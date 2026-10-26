"""Verified file transport independent of DriveFS; no credentials or training.

Use unchanged native crops and the full trusted checkpoints already retained
locally. Stream tar and 8 MiB parts; never allocate the whole input in memory.
"""
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pack(dataset, checkpoints, destination, *, required_checkpoints=('B2', 'C2'), require_same_patient_style=True):
    from tools.nonaligned_stain import data
    dataset, destination = Path(dataset), Path(destination)
    doc = data.validate(dataset, require_same_patient_style=require_same_patient_style)
    files = {'dataset/manifest.json': dataset/'manifest.json'}
    files.update({'dataset/'+r['file']: data.contained(dataset, r['file']) for r in doc['records']})
    if not set(required_checkpoints) <= {'B2', 'C2'} or set(checkpoints) != set(required_checkpoints):
        raise ValueError('Explicit trusted diagnostic checkpoints do not match this phase')
    for branch, item in checkpoints.items():
        path = Path(item['path'])
        if digest(path) != item['sha256']:
            raise ValueError('Local checkpoint checksum mismatch')
        files[branch+'.pt'] = path
    manifest = {name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for name, p in files.items()}
    if sum(m['bytes'] for m in manifest.values()) > 384*1024**2:
        raise ValueError('Local input byte budget exceeded')
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/'inputs.tar'
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w|') as tar:
        for name, path in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = path.stat().st_size, 0, 0o600
            with path.open('rb') as source:
                tar.addfile(info, source)
    parts = []
    with archive.open('rb') as stream:
        while block := stream.read(8*1024**2):
            name = f'input_part_{len(parts):03d}'
            with (destination/name).open('xb') as output:
                output.write(block)
            parts.append({'name': name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
    return {'input_transport': 'local_parts', 'allow_local_verified_completion': True,
            'inputs_sha256': digest(archive), 'inputs_bytes': archive.stat().st_size,
            'input_files': manifest, 'input_parts': parts, 'dataset_sha256': digest(dataset/'manifest.json'),
            'required_checkpoints': sorted(required_checkpoints)}


def verify_parts(root, cfg):
    root = Path(root)
    parts = cfg['input_parts']
    if (not 1 <= len(parts) <= 64 or not 0 < cfg['inputs_bytes'] <= 384*1024**2
            or sum(p['bytes'] for p in parts) != cfg['inputs_bytes']
            or not re.fullmatch('[0-9a-f]{64}', cfg['inputs_sha256'])):
        raise ValueError('Invalid local input transfer manifest')
    for i, part in enumerate(parts):
        if (part['name'] != f'input_part_{i:03d}' or not 0 < part['bytes'] <= 8*1024**2
                or not re.fullmatch('[0-9a-f]{64}', part['sha256'])):
            raise ValueError('Unsafe local input part')
        path = root/part['name']
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError('Local input part checksum mismatch')


def stage(root, cfg):
    """Verify→assemble→extract regular whitelisted files→check native identities."""
    root = Path(root).resolve()
    verify_parts(root, cfg)
    expected = cfg['input_files']
    required = cfg.get('required_checkpoints', ['B2', 'C2'])
    if (not set(required) <= {'B2', 'C2'} or len(set(required)) != len(required)
            or not 1 <= len(expected) <= 2051
            or not {'dataset/manifest.json', *(p+'.pt' for p in required)} <= expected.keys()
            or sum(m['bytes'] for m in expected.values()) > 384*1024**2):
        raise ValueError('Invalid local input member manifest')
    archive = root/'received_inputs.tar'
    h = hashlib.sha256()
    with archive.open('xb') as output:
        for part in cfg['input_parts']:
            with (root/part['name']).open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    output.write(block)
                    h.update(block)
    if h.hexdigest() != cfg['inputs_sha256']:
        raise ValueError('Assembled local input checksum mismatch')
    destination = root/'inputs'
    destination.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(archive, 'r|') as tar:
        for member in tar:
            name, path = member.name, PurePosixPath(member.name)
            allowed = name in ('B2.pt', 'C2.pt', 'dataset/manifest.json') or (
                name.startswith('dataset/') and path.suffix == '.png')
            if (not allowed or name not in expected or name in seen or not member.isfile()
                    or path.is_absolute() or '..' in path.parts):
                raise ValueError('Unsafe local input archive member')
            metadata = expected[name]
            if (not 0 < member.size <= (256*1024**2 if name.endswith('.pt') else 8*1024**2)
                    or member.size != metadata['bytes']):
                raise ValueError('Local input member size mismatch')
            target = destination/name
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Local input archive path escape')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024*1024)
            if digest(target) != metadata['sha256']:
                raise ValueError('Extracted local input checksum mismatch')
            seen.add(name)
    if seen != expected.keys():
        raise ValueError('Missing local input archive members')
    for branch in required:
        if digest(destination/(branch+'.pt')) != cfg['archives'][branch]['checkpoint_sha256']:
            raise ValueError('Transport changed a trusted diagnostic checkpoint')
    if digest(destination/'dataset/manifest.json') != cfg['dataset_sha256']:
        raise ValueError('Transport changed the native release')
    return destination/'dataset'
