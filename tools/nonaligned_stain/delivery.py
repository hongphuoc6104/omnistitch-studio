"""Immutable checkpoint receipts across an ephemeral runtime boundary.

No model execution: streams a published checkpoint inode and bounded metadata.
An atomic sidecar/SHA distinguishes normal concurrent replacement from a valid
snapshot. Only a verified host receipt makes a checkpoint durable locally.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(root, phase):
    root = Path(root).resolve()
    if not re.fullmatch('[A-Za-z0-9_-]{1,32}', phase):
        raise ValueError('Unsafe checkpoint phase')
    source = root/'results'/phase
    sidecar, checkpoint = source/'checkpoint_meta.json', source/'checkpoint.pt'
    if not sidecar.is_file() or not checkpoint.is_file():
        return None
    if sidecar.stat().st_size > 16384:
        raise ValueError('Oversized checkpoint sidecar')
    meta = json.loads(sidecar.read_text())
    if (type(meta['step']) is not int or not 0 <= meta['step'] <= 1000
            or not 0 < meta['bytes'] <= 256*1024**2
            or not re.fullmatch('[0-9a-f]{64}', meta['sha256'])):
        raise ValueError('Invalid checkpoint sidecar')
    folder = root/'publications'
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        archive = Path(temp)/'checkpoint.tar.gz'
        h = hashlib.sha256()
        class Reader:
            def __init__(self, stream):
                self.stream = stream
            def read(self, count):
                block = self.stream.read(count)
                h.update(block)
                return block
        with checkpoint.open('rb') as cp:
            if os.fstat(cp.fileno()).st_size != meta['bytes']:
                return None
            with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0, compresslevel=1) as gz:
                with tarfile.open(fileobj=gz, mode='w|') as tar:
                    info = tarfile.TarInfo('checkpoint.pt')
                    info.size, info.mtime, info.mode = meta['bytes'], 0, 0o600
                    tar.addfile(info, Reader(cp))
                    values = {'checkpoint_meta.json': json.dumps(meta, sort_keys=True).encode()}
                    if (source/'run.json').is_file():
                        if (source/'run.json').stat().st_size > 1024**2:
                            raise ValueError('Oversized checkpoint run metadata')
                        values['run.json'] = (source/'run.json').read_bytes()
                    if (source/'loss.jsonl').is_file():
                        if (source/'loss.jsonl').stat().st_size > 8*1024**2:
                            raise ValueError('Oversized checkpoint loss history')
                        lines = []
                        for line in (source/'loss.jsonl').read_text().splitlines():
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # The producer may currently append its final line.
                            if row['step'] <= meta['step']:
                                lines.append(json.dumps(row)+'\n')
                        values['loss_prefix.jsonl'] = ''.join(lines).encode()
                    for p in sorted(source.glob('gradient_*.json')):
                        if (re.fullmatch(r'gradient_[0-9]{4}\.json', p.name)
                                and int(p.stem.split('_')[1]) <= meta['step'] and p.stat().st_size < 1024**2):
                            values[p.name] = p.read_bytes()
                    for name, content in sorted(values.items()):
                        info = tarfile.TarInfo(name)
                        info.size, info.mtime, info.mode = len(content), 0, 0o600
                        tar.addfile(info, io.BytesIO(content))
        if h.hexdigest() != meta['sha256']:
            return None  # Checkpoint and sidecar changed between their atomic publications.
        sha = digest(archive)
        identifier = sha[:32]
        target = folder/(identifier+'.tar.gz')
        if target.exists():
            if digest(target) != sha:
                raise ValueError('Checkpoint publication identity conflict')
        else:
            os.link(archive, target)
        parts = []
        with target.open('rb') as stream:
            while block := stream.read(8*1024**2):
                name = f'publication_{identifier}_part_{len(parts):03d}'
                path = root/name
                if not path.exists():
                    with path.open('xb') as output:
                        output.write(block)
                elif digest(path) != hashlib.sha256(block).hexdigest():
                    raise ValueError('Checkpoint part identity conflict')
                parts.append({'name': name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
        return {'id': identifier, 'kind': 'checkpoint', 'phase': phase, 'step': meta['step'],
                'checkpoint_sha256': meta['sha256'], 'archive_sha256': sha,
                'archive_bytes': target.stat().st_size, 'parts': parts}


def receive(publication, destination, fetch):
    """Fetch bounded parts, verify all bytes, then expose one immutable receipt."""
    p = publication
    if (p.get('kind') != 'checkpoint' or not re.fullmatch('[A-Za-z0-9_-]{1,32}', p['phase'])
            or type(p['step']) is not int or not 0 <= p['step'] <= 1000
            or not re.fullmatch('[0-9a-f]{64}', p['archive_sha256'])
            or not re.fullmatch('[0-9a-f]{64}', p['checkpoint_sha256'])
            or p['id'] != p['archive_sha256'][:32] or not 0 < p['archive_bytes'] <= 256*1024**2
            or not 1 <= len(p['parts']) <= 32
            or sum(q['bytes'] for q in p['parts']) != p['archive_bytes']):
        raise ValueError('Invalid checkpoint publication')
    destination = Path(destination)/'deliveries'/p['phase']
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/f"{p['step']:04d}_{p['id']}.tar.gz"
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=destination) as temp:
            temp = Path(temp)
            archive = temp/'checkpoint.tar.gz'
            with archive.open('wb') as output:
                for i, part in enumerate(p['parts']):
                    if (part['name'] != f"publication_{p['id']}_part_{i:03d}"
                            or not 0 < part['bytes'] <= 8*1024**2):
                        raise ValueError('Unsafe checkpoint publication part')
                    path = temp/'part'
                    fetch(part['name'], path)
                    if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                        raise ValueError('Checkpoint delivery checksum mismatch')
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, output, 1024*1024)
            if digest(archive) != p['archive_sha256']:
                raise ValueError('Complete checkpoint delivery mismatch')
            os.link(archive, target)
    if target.stat().st_size != p['archive_bytes'] or digest(target) != p['archive_sha256']:
        raise ValueError('Existing checkpoint receipt is corrupt')
    record = {k: p[k] for k in ('id', 'kind', 'phase', 'step', 'checkpoint_sha256', 'archive_sha256', 'archive_bytes')}
    record.update(archive=str(target), verified_local=True)
    sidecar = target.with_suffix('.json')
    tmp = sidecar.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2)+'\n')
    tmp.replace(sidecar)
    return record


def restore(record, destination):
    """Restore only trusted checkpoint/provenance files; never execute an archive."""
    archive = Path(record['archive'])
    if not record.get('verified_local') or digest(archive) != record['archive_sha256']:
        raise ValueError('Unverified checkpoint receipt')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            allowed = member.name in ('checkpoint.pt', 'checkpoint_meta.json', 'run.json', 'loss_prefix.jsonl') or re.fullmatch(r'gradient_[0-9]{4}\.json', member.name)
            if (not allowed or not member.isfile() or member.name in seen
                    or not 0 <= member.size <= (256*1024**2 if member.name == 'checkpoint.pt' else 8*1024**2)):
                raise ValueError('Unsafe checkpoint receipt member')
            total += member.size
            if total > 280*1024**2:
                raise ValueError('Checkpoint restoration byte budget exceeded')
            target = destination/member.name
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Checkpoint restoration path escape')
            with tar.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024*1024)
            seen.add(member.name)
    if not {'checkpoint.pt', 'checkpoint_meta.json'} <= seen:
        raise ValueError('Incomplete checkpoint receipt')
    meta = json.loads((destination/'checkpoint_meta.json').read_text())
    if (meta['step'] != record['step'] or meta['sha256'] != record['checkpoint_sha256']
            or digest(destination/'checkpoint.pt') != record['checkpoint_sha256']):
        raise ValueError('Restored checkpoint source mismatch')
    return destination/'checkpoint.pt'
