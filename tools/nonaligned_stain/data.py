"""Native sensor crops for explicitly unpaired experiments; no positive pairs."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from tools.paired_stain.data import contained, read_image, sha256


def validate(root):
    root = Path(root).resolve()
    path = root/'manifest.json'
    if path.stat().st_size > 8*1024**2:
        raise ValueError('Oversized unpaired manifest')
    doc = json.loads(path.read_text())
    if (doc.get('schema') != 'nonaligned-stain-v1' or doc.get('paired') is not False
            or doc.get('ready_for_unpaired_pilot') is not True):
        raise ValueError('Expected explicit unpaired release, not paired candidates')
    rows = doc.get('records', [])
    if not 1 <= len(rows) <= 2048:
        raise ValueError('Empty/oversized unpaired release')
    owners, pixels, sources, buckets, files = {}, {}, {}, {}, set()
    size = 32 if doc.get('synthetic_fixture') else 256
    for row in rows:
        patient, split, domain = row['patient_id'], row['split'], row['domain']
        if not patient or split not in ('train', 'validation') or domain not in ('THO', 'HE'):
            raise ValueError('Forbidden test data/identity')
        if owners.setdefault(patient, split) != split:
            raise ValueError('Patient leakage')
        if row['file'] in files:
            raise ValueError('Duplicate crop record')
        files.add(row['file'])
        if row.get('coordinate_system') != 'original_sensor_pixels':
            raise ValueError('This version expects original sensor crops')
        box = row['box']
        if (len(box) != 4 or any(type(v) is not int or v < 0 for v in box)
                or box[2:] != [size, size]):
            raise ValueError('Native unresized crop coordinates/size mismatch')
        if not .75 <= row.get('structured_fraction', -1) <= 1:
            raise ValueError('Native technical texture curation not passed')
        if not re.fullmatch('[0-9a-f]{64}', row.get('source_sha256', '')):
            raise ValueError('Missing original source digest')
        identity = (patient, split, domain)
        if sources.setdefault(row['source_sha256'], identity) != identity:
            raise ValueError('Source file leakage')
        image = read_image(root, row, 'RGB', (size, size))
        pixel_hash = __import__('hashlib').sha256(image.tobytes()).hexdigest()
        if pixel_hash != row['pixel_sha256']:
            raise ValueError('Native pixel digest mismatch')
        if pixels.setdefault(pixel_hash, identity) != identity:
            raise ValueError('Pixel leakage')
        if not np.isfinite(row.get('native_laplacian_variance', float('nan'))):
            raise ValueError('Original sensor QC evidence missing')
        buckets.setdefault((split, domain), set()).add(patient)
    if any(not buckets.get((s, d)) for s in ('train', 'validation') for d in ('THO', 'HE')):
        raise ValueError('Missing independent train/validation domains')
    if len(buckets[('train', 'THO')]) < 2:
        raise ValueError('Need multiple train patients')
    if buckets[('train', 'THO')] - buckets[('train', 'HE')]:
        raise ValueError('B2 needs same-patient TRAIN style availability')
    return doc


def publish(source, output):
    """Freeze the existing core selection, retaining unchanged bytes and split."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source/'manifest.json').read_text())
    rows = [r for r in original['records'] if r.get('structured_fraction', 0) >= .75]
    doc = {'schema': 'nonaligned-stain-v1', 'paired': False,
           'ready_for_unpaired_pilot': True, 'trainingAccepted': False,
           'pixel_ground_truth': False, 'source_manifest_sha256': sha256(source/'manifest.json'),
           'sampling': 'native texture-rich core; technical selection, not tissue segmentation',
           'records': rows}
    if original.get('synthetic_fixture'):
        doc['synthetic_fixture'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='unpaired-stage-') as temp:
        stage = Path(temp)
        for row in rows:
            path = contained(source, row['file'])
            if sha256(path) != row['sha256']:
                raise ValueError('Source crop checksum changed')
            target = stage/row['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (stage/'manifest.json').write_text(json.dumps(doc, indent=2)+'\n')
        validate(stage)
        output.mkdir()
        for p in [p for p in stage.iterdir() if p.name != 'manifest.json']+[stage/'manifest.json']:
            p.rename(output/p.name)
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    doc = validate(args.dataset) if args.dataset else publish(args.source, args.output)
    print('UNPAIRED_RELEASE_VERIFIED', len(doc['records']))
