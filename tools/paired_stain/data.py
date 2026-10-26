"""Bounded, lossless paired releases; candidate overlap is never supervision.

This is a research data contract, not an anatomical validator. The signed-off
review must come from an actual correspondence/mask review, not an optimizer.
No patient data, reviews or derived weights belong in the source repository.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contained(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Expected contained relative file path')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing/escaping release file')
    return path


def read_image(root, item, mode, size=None):
    """Validate encoded size and headers before decoding bounded PNG pixels."""
    path = contained(root, item['file'])
    if path.suffix.lower() != '.png' or path.stat().st_size > 8 * 1024**2:
        raise ValueError('Expected bounded lossless PNG')
    if sha256(path) != item['sha256']:
        raise ValueError('Image checksum mismatch')
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != mode or not 16 <= min(image.size) <= max(image.size) <= 512:
            raise ValueError('Unexpected image type/size')
        if size is not None and image.size != size:
            raise ValueError('Mismatched paired dimensions')
        return np.array(image)


def validate(root, *, require_release=True, full_mask=True):
    """Reject test leakage, corrupt images, missing independent QC and masks.

    The fixed 4px gate is an engineering ceiling for this fine-scale pilot, not
    a clinical threshold or evidence that cells in serial sections are equal.
    Latent encoding mixes context; the shared B/C pilot requires completely
    approved patches rather than masking a few bad pixels after VAE encoding.
    """
    root = Path(root).resolve()
    manifest_file = root / 'manifest.json'
    if manifest_file.stat().st_size > 16 * 1024**2:
        raise ValueError('Oversized manifest')
    doc = json.loads(manifest_file.read_text())
    if doc.get('schema') != 'paired-stain-v1' or not isinstance(doc.get('records'), list):
        raise ValueError('Unsupported paired release schema')
    rows = doc['records']
    if not 1 <= len(rows) <= 4096:
        raise ValueError('Empty/oversized paired dataset')
    if require_release and (doc.get('kind') != 'approved_release' or doc.get('trainingAccepted') is not True):
        raise ValueError('Candidate dataset is not an approved training release')
    ids, patients, pixel_owners, source_owners, sizes = set(), {}, {}, {}, set()
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate pair-window identifier')
        ids.add(row['id'])
        patient, split = row['patient_id'], row['split']
        if not isinstance(patient, str) or not patient or split not in ('train', 'validation'):
            raise ValueError('Missing patient or forbidden test split')
        if patients.setdefault(patient, split) != split:
            raise ValueError('Patient split leakage')
        h = np.asarray(row['HE_to_THO_3x3'], dtype=float)
        box = row['THO_native_box_xyxy']
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-10:
            raise ValueError('Invalid full 3x3 transform')
        if len(box) != 4 or any(type(v) is not int or v < 0 for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Invalid native coordinate box')
        tho = read_image(root, row['THO'], 'RGB')
        size = (tho.shape[1], tho.shape[0])
        if (box[2] - box[0], box[3] - box[1]) != size:
            raise ValueError('Native size differs from unresized patch')
        he = read_image(root, row['HE'], 'RGB', size)
        capture = read_image(root, row['capture'], 'L', size)
        mask = read_image(root, row['supervision'], 'L', size)
        if not np.isin(mask, [0, 255]).all() or not np.isin(capture, [0, 255]).all():
            raise ValueError('Expected binary masks')
        if np.any((mask > 0) & (capture == 0)):
            raise ValueError('Supervision outside observed support')
        for domain, image in [('THO', tho), ('HE', he)]:
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            identity = (patient, split, domain)
            if pixel_owners.setdefault(digest, identity) != identity:
                raise ValueError('Identical pixels across patient/split/domain')
        sizes.add(size)
        if not require_release:
            continue
        for domain in ('THO', 'HE'):
            source = row.get('source_evidence', {}).get(domain, {})
            if (source.get('patient_id') != patient or source.get('split') != split
                    or source.get('domain') != domain
                    or any(not re.fullmatch('[0-9a-f]{64}', str(source.get(k, '')))
                           for k in ('original_mosaic_sha256', 'original_capture_mask_sha256', 'raw_crop_rgb_sha256'))):
                raise ValueError('Original source/crop provenance is missing or inconsistent')
            identity = (patient, split, domain)
            if source_owners.setdefault(source['original_mosaic_sha256'], identity) != identity:
                raise ValueError('Original source leaks across patient/split/domain')
        qc = row.get('qc', {})
        fit, holdout = qc.get('fit_ids', []), qc.get('holdout_ids', [])
        if (qc.get('status') != 'approved' or not qc.get('reviewer') or not qc.get('review_evidence')
                or qc.get('mask_reviewed') is not True or qc.get('anatomical_identity_confirmed') is not True
                or qc.get('supervision_scale') != 'fine_spatial_correspondence'
                or not {'B', 'C'}.issubset(set(qc.get('approved_branches', [])))):
            raise ValueError('Fine-scale anatomy/mask review is missing')
        if (len(set(fit)) < 2 or len(set(holdout)) < 3 or set(fit) & set(holdout)
                or not qc.get('heldout_regions_cover_patch')
                or not 0 <= float(qc.get('heldout_max_error_px', float('inf'))) <= 4):
            raise ValueError('Independent spatial validation has not passed')
        if not mask.any() or (full_mask and not (mask == 255).all()):
            raise ValueError('Shared latent pilot needs a fully reviewed patch')
    if require_release:
        if len(sizes) != 1 or next(iter(sizes))[0] != next(iter(sizes))[1]:
            raise ValueError('Pilot requires one square native patch size')
        if sum(s == 'train' for s in patients.values()) < 2 or 'validation' not in patients.values():
            raise ValueError('Need at least two train and one independent validation patients')
    return doc


def publish(root, reviews_file, output):
    """Create a new immutable release from separately reviewed masks/evidence.

    Existing outputs are never overwritten. A review file cannot turn capture
    into supervision by omission: it must supply its own explicit binary mask.
    """
    root, output, reviews_file = Path(root).resolve(), Path(output).resolve(), Path(reviews_file).resolve()
    source = validate(root, require_release=False)
    reviews = json.loads(reviews_file.read_text())
    accepted = {r['id']: r for r in reviews['records'] if r.get('qc', {}).get('status') == 'approved'}
    if not accepted or len(accepted) != len([r for r in reviews['records'] if r.get('qc', {}).get('status') == 'approved']):
        raise ValueError('No approved unique reviews')
    source_ids = {r['id'] for r in source['records']}
    if set(accepted) - source_ids:
        raise ValueError('Review references unknown candidate')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='paired-release-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'release'
        stage.mkdir()
        result = {'schema': 'paired-stain-v1', 'kind': 'approved_release', 'trainingAccepted': True,
                  'pixel_ground_truth': False, 'source_manifest_sha256': sha256(root/'manifest.json'),
                  'reviews_sha256': sha256(reviews_file), 'records': []}
        for index, row in enumerate(source['records']):
            if row['id'] not in accepted:
                continue
            row = json.loads(json.dumps(row))
            folder = stage / f'{index:04d}'
            folder.mkdir()
            for key in ('THO', 'HE', 'capture'):
                path = contained(root, row[key]['file'])
                target = folder / (key + '.png')
                shutil.copyfile(path, target)
                row[key] = {'file': str(target.relative_to(stage)), 'sha256': sha256(target)}
            approval = accepted[row['id']]
            mask = contained(reviews_file.parent, approval['supervision']['file'])
            if sha256(mask) != approval['supervision']['sha256']:
                raise ValueError('Reviewed mask checksum mismatch')
            target = folder / 'supervision.png'
            shutil.copyfile(mask, target)
            row['supervision'] = {'file': str(target.relative_to(stage)), 'sha256': sha256(target)}
            row['qc'] = approval['qc']
            result['records'].append(row)
        (stage/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
        validate(stage)
        # An exclusive reservation prevents replacement of a concurrent result.
        output.mkdir()
        try:
            members = [p for p in stage.iterdir() if p.name != 'manifest.json']
            for path in members + [stage/'manifest.json']:
                os.rename(path, output/path.name)
        except BaseException:
            # No valid manifest is published until last; an interrupted output
            # remains visibly incomplete and requires inspection before retry.
            raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--reviews')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.reviews:
        if not args.output:
            parser.error('--reviews requires --output')
        doc = publish(args.dataset, args.reviews, args.output)
    else:
        doc = validate(args.dataset)
    print('PAIRED_RELEASE_VERIFIED', len(doc['records']))


if __name__ == '__main__':
    main()
