"""Collect bounded, lossless native RGB/capture windows without registering them.

The workorder freezes window locations and original TIFF hashes. A window is a
search region, never a correspondence label, focus measurement, or train pair.
Only the requested owner's files are read; original mosaics stay unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import tarfile
import time
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def job_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Expected contained relative path')
    return path


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative_path(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source escaped containment')
    return path


def publish(source, target, expected):
    """Exclusive publication, including when two collectors race on Drive."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError('Existing immutable output has different checksum')
        return
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    reservation = None
    published = False
    try:
        with Path(source).open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst, 4 * 1024**2)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(temporary) != expected:
            raise ValueError('Copy checksum mismatch')
        # Drive FUSE does not reliably implement hard links. Reserve the name
        # exclusively, then rename a fully copied/verified same-directory file.
        # Readers accept only the completion marker, published after this step.
        try:
            with target.open('xb') as reserved:
                stat = os.fstat(reserved.fileno())
                reservation = (stat.st_dev, stat.st_ino)
        except FileExistsError:
            if digest(target) != expected:
                raise ValueError('Concurrent immutable output conflict')
            return
        temporary.replace(target)
        published = True
        if digest(target) != expected:
            raise ValueError('Published output checksum mismatch')
    finally:
        temporary.unlink(missing_ok=True)
        if reservation is not None and not published and target.exists():
            stat = target.stat()
            if (stat.st_dev, stat.st_ino) == reservation and stat.st_size == 0:
                target.unlink()


def validate(config):
    """Validate inputs and return deduplicated native windows for one owner."""
    import numpy as np
    from backend.serial_pairing import matrix3, transform_points
    owner = config['profile']
    order = config['workorder']
    if order.get('trainingAccepted') is not False or order.get('pixel_ground_truth') is not False:
        raise ValueError('Collector only accepts non-training search workorders')
    groups = order['groups']
    if not groups or len(groups) > 1000 or not order['candidates'] or len(order['candidates']) > 10000:
        raise ValueError('Workorder size outside bounds')
    for gid, row in groups.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', gid) or row['group_id'] != gid:
            raise ValueError('Invalid group identifier')
        for field in ('original_mosaic_sha256', 'original_capture_mask_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row[field]):
                raise ValueError('Invalid original TIFF digest')
        relative_path(row['source_reference']['drive_relative_path'])
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Test/unassigned sources are excluded')
        if any(type(v) is not int or v < 1 for v in row['native_size']):
            raise ValueError('Invalid native dimensions')
    requests, links, seen_pairs = {}, [], set()
    for pair in order['candidates']:
        pid = pair['pair_id']
        if pid in seen_pairs or pair.get('trainingAccepted') is not False:
            raise ValueError('Duplicate or accepted candidate')
        seen_pairs.add(pid)
        fixed, moving = [groups[pair[key]] for key in ('fixed_group', 'moving_group')]
        if (fixed['patient_id'] != moving['patient_id'] or fixed['patient_id'] != pair['patient_id']
                or fixed['split'] != moving['split'] or fixed['split'] != pair['split']
                or fixed['domain'] != 'THO' or moving['domain'] != 'HE'):
            raise ValueError('Patient, split, or domain mismatch')
        inverse = np.linalg.inv(matrix3(pair['HE_native_to_THO_native']))
        for roi in pair['ROI_search_windows']:
            box = roi['fixed_native_box_xyxy']
            if len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError('Native window needs four integers')
            x1, y1, x2, y2 = box
            fw, fh = fixed['native_size']
            if not (0 <= x1 < x2 <= fw and 0 <= y1 < y2 <= fh) or (x2-x1)*(y2-y1) > 1024**2:
                raise ValueError('Fixed search window outside bounds')
            corners = transform_points(inverse, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            mw, mh = moving['native_size']
            left = max(0, min(mw, int(np.floor(corners[:, 0].min())) - 2))
            top = max(0, min(mh, int(np.floor(corners[:, 1].min())) - 2))
            right = max(left, min(mw, int(np.ceil(corners[:, 0].max())) + 3))
            bottom = max(top, min(mh, int(np.ceil(corners[:, 1].max())) + 3))
            support = [left, top, right, bottom]
            if support != roi['moving_native_support_xyxy']:
                raise ValueError('Moving support does not follow frozen 3x3 transform')
            working_bytes = (x2-x1)*(y2-y1)*40 + (right-left)*(bottom-top)*16
            if working_bytes >= 32*1024**2:
                raise MemoryError('ROI exceeds bounded array working budget')
            for side, source, region in (('fixed', fixed, box), ('moving', moving, support)):
                if source['source_reference']['owner_profile'] != owner:
                    continue
                key = None
                if region[0] < region[2] and region[1] < region[3]:
                    key = hashlib.sha256(json.dumps([source['group_id'], region,
                        source['original_mosaic_sha256']], separators=(',', ':')).encode()).hexdigest()[:24]
                    requests[key] = {'request_id': key, 'group_id': source['group_id'], 'box_xyxy': region}
                links.append({'pair_id': pid, 'window': roi['search_window_id'], 'side': side, 'request_id': key})
    if not requests:
        raise ValueError('No regions for requested owner')
    return groups, requests, links


def collect(config, run_root, content_root):
    import cv2
    import numpy as np
    import tifffile
    from PIL import Image
    from backend.dataset_mosaic import read_dataset_region
    started = time.monotonic()
    cv2.setNumThreads(1)
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = lambda value: dump(root/'progress.json', value)
    groups, requests, links = validate(config)
    token = job_digest(config)
    drive = Path(content_root)/'drive/MyDrive'
    destination = contained(drive, config['drive_destination'])/token[:20]
    destination.mkdir(parents=True, exist_ok=True)
    archive = root/'native_windows.tar'
    previous = destination/'completed.json'
    if previous.is_file():
        completed = json.loads(previous.read_text())
        if completed['job_sha256'] != token or not completed['Drive_backup_verified']:
            raise ValueError('Existing completion does not match workorder')
        source = destination/'native_windows.tar'
        if digest(source) != completed['archive_sha256']:
            raise ValueError('Previously persisted archive checksum mismatch')
        shutil.copyfile(source, archive)
    else:
        data = root/'data'
        data.mkdir(exist_ok=True)
        records, verified = [], {}
        by_group = {}
        for request in requests.values():
            by_group.setdefault(request['group_id'], []).append(request)
        for number, (gid, regions) in enumerate(sorted(by_group.items()), 1):
            group = groups[gid]
            source = contained(drive, group['source_reference']['drive_relative_path'])
            cache = root/'cache'
            cache.mkdir(exist_ok=True)
            try:
                for filename, field in (('mosaic.tif', 'original_mosaic_sha256'), ('valid.tif', 'original_capture_mask_sha256')):
                    original = contained(source, filename)
                    local = cache/filename
                    shutil.copyfile(original, local)
                    if digest(local) != group[field]:
                        raise ValueError('Original TIFF checksum mismatch: ' + gid + '/' + filename)
                    with tifffile.TiffFile(local) as tif:
                        p = tif.pages[0]
                        if (len(tif.pages) != 1 or not p.is_tiled or p.compression != 1 or p.dtype != np.uint8
                                or p.samplesperpixel != (3 if filename == 'mosaic.tif' else 1)
                                or [p.imagewidth, p.imagelength] != group['native_size']):
                            raise ValueError('Original TIFF header differs from native contract')
                verified[gid] = {k: group[k] for k in ('original_mosaic_sha256', 'original_capture_mask_sha256', 'native_size', 'patient_id', 'split', 'domain')}
                for request in regions:
                    target = data/request['request_id']
                    target.mkdir(exist_ok=True)
                    arrays = {}
                    for filename, output in (('mosaic.tif', 'RGB.png'), ('valid.tif', 'capture.png')):
                        array = read_dataset_region(cache/filename, tuple(request['box_xyxy']), str(root),
                                                    memory_budget_bytes=32*1024**2)
                        Image.fromarray(array).save(target/output)
                        arrays[output] = {'shape': list(array.shape), 'raw_pixel_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                                          'png_sha256': digest(target/output)}
                        del array
                    record = {**request, 'pixels': arrays, 'native_sampling': 'unaltered integer crop, no resampling',
                              'trainingAccepted': False, 'sensor_focus_measured': False}
                    dump(target/'region.json', record)
                    records.append(record)
                state({'stage': 'reading_native', 'groups_done': number, 'groups_total': len(by_group),
                       'regions_done': len(records), 'job_sha256': token})
            finally:
                shutil.rmtree(cache)
        manifest = {'job_sha256': token, 'profile': config['profile'], 'requests': records, 'links': links,
                    'native_source_digests_verified': verified, 'trainingAccepted': False,
                    'capture_mask_meaning': 'camera capture support only, not anatomical correspondence',
                    'pixel_ground_truth': False, 'workorder': config['workorder']}
        dump(data/'manifest.json', manifest)
        with tarfile.open(archive, 'w') as tar:
            tar.add(data, arcname='data')
        expected = digest(archive)
        publish(archive, destination/'native_windows.tar', expected)
        completed = {'job_sha256': token, 'archive_sha256': expected, 'archive_bytes': archive.stat().st_size,
                     'profile': config['profile'], 'regions': len(records), 'groups': len(verified),
                     'Drive_backup_verified': True, 'drive_destination': str(destination),
                     'elapsed_seconds': round(time.monotonic()-started, 2),
                     'peak_RSS_MiB': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
                     'trainingAccepted': False}
        # Completion is last. Readers never accept incomplete archive writes.
        dump(previous, completed)
    parts = []
    with archive.open('rb') as stream:
        while block := stream.read(16*1024**2):
            name = f'part_{len(parts):03d}'
            (root/name).write_bytes(block)
            parts.append({'name': name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
    completed.update(stage='completed', parts=parts)
    dump(root/'completed.json', completed)
    state(completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--content-root', default='/content')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        collect(config, args.run_root, args.content_root)
    except Exception as error:
        dump(Path(args.run_root)/'progress.json', {'stage': 'failed', 'error': type(error).__name__+': '+str(error)})
        raise


if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
