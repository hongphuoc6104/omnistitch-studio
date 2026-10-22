"""Serial-section coordinate contracts and bounded native pair inspection.

THO is fixed. H&E maps moving→fixed in pixel-center coordinates. Capture
overlap is NOT anatomical correspondence. No automatic operation accepts a
serial pair as cell/pixel ground truth or mutates the original mosaics.
"""

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

from .dataset_mosaic import read_dataset_region
from .io_utils import resolve_file_path
from .wsi_exporter import _publish_staged_artifacts, _reclaim_memmap_pages, write_tiled_tiff


def matrix3(values):
    """Validate a finite, invertible 3×3 affine/projective matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.size != 9 or not np.isfinite(array).all():
        raise ValueError("Expected finite 3x3 transform")
    array = array.reshape(3, 3)
    if abs(array[2, 2]) < 1e-12:
        raise ValueError("Invalid transform normalization")
    array = array / array[2, 2]
    if abs(np.linalg.det(array)) < 1e-10 or np.linalg.cond(array) > 1e12:
        raise ValueError("Singular or ill-conditioned transform")
    return array


def transform_points(matrix, points):
    """Map x,y pixel centers; never silently drop a projective third row."""
    matrix = matrix3(matrix)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 points")
    mapped = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(mapped[:, 2] <= 1e-6):
        raise ValueError("Transform crosses projective horizon")
    return mapped[:, :2] / mapped[:, 2:3]


def proxy_to_native(native_size, proxy_size):
    """OpenCV resize pixel-center contract: native=(proxy+.5)*scale-.5.

    Only use this exact lift for proxies generated with that sampling contract.
    Legacy tile-pasted JPEGs are screening-only and cannot claim exact mapping.
    """
    if any(type(value) is not int or value <= 0 for value in (*native_size, *proxy_size)):
        raise ValueError("Positive integer image sizes required")
    sx, sy = native_size[0] / proxy_size[0], native_size[1] / proxy_size[1]
    return np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]])


def lift_proxy_transform(moving_to_fixed, fixed_native, fixed_proxy,
                         moving_native, moving_proxy):
    """Lift moving-proxy→fixed-proxy to moving-native→fixed-native."""
    return matrix3(proxy_to_native(fixed_native, fixed_proxy)
                   @ matrix3(moving_to_fixed)
                   @ np.linalg.inv(proxy_to_native(moving_native, moving_proxy)))


def landmark_report(moving_to_fixed, landmarks):
    """Evaluate explicitly held-out native landmarks without fitting them.

    A caller records reviewer and identity provenance separately. This function
    does not certify whether an annotation depicts the same biological object.
    """
    held = [row for row in landmarks if row.get("role") == "validation"]
    if any(row.get("used_for_fit", True) for row in held):
        raise ValueError("Validation landmarks must explicitly exclude fit use")
    if not held:
        return {"count": 0, "status": "pending_independent_landmarks",
                "median_native_px": None, "p95_native_px": None}
    predicted = transform_points(moving_to_fixed, [row["moving_xy"] for row in held])
    reference = np.asarray([row["fixed_xy"] for row in held], dtype=float)
    if not np.isfinite(reference).all():
        raise ValueError("Invalid fixed landmarks")
    errors = np.linalg.norm(predicted - reference, axis=1)
    return {"count": len(held), "status": "measured_not_biologically_certified",
            "median_native_px": float(np.median(errors)),
            "p95_native_px": float(np.percentile(errors, 95)),
            "errors_native_px": errors.tolist()}


def _info(path):
    with tifffile.TiffFile(path) as image:
        page = image.pages[0]
        if (len(image.pages) != 1 or not page.is_tiled or page.compression != 1
                or page.dtype != np.uint8 or page.samplesperpixel not in (1, 3)):
            raise ValueError("Expected original uncompressed uint8 tiled dataset TIFF")
        return page.imagewidth, page.imagelength, page.samplesperpixel


def make_registration_proxy(path, workspace_root, *, max_dimension=1200,
                            block=96, memory_budget_bytes=64 * 1024**2):
    """Bounded fixed-grid proxy. Never use its pixels for native clarity QC."""
    path = resolve_file_path(str(path), workspace_root)
    width, height, channels = _info(path)
    if channels != 3 or not 64 <= max_dimension <= 1600 or not 16 <= block <= 128:
        raise ValueError("Invalid proxy specification")
    scale = min(1., max_dimension / max(width, height))
    pw, ph = max(1, round(width * scale)), max(1, round(height * scale))
    if pw * ph * 3 + block * block * 32 > memory_budget_bytes // 2:
        raise MemoryError("Proxy exceeds bounded budget")
    result = np.empty((ph, pw, 3), dtype=np.uint8)
    spacing_x, spacing_y = width / pw, height / ph
    sigma = max(0., .5 * (max(spacing_x, spacing_y) - 1))
    halo = int(np.ceil(3 * sigma)) + 2
    for y in range(0, ph, block):
        for x in range(0, pw, block):
            x2, y2 = min(pw, x + block), min(ph, y + block)
            xs = (np.arange(x, x2, dtype=np.float32) + .5) * spacing_x - .5
            ys = (np.arange(y, y2, dtype=np.float32) + .5) * spacing_y - .5
            left, top = max(0, int(np.floor(xs[0])) - halo), max(0, int(np.floor(ys[0])) - halo)
            right = min(width, int(np.ceil(xs[-1])) + halo + 1)
            bottom = min(height, int(np.ceil(ys[-1])) + halo + 1)
            if (right - left) * (bottom - top) * 24 > memory_budget_bytes // 2:
                raise MemoryError("Native proxy support exceeds budget")
            rgb = read_dataset_region(path, (left, top, right, bottom), workspace_root,
                                      memory_budget_bytes=memory_budget_bytes // 2)
            if sigma > .01:
                rgb = cv2.GaussianBlur(rgb, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
            mx, my = np.meshgrid(xs - left, ys - top)
            result[y:y2, x:x2] = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
    return result, {"native_size": [width, height], "proxy_size": [pw, ph],
                    "pixel_center_mapping": proxy_to_native([width, height], [pw, ph]).tolist(),
                    "sampling": "fixed-grid bilinear after Gaussian anti-alias; registration only"}


def inspect_native_pair(fixed_path, moving_path, fixed_valid_path, moving_valid_path,
                        moving_to_fixed, box, workspace_root, *,
                        memory_budget_bytes=64 * 1024**2):
    """Read a bounded THO ROI and warp H&E once into that fixed ROI.

    ``box`` is x1,y1,x2,y2 in THO mosaic native pixels. The returned capture
    overlap only indicates observed camera coverage, never true tissue identity.
    Original capture-valid masks are mandatory (white glass is not padding).
    """
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Expected four built-in integer ROI coordinates")
    paths = [resolve_file_path(str(p), workspace_root) for p in
             (fixed_path, moving_path, fixed_valid_path, moving_valid_path)]
    fw, fh, fc = _info(paths[0])
    mw, mh, mc = _info(paths[1])
    if fc != 3 or mc != 3 or _info(paths[2]) != (fw, fh, 1) or _info(paths[3]) != (mw, mh, 1):
        raise ValueError("RGB/mask size or channel mismatch")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= fw and 0 <= y1 < y2 <= fh):
        raise ValueError("ROI outside fixed mosaic")
    h = matrix3(moving_to_fixed)
    # Denominator is linear; positive corner denominators protect the ROI interior.
    inverse_corners = transform_points(np.linalg.inv(h), [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    left = max(0, min(mw, int(np.floor(inverse_corners[:, 0].min())) - 2))
    top = max(0, min(mh, int(np.floor(inverse_corners[:, 1].min())) - 2))
    right = max(left, min(mw, int(np.ceil(inverse_corners[:, 0].max())) + 3))
    bottom = max(top, min(mh, int(np.ceil(inverse_corners[:, 1].max())) + 3))
    pixels = (x2 - x1) * (y2 - y1)
    if pixels * 40 + (right - left) * (bottom - top) * 16 > memory_budget_bytes // 2:
        raise MemoryError("Pair ROI/warp exceeds memory budget")
    fixed = read_dataset_region(paths[0], tuple(box), workspace_root,
                                memory_budget_bytes=memory_budget_bytes // 2)
    fixed_valid = read_dataset_region(paths[2], tuple(box), workspace_root,
                                      memory_budget_bytes=memory_budget_bytes // 2) > 0
    aligned = np.full_like(fixed, 255)
    observed = np.zeros(fixed_valid.shape, dtype=np.uint8)
    if right > left and bottom > top:
        moving = read_dataset_region(paths[1], (left, top, right, bottom), workspace_root,
                                     memory_budget_bytes=memory_budget_bytes // 2)
        moving_valid = read_dataset_region(paths[3], (left, top, right, bottom), workspace_root,
                                           memory_budget_bytes=memory_budget_bytes // 2)
        to_roi = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1.]])
        from_crop = np.array([[1, 0, left], [0, 1, top], [0, 0, 1.]])
        local = to_roi @ h @ from_crop
        aligned = cv2.warpPerspective(moving, local, (x2 - x1, y2 - y1),
                                      flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        observed = cv2.warpPerspective((moving_valid > 0).astype(np.uint8), local,
                                       (x2 - x1, y2 - y1), flags=cv2.INTER_NEAREST)
    overlap = fixed_valid & (observed > 0)
    return fixed, aligned, overlap.astype(np.uint8), {
        "fixed_roi_xyxy": list(box), "moving_support_xyxy": [left, top, right, bottom],
        "HE_native_to_THO_native": h.tolist(), "output_coordinate_system": "THO_native_pixels",
        "observed_overlap_fraction": float(overlap.mean()),
        "mask_meaning": "capture coverage intersection, NOT anatomical correspondence",
        "trainingAccepted": False, "pixel_ground_truth": False}


def sha256_file(path):
    """Streaming provenance checksum."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_hypotheses(fixed_ids, moving_ids, edges, *, max_states=100000, keep=20):
    """Enumerate one-to-one *hypotheses* with explicit unmatched sections.

    Labels have no matching semantics. This is a bounded diagnostic graph
    analysis, not an anatomical identity classifier. Fragment relationships may
    invalidate the one-to-one assumption; nothing is automatically selected.
    Edge scores (e.g. outline Dice) are ranking aids, not confidence values.
    """
    fixed_ids, moving_ids = list(fixed_ids), list(moving_ids)
    if (not fixed_ids or not moving_ids or max(len(fixed_ids), len(moving_ids)) > 8
            or len(set(fixed_ids)) != len(fixed_ids) or len(set(moving_ids)) != len(moving_ids)
            or not 1 <= keep <= 100 or max_states < 1):
        raise ValueError("Invalid bounded assignment graph")
    scores = {}
    for edge in edges:
        key = edge["fixed_id"], edge["moving_id"]
        score = float(edge["diagnostic_score"])
        if (key in scores or key[0] not in fixed_ids or key[1] not in moving_ids
                or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid or duplicate assignment edge")
        scores[key] = score
    complete, partial = [], []
    states = 0
    target_count = min(len(fixed_ids), len(moving_ids))
    def visit(index, used, selected):
        nonlocal states
        states += 1
        if states > max_states:
            raise ValueError("Assignment state budget exceeded; review/limit fragment candidates")
        if index == len(fixed_ids):
            if not selected:
                return
            values = [scores[key] for key in selected]
            row = {"edges": [{"fixed_id": a, "moving_id": b, "diagnostic_score": scores[a, b]}
                              for a, b in selected],
                   "unmatched_fixed": [a for a in fixed_ids if all(a != key[0] for key in selected)],
                   "unmatched_moving": [b for b in moving_ids if b not in used],
                   "mean_diagnostic_score": float(np.mean(values)),
                   "minimum_edge_diagnostic_score": min(values),
                   "cardinality": len(selected), "selected_as_true_pairing": False}
            bucket = complete if len(selected) == target_count else partial
            bucket.append(row)
            bucket.sort(key=lambda r: (r["cardinality"], r["mean_diagnostic_score"],
                                        r["minimum_edge_diagnostic_score"]), reverse=True)
            del bucket[keep:]
            return
        fixed = fixed_ids[index]
        visit(index + 1, used, selected)  # Never force a missing/bad section.
        for moving in moving_ids:
            if moving not in used and (fixed, moving) in scores:
                visit(index + 1, used | {moving}, selected + [(fixed, moving)])
    visit(0, set(), [])
    return {"full_cardinality_hypotheses": complete, "partial_hypotheses": partial,
            "mean_diagnostic_gap": complete[0]["mean_diagnostic_score"] - complete[1]["mean_diagnostic_score"]
            if len(complete) >= 2 else None,
            "states_enumerated": states, "status": "requires_native_anatomical_correspondence_review",
            "assumption": "one-to-one hypotheses only; physical fragment mapping still needs checking",
            "score_meaning": "diagnostic ranking, not confidence or anatomical truth",
            "trainingAccepted": False}


def export_aligned_pair(fixed_path, moving_path, fixed_valid_path, moving_valid_path,
                        moving_to_fixed, output_dir, workspace_root, *,
                        fixed_reference, moving_reference, tile_size=512,
                        memory_budget_bytes=64 * 1024**2, progress_callback=None):
    """Stream a review-only H&E derivative in the unchanged THO canvas.

    Original THO/HE are referenced by immutable hashes, never overwritten.
    New nonempty output directories are refused. Anatomical ignore/loss masks
    require a subsequent independent review; capture overlap alone is not enough.
    """
    fixed_path, moving_path, fixed_valid_path, moving_valid_path = [
        resolve_file_path(str(path), workspace_root) for path in
        (fixed_path, moving_path, fixed_valid_path, moving_valid_path)]
    output_dir = Path(resolve_file_path(str(output_dir), workspace_root))
    if tile_size < 32 or tile_size > 512 or tile_size % 16:
        raise ValueError("Expected tile size 32..512, multiple of 16")
    h = matrix3(moving_to_fixed)
    width, height, channels = _info(fixed_path)
    if channels != 3:
        raise ValueError("Fixed reference must be RGB")
    transform_points(np.linalg.inv(h), [[0, 0], [width, 0], [width, height], [0, height]])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Pair output already exists; choose a new version")
    # Disk maps plus padded TIFF payload and publication safety margin.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    padded = math.ceil(width / tile_size) * math.ceil(height / tile_size) * tile_size**2
    needed = width * height * 4 + padded * 4 + 64 * 1024**2
    if needed > shutil.disk_usage(output_dir.parent).free:
        raise OSError("Insufficient disk for pair maps and tiled TIFF")
    if tile_size**2 * 80 > memory_budget_bytes // 2:
        raise MemoryError("Tile buffers exceed pair memory budget")
    for reference, path in ((fixed_reference, fixed_path), (moving_reference, moving_path)):
        if reference.get("sha256") != sha256_file(path):
            raise ValueError("Original mosaic disagrees with source reference SHA256")
    output_dir.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".serial-pair-", dir=output_dir.parent))
    maps = []
    try:
        aligned = np.memmap(stage / "aligned.raw", mode="w+", dtype=np.uint8, shape=(height, width, 3))
        maps.append(aligned)
        overlap = np.memmap(stage / "overlap.raw", mode="w+", dtype=np.uint8, shape=(height, width))
        maps.append(overlap)
        total = math.ceil(width / tile_size) * math.ceil(height / tile_size)
        done, captured = 0, 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                x2, y2 = min(width, x + tile_size), min(height, y + tile_size)
                _, rgb, valid, _ = inspect_native_pair(
                    fixed_path, moving_path, fixed_valid_path, moving_valid_path,
                    h, (x, y, x2, y2), workspace_root,
                    memory_budget_bytes=memory_budget_bytes)
                aligned[y:y2, x:x2] = rgb
                overlap[y:y2, x:x2] = valid
                captured += int(valid.sum())
                done += 1
                if progress_callback:
                    progress_callback(done, total)
            for mapped in maps:
                _reclaim_memmap_pages(mapped)
        artifacts = []
        for name, mapped in (("HE_aligned_to_THO.tif", aligned), ("capture_overlap.tif", overlap)):
            path = stage / name
            write_tiled_tiff(mapped, str(path), tile_size=tile_size,
                             is_mask=name.startswith("capture"), release_pages=True)
            artifacts.append((str(path), str(output_dir / name)))
        metadata = {"fixed_reference": fixed_reference, "moving_reference": moving_reference,
                    "fixed_capture_mask_sha256": sha256_file(fixed_valid_path),
                    "moving_capture_mask_sha256": sha256_file(moving_valid_path),
                    "fixed_size": [width, height], "HE_native_to_THO_native": h.tolist(),
                    "THO_native_to_HE_native": np.linalg.inv(h).tolist(),
                    "coordinate_system": "THO_native_pixel_centers",
                    "captured_overlap_fraction": captured / (width * height),
                    "capture_mask_meaning": "observation coverage only; not anatomical correspondence",
                    "loss_mask_available": False, "landmark_review": "pending",
                    "trainingAccepted": False, "pixel_ground_truth": False,
                    "status": "globally_aligned_derivative_for_review",
                    "photometricProcessing": "none; geometric bilinear interpolation only",
                    "artifacts": {Path(a).name: sha256_file(a) for a, _ in artifacts}}
        manifest = stage / "pair_manifest.json"
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        artifacts.append((str(manifest), str(output_dir / manifest.name)))
        _publish_staged_artifacts(str(stage), artifacts)
        return metadata
    finally:
        for mapped in maps:
            mapped._mmap.close()
        shutil.rmtree(stage, ignore_errors=True)
