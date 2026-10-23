"""Bounded review sampling with a separate THO-proxy backward residual field.

Global transforms remain 3×3. A residual field never substitutes for that
matrix. Native H&E RGB is sampled once from its original mosaic; proxy previews
and deformation/Jacobian diagnostics are not training acceptance or anatomy GT.
"""

import hashlib

import cv2
import numpy as np

from .dataset_mosaic import read_dataset_region
from .io_utils import resolve_file_path
from .serial_pairing import _info, matrix3, transform_points


def validate_residual_field(field, *, memory_budget_bytes=64 * 1024**2,
                            maximum_displacement_proxy_px=64.):
    """Check a bounded float32 field and sampled I+∇u Jacobian in small rows.

    Field[...,0:2] is additive x,y displacement in THO proxy pixels, mapping
    fixed output locations to locations in the globally aligned H&E frame.
    Positive sampled determinants are a geometry check, never an anatomy check.
    """
    if (not isinstance(field, np.ndarray) or field.dtype != np.float32
            or field.ndim != 3 or field.shape[2] != 2
            or min(field.shape[:2]) < 8 or max(field.shape[:2]) > 1600
            or not field.flags.c_contiguous):
        raise ValueError("Expected contiguous float32 HxWx2 bounded proxy field")
    if field.nbytes + 66 * field.shape[1] * 4 * 16 > memory_budget_bytes:
        raise MemoryError("Residual field validation exceeds memory budget")
    if (not np.isfinite(field).all() or not np.isfinite(maximum_displacement_proxy_px)
            or maximum_displacement_proxy_px <= 0):
        raise ValueError("Invalid residual field values/limit")
    low, high, folds, maximum = float("inf"), -float("inf"), 0, 0.
    height = field.shape[0]
    for y in range(0, height, 64):
        bottom = min(height, y + 64)
        top, stop = max(0, y - 1), min(height, bottom + 1)
        uy, ux = np.gradient(field[top:stop, :, 0])
        vy, vx = np.gradient(field[top:stop, :, 1])
        jac = ((1 + ux) * (1 + vy) - uy * vx)[y - top:bottom - top]
        if not np.isfinite(jac).all():
            raise ValueError("Nonfinite sampled field Jacobian")
        low, high = min(low, float(jac.min())), max(high, float(jac.max()))
        folds += int(np.count_nonzero(jac <= 0))
        displacement = np.linalg.norm(field[y:bottom], axis=2)
        maximum = max(maximum, float(displacement.max()))
    if folds:
        raise ValueError("Folding residual field rejected")
    if maximum > maximum_displacement_proxy_px:
        raise ValueError("Residual displacement exceeds configured proxy-pixel limit")
    return {"raw_field_sha256": hashlib.sha256(memoryview(field).cast("B")).hexdigest(),
            "field_proxy_size": [field.shape[1], field.shape[0]],
            "sampled_jacobian_min": low, "sampled_jacobian_max": high,
            "sampled_fold_pixels": folds, "max_displacement_proxy_px": maximum,
            "interpretation": "sampled geometry validity only; not biological correspondence"}


def _backward(field, global_he_to_tho, fixed_native_size, points):
    h = matrix3(global_he_to_tho)
    if not np.allclose(h[2], [0, 0, 1.]):
        raise ValueError("Residual renderer supports affine globals only; never truncate perspective")
    if (len(fixed_native_size) != 2
            or any(type(v) is not int or v <= 0 for v in fixed_native_size)):
        raise ValueError("Invalid fixed native size")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 fixed native points")
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    if np.any(points < 0) or np.any(points > np.asarray(fixed_native_size) - 1):
        raise ValueError("Fixed points outside native canvas")
    scale = np.asarray(fixed_native_size, dtype=float) / [field.shape[1], field.shape[0]]
    proxy = (points + .5) / scale - .5
    columns = min(1024, len(points))
    rows = (len(points) + columns - 1) // columns
    padding = rows * columns - len(points)
    mapx = np.pad(proxy[:, 0].astype(np.float32), (0, padding), mode="edge").reshape(rows, columns)
    mapy = np.pad(proxy[:, 1].astype(np.float32), (0, padding), mode="edge").reshape(rows, columns)
    displacement = cv2.remap(field, mapx, mapy, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)[:len(points)]
    global_native = (proxy + displacement + .5) * scale - .5
    return transform_points(np.linalg.inv(h), global_native)


def backward_native_points(field, global_he_to_tho, fixed_native_size, points, *,
                            memory_budget_bytes=64 * 1024**2):
    """Map fixed THO-native points directly to original H&E-native samples."""
    validate_residual_field(field, memory_budget_bytes=memory_budget_bytes)
    if len(points) * 160 + field.nbytes > memory_budget_bytes:
        raise MemoryError("Backward point sampling exceeds memory budget")
    return _backward(field, global_he_to_tho, fixed_native_size, points)


def inspect_deformed_native_pair(fixed_path, moving_path, fixed_valid_path, moving_valid_path,
                                  global_he_to_tho, field, box, workspace_root, *,
                                  memory_budget_bytes=64 * 1024**2):
    """Compose global affine and separate residual, sampling original HE once.

    The small field spans the complete THO proxy canvas, using the same half-
    pixel convention as the fixed-grid sampler. Half-pixel edge extensions use
    replicated field boundaries. Capture overlap is not an anatomical loss mask.
    """
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Expected built-in integer native ROI")
    validation = validate_residual_field(field, memory_budget_bytes=memory_budget_bytes)
    paths = [resolve_file_path(str(path), workspace_root) for path in
             (fixed_path, moving_path, fixed_valid_path, moving_valid_path)]
    fw, fh, fc = _info(paths[0])
    mw, mh, mc = _info(paths[1])
    if fc != 3 or mc != 3 or _info(paths[2]) != (fw, fh, 1) or _info(paths[3]) != (mw, mh, 1):
        raise ValueError("Invalid native RGB/capture-mask dimensions")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= fw and 0 <= y1 < y2 <= fh):
        raise ValueError("Native ROI outside THO canvas")
    pixels = (x2 - x1) * (y2 - y1)
    if pixels * 160 + field.nbytes + 8 * 1024**2 > memory_budget_bytes:
        raise MemoryError("Native deformation ROI exceeds memory budget")
    xs, ys = np.meshgrid(np.arange(x1, x2, dtype=float), np.arange(y1, y2, dtype=float))
    moving_points = _backward(field, global_he_to_tho, [fw, fh], np.column_stack((xs.ravel(), ys.ravel())))
    # Nonlinear extrema may lie inside a tile, so bound ALL mapped samples.
    left = max(0, min(mw, int(np.floor(moving_points[:, 0].min())) - 2))
    top = max(0, min(mh, int(np.floor(moving_points[:, 1].min())) - 2))
    right = max(left, min(mw, int(np.ceil(moving_points[:, 0].max())) + 3))
    bottom = max(top, min(mh, int(np.ceil(moving_points[:, 1].max())) + 3))
    if (pixels * 160 + field.nbytes + (right - left) * (bottom - top) * 12
            + 8 * 1024**2 > memory_budget_bytes):
        raise MemoryError("Mapped source support exceeds deformation budget")
    if max(right - left, bottom - top) >= 32767:
        raise ValueError("Mapped source support exceeds remap dimension limit")
    fixed = read_dataset_region(paths[0], tuple(box), workspace_root,
                                memory_budget_bytes=memory_budget_bytes // 2)
    capture = read_dataset_region(paths[2], tuple(box), workspace_root,
                                  memory_budget_bytes=memory_budget_bytes // 2) > 0
    aligned, observed = np.full_like(fixed, 255), np.zeros(capture.shape, np.uint8)
    if right > left and bottom > top:
        moving = read_dataset_region(paths[1], (left, top, right, bottom), workspace_root,
                                     memory_budget_bytes=memory_budget_bytes // 2)
        valid = read_dataset_region(paths[3], (left, top, right, bottom), workspace_root,
                                    memory_budget_bytes=memory_budget_bytes // 2)
        mx = (moving_points[:, 0] - left).astype(np.float32).reshape(capture.shape)
        my = (moving_points[:, 1] - top).astype(np.float32).reshape(capture.shape)
        aligned = cv2.remap(moving, mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        observed = cv2.remap((valid > 0).astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return fixed, aligned, (capture & (observed > 0)).astype(np.uint8), {
        "fixed_roi_xyxy": list(box), "moving_support_xyxy": [left, top, right, bottom],
        "global_HE_native_to_THO_native": matrix3(global_he_to_tho).tolist(),
        "field_direction": "THO proxy -> globally aligned HE proxy, additive proxy-pixel displacement",
        "fixed_native_size": [fw, fh], "residual_geometry": validation,
        "interpolation": "single original HE sampling after composition; no repeated RGB warp",
        "mask_meaning": "capture intersection only; not anatomical correspondence",
        "trainingAccepted": False, "pixel_ground_truth": False}
