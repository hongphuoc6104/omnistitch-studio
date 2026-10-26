"""Experimental fixed-owner photometry. No geometry/focus/owner modifications."""

import cv2
import numpy as np


def sensor_gain_roi(field, width, height, box):
    x1, y1, x2, y2 = native_box(box)
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) > 1024 * 1024:
        raise MemoryError("Native field ROI exceeds bounded pilot limits")
    if field.ndim != 2 or not np.isfinite(field).all() or (field <= 0).any():
        raise ValueError("Field must be a finite positive 2D gain")
    x, y = np.meshgrid(
        np.arange(x1, x2, dtype=np.float32), np.arange(y1, y2, dtype=np.float32)
    )
    return cv2.remap(
        field.astype(np.float32),
        (x + 0.5) * field.shape[1] / width - 0.5,
        (y + 0.5) * field.shape[0] / height - 0.5,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def render_fixed_crop(manifest, sources, fields, gains, box, owner, reader):
    """Three float RGB crops at fixed original owner IDs and world coordinates."""
    x1, y1, x2, y2 = native_box(box)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0 or h * w > 1024 * 1024:
        raise MemoryError("Crop exceeds bounded pilot limits")
    if owner.shape != (h, w) or owner.dtype.kind != "u":
        raise ValueError("Owner must be an unsigned native crop label map")
    world = np.array(manifest["outputPixelToWorld"]).reshape(3, 3) @ np.stack(
        [*np.meshgrid(np.arange(x1, x2), np.arange(y1, y2)), np.ones((h, w))]
    ).reshape(3, -1)
    result = [np.full((h, w, 3), 255, np.float32) for _ in range(3)]
    by_owner = {s["ownerId"]: s for s in sources}
    world_origin = np.array(manifest["outputPixelToWorld"]).reshape(3, 3) @ np.array(
        [x1, y1, 1.0]
    )
    T = np.array(
        [
            [1, 0, -world_origin[0] / world_origin[2]],
            [0, 1, -world_origin[1] / world_origin[2]],
            [0, 0, 1.0],
        ]
    )
    for oid in np.unique(owner):
        if oid == 0:
            continue
        s = by_owner[int(oid)]
        mask = owner == oid
        p = np.array(s["worldToSource"]).reshape(3, 3) @ world
        sx = (p[0] / p[2]).reshape(h, w)
        sy = (p[1] / p[2]).reshape(h, w)
        rx1 = max(0, int(np.floor(sx[mask].min())) - 3)
        ry1 = max(0, int(np.floor(sy[mask].min())) - 3)
        rx2 = min(s["width"], int(np.ceil(sx[mask].max())) + 4)
        ry2 = min(s["height"], int(np.ceil(sy[mask].max())) + 4)
        raw = reader(s["local_path"], (rx1, ry1, rx2, ry2)).astype(np.float32)
        field = sensor_gain_roi(
            fields[s["id"]], s["width"], s["height"], (rx1, ry1, rx2, ry2)
        )
        H = (
            T
            @ np.array(s["sourceToWorld"]).reshape(3, 3)
            @ np.array([[1, 0, rx1], [0, 1, ry1], [0, 0, 1.0]])
        )
        a = cv2.warpPerspective(
            raw, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        b = cv2.warpPerspective(
            raw * field[:, :, None],
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        result[0][mask] = a[mask]
        result[1][mask] = b[mask]
        result[2][mask] = b[mask] * gains.get(s["id"], 1.0)
    return result


def make_preview(manifest, sources, proxies, fields, gains, owner):
    """Overview only: one source proxy at a time, fixed original ownership."""
    if owner.ndim != 2 or owner.size > 2_000_000 or owner.dtype.kind != "u":
        raise MemoryError("Preview owner map must be a bounded unsigned 2D array")
    h, w = owner.shape
    W, H = manifest["outputWidth"], manifest["outputHeight"]
    out = [np.full((h, w, 3), 255, np.float32) for _ in range(3)]
    for s, proxy in zip(sources, proxies):
        ph, pw = proxy.shape[:2]
        gain = cv2.resize(fields[s["id"]], (pw, ph), interpolation=cv2.INTER_LINEAR)
        matrix = (
            np.diag([w / W, h / H, 1.0])
            @ np.array(manifest["worldToOutputPixel"]).reshape(3, 3)
            @ np.array(s["sourceToWorld"]).reshape(3, 3)
            @ np.diag([s["width"] / pw, s["height"] / ph, 1.0])
        )
        mask = owner == s["ownerId"]
        a = cv2.warpPerspective(
            proxy.astype(np.float32),
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        b = cv2.warpPerspective(
            proxy.astype(np.float32) * gain[:, :, None],
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out[0][mask] = a[mask]
        out[1][mask] = b[mask]
        out[2][mask] = b[mask] * gains.get(s["id"], 1.0)
    return out


def native_box(values):
    """Adapt NumPy integer coordinates to the strict public region-read contract."""
    if len(values) != 4 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in values
    ):
        raise ValueError("Native box requires four integer coordinates")
    return tuple(int(v) for v in values)
