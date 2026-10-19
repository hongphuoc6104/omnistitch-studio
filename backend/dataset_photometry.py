"""Conservative relative exposure proposals from existing overlapping frames.

No blank/dark calibration is assumed. Scalar gains are estimated in camera RGB
code space and normalized to dim only, preventing new highlight clipping. The
proposal is not a physical flat-field or a white-balance/stain normalization.
"""

import math

import cv2
import numpy as np

from .dataset_mosaic import _validate_sources
from .io_utils import read_image_region


def estimate_overlap_scalar_gains(
    sources,
    workspace_root,
    *,
    edges,
    proxy_max_dimension=512,
    samples_per_edge=1024,
    minimum_gain=0.85,
    memory_budget_bytes=512 * 1024 * 1024,
):
    """Return dim-only source gains and held-position photometric diagnostics.

    ``sources`` and ``edges`` use the registered mosaic API format. Alternate
    sample positions fit/evaluate relative log-luminance errors; these are not
    independent biological validation. Missing photometric evidence keeps a
    source at gain one and is reported explicitly.
    """
    if (
        not 64 <= proxy_max_dimension <= 512
        or not 64 <= samples_per_edge <= 4096
        or not 0.5 <= minimum_gain <= 1
    ):
        raise ValueError("Invalid relative exposure configuration")
    edges = tuple((str(a), str(b)) for a, b in edges)
    sources, _ = _validate_sources(sources, workspace_root, edges, ())
    n = len(sources)
    cache_bytes = 0
    for s in sources:
        scale = min(1.0, proxy_max_dimension / max(s["width"], s["height"]))
        s["proxyWidth"], s["proxyHeight"] = (
            max(1, round(s["width"] * scale)),
            max(1, round(s["height"] * scale)),
        )
        cache_bytes += s["proxyWidth"] * s["proxyHeight"] * 5
    if cache_bytes > memory_budget_bytes // 4 or any(
        s["width"] * s["height"] * 3 * 8 + cache_bytes > memory_budget_bytes
        for s in sources
    ):
        raise MemoryError("Exposure proxy/decode cache exceeds memory budget")
    # Dense solve is bounded separately from the image cache.
    if (len(edges) + n) * n * 8 * 4 + cache_bytes > memory_budget_bytes:
        raise MemoryError("Exposure solve exceeds memory budget")
    for s in sources:
        rgb = read_image_region(s["path"], (0, 0, s["width"], s["height"]))
        proxy = cv2.resize(
            rgb, (s["proxyWidth"], s["proxyHeight"]), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(proxy, cv2.COLOR_RGB2GRAY).astype(np.float32)
        s["luma"] = cv2.GaussianBlur(gray, (5, 5), 1)
        s["photometricMask"] = (
            (proxy.max(axis=-1) < 250) & (proxy.min(axis=-1) > 2)
        ).astype(np.uint8)
        del rgb, proxy
    by_id = {s["id"]: i for i, s in enumerate(sources)}
    records = []
    rows = []
    targets = []
    supported = set()
    side = int(math.sqrt(samples_per_edge))
    for aid, bid in edges:
        a, b = sources[by_id[aid]], sources[by_id[bid]]
        x1, y1 = (
            max(a["bounds"][0], b["bounds"][0]),
            max(a["bounds"][1], b["bounds"][1]),
        )
        x2, y2 = (
            min(a["bounds"][2], b["bounds"][2]),
            min(a["bounds"][3], b["bounds"][3]),
        )
        if x2 <= x1 or y2 <= y1:
            continue
        xx, yy = np.meshgrid(
            np.linspace(x1, x2, side + 2)[1:-1], np.linspace(y1, y2, side + 2)[1:-1]
        )
        values = []
        valid = np.ones(xx.shape, dtype=bool)
        for s in [a, b]:
            p = s["inverse"] @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
            sx = (p[0] / p[2]).reshape(xx.shape)
            sy = (p[1] / p[2]).reshape(xx.shape)
            valid &= (
                (sx >= 1) & (sy >= 1) & (sx < s["width"] - 1) & (sy < s["height"] - 1)
            )
            sampled = cv2.remap(
                s["luma"],
                (sx * s["proxyWidth"] / s["width"]).astype(np.float32),
                (sy * s["proxyHeight"] / s["height"]).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            valid &= (sampled > 20) & (sampled < 240)
            valid &= (
                cv2.remap(
                    s["photometricMask"],
                    (sx * s["proxyWidth"] / s["width"]).astype(np.float32),
                    (sy * s["proxyHeight"] / s["height"]).astype(np.float32),
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                > 0
            )
            values.append(sampled)
        log_ratio = np.log(values[1][valid] / values[0][valid])
        if len(log_ratio) < 64:
            continue
        fit, held = log_ratio[::2], log_ratio[1::2]
        row = np.zeros(n)
        row[by_id[aid]] = 1
        row[by_id[bid]] = -1
        weight = math.sqrt(len(fit) / max(samples_per_edge / 2, 1))
        rows.append(row * weight)
        targets.append(float(np.median(fit)) * weight)
        records.append(
            {"source": aid, "target": bid, "sampleCount": len(log_ratio), "held": held}
        )
        supported.update([aid, bid])
    gains = np.ones(n)
    if rows:
        matrix = np.vstack([np.array(rows), 0.05 * np.eye(n)])
        rhs = np.r_[targets, np.zeros(n)]
        log_gains = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        # Normalize each evidenced component; no evidence means no correction.
        neighbors = {key: set() for key in by_id}
        for r in records:
            neighbors[r["source"]].add(r["target"])
            neighbors[r["target"]].add(r["source"])
        pending = set(supported)
        while pending:
            todo = [min(pending)]
            component = set()
            while todo:
                key = todo.pop()
                if key not in component:
                    component.add(key)
                    todo.extend(neighbors[key] - component)
            pending -= component
            indices = [by_id[k] for k in component]
            normalized = log_gains[indices] - max(log_gains[indices])
            gains[indices] = np.exp(np.clip(normalized, math.log(minimum_gain), 0))
    diagnostics = []
    for r in records:
        held = r.pop("held")
        correction = math.log(gains[by_id[r["source"]]]) - math.log(
            gains[by_id[r["target"]]]
        )
        r.update(
            beforeHeldMedianAbsLogRatio=float(np.median(np.abs(held))),
            afterHeldMedianAbsLogRatio=float(np.median(np.abs(held - correction))),
        )
        diagnostics.append(r)
    return {
        "sourceGains": {s["id"]: float(gains[i]) for i, s in enumerate(sources)},
        "method": "overlap_relative_scalar_dim_only",
        "minimumGain": minimum_gain,
        "unsupportedSourceIds": sorted(set(by_id) - supported),
        "proxyBytes": cache_bytes,
        "pairs": diagnostics,
        "units": "camera code-space log luminance ratio; no physical flat-field inferred",
        "trainingAccepted": False,
    }
