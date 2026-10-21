"""P3 exposure-only fitting and bounded, fixed-owner derivative TIFF export.

No registration, source selection, field estimation, sharpening or blending.
Float32 Deflate TIFF retains over-range RGB values instead of clipping channels.
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
from PIL import Image

from .dataset_mosaic import _matrix, read_dataset_region
from .io_utils import resolve_file_path
from .wsi_exporter import _publish_staged_artifacts


def estimate_p3_gains(manifest, proxies, *, memory_budget_bytes=256 * 1024 * 1024):
    """Reproduce the approved pilot's identity-field, median-anchored exposure fit.

    Proxies are raw RGB uint8, shape (N,216,384,3), in manifest source order.
    Alternate overlap positions fit/evaluate; this is not independent biological
    validation. Unsupported observations keep gain one. No physical calibration.
    """
    sources = manifest["sources"]
    n = len(sources)
    if n > 1000 or type(memory_budget_bytes) is not int or memory_budget_bytes < 1:
        raise ValueError("Invalid source count or memory budget")
    for source in sources:
        matrix, _ = _matrix(source["sourceToWorld"], source["width"], source["height"])
        inverse = np.asarray(source["worldToSource"], dtype=float).reshape(3, 3)
        if not np.isfinite(inverse).all() or not np.allclose(
            matrix @ inverse, np.eye(3), atol=1e-7
        ):
            raise ValueError("Inconsistent source coordinate transforms")
    if not n or proxies.shape != (n, 216, 384, 3) or proxies.dtype != np.uint8:
        raise ValueError("Expected one raw 384x216 RGB uint8 proxy per source")
    if n * 384 * 216 * 16 + n * n * 8 + 160 * n * 32 > memory_budget_bytes:
        raise MemoryError("Exposure feature/solve cache exceeds budget")
    ids = {s["id"]: i for i, s in enumerate(sources)}
    if len(ids) != n:
        raise ValueError("Source IDs must be unique")
    lumas, masks = [], []
    for rgb in proxies:
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lumas.append(cv2.GaussianBlur(y, (5, 5), 1))
        masks.append(
            ((rgb.max(-1) < 250) & (rgb.min(-1) > 2) & (y > 20) & (y < 240)).astype(
                np.uint8
            )
        )
    edges = manifest["matchGraphEdges"]
    edges = (
        [
            edges[i]
            for i in np.unique(
                np.linspace(0, len(edges) - 1, min(160, len(edges))).astype(int)
            )
        ]
        if edges
        else []
    )
    records, A, rhs = [], [], []
    for aid, bid in edges:
        if aid not in ids or bid not in ids:
            raise ValueError("Edge references an unknown source")
        a, b = sources[ids[aid]], sources[ids[bid]]
        corners = np.array(
            [
                [0, 0, 1],
                [a["width"], 0, 1],
                [0, a["height"], 1],
                [a["width"], a["height"], 1],
            ],
            float,
        ).T
        q = np.array(a["sourceToWorld"]).reshape(3, 3) @ corners
        q = q[:2] / q[2]
        xx, yy = np.meshgrid(
            np.linspace(q[0].min(), q[0].max(), 26)[1:-1],
            np.linspace(q[1].min(), q[1].max(), 26)[1:-1],
        )
        grid = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
        valid = np.ones(xx.size, bool)
        samples = []
        for s in [a, b]:
            i = ids[s["id"]]
            p = np.array(s["worldToSource"]).reshape(3, 3) @ grid
            sx, sy = p[0] / p[2], p[1] / p[2]
            valid &= (
                (sx >= 2) & (sy >= 2) & (sx < s["width"] - 2) & (sy < s["height"] - 2)
            )
            mx = ((sx + 0.5) * 384 / s["width"] - 0.5).astype(np.float32).reshape(1, -1)
            my = (
                ((sy + 0.5) * 216 / s["height"] - 0.5).astype(np.float32).reshape(1, -1)
            )
            valid &= (
                cv2.remap(
                    masks[i], mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
                ).ravel()
                > 0
            )
            samples.append(
                cv2.remap(
                    lumas[i], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                ).ravel()
            )
        if valid.sum() < 64:
            continue
        ratio = np.log(
            np.maximum(samples[1][valid], 1) / np.maximum(samples[0][valid], 1)
        )
        row = np.zeros(n)
        row[ids[aid]], row[ids[bid]] = 1, -1
        A.append(row)
        rhs.append(float(np.median(ratio[::2])))
        records.append({"a": aid, "b": bid, "held": ratio[1::2]})
    logs = np.zeros(n)
    if A:
        logs = np.linalg.lstsq(
            np.vstack([A, 0.05 * np.eye(n)]), np.r_[rhs, np.zeros(n)], rcond=None
        )[0]
        logs -= np.median(logs)
        logs = np.clip(logs, math.log(0.75), math.log(4 / 3))
    supported = {r[k] for r in records for k in ["a", "b"]}
    gains = {
        s["id"]: float(np.exp(logs[i])) if s["id"] in supported else 1.0
        for i, s in enumerate(sources)
    }

    neighbors = {key: set() for key in supported}
    for record in records:
        neighbors[record["a"]].add(record["b"])
        neighbors[record["b"]].add(record["a"])
    pending = set(supported)
    components = []
    while pending:
        todo, component = [min(pending)], set()
        while todo:
            key = todo.pop()
            if key not in component:
                component.add(key)
                todo.extend(neighbors[key] - component)
        pending -= component
        components.append(sorted(component))

    def median(arrays):
        return float(np.median(np.concatenate(arrays))) if arrays else None

    return {
        "gains": gains,
        "diagnostics": {
            "photometric_components": components,
            "evaluated_edges": len(records),
            "held_positions": sum(len(r["held"]) for r in records),
            "before_abs_log_ratio": median([abs(r["held"]) for r in records]),
            "after_abs_log_ratio": median(
                [
                    abs(r["held"] - (math.log(gains[r["a"]]) - math.log(gains[r["b"]])))
                    for r in records
                ]
            ),
            "gain_min_max": [min(gains.values()), max(gains.values())],
            "unsupported_sources": sorted(set(ids) - supported),
            "method": "P3 pilot: median anchor, gains [0.75,4/3], 160 edges, alternate held positions",
            "validation_limit": "Same-overlap held positions, not independent biological validation",
        },
    }


def read_exposure_region(
    path, box, workspace_root, *, memory_budget_bytes=64 * 1024 * 1024
):
    """Bounded tile reads of this module's Deflate float32 RGB TIFFs."""
    path = resolve_file_path(str(path), workspace_root)
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Region box must contain four built-in integers")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid region box")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if (
            len(tif.pages) != 1
            or not page.is_tiled
            or page.dtype != np.float32
            or page.samplesperpixel != 3
            or page.planarconfig != 1
            or page.compression not in (1, 8, 32946)
        ):
            raise ValueError("Expected single-plane tiled float32 RGB exposure TIFF")
        x1, x2 = max(0, min(page.imagewidth, x1)), max(0, min(page.imagewidth, x2))
        y1, y2 = max(0, min(page.imagelength, y1)), max(0, min(page.imagelength, y2))
        tile_bytes = page.tilewidth * page.tilelength * 12
        if (x2 - x1) * (y2 - y1) * 12 + tile_bytes * 4 > memory_budget_bytes:
            raise MemoryError("Exposure crop exceeds decode budget")
        result = np.empty((y2 - y1, x2 - x1, 3), np.float32)
        if x1 == x2 or y1 == y2:
            return result
        columns = math.ceil(page.imagewidth / page.tilewidth)
        for ty in range(y1 // page.tilelength, (y2 - 1) // page.tilelength + 1):
            for tx in range(x1 // page.tilewidth, (x2 - 1) // page.tilewidth + 1):
                index = ty * columns + tx
                if page.databytecounts[index] > tile_bytes + 65536:
                    raise ValueError("Encoded tile exceeds bounded size")
                tif.filehandle.seek(page.dataoffsets[index])
                raw = tif.filehandle.read(page.databytecounts[index])
                decoded, _, _ = page.decode(raw, index)
                if decoded is None:
                    raise ValueError("Exposure tile missing")
                tile = decoded.reshape(page.tilelength, page.tilewidth, 3)
                ox, oy = tx * page.tilewidth, ty * page.tilelength
                ax, ay, bx, by = (
                    max(x1, ox),
                    max(y1, oy),
                    min(x2, ox + page.tilewidth),
                    min(y2, oy + page.tilelength),
                )
                result[ay - y1 : by - y1, ax - x1 : bx - x1] = tile[
                    ay - oy : by - oy, ax - ox : bx - ox
                ]
        return result


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_p3_mosaic(
    baseline_path,
    owner_path,
    manifest,
    gains,
    output_dir,
    workspace_root,
    *,
    tile_size=512,
    memory_budget_bytes=128 * 1024 * 1024,
    progress_callback=None,
):
    """Multiply baseline pixels by their immutable source gain; stream float TIFF.

    Units remain camera RGB code values. Values above 255 are retained, not
    channel-clipped. JPEG is display only. No new geometric interpolation.
    Original owner/mask/transform artifacts are referenced, never modified.
    """
    if (
        type(tile_size) is not int
        or tile_size < 32
        or tile_size > 1024
        or tile_size % 16
    ):
        raise ValueError("Tile size must be a multiple of 16 within 32..1024")
    if tile_size * tile_size * 112 + 16 * 1024 * 1024 > memory_budget_bytes:
        raise MemoryError("Streaming TIFF working set exceeds budget")
    baseline_path = Path(resolve_file_path(str(baseline_path), workspace_root))
    owner_path = Path(resolve_file_path(str(owner_path), workspace_root))
    output_dir = Path(resolve_file_path(str(output_dir), workspace_root))
    if output_dir in [baseline_path.parent, owner_path.parent]:
        raise ValueError("Output must be separate from baseline")
    width, height = manifest["outputWidth"], manifest["outputHeight"]
    sources = manifest["sources"]
    if not sources or set(gains) != {s["id"] for s in sources}:
        raise ValueError("Supply exactly one gain per manifest source")
    owner_ids = [s["ownerId"] for s in sources]
    if (
        min(owner_ids) < 1
        or max(owner_ids) > 1000
        or len(set(owner_ids)) != len(owner_ids)
    ):
        raise ValueError("Invalid ownership ID mapping")
    lookup = np.full(max(owner_ids) + 1, np.nan, np.float32)
    lookup[0] = 1
    for s in sources:
        gain = float(gains[s["id"]])
        if not math.isfinite(gain) or not 0.75 - 1e-7 <= gain <= 4 / 3 + 1e-7:
            raise ValueError("Gain outside approved P3 range")
        lookup[s["ownerId"]] = gain
    for path, channels, dtype in [
        (baseline_path, 3, np.uint8),
        (owner_path, 1, np.uint32),
    ]:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if (page.imagewidth, page.imagelength, page.samplesperpixel) != (
                width,
                height,
                channels,
            ) or page.dtype != dtype:
                raise ValueError(
                    "Baseline or owner dimensions/dtype do not match manifest"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    padded = math.ceil(width / tile_size) * math.ceil(height / tile_size) * tile_size**2
    if padded * 12 + 32 * 1024 * 1024 > shutil.disk_usage(output_dir).free:
        raise OSError("Insufficient staging disk for worst-case float TIFF")
    stage = Path(tempfile.mkdtemp(prefix=".p3.", dir=output_dir))
    pw, ph = (
        max(1, round(width * min(1, 1200 / max(width, height)))),
        max(1, round(height * min(1, 1200 / max(width, height)))),
    )
    preview = Image.new("RGB", (pw, ph), "white")
    stats = {"tiles": 0, "pixels": 0, "display_over_range_pixels": 0}
    total = math.ceil(width / tile_size) * math.ceil(height / tile_size)

    def tiles():
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                x2, y2 = min(width, x + tile_size), min(height, y + tile_size)
                rgb = read_dataset_region(
                    baseline_path,
                    (x, y, x2, y2),
                    workspace_root,
                    memory_budget_bytes=memory_budget_bytes // 2,
                )
                owner = read_dataset_region(
                    owner_path,
                    (x, y, x2, y2),
                    workspace_root,
                    memory_budget_bytes=memory_budget_bytes // 2,
                )
                if owner.max() >= len(lookup) or np.isnan(lookup[owner]).any():
                    raise ValueError("Owner contains an unmapped source ID")
                corrected = rgb.astype(np.float32) * lookup[owner, None]
                stats["pixels"] += corrected.shape[0] * corrected.shape[1]
                stats["display_over_range_pixels"] += int(
                    (corrected > 255).any(-1).sum()
                )
                l, t, r, b = (
                    round(x * pw / width),
                    round(y * ph / height),
                    round(x2 * pw / width),
                    round(y2 * ph / height),
                )
                if r > l and b > t:
                    image = Image.fromarray(
                        np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
                    )
                    preview.paste(
                        image.resize((r - l, b - t), Image.Resampling.LANCZOS), (l, t)
                    )
                tile = np.full((tile_size, tile_size, 3), 255, np.float32)
                tile[: y2 - y, : x2 - x] = corrected
                stats["tiles"] += 1
                if progress_callback:
                    progress_callback(stats["tiles"], total)
                yield tile

    try:
        tifffile.imwrite(
            stage / "mosaic.tif",
            data=tiles(),
            shape=(height, width, 3),
            dtype=np.float32,
            tile=(tile_size, tile_size),
            bigtiff=True,
            photometric="rgb",
            compression="deflate",
            maxworkers=1,
            buffersize=8 * 1024 * 1024,
            metadata=None,
        )
        preview.save(stage / "preview.jpg", quality=90)
        stats["display_over_range_fraction"] = (
            stats["display_over_range_pixels"] / stats["pixels"]
        )
        metadata = {
            "method": "P3 exposure-only, fixed baseline pixels and owners",
            "outputWidth": width,
            "outputHeight": height,
            "outputPixelToWorld": manifest["outputPixelToWorld"],
            "worldToOutputPixel": manifest["worldToOutputPixel"],
            "units": "float32 camera RGB code values; do not infer 0..1 or clip before analysis",
            "source_gains": gains,
            "source_manifest": manifest,
            "owner_reference": str(owner_path),
            "owner_sha256": _sha(owner_path),
            "baseline_sha256": _sha(baseline_path),
            "renderStats": stats,
            "trainingAccepted": False,
            "artifacts": {
                name: _sha(stage / name) for name in ["mosaic.tif", "preview.jpg"]
            },
        }
        (stage / "exposure_manifest.json").write_text(json.dumps(metadata, indent=2))
        _publish_staged_artifacts(
            str(stage),
            [
                (str(stage / name), str(output_dir / name))
                for name in ["mosaic.tif", "preview.jpg", "exposure_manifest.json"]
            ],
        )
        return metadata
    finally:
        shutil.rmtree(stage, ignore_errors=True)
