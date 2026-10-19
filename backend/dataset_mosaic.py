"""Bounded, provenance-preserving mosaics from registered RGB camera frames.

This opt-in research export does not alter the interactive/legacy blender. Native
Laplacian variance selects one captured observation per output pixel. It is a
texture/noise-sensitive ranking cue, not a biological focus ground truth.
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

from .io_utils import read_image_region, resolve_file_path
from .wsi_exporter import (
    _publish_staged_artifacts,
    _reclaim_memmap_pages,
    write_tiled_tiff,
)


def read_dataset_region(
    path, box, workspace_root, *, memory_budget_bytes=64 * 1024 * 1024
):
    """Read only intersecting uncompressed contiguous TIFF tiles, preserving dtype.

    This reader is for dataset artifacts produced by ``write_tiled_tiff``;
    uint32 owner labels are not converted to display RGB. No full-image fallback
    is used, including for output mosaics larger than the camera decode guard.
    """
    path = resolve_file_path(str(path), workspace_root)
    if len(box) != 4 or any(not isinstance(v, int) or isinstance(v, bool) for v in box):
        raise ValueError("Region box must contain four integers")
    if memory_budget_bytes < 1:
        raise ValueError("Memory budget must be positive")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid region box")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if (
            len(tif.pages) != 1
            or not page.is_tiled
            or page.compression != 1
            or page.planarconfig != 1
            or page.imagedepth != 1
            or page.dtype.kind != "u"
        ):
            raise ValueError(
                "Expected single-plane uncompressed contiguous tiled unsigned TIFF"
            )
        width, height = page.imagewidth, page.imagelength
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        x2, y2 = max(x1, min(width, x2)), max(y1, min(height, y2))
        channels = page.samplesperpixel
        shape = (y2 - y1, x2 - x1) if channels == 1 else (y2 - y1, x2 - x1, channels)
        roi_bytes = (y2 - y1) * (x2 - x1) * channels * page.dtype.itemsize
        tile_bytes = page.tilewidth * page.tilelength * channels * page.dtype.itemsize
        if roi_bytes + tile_bytes * 3 > memory_budget_bytes:
            raise MemoryError("Dataset region decode exceeds memory budget")
        result = np.empty(shape, dtype=page.dtype)
        if x1 == x2 or y1 == y2:
            return result
        columns = math.ceil(width / page.tilewidth)
        for ty in range(y1 // page.tilelength, (y2 - 1) // page.tilelength + 1):
            for tx in range(x1 // page.tilewidth, (x2 - 1) // page.tilewidth + 1):
                index = ty * columns + tx
                if page.databytecounts[index] > tile_bytes:
                    raise ValueError("Uncompressed TIFF tile byte count is invalid")
                tif.filehandle.seek(page.dataoffsets[index])
                raw = tif.filehandle.read(page.databytecounts[index])
                decoded, _, _ = page.decode(raw, index)
                if decoded is None:
                    raise ValueError("Dataset tile is missing")
                tile = decoded.reshape(
                    (page.tilelength, page.tilewidth)
                    if channels == 1
                    else (page.tilelength, page.tilewidth, channels)
                )
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(values, width, height):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 9 or not np.all(np.isfinite(matrix)):
        raise ValueError("sourceToWorld must be a finite 3x3 matrix")
    matrix = matrix.reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("sourceToWorld normalization is invalid")
    matrix = matrix / matrix[2, 2]
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("sourceToWorld is singular")
    corners = matrix @ np.array(
        [[0, width, width, 0], [0, 0, height, height], [1, 1, 1, 1]]
    )
    if np.min(corners[2]) <= 1e-6:
        raise ValueError("Projective horizon intersects source")
    points = (corners[:2] / corners[2]).T
    return matrix, (
        points[:, 0].min(),
        points[:, 1].min(),
        points[:, 0].max(),
        points[:, 1].max(),
    )


def _native_focus_grid(rgb, step, window):
    """Measure variance on native crops; never resize/warp before measuring."""
    height, width = rgb.shape[:2]
    grid = np.empty(
        (math.ceil(height / step), math.ceil(width / step)), dtype=np.float32
    )
    for gy in range(grid.shape[0]):
        y = min(height - 1, int((gy + 0.5) * step))
        for gx in range(grid.shape[1]):
            x = min(width - 1, int((gx + 0.5) * step))
            crop = rgb[
                max(0, y - window // 2) : min(height, y + window // 2),
                max(0, x - window // 2) : min(width, x + window // 2),
            ]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            grid[gy, gx] = cv2.Laplacian(gray, cv2.CV_64F).var()
    return grid


def _camera_info(path):
    """Inspect headers before decoding; reject implicit depth/channel conversion."""
    if Path(path).suffix.lower() not in {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }:
        raise ValueError("Unsupported camera image format")
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as image:
            page = image.pages[0]
            if (
                len(image.pages) != 1
                or page.dtype != np.uint8
                or len(page.shape) != 3
                or page.shape[2] != 3
            ):
                raise ValueError(
                    "Dataset export requires single-plane RGB uint8 camera frames"
                )
            return page.imagewidth, page.imagelength
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Dataset export requires RGB uint8 camera frames")
        return image.size


def _validate_sources(sources, workspace_root, edges, landmarks):
    if not sources or len(sources) > 10000:
        raise ValueError("Supply between 1 and 10000 camera frames")
    ids = [str(source["id"]) for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("Source IDs must be unique")
    neighbors = {key: set() for key in ids}
    for a, b in edges:
        if str(a) not in neighbors or str(b) not in neighbors or str(a) == str(b):
            raise ValueError("Invalid match graph edge")
        neighbors[str(a)].add(str(b))
        neighbors[str(b)].add(str(a))
    visited, stack = set(), [ids[0]]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(neighbors[node] - visited)
    if len(visited) != len(ids):
        raise ValueError("Disconnected match graph; guessed placement is not accepted")
    prepared = []
    for source in sorted(sources, key=lambda source: str(source["id"])):
        path = resolve_file_path(source["path"], workspace_root)
        width, height = _camera_info(path)
        if (int(source["width"]), int(source["height"])) != (width, height):
            raise ValueError("Declared source dimensions do not match file")
        matrix, bounds = _matrix(source["sourceToWorld"], width, height)
        prepared.append(
            {
                "id": str(source["id"]),
                "path": path,
                "width": width,
                "height": height,
                "matrix": matrix,
                "inverse": np.linalg.inv(matrix),
                "bounds": bounds,
            }
        )
    by_id = {source["id"]: source for source in prepared}
    errors = []
    for point in landmarks:
        positions = []
        for suffix in ["a", "b"]:
            source = by_id[str(point["source_" + suffix])]
            xy = np.asarray(point["xy_" + suffix], dtype=np.float64)
            if (
                xy.shape != (2,)
                or not np.all(np.isfinite(xy))
                or not (0 <= xy[0] < source["width"] and 0 <= xy[1] < source["height"])
            ):
                raise ValueError("Landmark lies outside source")
            mapped = source["matrix"] @ np.r_[xy, 1.0]
            positions.append(mapped[:2] / mapped[2])
        errors.append(float(np.linalg.norm(positions[0] - positions[1])))
    qc = {
        "graphConnected": True,
        "sourceCount": len(prepared),
        "landmarkCount": len(errors),
        "landmarkUnits": "world pixels; no physical calibration inferred",
        "status": "needs_independent_landmarks",
        "trainingAccepted": False,
    }
    if errors:
        qc.update(
            landmarkMedian=float(np.median(errors)),
            landmarkP95=float(np.percentile(errors, 95)),
            status="landmarks_within_target"
            if np.median(errors) <= 1 and np.percentile(errors, 95) <= 3
            else "landmark_review_required",
        )
    return prepared, qc


def _render_tile(sources, bounds, box, step, budget, stats):
    """Render a tile plus one-pixel owner halo without a full-canvas array."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    pixel_count = width * height
    tile_bytes = pixel_count * 256 + sum(source["focus"].nbytes for source in sources)
    if tile_bytes > budget:
        raise MemoryError("Tile working set exceeds memory budget")
    yy, xx = np.indices((height, width), dtype=np.float64)
    wx, wy = xx + x1 + bounds[0], yy + y1 + bounds[1]
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    owner = np.zeros((height, width), dtype=np.uint32)
    best = np.full((height, width), -np.inf, dtype=np.float32)
    second = np.full_like(best, -np.inf)
    world_box = (x1 + bounds[0], y1 + bounds[1], x2 + bounds[0], y2 + bounds[1])
    for index, source in enumerate(sources, start=1):
        bx1, by1, bx2, by2 = source["bounds"]
        if (
            bx1 >= world_box[2]
            or bx2 <= world_box[0]
            or by1 >= world_box[3]
            or by2 <= world_box[1]
        ):
            continue
        inverse = source["inverse"]
        denominator = inverse[2, 0] * wx + inverse[2, 1] * wy + inverse[2, 2]
        safe = np.abs(denominator) > 1e-9
        sx = np.divide(
            inverse[0, 0] * wx + inverse[0, 1] * wy + inverse[0, 2],
            denominator,
            out=np.full_like(wx, -1),
            where=safe,
        )
        sy = np.divide(
            inverse[1, 0] * wx + inverse[1, 1] * wy + inverse[1, 2],
            denominator,
            out=np.full_like(wy, -1),
            where=safe,
        )
        valid = (
            safe
            & (sx >= 0)
            & (sy >= 0)
            & (sx < source["width"])
            & (sy < source["height"])
        )
        if not np.any(valid):
            continue
        rx1, ry1 = (
            max(0, int(np.floor(sx[valid].min())) - 2),
            max(0, int(np.floor(sy[valid].min())) - 2),
        )
        rx2 = min(source["width"], int(np.ceil(sx[valid].max())) + 3)
        ry2 = min(source["height"], int(np.ceil(sy[valid].max())) + 3)
        roi_pixels = (rx2 - rx1) * (ry2 - ry1)
        estimated = tile_bytes + roi_pixels * 16
        if estimated > budget:
            raise MemoryError("Source ROI working set exceeds memory budget")
        stats["maxEstimatedWorkingBytes"] = max(
            stats["maxEstimatedWorkingBytes"], estimated
        )
        stats["maxSourceRoiPixels"] = max(stats["maxSourceRoiPixels"], roi_pixels)
        region = read_image_region(source["cache"], (rx1, ry1, rx2, ry2))
        offset = np.array(
            [[1.0, 0.0, -world_box[0]], [0.0, 1.0, -world_box[1]], [0.0, 0.0, 1.0]]
        )
        roi_origin = np.array([[1.0, 0.0, rx1], [0.0, 1.0, ry1], [0.0, 0.0, 1.0]])
        warped = cv2.warpPerspective(
            region,
            offset @ source["matrix"] @ roi_origin,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        score = cv2.remap(
            source["focus"],
            ((sx - step / 2) / step).astype(np.float32),
            ((sy - step / 2) / step).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        edge_distance = np.minimum.reduce(
            [sx, sy, source["width"] - 1 - sx, source["height"] - 1 - sy]
        )
        score = (score + 1e-6) * (0.8 + 0.2 * np.clip(edge_distance / 32, 0, 1))
        score = np.where(valid, score, -np.inf)
        better = score > best
        second = np.where(better, best, np.maximum(second, score))
        best = np.maximum(best, score)
        if source.get("gain", 1.0) == 1.0:
            rgb[better] = warped[better]
        else:
            rgb[better] = np.rint(
                warped[better].astype(np.float32) * source["gain"]
            ).astype(np.uint8)
        owner[better] = index
    valid = owner != 0
    margin = np.zeros((height, width), dtype=np.uint8)
    has_two = valid & np.isfinite(second)
    margin[valid & ~has_two] = 255
    margin[has_two] = np.clip(
        255 * (best[has_two] - second[has_two]) / np.maximum(best[has_two], 1e-6),
        0,
        255,
    ).astype(np.uint8)
    padded = np.pad(owner, 1, mode="edge")
    seam = (
        (owner != padded[:-2, 1:-1])
        | (owner != padded[2:, 1:-1])
        | (owner != padded[1:-1, :-2])
        | (owner != padded[1:-1, 2:])
    ) & valid
    stats["maxTilePixels"] = max(stats["maxTilePixels"], pixel_count)
    return rgb, owner, valid.astype(np.uint8) * 255, margin, seam.astype(np.uint8) * 255


def export_dataset_mosaic(
    sources,
    output_dir,
    workspace_root,
    *,
    edges=(),
    landmarks=(),
    tile_size=512,
    focus_step=128,
    focus_window=256,
    memory_budget_bytes=512 * 1024 * 1024,
    max_output_pixels=2_000_000_000,
    progress_callback=None,
    source_scalar_gains=None,
):
    """Export registered RGB frames and owner/mask maps using native focus scores.

    ``sources`` entries: id, path, width, height, sourceToWorld (row-major 3x3).
    Match edges must connect every source; no guessed component placements.
    Landmark correspondences are supplied separately and never inferred from
    tissue-mask Dice. A successful export does not imply training acceptance.
    This path does not alter color, sharpen, deblur or correct illumination.
    """
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (
            tile_size,
            focus_step,
            focus_window,
            memory_budget_bytes,
            max_output_pixels,
        )
    ):
        raise ValueError("Budgets and sampling sizes must be integers")
    if (
        not 32 <= tile_size <= 2048
        or not 16 <= focus_step <= 512
        or not focus_step <= focus_window <= 1024
        or focus_window % 2
    ):
        raise ValueError("Invalid tile/focus configuration")
    if memory_budget_bytes < 8 * 1024 * 1024:
        raise ValueError("Memory budget must be at least 8 MiB")
    edges = tuple((str(a), str(b)) for a, b in edges)
    landmarks = tuple(landmarks)
    sources, qc = _validate_sources(sources, workspace_root, edges, landmarks)
    gains = {} if source_scalar_gains is None else source_scalar_gains
    if set(gains) - {s["id"] for s in sources}:
        raise ValueError("Exposure gains contain unknown source IDs")
    for source in sources:
        gain = float(gains.get(source["id"], 1.0))
        if not np.isfinite(gain) or not 0.5 <= gain <= 1.0:
            raise ValueError(
                "Exposure gains must be finite dim-only scalars in [0.5,1]"
            )
        source["gain"] = gain
    output_dir = Path(resolve_file_path(str(output_dir), workspace_root))
    bounds = (
        math.floor(min(s["bounds"][0] for s in sources)),
        math.floor(min(s["bounds"][1] for s in sources)),
        math.ceil(max(s["bounds"][2] for s in sources)),
        math.ceil(max(s["bounds"][3] for s in sources)),
    )
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if (
        min(width, height) < 1
        or max(width, height) > 100000
        or width * height > max_output_pixels
    ):
        raise ValueError("Output dimensions exceed configured limit")
    source_bytes = sum(s["width"] * s["height"] * 3 for s in sources)
    focus_bytes = sum(
        math.ceil(s["width"] / focus_step) * math.ceil(s["height"] / focus_step) * 4
        for s in sources
    )
    if focus_bytes > memory_budget_bytes // 8:
        raise MemoryError("Native focus map cache exceeds memory budget")
    if any(
        s["width"] * s["height"] * 3 * 8 + focus_bytes > memory_budget_bytes
        for s in sources
    ):
        raise MemoryError(
            "Camera frame decode exceeds memory budget; use bounded FOV inputs"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(Path(s["path"]).parent == output_dir for s in sources):
        raise ValueError("Output directory must be separate from source directory")
    encoded_tile = math.ceil(tile_size / 16) * 16
    padded_pixels = (
        math.ceil(width / encoded_tile)
        * math.ceil(height / encoded_tile)
        * encoded_tile**2
    )
    required_disk = (
        max(width * height * 24, width * height * 10 + padded_pixels * 10)
        + source_bytes
        + 16 * 1024 * 1024
    )
    if required_disk > shutil.disk_usage(output_dir).free:
        raise OSError("Insufficient disk capacity for staged dataset mosaic")
    stage = Path(tempfile.mkdtemp(prefix=".dataset.", dir=output_dir))
    maps = []
    stats = {
        "memoryBudgetBytes": memory_budget_bytes,
        "maxEstimatedWorkingBytes": 0,
        "maxSourceRoiPixels": 0,
        "maxTilePixels": 0,
        "tileCount": 0,
        "estimatedDiskBytes": required_disk,
        "sourceDecodeLimitBytes": (memory_budget_bytes - focus_bytes) // 8,
    }
    try:
        provenance = []
        for index, source in enumerate(sources, start=1):
            rgb = read_image_region(
                source["path"], (0, 0, source["width"], source["height"])
            )
            if rgb.dtype != np.uint8 or rgb.shape != (
                source["height"],
                source["width"],
                3,
            ):
                raise ValueError("Decoded source violates RGB uint8 shape contract")
            source["cache"] = str(stage / f"source_{index}.tif")
            tifffile.imwrite(
                source["cache"], rgb, photometric="rgb", compression=None, metadata=None
            )
            source["focus"] = _native_focus_grid(rgb, focus_step, focus_window)
            provenance.append(
                {
                    "ownerId": index,
                    "id": source["id"],
                    "sourcePath": source["path"],
                    "sourceSha256": _sha256(source["path"]),
                    "decodedRgbSha256": hashlib.sha256(
                        memoryview(rgb).cast("B")
                    ).hexdigest(),
                    "width": source["width"],
                    "height": source["height"],
                    "sourceToWorld": source["matrix"].ravel().tolist(),
                    "worldToSource": source["inverse"].ravel().tolist(),
                    "scalarGain": source["gain"],
                }
            )
            del rgb
        if sum(s["focus"].nbytes for s in sources) > memory_budget_bytes // 8:
            raise MemoryError("Native focus map cache exceeds memory budget")
        specs = [
            ("mosaic", np.uint8, (height, width, 3)),
            ("owner", np.uint32, (height, width)),
            ("valid", np.uint8, (height, width)),
            ("quality_margin", np.uint8, (height, width)),
            ("seam", np.uint8, (height, width)),
        ]
        for name, dtype, shape in specs:
            maps.append(
                np.memmap(stage / (name + ".dat"), mode="w+", dtype=dtype, shape=shape)
            )
        tile_total = math.ceil(width / tile_size) * math.ceil(height / tile_size)
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                x2, y2 = min(width, x + tile_size), min(height, y + tile_size)
                ex1, ey1, ex2, ey2 = (
                    max(0, x - 1),
                    max(0, y - 1),
                    min(width, x2 + 1),
                    min(height, y2 + 1),
                )
                rendered = _render_tile(
                    sources,
                    bounds,
                    (ex1, ey1, ex2, ey2),
                    focus_step,
                    memory_budget_bytes,
                    stats,
                )
                for mapped, tile in zip(maps, rendered):
                    mapped[y:y2, x:x2] = tile[y - ey1 : y2 - ey1, x - ex1 : x2 - ex1]
                stats["tileCount"] += 1
                if progress_callback:
                    progress_callback(stats["tileCount"], tile_total)
            for mapped in maps:
                _reclaim_memmap_pages(mapped)
        artifacts = []
        for (name, _, _), mapped in zip(specs, maps):
            path = stage / (name + ".tif")
            write_tiled_tiff(
                mapped,
                str(path),
                tile_size=tile_size,
                is_mask=name != "mosaic",
                release_pages=True,
            )
            artifacts.append((str(path), str(output_dir / path.name)))
        metadata = {
            "version": 1,
            "method": "native_laplacian_single_source",
            "outputWidth": width,
            "outputHeight": height,
            "outputPixelToWorld": [1, 0, bounds[0], 0, 1, bounds[1], 0, 0, 1],
            "worldToOutputPixel": [1, 0, -bounds[0], 0, 1, -bounds[1], 0, 0, 1],
            "focusStep": focus_step,
            "focusWindow": focus_window,
            "qualityMarginMeaning": "relative score gap, not calibrated confidence; single coverage=255",
            "photometricProcessing": "dim-only scalar RGB after interpolation; native owner fixed"
            if source_scalar_gains is not None
            else "none; RGB from one observation, geometric interpolation only",
            "geometryQC": qc,
            "matchGraphEdges": edges,
            "landmarks": [
                {
                    "source_a": str(p["source_a"]),
                    "source_b": str(p["source_b"]),
                    "xy_a": np.asarray(p["xy_a"], dtype=float).tolist(),
                    "xy_b": np.asarray(p["xy_b"], dtype=float).tolist(),
                }
                for p in landmarks
            ],
            "sources": provenance,
            "renderStats": stats,
            "artifacts": {Path(a).name: _sha256(a) for a, _ in artifacts},
        }
        (stage / "manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        artifacts.append(
            (str(stage / "manifest.json"), str(output_dir / "manifest.json"))
        )
        _publish_staged_artifacts(str(stage), artifacts)
        return metadata
    finally:
        for mapped in maps:
            mapped._mmap.close()
        shutil.rmtree(stage, ignore_errors=True)
