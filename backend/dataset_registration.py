"""Camera-frame registration using bounded sequential decodes and proxy features.

No source RGB collection or full mosaic canvas is retained. Correspondences and
RANSAC residuals are measured in native source pixels. Disconnected graphs fail
instead of synthesizing placements. Tree registration still needs independent
landmarks/loop review before training acceptance.
"""

import cv2
import numpy as np

from .dataset_mosaic import _camera_info, _matrix
from .feature_engine import FeatureEngine
from .io_utils import read_image_region, resolve_file_path
from .matcher import FeatureMatcher


def register_camera_frames(
    sources,
    workspace_root,
    *,
    proxy_max_dimension=1536,
    max_features=2000,
    min_inliers=16,
    memory_budget_bytes=512 * 1024 * 1024,
    max_pairs=6000,
    progress_callback=None,
):
    """Return sources/edges/diagnostics for ``export_dataset_mosaic``.

    Input entries contain unique id/path; optional width/height are verified.
    All-pairs proposals are intentional without stage metadata; feature count,
    decode size and retained descriptor bytes have explicit limits. Brightness
    enhancement is applied only inside the feature detector, never to RGB data.
    """
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (
            proxy_max_dimension,
            max_features,
            min_inliers,
            memory_budget_bytes,
            max_pairs,
        )
    ):
        raise ValueError("Registration limits must be integers")
    if len(sources) * (len(sources) - 1) // 2 > max_pairs or max_pairs < 0:
        raise ValueError("Registration pair budget exceeded")
    if (
        not sources
        or len(sources) > 1000
        or len({str(s["id"]) for s in sources}) != len(sources)
    ):
        raise ValueError("Supply 1..1000 sources with unique IDs")
    if (
        not 128 <= proxy_max_dimension <= 2048
        or not 64 <= max_features <= 8000
        or not 8 <= min_inliers <= max_features
    ):
        raise ValueError("Invalid registration feature configuration")
    if memory_budget_bytes < 8 * 1024 * 1024:
        raise ValueError("Memory budget must be at least 8 MiB")
    detector = FeatureEngine(
        method="sift", max_features=max_features, enable_clahe=True
    )
    matcher = FeatureMatcher(
        method="sift", ratio_threshold=0.75, min_inliers=min_inliers
    )
    features = []
    descriptors_bytes = 0
    max_decode_bytes = 0
    for source in sorted(sources, key=lambda s: str(s["id"])):
        path = resolve_file_path(source["path"], workspace_root)
        width, height = _camera_info(path)
        if ("width" in source and source["width"] != width) or (
            "height" in source and source["height"] != height
        ):
            raise ValueError("Declared source dimensions do not match file")
        decode_bytes = width * height * 3
        if decode_bytes * 8 + descriptors_bytes > memory_budget_bytes:
            raise MemoryError("Camera frame decode exceeds registration memory budget")
        max_decode_bytes = max(max_decode_bytes, decode_bytes)
        rgb = read_image_region(path, (0, 0, width, height))
        factor = min(1.0, proxy_max_dimension / max(width, height))
        pw, ph = max(1, round(width * factor)), max(1, round(height * factor))
        proxy = (
            cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
            if factor < 1
            else rgb
        )
        keypoints, desc, _ = detector.detect_and_compute(proxy)
        keypoints = keypoints[:max_features]
        desc = desc[:max_features].copy() if desc is not None else None
        points = np.array(
            [(k.pt[0] * width / pw, k.pt[1] * height / ph) for k in keypoints],
            dtype=np.float32,
        ).reshape(-1, 2)
        descriptors_bytes += points.nbytes + (desc.nbytes if desc is not None else 0)
        if (
            descriptors_bytes > memory_budget_bytes // 3
            or max_decode_bytes * 8 + descriptors_bytes > memory_budget_bytes
        ):
            raise MemoryError("Descriptor cache exceeds registration memory budget")
        features.append(
            {
                "id": str(source["id"]),
                "path": path,
                "width": width,
                "height": height,
                "points": points,
                "desc": desc,
            }
        )
        del rgb, proxy, keypoints
    transforms = {}
    accepted = []
    proposed = 0
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            proposed += 1
            matches = matcher.match_pair(features[i]["desc"], features[j]["desc"])
            if len(matches) >= min_inliers:
                a = np.array([features[i]["points"][m.queryIdx] for m in matches])
                b = np.array([features[j]["points"][m.trainIdx] for m in matches])
                affine, inliers = cv2.estimateAffinePartial2D(
                    b,
                    a,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.5,
                    maxIters=3000,
                    confidence=0.995,
                )
                if affine is not None and inliers is not None:
                    keep = inliers.ravel().astype(bool)
                    count = int(keep.sum())
                    H = np.eye(3)
                    H[:2] = affine
                    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
                    if (
                        count >= min_inliers
                        and count / len(matches) >= 0.2
                        and 0.85 <= scale <= 1.15
                    ):
                        residual = np.linalg.norm(
                            cv2.transform(b[None], affine)[0] - a, axis=1
                        )
                        transforms[(i, j)] = H
                        transforms[(j, i)] = np.linalg.inv(H)
                        accepted.append(
                            {
                                "source": features[i]["id"],
                                "target": features[j]["id"],
                                "inliers": count,
                                "matchCount": len(matches),
                                "score": count * count / len(matches),
                                "inlierResidualMedianNativePx": float(
                                    np.median(residual[keep])
                                ),
                                "inlierResidualP95NativePx": float(
                                    np.percentile(residual[keep], 95)
                                ),
                            }
                        )
            if progress_callback:
                progress_callback(proposed, n * (n - 1) // 2)
    visited = {0}
    world = {0: np.eye(3)}
    tree = []
    by_id = {f["id"]: i for i, f in enumerate(features)}
    while len(visited) < n:
        candidates = []
        for edge in accepted:
            a, b = by_id[edge["source"]], by_id[edge["target"]]
            if (a in visited) != (b in visited):
                candidates.append((edge["score"], a, b))
        if not candidates:
            missing = [features[i]["id"] for i in range(n) if i not in visited]
            raise ValueError(
                "Disconnected match graph; unresolved source IDs: " + ",".join(missing)
            )
        _, a, b = max(candidates, key=lambda c: (c[0], -c[1], -c[2]))
        if a not in visited:
            a, b = b, a
        world[b] = world[a] @ transforms[(a, b)]
        visited.add(b)
        tree.append((a, b))
    registered = []
    for i, f in enumerate(features):
        _matrix(world[i], f["width"], f["height"])
        registered.append(
            {
                "id": f["id"],
                "path": f["path"],
                "width": f["width"],
                "height": f["height"],
                "sourceToWorld": world[i].ravel().tolist(),
            }
        )
    # Loop consistency is diagnostic, not automatic non-rigid correction.
    loop_errors = []
    for edge in accepted:
        a, b = by_id[edge["source"]], by_id[edge["target"]]
        h, w = features[b]["height"], features[b]["width"]
        p = np.array([[0, 0, 1], [w / 2, h / 2, 1], [w, h, 1]], dtype=float).T
        direct = world[a] @ transforms[(a, b)] @ p
        tree_mapped = world[b] @ p
        loop_errors.extend(
            np.linalg.norm(
                (direct[:2] / direct[2] - tree_mapped[:2] / tree_mapped[2]).T, axis=1
            ).tolist()
        )
    return {
        "sources": registered,
        "edges": [(e["source"], e["target"]) for e in accepted],
        "diagnostics": {
            "sourceCount": n,
            "pairProposals": proposed,
            "acceptedPairs": accepted,
            "maxDecodedRgbBytes": max_decode_bytes,
            "retainedFeatureBytes": descriptors_bytes,
            "proxyMaxDimension": proxy_max_dimension,
            "maxFeatures": max_features,
            "treeEdges": [[features[a]["id"], features[b]["id"]] for a, b in tree],
            "loopResidualP95WorldPx": float(np.percentile(loop_errors, 95))
            if loop_errors
            else 0,
            "loopResidualSampling": "source corners/center, including non-overlap extrapolation; not independent landmark error",
            "trainingAccepted": False,
            "status": "needs_independent_landmark_and_loop_review",
        },
    }
