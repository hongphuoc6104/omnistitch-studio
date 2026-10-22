"""Bounded serial-section proposals. Metrics never automatically approve a pair.

SIFT/RANSAC and silhouette hypotheses initialize multimodal affine registration.
Optional SimpleITK Mattes mutual information follows official registration APIs.
RGB derivatives are for inspection; intensity preprocessing never changes sources.
"""

import cv2
import numpy as np

from backend.serial_pairing import matrix3, transform_points, proxy_to_native


def tissue_proxy(rgb):
    """Filled outer-body silhouette for registration; NOT a tissue/loss mask.

    Visible gland lumens can differ greatly between THO and H&E. Counting their
    dark pixels as tissue area would bias the initial scale between modalities.
    Disconnected external fragments remain disconnected; enclosed holes are filled.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or max(rgb.shape[:2]) > 1600:
        raise ValueError("Expected bounded uint8 RGB proxy")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= max(64, mask.size * .003):
            clean[labels == index] = 255
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(clean)
    cv2.drawContours(outline, contours, -1, 255, thickness=cv2.FILLED)
    return outline


def silhouette_metrics(fixed_mask, moving_mask, moving_to_fixed):
    """Overlap diagnostic only: similar outlines can hide wrong internal anatomy."""
    warped = cv2.warpPerspective(moving_mask, matrix3(moving_to_fixed),
                                 (fixed_mask.shape[1], fixed_mask.shape[0]), flags=cv2.INTER_NEAREST)
    a, b = fixed_mask > 0, warped > 0
    intersection = int(np.count_nonzero(a & b))
    return {"silhouette_dice": 2 * intersection / max(1, int(a.sum() + b.sum())),
            "fixed_covered_fraction": intersection / max(1, int(a.sum())),
            "warped_moving_covered_fraction": intersection / max(1, int(b.sum())),
            "meaning": "heuristic outline overlap, not gland/cell correspondence"}


def _gray(rgb):
    return cv2.createCLAHE(clipLimit=2., tileGridSize=(12, 12)).apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def feature_hypothesis(fixed, moving, fixed_mask, moving_mask, *, mirror=False):
    """Mutual ratio SIFT, fit/diagnostic split, similarity RANSAC in proxy pixels."""
    flip = np.array([[-1., 0, moving.shape[1] - 1], [0, 1, 0], [0, 0, 1]]) if mirror else np.eye(3)
    working = moving[:, ::-1].copy() if mirror else moving
    working_mask = moving_mask[:, ::-1].copy() if mirror else moving_mask
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.012, edgeThreshold=12, sigma=1.4)
    fk, fd = detector.detectAndCompute(_gray(fixed), fixed_mask)
    mk, md = detector.detectAndCompute(_gray(working), working_mask)
    if fd is None or md is None or min(len(fd), len(md)) < 16:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = {a.queryIdx: a.trainIdx for pair in matcher.knnMatch(md, fd, k=2)
               if len(pair) == 2 for a, b in [pair] if a.distance < .75 * b.distance}
    reverse = {a.queryIdx: a.trainIdx for pair in matcher.knnMatch(fd, md, k=2)
               if len(pair) == 2 for a, b in [pair] if a.distance < .75 * b.distance}
    pairs = sorted((m, f) for m, f in forward.items() if reverse.get(f) == m)
    if len(pairs) < 16:
        return None
    source = np.asarray([mk[m].pt for m, _ in pairs], dtype=np.float32)
    target = np.asarray([fk[f].pt for _, f in pairs], dtype=np.float32)
    # Diagnostic descriptor holdout is not an independent anatomical annotation.
    held = np.arange(len(source)) % 4 == 0
    cv2.setRNGSeed(2026)
    affine, inliers = cv2.estimateAffinePartial2D(source[~held], target[~held], method=cv2.RANSAC,
                                                ransacReprojThreshold=3., maxIters=5000, confidence=.999)
    if affine is None or inliers is None or int(inliers.sum()) < 10:
        return None
    h = np.eye(3)
    h[:2] = affine
    singular = np.linalg.svd(h[:2, :2], compute_uv=False)
    if singular.min() < .35 or singular.max() > 3:
        return None
    prediction = transform_points(h, source[held])
    residual = np.linalg.norm(prediction - target[held], axis=1)
    kept_source = source[~held][inliers.ravel() > 0]
    spread = float(cv2.contourArea(cv2.convexHull(kept_source))) / max(1, int((moving_mask > 0).sum()))
    original_source = transform_points(flip, source)
    return {"HE_proxy_to_THO_proxy": (h @ flip).tolist(), "initializer": "mutual_SIFT_RANSAC_similarity",
            "reflection_proposed": mirror, "mutual_matches": len(pairs),
            "fit_inliers": int(inliers.sum()), "fit_spread_fraction": spread,
            "descriptor_holdout_median_proxy_px": float(np.median(residual)),
            "descriptor_holdout_p95_proxy_px": float(np.percentile(residual, 95)),
            "matches": [{"moving_xy": a.tolist(), "fixed_xy": b.tolist(),
                         "role": "descriptor_diagnostic" if diagnostic else "fit_candidate"}
                        for a, b, diagnostic in zip(original_source, target, held)]}


def shape_hypothesis(fixed_mask, moving_mask):
    """Grid silhouette initializer; cannot establish an internal correspondence."""
    maximum = max(*fixed_mask.shape, *moving_mask.shape)
    scale = min(1., 256 / maximum)
    fs = (max(1, round(fixed_mask.shape[1] * scale)), max(1, round(fixed_mask.shape[0] * scale)))
    ms = (max(1, round(moving_mask.shape[1] * scale)), max(1, round(moving_mask.shape[0] * scale)))
    fm = cv2.resize(fixed_mask, fs, interpolation=cv2.INTER_NEAREST)
    mm = cv2.resize(moving_mask, ms, interpolation=cv2.INTER_NEAREST)
    fy, fx = np.where(fm > 0)
    my, mx = np.where(mm > 0)
    if min(len(fx), len(mx)) < 100:
        return None
    cf, cm = np.array([fx.mean(), fy.mean()]), np.array([mx.mean(), my.mean()])
    base_scale = np.sqrt(len(fx) / len(mx))
    best = None
    for mirror in (False, True):
        reflection = np.diag([-1 if mirror else 1, 1])
        for degree in range(0, 360, 15):
            angle = np.deg2rad(degree)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for variation in (.9, 1., 1.1):
                linear = rotation @ reflection * base_scale * variation
                h = np.eye(3)
                h[:2, :2], h[:2, 2] = linear, cf - linear @ cm
                score = silhouette_metrics(fm, mm, h)["silhouette_dice"]
                if best is None or score > best[0]:
                    best = (score, h.copy(), mirror, degree)
    # Small mask resize is only initialization, then refined/checked on full proxy.
    pf = proxy_to_native([fixed_mask.shape[1], fixed_mask.shape[0]], list(fs))
    pm = proxy_to_native([moving_mask.shape[1], moving_mask.shape[0]], list(ms))
    return {"HE_proxy_to_THO_proxy": (pf @ best[1] @ np.linalg.inv(pm)).tolist(),
            "initializer": "silhouette_grid_only", "reflection_proposed": best[2],
            "initial_rotation_degrees": best[3], "fit_inliers": 0, "matches": [],
            "warning": "outline-only; do not infer anatomical identity"}


def refine_affine_mi(fixed, moving, initial, fixed_mask, moving_mask, *, iterations=60):
    """Optional SimpleITK MI: optimizer maps fixed→moving, stored result is inverse.

    See SimpleITK registrationOverview and ImageRegistrationMethod documentation.
    This global affine stage does not attempt a deformable/anatomical correction.
    """
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    h = matrix3(initial)
    inverse = np.linalg.inv(h)
    if not np.allclose(inverse[2], [0, 0, 1]):
        raise ValueError("MI affine refinement expects an affine initializer")
    fixed_image = sitk.GetImageFromArray(_gray(fixed).astype(np.float32) / 255)
    moving_image = sitk.GetImageFromArray(_gray(moving).astype(np.float32) / 255)
    transform = sitk.AffineTransform(2)
    center = np.array([(fixed.shape[1] - 1) / 2, (fixed.shape[0] - 1) / 2])
    transform.SetCenter(center.tolist())
    transform.SetMatrix(inverse[:2, :2].ravel().tolist())
    transform.SetTranslation((inverse[:2, 2] - center + inverse[:2, :2] @ center).tolist())
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(.25, seed=2026)
    registration.SetMetricFixedMask(sitk.GetImageFromArray((fixed_mask > 0).astype(np.uint8)))
    registration.SetMetricMovingMask(sitk.GetImageFromArray((moving_mask > 0).astype(np.uint8)))
    registration.SetInterpolator(sitk.sitkLinear)
    # An unrestricted affine line search can leave all moving-image support even
    # from substantial initial overlap. Bound physical displacement each update.
    registration.SetOptimizerAsGradientDescent(
        learningRate=1., numberOfIterations=iterations,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
        estimateLearningRate=registration.EachIteration,
        maximumStepSizeInPhysicalUnits=1.)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=False)
    initial_metric = float(registration.MetricEvaluate(fixed_image, moving_image))
    result = registration.Execute(fixed_image, moving_image)
    origin = np.asarray(result.TransformPoint((0., 0.)))
    inverse_refined = np.eye(3)
    inverse_refined[:2, 2] = origin
    inverse_refined[:2, 0] = np.asarray(result.TransformPoint((1., 0.))) - origin
    inverse_refined[:2, 1] = np.asarray(result.TransformPoint((0., 1.))) - origin
    refined = matrix3(np.linalg.inv(inverse_refined))
    singular = np.linalg.svd(refined[:2, :2], compute_uv=False)
    if singular.min() < .3 or singular.max() > 3 or singular.max() / singular.min() > 1.5:
        raise ValueError("MI proposed excessive scale/anisotropy; keep initial for review")
    if np.sign(np.linalg.det(refined[:2, :2])) != np.sign(np.linalg.det(h[:2, :2])):
        raise ValueError("MI changed reflection topology")
    return refined, {"metric": "Mattes mutual information, optimizer diagnostic only",
                     "initial_full_resolution_metric": initial_metric,
                     "value": float(registration.GetMetricValue()),
                     "stop": registration.GetOptimizerStopConditionDescription(),
                     "threads": 2, "sampling_seed": 2026,
                     "optimizer": "gradient descent, per-iteration learning-rate estimate",
                     "maximum_step_physical_proxy_units": 1.}


def refine_outline_ecc(fixed_mask, moving_mask, initial):
    """Continuously refine a rough affine using clipped signed outline distances.

    OpenCV ECC estimates fixed→moving sampling parameters. This function accepts
    affine matrices only, restores a full 3×3 and stores the inverse direction.
    Filled-outline geometry is initialization, not anatomical correspondence.
    """
    h = matrix3(initial)
    if not np.allclose(h[2], [0, 0, 1.]):
        raise ValueError("Outline ECC is affine-only; do not truncate a homography")
    if min(np.count_nonzero(fixed_mask), np.count_nonzero(moving_mask)) < 100:
        raise ValueError("Insufficient silhouette for outline refinement")
    if max(*fixed_mask.shape, *moving_mask.shape) > 1600:
        raise ValueError("Outline exceeds bounded proxy dimensions")
    before = silhouette_metrics(fixed_mask, moving_mask, h)
    levels = []
    for shrink in (4, 2, 1):
        fs = [max(16, round(fixed_mask.shape[1] / shrink)), max(16, round(fixed_mask.shape[0] / shrink))]
        ms = [max(16, round(moving_mask.shape[1] / shrink)), max(16, round(moving_mask.shape[0] / shrink))]
        pf = proxy_to_native([fixed_mask.shape[1], fixed_mask.shape[0]], fs)
        pm = proxy_to_native([moving_mask.shape[1], moving_mask.shape[0]], ms)
        def distance(mask, size):
            small = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
            signed = cv2.distanceTransform((small == 0).astype(np.uint8), cv2.DIST_L2, 3)
            signed -= cv2.distanceTransform((small > 0).astype(np.uint8), cv2.DIST_L2, 3)
            return np.clip(signed / max(8., 48. / shrink), -1, 1).astype(np.float32)
        small_h = np.linalg.inv(pf) @ h @ pm
        inverse = np.linalg.inv(small_h)
        # Explicit affine parameterization, with projective inputs rejected above.
        score, affine = cv2.findTransformECC(
            distance(fixed_mask, fs), distance(moving_mask, ms),
            inverse[:2].astype(np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 150, 1e-5),
            inputMask=None, gaussFiltSize=5)
        inverse = np.eye(3)
        inverse[:2] = affine
        h = matrix3(pf @ np.linalg.inv(inverse) @ np.linalg.inv(pm))
        levels.append({"shrink": shrink, "outline_ecc": float(score)})
    singular = np.linalg.svd(h[:2, :2], compute_uv=False)
    if singular.min() < .3 or singular.max() > 3 or singular.max() / singular.min() > 1.5:
        raise ValueError("Outline ECC proposed excessive affine distortion")
    if np.sign(np.linalg.det(h[:2, :2])) != np.sign(np.linalg.det(matrix3(initial)[:2, :2])):
        raise ValueError("Outline refinement changed reflection topology")
    after = silhouette_metrics(fixed_mask, moving_mask, h)
    if after["silhouette_dice"] + .01 < before["silhouette_dice"]:
        raise ValueError("Outline ECC degraded silhouette; preserve initial hypothesis")
    return h, {"levels": levels, "before": before, "after": after,
                "interpretation": "continuous global outline initialization only"}


def propose_pair(fixed, moving, *, refine=False):
    """Return all hypotheses and a review proposal, never an accepted pair."""
    fm, mm = tissue_proxy(fixed), tissue_proxy(moving)
    proposals = [feature_hypothesis(fixed, moving, fm, mm, mirror=mirror) for mirror in (False, True)]
    proposals.append(shape_hypothesis(fm, mm))
    candidates = []
    for proposal in (p for p in proposals if p is not None):
        h = matrix3(proposal["HE_proxy_to_THO_proxy"])
        proposal["initial_silhouette"] = silhouette_metrics(fm, mm, h)
        if refine:
            try:
                h, details = refine_outline_ecc(fm, mm, h)
                proposal["outline_refinement"] = details
                proposal["HE_proxy_to_THO_proxy_outline"] = h.tolist()
                proposal["HE_proxy_to_THO_proxy"] = h.tolist()
            except (cv2.error, ValueError) as error:
                proposal["outline_refinement_rejected"] = str(error)
            try:
                refined, details = refine_affine_mi(fixed, moving, h, fm, mm)
                proposal["MI_refinement"] = details
                proposal["HE_proxy_to_THO_proxy_MI"] = refined.tolist()
                proposal["HE_proxy_to_THO_proxy"] = refined.tolist()
            except (ValueError, RuntimeError) as error:
                proposal["MI_refinement_rejected"] = str(error)
        proposal["final_silhouette"] = silhouette_metrics(fm, mm, proposal["HE_proxy_to_THO_proxy"])
        candidates.append(proposal)
    # Ordering is a triage aid, no identity claim. Silhouette-first keeps fragments
    # with accidental repeated-gland matches from winning by inlier count alone.
    candidates.sort(key=lambda p: p["final_silhouette"]["silhouette_dice"], reverse=True)
    return {"hypotheses": candidates, "status": "proposal_pending_native_and_landmark_review" if candidates else "no_proposal",
            "trainingAccepted": False, "pixel_ground_truth": False,
            "fixed_reference": "THO", "moving": "HE",
            "selection_limit": "ranking is heuristic, not independent anatomical validation"}
