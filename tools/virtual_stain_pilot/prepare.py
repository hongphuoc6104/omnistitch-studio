"""Prepare a bounded, unpaired technical pilot from native RGB sensor crops.

This deliberately selects one FOV per patient/domain to reduce overlapping-FOV
redundancy. Foreground/focus selection is heuristic, never a tissue annotation.
No resizing, illumination correction, warping, or sharpening touches saved crops.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    """Hash without loading the complete file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, relative):
    """Reject absolute paths, traversal, and symlinks escaping the source root."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected contained relative source path")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Source escapes root")
    return path


def validate_orders(config):
    """Fail before I/O for split leakage or unsupported pilot configuration."""
    owners = {}
    hashes = {}
    if config["patch_size"] not in (128, 256, 512):
        raise ValueError("Unsupported native patch size")
    if not 1 <= config["max_patches_per_patient_domain"] <= 128:
        raise ValueError("Patch count exceeds bounded pilot")
    seen = set()
    for job in config["jobs"]:
        patient, split, domain = job["patient_id"], job["split"], job["domain"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", patient):
            raise ValueError("Unsafe patient identifier")
        if split not in ("train", "validation") or domain not in ("THO", "HE"):
            raise ValueError("Pilot accepts train/validation THO/HE only")
        if patient in owners and owners[patient] != split:
            raise ValueError("Patient leakage across splits")
        owners[patient] = split
        if (patient, domain) in seen:
            raise ValueError("Duplicate patient/domain job")
        seen.add((patient, domain))
        if not 1 <= len(job["candidates"]) <= 3:
            raise ValueError("Expected one to three candidate FOVs")
        for frame in job["candidates"]:
            if frame["patient_id"] != patient or frame["magnification_label"] != "4X":
                raise ValueError("Source identity/magnification mismatch")
            key = frame["md5_index"]
            if key in hashes and hashes[key] != (patient, split, domain):
                raise ValueError("Identical source reused across identities/domains")
            hashes[key] = (patient, split, domain)
    for patient in owners:
        if not {(patient, "THO"), (patient, "HE")}.issubset(seen):
            raise ValueError("Pilot requires both domains per selected patient")


def native_candidates(rgb, size):
    """Score original, nonoverlapping crops; Laplacian never sees a resize."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected native uint8 RGB")
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    white = float(np.percentile(gray, 95))
    candidates = []
    for y in range(int(height * .1), int(height * .9) - size + 1, size):
        for x in range(int(width * .1), int(width * .9) - size + 1, size):
            crop = rgb[y:y + size, x:x + size]
            local = gray[y:y + size, x:x + size]
            maximum = crop.max(axis=2).astype(np.float32)
            minimum = crop.min(axis=2).astype(np.float32)
            saturation = (maximum - minimum) / np.maximum(maximum, 1)
            foreground = float(np.mean((local < white - 12) | (saturation > .07)))
            focus = float(cv2.Laplacian(local, cv2.CV_32F).var())
            candidates.append({"box": [x, y, size, size],
                               "foreground_heuristic": foreground,
                               "native_laplacian_variance": focus,
                               "eligible": foreground >= .20 and focus > 1.0})
    return candidates


def prepare(config, source_root, output):
    """Write lossless crops, provenance, QC contacts, and a completion manifest."""
    validate_orders(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Use a new output directory for a new dataset version")
    records, reports, patch_hashes = [], [], {}
    for job in config["jobs"]:
        best = None
        report = {key: job[key] for key in ("patient_id", "split", "domain")}
        report["candidates"] = []
        for frame in job["candidates"]:
            paths = [contained(source_root, name) for name in frame["raw_source_candidates"]]
            source = next((path for path in paths if path.is_file()), None)
            if source is None:
                raise FileNotFoundError("Indexed source candidates missing")
            md5 = hashlib.md5()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    md5.update(block)
            if md5.hexdigest() != frame["md5_index"]:
                raise ValueError("Source MD5 disagrees with inventory")
            with Image.open(source) as image:
                # Room for two candidate RGB FOVs, PIL decode and QC temporaries.
                if image.mode != "RGB" or image.width * image.height * 24 > 256 * 1024**2:
                    raise ValueError("Unsupported mode or native FOV memory budget exceeded")
                rgb = np.array(image)
            scores = native_candidates(rgb, config["patch_size"])
            eligible = [score for score in scores if score["eligible"]]
            report["candidates"].append({"group_id": frame["group_id"],
                                         "raw_relative_path": str(source.relative_to(Path(source_root).resolve())),
                                         "scores": scores})
            rank = (len(eligible), float(np.median([p["native_laplacian_variance"] for p in eligible])) if eligible else -1)
            if eligible and (best is None or rank > best[0]):
                best = (rank, rgb, source, frame, eligible)
        if best is None:
            report["status"] = "no_eligible_native_crops"
            reports.append(report)
            continue
        _, rgb, source, frame, eligible = best
        # Keep top 80% by within-frame focus, never use a global cross-stain cutoff.
        eligible.sort(key=lambda p: p["native_laplacian_variance"], reverse=True)
        pool = eligible[:max(1, int(np.ceil(len(eligible) * .8)))]
        random.Random(config["seed"]).shuffle(pool)
        chosen = pool[:config["max_patches_per_patient_domain"]]
        source_hash = sha256(source)
        panel = Image.new("RGB", (4 * 160, 2 * 184), "white")
        draw = ImageDraw.Draw(panel)
        for i, score in enumerate(chosen):
            x, y, size, _ = score["box"]
            crop = rgb[y:y + size, x:x + size]
            pixel_hash = hashlib.sha256(crop.tobytes()).hexdigest()
            identity = (job["patient_id"], job["split"], job["domain"])
            if pixel_hash in patch_hashes:
                if patch_hashes[pixel_hash] != identity:
                    raise ValueError("Identical native patch across identities/domains")
                continue
            patch_hashes[pixel_hash] = identity
            patch_id = f"{job['split']}_{job['patient_id']}_{job['domain']}_{i:03d}"
            relative = f"patches/{patch_id}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(crop)
            image.save(target)
            records.append({**{key: job[key] for key in ("patient_id", "split", "domain")},
                            **score, "file": relative, "sha256": sha256(target),
                            "pixel_sha256": pixel_hash, "source_sha256": source_hash,
                            "source_md5": frame["md5_index"], "group_id": frame["group_id"],
                            "raw_relative_path": str(source.relative_to(Path(source_root).resolve())),
                            "coordinate_system": "original_sensor_pixels", "microns_per_pixel": None})
            if i < 8:
                image.thumbnail((160, 160))
                left, top = (i % 4) * 160, (i // 4) * 184
                panel.paste(image, (left, top))
                draw.text((left, top + 160), f"{x},{y} F={score['native_laplacian_variance']:.0f}", fill="black")
        contact = output / "qc" / f"{job['split']}_{job['patient_id']}_{job['domain']}.jpg"
        contact.parent.mkdir(parents=True, exist_ok=True)
        panel.save(contact, quality=90)
        report.update(status="prepared_pending_visual_QC", selected_source_sha256=source_hash,
                      selected_count=len(chosen), contact=str(contact.relative_to(output)))
        reports.append(report)
        print("PREPARED", job["split"], job["patient_id"], job["domain"], len(chosen), flush=True)
        del best, rgb
    manifest = {"config": config, "records": records, "qc": reports,
                "paired": False, "trainingAccepted": False,
                "claim": "technical unpaired style-transfer pilot; foreground/focus heuristic"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(json.loads(Path(args.orders).read_text()), args.source_root, args.output)
