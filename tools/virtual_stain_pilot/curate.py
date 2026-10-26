"""Versioned texture curation of an existing native pilot, preserving pixels."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from tools.virtual_stain_pilot.prepare import contained, sha256, structured_fraction


def curate(source, output):
    """Filter featureless tinted glass; leave the original dataset immutable."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Curation needs a new version directory")
    original_manifest = source / "manifest.json"
    manifest = json.loads(original_manifest.read_text())
    kept, rejected, by_patient = [], [], defaultdict(lambda: defaultdict(int))
    for row in manifest["records"]:
        path = contained(source, row["file"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Original crop checksum mismatch")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.width * image.height > 512**2:
                raise ValueError("Unexpected pilot patch")
            score = structured_fraction(np.array(image))
        updated = {**row, "structured_fraction": score}
        if score >= .25:
            kept.append(updated)
            by_patient[(row["split"], row["patient_id"])][row["domain"]] += 1
        else:
            rejected.append({**updated, "reason": "insufficient_native_block_contrast_for_technical_pilot"})
    incomplete = {key for key, counts in by_patient.items()
                  if min(counts.get(domain, 0) for domain in ("THO", "HE")) < 2}
    final = []
    for row in kept:
        if (row["split"], row["patient_id"]) in incomplete:
            rejected.append({**row, "reason": "patient_lacks_two_curated_crops_in_each_domain"})
        else:
            final.append(row)
    if not final:
        raise ValueError("No curated records; inspect data before training")
    output.mkdir(parents=True)
    for row in final:
        path = contained(output, row["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(source, row["file"]), path)
        if sha256(path) != row["sha256"]:
            raise ValueError("Curation changed a crop")
    contacts = defaultdict(list)
    for row in final:
        contacts[(row["split"], row["patient_id"], row["domain"])].append(row)
    for (split, patient, domain), rows in contacts.items():
        panel = Image.new("RGB", (640, 368), "white")
        draw = ImageDraw.Draw(panel)
        for index, row in enumerate(rows[:8]):
            with Image.open(output / row["file"]) as image:
                image.thumbnail((160, 160))
                left, top = index % 4 * 160, index // 4 * 184
                panel.paste(image, (left, top))
                draw.text((left, top + 160), f"S={row['structured_fraction']:.2f}", fill="black")
        target = output / "qc" / f"{split}_{patient}_{domain}.jpg"
        target.parent.mkdir(exist_ok=True)
        panel.save(target, quality=90)
    result = {**manifest, "records": final, "curation_rejected": rejected,
              "curation": {"source_manifest_sha256": sha256(original_manifest),
                           "method": "native 64px block p90-p10 >=12; fraction >=0.25",
                           "limits": "texture-rich technical pilot; may exclude low-contrast tissue",
                           "input_count": len(manifest["records"]), "output_count": len(final)}}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    curate(args.source, args.output)
