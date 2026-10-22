"""End-to-end native crop preparation and leakage/path regression checks."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.virtual_stain_pilot.prepare import contained, native_candidates, prepare, validate_orders


class VirtualStainPreparationTests(unittest.TestCase):
    def make_dataset(self, root):
        jobs = []
        for index, (patient, split, domain) in enumerate([
            ("101", "train", "THO"), ("101", "train", "HE"),
            ("202", "validation", "THO"), ("202", "validation", "HE")]):
            rgb = np.random.default_rng(index).integers(40, 240, (640, 640, 3), dtype=np.uint8)
            path = root / f"{index}.tif"
            Image.fromarray(rgb).save(path)
            frame = {"patient_id": patient, "magnification_label": "4X",
                     "group_id": f"4X_{patient}_{domain}",
                     "md5_index": hashlib.md5(path.read_bytes()).hexdigest(),
                     "raw_source_candidates": [path.name]}
            jobs.append({"patient_id": patient, "split": split, "domain": domain,
                         "candidates": [frame]})
        return {"seed": 2026, "patch_size": 128, "max_patches_per_patient_domain": 4,
                "jobs": jobs}

    def test_e2e_lossless_native_crop_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_dataset(root)
            manifest = prepare(config, root, root / "output")
            self.assertEqual(len(manifest["records"]), 16)
            self.assertFalse(manifest["paired"])
            self.assertFalse(manifest["trainingAccepted"])
            saved = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(saved["records"], manifest["records"])
            for row in saved["records"]:
                with Image.open(root / row["raw_relative_path"]) as image:
                    raw = np.array(image)
                with Image.open(root / "output" / row["file"]) as image:
                    crop = np.array(image)
                x, y, width, height = row["box"]
                np.testing.assert_array_equal(crop, raw[y:y + height, x:x + width])
                self.assertEqual(row["coordinate_system"], "original_sensor_pixels")

    def test_leakage_and_missing_domain_rejected_before_io(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_dataset(Path(directory))
            leak = copy.deepcopy(config)
            leak["jobs"][1]["split"] = "validation"
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                validate_orders(leak)
            missing = copy.deepcopy(config)
            missing["jobs"] = missing["jobs"][:-1]
            with self.assertRaisesRegex(ValueError, "both domains"):
                validate_orders(missing)

    def test_checksum_mismatch_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_dataset(root)
            config["jobs"][0]["candidates"][0]["md5_index"] = "0" * 32
            with self.assertRaisesRegex(ValueError, "MD5"):
                prepare(config, root, root / "output")
            self.assertFalse((root / "output/manifest.json").exists())

    def test_cross_identity_source_hash_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_dataset(root)
            config["jobs"][2]["candidates"][0]["md5_index"] = config["jobs"][0]["candidates"][0]["md5_index"]
            with self.assertRaisesRegex(ValueError, "Identical source"):
                validate_orders(config)
            with self.assertRaises(ValueError):
                contained(root, "../secret")
            with self.assertRaises(ValueError):
                contained(root, "/tmp/secret")

    def test_native_focus_detects_texture_and_blank_is_ineligible(self):
        blank = np.full((640, 640, 3), 240, dtype=np.uint8)
        self.assertFalse(any(row["eligible"] for row in native_candidates(blank, 128)))
        checker = (np.indices((640, 640)).sum(axis=0) % 2 * 180 + 40).astype(np.uint8)
        textured = np.repeat(checker[:, :, None], 3, axis=2)
        scores = native_candidates(textured, 128)
        self.assertTrue(all(row["eligible"] for row in scores))
        self.assertGreater(min(row["native_laplacian_variance"] for row in scores), 100)


if __name__ == "__main__":
    unittest.main()
