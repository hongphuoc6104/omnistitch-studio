import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.dataset_mosaic import export_dataset_mosaic, read_dataset_region
from backend.dataset_registration import register_camera_frames


class TestDatasetRegistration(unittest.TestCase):
    def test_file_to_registration_to_tiled_export_known_translation(self):
        with tempfile.TemporaryDirectory() as root:
            rng = np.random.default_rng(3)
            world = cv2.GaussianBlur(
                rng.integers(0, 255, (256, 384, 3), dtype=np.uint8), (3, 3), 0.5
            )
            sources = []
            for name, rgb in [("a", world[:, :256]), ("b", world[:, 128:])]:
                p = Path(root) / (name + ".png")
                Image.fromarray(rgb).save(p)
                sources.append({"id": name, "path": str(p)})
            result = register_camera_frames(
                sources,
                root,
                proxy_max_dimension=256,
                max_features=1500,
                min_inliers=12,
            )
            H = np.array(result["sources"][1]["sourceToWorld"]).reshape(3, 3)
            self.assertAlmostEqual(H[0, 2], 128, delta=0.5)
            self.assertAlmostEqual(H[1, 2], 0, delta=0.5)
            self.assertLess(
                result["diagnostics"]["retainedFeatureBytes"], 8 * 1024 * 1024
            )
            meta = export_dataset_mosaic(
                result["sources"],
                Path(root) / "out",
                root,
                edges=result["edges"],
                tile_size=64,
            )
            self.assertGreaterEqual(meta["outputWidth"], 383)
            self.assertFalse(meta["geometryQC"]["trainingAccepted"])

    def test_blank_disconnected_frames_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as root:
            sources = []
            for name in ["a", "b"]:
                p = Path(root) / (name + ".png")
                Image.fromarray(np.full((128, 128, 3), 100, dtype=np.uint8)).save(p)
                sources.append({"id": name, "path": str(p)})
            with self.assertRaisesRegex(ValueError, "Disconnected"):
                register_camera_frames(sources, root)

    def test_disconnected_components_export_separately(self):
        with tempfile.TemporaryDirectory() as root:
            sources = []
            for name, value in [("a", 60), ("b", 170)]:
                path = Path(root) / (name + ".png")
                Image.fromarray(np.full((128, 128, 3), value, dtype=np.uint8)).save(
                    path
                )
                sources.append({"id": name, "path": str(path)})
            result = register_camera_frames(sources, root, allow_disconnected=True)
            self.assertEqual(result["diagnostics"]["componentCount"], 2)
            self.assertEqual(len(result["components"]), 2)
            for component, value in zip(result["components"], [60, 170]):
                self.assertEqual(len(component["sources"]), 1)
                self.assertEqual(component["edges"], [])
                meta = export_dataset_mosaic(
                    component["sources"],
                    Path(root) / component["id"],
                    root,
                    tile_size=64,
                )
                self.assertEqual(meta["outputWidth"], 128)
                crop = read_dataset_region(
                    Path(root) / component["id"] / "mosaic.tif", (0, 0, 128, 128), root
                )
                self.assertTrue(np.all(crop == value))
                self.assertFalse(meta["geometryQC"]["trainingAccepted"])

    def test_rejects_duplicate_ids_and_decode_budget(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "a.png"
            Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)).save(p)
            source = {"id": "a", "path": str(p)}
            with self.assertRaisesRegex(ValueError, "unique"):
                register_camera_frames([source, source], root)
            with self.assertRaisesRegex(MemoryError, "decode"):
                register_camera_frames(
                    [source], root, memory_budget_bytes=8 * 1024 * 1024
                )
            with self.assertRaisesRegex(ValueError, "pair budget"):
                register_camera_frames(
                    [source, dict(source, id="b")], root, max_pairs=0
                )


if __name__ == "__main__":
    unittest.main()
