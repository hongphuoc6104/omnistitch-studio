import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.dataset_exposure import (
    estimate_p3_gains,
    export_p3_mosaic,
    read_exposure_region,
)
from backend.wsi_exporter import write_tiled_tiff


class DatasetExposureTest(unittest.TestCase):
    def fixture(self, root, unknown=False):
        h, w = 65, 97
        rgb = np.full((h, w, 3), 240, np.uint8)
        rgb[:, :, 0] = np.arange(w, dtype=np.uint8)
        owner = np.ones((h, w), np.uint32)
        owner[:, w // 2 :] = 3 if unknown else 2
        baseline = Path(root) / "baseline.tif"
        labels = Path(root) / "owner.tif"
        write_tiled_tiff(rgb, str(baseline), tile_size=32)
        write_tiled_tiff(owner, str(labels), tile_size=32, is_mask=True)
        I = np.eye(3).ravel().tolist()
        meta = {
            "outputWidth": w,
            "outputHeight": h,
            "outputPixelToWorld": I,
            "worldToOutputPixel": I,
            "sources": [{"id": "a", "ownerId": 1}, {"id": "b", "ownerId": 2}],
        }
        return rgb, owner, baseline, labels, meta

    def test_file_export_read_float_highlights_geometry_and_color(self):
        with tempfile.TemporaryDirectory() as root:
            rgb, owner, path, labels, meta = self.fixture(root)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            owner_before = hashlib.sha256(labels.read_bytes()).hexdigest()
            result = export_p3_mosaic(
                path,
                labels,
                meta,
                {"a": 0.75, "b": 4 / 3},
                Path(root) / "out",
                root,
                tile_size=32,
            )
            image = read_exposure_region(
                Path(root) / "out/mosaic.tif", (0, 0, 97, 65), root
            )
            expected = (
                rgb.astype(np.float32)
                * np.where(owner == 1, 0.75, 4 / 3).astype(np.float32)[:, :, None]
            )
            self.assertTrue(np.allclose(image, expected, atol=1e-4))
            self.assertGreater(image.max(), 255)
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual(result["outputPixelToWorld"], meta["outputPixelToWorld"])
            self.assertEqual(result["renderStats"]["pixels"], 65 * 97)
            self.assertFalse(result["trainingAccepted"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(
                hashlib.sha256(labels.read_bytes()).hexdigest(), owner_before
            )
            edge = read_exposure_region(
                Path(root) / "out/mosaic.tif", (83, 54, 200, 200), root
            )
            self.assertTrue(np.array_equal(edge, image[54:, 83:]))
            with self.assertRaises(MemoryError):
                read_exposure_region(
                    Path(root) / "out/mosaic.tif",
                    (0, 0, 97, 65),
                    root,
                    memory_budget_bytes=1,
                )

    def test_unknown_owner_fails_without_publishing_partial_output(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, path, labels, meta = self.fixture(root, unknown=True)
            with self.assertRaisesRegex(ValueError, "unmapped"):
                export_p3_mosaic(
                    path,
                    labels,
                    meta,
                    {"a": 1.0, "b": 1.0},
                    Path(root) / "out",
                    root,
                    tile_size=32,
                )
            self.assertFalse((Path(root) / "out/mosaic.tif").exists())
            self.assertEqual(list((Path(root) / "out").glob(".p3.*")), [])

    def test_guards_before_allocating_canvas(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, path, labels, meta = self.fixture(root)
            with self.assertRaises(MemoryError):
                export_p3_mosaic(
                    path,
                    labels,
                    meta,
                    {"a": 1.0, "b": 1.0},
                    Path(root) / "out",
                    root,
                    memory_budget_bytes=1,
                )
            with self.assertRaises(ValueError):
                export_p3_mosaic(
                    path, labels, meta, {"a": 1.0, "b": 2.0}, Path(root) / "out", root
                )
            with self.assertRaises(ValueError):
                export_p3_mosaic(
                    path, labels, meta, {"a": 1.0, "b": 1.0}, Path(root), root
                )
            self.assertFalse((Path(root) / "out/mosaic.tif").exists())

    def test_fit_recovers_known_exposure_ratio(self):
        I = np.eye(3).ravel().tolist()
        sources = [
            {
                "id": key,
                "width": 384,
                "height": 216,
                "sourceToWorld": I,
                "worldToSource": I,
            }
            for key in ["a", "b"]
        ]
        gray = np.tile(np.linspace(50, 150, 384), (216, 1))
        raw = np.repeat(gray[:, :, None], 3, -1).astype(np.uint8)
        bright = np.rint(raw.astype(float) * 1.2).astype(np.uint8)
        result = estimate_p3_gains(
            {"sources": sources, "matchGraphEdges": [["a", "b"]]},
            np.stack([raw, bright]),
        )
        self.assertAlmostEqual(
            result["gains"]["a"] / result["gains"]["b"], 1.2, delta=0.015
        )
        self.assertLess(
            result["diagnostics"]["after_abs_log_ratio"],
            result["diagnostics"]["before_abs_log_ratio"] / 10,
        )
        self.assertEqual(result["diagnostics"]["unsupported_sources"], [])

    def test_no_photometric_evidence_keeps_identity(self):
        I = np.eye(3).ravel().tolist()
        sources = [
            {
                "id": key,
                "width": 384,
                "height": 216,
                "sourceToWorld": I,
                "worldToSource": I,
            }
            for key in ["a", "b"]
        ]
        manifest = {"sources": sources, "matchGraphEdges": [["a", "b"]]}
        proxies = np.zeros((2, 216, 384, 3), np.uint8)
        result = estimate_p3_gains(manifest, proxies)
        self.assertEqual(result["gains"], {"a": 1.0, "b": 1.0})
        self.assertIsNone(result["diagnostics"]["before_abs_log_ratio"])
        with self.assertRaises(MemoryError):
            estimate_p3_gains(manifest, proxies, memory_budget_bytes=1)


if __name__ == "__main__":
    unittest.main()
