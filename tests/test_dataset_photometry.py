import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from backend.dataset_mosaic import export_dataset_mosaic
from backend.dataset_photometry import estimate_overlap_scalar_gains


class TestDatasetPhotometry(unittest.TestCase):
    def _fixture(self, root):
        y, x = np.indices((128, 128))
        base = (70 + ((x // 8 + y // 8) % 2) * 50).astype(np.uint8)
        rgb = np.stack(
            [base, (base * 0.8).astype(np.uint8), (base * 1.2).astype(np.uint8)],
            axis=-1,
        )
        sources = []
        for name, image in [
            ("a", rgb),
            ("b", np.rint(rgb.astype(float) * 1.1).astype(np.uint8)),
        ]:
            p = Path(root) / (name + ".png")
            Image.fromarray(image).save(p)
            sources.append(
                {
                    "id": name,
                    "path": str(p),
                    "width": 128,
                    "height": 128,
                    "sourceToWorld": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                }
            )
        return sources

    def test_overlap_exposure_reduces_held_error_without_brightening(self):
        with tempfile.TemporaryDirectory() as root:
            sources = self._fixture(root)
            proposal = estimate_overlap_scalar_gains(sources, root, edges=[("a", "b")])
            self.assertEqual(max(proposal["sourceGains"].values()), 1.0)
            self.assertGreaterEqual(min(proposal["sourceGains"].values()), 0.85)
            p = proposal["pairs"][0]
            self.assertLess(
                p["afterHeldMedianAbsLogRatio"], p["beforeHeldMedianAbsLogRatio"] * 0.3
            )

    def test_photometry_keeps_owner_map_fixed_and_scales_rgb_together(self):
        with tempfile.TemporaryDirectory() as root:
            sources = self._fixture(root)
            for name, gains in [("raw", None), ("dim", {"a": 0.9, "b": 0.9})]:
                export_dataset_mosaic(
                    sources,
                    Path(root) / name,
                    root,
                    edges=[("a", "b")],
                    source_scalar_gains=gains,
                    tile_size=64,
                )
            a = tifffile.imread(Path(root) / "raw/mosaic.tif")
            b = tifffile.imread(Path(root) / "dim/mosaic.tif")
            self.assertTrue(
                np.array_equal(b, np.rint(a.astype(np.float32) * 0.9).astype(np.uint8))
            )
            self.assertTrue(
                np.array_equal(
                    tifffile.imread(Path(root) / "raw/owner.tif"),
                    tifffile.imread(Path(root) / "dim/owner.tif"),
                )
            )

    def test_unsupported_exposure_evidence_and_bad_gains_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            sources = self._fixture(root)
            for s in sources:
                Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8)).save(
                    s["path"]
                )
            p = estimate_overlap_scalar_gains(sources, root, edges=[("a", "b")])
            self.assertEqual(p["unsupportedSourceIds"], ["a", "b"])
            self.assertEqual(p["sourceGains"], {"a": 1.0, "b": 1.0})
            for s in sources:
                Image.fromarray(
                    np.full((128, 128, 3), [30, 10, 255], dtype=np.uint8)
                ).save(s["path"])
            saturated = estimate_overlap_scalar_gains(sources, root, edges=[("a", "b")])
            self.assertEqual(saturated["unsupportedSourceIds"], ["a", "b"])
            with self.assertRaisesRegex(ValueError, "dim-only"):
                export_dataset_mosaic(
                    sources,
                    Path(root) / "out",
                    root,
                    edges=[("a", "b")],
                    source_scalar_gains={"a": 1.1},
                )


if __name__ == "__main__":
    unittest.main()
