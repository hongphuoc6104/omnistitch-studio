import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import tifffile
from PIL import Image

from backend.dataset_mosaic import _native_focus_grid, export_dataset_mosaic
from backend.wsi_exporter import _reclaim_memmap_pages, write_tiled_tiff


class TestDatasetMosaic(unittest.TestCase):
    def _source(self, root, name, rgb, matrix=None):
        path = Path(root) / (name + ".png")
        Image.fromarray(rgb).save(path)
        return {
            "id": name,
            "path": str(path),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "sourceToWorld": matrix or [1, 0, 0, 0, 1, 0, 0, 0, 1],
        }

    def _counterexample(self, root):
        rng = np.random.default_rng(2026)
        gray = cv2.GaussianBlur(
            rng.integers(25, 220, (704, 704), dtype=np.uint8), (3, 3), 0.6
        )
        world = np.stack(
            [
                gray,
                (gray * 0.65).astype(np.uint8),
                np.minimum(gray.astype(np.uint16) + 25, 255).astype(np.uint8),
            ],
            axis=-1,
        )
        sharp = world[192:704, 192:704]
        blurry = cv2.GaussianBlur(world[:512, :512], (21, 21), 4)
        return sharp, [
            self._source(root, "corner", sharp, [1, 0, 192, 0, 1, 192, 0, 0, 1]),
            self._source(root, "center", blurry),
        ]

    def test_e2e_sharp_corner_beats_blurry_center_with_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            sharp, sources = self._counterexample(root)
            meta = export_dataset_mosaic(
                sources,
                Path(root) / "out",
                root,
                edges=[("corner", "center")],
                tile_size=64,
                focus_step=64,
                focus_window=128,
                memory_budget_bytes=8 * 1024 * 1024,
            )
            actual = tifffile.imread(Path(root) / "out/mosaic.tif")
            owner = tifffile.imread(Path(root) / "out/owner.tif")
            patch = actual[240:272, 240:272]
            self.assertLessEqual(
                float(np.abs(patch.astype(float) - sharp[48:80, 48:80]).mean()), 2
            )
            self.assertTrue(np.all(owner[240:272, 240:272] == 2))
            self.assertEqual(owner.dtype, np.uint32)
            self.assertEqual(meta["sources"][1]["id"], "corner")
            self.assertFalse(meta["geometryQC"]["trainingAccepted"])
            self.assertGreater(meta["renderStats"]["tileCount"], 1)
            self.assertLessEqual(
                meta["renderStats"]["maxEstimatedWorkingBytes"], 8 * 1024 * 1024
            )
            self.assertLess(meta["renderStats"]["maxSourceRoiPixels"], 512 * 512)

    def test_chunk_size_order_and_overlap_seams_are_invariant(self):
        with tempfile.TemporaryDirectory() as root:
            _, sources = self._counterexample(root)
            outputs = []
            for name, tile, order in [
                ("a", 64, sources),
                ("b", 128, list(reversed(sources))),
            ]:
                export_dataset_mosaic(
                    order,
                    Path(root) / name,
                    root,
                    edges=[("center", "corner")],
                    tile_size=tile,
                    focus_step=64,
                    focus_window=128,
                )
                outputs.append(
                    [
                        tifffile.imread(Path(root) / name / (m + ".tif"))
                        for m in ["mosaic", "owner", "valid", "seam"]
                    ]
                )
            for a, b in zip(*outputs):
                self.assertTrue(np.array_equal(a, b))

    def test_native_focus_is_measured_before_warp(self):
        rgb = np.random.default_rng(9).integers(0, 255, (128, 128, 3), dtype=np.uint8)
        grid = _native_focus_grid(rgb, 64, 64)
        crop = cv2.cvtColor(rgb[:64, :64], cv2.COLOR_RGB2GRAY)
        self.assertAlmostEqual(
            float(grid[0, 0]), float(cv2.Laplacian(crop, cv2.CV_64F).var()), delta=0.05
        )

    def test_preserves_purple_colors_and_separates_missing_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            rgb = np.full((64, 64, 3), [40, 10, 180], dtype=np.uint8)
            a = self._source(root, "a", rgb, [1, 0, -16, 0, 1, 0, 0, 0, 1])
            b = self._source(root, "b", rgb, [1, 0, 80, 0, 1, 0, 0, 0, 1])
            export_dataset_mosaic(
                [a, b],
                Path(root) / "out",
                root,
                edges=[("a", "b")],
                tile_size=32,
                focus_step=32,
                focus_window=32,
            )
            image = tifffile.imread(Path(root) / "out/mosaic.tif")
            valid = tifffile.imread(Path(root) / "out/valid.tif")
            self.assertTrue(np.all(image[:, 10:20] == [40, 10, 180]))
            self.assertTrue(np.all(valid[:, 64:96] == 0))
            self.assertTrue(np.all(image[:, 64:96] == 255))

    def test_homography_and_landmarks_use_full_3x3_contract(self):
        with tempfile.TemporaryDirectory() as root:
            rgb = np.random.default_rng(61).integers(
                0, 255, (64, 64, 3), dtype=np.uint8
            )
            H = [1, 0, 0, 0, 1, 0, 0.001, 0, 1]
            sources = [self._source(root, "a", rgb, H), self._source(root, "b", rgb, H)]
            meta = export_dataset_mosaic(
                sources,
                Path(root) / "out",
                root,
                edges=[("a", "b")],
                tile_size=32,
                landmarks=[
                    {
                        "source_a": "a",
                        "xy_a": [30, 30],
                        "source_b": "b",
                        "xy_b": [30, 30],
                    }
                ],
            )
            self.assertEqual(meta["sources"][0]["sourceToWorld"][6], 0.001)
            self.assertEqual(meta["geometryQC"]["landmarkP95"], 0)
            self.assertEqual(meta["geometryQC"]["status"], "landmarks_within_target")
            actual = tifffile.imread(Path(root) / "out/mosaic.tif")
            matrix = np.array(H, dtype=float).reshape(3, 3)
            reference = cv2.warpPerspective(
                rgb,
                matrix,
                (meta["outputWidth"], meta["outputHeight"]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            mask = cv2.warpPerspective(
                np.full((64, 64), 255, dtype=np.uint8),
                matrix,
                (meta["outputWidth"], meta["outputHeight"]),
                flags=cv2.INTER_NEAREST,
            )
            inner = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
            self.assertLessEqual(
                int(
                    np.max(
                        np.abs(
                            actual[inner].astype(np.int16)
                            - reference[inner].astype(np.int16)
                        )
                    )
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "horizon"):
                sources[0]["sourceToWorld"] = [1, 0, 0, 0, 1, 0, -0.1, 0, 1]
                export_dataset_mosaic(
                    sources, Path(root) / "bad", root, edges=[("a", "b")]
                )

    def test_rejects_disconnected_singular_paths_and_unsupported_bit_depth(self):
        with tempfile.TemporaryDirectory() as root:
            rgb = np.full((32, 32, 3), 100, dtype=np.uint8)
            a = self._source(root, "a", rgb)
            b = self._source(root, "b", rgb)
            with self.assertRaisesRegex(ValueError, "Disconnected"):
                export_dataset_mosaic([a, b], Path(root) / "out", root)
            bad = dict(a, sourceToWorld=[0] * 9)
            with self.assertRaises(ValueError):
                export_dataset_mosaic([bad], Path(root) / "out", root)
            with self.assertRaisesRegex(ValueError, "workspace"):
                export_dataset_mosaic(
                    [dict(a, path="/etc/passwd")], Path(root) / "out", root
                )
            path = Path(root) / "deep.tif"
            tifffile.imwrite(
                path, np.ones((32, 32, 3), dtype=np.uint16), photometric="rgb"
            )
            with self.assertRaisesRegex(ValueError, "uint8"):
                export_dataset_mosaic(
                    [dict(a, path=str(path))], Path(root) / "out", root
                )

    def test_failed_export_keeps_old_artifacts_and_closes_memmaps(self):
        with tempfile.TemporaryDirectory() as root:
            a = self._source(root, "a", np.full((32, 32, 3), 100, dtype=np.uint8))
            out = Path(root) / "out"
            out.mkdir()
            (out / "mosaic.tif").write_bytes(b"old")
            with (
                mock.patch(
                    "backend.dataset_mosaic.write_tiled_tiff",
                    side_effect=OSError("write failure"),
                ),
                self.assertRaisesRegex(OSError, "write failure"),
            ):
                export_dataset_mosaic([a], out, root)
            self.assertEqual((out / "mosaic.tif").read_bytes(), b"old")
            self.assertFalse(any(p.name.startswith(".dataset.") for p in out.iterdir()))

    def test_dimension_disk_decode_and_working_set_guards(self):
        with tempfile.TemporaryDirectory() as root:
            a = self._source(root, "a", np.full((64, 64, 3), 100, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                export_dataset_mosaic(
                    [a], Path(root) / "out", root, max_output_pixels=10
                )
            with (
                mock.patch(
                    "backend.dataset_mosaic.shutil.disk_usage",
                    return_value=mock.Mock(free=1),
                ),
                self.assertRaisesRegex(OSError, "capacity"),
            ):
                export_dataset_mosaic([a], Path(root) / "out", root)
            with self.assertRaisesRegex(MemoryError, "Tile working"):
                self._large_extent(root, a)
            large = self._source(
                root, "large", np.full((1024, 1024, 3), 100, dtype=np.uint8)
            )
            with self.assertRaisesRegex(MemoryError, "decode"):
                export_dataset_mosaic(
                    [large],
                    Path(root) / "decode",
                    root,
                    memory_budget_bytes=8 * 1024 * 1024,
                )

    def _large_extent(self, root, a):
        large = dict(a, sourceToWorld=[32, 0, 0, 0, 32, 0, 0, 0, 1])
        return export_dataset_mosaic(
            [large],
            Path(root) / "large",
            root,
            tile_size=2048,
            memory_budget_bytes=8 * 1024 * 1024,
        )

    def test_tiled_writer_preserves_large_owner_ids(self):
        with tempfile.TemporaryDirectory() as root:
            values = np.array([[0, 300], [65536, 2**31]], dtype=np.uint32)
            path = Path(root) / "owner.tif"
            write_tiled_tiff(values, str(path), tile_size=32, is_mask=True)
            self.assertTrue(np.array_equal(tifffile.imread(path), values))

    def test_writer_reclaims_mapped_rows_without_losing_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            mapped = np.memmap(
                Path(root) / "map.dat", mode="w+", dtype=np.uint32, shape=(64, 64)
            )
            mapped[:] = 123456
            try:
                with mock.patch(
                    "backend.wsi_exporter._reclaim_memmap_pages",
                    wraps=_reclaim_memmap_pages,
                ) as reclaim:
                    write_tiled_tiff(
                        mapped,
                        str(Path(root) / "map.tif"),
                        tile_size=32,
                        is_mask=True,
                        release_pages=True,
                    )
                self.assertGreaterEqual(reclaim.call_count, 2)
                self.assertTrue(
                    np.all(tifffile.imread(Path(root) / "map.tif") == 123456)
                )
            finally:
                mapped._mmap.close()

    def test_focus_cache_budget_is_rejected_before_decoding_any_frame(self):
        with tempfile.TemporaryDirectory() as root:
            base = self._source(root, "base", np.zeros((512, 512, 3), dtype=np.uint8))
            sources = [dict(base, id=str(i)) for i in range(257)]
            edges = [(str(i), str(i + 1)) for i in range(256)]
            with (
                mock.patch(
                    "backend.dataset_mosaic.read_image_region",
                    side_effect=AssertionError("decode should not run"),
                ),
                self.assertRaisesRegex(MemoryError, "focus map cache"),
            ):
                export_dataset_mosaic(
                    sources,
                    Path(root) / "out",
                    root,
                    edges=edges,
                    focus_step=16,
                    focus_window=16,
                    memory_budget_bytes=8 * 1024 * 1024,
                )

    def test_disk_preflight_counts_padded_tiles_for_thin_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            a = self._source(
                root,
                "thin",
                np.zeros((1, 64, 3), dtype=np.uint8),
                [1000, 0, 0, 0, 1, 0, 0, 0, 1],
            )
            with (
                mock.patch(
                    "backend.dataset_mosaic.shutil.disk_usage",
                    return_value=mock.Mock(free=32 * 1024 * 1024),
                ),
                mock.patch(
                    "backend.dataset_mosaic.write_tiled_tiff",
                    side_effect=AssertionError("preflight must reject before writing"),
                ),
                self.assertRaisesRegex(OSError, "capacity"),
            ):
                export_dataset_mosaic([a], Path(root) / "out", root)


if __name__ == "__main__":
    unittest.main()
