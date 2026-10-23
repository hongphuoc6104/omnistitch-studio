"""File-to-native residual review, coordinate direction and bounded memory."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.serial_deformation import (backward_native_points,
                                        inspect_deformed_native_pair,
                                        validate_residual_field)
from backend.serial_pairing import inspect_native_pair, sha256_file
from backend.wsi_exporter import write_tiled_tiff


class TestSerialDeformation(unittest.TestCase):
    def sources(self, root):
        rgb = np.random.default_rng(9).integers(0, 255, (160, 224, 3), dtype=np.uint8)
        mask = np.ones((160, 224), np.uint8)
        paths = [Path(root) / name for name in
                 ("THO.tif", "HE.tif", "THO_valid.tif", "HE_valid.tif")]
        for path, array in zip(paths, (rgb, rgb, mask, mask)):
            write_tiled_tiff(array, str(path), tile_size=64, is_mask=array.ndim == 2)
        return paths, rgb

    def test_file_to_native_identity_keeps_sources_and_capture_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            paths, rgb = self.sources(root)
            hashes = [sha256_file(p) for p in paths]
            fixed, moved, mask, meta = inspect_deformed_native_pair(
                *paths, np.eye(3), np.zeros((40, 56, 2), np.float32),
                (40, 32, 136, 112), root)
            np.testing.assert_array_equal(fixed, rgb[32:112, 40:136])
            np.testing.assert_array_equal(moved, fixed)
            self.assertTrue(mask.all())
            self.assertFalse(meta["trainingAccepted"])
            self.assertIn("not anatomical", meta["mask_meaning"])
            self.assertEqual(hashes, [sha256_file(p) for p in paths])

    def test_constant_backward_field_is_composed_before_one_native_sampling(self):
        with tempfile.TemporaryDirectory() as root:
            paths, _ = self.sources(root)
            field = np.empty((40, 56, 2), np.float32)
            field[:] = [2., -1.]
            global_h = np.array([[1., .03, 7.], [0, 1., -3.], [0, 0, 1.]])
            # Native scale is four: fixed -> globally aligned HE is (+8,-4).
            local_forward = np.array([[1., 0, -8.], [0, 1., 4.], [0, 0, 1.]]) @ global_h
            box = (40, 32, 136, 112)
            _, expected, _, _ = inspect_native_pair(*paths, local_forward, box, root)
            _, actual, _, _ = inspect_deformed_native_pair(*paths, global_h, field, box, root)
            self.assertLessEqual(float(np.abs(actual.astype(float) - expected).max()), 1.)

    def test_half_pixel_backward_direction_for_unequal_native_and_proxy_sizes(self):
        field = np.empty((17, 23, 2), np.float32)
        field[:] = [1.25, -2.5]
        points = np.array([[0., 0.], [100., 70.], [232., 170.]])
        h = np.array([[1., 0, 9.], [0, 1., -4.], [0, 0, 1.]])
        expected = points + [1.25 * 233 / 23, -2.5 * 171 / 17] - [9., -4.]
        result = backward_native_points(field, h, [233, 171], points)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        empty = np.empty((0, 2), dtype=float)
        self.assertEqual(backward_native_points(field, h, [233, 171], empty).shape, (0, 2))

    def test_interior_nonlinear_support_is_not_bounded_by_corners(self):
        with tempfile.TemporaryDirectory() as root:
            paths, rgb = self.sources(root)
            field = np.zeros((160, 224, 2), np.float32)
            # A triangular y shear has det(I+grad u)=1, with an interior extremum.
            shift = np.maximum(0, 24 - np.abs(np.arange(224) - 112) / 2).astype(np.float32)
            field[:, :, 1] = -shift[None, :]
            box = (48, 40, 176, 96)
            _, result, _, meta = inspect_deformed_native_pair(*paths, np.eye(3), field, box, root)
            self.assertLess(meta["moving_support_xyxy"][1], 20)
            np.testing.assert_array_equal(result[:, 64], rgb[16:72, 112])
            self.assertEqual(meta["residual_geometry"]["sampled_fold_pixels"], 0)

    def test_missing_capture_does_not_become_anatomical_correspondence(self):
        with tempfile.TemporaryDirectory() as root:
            paths, _ = self.sources(root)
            mask = np.ones((160, 224), np.uint8)
            mask[40:80, 60:100] = 0
            write_tiled_tiff(mask, str(paths[3]), tile_size=64, is_mask=True)
            _, _, coverage, meta = inspect_deformed_native_pair(
                *paths, np.eye(3), np.zeros((40, 56, 2), np.float32),
                (40, 32, 136, 112), root)
            self.assertEqual(int(coverage.sum()), 96 * 80 - 40 * 40)
            self.assertFalse(meta["pixel_ground_truth"])

    def test_geometry_rejects_folds_nonfinite_excessive_and_invalid_fields(self):
        field = np.zeros((20, 24, 2), np.float32)
        field[:, :, 0] = -2 * np.arange(24)[None, :]
        with self.assertRaisesRegex(ValueError, "Folding"):
            validate_residual_field(field)
        field[:] = 0
        field[3, 4] = np.nan
        with self.assertRaises(ValueError):
            validate_residual_field(field)
        field[:] = [65., 0]
        with self.assertRaisesRegex(ValueError, "displacement"):
            validate_residual_field(field)
        with self.assertRaises(ValueError):
            backward_native_points([], np.eye(3), [224, 160], [[20, 30]])
        with tempfile.TemporaryDirectory() as root:
            paths, _ = self.sources(root)
            with self.assertRaises(ValueError):
                inspect_deformed_native_pair(*paths, np.eye(3), [], (20, 30, 80, 90), root)

    def test_budget_and_projective_global_guards_precede_native_allocation(self):
        field = np.zeros((40, 56, 2), np.float32)
        with self.assertRaises(MemoryError):
            backward_native_points(field, np.eye(3), [224, 160], [[2, 3]], memory_budget_bytes=128)
        with tempfile.TemporaryDirectory() as root:
            paths, _ = self.sources(root)
            with self.assertRaises(MemoryError):
                inspect_deformed_native_pair(*paths, np.eye(3), field, (0, 0, 224, 160), root,
                                              memory_budget_bytes=1024)
            h = np.array([[1., 0, 0], [0, 1., 0], [.001, 0, 1.]])
            with self.assertRaisesRegex(ValueError, "affine"):
                inspect_deformed_native_pair(*paths, h, field, (10, 10, 90, 90), root)
            with self.assertRaises(ValueError):
                inspect_deformed_native_pair(*paths, np.eye(3), field, (-1, 0, 50, 50), root)


if __name__ == "__main__":
    unittest.main()
