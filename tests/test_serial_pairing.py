"""File→native THO-anchored inspection; neither coverage nor fit implies GT."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from backend.serial_pairing import (inspect_native_pair, landmark_report,
                                    lift_proxy_transform, make_registration_proxy,
                                    matrix3, proxy_to_native, sha256_file, transform_points,
                                    export_aligned_pair)
from backend.dataset_mosaic import read_dataset_region
from tools.serial_pairing.registration import propose_pair, tissue_proxy, shape_hypothesis
from backend.wsi_exporter import write_tiled_tiff


class TestSerialPairing(unittest.TestCase):
    def write_pair(self, root, *, homography=None):
        fixed = np.random.default_rng(2).integers(20, 230, (180, 240, 3), dtype=np.uint8)
        if homography is None:
            moving = fixed.copy()
        else:
            moving = cv2.warpPerspective(fixed, np.linalg.inv(homography), (240, 180),
                                          flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        valid = np.ones((180, 240), dtype=np.uint8)
        paths = [Path(root) / name for name in ("fixed.tif", "moving.tif", "fixed_valid.tif", "moving_valid.tif")]
        for p, image in zip(paths, (fixed, moving, valid, valid)):
            write_tiled_tiff(image, str(p), tile_size=64, is_mask=image.ndim == 2)
        return paths, fixed, moving

    def test_e2e_translation_preserves_fixed_and_native_pixel_correspondence(self):
        with tempfile.TemporaryDirectory() as root:
            h = np.array([[1., 0, 16], [0, 1, -8], [0, 0, 1]])
            paths, original, _ = self.write_pair(root, homography=h)
            hashes = [sha256_file(p) for p in paths]
            fixed, aligned, mask, metadata = inspect_native_pair(*paths, h, (48, 40, 144, 120), root)
            np.testing.assert_array_equal(fixed, original[40:120, 48:144])
            np.testing.assert_array_equal(aligned, fixed)
            self.assertTrue(mask.all())
            self.assertFalse(metadata["trainingAccepted"])
            self.assertEqual(hashes, [sha256_file(p) for p in paths])

    def test_e2e_projective_third_row_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            h = np.array([[1., .02, 5], [.01, 1., -3], [.0007, .0004, 1.]])
            paths, _, moving = self.write_pair(root, homography=h)
            _, aligned, _, _ = inspect_native_pair(*paths, h, (40, 30, 130, 110), root)
            expected = cv2.warpPerspective(moving, h, (240, 180), flags=cv2.INTER_LINEAR,
                                           borderValue=(255, 255, 255))[30:110, 40:130]
            self.assertLess(float(np.abs(aligned.astype(float) - expected).max()), 1.1)
            wrong = cv2.warpAffine(moving, h[:2], (240, 180))[30:110, 40:130]
            self.assertGreater(float(np.abs(wrong.astype(float) - expected).mean()), 10)

    def test_proxy_pixel_centers_lift_between_different_shapes(self):
        h = np.array([[1.1, -.1, 12], [.1, 1.1, -5], [0, 0, 1.]])
        fixed, moving = ([1001, 777], [2003, 1599])
        fp, mp = ([301, 234], [401, 320])
        native = lift_proxy_transform(h, fixed, fp, moving, mp)
        point = np.array([[41.2, 66.8]])
        a = transform_points(native, transform_points(proxy_to_native(moving, mp), point))
        b = transform_points(proxy_to_native(fixed, fp), transform_points(h, point))
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_native_proxy_is_bounded_and_has_exact_identity_when_not_resized(self):
        with tempfile.TemporaryDirectory() as root:
            paths, original, _ = self.write_pair(root)
            proxy, metadata = make_registration_proxy(paths[0], root, max_dimension=256)
            np.testing.assert_array_equal(proxy, original)
            self.assertEqual(metadata["native_size"], [240, 180])
            small, _ = make_registration_proxy(paths[0], root, max_dimension=120)
            self.assertEqual(small.shape, (90, 120, 3))
            with self.assertRaises(MemoryError):
                make_registration_proxy(paths[0], root, memory_budget_bytes=1024)

    def test_capture_mask_does_not_accept_white_padding_or_assert_anatomy(self):
        with tempfile.TemporaryDirectory() as root:
            paths, _, _ = self.write_pair(root)
            valid = np.ones((180, 240), dtype=np.uint8)
            valid[40:80, 40:80] = 0
            write_tiled_tiff(valid, str(paths[3]), tile_size=64, is_mask=True)
            _, _, overlap, metadata = inspect_native_pair(*paths, np.eye(3), (30, 30, 90, 90), root)
            self.assertEqual(int(overlap.sum()), 60 * 60 - 40 * 40)
            self.assertIn("NOT anatomical", metadata["mask_meaning"])

    def test_landmarks_are_held_out_and_never_implicitly_fit_or_accepted(self):
        self.assertEqual(landmark_report(np.eye(3), [])["status"], "pending_independent_landmarks")
        point = {"role": "validation", "used_for_fit": False, "moving_xy": [5, 8], "fixed_xy": [8, 12]}
        report = landmark_report(np.eye(3), [point])
        self.assertEqual(report["median_native_px"], 5)
        with self.assertRaises(ValueError):
            landmark_report(np.eye(3), [{**point, "used_for_fit": True}])

    def test_degenerate_horizon_roi_and_memory_guards(self):
        with self.assertRaises(ValueError):
            matrix3(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            transform_points([[1, 0, 0], [0, 1, 0], [-.01, 0, 1]], [[200, 20]])
        with tempfile.TemporaryDirectory() as root:
            paths, _, _ = self.write_pair(root)
            with self.assertRaises(ValueError):
                inspect_native_pair(*paths, np.eye(3), (-1, 0, 100, 100), root)
            with self.assertRaises(MemoryError):
                inspect_native_pair(*paths, np.eye(3), (0, 0, 200, 160), root, memory_budget_bytes=1024)

    def test_streamed_pair_publication_immutable_sources_and_repeat_refused(self):
        with tempfile.TemporaryDirectory() as root:
            paths, original, _ = self.write_pair(root)
            references = [{"sha256": sha256_file(path), "logical_path": path.name} for path in paths[:2]]
            out = Path(root) / "pair_v1"
            metadata = export_aligned_pair(*paths, np.eye(3), out, root,
                                           fixed_reference=references[0], moving_reference=references[1], tile_size=64)
            result = read_dataset_region(out / "HE_aligned_to_THO.tif", (0, 0, 240, 180), root)
            np.testing.assert_array_equal(result, original)
            self.assertFalse(metadata["loss_mask_available"])
            self.assertFalse(metadata["trainingAccepted"])
            self.assertEqual(metadata["artifacts"]["HE_aligned_to_THO.tif"], sha256_file(out / "HE_aligned_to_THO.tif"))
            self.assertFalse(any(Path(root).glob(".serial-pair-*")))
            with self.assertRaises(ValueError):
                export_aligned_pair(*paths, np.eye(3), out, root,
                                    fixed_reference=references[0], moving_reference=references[1], tile_size=64)

    def test_proxy_proposal_never_automatically_accepts_anatomical_pair(self):
        rng = np.random.default_rng(17)
        fixed = np.full((256, 320, 3), 240, dtype=np.uint8)
        texture = rng.integers(25, 150, (160, 220, 3), dtype=np.uint8)
        fixed[45:205, 50:270] = cv2.GaussianBlur(texture, (3, 3), .6)
        moving = np.roll(fixed, 10, axis=1)
        result = propose_pair(fixed, moving)
        self.assertTrue(result["hypotheses"])
        self.assertFalse(result["trainingAccepted"])
        self.assertFalse(result["pixel_ground_truth"])
        best = result["hypotheses"][0]
        self.assertGreater(best["final_silhouette"]["silhouette_dice"], .9)

    def test_lumen_contrast_does_not_change_outline_initializer_scale(self):
        fixed = np.full((256, 256, 3), 240, dtype=np.uint8)
        cv2.ellipse(fixed, (128, 128), (95, 75), 0, 0, 360, (60, 60, 60), -1)
        moving = fixed.copy()
        for y in range(78, 181, 26):
            for x in range(62, 195, 26):
                if ((x - 128) / 82)**2 + ((y - 128) / 62)**2 < 1:
                    cv2.circle(moving, (x, y), 11, (240, 240, 240), -1)
        with tempfile.TemporaryDirectory() as root:
            Image.fromarray(fixed).save(Path(root) / "THO.png")
            Image.fromarray(moving).save(Path(root) / "HE.png")
            with Image.open(Path(root) / "THO.png") as image:
                read_fixed = np.array(image)
            with Image.open(Path(root) / "HE.png") as image:
                read_moving = np.array(image)
            proposal = propose_pair(read_fixed, read_moving)["hypotheses"][0]
        scale = abs(np.linalg.det(np.asarray(proposal["HE_proxy_to_THO_proxy"])[:2, :2]))**.5
        self.assertAlmostEqual(scale, 1., delta=.03)


if __name__ == "__main__":
    unittest.main()
