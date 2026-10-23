import unittest

import numpy as np

from tools.photometry_pilot.lighting_math import (
    native_box,
    render_fixed_crop,
    sensor_gain_roi,
)


class TrialMathTest(unittest.TestCase):
    def test_identity_fixed_owner_and_native_field(self):
        raw = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        identity = np.eye(3).ravel().tolist()
        manifest = {"outputPixelToWorld": identity}
        s = {
            "id": "a",
            "ownerId": 1,
            "width": 8,
            "height": 8,
            "worldToSource": identity,
            "sourceToWorld": identity,
            "local_path": "fixture",
        }
        owner = np.ones((4, 4), np.uint32)
        reader = lambda path, box: raw[box[1] : box[3], box[0] : box[2]]
        result = render_fixed_crop(
            manifest,
            [s],
            {"a": np.full((4, 4), 1.25, np.float32)},
            {"a": 0.8},
            (2, 2, 6, 6),
            owner,
            reader,
        )
        self.assertTrue(np.array_equal(result[0], raw[2:6, 2:6]))
        self.assertTrue(np.allclose(result[1], raw[2:6, 2:6] * 1.25))
        self.assertTrue(np.allclose(result[2], raw[2:6, 2:6]))
        self.assertTrue(
            np.all(sensor_gain_roi(np.ones((2, 2)), 8, 8, (0, 0, 8, 8)) == 1)
        )

    def test_numpy_coordinate_adapter(self):
        box = native_box(tuple(np.int64(x) for x in [1, 2, 5, 6]))
        self.assertEqual(box, (1, 2, 5, 6))
        self.assertTrue(all(type(x) is int for x in box))
        with self.assertRaises(ValueError):
            native_box((1.2, 2, 5, 6))

    def test_rejects_gigapixel_roi_before_allocation(self):
        with self.assertRaises(MemoryError):
            sensor_gain_roi(np.ones((2, 2)), 100000, 100000, (0, 0, 100000, 100000))

    def test_rejects_invalid_field(self):
        with self.assertRaises(ValueError):
            sensor_gain_roi(np.full((2, 2), np.nan), 8, 8, (0, 0, 4, 4))

    def test_two_owners_no_blending_or_clipping(self):
        I = np.eye(3).ravel().tolist()
        sources = []
        for i in [1, 2]:
            sources.append(
                {
                    "id": str(i),
                    "ownerId": i,
                    "width": 4,
                    "height": 4,
                    "worldToSource": I,
                    "sourceToWorld": I,
                    "local_path": str(i),
                }
            )
        owner = np.array([[1, 1, 2, 2]] * 4, np.uint32)
        out = render_fixed_crop(
            {"outputPixelToWorld": I},
            sources,
            {str(i): np.full((2, 2), 2, np.float32) for i in [1, 2]},
            {},
            (0, 0, 4, 4),
            owner,
            lambda path, box: np.full(
                (box[3] - box[1], box[2] - box[0], 3), int(path) * 100, np.uint8
            ),
        )
        self.assertTrue(np.all(out[1][:, :2] == 200))
        self.assertTrue(np.all(out[1][:, 2:] == 400))
        self.assertEqual(out[1].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
