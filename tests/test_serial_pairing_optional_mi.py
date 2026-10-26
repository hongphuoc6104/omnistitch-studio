"""Optional real-file regression: private fixed-grid proxies stay outside Git.

Set CTUP_MI_REGRESSION_DIR to a directory with fixed.png, moving.png and
initial.json (HE_proxy_to_THO_proxy) and install SimpleITK==2.4.0. This exercises
the same file→mask→MI→native-coordinate transform path as the real incident.
"""

import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from tools.serial_pairing.registration import refine_affine_mi, tissue_proxy


@unittest.skipUnless(importlib.util.find_spec("SimpleITK") and os.environ.get("CTUP_MI_REGRESSION_DIR"),
                     "Optional private-proxy SimpleITK E2E fixture")
class TestSerialPairingRealMI(unittest.TestCase):
    def test_file_to_mask_to_bounded_mi_preserves_overlap(self):
        root = Path(os.environ["CTUP_MI_REGRESSION_DIR"])
        with Image.open(root / "fixed.png") as image:
            fixed = np.array(image)
        with Image.open(root / "moving.png") as image:
            moving = np.array(image)
        h = json.loads((root / "initial.json").read_text())["HE_proxy_to_THO_proxy"]
        result, report = refine_affine_mi(fixed, moving, h, tissue_proxy(fixed), tissue_proxy(moving))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(abs(float(np.linalg.det(result))), 1e-8)
        self.assertEqual(report["maximum_step_physical_proxy_units"], 1)
        self.assertTrue(np.isfinite(report["value"]))


if __name__ == "__main__":
    unittest.main()
