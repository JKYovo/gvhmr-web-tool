import unittest

import numpy as np
import torch

from common import contact_y_correction, hybrid_root_y, interpolate_track, window_metrics


class CommonTest(unittest.TestCase):
    def test_contact_y_correction_cancels_stationary_foot_drift(self):
        root_y = torch.tensor([0.0, 0.1, 0.2, 0.3])
        joints = torch.zeros(4, 4, 3)
        joints[:, :, 1] = root_y[:, None]
        contact = torch.ones(4, 4)
        corrected, displacement, mask = contact_y_correction(root_y, joints, contact)
        torch.testing.assert_close(corrected, torch.zeros_like(root_y))
        torch.testing.assert_close(displacement, torch.full((3,), 0.1))
        self.assertTrue(mask.all())

    def test_no_contact_preserves_root(self):
        root_y = torch.tensor([0.0, 0.1, 0.2])
        joints = torch.randn(3, 4, 3)
        corrected, _, mask = contact_y_correction(root_y, joints, torch.zeros(3, 4))
        torch.testing.assert_close(corrected, root_y)
        self.assertFalse(mask.any())

    def test_interpolate_track(self):
        values = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        frames = np.array([0, 2, 4])
        actual = interpolate_track(values, frames, 5, min_coverage=0.5)
        np.testing.assert_allclose(actual, np.arange(5, dtype=np.float32))

    def test_hybrid_root_y_anchors_pre_window(self):
        gv = np.arange(6, dtype=np.float32)
        w0 = np.zeros(6, dtype=np.float32)
        w2 = np.arange(6, dtype=np.float32) * 2
        c_delta, c_root = hybrid_root_y(gv, w0, w2, (0, 2))
        self.assertAlmostEqual(float(np.median(c_delta[:2]) - np.median(gv[:2])), 0.0)
        self.assertAlmostEqual(float(np.median(c_root[:2]) - np.median(gv[:2])), 0.0)

    def test_window_metrics(self):
        curve = np.array([0.0, 0.0, 0.5, 0.5, 0.02, 0.02])
        metrics = window_metrics(curve, {"pre": (0, 2), "top": (2, 4), "post": (4, 6)})
        self.assertAlmostEqual(metrics.floor_return_cm, 2.0)
        self.assertAlmostEqual(metrics.top_relative_cm, 50.0)


if __name__ == "__main__":
    unittest.main()
