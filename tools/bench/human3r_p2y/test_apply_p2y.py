#!/usr/bin/env python3
"""CPU-only checks for the Human3R P2-Y correction helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from apply_p2y import WINDOWS, build_offset, clone_with_root_y, tree_equal


class P2YHelpersTest(unittest.TestCase):
    def test_contact_interval_prevents_compressed_descent(self) -> None:
        curve = np.full(1060, 0.01, dtype=np.float64)
        curve[250:401] = np.linspace(0.01, 0.50, 151)
        curve[401:929] = 0.50
        curve[929:941] = np.linspace(0.50, 0.20, 12)
        curve[941:] = 0.20
        contact = np.ones((len(curve), 4), dtype=np.float64)
        contact[878:955] = 0.1

        correction, diagnostics = build_offset(curve, contact, box_height=0.70)

        self.assertEqual(diagnostics["descent_start"], 878)
        self.assertEqual(diagnostics["descent_end"], 955)
        self.assertGreaterEqual(
            diagnostics["descent_end"] - diagnostics["descent_start"] + 1,
            diagnostics["minimum_descent_frames"],
        )
        for name, expected in (("pre", -0.01), ("top", 0.20), ("post", -0.20)):
            start, end = WINDOWS[name]
            self.assertAlmostEqual(float(np.median(correction[start:end])), expected)
        self.assertTrue(np.isfinite(correction).all())

    def test_clone_changes_only_global_root_y(self) -> None:
        source = {
            "smpl_params_global": {
                "transl": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "body_pose": torch.arange(12).reshape(2, 6),
            },
            "smpl_params_incam": {"transl": torch.ones(2, 3)},
            "K_fullimg": torch.eye(3).repeat(2, 1, 1),
            "net_outputs": {"feature": torch.arange(4)},
        }

        target = clone_with_root_y(source, np.array([7.0, 8.0]))

        self.assertEqual(source.keys(), target.keys())
        self.assertTrue(torch.equal(source["smpl_params_global"]["transl"][:, 0], target["smpl_params_global"]["transl"][:, 0]))
        self.assertTrue(torch.equal(source["smpl_params_global"]["transl"][:, 2], target["smpl_params_global"]["transl"][:, 2]))
        self.assertTrue(torch.equal(target["smpl_params_global"]["transl"][:, 1], torch.tensor([7.0, 8.0])))
        self.assertTrue(tree_equal(source["smpl_params_incam"], target["smpl_params_incam"]))
        self.assertTrue(tree_equal(source["K_fullimg"], target["K_fullimg"]))
        self.assertTrue(tree_equal(source["net_outputs"], target["net_outputs"]))


if __name__ == "__main__":
    unittest.main()
