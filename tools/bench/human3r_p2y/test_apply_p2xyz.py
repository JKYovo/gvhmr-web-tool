#!/usr/bin/env python3
"""CPU-only checks for the Human3R P2-XYZ trajectory helpers."""

from __future__ import annotations

import unittest

import numpy as np

from apply_p2xyz import build_xz_correction, fit_rigid_2d


class P2XYZHelpersTest(unittest.TestCase):
    def test_rigid_alignment_recovers_rotation_and_translation(self) -> None:
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
        expected_rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        expected_translation = np.array([2.0, -3.0])
        target = source @ expected_rotation + expected_translation

        rotation, translation = fit_rigid_2d(source, target)

        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-12)
        np.testing.assert_allclose(source @ rotation + translation, target, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_xz_correction_is_continuous_and_hits_anchors(self) -> None:
        anchors = {
            "pre": np.array([0.0, 0.0]),
            "top": np.array([0.08, 0.03]),
            "post": np.array([0.17, 0.06]),
        }
        transition = {
            "ascent_start": 20,
            "ascent_end": 60,
            "descent_start": 120,
            "descent_end": 180,
        }

        correction = build_xz_correction(220, anchors, transition)

        np.testing.assert_allclose(correction[:20] - anchors["pre"], 0.0)
        np.testing.assert_allclose(correction[60:120] - anchors["top"], 0.0)
        np.testing.assert_allclose(correction[180:] - anchors["post"], 0.0)
        self.assertTrue(np.isfinite(correction).all())
        self.assertLess(float(np.linalg.norm(np.diff(correction, axis=0), axis=1).max()), 0.01)


if __name__ == "__main__":
    unittest.main()
