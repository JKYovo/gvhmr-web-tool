#!/usr/bin/env python3
"""CPU checks for the self-calibrated flat-ground-Y constraint."""

from __future__ import annotations

import unittest

import numpy as np

from apply_flat_ground_y import (
    build_height_observations,
    calibrate_marker_heights,
    foot_confidence,
)
from apply_ground_xyz import smooth_observations


class FlatGroundYTest(unittest.TestCase):
    def test_contact_expansion(self) -> None:
        contact = np.array([[0.1, 0.9, 0.2, 0.8]])
        confidence = foot_confidence(contact)
        np.testing.assert_allclose(confidence, [[0.9, 0.9, 0.9, 0.8, 0.8, 0.8]])

    def test_recovers_slow_vertical_drift(self) -> None:
        length = 180
        fps = 30.0
        levels = np.array([0.01, 0.012, 0.008, 0.011, 0.009, 0.013])
        drift = np.linspace(0.0, 0.12, length)
        feet_y = levels[None] + drift[:, None]
        confidence = np.ones_like(feet_y)
        calibrated, marker_mask, calibration_frames = calibrate_marker_heights(
            feet_y,
            confidence,
            fps=fps,
            threshold=0.8,
            calibration_seconds=1.0,
            minimum_observations=20,
        )
        observations, observation_mask = build_height_observations(
            feet_y, calibrated, marker_mask
        )
        correction, diagnostics = smooth_observations(
            observations,
            observation_mask,
            calibration_frames,
            fps=fps,
            smoothing_seconds=0.5,
            minimum_observations=20,
        )
        corrected = feet_y + correction[:, None, 0]
        baseline_error = np.abs(feet_y - calibrated[None])
        corrected_error = np.abs(corrected - calibrated[None])
        self.assertTrue(np.isfinite(correction).all())
        self.assertEqual(diagnostics["outlier_frames"], 0)
        self.assertLess(float(np.percentile(corrected_error, 95)), 0.02)
        self.assertLess(float(np.percentile(corrected_error, 95)), float(np.percentile(baseline_error, 95)))
        self.assertAlmostEqual(float(correction[-1, 0]), -0.12, delta=0.02)


if __name__ == "__main__":
    unittest.main()
