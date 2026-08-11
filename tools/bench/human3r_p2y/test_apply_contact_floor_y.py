#!/usr/bin/env python3
"""CPU checks for the shared contact-floor root-Y experiment."""

from __future__ import annotations

import unittest

import numpy as np

from apply_contact_floor_y import (
    build_floor_observations,
    calibrate_floor_height,
    foot_contact_mask,
    guardrail_decision,
    sole_heights,
    sole_residual_metrics,
)
from apply_ground_xyz import smooth_observations


class ContactFloorYTest(unittest.TestCase):
    def test_large_correction_can_be_diagnostic_only(self) -> None:
        checks = {
            "max_abs_y_pass": False,
            "root_step_pass": True,
            "height_p95_effective": True,
        }
        strict, strict_enforced = guardrail_decision(
            checks, allow_large_correction=False
        )
        relaxed, relaxed_enforced = guardrail_decision(
            checks, allow_large_correction=True
        )
        self.assertEqual(strict, "guardrail_failed")
        self.assertIn("max_abs_y_pass", strict_enforced)
        self.assertEqual(relaxed, "diagnostic_pass")
        self.assertNotIn("max_abs_y_pass", relaxed_enforced)

    def test_contact_reduction_and_sole_minima(self) -> None:
        contact = np.array([[0.1, 0.9, 0.2, 0.3], [0.1, 0.2, 0.95, 0.1]])
        np.testing.assert_array_equal(
            foot_contact_mask(contact, 0.8), [[True, False], [False, True]]
        )
        feet_y = np.array(
            [[0.03, 0.02, 0.04, 0.08, 0.05, 0.06], [0.07, 0.04, 0.06, 0.01, 0.03, 0.02]]
        )
        np.testing.assert_allclose(sole_heights(feet_y), [[0.02, 0.05], [0.04, 0.01]])

    def test_lowest_contacting_sole_is_anchored(self) -> None:
        sole_y = np.array([[0.10, 0.14], [0.12, 0.09], [0.11, 0.13]])
        contact = np.array([[True, True], [True, False], [False, True]])
        observations, valid = build_floor_observations(sole_y, contact, floor_y=0.05)
        np.testing.assert_array_equal(valid, [True, True, True])
        np.testing.assert_allclose(observations[:, 0], [-0.05, -0.07, -0.08])

    def test_recovers_shared_floor_drift(self) -> None:
        length = 180
        fps = 30.0
        drift = np.linspace(0.0, 0.12, length)
        sole_y = np.stack([0.02 + drift, 0.025 + drift], axis=1)
        contact = np.ones_like(sole_y, dtype=bool)
        floor_y, calibration_frames = calibrate_floor_height(
            sole_y,
            contact,
            fps=fps,
            calibration_seconds=1.0,
            minimum_observations=20,
        )
        observations, valid = build_floor_observations(sole_y, contact, floor_y)
        correction, diagnostics = smooth_observations(
            observations,
            valid,
            calibration_frames,
            fps,
            smoothing_seconds=0.5,
            minimum_observations=20,
        )
        baseline = sole_residual_metrics(
            sole_y, contact, floor_y, np.zeros(length, dtype=np.float64)
        )
        enhanced = sole_residual_metrics(sole_y, contact, floor_y, correction[:, 0])
        self.assertEqual(diagnostics["outlier_frames"], 0)
        self.assertLess(enhanced["median_cm"], baseline["median_cm"])
        self.assertLess(enhanced["p95_cm"], 2.0)
        self.assertAlmostEqual(float(correction[-1, 0]), -0.12, delta=0.02)


if __name__ == "__main__":
    unittest.main()
