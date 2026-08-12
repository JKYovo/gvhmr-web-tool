#!/usr/bin/env python3
"""CPU checks for the no-depth contact-aware global root optimizer."""

from __future__ import annotations

import unittest

import numpy as np

from tools.bench.human3r_p2y.apply_contact_global_root import (
    add_temporal_rows,
    apply_segment_confidence,
    build_segments,
    contact_anchor_points,
    contact_metrics,
    contact_point_probabilities,
    optimize_root,
    refine_contacts,
    sole_points,
)


class ContactGlobalRootTest(unittest.TestCase):
    def test_contact_channels_select_toe_and_weak_heel_proxy(self) -> None:
        probability = np.array(
            [
                [0.95, 0.10, 0.10, 0.95],
                [0.95, 0.95, 0.95, 0.95],
            ],
            dtype=np.float64,
        )
        points = contact_point_probabilities(probability)
        self.assertEqual(points.shape, (2, 2, 2))
        self.assertAlmostEqual(points[0, 0, 0], 0.10)
        self.assertLess(points[0, 0, 1], 0.5)  # ankle-only is a weak heel proxy
        self.assertAlmostEqual(points[0, 1, 0], 0.95)
        self.assertGreater(points[1, 0, 1], 0.85)  # full-foot support

    def test_temporal_derivatives_are_fps_normalized(self) -> None:
        def coefficients(fps: float) -> tuple[float, float]:
            rows, columns, values, targets = [], [], [], []
            add_temporal_rows(
                rows,
                columns,
                values,
                targets,
                3,
                fps=fps,
                data_weight=0.0,
                velocity_weight=4.0,
                acceleration_weight=9.0,
            )
            velocity = max(abs(v) for r, v in zip(rows, values) if r == 3)
            acceleration = max(abs(v) for r, v in zip(rows, values) if r == 5)
            return velocity, acceleration

        velocity_30, acceleration_30 = coefficients(30.0)
        velocity_60, acceleration_60 = coefficients(60.0)
        self.assertAlmostEqual(velocity_60 / velocity_30, 2.0)
        self.assertAlmostEqual(acceleration_60 / acceleration_30, 4.0)

    def test_same_physical_motion_is_stable_across_fps(self) -> None:
        curves = {}
        for fps in (24.0, 30.0, 60.0):
            time = np.arange(int(4.0 * fps) + 1) / fps
            length = len(time)
            root = np.zeros((length, 3), dtype=np.float64)
            points = np.zeros((length, 2, 2, 3), dtype=np.float64)
            drift = 0.04 * time + 0.01 * np.sin(2.0 * np.pi * 0.7 * time)
            points[:, 0, :, 0] = drift[:, None]
            points[:, 0, :, 2] = (-0.02 * time)[:, None]
            points[:, 0, :, 1] = (0.03 * time)[:, None]
            points[:, 1, :, 1] = 0.20
            heights = np.stack((0.03 * time, np.full(length, 0.20)), axis=1)
            contact = np.zeros((length, 2), dtype=bool)
            contact[:, 0] = True
            point_weights = np.zeros((length, 2, 2), dtype=np.float64)
            point_weights[:, 0] = 0.8
            segments = build_segments(contact, point_weights)
            point_weights = apply_segment_confidence(
                point_weights, segments, max(1, round(0.10 * fps))
            )
            correction, _ = optimize_root(
                root,
                points,
                heights,
                contact,
                point_weights,
                segments,
                floor_y=0.0,
                fps=fps,
                data_weight=0.5,
                velocity_weight=8.0,
                acceleration_weight=480.0,
                height_contact_weight=30720.0,
                slip_contact_weight=960.0,
            )
            curves[fps] = (time, correction)

        reference_time, reference = curves[30.0]
        for fps in (24.0, 60.0):
            time, correction = curves[fps]
            interpolated = np.stack(
                [
                    np.interp(reference_time, time, correction[:, axis])
                    for axis in range(3)
                ],
                axis=1,
            )
            error = np.linalg.norm(interpolated - reference, axis=1)
            self.assertLess(np.percentile(error, 95), 1.0e-4)

    def test_contact_refinement_uses_hysteresis_height_speed_and_duration(self) -> None:
        length = 40
        confidence = np.zeros((length, 2), dtype=np.float64)
        confidence[3:20, 0] = 0.95
        confidence[10, 0] = 0.60  # short gap remains active through hysteresis
        confidence[24:27, 1] = 0.99  # too short and must be removed
        heights = np.full((length, 2), 0.02)
        heights[3:20, 1] = 0.20
        centroids = np.zeros((length, 2, 3), dtype=np.float64)
        centroids[15, 0, 0] = 0.5  # implausibly fast frame splits/rejects contact
        mask, diagnostics = refine_contacts(
            confidence,
            heights,
            centroids,
            fps=30.0,
            enter_threshold=0.85,
            exit_threshold=0.65,
            relative_height_margin=0.08,
            max_contact_speed=0.5,
            minimum_frames=5,
            maximum_gap=2,
        )
        self.assertFalse(mask[:, 1].any())
        self.assertTrue(mask[3:10, 0].all())
        self.assertFalse(mask[14:17, 0].any())
        self.assertGreater(diagnostics["speed_rejected_samples"], 0)

    def test_global_solve_reduces_height_and_slip(self) -> None:
        length = 180
        fps = 30.0
        feet = np.zeros((length, 6, 3), dtype=np.float64)
        contact = np.zeros((length, 2), dtype=bool)
        contact[:80, 0] = True
        contact[100:, 1] = True
        # Left and right contact segments drift in X/Z and vertically away from
        # the fixed floor.  The airborne gap has no direct contact constraints.
        left_phase = np.linspace(0.0, 1.0, 80)
        right_phase = np.linspace(0.0, 1.0, 80)
        feet[:80, :3, 0] = (0.12 * left_phase)[:, None]
        feet[:80, :3, 2] = (-0.06 * left_phase)[:, None]
        feet[:80, :3, 1] = (0.10 * left_phase)[:, None]
        feet[100:, 3:, 0] = (-0.10 * right_phase)[:, None]
        feet[100:, 3:, 2] = (0.05 * right_phase)[:, None]
        feet[100:, 3:, 1] = (0.08 * right_phase)[:, None]
        # Keep swing feet lifted so the support choice is unambiguous.
        feet[:80, 3:, 1] = 0.20
        feet[80:100, :, 1] = 0.20
        feet[100:, :3, 1] = 0.20

        grouped, heights, _centroids = sole_points(feet)
        anchor_points = contact_anchor_points(grouped)
        point_weights = np.repeat(contact[:, :, None], 2, axis=2).astype(np.float64)
        segments = build_segments(contact, point_weights)
        point_weights = apply_segment_confidence(point_weights, segments, 3)
        root = np.zeros((length, 3), dtype=np.float64)
        correction, _ = optimize_root(
            root,
            anchor_points,
            heights,
            contact,
            point_weights,
            segments,
            floor_y=0.0,
            fps=fps,
            data_weight=0.5,
            velocity_weight=4.0,
            acceleration_weight=80.0,
            height_contact_weight=200.0,
            slip_contact_weight=120.0,
        )
        anchor_points_after = anchor_points + correction[:, None, None, :]
        heights_after = heights + correction[:, None, 1]
        before = contact_metrics(
            root, anchor_points, heights, contact, segments, 0.0, fps
        )
        after = contact_metrics(
            root + correction,
            anchor_points_after,
            heights_after,
            contact,
            segments,
            0.0,
            fps,
        )
        self.assertLess(after["support_height_abs_cm"]["p95"], 1.0)
        self.assertLess(
            after["contact_segment_endpoint_drift_cm"]["p95"],
            0.2 * before["contact_segment_endpoint_drift_cm"]["p95"],
        )
        self.assertTrue(np.isfinite(correction).all())
        self.assertLess(np.max(np.abs(np.diff(correction[:, 1], n=2))), 0.01)


if __name__ == "__main__":
    unittest.main()
