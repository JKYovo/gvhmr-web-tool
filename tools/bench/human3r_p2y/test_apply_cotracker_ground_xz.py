#!/usr/bin/env python3
"""CPU-only checks for CoTracker3 Ground-XZ fusion."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from apply_cotracker_ground_xz import (
    RelativeConstraint,
    TrackWindow,
    build_track_windows,
    clone_with_root_xz,
    foot_crop_bounds,
    solve_relative_constraints,
    tracked_relative_constraint,
)


class CoTrackerGroundXZHelpersTest(unittest.TestCase):
    def test_foot_crop_contains_points_and_stays_in_image(self) -> None:
        points = np.array([[3.0, 270.0], [20.0, 280.0], [12.0, 275.0]])
        box = foot_crop_bounds(points, (512, 288), (128, 96), 16)
        x0, y0, x1, y1 = box
        self.assertEqual((x1 - x0, y1 - y0), (128, 96))
        self.assertGreaterEqual(float(points[:, 0].min()), x0)
        self.assertLess(float(points[:, 0].max()), x1)
        self.assertGreaterEqual(float(points[:, 1].min()), y0)
        self.assertLess(float(points[:, 1].max()), y1)

    def test_contact_runs_merge_short_gaps_and_split_long_windows(self) -> None:
        contact = np.zeros((100, 4), dtype=np.float64)
        contact[10:35, 0] = 0.95
        contact[20:22, 0] = 0.0
        contact[50:57, 2] = 0.99
        windows, masks = build_track_windows(
            contact, 0.8, merge_gap=2, minimum_frames=6, window_frames=16, overlap=4
        )
        self.assertTrue(masks[20:22, 0].all())
        self.assertEqual([(item.foot, item.start, item.end) for item in windows], [(0, 10, 26), (0, 22, 35), (1, 50, 57)])

    def test_bad_track_point_and_low_visibility_are_rejected(self) -> None:
        window = TrackWindow(0, 0, 8)
        tracks = np.zeros((8, 3, 2), dtype=np.float64)
        tracks[..., 0] = np.array([40.0, 50.0, 60.0])
        tracks[..., 1] = 60.0
        tracks[:, :, 0] += np.arange(8)[:, None]
        tracks[4, 2] += 100.0
        visibility = np.ones((8, 3), dtype=bool)
        visibility[6, :2] = False
        intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        source = np.zeros((8, 3, 2), dtype=np.float64)
        constraint = tracked_relative_constraint(
            window,
            tracks,
            visibility,
            source,
            intrinsics,
            np.array([0.0, 1.0, 0.0]),
            -1.0,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.eye(2),
            np.zeros(2),
            (200, 200),
            max_track_step_pixels=35.0,
            max_point_disagreement=0.06,
        )
        self.assertIsNotNone(constraint)
        assert constraint is not None
        self.assertNotIn(6, constraint.frames.tolist())
        self.assertTrue(np.isfinite(constraint.delta_xz).all())
        self.assertLessEqual(int(constraint.points_used[4]), 2)

    def test_constraint_solver_recovers_relative_motion_continuously(self) -> None:
        frames = np.arange(0, 20)
        expected = np.stack([0.002 * frames, -0.001 * frames], axis=-1)
        constraint = RelativeConstraint(
            foot=0,
            start=0,
            frames=frames,
            delta_xz=expected,
            weights=np.ones(len(frames)),
            points_used=np.full(len(frames), 3),
        )
        correction, diagnostics = solve_relative_constraints(
            40, [constraint], smooth_weight=0.01, absolute_weight=0.0
        )
        np.testing.assert_allclose(correction[19], expected[19], atol=0.002)
        self.assertLess(float(np.max(np.linalg.norm(np.diff(correction, axis=0), axis=1))), 0.005)
        self.assertEqual(diagnostics["constraint_windows"], 1)

    def test_clone_changes_only_global_root_xz(self) -> None:
        source = {
            "smpl_params_global": {
                "transl": torch.zeros(5, 3),
                "body_pose": torch.ones(5, 63),
                "global_orient": torch.ones(5, 3),
                "betas": torch.ones(5, 10),
            },
            "smpl_params_incam": {"transl": torch.randn(5, 3)},
            "K_fullimg": torch.eye(3).repeat(5, 1, 1),
            "net_outputs": {"value": torch.randn(1)},
        }
        correction = np.tile([0.1, -0.2], (5, 1))
        enhanced = clone_with_root_xz(source, correction)
        np.testing.assert_allclose(enhanced["smpl_params_global"]["transl"][:, (0, 2)], correction)
        self.assertTrue(torch.equal(source["smpl_params_global"]["transl"][:, 1], enhanced["smpl_params_global"]["transl"][:, 1]))
        self.assertTrue(torch.equal(source["smpl_params_global"]["body_pose"], enhanced["smpl_params_global"]["body_pose"]))
        self.assertTrue(torch.equal(source["smpl_params_incam"]["transl"], enhanced["smpl_params_incam"]["transl"]))


if __name__ == "__main__":
    unittest.main()
