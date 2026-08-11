#!/usr/bin/env python3
"""CPU-only checks for automatic Human3R ground constraints."""

from __future__ import annotations

import unittest

import numpy as np

from apply_ground_xyz import (
    first_stable_calibration,
    intersect_ground,
    smooth_observations,
)
from extract_ground_plane import plane_score


class GroundXYZHelpersTest(unittest.TestCase):
    def test_ray_plane_intersection(self) -> None:
        intrinsics = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )
        pixels = np.array([[[50.0, 60.0], [70.0, 70.0]]])
        points = intersect_ground(
            pixels, intrinsics, np.array([0.0, 1.0, 0.0]), -1.0
        )

        np.testing.assert_allclose(points[0, 0], [0.0, 1.0, 10.0])
        np.testing.assert_allclose(points[0, 1], [1.0, 1.0, 5.0])

    def test_contact_smoothing_rejects_one_outlier_and_interpolates(self) -> None:
        length = 180
        frame = np.arange(length)
        expected = np.stack(
            [0.12 * frame / (length - 1), -0.06 * frame / (length - 1), 0.04 * frame / (length - 1)],
            axis=-1,
        )
        observed = expected.copy()
        mask = np.ones(length, dtype=bool)
        mask[55:85] = False
        observed[~mask] = np.nan
        observed[120] += np.array([1.0, -1.0, 1.0])
        calibration = np.arange(20)

        correction, diagnostics = smooth_observations(
            observed,
            mask,
            calibration,
            fps=30.0,
            smoothing_seconds=0.5,
            minimum_observations=30,
        )

        self.assertEqual(diagnostics["outlier_frames"], 1)
        self.assertTrue(np.isfinite(correction).all())
        self.assertLess(float(np.max(np.linalg.norm(np.diff(correction, axis=0), axis=1))), 0.01)
        np.testing.assert_allclose(correction[-1], expected[-1] - expected[9:11].mean(axis=0), atol=0.015)

    def test_calibration_and_ground_scoring_are_automatic(self) -> None:
        marker_mask = np.zeros((100, 6), dtype=bool)
        marker_mask[12:52, :3] = True
        frames = first_stable_calibration(marker_mask, 30.0, 2.0, 30)
        self.assertEqual(int(frames[0]), 12)
        self.assertEqual(int(frames[-1]), 51)

        floor_pixels = np.array([[u, v] for u in range(20, 180, 4) for v in range(110, 195, 4)])
        wall_pixels = np.array([[u, v] for u in range(20, 180, 4) for v in range(10, 110, 4)])
        floor = {
            "pixels": floor_pixels,
            "pixel_bounds": [floor_pixels.min(axis=0), floor_pixels.max(axis=0)],
            "points": len(floor_pixels),
        }
        wall = {
            "pixels": wall_pixels,
            "pixel_bounds": [wall_pixels.min(axis=0), wall_pixels.max(axis=0)],
            "points": len(wall_pixels),
        }
        self.assertGreater(
            plane_score(floor, 200, 200)["score"],
            plane_score(wall, 200, 200)["score"],
        )

        feet = np.zeros((100, 6, 3))
        selected_xz = feet[frames][:, :, (0, 2)][marker_mask[frames]]
        self.assertEqual(selected_xz.shape, (len(frames) * 3, 2))


if __name__ == "__main__":
    unittest.main()
