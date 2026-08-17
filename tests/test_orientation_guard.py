import copy
import unittest

import torch

from hmr4d.utils.orientation_guard import (
    detect_isolated_orientation_jumps,
    guard_isolated_orientation_jumps,
    rotation_step_degrees,
)


def make_orientations(length=30, *, boundary=14, impulse_deg=107.0):
    increments = torch.full((length - 1,), 3.0)
    increments[boundary] = impulse_deg
    yaw = torch.cat((torch.zeros(1), torch.cumsum(increments, dim=0)))
    global_orient = torch.zeros(length, 3)
    global_orient[:, 1] = torch.deg2rad(yaw)
    incam_orient = global_orient.clone()
    incam_orient[:, 1] += torch.deg2rad(torch.linspace(0.0, -12.0, length))
    return global_orient, incam_orient


def make_result(global_orient, incam_orient):
    length = len(global_orient)
    global_params = {
        "global_orient": global_orient.clone(),
        "body_pose": torch.randn(length, 63),
        "betas": torch.randn(length, 10),
        "transl": torch.randn(length, 3),
    }
    incam_params = {
        "global_orient": incam_orient.clone(),
        "body_pose": global_params["body_pose"].clone(),
        "betas": global_params["betas"].clone(),
        "transl": torch.randn(length, 3),
    }
    return {
        "smpl_params_global": global_params,
        "smpl_params_incam": incam_params,
        "K_fullimg": torch.eye(3).repeat(length, 1, 1),
        "net_outputs": {
            "pred_smpl_params_global": {
                key: value.unsqueeze(0).clone() for key, value in global_params.items()
            },
            "pred_smpl_params_incam": {
                key: value.unsqueeze(0).clone() for key, value in incam_params.items()
            },
            "unrelated": torch.randn(1, length, 7),
        },
    }


class OrientationGuardTest(unittest.TestCase):
    def test_repairs_only_isolated_root_orientation_window(self):
        global_orient, incam_orient = make_orientations()
        original = make_result(global_orient, incam_orient)
        snapshot = copy.deepcopy(original)

        guarded, metrics = guard_isolated_orientation_jumps(original)

        self.assertTrue(metrics["triggered"])
        self.assertEqual(metrics["num_detections"], 1)
        detection = metrics["detections"][0]
        self.assertEqual(detection["boundary_frame"], 14)
        self.assertLess(metrics["max_global_step_after_deg"], 30.0)
        self.assertLess(metrics["max_incam_step_after_deg"], 30.0)

        for space in ("global", "incam"):
            params = guarded[f"smpl_params_{space}"]
            original_params = snapshot[f"smpl_params_{space}"]
            for name in ("body_pose", "betas", "transl"):
                torch.testing.assert_close(params[name], original_params[name], rtol=0, atol=0)
            orient = params["global_orient"]
            original_orient = original_params["global_orient"]
            start = detection["window_start"]
            end = detection["window_end"]
            torch.testing.assert_close(orient[: start + 1], original_orient[: start + 1], rtol=0, atol=0)
            torch.testing.assert_close(orient[end:], original_orient[end:], rtol=0, atol=0)
            torch.testing.assert_close(orient[start], original_orient[start], rtol=0, atol=0)
            torch.testing.assert_close(orient[end], original_orient[end], rtol=0, atol=0)
            self.assertFalse(torch.equal(orient[start + 1 : end], original_orient[start + 1 : end]))
            torch.testing.assert_close(
                guarded["net_outputs"][f"pred_smpl_params_{space}"]["global_orient"][0],
                orient,
                rtol=0,
                atol=0,
            )

        # The function is non-mutating when a repair is applied.
        torch.testing.assert_close(
            original["smpl_params_global"]["global_orient"],
            snapshot["smpl_params_global"]["global_orient"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            guarded["K_fullimg"], snapshot["K_fullimg"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            guarded["net_outputs"]["unrelated"],
            snapshot["net_outputs"]["unrelated"],
            rtol=0,
            atol=0,
        )

    def test_normal_fast_turn_is_not_modified(self):
        global_orient, incam_orient = make_orientations(impulse_deg=29.0)
        result = make_result(global_orient, incam_orient)
        guarded, metrics = guard_isolated_orientation_jumps(result)
        self.assertIs(guarded, result)
        self.assertFalse(metrics["triggered"])
        self.assertEqual(metrics["num_detections"], 0)

    def test_nearby_second_extreme_peak_disables_repair(self):
        global_orient, incam_orient = make_orientations()
        for orient in (global_orient, incam_orient):
            rotations = orient[:, 1].clone()
            rotations[18:] += torch.deg2rad(torch.tensor(85.0))
            orient[:, 1] = rotations
        detections, _, _ = detect_isolated_orientation_jumps(
            global_orient, incam_orient
        )
        self.assertEqual(detections, [])

    def test_step_angles_are_finite_and_have_expected_shape(self):
        global_orient, _ = make_orientations()
        steps = rotation_step_degrees(global_orient)
        self.assertEqual(tuple(steps.shape), (len(global_orient) - 1,))
        self.assertTrue(torch.isfinite(steps).all())


if __name__ == "__main__":
    unittest.main()
