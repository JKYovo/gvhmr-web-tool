import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from scipy.spatial.transform import Rotation, Slerp

from hmr4d.utils.long_video import (
    predict_long_video,
    slice_data,
    stitch_predictions,
    window_starts,
)
from hmr4d.model.gvhmr.utils.postprocess import (
    _cumulative_displacement,
    _static_camera_root_corrections,
)
from hmr4d.utils.preproc.vitfeat_extractor import (
    create_batch_memmap,
    get_batch,
    get_batch_memmap,
    get_or_create_batch_memmap,
    open_batch_memmap,
)
from hmr4d.network.base_arch.transformer.encoder_rope import RoPEAttention


def make_result(length, *, yaw=0.0, offset=(0.0, 0.0, 0.0), value=0.0):
    body = torch.zeros(length, 63)
    body[:, 0] = value
    orient = torch.tensor([[0.0, yaw, 0.0]]).repeat(length, 1)
    transl = torch.stack(
        (torch.arange(length, dtype=torch.float32), torch.zeros(length), torch.zeros(length)),
        dim=-1,
    )
    transl += torch.tensor(offset)
    params = {
        "body_pose": body,
        "betas": torch.full((length, 10), value),
        "global_orient": orient,
        "transl": transl,
    }
    incam = {key: tensor.clone() for key, tensor in params.items()}
    global_params = {key: tensor.clone() for key, tensor in params.items()}
    return {
        "smpl_params_global": global_params,
        "smpl_params_incam": incam,
        "K_fullimg": torch.eye(3).repeat(length, 1, 1),
        "net_outputs": {
            "pred_smpl_params_global": {
                key: tensor.unsqueeze(0) for key, tensor in global_params.items()
            },
            "pred_smpl_params_incam": {
                key: tensor.unsqueeze(0) for key, tensor in incam.items()
            },
            "static_conf_logits": torch.zeros(1, length, 6),
            "decode_dict": {"body_pose": body.unsqueeze(0)},
        },
    }


class FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, data, static_cam=False, no_postproc=False):
        self.calls += 1
        self.last_no_postproc = no_postproc
        return make_result(int(data["length"]), value=float(self.calls))


class LongVideoTest(unittest.TestCase):
    def test_dense_and_memory_bounded_local_attention_are_exact(self):
        torch.manual_seed(20260813)
        attention = RoPEAttention(64, 4, dropout=0.0, attention_chunk_size=31).eval()
        length = 257
        window = 120
        inputs = torch.randn(1, length, 64)
        padding = torch.zeros(1, length, dtype=torch.bool)
        dense_mask = torch.ones(length, length, dtype=torch.bool)
        for index in range(length):
            start = max(0, index - window // 2)
            end = min(length, index + window // 2)
            end = max(window, end)
            start = min(length - window, start)
            dense_mask[index, start:end] = False
        with torch.inference_mode():
            dense = attention(inputs, dense_mask, padding)
            local = attention(inputs, ("local", window), padding)
        torch.testing.assert_close(local, dense, rtol=2e-5, atol=2e-6)

    def test_streaming_shared_crop_matches_in_memory_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "crop.mp4"
            frames = np.random.default_rng(20260813).integers(
                0, 256, size=(5, 72, 96, 3), dtype=np.uint8
            )
            iio.imwrite(video, frames, plugin="pyav", fps=30, codec="libx264")
            boxes = torch.tensor(
                [[48.0, 36.0, 60.0], [47.0, 35.0, 58.0], [49.0, 34.0, 55.0],
                 [48.0, 37.0, 62.0], [46.0, 36.0, 57.0]]
            )
            expected_images, expected_boxes = get_batch(str(video), boxes, img_ds=0.5)
            mapped_images, mapped_boxes = get_batch_memmap(
                video, boxes, root / "crop.mmap", img_ds=0.5
            )
            torch.testing.assert_close(mapped_images, expected_images, rtol=0, atol=0)
            torch.testing.assert_close(mapped_boxes, expected_boxes, rtol=0, atol=0)

            persisted_images, persisted_boxes = create_batch_memmap(
                video,
                boxes,
                root / "persisted.mmap",
                source_sha256="unit-test-video",
                img_ds=0.5,
            )
            reopened, metadata = open_batch_memmap(root / "persisted.mmap")
            torch.testing.assert_close(reopened, persisted_images, rtol=0, atol=0)
            torch.testing.assert_close(
                torch.tensor(metadata["bbx_xys"]), persisted_boxes, rtol=0, atol=0
            )
            reused_images, reused_boxes, reused = get_or_create_batch_memmap(
                video,
                boxes,
                root / "persisted.mmap",
                source_sha256="unit-test-video",
                img_ds=0.5,
            )
            self.assertTrue(reused)
            torch.testing.assert_close(reused_images, persisted_images, rtol=0, atol=0)
            torch.testing.assert_close(reused_boxes, persisted_boxes, rtol=0, atol=0)

    def test_linear_static_camera_rollout_matches_original_suffix_loop(self):
        generator = torch.Generator().manual_seed(20260813)
        pred_w = torch.randn(2, 37, 3, generator=generator)
        pred_c = torch.randn(2, 37, 3, generator=generator)
        threshold = torch.tensor([0.25, 0.25, 0.25])
        original = pred_w.clone()
        for index in range(1, len(original[0])):
            difference = original[:, index] - pred_c[:, index]
            difference *= ~(
                (difference > -threshold) * (difference < threshold)
            )
            difference = torch.clamp(difference, -0.02, 0.02)
            original[:, index:] -= difference[:, None]
        correction = _static_camera_root_corrections(pred_w, pred_c, threshold)
        torch.testing.assert_close(pred_w - correction, original, rtol=0, atol=2e-7)

    def test_cumulative_displacement_matches_original_suffix_loop(self):
        generator = torch.Generator().manual_seed(20260813)
        initial = torch.randn(2, 37, 3, generator=generator)
        displacement = torch.randn(2, 36, 3, generator=generator)
        original = initial.clone()
        for index in range(1, len(original[0])):
            original[:, index:] -= displacement[:, [index - 1]]
        optimized = initial - _cumulative_displacement(displacement)
        torch.testing.assert_close(optimized, original, rtol=0, atol=2e-6)

    def test_window_starts_covers_special_final_overlap(self):
        starts = window_starts(7162, 600, 480)
        self.assertEqual(len(starts), 15)
        self.assertEqual(starts[-2:], [6240, 6562])
        self.assertEqual(starts[-1], 7162 - 600)

    def test_slice_data_only_slices_frame_aligned_tensors(self):
        data = {
            "length": torch.tensor(10),
            "frames": torch.arange(20).reshape(10, 2),
            "constant": torch.ones(3),
            "label": "keep",
        }
        sliced = slice_data(data, 3, 7)
        self.assertEqual(int(sliced["length"]), 4)
        torch.testing.assert_close(sliced["frames"], data["frames"][3:7])
        self.assertIs(sliced["constant"], data["constant"])
        self.assertEqual(sliced["label"], "keep")

    def test_stitch_is_finite_and_uses_rotation_slerp(self):
        first = make_result(6, yaw=0.0, value=0.0)
        second = make_result(6, yaw=np.pi / 2, value=2.0)
        first["smpl_params_incam"]["global_orient"][:] = torch.tensor(
            Rotation.from_euler("x", 90, degrees=True).as_rotvec(), dtype=torch.float32
        )
        second["smpl_params_incam"]["global_orient"][:] = torch.tensor(
            Rotation.from_euler("y", 90, degrees=True).as_rotvec(), dtype=torch.float32
        )
        # Give both windows the same global trajectory in their shared range,
        # so alignment does not obscure the incam SLERP assertion.
        second["smpl_params_global"]["transl"][:, 0] += 4
        second["net_outputs"]["pred_smpl_params_global"]["transl"] = second[
            "smpl_params_global"
        ]["transl"].unsqueeze(0)
        stitched, alignments = stitch_predictions([first, second], [0, 4], 10)
        self.assertEqual(tuple(stitched["smpl_params_global"]["body_pose"].shape), (10, 63))
        self.assertTrue(torch.isfinite(stitched["smpl_params_global"]["transl"]).all())
        self.assertEqual(len(alignments), 1)
        midpoint = stitched["smpl_params_incam"]["global_orient"][5].numpy()
        endpoints = Rotation.from_euler("xy", [[90, 0], [0, 90]], degrees=True)
        expected = Slerp([0.0, 1.0], endpoints)([0.5]).as_rotvec()[0]
        np.testing.assert_allclose(midpoint, expected, atol=1e-5)

    def test_stitch_rejects_gap_and_triple_overlap(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            stitch_predictions([make_result(4), make_result(4)], [0, 5], 9)
        with self.assertRaisesRegex(ValueError, "Triple"):
            stitch_predictions(
                [make_result(6), make_result(6), make_result(6)], [0, 2, 4], 10
            )

    def test_prediction_windows_resume_without_reusing_other_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            video.write_bytes(b"stable-video-identity")
            data = {"length": torch.tensor(10), "feature": torch.arange(10)}
            model = FakeModel()
            kwargs = dict(
                model=model,
                data=data,
                normalized_video=video,
                work_root=root / "work",
                static_cam=True,
                no_postproc=True,
                detach_to_cpu=lambda value: copy.deepcopy(value),
                window_frames=6,
                stride_frames=4,
                log=lambda _message: None,
            )
            result, metrics = predict_long_video(**kwargs, cache_identity="model-a")
            self.assertEqual(model.calls, 2)
            self.assertEqual(metrics["reused_windows"], 0)
            self.assertEqual(result["smpl_params_global"]["body_pose"].shape[0], 10)

            _, resumed = predict_long_video(**kwargs, cache_identity="model-a")
            self.assertEqual(model.calls, 2)
            self.assertEqual(resumed["reused_windows"], 2)

            _, changed = predict_long_video(**kwargs, cache_identity="model-b")
            self.assertEqual(model.calls, 4)
            self.assertEqual(changed["reused_windows"], 0)


if __name__ == "__main__":
    unittest.main()
