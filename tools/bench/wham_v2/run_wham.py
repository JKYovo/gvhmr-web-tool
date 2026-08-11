#!/usr/bin/env python3
"""Run official WHAM and persist W0/W1/W2 trajectories for one static-camera video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import cv2
import joblib
import numpy as np
import torch
from progress.bar import Bar


EXPECTED_WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wham-root", type=Path, default=Path("/home/user-kevien/gvhmr_pkg/WHAM"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def select_primary_track(tracking_results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    if not tracking_results:
        raise RuntimeError("WHAM did not find a person track longer than 30 frames")

    def score(item: Any) -> Any:
        _, track = item
        bbox_scale = np.asarray(track["bbox"])[..., 2]
        return len(track["frame_id"]), float(np.median(bbox_scale))

    track_id, track = max(tracking_results.items(), key=score)
    print(
        f"Selected WHAM track id={track_id}, frames={len(track['frame_id'])}, "
        f"range={int(track['frame_id'][0])}:{int(track['frame_id'][-1])}"
    )
    return {track_id: track}


@torch.inference_mode()
def preprocess_static_video(cfg: Any, video: Path, cache_path: Path) -> tuple:
    from lib.models.preproc.detector import DetectionModel
    from lib.models.preproc.extractor import FeatureExtractor

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = DetectionModel(cfg.DEVICE.lower())
    bar = Bar("WHAM detection", fill="#", max=length)
    while cap.isOpened():
        ok, image = cap.read()
        if not ok:
            break
        detector.track(image, fps, length)
        bar.next()
    bar.finish()
    cap.release()
    tracking_results = select_primary_track(detector.process(fps))
    del detector
    torch.cuda.empty_cache()

    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
    tracking_results = extractor.run(str(video), tracking_results)
    del extractor
    torch.cuda.empty_cache()
    joblib.dump(tracking_results, cache_path)

    slam_results = np.zeros((length, 7), dtype=np.float32)
    slam_results[:, 3] = 1.0
    return tracking_results, slam_results, width, height, fps, length


def clone_tensor_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tensor_tree(item) for key, item in value.items()}
    return value


def world_feet(smpl: Any, network_output: Any, root_world: torch.Tensor, trans_world: torch.Tensor) -> torch.Tensor:
    batch, frames = root_world.shape[:2]
    output = smpl.get_output(
        body_pose=network_output.body_pose.detach().reshape(batch * frames, 23, 3, 3),
        global_orient=root_world.detach().reshape(batch * frames, 1, 3, 3),
        betas=network_output.betas.detach().reshape(batch * frames, 10),
        pose2rot=False,
    )
    return output.feet.reshape(batch, frames, 4, 3) + trans_world.unsqueeze(-2)


def pack_variant(
    name: str,
    output: Dict[str, torch.Tensor],
    smpl: Any,
    network_output: Any,
    root_r6d: torch.Tensor,
    velocity: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    from lib.utils.transforms import matrix_to_axis_angle

    root_world = output["poses_root_world"]
    trans_world = output["trans_world"]
    feet_world = world_feet(smpl, network_output, root_world, trans_world)
    return {
        "name": name,
        "root_r6d": root_r6d.detach().cpu().squeeze(0),
        "velocity_root": velocity.detach().cpu().squeeze(0),
        "root_orient_world": matrix_to_axis_angle(root_world).detach().cpu().squeeze(0),
        "trans_world": trans_world.detach().cpu().squeeze(0),
        "feet_world": feet_world.detach().cpu().squeeze(0),
    }


@torch.no_grad()
def infer_subject(network: Any, smpl: Any, batch: tuple, flip_eval: bool) -> Dict[str, Any]:
    from lib.models.layers import reset_root_velocity
    from lib.utils.imutils import avg_preds
    from lib.utils.transforms import matrix_to_axis_angle

    if flip_eval:
        flipped = batch["flipped"]
        network(
            flipped[1], flipped[2], flipped[3], mask=flipped[4],
            init_root=flipped[5], cam_angvel=flipped[6],
            return_y_up=True, refine_traj=False, **flipped[8],
        )
        flipped_pose = network.pred_pose.detach().clone().reshape(-1, 24, 6)
        flipped_shape = network.pred_shape.detach().clone().squeeze(0)
        flipped_contact = network.pred_contact.detach().clone()

    normal = batch["normal"]
    network(
        normal[1], normal[2], normal[3], mask=normal[4],
        init_root=normal[5], cam_angvel=normal[6],
        return_y_up=True, refine_traj=False, **normal[8],
    )

    if flip_eval:
        pose = network.pred_pose.detach().clone().reshape(-1, 24, 6)
        shape = network.pred_shape.detach().clone().squeeze(0)
        avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
        avg_contact = (flipped_contact[..., [2, 3, 0, 1]] + network.pred_contact) / 2.0
        network.pred_pose = avg_pose.reshape_as(network.pred_pose)
        network.pred_shape = avg_shape.reshape_as(network.pred_shape)
        network.pred_contact = avg_contact.reshape_as(network.pred_contact)

    base_output = network.forward_smpl(**normal[8])
    w0_output = network.rollout(
        clone_tensor_tree(base_output), network.pred_root, network.pred_vel, return_y_up=True
    )
    reset_velocity = reset_root_velocity(
        smpl, network.output, network.pred_contact, network.pred_root, network.pred_vel, thr=0.5
    )
    w1_output = network.rollout(
        clone_tensor_tree(base_output), network.pred_root, reset_velocity, return_y_up=True
    )
    w2_output = network.trajectory_refiner(
        network.old_motion_context,
        reset_velocity,
        clone_tensor_tree(base_output),
        normal[6],
        return_y_up=True,
    )
    w2_output = network.rollout(
        w2_output,
        w2_output["poses_root_r6d_refined"],
        w2_output["vel_root_refined"],
        return_y_up=True,
    )

    body_pose = matrix_to_axis_angle(network.output.body_pose).reshape(-1, 69).detach().cpu()
    return {
        "subject_id": int(normal[0]),
        "frame_ids": torch.as_tensor(normal[7], dtype=torch.long),
        "contact": network.pred_contact.detach().cpu().squeeze(0),
        "body_pose": body_pose,
        "betas": network.pred_shape.detach().cpu().squeeze(0),
        "variants": {
            "w0": pack_variant("trajectory_decoder", w0_output, smpl, network.output, network.pred_root, network.pred_vel),
            "w1": pack_variant("contact_reset", w1_output, smpl, network.output, network.pred_root, reset_velocity),
            "w2": pack_variant(
                "trajectory_refiner",
                w2_output,
                smpl,
                network.output,
                w2_output["poses_root_r6d_refined"],
                w2_output["vel_root_refined"],
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.video = args.video.resolve()
    args.output_dir = args.output_dir.resolve()
    args.wham_root = args.wham_root.resolve()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    commit = git_commit(args.wham_root)
    if commit != EXPECTED_WHAM_COMMIT:
        raise RuntimeError(f"Expected WHAM {EXPECTED_WHAM_COMMIT}, found {commit}")

    sys.path.insert(0, str(args.wham_root))
    os.chdir(args.wham_root)
    from configs.config import get_cfg_defaults
    from lib.data.datasets import CustomDataset
    from lib.models import build_body_model, build_network

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "tracking_results.pth"
    result_path = args.output_dir / "wham_w0_w1_w2.pt"
    manifest_path = args.output_dir / "wham_manifest.json"

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    cfg.DEVICE = args.device
    cfg.FLIP_EVAL = True

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if cache_path.exists() and not args.force_preprocess:
        tracking_results = joblib.load(cache_path)
        slam_results = np.zeros((length, 7), dtype=np.float32)
        slam_results[:, 3] = 1.0
        print(f"Loaded WHAM preprocessing cache: {cache_path}")
    else:
        tracking_results, slam_results, width, height, fps, length = preprocess_static_video(
            cfg, args.video, cache_path
        )

    assets = {
        "wham_checkpoint": args.wham_root / cfg.TRAIN.CHECKPOINT,
        "hmr2_checkpoint": args.wham_root / "checkpoints/hmr2a.ckpt",
        "vitpose_checkpoint": args.wham_root / "checkpoints/vitpose-h-multi-coco.pth",
        "yolo_checkpoint": args.wham_root / "checkpoints/yolov8x.pt",
        "smpl_neutral": args.wham_root / "dataset/body_models/smpl/SMPL_NEUTRAL.pkl",
        "foot_regressor": args.wham_root / "dataset/body_models/J_regressor_feet.npy",
    }
    missing = [str(path) for path in assets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing WHAM assets: {missing}")
    manifest = {
        "wham_commit": commit,
        "config": "configs/yamls/demo.yaml",
        "checkpoint": "wham_vit_bedlam_w_3dpw.pth.tar",
        "flip_eval": True,
        "temporal_smplify": False,
        "static_camera": True,
        "dpvo": False,
        "video": str(args.video),
        "video_frames": length,
        "fps": fps,
        "assets_sha256": {name: sha256(path.resolve()) for name, path in assets.items()},
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if result_path.exists() and not args.force_inference:
        print(f"WHAM result already exists: {result_path}")
        return

    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one selected WHAM track, found {len(dataset)}")
    smpl = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
    network = build_network(cfg, smpl).eval()
    normal = dataset.load_data(0)
    flipped = dataset.load_data(0, True)
    result = infer_subject(network, smpl, {"normal": normal, "flipped": flipped}, cfg.FLIP_EVAL)
    result["meta"] = manifest
    torch.save(result, result_path)
    print(f"Saved WHAM W0/W1/W2 result: {result_path}")


if __name__ == "__main__":
    main()
