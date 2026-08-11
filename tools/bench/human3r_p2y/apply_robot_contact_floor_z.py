#!/usr/bin/env python3
"""Anchor a retargeted robot's contacting foot sites to the MuJoCo floor."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-site", default="lf_tc")
    parser.add_argument("--right-site", default="rf_tc")
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--target-clearance", type=float, default=0.03)
    parser.add_argument(
        "--minimum-clearance",
        type=float,
        default=0.005,
        help="Hard lower bound for either robot foot site after correction.",
    )
    parser.add_argument("--smoothing-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-observations", type=int, default=30)
    return parser.parse_args()


def foot_contact_mask(contact: np.ndarray, threshold: float) -> np.ndarray:
    contact = np.asarray(contact)
    if contact.ndim != 2 or contact.shape[1] < 4:
        raise ValueError(f"Expected contact probabilities (T, >=4), got {contact.shape}")
    return np.stack(
        [
            np.max(contact[:, :2], axis=1) > threshold,
            np.max(contact[:, 2:4], axis=1) > threshold,
        ],
        axis=1,
    )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(views, axis=-1)


def smooth_contact_correction(
    observations: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    smoothing_seconds: float,
    minimum_observations: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    valid = valid & np.isfinite(observations)
    if int(valid.sum()) < minimum_observations:
        raise RuntimeError(f"Too few robot contact observations: {int(valid.sum())}")

    frames = np.arange(len(observations))
    indices = np.flatnonzero(valid)
    filled = np.interp(frames, indices, observations[indices])
    median_window = max(3, int(round(0.25 * fps)))
    trend = rolling_median(filled, median_window)

    smooth_window = max(3, int(round(smoothing_seconds * fps)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.hanning(smooth_window)
    if not np.any(kernel):
        kernel = np.ones(smooth_window)
    kernel /= kernel.sum()
    radius = smooth_window // 2
    correction = np.convolve(
        np.pad(trend, (radius, radius), mode="edge"), kernel, mode="valid"
    )
    return correction, {
        "observation_frames": int(valid.sum()),
        "first_observation_frame": int(indices[0]),
        "last_observation_frame": int(indices[-1]),
        "median_window_frames": int(median_window),
        "smoothing_window_frames": int(smooth_window),
    }


def motion_qpos(motion: dict, frame: int) -> np.ndarray:
    root_rot_xyzw = np.asarray(motion["root_rot"][frame], dtype=np.float64)
    root_rot_xyzw /= max(float(np.linalg.norm(root_rot_xyzw)), 1e-12)
    return np.concatenate(
        [
            np.asarray(motion["root_pos"][frame], dtype=np.float64),
            root_rot_xyzw[[3, 0, 1, 2]],
            np.asarray(motion["dof_pos"][frame], dtype=np.float64),
        ]
    )


def foot_site_heights(
    model: mujoco.MjModel,
    motion: dict,
    site_names: tuple[str, str],
) -> np.ndarray:
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    ]
    if any(site_id < 0 for site_id in site_ids):
        raise ValueError(f"Robot XML does not contain both foot sites: {site_names}")
    length = len(motion["root_pos"])
    heights = np.empty((length, 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(length):
        qpos = motion_qpos(motion, frame)
        if len(qpos) != model.nq:
            raise ValueError(f"qpos/model mismatch: {len(qpos)} vs {model.nq}")
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        heights[frame] = [data.site_xpos[site_id, 2] for site_id in site_ids]
    return heights


def root_metrics(root: np.ndarray, fps: float) -> dict[str, float]:
    step = np.abs(np.diff(root[:, 2]))
    acceleration = np.abs(np.diff(root[:, 2], n=2)) * fps * fps
    return {
        "root_z_step_max_cm_per_frame": float(np.max(step) * 100.0),
        "root_z_step_p95_cm_per_frame": float(np.percentile(step, 95) * 100.0),
        "root_z_accel_p95_m_per_s2": float(np.percentile(acceleration, 95)),
    }


def contact_metrics(heights: np.ndarray, contact: np.ndarray) -> dict[str, float | int]:
    frame_mask = np.any(contact, axis=1)
    anchor = np.asarray(
        [np.min(heights[i, contact[i]]) for i in np.flatnonzero(frame_mask)]
    )
    samples = heights[contact]
    return {
        "contact_frames": int(frame_mask.sum()),
        "contact_foot_samples": int(contact.sum()),
        "anchor_median_cm": float(np.median(anchor) * 100.0),
        "anchor_p05_cm": float(np.percentile(anchor, 5) * 100.0),
        "anchor_p95_cm": float(np.percentile(anchor, 95) * 100.0),
        "contact_sample_median_cm": float(np.median(samples) * 100.0),
        "minimum_site_height_cm": float(np.min(heights) * 100.0),
        "below_floor_frames": int(np.any(heights < 0.0, axis=1).sum()),
    }


def main() -> None:
    args = parse_args()
    for path in (args.motion, args.gvhmr_result, args.robot_xml):
        if not path.is_file():
            raise FileNotFoundError(path)

    with args.motion.open("rb") as file:
        source_motion = pickle.load(file)
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = required - set(source_motion)
    if missing:
        raise ValueError(f"robot_motion.pkl missing fields: {sorted(missing)}")

    gvhmr_result = torch.load(args.gvhmr_result, map_location="cpu")
    probabilities = (
        gvhmr_result["net_outputs"]["static_conf_logits"][0, :, :4]
        .sigmoid()
        .numpy()
    )
    contact = foot_contact_mask(probabilities, args.contact_threshold)
    length = len(source_motion["root_pos"])
    if len(contact) != length:
        raise ValueError(f"GVHMR/robot length mismatch: {len(contact)} vs {length}")

    fps = float(source_motion["fps"])
    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    site_names = (args.left_site, args.right_site)
    before_heights = foot_site_heights(model, source_motion, site_names)
    frame_mask = np.any(contact, axis=1)
    observations = np.full(length, np.nan, dtype=np.float64)
    for frame in np.flatnonzero(frame_mask):
        observations[frame] = args.target_clearance - float(
            np.min(before_heights[frame, contact[frame]])
        )
    correction, smoothing = smooth_contact_correction(
        observations,
        frame_mask,
        fps=fps,
        smoothing_seconds=args.smoothing_seconds,
        minimum_observations=args.minimum_observations,
    )
    floor_lower_bound = args.minimum_clearance - np.min(before_heights, axis=1)
    unclamped_correction = correction.copy()
    correction = np.maximum(correction, floor_lower_bound)
    floor_clamped = correction > unclamped_correction + 1e-12

    enhanced_motion = copy.deepcopy(source_motion)
    root_before = np.asarray(source_motion["root_pos"], dtype=np.float64)
    root_after = root_before.copy()
    root_after[:, 2] += correction
    original_dtype = np.asarray(source_motion["root_pos"]).dtype
    enhanced_motion["root_pos"] = root_after.astype(original_dtype, copy=False)
    after_heights = foot_site_heights(model, enhanced_motion, site_names)

    metrics = {
        "method": "robot_contact_floor_z",
        "motion": str(args.motion),
        "gvhmr_result": str(args.gvhmr_result),
        "robot_xml": str(args.robot_xml),
        "fps": fps,
        "frames": length,
        "contact_threshold": args.contact_threshold,
        "target_clearance_cm": args.target_clearance * 100.0,
        "minimum_clearance_cm": args.minimum_clearance * 100.0,
        "smoothing_seconds": args.smoothing_seconds,
        "correction_cm": {
            "min": float(np.min(correction) * 100.0),
            "median": float(np.median(correction) * 100.0),
            "max": float(np.max(correction) * 100.0),
        },
        "smoothing": smoothing,
        "floor_guard": {
            "clamped_frames": int(floor_clamped.sum()),
            "max_clamp_delta_cm": float(
                np.max(correction - unclamped_correction) * 100.0
            ),
        },
        "before": {
            **contact_metrics(before_heights, contact),
            **root_metrics(root_before, fps),
        },
        "after": {
            **contact_metrics(after_heights, contact),
            **root_metrics(root_after, fps),
        },
    }
    if not np.isfinite(root_after).all() or not np.isfinite(after_heights).all():
        raise RuntimeError("Non-finite values after robot floor correction")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = args.output_dir / "robot_motion_contact_floor_z.pkl"
    metrics_path = args.output_dir / "robot_contact_floor_z_metrics.json"
    with motion_path.open("wb") as file:
        pickle.dump(enhanced_motion, file)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved motion: {motion_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
