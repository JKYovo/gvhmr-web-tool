#!/usr/bin/env python3
"""Constrain a FootMR global root-Y trajectory to one self-calibrated floor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from apply_ground_xyz import smooth_observations, trajectory_metrics
from apply_p2y import foot_landmarks, tree_equal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-observations", type=int, default=30)
    parser.add_argument("--smoothing-seconds", type=float, default=2.0)
    parser.add_argument("--max-y-correction", type=float, default=0.25)
    parser.add_argument("--max-contact-speed-ratio", type=float, default=1.25)
    return parser.parse_args()


def foot_confidence(contact: np.ndarray) -> np.ndarray:
    """Expand FootMR's two contact logits per foot to its three sole markers."""
    if contact.ndim != 2 or contact.shape[1] < 4:
        raise ValueError(f"Expected four contact probabilities, got {contact.shape}")
    return np.concatenate(
        [
            np.repeat(np.max(contact[:, :2], axis=1, keepdims=True), 3, axis=1),
            np.repeat(np.max(contact[:, 2:4], axis=1, keepdims=True), 3, axis=1),
        ],
        axis=1,
    )


def calibrate_marker_heights(
    feet_y: np.ndarray,
    confidence: np.ndarray,
    *,
    fps: float,
    threshold: float,
    calibration_seconds: float,
    minimum_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate each sole marker's initial support height from contact frames."""
    if feet_y.shape != confidence.shape or feet_y.ndim != 2:
        raise ValueError(f"feet/confidence shape mismatch: {feet_y.shape}, {confidence.shape}")
    length, markers = feet_y.shape
    search_end = min(length, max(1, int(round(calibration_seconds * fps))))
    marker_mask = confidence > threshold
    levels = np.empty(markers, dtype=np.float64)
    calibration_mask = np.zeros_like(marker_mask)
    for marker in range(markers):
        early = np.flatnonzero(marker_mask[:search_end, marker])
        selected = early
        if len(selected) < minimum_observations:
            selected = np.flatnonzero(marker_mask[:, marker])[:minimum_observations]
        if len(selected) < minimum_observations:
            raise RuntimeError(
                f"Too few contact samples for foot marker {marker}: {len(selected)}"
            )
        calibration_mask[selected, marker] = True
        levels[marker] = float(np.median(feet_y[selected, marker]))
    calibration_frames = np.flatnonzero(np.any(calibration_mask, axis=1))
    return levels, marker_mask, calibration_frames


def build_height_observations(
    feet_y: np.ndarray,
    marker_levels: np.ndarray,
    marker_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one robust root-Y residual from the contacting sole markers."""
    length = len(feet_y)
    observations = np.full((length, 1), np.nan, dtype=np.float64)
    observation_mask = np.any(marker_mask, axis=1)
    residual = marker_levels[None] - feet_y
    for frame_id in np.flatnonzero(observation_mask):
        observations[frame_id, 0] = float(
            np.median(residual[frame_id, marker_mask[frame_id]])
        )
    return observations, observation_mask


def height_residual_metrics(
    feet_y: np.ndarray,
    marker_levels: np.ndarray,
    marker_mask: np.ndarray,
    correction: np.ndarray,
) -> dict[str, dict[str, float]]:
    baseline = np.abs(feet_y - marker_levels[None])[marker_mask]
    enhanced = np.abs(feet_y + correction[:, None] - marker_levels[None])[marker_mask]

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "median_cm": float(np.median(values) * 100.0),
            "p95_cm": float(np.percentile(values, 95) * 100.0),
        }

    return {"baseline": summarize(baseline), "flat_ground_y": summarize(enhanced)}


def save_curve(
    output_dir: Path,
    root_before: np.ndarray,
    root_after: np.ndarray,
    correction: np.ndarray,
    observations: np.ndarray,
    observation_mask: np.ndarray,
    fps: float,
) -> None:
    time = np.arange(len(correction)) / fps
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(time, root_before, color="black", label="FootMR root Y")
    axes[0].plot(time, root_after, color="tab:blue", label="flat-ground root Y")
    axes[0].set_ylabel("root Y (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(time, correction, color="tab:blue", label="smoothed correction")
    axes[1].scatter(
        time[observation_mask],
        observations[observation_mask, 0],
        s=3,
        alpha=0.15,
        label="contact observation",
    )
    axes[1].set_ylabel("Y correction (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "flat_ground_y_curve.png", dpi=160)
    plt.close(figure)


def invariants(source: Mapping[str, Any], enhanced: Mapping[str, Any]) -> dict[str, bool]:
    source_translation = source["smpl_params_global"]["transl"]
    target_translation = enhanced["smpl_params_global"]["transl"]
    return {
        "top_level_keys_equal": source.keys() == enhanced.keys(),
        "global_param_keys_equal": source["smpl_params_global"].keys()
        == enhanced["smpl_params_global"].keys(),
        "incam_param_keys_equal": source["smpl_params_incam"].keys()
        == enhanced["smpl_params_incam"].keys(),
        "global_root_xz_equal": bool(
            torch.equal(source_translation[:, (0, 2)], target_translation[:, (0, 2)])
        ),
        "body_pose_equal": bool(
            torch.equal(
                source["smpl_params_global"]["body_pose"],
                enhanced["smpl_params_global"]["body_pose"],
            )
        ),
        "global_orient_equal": bool(
            torch.equal(
                source["smpl_params_global"]["global_orient"],
                enhanced["smpl_params_global"]["global_orient"],
            )
        ),
        "betas_equal": bool(
            torch.equal(
                source["smpl_params_global"]["betas"],
                enhanced["smpl_params_global"]["betas"],
            )
        ),
        "incam_equal": tree_equal(source["smpl_params_incam"], enhanced["smpl_params_incam"]),
        "K_fullimg_equal": tree_equal(source["K_fullimg"], enhanced["K_fullimg"]),
        "net_outputs_equal": tree_equal(source["net_outputs"], enhanced["net_outputs"]),
        "finite_root_xyz": bool(torch.isfinite(target_translation).all()),
    }


def main() -> None:
    args = parse_args()
    for path in (args.gvhmr_result, args.video):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.gvhmr_result, map_location="cpu")
    length = len(source["smpl_params_global"]["transl"])

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if video_frames != length:
        raise RuntimeError(f"Video/result length mismatch: {video_frames} vs {length}")

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23

    model = SmplxLiteV437Coco23().eval()
    feet = foot_landmarks(source, model)
    contact = source["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().numpy()
    confidence = foot_confidence(contact)
    marker_levels, marker_mask, calibration_frames = calibrate_marker_heights(
        feet[..., 1],
        confidence,
        fps=fps,
        threshold=args.contact_threshold,
        calibration_seconds=args.calibration_seconds,
        minimum_observations=args.minimum_observations,
    )
    observations, observation_mask = build_height_observations(
        feet[..., 1], marker_levels, marker_mask
    )
    correction_2d, smoothing = smooth_observations(
        observations,
        observation_mask,
        calibration_frames,
        fps,
        args.smoothing_seconds,
        args.minimum_observations,
    )
    correction = correction_2d[:, 0]

    enhanced = copy.deepcopy(source)
    translation = enhanced["smpl_params_global"]["transl"].clone()
    root_before = translation.numpy().copy()
    translation[:, 1] += torch.as_tensor(correction, dtype=translation.dtype)
    enhanced["smpl_params_global"]["transl"] = translation
    root_after = translation.numpy()
    enhanced_feet = feet.copy()
    enhanced_feet[..., 1] += correction[:, None]

    baseline_metrics = trajectory_metrics(root_before, feet, marker_mask, fps)
    enhanced_metrics = trajectory_metrics(root_after, enhanced_feet, marker_mask, fps)
    residual = height_residual_metrics(feet[..., 1], marker_levels, marker_mask, correction)
    invariant_checks = invariants(source, enhanced)
    if not all(invariant_checks.values()):
        raise RuntimeError(f"Flat-ground invariant failed: {invariant_checks}")

    max_y = float(np.max(np.abs(correction)))
    step_limit = max(3.0, baseline_metrics["root_step_max_cm_per_frame"] * 1.5)
    acceleration_limit = baseline_metrics["root_accel_p95_m_per_s2"] * 1.25
    contact_speed_limit = (
        baseline_metrics["contact_foot_speed_p95_mm_per_frame"]
        * args.max_contact_speed_ratio
    )
    height_improved = (
        residual["flat_ground_y"]["median_cm"] < residual["baseline"]["median_cm"]
        and residual["flat_ground_y"]["p95_cm"] < residual["baseline"]["p95_cm"]
    )
    guardrails = {
        "max_abs_y_pass": max_y <= args.max_y_correction,
        "root_step_pass": enhanced_metrics["root_step_max_cm_per_frame"] <= step_limit,
        "root_accel_pass": enhanced_metrics["root_accel_p95_m_per_s2"] <= acceleration_limit,
        "contact_speed_pass": enhanced_metrics["contact_foot_speed_p95_mm_per_frame"]
        <= contact_speed_limit,
        "height_median_and_p95_improved": height_improved,
    }
    decision = "diagnostic_pass" if all(guardrails.values()) else "guardrail_failed"
    payload = {
        "method": "FootMR self-calibrated flat-ground Y",
        "decision": decision,
        "frames": length,
        "fps": fps,
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "video": str(args.video.resolve()),
        "contact_threshold": args.contact_threshold,
        "contact_frames": int(observation_mask.sum()),
        "marker_levels_m": marker_levels.tolist(),
        "calibration_frames": calibration_frames.tolist(),
        "smoothing": smoothing,
        "correction": {
            "start_y_m": float(correction[0]),
            "end_y_m": float(correction[-1]),
            "max_abs_y_m": max_y,
        },
        "height_residual_cm": residual,
        "baseline": baseline_metrics,
        "flat_ground_y": enhanced_metrics,
        "limits": {
            "max_abs_y_m": args.max_y_correction,
            "root_step_cm_per_frame": step_limit,
            "root_accel_m_per_s2": acceleration_limit,
            "contact_speed_p95_mm_per_frame": contact_speed_limit,
        },
        "guardrails": guardrails,
        "invariants": invariant_checks,
    }
    result_name = (
        "flat_ground_y_hmr4d_results.pt"
        if decision == "diagnostic_pass"
        else "candidate_flat_ground_y_hmr4d_results.pt"
    )
    torch.save(enhanced, args.output_dir / result_name)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_curve(
        args.output_dir,
        root_before[:, 1],
        root_after[:, 1],
        correction,
        observations,
        observation_mask,
        fps,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
