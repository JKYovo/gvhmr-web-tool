#!/usr/bin/env python3
"""Constrain FootMR root-Y to one shared floor using contacting sole minima."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from apply_flat_ground_y import invariants
from apply_ground_xyz import smooth_observations, trajectory_metrics
from apply_p2y import foot_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-observations", type=int, default=30)
    parser.add_argument("--smoothing-seconds", type=float, default=0.5)
    parser.add_argument("--max-y-correction", type=float, default=0.25)
    parser.add_argument(
        "--allow-large-correction",
        action="store_true",
        help="Record but do not enforce the fixed max-Y threshold.",
    )
    parser.add_argument("--max-contact-speed-ratio", type=float, default=1.25)
    parser.add_argument("--max-median-residual-cm", type=float, default=1.0)
    parser.add_argument("--max-p95-residual-cm", type=float, default=5.0)
    return parser.parse_args()


def foot_contact_mask(contact: np.ndarray, threshold: float) -> np.ndarray:
    """Reduce FootMR's four static probabilities to left/right contact masks."""
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


def sole_heights(feet_y: np.ndarray) -> np.ndarray:
    """Return the lowest of three sole landmarks for each foot."""
    feet_y = np.asarray(feet_y)
    if feet_y.ndim != 2 or feet_y.shape[1] != 6:
        raise ValueError(f"Expected six sole landmarks, got {feet_y.shape}")
    return np.stack(
        [np.min(feet_y[:, :3], axis=1), np.min(feet_y[:, 3:], axis=1)],
        axis=1,
    )


def calibrate_floor_height(
    sole_y: np.ndarray,
    contact_mask: np.ndarray,
    *,
    fps: float,
    calibration_seconds: float,
    minimum_observations: int,
) -> tuple[float, np.ndarray]:
    """Estimate one shared floor from early high-confidence support samples."""
    if sole_y.shape != contact_mask.shape or sole_y.ndim != 2:
        raise ValueError(f"sole/contact shape mismatch: {sole_y.shape}, {contact_mask.shape}")
    search_end = min(len(sole_y), max(1, int(round(calibration_seconds * fps))))
    early_values = sole_y[:search_end][contact_mask[:search_end]]
    frame_mask = np.any(contact_mask, axis=1)
    calibration_frames = np.flatnonzero(frame_mask[:search_end])
    if len(early_values) < minimum_observations:
        all_values = sole_y[contact_mask]
        if len(all_values) < minimum_observations:
            raise RuntimeError(
                f"Too few contacting sole samples for floor calibration: {len(all_values)}"
            )
        early_values = all_values[:minimum_observations]
        calibration_frames = np.flatnonzero(frame_mask)[:minimum_observations]
    return float(np.median(early_values)), calibration_frames


def build_floor_observations(
    sole_y: np.ndarray,
    contact_mask: np.ndarray,
    floor_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor the lowest currently supporting sole to the shared floor."""
    if sole_y.shape != contact_mask.shape:
        raise ValueError(f"sole/contact shape mismatch: {sole_y.shape}, {contact_mask.shape}")
    frame_mask = np.any(contact_mask, axis=1)
    observations = np.full((len(sole_y), 1), np.nan, dtype=np.float64)
    for frame in np.flatnonzero(frame_mask):
        observations[frame, 0] = floor_y - float(
            np.min(sole_y[frame, contact_mask[frame]])
        )
    return observations, frame_mask


def sole_residual_metrics(
    sole_y: np.ndarray,
    contact_mask: np.ndarray,
    floor_y: float,
    correction: np.ndarray,
) -> dict[str, float]:
    residual = np.abs(sole_y + correction[:, None] - floor_y)[contact_mask]
    if not len(residual):
        raise RuntimeError("No contacting sole samples for residual metrics")
    return {
        "median_cm": float(np.median(residual) * 100.0),
        "p95_cm": float(np.percentile(residual, 95) * 100.0),
    }


def guardrail_decision(
    guardrails: dict[str, bool], *, allow_large_correction: bool
) -> tuple[str, list[str]]:
    """Evaluate safety/effectiveness checks with an optional amplitude waiver."""
    enforced = [
        key
        for key in guardrails
        if key != "max_abs_y_pass" or not allow_large_correction
    ]
    failed = [key for key in enforced if not guardrails[key]]
    return ("diagnostic_pass" if not failed else "guardrail_failed"), enforced


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
    axes[0].plot(time, root_after, color="tab:orange", label="contact-floor root Y")
    axes[0].set_ylabel("root Y (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(time, correction, color="tab:orange", label="smoothed correction")
    axes[1].scatter(
        time[observation_mask],
        observations[observation_mask, 0],
        s=3,
        alpha=0.15,
        label="lowest support sole observation",
    )
    axes[1].set_ylabel("Y correction (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "contact_floor_y_curve.png", dpi=160)
    plt.close(figure)


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

    feet = foot_landmarks(source, SmplxLiteV437Coco23().eval())
    sole_y = sole_heights(feet[..., 1])
    contact = source["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().numpy()
    contact_feet = foot_contact_mask(contact, args.contact_threshold)
    floor_y, calibration_frames = calibrate_floor_height(
        sole_y,
        contact_feet,
        fps=fps,
        calibration_seconds=args.calibration_seconds,
        minimum_observations=args.minimum_observations,
    )
    observations, observation_mask = build_floor_observations(
        sole_y, contact_feet, floor_y
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

    marker_contact = np.repeat(contact_feet, 3, axis=1)
    baseline_trajectory = trajectory_metrics(root_before, feet, marker_contact, fps)
    enhanced_trajectory = trajectory_metrics(
        root_after, enhanced_feet, marker_contact, fps
    )
    baseline_residual = sole_residual_metrics(
        sole_y, contact_feet, floor_y, np.zeros(length, dtype=np.float64)
    )
    enhanced_residual = sole_residual_metrics(
        sole_y, contact_feet, floor_y, correction
    )
    invariant_checks = invariants(source, enhanced)
    if not all(invariant_checks.values()):
        raise RuntimeError(f"Contact-floor invariant failed: {invariant_checks}")

    max_y = float(np.max(np.abs(correction)))
    step_limit = max(3.0, baseline_trajectory["root_step_max_cm_per_frame"] * 1.5)
    acceleration_limit = baseline_trajectory["root_accel_p95_m_per_s2"] * 1.25
    contact_speed_limit = (
        baseline_trajectory["contact_foot_speed_p95_mm_per_frame"]
        * args.max_contact_speed_ratio
    )
    guardrails = {
        "max_abs_y_pass": max_y <= args.max_y_correction,
        "root_step_pass": enhanced_trajectory["root_step_max_cm_per_frame"] <= step_limit,
        "root_accel_pass": enhanced_trajectory["root_accel_p95_m_per_s2"]
        <= acceleration_limit,
        "contact_speed_pass": enhanced_trajectory[
            "contact_foot_speed_p95_mm_per_frame"
        ]
        <= contact_speed_limit,
        "height_median_improved": enhanced_residual["median_cm"]
        < baseline_residual["median_cm"],
        "height_p95_improved": enhanced_residual["p95_cm"]
        < baseline_residual["p95_cm"],
        "height_median_effective": enhanced_residual["median_cm"]
        <= args.max_median_residual_cm,
        "height_p95_effective": enhanced_residual["p95_cm"]
        <= args.max_p95_residual_cm,
    }
    decision, enforced_guardrails = guardrail_decision(
        guardrails, allow_large_correction=args.allow_large_correction
    )
    payload = {
        "method": "FootMR shared-floor lowest-support-sole Y",
        "decision": decision,
        "frames": length,
        "fps": fps,
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "video": str(args.video.resolve()),
        "contact_threshold": args.contact_threshold,
        "contact_frames": int(observation_mask.sum()),
        "contact_foot_samples": int(contact_feet.sum()),
        "floor_y_m": floor_y,
        "calibration_frames": calibration_frames.tolist(),
        "smoothing": smoothing,
        "correction": {
            "start_y_m": float(correction[0]),
            "end_y_m": float(correction[-1]),
            "max_abs_y_m": max_y,
        },
        "support_sole_residual_cm": {
            "baseline": baseline_residual,
            "contact_floor_y": enhanced_residual,
        },
        "baseline": baseline_trajectory,
        "contact_floor_y": enhanced_trajectory,
        "limits": {
            "max_abs_y_m": args.max_y_correction,
            "max_abs_y_enforced": not args.allow_large_correction,
            "root_step_cm_per_frame": step_limit,
            "root_accel_m_per_s2": acceleration_limit,
            "contact_speed_p95_mm_per_frame": contact_speed_limit,
            "median_residual_cm": args.max_median_residual_cm,
            "p95_residual_cm": args.max_p95_residual_cm,
        },
        "guardrails": guardrails,
        "enforced_guardrails": enforced_guardrails,
        "invariants": invariant_checks,
    }
    result_name = (
        "contact_floor_y_hmr4d_results.pt"
        if decision == "diagnostic_pass"
        else "candidate_contact_floor_y_hmr4d_results.pt"
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
