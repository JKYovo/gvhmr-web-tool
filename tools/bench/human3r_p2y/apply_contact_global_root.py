#!/usr/bin/env python3
"""Globally optimize a GVHMR root trajectory from FootMR contacts and a flat floor.

V1.1 deliberately uses no depth or scene reconstruction.  It jointly solves the
root correction for every frame.  Vertical contact rows constrain the soles to
one early-calibrated floor; marker-aware horizontal rows independently anchor
toe and heel proxies with continuous confidence.  Data and FPS-normalized
temporal terms keep the solution close to the original GVHMR trajectory and
smooth through airborne gaps.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lsqr

try:
    from .apply_p2y import foot_landmarks, tree_equal
except ImportError:  # Direct script execution adds this directory to sys.path.
    from apply_p2y import foot_landmarks, tree_equal


@dataclass(frozen=True)
class ContactSegment:
    foot: int
    start: int
    end: int
    marker: int


CONTACT_POINT_NAMES = ("toe", "heel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-enter-threshold", type=float, default=0.85)
    parser.add_argument("--contact-exit-threshold", type=float, default=0.65)
    parser.add_argument("--relative-height-margin", type=float, default=0.02)
    parser.add_argument(
        "--max-contact-speed",
        type=float,
        default=0.15,
        help="m/s; matches the static-contact velocity target used by GVHMR training",
    )
    parser.add_argument("--minimum-contact-frames", type=int, default=5)
    parser.add_argument("--maximum-contact-gap", type=int, default=2)
    parser.add_argument("--minimum-anchor-weight", type=float, default=0.01)
    parser.add_argument("--segment-fade-seconds", type=float, default=0.10)
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-floor-samples", type=int, default=20)
    parser.add_argument("--data-weight", type=float, default=0.5)
    parser.add_argument("--velocity-weight", type=float, default=8.0)
    parser.add_argument("--acceleration-weight", type=float, default=480.0)
    parser.add_argument("--height-contact-weight", type=float, default=30720.0)
    parser.add_argument("--slip-contact-weight", type=float, default=960.0)
    parser.add_argument("--max-horizontal-correction", type=float, default=0.35)
    parser.add_argument("--max-vertical-correction", type=float, default=0.30)
    return parser.parse_args()


def foot_confidence(contact: np.ndarray) -> np.ndarray:
    """Reduce ankle/foot logits to one continuous confidence per foot."""
    contact = np.asarray(contact, dtype=np.float64)
    if contact.ndim != 2 or contact.shape[1] < 4:
        raise ValueError(f"Expected contact probabilities (T, >=4), got {contact.shape}")
    left = contact[:, :2]
    right = contact[:, 2:4]
    # A single strong endpoint can establish contact, while the weaker endpoint
    # still contributes so isolated overconfident logits are less dominant.
    return np.stack(
        [0.7 * left.max(axis=1) + 0.3 * left.min(axis=1),
         0.7 * right.max(axis=1) + 0.3 * right.min(axis=1)],
        axis=1,
    )


def sole_points(feet: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feet = np.asarray(feet, dtype=np.float64)
    if feet.ndim != 3 or feet.shape[1:] != (6, 3):
        raise ValueError(f"Expected sole landmarks (T, 6, 3), got {feet.shape}")
    grouped = np.stack((feet[:, :3], feet[:, 3:]), axis=1)
    heights = grouped[..., 1].min(axis=2)
    centroids = grouped.mean(axis=2)
    lowest_marker = grouped[..., 1].argmin(axis=2)
    return grouped, heights, centroids


def contact_anchor_points(grouped_feet: np.ndarray) -> np.ndarray:
    """Return one forefoot-center and one heel point for each foot."""
    grouped_feet = np.asarray(grouped_feet, dtype=np.float64)
    if grouped_feet.ndim != 4 or grouped_feet.shape[1:] != (2, 3, 3):
        raise ValueError(
            f"Expected grouped feet (T, 2, 3, 3), got {grouped_feet.shape}"
        )
    toe = grouped_feet[:, :, :2].mean(axis=2)
    heel = grouped_feet[:, :, 2]
    return np.stack((toe, heel), axis=2)


def contact_point_probabilities(contact: np.ndarray) -> np.ndarray:
    """Map GVHMR ankle/foot static probabilities to toe/heel proxies.

    The network does not predict an explicit heel label.  Its SMPL ``foot``
    joint is used for the forefoot, while ``ankle`` is only a heel proxy.  An
    ankle-only response is therefore deliberately weak; coincident foot
    confidence raises the heel proxy for full-foot support.
    """
    contact = np.asarray(contact, dtype=np.float64)
    if contact.ndim != 2 or contact.shape[1] < 4:
        raise ValueError(f"Expected contact probabilities (T, >=4), got {contact.shape}")
    output = np.zeros((len(contact), 2, 2), dtype=np.float64)
    for foot, (ankle_index, foot_index) in enumerate(((0, 1), (2, 3))):
        ankle_probability = contact[:, ankle_index]
        toe_probability = contact[:, foot_index]
        heel_proxy = ankle_probability * (0.35 + 0.65 * toe_probability)
        output[:, foot, 0] = toe_probability
        output[:, foot, 1] = heel_proxy
    return output


def hysteresis_mask(probability: np.ndarray, enter: float, exit: float) -> np.ndarray:
    if not (0.0 <= exit <= enter <= 1.0):
        raise ValueError(f"Invalid hysteresis thresholds: enter={enter}, exit={exit}")
    output = np.zeros_like(probability, dtype=bool)
    for foot in range(probability.shape[1]):
        active = False
        for frame in range(len(probability)):
            if active:
                active = bool(probability[frame, foot] >= exit)
            else:
                active = bool(probability[frame, foot] >= enter)
            output[frame, foot] = active
    return output


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def close_short_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    if maximum_gap <= 0:
        return output
    inverse_runs = true_runs(~output)
    for start, end in inverse_runs:
        if start > 0 and end < len(output) and end - start <= maximum_gap:
            output[start:end] = True
    return output


def remove_short_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    for start, end in true_runs(output):
        if end - start < minimum_frames:
            output[start:end] = False
    return output


def refine_contacts(
    confidence: np.ndarray,
    heights: np.ndarray,
    centroids: np.ndarray,
    *,
    fps: float,
    enter_threshold: float,
    exit_threshold: float,
    relative_height_margin: float,
    max_contact_speed: float,
    minimum_frames: int,
    maximum_gap: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if confidence.shape != heights.shape or confidence.shape != centroids.shape[:2]:
        raise ValueError("Contact confidence/sole shape mismatch")
    initial = hysteresis_mask(confidence, enter_threshold, exit_threshold)

    # A supporting foot should be close to the lower of the currently plausible
    # soles.  This rejects a high-confidence lifted foot without assuming the
    # absolute floor height, which may itself drift in GVHMR.
    plausible_height = np.full(len(heights), np.nan, dtype=np.float64)
    for frame in range(len(heights)):
        active = initial[frame]
        if np.any(active):
            plausible_height[frame] = float(np.min(heights[frame, active]))
        else:
            plausible_height[frame] = float(np.min(heights[frame]))
    height_gate = heights <= plausible_height[:, None] + relative_height_margin

    speed = np.zeros_like(confidence, dtype=np.float64)
    speed[1:] = np.linalg.norm(np.diff(centroids[..., (0, 2)], axis=0), axis=-1) * fps
    if len(speed) > 1:
        speed[:-1] = np.maximum(speed[:-1], speed[1:])
    speed_gate = speed <= max_contact_speed
    refined = initial & height_gate & speed_gate
    for foot in range(2):
        refined[:, foot] = close_short_gaps(refined[:, foot], maximum_gap)
        refined[:, foot] &= height_gate[:, foot] & speed_gate[:, foot]
        refined[:, foot] = remove_short_runs(refined[:, foot], minimum_frames)

    return refined, {
        "hysteresis_samples": int(initial.sum()),
        "height_rejected_samples": int((initial & ~height_gate).sum()),
        "speed_rejected_samples": int((initial & height_gate & ~speed_gate).sum()),
        "refined_samples": int(refined.sum()),
        "refined_frames": int(np.any(refined, axis=1).sum()),
        "sole_speed_p95_m_per_s": float(np.percentile(speed[refined], 95)) if refined.any() else None,
    }


def continuous_anchor_weights(
    point_probability: np.ndarray,
    anchor_points: np.ndarray,
    contact_mask: np.ndarray,
    *,
    fps: float,
    relative_height_margin: float,
    max_contact_speed: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build soft toe/heel weights from probability, height and velocity."""
    if point_probability.shape != anchor_points.shape[:3]:
        raise ValueError("Point probability/anchor shape mismatch")
    if contact_mask.shape != point_probability.shape[:2]:
        raise ValueError("Foot contact/point probability shape mismatch")
    if fps <= 0 or relative_height_margin <= 0 or max_contact_speed <= 0:
        raise ValueError("FPS, height margin and speed limit must be positive")

    point_height = anchor_points[..., 1]
    plausible_height = np.full(len(anchor_points), np.nan, dtype=np.float64)
    for frame in range(len(anchor_points)):
        active_feet = contact_mask[frame]
        candidates = point_height[frame, active_feet]
        if not candidates.size:
            candidates = point_height[frame]
        plausible_height[frame] = float(np.min(candidates))
    height_excess = np.maximum(point_height - plausible_height[:, None, None], 0.0)
    height_confidence = np.clip(
        1.0 - height_excess / relative_height_margin, 0.0, 1.0
    )
    height_confidence *= height_confidence

    speed = np.zeros_like(point_probability, dtype=np.float64)
    speed[1:] = (
        np.linalg.norm(np.diff(anchor_points[..., (0, 2)], axis=0), axis=-1) * fps
    )
    if len(speed) > 1:
        speed[:-1] = np.maximum(speed[:-1], speed[1:])
    velocity_confidence = np.clip(1.0 - speed / max_contact_speed, 0.0, 1.0)
    velocity_confidence *= velocity_confidence

    probability_confidence = np.clip(point_probability, 0.0, 1.0)
    weights = (
        probability_confidence
        * height_confidence
        * velocity_confidence
        * contact_mask[:, :, None]
    )
    return weights, {
        "point_weight_nonzero_samples": int(np.count_nonzero(weights)),
        "point_weight_median": float(np.median(weights[weights > 0]))
        if np.any(weights > 0)
        else None,
        "point_weight_p95": float(np.percentile(weights[weights > 0], 95))
        if np.any(weights > 0)
        else None,
        "point_speed_p95_m_per_s": float(np.percentile(speed[weights > 0], 95))
        if np.any(weights > 0)
        else None,
    }


def build_segments(
    contact_mask: np.ndarray,
    point_weights: np.ndarray,
    *,
    minimum_anchor_weight: float = 0.01,
) -> list[ContactSegment]:
    """Build independent toe/heel anchor segments from soft support weights."""
    if point_weights.shape != (*contact_mask.shape, 2):
        raise ValueError("Contact mask/point weight shape mismatch")
    segments = []
    for foot in range(2):
        for marker in range(2):
            active = contact_mask[:, foot] & (
                point_weights[:, foot, marker] >= minimum_anchor_weight
            )
            for start, end in true_runs(active):
                if end - start >= 2:
                    segments.append(ContactSegment(foot, start, end, marker))
    return segments


def apply_segment_confidence(
    point_weights: np.ndarray,
    segments: list[ContactSegment],
    fade_frames: int,
) -> np.ndarray:
    """Fade anchors at landing/liftoff while retaining continuous weights."""
    if fade_frames < 1:
        raise ValueError("fade_frames must be positive")
    output = np.zeros_like(point_weights, dtype=np.float64)
    for segment in segments:
        length = segment.end - segment.start
        index = np.arange(length)
        ramp = np.minimum.reduce(
            (
                (index + 1) / fade_frames,
                (length - index) / fade_frames,
                np.ones(length, dtype=np.float64),
            )
        )
        output[segment.start : segment.end, segment.foot, segment.marker] = (
            point_weights[segment.start : segment.end, segment.foot, segment.marker]
            * ramp
        )
    return output


def calibrate_floor(
    heights: np.ndarray,
    contact_mask: np.ndarray,
    *,
    fps: float,
    calibration_seconds: float,
    minimum_samples: int,
) -> tuple[float, np.ndarray]:
    search_end = min(len(heights), max(1, int(round(calibration_seconds * fps))))
    early_mask = contact_mask[:search_end]
    samples = heights[:search_end][early_mask]
    frames = np.flatnonzero(np.any(early_mask, axis=1))
    if len(samples) < minimum_samples:
        all_indices = np.argwhere(contact_mask)
        if len(all_indices) < minimum_samples:
            raise RuntimeError(f"Too few reliable floor contacts: {len(all_indices)}")
        selected = all_indices[:minimum_samples]
        samples = heights[selected[:, 0], selected[:, 1]]
        frames = np.unique(selected[:, 0])
    return float(np.median(samples)), frames.astype(np.int64)


def add_temporal_rows(
    rows: list[int],
    columns: list[int],
    values: list[float],
    targets: list[float],
    length: int,
    *,
    fps: float,
    data_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    row = 0
    data_scale = np.sqrt(data_weight)
    for frame in range(length):
        rows.append(row)
        columns.append(frame)
        values.append(data_scale)
        targets.append(0.0)
        row += 1
    # CLI weights retain their 30-FPS numerical meaning, while residuals are
    # evaluated as physical derivatives.  At 60 FPS the velocity coefficient
    # is therefore 2x and the acceleration coefficient 4x.
    reference_dt = 1.0 / 30.0
    dt = 1.0 / fps
    velocity_scale = np.sqrt(velocity_weight) * reference_dt / dt
    for frame in range(1, length):
        rows.extend((row, row))
        columns.extend((frame - 1, frame))
        values.extend((-velocity_scale, velocity_scale))
        targets.append(0.0)
        row += 1
    acceleration_scale = np.sqrt(acceleration_weight) * (reference_dt / dt) ** 2
    for frame in range(1, length - 1):
        rows.extend((row, row, row))
        columns.extend((frame - 1, frame, frame + 1))
        values.extend((acceleration_scale, -2.0 * acceleration_scale, acceleration_scale))
        targets.append(0.0)
        row += 1
    return row


def solve_vertical(
    heights: np.ndarray,
    contact_mask: np.ndarray,
    weights: np.ndarray,
    floor_y: float,
    *,
    fps: float,
    data_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
    contact_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    length = len(heights)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    targets: list[float] = []
    row = add_temporal_rows(
        rows,
        columns,
        values,
        targets,
        length,
        fps=fps,
        data_weight=data_weight,
        velocity_weight=velocity_weight,
        acceleration_weight=acceleration_weight,
    )
    # Root translation cannot resolve pose-caused disagreement between two feet.
    # Anchor the lowest reliable support sole once per frame, rather than
    # averaging two incompatible heights and pushing one foot through the floor.
    for frame in np.flatnonzero(np.any(contact_mask, axis=1)):
        active_feet = np.flatnonzero(contact_mask[frame])
        foot = int(active_feet[np.argmin(heights[frame, active_feet])])
        scale = np.sqrt(contact_weight * weights[frame, foot])
        rows.append(row)
        columns.append(int(frame))
        values.append(float(scale))
        targets.append(float(scale * (floor_y - heights[frame, foot])))
        row += 1
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(row, length)).tocsr()
    solution = lsqr(matrix, np.asarray(targets), atol=1.0e-9, btol=1.0e-9, iter_lim=4000)
    return solution[0], {
        "rows": row,
        "iterations": int(solution[2]),
        "stop_code": int(solution[1]),
        "residual_norm": float(solution[3]),
    }


def solve_horizontal_axis(
    anchor_points: np.ndarray,
    segments: list[ContactSegment],
    point_weights: np.ndarray,
    axis: int,
    *,
    fps: float,
    data_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
    contact_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    length = len(anchor_points)
    variables = length + len(segments)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    targets: list[float] = []
    row = add_temporal_rows(
        rows,
        columns,
        values,
        targets,
        length,
        fps=fps,
        data_weight=data_weight,
        velocity_weight=velocity_weight,
        acceleration_weight=acceleration_weight,
    )
    for segment_index, segment in enumerate(segments):
        anchor_column = length + segment_index
        for frame in range(segment.start, segment.end):
            weight = point_weights[frame, segment.foot, segment.marker]
            scale = np.sqrt(contact_weight * weight)
            point = anchor_points[frame, segment.foot, segment.marker, axis]
            rows.extend((row, row))
            columns.extend((frame, anchor_column))
            values.extend((float(scale), float(-scale)))
            targets.append(float(-scale * point))
            row += 1
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(row, variables)).tocsr()
    solution = lsqr(matrix, np.asarray(targets), atol=1.0e-9, btol=1.0e-9, iter_lim=6000)
    return solution[0][:length], solution[0][length:], {
        "rows": row,
        "variables": variables,
        "iterations": int(solution[2]),
        "stop_code": int(solution[1]),
        "residual_norm": float(solution[3]),
    }


def optimize_root(
    root: np.ndarray,
    anchor_points: np.ndarray,
    heights: np.ndarray,
    contact_mask: np.ndarray,
    point_weights: np.ndarray,
    segments: list[ContactSegment],
    floor_y: float,
    *,
    fps: float,
    data_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
    height_contact_weight: float,
    slip_contact_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    support_weights = point_weights.max(axis=2)
    vertical_mask = contact_mask & (support_weights > 0)
    correction = np.zeros_like(root, dtype=np.float64)
    correction[:, 1], y_solver = solve_vertical(
        heights,
        vertical_mask,
        support_weights,
        floor_y,
        fps=fps,
        data_weight=data_weight,
        velocity_weight=velocity_weight,
        acceleration_weight=acceleration_weight,
        contact_weight=height_contact_weight,
    )
    anchors = {}
    solvers = {"y": y_solver}
    for axis, label in ((0, "x"), (2, "z")):
        correction[:, axis], axis_anchors, solver = solve_horizontal_axis(
            anchor_points,
            segments,
            point_weights,
            axis,
            fps=fps,
            data_weight=data_weight,
            velocity_weight=velocity_weight,
            acceleration_weight=acceleration_weight,
            contact_weight=slip_contact_weight,
        )
        anchors[label] = axis_anchors.tolist()
        solvers[label] = solver
    return correction, {"anchors": anchors, "solvers": solvers}


def summarize(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "median": float(np.median(values) * scale),
        "p95": float(np.percentile(values, 95) * scale),
        "max": float(np.max(values) * scale),
    }


def contact_metrics(
    root: np.ndarray,
    anchor_points: np.ndarray,
    heights: np.ndarray,
    contact_mask: np.ndarray,
    segments: list[ContactSegment],
    floor_y: float,
    fps: float,
) -> dict[str, Any]:
    all_contact_residual = (heights - floor_y)[contact_mask]
    support_residual = np.asarray(
        [
            np.min(heights[frame, contact_mask[frame]]) - floor_y
            for frame in np.flatnonzero(np.any(contact_mask, axis=1))
        ],
        dtype=np.float64,
    )
    anchor_steps = []
    segment_drifts = []
    segment_scatter = []
    for segment in segments:
        points = anchor_points[
            segment.start : segment.end,
            segment.foot,
            segment.marker,
        ][:, (0, 2)]
        if len(points) > 1:
            anchor_steps.extend(np.linalg.norm(np.diff(points, axis=0), axis=1).tolist())
            segment_drifts.append(float(np.linalg.norm(points[-1] - points[0])))
        center = np.median(points, axis=0)
        segment_scatter.extend(np.linalg.norm(points - center, axis=1).tolist())
    root_step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    root_acceleration = np.linalg.norm(np.diff(root, n=2, axis=0), axis=1) * fps * fps
    return {
        "support_height_signed_cm": {
            "median": float(np.median(support_residual) * 100.0),
            "p05": float(np.percentile(support_residual, 5) * 100.0),
            "p95": float(np.percentile(support_residual, 95) * 100.0),
        },
        "support_height_abs_cm": summarize(np.abs(support_residual), 100.0),
        "all_labeled_contact_height_abs_cm": summarize(
            np.abs(all_contact_residual), 100.0
        ),
        "hover_over_3cm_pct": float(np.mean(support_residual > 0.03) * 100.0),
        "penetration_over_1cm_pct": float(np.mean(support_residual < -0.01) * 100.0),
        "contact_anchor_speed_mm_per_frame": summarize(np.asarray(anchor_steps), 1000.0),
        "contact_segment_endpoint_drift_cm": summarize(np.asarray(segment_drifts), 100.0),
        "contact_anchor_scatter_cm": summarize(np.asarray(segment_scatter), 100.0),
        "root_step_cm_per_frame": summarize(root_step, 100.0),
        "root_acceleration_m_per_s2": summarize(root_acceleration, 1.0),
    }


def invariants(source: Mapping[str, Any], enhanced: Mapping[str, Any]) -> dict[str, bool]:
    target_translation = enhanced["smpl_params_global"]["transl"]
    return {
        "top_level_keys_equal": source.keys() == enhanced.keys(),
        "global_param_keys_equal": source["smpl_params_global"].keys()
        == enhanced["smpl_params_global"].keys(),
        "incam_param_keys_equal": source["smpl_params_incam"].keys()
        == enhanced["smpl_params_incam"].keys(),
        "body_pose_equal": bool(torch.equal(source["smpl_params_global"]["body_pose"], enhanced["smpl_params_global"]["body_pose"])),
        "global_orient_equal": bool(torch.equal(source["smpl_params_global"]["global_orient"], enhanced["smpl_params_global"]["global_orient"])),
        "betas_equal": bool(torch.equal(source["smpl_params_global"]["betas"], enhanced["smpl_params_global"]["betas"])),
        "incam_equal": tree_equal(source["smpl_params_incam"], enhanced["smpl_params_incam"]),
        "K_fullimg_equal": tree_equal(source["K_fullimg"], enhanced["K_fullimg"]),
        "net_outputs_equal": tree_equal(source["net_outputs"], enhanced["net_outputs"]),
        "finite_root_xyz": bool(torch.isfinite(target_translation).all()),
    }


def save_curves(
    output_dir: Path,
    root_before: np.ndarray,
    root_after: np.ndarray,
    correction: np.ndarray,
    heights_before: np.ndarray,
    heights_after: np.ndarray,
    contact_mask: np.ndarray,
    floor_y: float,
    fps: float,
) -> None:
    time = np.arange(len(root_before)) / fps
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for axis, label in enumerate(("X", "Y", "Z")):
        axes[0].plot(time, root_before[:, axis], linewidth=0.8, label=f"baseline {label}")
        axes[0].plot(time, root_after[:, axis], linewidth=0.8, linestyle="--", label=f"global {label}")
    axes[0].set_ylabel("root (m)")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)
    for axis, label in enumerate(("X", "Y", "Z")):
        axes[1].plot(time, correction[:, axis], linewidth=0.9, label=label)
    axes[1].set_ylabel("root correction (m)")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.25)
    for foot, label in enumerate(("left", "right")):
        axes[2].plot(time, heights_before[:, foot], linewidth=0.7, label=f"baseline {label}")
        axes[2].plot(time, heights_after[:, foot], linewidth=0.8, linestyle="--", label=f"global {label}")
    axes[2].axhline(floor_y, color="black", linewidth=0.8, label="fixed floor")
    axes[2].set_ylabel("sole min Y (m)")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.25)
    for foot, label in enumerate(("left contact", "right contact")):
        axes[3].fill_between(time, 0, 1, where=contact_mask[:, foot], alpha=0.30, label=label)
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("reliable contact")
    axes[3].set_xlabel("time (s)")
    axes[3].legend()
    axes[3].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "contact_global_root_curves.png", dpi=160)
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
    grouped, heights, centroids = sole_points(feet)
    anchor_points = contact_anchor_points(grouped)
    contact_probability = source["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().numpy()
    confidence = foot_confidence(contact_probability)
    contact_mask, contact_diagnostics = refine_contacts(
        confidence,
        heights,
        centroids,
        fps=fps,
        enter_threshold=args.contact_enter_threshold,
        exit_threshold=args.contact_exit_threshold,
        relative_height_margin=args.relative_height_margin,
        max_contact_speed=args.max_contact_speed,
        minimum_frames=args.minimum_contact_frames,
        maximum_gap=args.maximum_contact_gap,
    )
    raw_point_weights, point_weight_diagnostics = continuous_anchor_weights(
        contact_point_probabilities(contact_probability),
        anchor_points,
        contact_mask,
        fps=fps,
        relative_height_margin=args.relative_height_margin,
        max_contact_speed=args.max_contact_speed,
    )
    segments = build_segments(
        contact_mask,
        raw_point_weights,
        minimum_anchor_weight=args.minimum_anchor_weight,
    )
    if not segments:
        raise RuntimeError("No reliable contact segments after refinement")
    segment_fade_frames = max(1, int(round(args.segment_fade_seconds * fps)))
    point_weights = apply_segment_confidence(
        raw_point_weights, segments, segment_fade_frames
    )
    floor_y, calibration_frames = calibrate_floor(
        heights,
        contact_mask,
        fps=fps,
        calibration_seconds=args.calibration_seconds,
        minimum_samples=args.minimum_floor_samples,
    )
    root_before = source["smpl_params_global"]["transl"].numpy().astype(np.float64)
    correction, solver_diagnostics = optimize_root(
        root_before,
        anchor_points,
        heights,
        contact_mask,
        point_weights,
        segments,
        floor_y,
        fps=fps,
        data_weight=args.data_weight,
        velocity_weight=args.velocity_weight,
        acceleration_weight=args.acceleration_weight,
        height_contact_weight=args.height_contact_weight,
        slip_contact_weight=args.slip_contact_weight,
    )
    root_after = root_before + correction
    anchor_points_after = anchor_points + correction[:, None, None, :]
    heights_after = heights + correction[:, None, 1]

    enhanced = copy.deepcopy(source)
    enhanced["smpl_params_global"]["transl"] = torch.as_tensor(
        root_after,
        dtype=source["smpl_params_global"]["transl"].dtype,
    )
    invariant_checks = invariants(source, enhanced)
    if not all(invariant_checks.values()):
        raise RuntimeError(f"Global contact optimizer invariant failed: {invariant_checks}")

    baseline_metrics = contact_metrics(
        root_before, anchor_points, heights, contact_mask, segments, floor_y, fps
    )
    optimized_metrics = contact_metrics(
        root_after,
        anchor_points_after,
        heights_after,
        contact_mask,
        segments,
        floor_y,
        fps,
    )
    max_horizontal = float(np.max(np.linalg.norm(correction[:, (0, 2)], axis=1)))
    max_vertical = float(np.max(np.abs(correction[:, 1])))
    slip_improved = (
        optimized_metrics["contact_anchor_speed_mm_per_frame"]["p95"]
        < baseline_metrics["contact_anchor_speed_mm_per_frame"]["p95"]
        and optimized_metrics["contact_segment_endpoint_drift_cm"]["p95"]
        < baseline_metrics["contact_segment_endpoint_drift_cm"]["p95"]
    )
    height_improved = (
        optimized_metrics["support_height_abs_cm"]["p95"]
        < baseline_metrics["support_height_abs_cm"]["p95"]
        and optimized_metrics["hover_over_3cm_pct"]
        <= baseline_metrics["hover_over_3cm_pct"]
        and optimized_metrics["penetration_over_1cm_pct"]
        <= baseline_metrics["penetration_over_1cm_pct"]
    )
    root_step_limit = max(
        3.0, 1.5 * baseline_metrics["root_step_cm_per_frame"]["max"]
    )
    root_acceleration_limit = (
        1.25 * baseline_metrics["root_acceleration_m_per_s2"]["p95"]
    )
    guardrails = {
        "horizontal_correction_pass": max_horizontal <= args.max_horizontal_correction,
        "vertical_correction_pass": max_vertical <= args.max_vertical_correction,
        "root_step_pass": optimized_metrics["root_step_cm_per_frame"]["max"]
        <= root_step_limit,
        "root_acceleration_pass": optimized_metrics["root_acceleration_m_per_s2"]["p95"]
        <= root_acceleration_limit,
        "slip_improved": slip_improved,
        "height_improved": height_improved,
    }
    guardrail_details = {
        "horizontal_correction_pass": {
            "actual_m": max_horizontal,
            "limit_m": args.max_horizontal_correction,
        },
        "vertical_correction_pass": {
            "actual_m": max_vertical,
            "limit_m": args.max_vertical_correction,
        },
        "root_step_pass": {
            "actual_cm_per_frame": optimized_metrics["root_step_cm_per_frame"]["max"],
            "limit_cm_per_frame": root_step_limit,
        },
        "root_acceleration_pass": {
            "actual_m_per_s2_p95": optimized_metrics["root_acceleration_m_per_s2"]["p95"],
            "limit_m_per_s2_p95": root_acceleration_limit,
        },
        "slip_improved": {
            "anchor_speed_before_mm_per_frame_p95": baseline_metrics["contact_anchor_speed_mm_per_frame"]["p95"],
            "anchor_speed_after_mm_per_frame_p95": optimized_metrics["contact_anchor_speed_mm_per_frame"]["p95"],
            "endpoint_drift_before_cm_p95": baseline_metrics["contact_segment_endpoint_drift_cm"]["p95"],
            "endpoint_drift_after_cm_p95": optimized_metrics["contact_segment_endpoint_drift_cm"]["p95"],
        },
        "height_improved": {
            "support_height_before_cm_p95": baseline_metrics["support_height_abs_cm"]["p95"],
            "support_height_after_cm_p95": optimized_metrics["support_height_abs_cm"]["p95"],
            "hover_before_pct": baseline_metrics["hover_over_3cm_pct"],
            "hover_after_pct": optimized_metrics["hover_over_3cm_pct"],
            "penetration_before_pct": baseline_metrics["penetration_over_1cm_pct"],
            "penetration_after_pct": optimized_metrics["penetration_over_1cm_pct"],
        },
    }
    failed_guardrails = [key for key, passed in guardrails.items() if not passed]
    decision = "diagnostic_pass" if all(guardrails.values()) else "guardrail_failed"
    payload = {
        "method": "contact-aware global root optimizer V1.1 (marker-aware, no depth)",
        "decision": decision,
        "frames": length,
        "fps": fps,
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "video": str(args.video.resolve()),
        "fixed_floor_y_m": floor_y,
        "calibration_frames": calibration_frames.tolist(),
        "contact": {
            "enter_threshold": args.contact_enter_threshold,
            "exit_threshold": args.contact_exit_threshold,
            "relative_height_margin_m": args.relative_height_margin,
            "max_contact_speed_m_per_s": args.max_contact_speed,
            "minimum_contact_frames": args.minimum_contact_frames,
            "maximum_contact_gap_frames": args.maximum_contact_gap,
            "minimum_anchor_weight": args.minimum_anchor_weight,
            "segment_fade_seconds": args.segment_fade_seconds,
            "segment_fade_frames": segment_fade_frames,
            "segments": [
                {**segment.__dict__, "marker_name": CONTACT_POINT_NAMES[segment.marker]}
                for segment in segments
            ],
            **contact_diagnostics,
            **point_weight_diagnostics,
        },
        "weights": {
            "data": args.data_weight,
            "velocity": args.velocity_weight,
            "acceleration": args.acceleration_weight,
            "height_contact": args.height_contact_weight,
            "slip_contact": args.slip_contact_weight,
            "temporal_reference_fps": 30.0,
            "temporal_derivatives_use_actual_dt": True,
        },
        "correction": {
            "start_xyz_m": correction[0].tolist(),
            "end_xyz_m": correction[-1].tolist(),
            "max_horizontal_m": max_horizontal,
            "max_abs_vertical_m": max_vertical,
            "rms_xyz_m": np.sqrt(np.mean(correction * correction, axis=0)).tolist(),
        },
        "baseline": baseline_metrics,
        "contact_global": optimized_metrics,
        "solver": solver_diagnostics,
        "guardrails": guardrails,
        "guardrail_details": guardrail_details,
        "failed_guardrails": failed_guardrails,
        "invariants": invariant_checks,
        "limitations": [
            "No depth or image-ground observation: the absolute floor gauge is calibrated from early reliable contacts.",
            "Horizontal anchors remove within-contact sliding but cannot observe absolute displacement between disconnected contact segments.",
            "Only global transl is modified; pose-caused disagreement between two feet cannot be fully corrected by root translation.",
            "GVHMR predicts ankle/foot static confidence, not an explicit heel label; ankle confidence is used only as a down-weighted heel proxy.",
        ],
    }
    torch.save(enhanced, args.output_dir / "contact_global_root_hmr4d_results.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_curves(
        args.output_dir,
        root_before,
        root_after,
        correction,
        heights,
        heights_after,
        contact_mask,
        floor_y,
        fps,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
