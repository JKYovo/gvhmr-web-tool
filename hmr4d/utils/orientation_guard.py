"""Strict post-process for isolated root-orientation impulses.

The guard deliberately targets only implausibly large, isolated frame-to-frame
root rotations which appear in both the global and in-camera predictions.  It
does not smooth ordinary turns or any body joints/translations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


@dataclass(frozen=True)
class OrientationGuardConfig:
    global_threshold_deg: float = 80.0
    incam_threshold_deg: float = 75.0
    global_prominence_deg: float = 45.0
    incam_prominence_deg: float = 40.0
    prominence_ratio: float = 2.5
    detection_radius: int = 8
    smoothing_radius: int = 4
    comparable_ratio: float = 0.65
    comparable_global_floor_deg: float = 60.0
    comparable_incam_floor_deg: float = 55.0
    smoothing_iterations: int = 40
    smoothing_blend: float = 0.35


def rotation_step_degrees(axis_angle: torch.Tensor) -> torch.Tensor:
    """Return geodesic rotation magnitudes between consecutive frames."""
    if axis_angle.ndim != 2 or axis_angle.shape[-1] != 3:
        raise ValueError(
            "global_orient must have shape (frames, 3), got "
            f"{tuple(axis_angle.shape)}"
        )
    if len(axis_angle) < 2:
        return axis_angle.new_zeros((0,))
    rotations = axis_angle_to_matrix(axis_angle)
    relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    return torch.rad2deg(matrix_to_axis_angle(relative).norm(dim=-1))


def detect_isolated_orientation_jumps(
    global_orient: torch.Tensor,
    incam_orient: torch.Tensor,
    config: OrientationGuardConfig = OrientationGuardConfig(),
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    """Detect synchronized, isolated extreme root-rotation boundaries."""
    if global_orient.shape != incam_orient.shape:
        raise ValueError(
            "global/incam global_orient shapes differ: "
            f"{tuple(global_orient.shape)} vs {tuple(incam_orient.shape)}"
        )
    if not torch.isfinite(global_orient).all() or not torch.isfinite(incam_orient).all():
        raise ValueError("global_orient contains non-finite values")

    global_steps = rotation_step_degrees(global_orient)
    incam_steps = rotation_step_degrees(incam_orient)
    candidates = torch.nonzero(
        (global_steps >= config.global_threshold_deg)
        & (incam_steps >= config.incam_threshold_deg),
        as_tuple=False,
    ).flatten()
    detections: list[dict] = []
    frame_count = len(global_orient)

    for candidate_tensor in candidates:
        boundary = int(candidate_tensor)
        # A complete endpoint-fixed window is required. Edge impulses are left
        # unchanged rather than extrapolating a pose that was never observed.
        window_start = boundary - config.smoothing_radius
        window_end = boundary + 1 + config.smoothing_radius
        if window_start < 0 or window_end >= frame_count:
            continue

        local_start = max(0, boundary - config.detection_radius)
        local_end = min(len(global_steps), boundary + config.detection_radius + 1)
        local_indices = torch.arange(local_start, local_end, device=global_steps.device)
        neighbors = local_indices != boundary
        neighbor_global = global_steps[local_start:local_end][neighbors]
        neighbor_incam = incam_steps[local_start:local_end][neighbors]
        if not len(neighbor_global):
            continue

        global_peak = float(global_steps[boundary])
        incam_peak = float(incam_steps[boundary])
        global_median = float(neighbor_global.median())
        incam_median = float(neighbor_incam.median())
        prominent = (
            global_peak - global_median >= config.global_prominence_deg
            and incam_peak - incam_median >= config.incam_prominence_deg
            and global_peak >= config.prominence_ratio * max(global_median, 1.0)
            and incam_peak >= config.prominence_ratio * max(incam_median, 1.0)
        )
        if not prominent:
            continue

        comparable_global = max(
            config.comparable_global_floor_deg,
            global_peak * config.comparable_ratio,
        )
        comparable_incam = max(
            config.comparable_incam_floor_deg,
            incam_peak * config.comparable_ratio,
        )
        has_second_peak = bool(
            ((neighbor_global >= comparable_global) | (neighbor_incam >= comparable_incam))
            .any()
            .item()
        )
        if has_second_peak:
            continue

        detections.append(
            {
                "boundary_frame": boundary,
                "from_frame": boundary,
                "to_frame": boundary + 1,
                "window_start": window_start,
                "window_end": window_end,
                "global_step_before_deg": global_peak,
                "incam_step_before_deg": incam_peak,
                "local_global_median_deg": global_median,
                "local_incam_median_deg": incam_median,
            }
        )

    return detections, global_steps, incam_steps


def _smooth_rotation_window(
    rotations: torch.Tensor,
    start: int,
    end: int,
    *,
    iterations: int,
    blend: float,
) -> None:
    """Lie-group Laplacian smoothing with fixed endpoint rotations."""
    window = rotations[start : end + 1].clone()
    for _ in range(iterations):
        previous = window
        updated = previous.clone()
        left = previous[:-2]
        center = previous[1:-1]
        right = previous[2:]
        midpoint_delta = matrix_to_axis_angle(left.transpose(-1, -2) @ right) * 0.5
        midpoint = left @ axis_angle_to_matrix(midpoint_delta)
        toward_midpoint = matrix_to_axis_angle(center.transpose(-1, -2) @ midpoint)
        updated[1:-1] = center @ axis_angle_to_matrix(toward_midpoint * blend)
        window = updated
    rotations[start : end + 1] = window


def _replace_orientation_copies(
    result: dict,
    global_orient: torch.Tensor,
    incam_orient: torch.Tensor,
) -> None:
    result["smpl_params_global"]["global_orient"] = global_orient
    result["smpl_params_incam"]["global_orient"] = incam_orient
    net_outputs = result.get("net_outputs")
    if not isinstance(net_outputs, dict):
        return
    for name, orient in (
        ("pred_smpl_params_global", global_orient),
        ("pred_smpl_params_incam", incam_orient),
    ):
        params = net_outputs.get(name)
        if not isinstance(params, dict) or "global_orient" not in params:
            continue
        existing = params["global_orient"]
        if existing.shape == orient.shape:
            params["global_orient"] = orient.clone()
        elif existing.ndim == 3 and existing.shape[0] == 1 and existing.shape[1:] == orient.shape:
            params["global_orient"] = orient.unsqueeze(0).clone()
        else:
            raise ValueError(
                f"Unexpected {name}.global_orient shape: {tuple(existing.shape)}"
            )


def guard_isolated_orientation_jumps(
    result: dict,
    config: OrientationGuardConfig = OrientationGuardConfig(),
) -> tuple[dict, dict]:
    """Return a guarded result and JSON-serializable diagnostics."""
    global_orient = result["smpl_params_global"]["global_orient"]
    incam_orient = result["smpl_params_incam"]["global_orient"]
    detections, global_before, incam_before = detect_isolated_orientation_jumps(
        global_orient, incam_orient, config
    )

    metrics = {
        "schema": "strict-isolated-orientation-guard-v1",
        "frames": int(len(global_orient)),
        "triggered": bool(detections),
        "num_detections": len(detections),
        "detections": detections,
        "thresholds": {
            "global_step_deg": config.global_threshold_deg,
            "incam_step_deg": config.incam_threshold_deg,
            "global_prominence_deg": config.global_prominence_deg,
            "incam_prominence_deg": config.incam_prominence_deg,
            "prominence_ratio": config.prominence_ratio,
            "detection_radius_frames": config.detection_radius,
            "smoothing_radius_frames": config.smoothing_radius,
        },
        "max_global_step_before_deg": float(global_before.max()) if len(global_before) else 0.0,
        "max_incam_step_before_deg": float(incam_before.max()) if len(incam_before) else 0.0,
    }
    if not detections:
        metrics["max_global_step_after_deg"] = metrics["max_global_step_before_deg"]
        metrics["max_incam_step_after_deg"] = metrics["max_incam_step_before_deg"]
        return result, metrics

    guarded = copy.deepcopy(result)
    global_rotations = axis_angle_to_matrix(global_orient).clone()
    incam_rotations = axis_angle_to_matrix(incam_orient).clone()
    for detection in detections:
        for rotations in (global_rotations, incam_rotations):
            _smooth_rotation_window(
                rotations,
                detection["window_start"],
                detection["window_end"],
                iterations=config.smoothing_iterations,
                blend=config.smoothing_blend,
            )

    # Preserve the original axis-angle tensors bit-for-bit outside each window,
    # including both fixed endpoints. Matrix conversion is limited to the
    # internal frames that the optimizer is allowed to change.
    guarded_global = global_orient.clone()
    guarded_incam = incam_orient.clone()
    for detection in detections:
        start = detection["window_start"] + 1
        end = detection["window_end"]
        guarded_global[start:end] = matrix_to_axis_angle(global_rotations[start:end])
        guarded_incam[start:end] = matrix_to_axis_angle(incam_rotations[start:end])
    _replace_orientation_copies(guarded, guarded_global, guarded_incam)
    global_after = rotation_step_degrees(guarded_global)
    incam_after = rotation_step_degrees(guarded_incam)
    metrics["max_global_step_after_deg"] = float(global_after.max())
    metrics["max_incam_step_after_deg"] = float(incam_after.max())
    for detection in metrics["detections"]:
        boundary = detection["boundary_frame"]
        detection["global_step_after_deg"] = float(global_after[boundary])
        detection["incam_step_after_deg"] = float(incam_after[boundary])
    return guarded, metrics


def config_as_dict(config: OrientationGuardConfig = OrientationGuardConfig()) -> dict:
    """Expose configuration without adding a dataclasses dependency to callers."""
    return {
        name: getattr(config, name)
        for name in config.__dataclass_fields__
    }


__all__ = [
    "OrientationGuardConfig",
    "config_as_dict",
    "detect_isolated_orientation_jumps",
    "guard_isolated_orientation_jumps",
    "rotation_step_degrees",
]
