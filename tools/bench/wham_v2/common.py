"""Pure helpers shared by the FootMR-v2 trajectory experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch


WINDOWS: Dict[str, Tuple[int, int]] = {
    "pre": (30, 105),
    "top": (600, 795),
    "post": (970, 1056),
}


@dataclass(frozen=True)
class WindowMetrics:
    pre_floor_m: float
    top_floor_m: float
    post_floor_m: float
    floor_return_cm: float
    signed_floor_return_cm: float
    top_relative_cm: float


def validate_windows(length: int, windows: Mapping[str, Tuple[int, int]] = WINDOWS) -> None:
    for name in ("pre", "top", "post"):
        if name not in windows:
            raise ValueError(f"Missing required window: {name}")
        start, end = windows[name]
        if not 0 <= start < end <= length:
            raise ValueError(f"Window {name}={start}:{end} is invalid for length={length}")


def contact_y_correction(
    root_y: torch.Tensor,
    support_joints_world: torch.Tensor,
    contact_probability: torch.Tensor,
    threshold: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the deterministic first stage of WHAM to GVHMR's vertical motion.

    Args:
        root_y: ``(T,)`` current world root height.
        support_joints_world: ``(T, 4, 3)`` ordered as left ankle/foot,
            right ankle/foot.
        contact_probability: ``(T, 4)`` probabilities matching the joints.
        threshold: hard contact threshold used by GVHMR static-camera postprocess.

    Returns:
        Corrected root Y, per-step support-foot Y displacement, and contact mask.
    """
    if root_y.ndim != 1:
        raise ValueError(f"root_y must be (T,), got {tuple(root_y.shape)}")
    if support_joints_world.shape != (root_y.shape[0], 4, 3):
        raise ValueError(
            "support_joints_world must be (T, 4, 3), got "
            f"{tuple(support_joints_world.shape)}"
        )
    if contact_probability.shape != (root_y.shape[0], 4):
        raise ValueError(
            f"contact_probability must be (T, 4), got {tuple(contact_probability.shape)}"
        )

    joint_displacement = support_joints_world[1:] - support_joints_world[:-1]
    contact_mask = contact_probability[:-1] > threshold
    denominator = contact_mask.sum(dim=-1).clamp_min(1)
    support_displacement = (
        joint_displacement * contact_mask[..., None]
    ).sum(dim=-2) / denominator[..., None]

    root_step = root_y[1:] - root_y[:-1]
    corrected_step = root_step - support_displacement[:, 1]
    corrected_y = torch.cat(
        [root_y[:1], root_y[:1] + torch.cumsum(corrected_step, dim=0)], dim=0
    )
    return corrected_y, support_displacement[:, 1], contact_mask


def interpolate_track(
    values: np.ndarray,
    frame_ids: np.ndarray,
    length: int,
    max_gap: int = 5,
    min_coverage: float = 0.95,
) -> np.ndarray:
    """Interpolate a tracked quantity onto the full video timeline.

    Long internal gaps and insufficient coverage are rejected instead of silently
    fabricating a WHAM trajectory.
    """
    values = np.asarray(values)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if values.shape[0] != frame_ids.shape[0]:
        raise ValueError("values and frame_ids must have the same leading dimension")
    if frame_ids.ndim != 1 or len(frame_ids) == 0:
        raise ValueError("frame_ids must be a non-empty 1D array")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("frame_ids must be strictly increasing")
    if frame_ids[0] < 0 or frame_ids[-1] >= length:
        raise ValueError("frame_ids fall outside the target video")
    if len(frame_ids) / float(length) < min_coverage:
        raise ValueError(
            f"WHAM track coverage {len(frame_ids)}/{length} is below {min_coverage:.0%}"
        )
    gaps = np.diff(frame_ids) - 1
    if len(gaps) and int(gaps.max()) > max_gap:
        raise ValueError(f"WHAM track contains a gap of {int(gaps.max())} frames (max={max_gap})")

    flat = values.reshape(values.shape[0], -1)
    timeline = np.arange(length)
    result = np.empty((length, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(timeline, frame_ids, flat[:, column])
    return result.reshape((length,) + values.shape[1:]).astype(values.dtype, copy=False)


def hybrid_root_y(
    gvhmr_y: np.ndarray,
    wham_w0_y: np.ndarray,
    wham_w2_y: np.ndarray,
    pre_window: Tuple[int, int] = WINDOWS["pre"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the C-delta and C-rootY diagnostic transfers."""
    gvhmr_y = np.asarray(gvhmr_y)
    wham_w0_y = np.asarray(wham_w0_y)
    wham_w2_y = np.asarray(wham_w2_y)
    if not (gvhmr_y.shape == wham_w0_y.shape == wham_w2_y.shape):
        raise ValueError("All root-Y inputs must have identical shapes")
    start, end = pre_window
    correction = wham_w2_y - wham_w0_y
    correction = correction - np.median(correction[start:end])
    c_delta = gvhmr_y + correction
    c_root_y = np.median(gvhmr_y[start:end]) + (
        wham_w2_y - np.median(wham_w2_y[start:end])
    )
    return c_delta, c_root_y


def foot_height_curve(feet_world: np.ndarray) -> np.ndarray:
    feet_world = np.asarray(feet_world)
    if feet_world.ndim != 3 or feet_world.shape[-1] != 3:
        raise ValueError(f"feet_world must be (T, N, 3), got {feet_world.shape}")
    return feet_world[..., 1].min(axis=1)


def window_metrics(
    foot_min_y: np.ndarray,
    windows: Mapping[str, Tuple[int, int]] = WINDOWS,
) -> WindowMetrics:
    foot_min_y = np.asarray(foot_min_y)
    validate_windows(len(foot_min_y), windows)
    medians = {
        name: float(np.median(foot_min_y[start:end]))
        for name, (start, end) in windows.items()
    }
    signed = (medians["post"] - medians["pre"]) * 100.0
    return WindowMetrics(
        pre_floor_m=medians["pre"],
        top_floor_m=medians["top"],
        post_floor_m=medians["post"],
        floor_return_cm=abs(signed),
        signed_floor_return_cm=signed,
        top_relative_cm=(medians["top"] - medians["pre"]) * 100.0,
    )


def percentile_abs(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(np.abs(array), percentile)) if array.size else 0.0
