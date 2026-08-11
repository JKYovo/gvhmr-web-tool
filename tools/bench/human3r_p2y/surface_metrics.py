"""Metrics shared by the Human3R multi-surface constraint experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


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
