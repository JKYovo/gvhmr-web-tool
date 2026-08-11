#!/usr/bin/env python3
"""Refine a Human3R Ground-XYZ trajectory with contact-foot CoTracker3 tracks.

This is an offline diagnostic.  CoTracker3 observes relative shoe motion during
FootMR contact intervals; it never replaces body pose or the Ground-XYZ height.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from apply_ground_xyz import (
    foot_confidence,
    intersect_ground,
    trajectory_metrics,
)
from apply_p2y import foot_landmarks, render_comparison, tree_equal
from apply_p2xyz import floor_basis, incam_foot_landmarks, project_points


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COTRACKER_ROOT = REPO_ROOT / "third-party/CoTracker"
DEFAULT_CHECKPOINT = REPO_ROOT / "inputs/cotracker_assets/scaled_offline.pth"
EXPECTED_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
OFFICIAL_CHECKPOINT_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
FOOT_NAMES = ("left", "right")


@dataclass(frozen=True)
class TrackWindow:
    foot: int
    start: int
    end: int


@dataclass
class RelativeConstraint:
    foot: int
    start: int
    frames: np.ndarray
    delta_xz: np.ndarray
    weights: np.ndarray
    points_used: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-xyz-result", type=Path, required=True)
    parser.add_argument("--ground-xyz-metrics", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--merge-gap-frames", type=int, default=3)
    parser.add_argument("--minimum-contact-frames", type=int, default=8)
    parser.add_argument("--window-frames", type=int, default=60)
    parser.add_argument("--window-overlap", type=int, default=8)
    parser.add_argument("--tracking-width", type=int, default=512)
    parser.add_argument("--foot-crop-width", type=int, default=128)
    parser.add_argument("--foot-crop-height", type=int, default=96)
    parser.add_argument("--crop-margin", type=int, default=16)
    parser.add_argument("--no-foot-crop", action="store_true")
    parser.add_argument("--max-track-step-pixels", type=float, default=35.0)
    parser.add_argument("--max-point-disagreement", type=float, default=0.06)
    parser.add_argument("--smooth-weight", type=float, default=100.0)
    parser.add_argument(
        "--absolute-weight",
        type=float,
        default=0.5,
        help="Ground-XYZ zero-correction prior preventing cross-window drift accumulation.",
    )
    parser.add_argument("--minimum-tracked-coverage", type=float, default=0.50)
    parser.add_argument("--max-xz-correction", type=float, default=0.15)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-track-overlay", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open true runs."""
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def merge_short_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if maximum_gap <= 0:
        return result
    for start, end in true_runs(~result):
        if start > 0 and end < len(result) and end - start <= maximum_gap:
            result[start:end] = True
    return result


def build_track_windows(
    contact: np.ndarray,
    threshold: float,
    merge_gap: int,
    minimum_frames: int,
    window_frames: int,
    overlap: int,
) -> tuple[list[TrackWindow], np.ndarray]:
    if contact.ndim != 2 or contact.shape[1] < 4:
        raise ValueError(f"Expected contact probabilities (T, >=4), got {contact.shape}")
    if not 0 <= overlap < window_frames:
        raise ValueError("window overlap must be non-negative and smaller than window")
    per_foot = np.stack(
        [np.max(contact[:, :2], axis=1), np.max(contact[:, 2:4], axis=1)], axis=1
    )
    masks = np.zeros_like(per_foot, dtype=bool)
    windows: list[TrackWindow] = []
    step = window_frames - overlap
    for foot in range(2):
        masks[:, foot] = merge_short_gaps(per_foot[:, foot] > threshold, merge_gap)
        for segment_start, segment_end in true_runs(masks[:, foot]):
            if segment_end - segment_start < minimum_frames:
                masks[segment_start:segment_end, foot] = False
                continue
            start = segment_start
            while start < segment_end:
                end = min(start + window_frames, segment_end)
                if end - start < minimum_frames and windows and windows[-1].foot == foot:
                    previous = windows.pop()
                    windows.append(
                        TrackWindow(foot, int(previous.start), int(segment_end))
                    )
                    break
                windows.append(TrackWindow(foot, int(start), int(end)))
                if end == segment_end:
                    break
                start += step
    windows.sort(key=lambda item: (item.start, item.foot))
    return windows, masks


def load_video_window(
    capture: cv2.VideoCapture,
    start: int,
    end: int,
    output_size: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for frame_id in range(start, end):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read video frame {frame_id}")
        resized = cv2.resize(bgr, output_size, interpolation=cv2.INTER_AREA)
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            resized = resized[y0:y1, x0:x1]
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def foot_crop_bounds(
    points: np.ndarray,
    image_size: tuple[int, int],
    minimum_size: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    """Build one fixed per-window crop containing all projected foot points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("Cannot crop an empty projected foot track")
    image_width, image_height = image_size
    minimum_width, minimum_height = minimum_size
    target_width = max(minimum_width, int(np.ceil(np.ptp(points[:, 0]))) + 2 * margin)
    target_height = max(minimum_height, int(np.ceil(np.ptp(points[:, 1]))) + 2 * margin)
    target_width = min(target_width, image_width)
    target_height = min(target_height, image_height)
    center = np.median(points, axis=0)
    x0 = int(np.clip(round(center[0] - target_width / 2), 0, image_width - target_width))
    y0 = int(np.clip(round(center[1] - target_height / 2), 0, image_height - target_height))
    return x0, y0, x0 + target_width, y0 + target_height


def tracked_relative_constraint(
    window: TrackWindow,
    tracks: np.ndarray,
    visibility: np.ndarray,
    source_foot_xz: np.ndarray,
    human_intrinsics: np.ndarray,
    ground_normal: np.ndarray,
    ground_offset: float,
    basis_x: np.ndarray,
    basis_z: np.ndarray,
    rigid_rotation: np.ndarray,
    rigid_translation: np.ndarray,
    image_size: tuple[int, int],
    max_track_step_pixels: float,
    max_point_disagreement: float,
) -> RelativeConstraint | None:
    """Convert one three-point foot track to relative GVHMR root constraints."""
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    if tracks.shape != (window.end - window.start, 3, 2):
        raise ValueError(f"Unexpected track shape {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(f"Unexpected visibility shape {visibility.shape}")
    width, height = image_size
    inside = (
        (tracks[..., 0] >= 0)
        & (tracks[..., 0] < width)
        & (tracks[..., 1] >= 0)
        & (tracks[..., 1] < height)
    )
    step = np.zeros(visibility.shape, dtype=np.float64)
    step[1:] = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
    step_ok = step <= max_track_step_pixels
    step_ok[1:] &= step[:-1] <= max_track_step_pixels
    valid = visibility & inside & step_ok & np.isfinite(tracks).all(axis=-1)

    intersections = intersect_ground(tracks, human_intrinsics, ground_normal, ground_offset)
    scene_xz = np.stack(
        [intersections @ basis_x, intersections @ basis_z], axis=-1
    )
    mapped_xz = scene_xz @ rigid_rotation + rigid_translation
    raw_offset = mapped_xz - source_foot_xz
    relative = raw_offset - raw_offset[0:1]
    valid &= np.isfinite(relative).all(axis=-1)

    observations = np.full((len(tracks), 2), np.nan, dtype=np.float64)
    points_used = np.zeros(len(tracks), dtype=np.int64)
    for local_frame in range(len(tracks)):
        frame_valid = valid[local_frame]
        if int(frame_valid.sum()) < 2:
            continue
        values = relative[local_frame, frame_valid]
        center = np.median(values, axis=0)
        disagreement = np.linalg.norm(values - center, axis=1)
        keep = disagreement <= max_point_disagreement
        if int(keep.sum()) < 2:
            continue
        observations[local_frame] = np.median(values[keep], axis=0)
        points_used[local_frame] = int(keep.sum())

    usable = np.isfinite(observations).all(axis=1)
    # A relative window must include its anchor and enough later evidence.
    if not usable[0] or int(usable.sum()) < 3:
        return None
    frames = np.arange(window.start, window.end, dtype=np.int64)[usable]
    return RelativeConstraint(
        foot=window.foot,
        start=window.start,
        frames=frames,
        delta_xz=observations[usable],
        weights=points_used[usable].astype(np.float64) / 3.0,
        points_used=points_used[usable],
    )


def solve_relative_constraints(
    length: int,
    constraints: Sequence[RelativeConstraint],
    smooth_weight: float,
    absolute_weight: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve c[t] - c[anchor] = tracked relative correction with smoothing."""
    if not constraints:
        raise RuntimeError("CoTracker produced no usable contact constraints")
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
    except ImportError as error:
        raise RuntimeError("CoTracker fusion requires SciPy for sparse constraint solving") from error
    system = lil_matrix((length, length), dtype=np.float64)
    rhs = np.zeros((length, 2), dtype=np.float64)
    observed_frames = np.zeros(length, dtype=bool)
    equation_count = 0
    for constraint in constraints:
        anchor = constraint.start
        for frame, delta, weight in zip(
            constraint.frames, constraint.delta_xz, constraint.weights
        ):
            if frame == anchor:
                continue
            weight = float(weight)
            system[frame, frame] += weight
            system[anchor, anchor] += weight
            system[frame, anchor] -= weight
            system[anchor, frame] -= weight
            rhs[frame] += weight * delta
            rhs[anchor] -= weight * delta
            observed_frames[frame] = True
            observed_frames[anchor] = True
            equation_count += 1
    if equation_count < 2:
        raise RuntimeError("Too few non-anchor CoTracker constraints")
    for frame in range(1, length):
        weight = float(smooth_weight)
        system[frame, frame] += weight
        system[frame - 1, frame - 1] += weight
        system[frame, frame - 1] -= weight
        system[frame - 1, frame] -= weight
    if absolute_weight < 0:
        raise ValueError("absolute weight must be non-negative")
    for frame in range(length):
        system[frame, frame] += float(absolute_weight) + 1e-8
    system[0, 0] += 100.0
    system = system.tocsr()
    correction = np.stack(
        [spsolve(system, rhs[:, axis]) for axis in range(2)], axis=-1
    )
    if not np.isfinite(correction).all():
        raise RuntimeError("Non-finite CoTracker correction")
    gaps = true_runs(~observed_frames)
    longest_gap = max((end - start for start, end in gaps), default=0)
    return correction, {
        "constraint_windows": len(constraints),
        "constraint_equations": equation_count,
        "observed_frames": int(observed_frames.sum()),
        "first_observed_frame": int(np.flatnonzero(observed_frames)[0]),
        "last_observed_frame": int(np.flatnonzero(observed_frames)[-1]),
        "longest_unobserved_run_frames": int(longest_gap),
        "smooth_weight": smooth_weight,
        "absolute_weight": absolute_weight,
    }


def clone_with_root_xz(source: Mapping[str, Any], correction_xz: np.ndarray) -> dict[str, Any]:
    result = copy.deepcopy(source)
    translation = result["smpl_params_global"]["transl"].clone()
    translation[:, 0] += torch.as_tensor(correction_xz[:, 0], dtype=translation.dtype)
    translation[:, 2] += torch.as_tensor(correction_xz[:, 1], dtype=translation.dtype)
    result["smpl_params_global"]["transl"] = translation
    return result


def summarize(values: np.ndarray, scale: float = 100.0) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"median": float("nan"), "p95": float("nan")}
    return {
        "median": float(np.median(finite) * scale),
        "p95": float(np.percentile(finite, 95) * scale),
    }


def constraint_residuals(
    correction: np.ndarray, constraints: Sequence[RelativeConstraint]
) -> tuple[np.ndarray, np.ndarray]:
    before, after = [], []
    for item in constraints:
        modeled = correction[item.frames] - correction[item.start]
        before.extend(np.linalg.norm(item.delta_xz, axis=1))
        after.extend(np.linalg.norm(modeled - item.delta_xz, axis=1))
    return np.asarray(before), np.asarray(after)


def scene_xz_residuals(
    global_feet: np.ndarray,
    projected_pixels: np.ndarray,
    marker_mask: np.ndarray,
    intrinsics: np.ndarray,
    normal: np.ndarray,
    offset: float,
    basis_x: np.ndarray,
    basis_z: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    correction_xz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    intersections = intersect_ground(projected_pixels, intrinsics, normal, offset)
    scene_xz = np.stack([intersections @ basis_x, intersections @ basis_z], axis=-1)
    mapped = scene_xz @ rotation + translation
    source = np.linalg.norm(global_feet[..., (0, 2)] - mapped, axis=-1)[marker_mask]
    enhanced = np.linalg.norm(
        global_feet[..., (0, 2)] + correction_xz[:, None, :] - mapped, axis=-1
    )[marker_mask]
    return source, enhanced


def save_track_overlay(
    video: Path,
    output: Path,
    track_sum: np.ndarray,
    track_count: np.ndarray,
    size: tuple[int, int],
    fps: float,
) -> None:
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    colors = ((40, 220, 40), (40, 170, 255))
    for frame_id in range(len(track_sum)):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read overlay frame {frame_id}")
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        for foot in range(2):
            for point in range(3):
                count = track_count[frame_id, foot, point]
                if count:
                    xy = track_sum[frame_id, foot, point] / count
                    cv2.circle(frame, tuple(np.rint(xy).astype(int)), 4, colors[foot], -1)
        writer.write(frame)
    capture.release()
    writer.release()


def save_curves(
    output_dir: Path,
    source_root: np.ndarray,
    enhanced_root: np.ndarray,
    correction: np.ndarray,
    fps: float,
) -> None:
    frame = np.arange(len(correction))
    time = frame / fps
    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(time, source_root[:, 0], label="Ground-XYZ X", color="black")
    axes[0].plot(time, enhanced_root[:, 0], label="CoTracker X", color="tab:blue")
    axes[1].plot(time, source_root[:, 2], label="Ground-XYZ Z", color="black")
    axes[1].plot(time, enhanced_root[:, 2], label="CoTracker Z", color="tab:blue")
    axes[2].plot(time, correction[:, 0], label="X correction")
    axes[2].plot(time, correction[:, 1], label="Z correction")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("correction (m)")
    figure.tight_layout()
    figure.savefig(output_dir / "cotracker_ground_xz_curves.png", dpi=160)
    plt.close(figure)
    with (output_dir / "cotracker_ground_xz_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame", "time_s", "source_x_m", "source_z_m", "enhanced_x_m", "enhanced_z_m", "correction_x_m", "correction_z_m"]
        )
        for index in frame:
            writer.writerow(
                [index, time[index], source_root[index, 0], source_root[index, 2], enhanced_root[index, 0], enhanced_root[index, 2], *correction[index]]
            )


def save_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    lock = payload["tracked_lock_residual_cm"]
    speed_a = payload["ground_xyz"]["contact_foot_speed_p95_mm_per_frame"]
    speed_b = payload["cotracker_ground_xz"]["contact_foot_speed_p95_mm_per_frame"]
    scene = payload["scene_xz_residual_cm"]
    text = f"""# CoTracker3 接触足 Ground-XZ 实验

## 结论

- 判定：`{payload['decision']}`。
- CoTracker3 只约束接触期间的 global root X/Z；Ground-XYZ 的 Y、全部人体姿态、incam、相机和网络输出精确不变。
- CoTracker3 固定于官方 commit `{payload['cotracker']['commit']}`，其主要代码使用 CC BY-NC 4.0，仅限符合该非商业许可的用途。

## 指标

| 指标 | Ground-XYZ | CoTracker Ground-XZ |
| --- | ---: | ---: |
| 跟踪锁定残差 median | {lock['before']['median']:.3f} cm | {lock['after']['median']:.3f} cm |
| 跟踪锁定残差 P95 | {lock['before']['p95']:.3f} cm | {lock['after']['p95']:.3f} cm |
| 场景 XZ 残差 median | {scene['before']['median']:.3f} cm | {scene['after']['median']:.3f} cm |
| 场景 XZ 残差 P95 | {scene['before']['p95']:.3f} cm | {scene['after']['p95']:.3f} cm |
| 接触足速度 P95 | {speed_a:.3f} mm/frame | {speed_b:.3f} mm/frame |

`diagnostic_pass` 不代表具有 X/Z 真值；它只表示接触足锁定、场景一致性、轨迹连续性和修改范围保护同时通过。若判定失败，正式推荐文件不会生成，继续使用 Ground-XYZ。
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (
        args.ground_xyz_result,
        args.ground_xyz_metrics,
        args.video,
        args.checkpoint,
        args.cotracker_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(args.cotracker_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"CoTracker revision mismatch: expected {EXPECTED_COMMIT}, got {commit}"
        )
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if (
        args.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve()
        and checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Official CoTracker checkpoint SHA256 mismatch: "
            f"expected {OFFICIAL_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    source = torch.load(args.ground_xyz_result, map_location="cpu")
    ground_metrics = json.loads(args.ground_xyz_metrics.read_text(encoding="utf-8"))
    alignment = ground_metrics["scene_alignment"]
    length = int(source["smpl_params_global"]["transl"].shape[0])

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != length:
        raise RuntimeError(f"Video/result frame mismatch: {video_frames} vs {length}")
    tracking_height = int(round(video_height * args.tracking_width / video_width))
    tracking_size = (args.tracking_width, tracking_height)
    human_width, human_height = map(int, alignment["human3r_image_size"])
    if tracking_size != (human_width, human_height):
        raise RuntimeError(
            f"Tracking size {tracking_size} must equal Human3R size {(human_width, human_height)}"
        )

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23

    landmark_model = SmplxLiteV437Coco23().eval()
    global_feet = foot_landmarks(source, landmark_model)
    incam_feet = incam_foot_landmarks(source, landmark_model)
    full_pixels = project_points(incam_feet, source["K_fullimg"].numpy())
    pixel_scale = np.array([human_width / video_width, human_height / video_height])
    projected_pixels = full_pixels * pixel_scale
    contact = source["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().numpy()
    windows, contact_feet = build_track_windows(
        contact,
        args.contact_threshold,
        args.merge_gap_frames,
        args.minimum_contact_frames,
        args.window_frames,
        args.window_overlap,
    )
    if not windows:
        raise RuntimeError("No FootMR contact interval is long enough for tracking")

    sys.path.insert(0, str(args.cotracker_root))
    try:
        from cotracker.predictor import CoTrackerPredictor
    except ImportError as error:
        raise RuntimeError(
            f"Cannot import CoTracker3 from fixed submodule {args.cotracker_root}"
        ) from error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The P5 video experiment requires CUDA; CPU helper tests remain available")
    predictor = CoTrackerPredictor(
        checkpoint=str(args.checkpoint), offline=True, window_len=args.window_frames
    ).to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
    torch.cuda.reset_peak_memory_stats()

    intrinsics = np.asarray(alignment["human3r_intrinsics"], dtype=np.float64)
    normal = np.asarray(alignment["ground_normal"], dtype=np.float64)
    offset = float(alignment["ground_offset"])
    basis_x, basis_z = floor_basis(normal)
    rotation = np.asarray(alignment["rigid_rotation"], dtype=np.float64)
    translation = np.asarray(alignment["rigid_translation"], dtype=np.float64)
    constraints: list[RelativeConstraint] = []
    window_records = []
    track_sum = np.zeros((length, 2, 3, 2), dtype=np.float64)
    track_count = np.zeros((length, 2, 3), dtype=np.int64)
    for window_id, window in enumerate(windows):
        crop_box = None
        if not args.no_foot_crop:
            crop_box = foot_crop_bounds(
                projected_pixels[
                    window.start : window.end,
                    window.foot * 3 : window.foot * 3 + 3,
                ],
                tracking_size,
                (args.foot_crop_width, args.foot_crop_height),
                args.crop_margin,
            )
        frames = load_video_window(
            capture, window.start, window.end, tracking_size, crop_box
        )
        query_xy = projected_pixels[window.start, window.foot * 3 : window.foot * 3 + 3]
        inside_query = (
            (query_xy[:, 0] >= 0)
            & (query_xy[:, 0] < human_width)
            & (query_xy[:, 1] >= 0)
            & (query_xy[:, 1] < human_height)
        )
        if int(inside_query.sum()) < 3:
            window_records.append({"id": window_id, "foot": FOOT_NAMES[window.foot], "start": window.start, "end": window.end, "status": "query_outside"})
            continue
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
        queries = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
        query_origin = np.zeros(2, dtype=np.float64)
        if crop_box is not None:
            query_origin = np.asarray(crop_box[:2], dtype=np.float64)
        queries[0, :, 1:] = torch.as_tensor(
            query_xy - query_origin, dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            predicted_tracks, visibility = predictor(video_tensor, queries=queries)
        tracks_np = predicted_tracks[0].float().cpu().numpy()
        tracks_np += query_origin
        visibility_np = visibility[0].cpu().numpy().astype(bool)
        del frames, video_tensor, queries, predicted_tracks, visibility
        constraint = tracked_relative_constraint(
            window,
            tracks_np,
            visibility_np,
            global_feet[window.start : window.end, window.foot * 3 : window.foot * 3 + 3][:, :, (0, 2)],
            intrinsics,
            normal,
            offset,
            basis_x,
            basis_z,
            rotation,
            translation,
            tracking_size,
            args.max_track_step_pixels,
            args.max_point_disagreement,
        )
        for local_frame, frame_id in enumerate(range(window.start, window.end)):
            for point in range(3):
                if visibility_np[local_frame, point]:
                    track_sum[frame_id, window.foot, point] += tracks_np[local_frame, point]
                    track_count[frame_id, window.foot, point] += 1
        record = {
            "id": window_id,
            "foot": FOOT_NAMES[window.foot],
            "start": window.start,
            "end": window.end,
            "frames": window.end - window.start,
            "visible_point_fraction": float(visibility_np.mean()),
            "crop_box_xyxy": None if crop_box is None else list(crop_box),
            "usable_frames": 0 if constraint is None else int(len(constraint.frames)),
            "status": "rejected" if constraint is None else "usable",
        }
        window_records.append(record)
        if constraint is not None:
            constraints.append(constraint)
        torch.cuda.empty_cache()
        print(
            f"[{window_id + 1}/{len(windows)}] {record['foot']} "
            f"{window.start}:{window.end} {record['status']} usable={record['usable_frames']}",
            flush=True,
        )
    capture.release()

    correction_xz, solver = solve_relative_constraints(
        length, constraints, args.smooth_weight, args.absolute_weight
    )
    tracked_contact_frames = np.zeros((length, 2), dtype=bool)
    for item in constraints:
        tracked_contact_frames[item.frames, item.foot] = True
    contact_frame_count = int(contact_feet.sum())
    tracked_coverage = float(
        (tracked_contact_frames & contact_feet).sum() / max(contact_frame_count, 1)
    )
    enhanced = clone_with_root_xz(source, correction_xz)
    source_root = source["smpl_params_global"]["transl"].numpy()
    enhanced_root = enhanced["smpl_params_global"]["transl"].numpy()
    correction_xyz = np.zeros((length, 3), dtype=np.float64)
    correction_xyz[:, (0, 2)] = correction_xz
    enhanced_feet = global_feet + correction_xyz[:, None, :]
    marker_contact = np.repeat(contact_feet, 3, axis=1)
    source_trajectory = trajectory_metrics(source_root, global_feet, marker_contact, fps)
    enhanced_trajectory = trajectory_metrics(enhanced_root, enhanced_feet, marker_contact, fps)
    before_lock, after_lock = constraint_residuals(correction_xz, constraints)

    confidence = foot_confidence(contact)
    projected_inside = (
        (projected_pixels[..., 0] >= 0)
        & (projected_pixels[..., 0] < human_width)
        & (projected_pixels[..., 1] >= 0)
        & (projected_pixels[..., 1] < human_height)
    )
    marker_mask = (
        (confidence > args.contact_threshold)
        & projected_inside
        & np.isfinite(projected_pixels).all(axis=-1)
    )
    before_scene, after_scene = scene_xz_residuals(
        global_feet,
        projected_pixels,
        marker_mask,
        intrinsics,
        normal,
        offset,
        basis_x,
        basis_z,
        rotation,
        translation,
        correction_xz,
    )

    max_correction = float(np.max(np.linalg.norm(correction_xz, axis=1)))
    step_limit = max(3.0, source_trajectory["root_step_max_cm_per_frame"] * 1.25)
    acceleration_limit = source_trajectory["root_accel_p95_m_per_s2"] * 1.25
    lock_before = summarize(before_lock)
    lock_after = summarize(after_lock)
    scene_before = summarize(before_scene)
    scene_after = summarize(after_scene)
    speed_improved = (
        enhanced_trajectory["contact_foot_speed_p95_mm_per_frame"]
        <= source_trajectory["contact_foot_speed_p95_mm_per_frame"] * 0.98
        and enhanced_trajectory["contact_foot_speed_median_mm_per_frame"]
        <= source_trajectory["contact_foot_speed_median_mm_per_frame"]
    )
    lock_improved = (
        lock_after["median"] < lock_before["median"]
        and lock_after["p95"] < lock_before["p95"]
    )
    scene_preserved = (
        scene_after["median"] <= scene_before["median"] * 1.10
        and scene_after["p95"] <= scene_before["p95"] * 1.10
    )
    invariants = {
        "top_level_keys_equal": source.keys() == enhanced.keys(),
        "global_param_keys_equal": source["smpl_params_global"].keys() == enhanced["smpl_params_global"].keys(),
        "body_pose_equal": bool(torch.equal(source["smpl_params_global"]["body_pose"], enhanced["smpl_params_global"]["body_pose"])),
        "global_orient_equal": bool(torch.equal(source["smpl_params_global"]["global_orient"], enhanced["smpl_params_global"]["global_orient"])),
        "betas_equal": bool(torch.equal(source["smpl_params_global"]["betas"], enhanced["smpl_params_global"]["betas"])),
        "global_y_equal": bool(torch.equal(source["smpl_params_global"]["transl"][:, 1], enhanced["smpl_params_global"]["transl"][:, 1])),
        "incam_equal": tree_equal(source["smpl_params_incam"], enhanced["smpl_params_incam"]),
        "K_fullimg_equal": tree_equal(source["K_fullimg"], enhanced["K_fullimg"]),
        "net_outputs_equal": tree_equal(source["net_outputs"], enhanced["net_outputs"]),
        "finite_root": bool(torch.isfinite(enhanced["smpl_params_global"]["transl"]).all()),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"CoTracker invariant failed: {invariants}")
    guardrails = {
        "minimum_tracked_coverage": args.minimum_tracked_coverage,
        "tracked_coverage_pass": tracked_coverage >= args.minimum_tracked_coverage,
        "max_xz_correction_m": args.max_xz_correction,
        "correction_pass": max_correction <= args.max_xz_correction,
        "root_step_limit_cm_per_frame": step_limit,
        "root_step_pass": enhanced_trajectory["root_step_max_cm_per_frame"] <= step_limit,
        "root_accel_limit_m_per_s2": acceleration_limit,
        "root_accel_pass": enhanced_trajectory["root_accel_p95_m_per_s2"] <= acceleration_limit,
        "tracked_lock_improved": lock_improved,
        "contact_speed_improved_by_2_percent": speed_improved,
        "scene_xz_within_10_percent": scene_preserved,
    }
    decision = "diagnostic_pass" if all(
        value for key, value in guardrails.items() if key.endswith("pass") or key in {"tracked_lock_improved", "contact_speed_improved_by_2_percent", "scene_xz_within_10_percent"}
    ) else "guardrail_failed"
    payload = {
        "method": "Ground-XYZ + CoTracker3 contact-foot relative XZ",
        "decision": decision,
        "frames": length,
        "fps": fps,
        "source": str(args.ground_xyz_result.resolve()),
        "video": str(args.video.resolve()),
        "cotracker": {
            "commit": commit,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "parameter_count": parameter_count,
            "license": "CC BY-NC 4.0 (majority of upstream CoTracker)",
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "tracking": {
            "contact_threshold": args.contact_threshold,
            "detected_windows": len(windows),
            "usable_windows": len(constraints),
            "contact_foot_frames": contact_frame_count,
            "tracked_contact_foot_frames": int((tracked_contact_frames & contact_feet).sum()),
            "tracked_coverage": tracked_coverage,
            "windows": window_records,
            **solver,
        },
        "correction": {
            "start_xz_m": correction_xz[0].tolist(),
            "end_xz_m": correction_xz[-1].tolist(),
            "max_norm_m": max_correction,
        },
        "tracked_lock_residual_cm": {"before": lock_before, "after": lock_after},
        "scene_xz_residual_cm": {"before": scene_before, "after": scene_after},
        "ground_xyz": source_trajectory,
        "cotracker_ground_xz": enhanced_trajectory,
        "guardrails": guardrails,
        "invariants": invariants,
    }
    torch.save(enhanced, args.output_dir / "candidate_hmr4d_results.pt")
    if decision == "diagnostic_pass":
        torch.save(enhanced, args.output_dir / "cotracker_ground_xz_hmr4d_results.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "tracked_points.npz",
        track_sum=track_sum,
        track_count=track_count,
        contact_feet=contact_feet,
        correction_xz=correction_xz,
        constraint_anchor=np.concatenate(
            [np.full(len(item.frames), item.start, dtype=np.int64) for item in constraints]
        ),
        constraint_frame=np.concatenate([item.frames for item in constraints]),
        constraint_delta_xz=np.concatenate([item.delta_xz for item in constraints]),
        constraint_weight=np.concatenate([item.weights for item in constraints]),
    )
    save_curves(args.output_dir, source_root, enhanced_root, correction_xz, fps)
    save_report(args.output_dir, payload)
    if not args.skip_track_overlay:
        save_track_overlay(
            args.video,
            args.output_dir / "cotracker_tracks.mp4",
            track_sum,
            track_count,
            tracking_size,
            fps,
        )
    if not args.skip_video:
        render_comparison(
            source,
            enhanced,
            args.output_dir / "ground_xyz_vs_cotracker_xz.mp4",
            baseline_label="FootMR + Human3R Ground-XYZ",
            enhanced_name="cotracker_ground_xz",
            enhanced_label="Ground-XYZ + CoTracker3 XZ",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
