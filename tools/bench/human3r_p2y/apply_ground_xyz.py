#!/usr/bin/env python3
"""Constrain a FootMR trajectory to one Human3R ground plane."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from apply_p2y import foot_landmarks, render_comparison, tree_equal
from apply_p2xyz import fit_rigid_2d, floor_basis, incam_foot_landmarks, project_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--human3r-dir", type=Path, required=True)
    parser.add_argument("--ground-plane", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-observations", type=int, default=30)
    parser.add_argument("--smoothing-seconds", type=float, default=2.0)
    parser.add_argument("--max-xz-correction", type=float, default=0.35)
    parser.add_argument("--max-y-correction", type=float, default=0.25)
    parser.add_argument("--max-contact-speed-ratio", type=float, default=1.25)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def intersect_ground(
    pixels: np.ndarray,
    intrinsics: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Intersect image rays with a plane; invalid/behind-camera rays become NaN."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    rays = np.concatenate(
        [
            (pixels - intrinsics[(0, 1), (2, 2)])
            / intrinsics[(0, 1), (0, 1)],
            np.ones((*pixels.shape[:-1], 1), dtype=np.float64),
        ],
        axis=-1,
    )
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -float(offset) / denominator
    valid = np.isfinite(distance) & (distance > 0.1) & (distance < 30.0)
    result = np.full_like(rays, np.nan, dtype=np.float64)
    result[valid] = rays[valid] * distance[valid, None]
    return result


def foot_confidence(contact: np.ndarray) -> np.ndarray:
    if contact.ndim != 2 or contact.shape[1] < 4:
        raise ValueError(f"Expected four contact probabilities, got {contact.shape}")
    return np.concatenate(
        [
            np.repeat(np.max(contact[:, :2], axis=1, keepdims=True), 3, axis=1),
            np.repeat(np.max(contact[:, 2:4], axis=1, keepdims=True), 3, axis=1),
        ],
        axis=1,
    )


def first_stable_calibration(
    marker_mask: np.ndarray,
    fps: float,
    calibration_seconds: float,
    minimum_observations: int,
) -> np.ndarray:
    frame_valid = np.any(marker_mask, axis=1)
    search_end = min(len(frame_valid), max(int(round(calibration_seconds * fps)), 1))
    selected = np.flatnonzero(frame_valid[:search_end])
    if len(selected) < minimum_observations:
        selected = np.flatnonzero(frame_valid)[: max(minimum_observations, search_end)]
    if len(selected) < minimum_observations:
        raise RuntimeError(
            f"Too few high-confidence ground contacts for calibration: {len(selected)}"
        )
    return selected


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.median(views, axis=-1)


def smooth_observations(
    observations: np.ndarray,
    observation_mask: np.ndarray,
    calibration_frames: np.ndarray,
    fps: float,
    smoothing_seconds: float,
    minimum_observations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    length, dimensions = observations.shape
    valid = observation_mask & np.isfinite(observations).all(axis=1)
    if int(valid.sum()) < minimum_observations:
        raise RuntimeError(f"Too few valid correction observations: {int(valid.sum())}")

    def interpolate(mask: np.ndarray) -> np.ndarray:
        frame = np.arange(length)
        output = np.empty((length, dimensions), dtype=np.float64)
        indices = np.flatnonzero(mask)
        for axis in range(dimensions):
            output[:, axis] = np.interp(frame, indices, observations[indices, axis])
        return output

    initial = interpolate(valid)
    median_window = max(5, int(round(0.5 * fps)))
    local = rolling_median(initial, median_window)
    residual = np.linalg.norm(observations - local, axis=1)
    residual_valid = residual[valid]
    center = float(np.median(residual_valid))
    mad = float(np.median(np.abs(residual_valid - center)))
    outlier_limit = max(0.03, center + 4.0 * 1.4826 * mad)
    filtered = valid & (residual <= outlier_limit)
    if int(filtered.sum()) < minimum_observations:
        raise RuntimeError(
            f"Outlier rejection left too few observations: {int(filtered.sum())}"
        )
    filled = interpolate(filtered)
    trend = rolling_median(filled, median_window)

    smooth_window = max(3, int(round(smoothing_seconds * fps)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.hanning(smooth_window)
    if not np.any(kernel):
        kernel = np.ones(smooth_window)
    kernel /= kernel.sum()
    radius = smooth_window // 2
    correction = np.empty_like(trend)
    for axis in range(dimensions):
        padded = np.pad(trend[:, axis], (radius, radius), mode="edge")
        correction[:, axis] = np.convolve(padded, kernel, mode="valid")

    calibration_mask = np.zeros(length, dtype=bool)
    calibration_mask[calibration_frames] = True
    calibration_mask &= filtered
    if not np.any(calibration_mask):
        calibration_mask[calibration_frames] = True
    origin = np.median(correction[calibration_mask], axis=0)
    correction -= origin
    diagnostics = {
        "raw_observation_frames": int(valid.sum()),
        "filtered_observation_frames": int(filtered.sum()),
        "outlier_frames": int(valid.sum() - filtered.sum()),
        "outlier_limit_cm": outlier_limit * 100.0,
        "median_window_frames": median_window,
        "smoothing_window_frames": smooth_window,
        "calibration_origin_m": origin.tolist(),
        "first_observation_frame": int(np.flatnonzero(filtered)[0]),
        "last_observation_frame": int(np.flatnonzero(filtered)[-1]),
    }
    return correction, diagnostics


def trajectory_metrics(root: np.ndarray, feet: np.ndarray, contact_mask: np.ndarray, fps: float) -> dict[str, float | list[float]]:
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    acceleration = np.linalg.norm(np.diff(root, n=2, axis=0), axis=1) * fps * fps
    foot_step = np.linalg.norm(np.diff(feet, axis=0), axis=-1)
    speed_mask = contact_mask[1:] & contact_mask[:-1]
    selected_speed = foot_step[speed_mask]
    return {
        "start_root_xyz_m": root[0].tolist(),
        "end_root_xyz_m": root[-1].tolist(),
        "root_displacement_xyz_m": (root[-1] - root[0]).tolist(),
        "root_step_max_cm_per_frame": float(np.max(step) * 100.0),
        "root_step_p95_cm_per_frame": float(np.percentile(step, 95) * 100.0),
        "root_accel_p95_m_per_s2": float(np.percentile(acceleration, 95)),
        "contact_foot_speed_median_mm_per_frame": float(np.median(selected_speed) * 1000.0),
        "contact_foot_speed_p95_mm_per_frame": float(np.percentile(selected_speed, 95) * 1000.0),
    }


def residual_metrics(
    global_feet: np.ndarray,
    mapped_scene_xz: np.ndarray,
    ground_y: float,
    marker_mask: np.ndarray,
    correction: np.ndarray,
) -> dict[str, dict[str, float]]:
    baseline_xz = np.linalg.norm(
        global_feet[..., (0, 2)] - mapped_scene_xz, axis=-1
    )[marker_mask]
    enhanced_xz = np.linalg.norm(
        global_feet[..., (0, 2)]
        + correction[:, None, (0, 2)]
        - mapped_scene_xz,
        axis=-1,
    )[marker_mask]
    baseline_y = np.abs(global_feet[..., 1] - ground_y)[marker_mask]
    enhanced_y = np.abs(
        global_feet[..., 1] + correction[:, None, 1] - ground_y
    )[marker_mask]

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "median_cm": float(np.median(values) * 100.0),
            "p95_cm": float(np.percentile(values, 95) * 100.0),
        }

    return {
        "baseline_xz": summarize(baseline_xz),
        "ground_xyz_xz": summarize(enhanced_xz),
        "baseline_y": summarize(baseline_y),
        "ground_xyz_y": summarize(enhanced_y),
    }


def save_curves(
    output_dir: Path,
    baseline_root: np.ndarray,
    enhanced_root: np.ndarray,
    correction: np.ndarray,
    observations: np.ndarray,
    observation_mask: np.ndarray,
    fps: float,
) -> None:
    frame = np.arange(len(correction))
    time = frame / fps
    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    labels = ("X", "Y", "Z")
    for axis, index, label in zip(axes[:3], range(3), labels):
        axis.plot(time, baseline_root[:, index], color="black", label=f"baseline root {label}")
        axis.plot(time, enhanced_root[:, index], color="tab:blue", label=f"ground XYZ root {label}")
        axis.set_ylabel(f"root {label} (m)")
        axis.legend()
        axis.grid(alpha=0.25)
    for index, label in enumerate(labels):
        axes[3].plot(time, correction[:, index], label=f"{label} correction")
        axes[3].scatter(
            time[observation_mask],
            observations[observation_mask, index],
            s=2,
            alpha=0.12,
        )
    axes[3].set_ylabel("correction (m)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(ncol=3)
    axes[3].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "ground_xyz_curves.png", dpi=160)
    plt.close(figure)

    with (output_dir / "ground_xyz_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame", "time_s", "baseline_x_m", "baseline_y_m", "baseline_z_m",
                "enhanced_x_m", "enhanced_y_m", "enhanced_z_m",
                "correction_x_m", "correction_y_m", "correction_z_m", "observed",
            ]
        )
        for index in frame:
            writer.writerow(
                [
                    int(index), float(time[index]), *baseline_root[index].tolist(),
                    *enhanced_root[index].tolist(), *correction[index].tolist(),
                    int(observation_mask[index]),
                ]
            )


def save_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    residual = payload["scene_residual_cm"]
    base = payload["baseline"]
    enhanced = payload["ground_xyz"]
    report = f"""# Human3R 单地面 XYZ 约束实验

## 结论

- 判定：`{payload['decision']}`。
- 人体姿态、脚踝细化和相机内人体来自 GVHMR + FootMR；Human3R 只提供静态地面几何。
- 只修改 `smpl_params_global.transl`，不修改 body pose、朝向、betas、incam、相机内参或 `net_outputs`。
- 自动使用高置信度接触帧，不使用爬箱视频的 `pre/top/post` 人工窗口。

## 场景残差

| 指标 | baseline | ground XYZ |
| --- | ---: | ---: |
| 接触足水平残差 median | {residual['baseline_xz']['median_cm']:.3f} cm | {residual['ground_xyz_xz']['median_cm']:.3f} cm |
| 接触足水平残差 P95 | {residual['baseline_xz']['p95_cm']:.3f} cm | {residual['ground_xyz_xz']['p95_cm']:.3f} cm |
| 接触足离地高度 median | {residual['baseline_y']['median_cm']:.3f} cm | {residual['ground_xyz_y']['median_cm']:.3f} cm |
| 接触足离地高度 P95 | {residual['baseline_y']['p95_cm']:.3f} cm | {residual['ground_xyz_y']['p95_cm']:.3f} cm |

## 连续性

| 指标 | baseline | ground XYZ |
| --- | ---: | ---: |
| root 最大步长 | {base['root_step_max_cm_per_frame']:.3f} cm/frame | {enhanced['root_step_max_cm_per_frame']:.3f} cm/frame |
| root 加速度 P95 | {base['root_accel_p95_m_per_s2']:.3f} m/s² | {enhanced['root_accel_p95_m_per_s2']:.3f} m/s² |
| 接触足速度 P95 | {base['contact_foot_speed_p95_mm_per_frame']:.3f} mm/frame | {enhanced['contact_foot_speed_p95_mm_per_frame']:.3f} mm/frame |

## 边界

- 单目场景没有测量真值，`diagnostic_pass` 只表示场景一致性和连续性保护条件通过，不证明绝对 XYZ 精度。
- 地面交点依赖固定相机、Human3R 深度尺度和足点投影；动态相机或地面弱纹理需要单独验证。
- 非接触区间仅平滑插值 root correction，人体姿态始终不变。

## 产物

- `ground_xyz_hmr4d_results.pt`：保持 GVHMR tensor schema，可供 GMR 读取。
- `metrics.json`、`ground_xyz_curves.csv`、`ground_xyz_curves.png`：完整指标与逐帧修正。
- `footmr_vs_ground_xyz.mp4`：可选的全局轨迹并排渲染。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (args.gvhmr_result, args.human3r_dir, args.ground_plane, args.video):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = torch.load(args.gvhmr_result, map_location="cpu")
    scene = json.loads(args.ground_plane.read_text(encoding="utf-8"))
    ground = scene["selected_ground"]
    length = baseline["smpl_params_global"]["transl"].shape[0]

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if video_frames != length:
        raise RuntimeError(f"Video/result length mismatch: {video_frames} vs {length}")

    depth_paths = sorted((args.human3r_dir / "depth").glob("*.npy"))
    camera_paths = sorted((args.human3r_dir / "camera").glob("*.npz"))
    if len(depth_paths) != length or len(camera_paths) != length:
        raise RuntimeError(
            f"Human3R/GVHMR length mismatch: depth={len(depth_paths)}, "
            f"camera={len(camera_paths)}, gvhmr={length}"
        )
    depth_height, depth_width = np.load(depth_paths[0], mmap_mode="r").shape
    video_width, video_height = video_size
    scale = np.array([depth_width / video_width, depth_height / video_height])
    if abs(scale[0] - scale[1]) > 1e-4:
        raise RuntimeError(f"Human3R/video aspect scaling is inconsistent: {scale}")
    human_intrinsics = np.median(
        np.stack([np.load(path)["intrinsics"] for path in camera_paths[::10]]), axis=0
    )

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23

    model = SmplxLiteV437Coco23().eval()
    global_feet = foot_landmarks(baseline, model)
    incam_feet = incam_foot_landmarks(baseline, model)
    pixels = project_points(incam_feet, baseline["K_fullimg"].numpy()) * scale
    normal = np.asarray(ground["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    intersections = intersect_ground(pixels, human_intrinsics, normal, float(ground["offset"]))
    basis_x, basis_z = floor_basis(normal)
    scene_xz = np.stack([intersections @ basis_x, intersections @ basis_z], axis=-1)

    contact = baseline["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().numpy()
    confidence = foot_confidence(contact)
    inside = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < depth_width)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < depth_height)
    )
    marker_mask = (
        (confidence > args.contact_threshold)
        & inside
        & np.isfinite(scene_xz).all(axis=-1)
    )
    calibration_frames = first_stable_calibration(
        marker_mask,
        fps,
        args.calibration_seconds,
        args.minimum_observations,
    )
    calibration_mask = marker_mask[calibration_frames]
    rotation, translation = fit_rigid_2d(
        scene_xz[calibration_frames][calibration_mask],
        global_feet[calibration_frames][:, :, (0, 2)][calibration_mask],
    )
    mapped_scene_xz = scene_xz @ rotation + translation
    ground_y = float(
        np.median(global_feet[calibration_frames, :, 1][calibration_mask])
    )

    observations = np.full((length, 3), np.nan, dtype=np.float64)
    observation_mask = np.any(marker_mask, axis=1)
    for frame_id in np.flatnonzero(observation_mask):
        valid = marker_mask[frame_id]
        xz_delta = np.median(
            mapped_scene_xz[frame_id, valid]
            - global_feet[frame_id, valid][:, (0, 2)],
            axis=0,
        )
        y_delta = float(np.median(ground_y - global_feet[frame_id, valid, 1]))
        observations[frame_id] = (xz_delta[0], y_delta, xz_delta[1])

    correction, smoothing = smooth_observations(
        observations,
        observation_mask,
        calibration_frames,
        fps,
        args.smoothing_seconds,
        args.minimum_observations,
    )
    max_xz = float(np.max(np.linalg.norm(correction[:, (0, 2)], axis=1)))
    max_y = float(np.max(np.abs(correction[:, 1])))

    enhanced = copy.deepcopy(baseline)
    source_translation = baseline["smpl_params_global"]["transl"]
    baseline_root = source_translation.numpy()
    enhanced_root = baseline_root + correction
    enhanced["smpl_params_global"]["transl"] = torch.as_tensor(
        enhanced_root, dtype=source_translation.dtype
    )
    enhanced_feet = global_feet + correction[:, None, :]

    baseline_metrics = trajectory_metrics(baseline_root, global_feet, marker_mask, fps)
    enhanced_metrics = trajectory_metrics(enhanced_root, enhanced_feet, marker_mask, fps)
    residual = residual_metrics(
        global_feet, mapped_scene_xz, ground_y, marker_mask, correction
    )
    step_limit = max(3.0, baseline_metrics["root_step_max_cm_per_frame"] * 1.5)
    acceleration_limit = baseline_metrics["root_accel_p95_m_per_s2"] * 1.25
    contact_speed_limit = (
        baseline_metrics["contact_foot_speed_p95_mm_per_frame"]
        * args.max_contact_speed_ratio
    )
    xz_improved = (
        residual["ground_xyz_xz"]["median_cm"] < residual["baseline_xz"]["median_cm"]
        and residual["ground_xyz_xz"]["p95_cm"] < residual["baseline_xz"]["p95_cm"]
    )
    y_improved = (
        residual["ground_xyz_y"]["median_cm"] < residual["baseline_y"]["median_cm"]
        and residual["ground_xyz_y"]["p95_cm"] < residual["baseline_y"]["p95_cm"]
    )

    target_translation = enhanced["smpl_params_global"]["transl"]
    invariants = {
        "top_level_keys_equal": baseline.keys() == enhanced.keys(),
        "global_param_keys_equal": baseline["smpl_params_global"].keys()
        == enhanced["smpl_params_global"].keys(),
        "incam_param_keys_equal": baseline["smpl_params_incam"].keys()
        == enhanced["smpl_params_incam"].keys(),
        "body_pose_equal": bool(torch.equal(baseline["smpl_params_global"]["body_pose"], enhanced["smpl_params_global"]["body_pose"])),
        "global_orient_equal": bool(torch.equal(baseline["smpl_params_global"]["global_orient"], enhanced["smpl_params_global"]["global_orient"])),
        "betas_equal": bool(torch.equal(baseline["smpl_params_global"]["betas"], enhanced["smpl_params_global"]["betas"])),
        "incam_equal": tree_equal(baseline["smpl_params_incam"], enhanced["smpl_params_incam"]),
        "K_fullimg_equal": tree_equal(baseline["K_fullimg"], enhanced["K_fullimg"]),
        "net_outputs_equal": tree_equal(baseline["net_outputs"], enhanced["net_outputs"]),
        "finite_root_xyz": bool(torch.isfinite(target_translation).all()),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"Ground XYZ invariant failed: {invariants}")

    decision = (
        "diagnostic_pass"
        if max_xz <= args.max_xz_correction
        and max_y <= args.max_y_correction
        and enhanced_metrics["root_step_max_cm_per_frame"] <= step_limit
        and enhanced_metrics["root_accel_p95_m_per_s2"] <= acceleration_limit
        and enhanced_metrics["contact_foot_speed_p95_mm_per_frame"] <= contact_speed_limit
        and xz_improved
        and y_improved
        else "guardrail_failed"
    )
    payload = {
        "method": "Human3R ground-only XYZ",
        "decision": decision,
        "frames": length,
        "fps": fps,
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "human3r_dir": str(args.human3r_dir.resolve()),
        "ground_plane": str(args.ground_plane.resolve()),
        "video": str(args.video.resolve()),
        "contact_threshold": args.contact_threshold,
        "contact_frames": int(observation_mask.sum()),
        "calibration_frames": calibration_frames.tolist(),
        "scene_alignment": {
            "human3r_image_size": [depth_width, depth_height],
            "video_size": [video_width, video_height],
            "image_scale": scale.tolist(),
            "human3r_intrinsics": human_intrinsics.tolist(),
            "ground_normal": normal.tolist(),
            "ground_offset": float(ground["offset"]),
            "ground_y_in_gvhmr_m": ground_y,
            "rigid_rotation": rotation.tolist(),
            "rigid_translation": translation.tolist(),
            "rigid_determinant": float(np.linalg.det(rotation)),
        },
        "smoothing": smoothing,
        "correction": {
            "start_xyz_m": correction[0].tolist(),
            "end_xyz_m": correction[-1].tolist(),
            "max_xz_m": max_xz,
            "max_abs_y_m": max_y,
        },
        "scene_residual_cm": residual,
        "baseline": baseline_metrics,
        "ground_xyz": enhanced_metrics,
        "guardrails": {
            "max_xz_correction_m": args.max_xz_correction,
            "max_abs_y_correction_m": args.max_y_correction,
            "root_step_limit_cm_per_frame": step_limit,
            "root_accel_limit_m_per_s2": acceleration_limit,
            "contact_foot_speed_p95_limit_mm_per_frame": contact_speed_limit,
            "xz_median_and_p95_improved": xz_improved,
            "y_median_and_p95_improved": y_improved,
        },
        "invariants": invariants,
    }
    torch.save(enhanced, args.output_dir / "ground_xyz_hmr4d_results.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_curves(
        args.output_dir,
        baseline_root,
        enhanced_root,
        correction,
        observations,
        observation_mask,
        fps,
    )
    save_report(args.output_dir, payload)
    if not args.skip_video:
        render_comparison(
            baseline,
            enhanced,
            args.output_dir / "footmr_vs_ground_xyz.mp4",
            baseline_label="GVHMR + FootMR",
            enhanced_name="ground_xyz",
            enhanced_label="FootMR + Human3R ground XYZ",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
