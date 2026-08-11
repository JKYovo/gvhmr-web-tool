#!/usr/bin/env python3
"""Add conservative Human3R scene-anchored root X/Z corrections to P2-Y."""

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

from apply_p2y import foot_landmarks, render_comparison, smoothstep, tree_equal


WINDOW_NAMES = ("pre", "top", "post")
SURFACE_LEVEL = {"pre": 0.0, "top": 1.0, "post": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--p2-y-result", type=Path, required=True)
    parser.add_argument("--p2-y-metrics", type=Path, required=True)
    parser.add_argument("--human3r-dir", type=Path, required=True)
    parser.add_argument("--scene-planes", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--max-anchor-correction", type=float, default=0.25)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def incam_foot_landmarks(result: Mapping[str, Any], model: torch.nn.Module) -> np.ndarray:
    params = {key: value[None] for key, value in result["smpl_params_incam"].items()}
    _, joints = model(**params)
    return joints[0, :, 17:23].cpu().numpy()


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (T, N, 3), got {points.shape}")
    if intrinsics.shape != (len(points), 3, 3):
        raise ValueError(f"intrinsics must be (T, 3, 3), got {intrinsics.shape}")
    if np.any(points[..., 2] <= 0) or not np.isfinite(points).all():
        raise ValueError("Projected foot points contain invalid camera depths")
    xy = points[..., :2] / points[..., 2:3]
    focal = intrinsics[:, None, (0, 1), (0, 1)]
    center = intrinsics[:, None, (0, 1), (2, 2)]
    return xy * focal + center


def fit_rigid_2d(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Rigid alignment needs at least three paired 2D points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center), full_matrices=False
    )
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def floor_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    camera_x = np.array([1.0, 0.0, 0.0])
    first = camera_x - normal * np.dot(camera_x, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def intersect_surface(
    pixels: np.ndarray,
    intrinsics: np.ndarray,
    normal: np.ndarray,
    floor_offset: float,
    height: float,
) -> np.ndarray:
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
    distance = -(float(floor_offset) + float(height)) / denominator
    if np.any(distance <= 0) or not np.isfinite(distance).all():
        raise RuntimeError("Foot ray does not intersect the selected support surface in front of camera")
    return rays * distance[..., None]


def build_scene_anchors(
    global_root_xz: np.ndarray,
    global_feet: np.ndarray,
    incam_feet: np.ndarray,
    gvhmr_intrinsics: np.ndarray,
    human3r_dir: Path,
    scene: Mapping[str, Any],
    windows: Mapping[str, list[int]],
    contact: np.ndarray,
    contact_threshold: float,
    video_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    depth_paths = sorted((human3r_dir / "depth").glob("*.npy"))
    camera_paths = sorted((human3r_dir / "camera").glob("*.npz"))
    if len(depth_paths) != len(global_feet) or len(camera_paths) != len(global_feet):
        raise RuntimeError(
            "Human3R/GVHMR length mismatch: "
            f"depth={len(depth_paths)}, camera={len(camera_paths)}, gvhmr={len(global_feet)}"
        )
    depth_height, depth_width = np.load(depth_paths[0], mmap_mode="r").shape
    video_width, video_height = video_size
    scale = np.array([depth_width / video_width, depth_height / video_height])
    if abs(scale[0] - scale[1]) > 1e-4:
        raise RuntimeError(f"Human3R/video aspect scaling is inconsistent: {scale}")
    human_intrinsics = np.median(
        np.stack([np.load(path)["intrinsics"] for path in camera_paths[::10]]), axis=0
    )
    pixels = project_points(incam_feet, gvhmr_intrinsics) * scale

    selected = scene["selected_pair"]
    floor = scene["planes"][int(selected["floor_id"])]
    normal = np.asarray(floor["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    basis_x, basis_z = floor_basis(normal)
    box_height = float(selected["separation_m"])
    scene_coordinates: dict[str, np.ndarray] = {}
    for name in WINDOW_NAMES:
        start, end = windows[name]
        points = intersect_surface(
            pixels[start:end],
            human_intrinsics,
            normal,
            float(floor["offset"]),
            SURFACE_LEVEL[name] * box_height,
        )
        scene_coordinates[name] = np.stack([points @ basis_x, points @ basis_z], axis=-1)

    pre_start, pre_end = windows["pre"]
    rotation, translation = fit_rigid_2d(
        scene_coordinates["pre"], global_feet[pre_start:pre_end, :, (0, 2)]
    )
    mapped: dict[str, np.ndarray] = {
        name: coordinates @ rotation + translation
        for name, coordinates in scene_coordinates.items()
    }

    foot_confidence = np.concatenate(
        [
            np.repeat(np.max(contact[:, :2], axis=1, keepdims=True), 3, axis=1),
            np.repeat(np.max(contact[:, 2:4], axis=1, keepdims=True), 3, axis=1),
        ],
        axis=1,
    )
    observed_anchors: dict[str, np.ndarray] = {}
    anchor_spread: dict[str, float] = {}
    residuals: dict[str, np.ndarray] = {}
    for name in WINDOW_NAMES:
        start, end = windows[name]
        baseline_feet = global_feet[start:end, :, (0, 2)]
        delta = mapped[name] - baseline_feet
        mask = foot_confidence[start:end] > contact_threshold
        frame_deltas = []
        for frame_delta, frame_mask in zip(delta, mask):
            if np.any(frame_mask):
                frame_deltas.append(np.median(frame_delta[frame_mask], axis=0))
        frame_deltas = np.asarray(frame_deltas)
        if len(frame_deltas) < 20:
            raise RuntimeError(f"Too few static support frames in {name}: {len(frame_deltas)}")
        observed_anchors[name] = np.median(frame_deltas, axis=0)
        anchor_spread[name] = float(
            np.percentile(
                np.linalg.norm(frame_deltas - observed_anchors[name], axis=1), 95
            )
        )
        residuals[name] = np.linalg.norm(delta, axis=-1)[mask]

    pre_anchor = observed_anchors["pre"]
    post_anchor = observed_anchors["post"] - pre_anchor
    root_centers = {
        name: np.median(global_root_xz[start:end], axis=0)
        for name, (start, end) in windows.items()
    }
    travel = root_centers["post"] - root_centers["pre"]
    denominator = float(np.dot(travel, travel))
    if denominator < 1e-6:
        raise RuntimeError("Pre/post horizontal travel is too small to place the top anchor")
    top_progress = float(
        np.clip(
            np.dot(root_centers["top"] - root_centers["pre"], travel) / denominator,
            0.0,
            1.0,
        )
    )
    # The reconstructed box top is reliable for height, but its small visible
    # support area makes ray/foot horizontal intersections unstable in this
    # clip (the per-frame anchor spread is much larger than on the floor).
    # Estimate horizontal drift from the same physical floor before/after the
    # action, then place the intermediate anchor by trajectory progress.
    anchors = {
        "pre": np.zeros(2, dtype=np.float64),
        "top": post_anchor * top_progress,
        "post": post_anchor,
    }
    diagnostics = {
        "human3r_image_size": [depth_width, depth_height],
        "video_size": [video_width, video_height],
        "image_scale": scale.tolist(),
        "human3r_intrinsics": human_intrinsics.tolist(),
        "floor_normal": normal.tolist(),
        "floor_offset": float(floor["offset"]),
        "rigid_rotation": rotation.tolist(),
        "rigid_translation": translation.tolist(),
        "rigid_determinant": float(np.linalg.det(rotation)),
        "observed_anchors_m": {name: value.tolist() for name, value in observed_anchors.items()},
        "observed_relative_anchors_m": {
            name: (value - pre_anchor).tolist()
            for name, value in observed_anchors.items()
        },
        "observed_anchor_spread_p95_cm": {
            name: value * 100.0 for name, value in anchor_spread.items()
        },
        "relative_anchors_m": {name: value.tolist() for name, value in anchors.items()},
        "top_anchor_method": "pre_post_floor_drift_scaled_by_root_trajectory_progress",
        "top_trajectory_progress": top_progress,
        "pre_alignment_residual_median_cm": float(np.median(residuals["pre"]) * 100.0),
        "pre_alignment_residual_p95_cm": float(np.percentile(residuals["pre"], 95) * 100.0),
    }
    return anchors, diagnostics, mapped


def build_xz_correction(
    length: int,
    anchors: Mapping[str, np.ndarray],
    transition: Mapping[str, Any],
) -> np.ndarray:
    ascent_start = int(transition["ascent_start"])
    ascent_end = int(transition["ascent_end"])
    descent_start = int(transition["descent_start"])
    descent_end = int(transition["descent_end"])
    if not 0 <= ascent_start < ascent_end < descent_start < descent_end < length:
        raise RuntimeError("Invalid P2-Y transition order for X/Z correction")
    correction = np.repeat(np.asarray(anchors["pre"])[None], length, axis=0)
    phase = smoothstep(ascent_end - ascent_start + 1)[:, None]
    correction[ascent_start : ascent_end + 1] = anchors["pre"] + (
        anchors["top"] - anchors["pre"]
    ) * phase
    correction[ascent_end + 1 : descent_start] = anchors["top"]
    phase = smoothstep(descent_end - descent_start + 1)[:, None]
    correction[descent_start : descent_end + 1] = anchors["top"] + (
        anchors["post"] - anchors["top"]
    ) * phase
    correction[descent_end + 1 :] = anchors["post"]
    return correction


def trajectory_metrics(root_xz: np.ndarray, fps: float) -> dict[str, float | list[float]]:
    step = np.linalg.norm(np.diff(root_xz, axis=0), axis=1)
    acceleration = np.linalg.norm(np.diff(root_xz, n=2, axis=0), axis=1) * fps * fps
    return {
        "start_xz_m": root_xz[0].tolist(),
        "end_xz_m": root_xz[-1].tolist(),
        "displacement_xz_m": (root_xz[-1] - root_xz[0]).tolist(),
        "root_xz_step_max_cm_per_frame": float(np.max(step) * 100.0),
        "root_xz_step_p95_cm_per_frame": float(np.percentile(step, 95) * 100.0),
        "root_xz_accel_p95_m_per_s2": float(np.percentile(acceleration, 95)),
    }


def scene_residual_metrics(
    feet: np.ndarray,
    mapped_scene: Mapping[str, np.ndarray],
    correction: np.ndarray,
    windows: Mapping[str, list[int]],
    contact: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    foot_confidence = np.concatenate(
        [
            np.repeat(np.max(contact[:, :2], axis=1, keepdims=True), 3, axis=1),
            np.repeat(np.max(contact[:, 2:4], axis=1, keepdims=True), 3, axis=1),
        ],
        axis=1,
    )
    result: dict[str, dict[str, float]] = {}
    for name in WINDOW_NAMES:
        start, end = windows[name]
        mask = foot_confidence[start:end] > threshold
        baseline_residual = np.linalg.norm(
            feet[start:end, :, (0, 2)] - mapped_scene[name], axis=-1
        )[mask]
        enhanced_residual = np.linalg.norm(
            feet[start:end, :, (0, 2)]
            + correction[start:end, None, :]
            - mapped_scene[name],
            axis=-1,
        )[mask]
        result[name] = {
            "baseline_median_cm": float(np.median(baseline_residual) * 100.0),
            "baseline_p95_cm": float(np.percentile(baseline_residual, 95) * 100.0),
            "p2_xyz_median_cm": float(np.median(enhanced_residual) * 100.0),
            "p2_xyz_p95_cm": float(np.percentile(enhanced_residual, 95) * 100.0),
        }
    return result


def save_curves(
    output_dir: Path,
    baseline_xz: np.ndarray,
    enhanced_xz: np.ndarray,
    correction: np.ndarray,
    fps: float,
) -> None:
    frame = np.arange(len(correction))
    time = frame / fps
    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for axis, index, label in ((axes[0], 0, "root X (m)"), (axes[1], 1, "root Z (m)")):
        axis.plot(time, baseline_xz[:, index], color="black", label="P2-Y / baseline XZ")
        axis.plot(time, enhanced_xz[:, index], color="tab:blue", label="P2-XYZ")
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=0.25)
    axes[2].plot(time, correction[:, 0], label="X correction")
    axes[2].plot(time, correction[:, 1], label="Z correction")
    axes[2].set_ylabel("correction (m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "p2_xyz_curves.png", dpi=160)
    plt.close(figure)

    with (output_dir / "p2_xyz_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame", "time_s", "baseline_x_m", "baseline_z_m", "p2_xyz_x_m", "p2_xyz_z_m", "correction_x_m", "correction_z_m"]
        )
        for index in frame:
            writer.writerow(
                [
                    int(index),
                    float(time[index]),
                    *baseline_xz[index].tolist(),
                    *enhanced_xz[index].tolist(),
                    *correction[index].tolist(),
                ]
            )


def save_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    base = payload["baseline_xz"]
    enhanced = payload["p2_xyz"]
    anchors = payload["scene_alignment"]["relative_anchors_m"]
    spread = payload["scene_alignment"]["observed_anchor_spread_p95_cm"]
    residual = payload["scene_residual_cm"]
    text = f"""# Human3R P2-XYZ 单视频离线实验

## 结论

- 判定：`{payload['decision']}`。
- P2-Y 的高度修正保持不变；新增 X/Z 只使用 Human3R 静态场景平面与 GVHMR 投影足点，不使用 Human3R 的人体 root。
- Human3R 人体分支在本视频 500–800 帧存在明显误检，因此直接迁移其 SMPL-X root 的方案已拒绝。
- 箱顶水平交点的逐帧离散 P95 为 {spread['top']:.2f} cm，明显大于 pre 的 {spread['pre']:.2f} cm 和 post 的 {spread['post']:.2f} cm，所以箱顶观测只用于 Y；X/Z 漂移只由同一地面上的 pre/post 估计，中间锚点按原轨迹进度放置。
- 这是无水平真值条件下的诊断性候选，不能把 `diagnostic_pass` 解读成 X/Z 精度已经得到证明。

## 水平修正锚点

| 稳定表面 | X 修正 | Z 修正 |
| --- | ---: | ---: |
| pre | {anchors['pre'][0]:.4f} m | {anchors['pre'][1]:.4f} m |
| top | {anchors['top'][0]:.4f} m | {anchors['top'][1]:.4f} m |
| post | {anchors['post'][0]:.4f} m | {anchors['post'][1]:.4f} m |

## 连续性

| 指标 | baseline/P2-Y | P2-XYZ | 保护上限 |
| --- | ---: | ---: | ---: |
| XZ 最大步长 | {base['root_xz_step_max_cm_per_frame']:.4f} cm/frame | {enhanced['root_xz_step_max_cm_per_frame']:.4f} cm/frame | {payload['guardrails']['root_xz_step_limit_cm_per_frame']:.4f} |
| XZ 加速度 P95 | {base['root_xz_accel_p95_m_per_s2']:.4f} m/s² | {enhanced['root_xz_accel_p95_m_per_s2']:.4f} m/s² | {payload['guardrails']['root_xz_accel_limit_m_per_s2']:.4f} |

## 稳定窗口场景残差

| 窗口 | baseline median | P2-XYZ median | baseline P95 | P2-XYZ P95 |
| --- | ---: | ---: | ---: | ---: |
| pre | {residual['pre']['baseline_median_cm']:.3f} cm | {residual['pre']['p2_xyz_median_cm']:.3f} cm | {residual['pre']['baseline_p95_cm']:.3f} cm | {residual['pre']['p2_xyz_p95_cm']:.3f} cm |
| top | {residual['top']['baseline_median_cm']:.3f} cm | {residual['top']['p2_xyz_median_cm']:.3f} cm | {residual['top']['baseline_p95_cm']:.3f} cm | {residual['top']['p2_xyz_p95_cm']:.3f} cm |
| post | {residual['post']['baseline_median_cm']:.3f} cm | {residual['post']['p2_xyz_median_cm']:.3f} cm | {residual['post']['baseline_p95_cm']:.3f} cm | {residual['post']['p2_xyz_p95_cm']:.3f} cm |

## 限制

- 水平锚点由当前视频的人工稳定窗口估计，尚未自动检测。
- Human3R 场景尺度、地面平面和相机静止假设仍会影响结果。
- 当前没有测量标定或 3D ground truth，只能检查场景一致性和连续性。

## 产物

- `p2_xyz_hmr4d_results.pt`：供 GMR 读取的实验候选。
- `p2_y_vs_p2_xyz.mp4`：隔离显示新增 X/Z 修正的并排视频。
- `metrics.json`、`p2_xyz_curves.csv`、`p2_xyz_curves.png`：指标和曲线。
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (
        args.gvhmr_result,
        args.p2_y_result,
        args.p2_y_metrics,
        args.human3r_dir,
        args.scene_planes,
        args.video,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = torch.load(args.gvhmr_result, map_location="cpu")
    p2_y = torch.load(args.p2_y_result, map_location="cpu")
    p2_y_metrics = json.loads(args.p2_y_metrics.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_planes.read_text(encoding="utf-8"))
    length = baseline["smpl_params_global"]["transl"].shape[0]
    if p2_y["smpl_params_global"]["transl"].shape[0] != length:
        raise RuntimeError("Baseline/P2-Y length mismatch")
    windows = p2_y_metrics["windows"]
    transition = p2_y_metrics["transition_diagnostics"]

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

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23

    model = SmplxLiteV437Coco23().eval()
    global_feet = foot_landmarks(baseline, model)
    incam_feet = incam_foot_landmarks(baseline, model)
    contact = baseline["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().numpy()
    anchors, alignment, mapped_scene = build_scene_anchors(
        baseline["smpl_params_global"]["transl"][:, (0, 2)].numpy(),
        global_feet,
        incam_feet,
        baseline["K_fullimg"].numpy(),
        args.human3r_dir,
        scene,
        windows,
        contact,
        args.contact_threshold,
        video_size,
    )
    correction = build_xz_correction(length, anchors, transition)
    max_anchor = max(float(np.linalg.norm(value)) for value in anchors.values())

    enhanced = copy.deepcopy(p2_y)
    translation = enhanced["smpl_params_global"]["transl"].clone()
    baseline_xz = baseline["smpl_params_global"]["transl"][:, (0, 2)].numpy()
    enhanced_xz = baseline_xz + correction
    translation[:, (0, 2)] = torch.as_tensor(enhanced_xz, dtype=translation.dtype)
    enhanced["smpl_params_global"]["transl"] = translation

    baseline_trajectory = trajectory_metrics(baseline_xz, fps)
    enhanced_trajectory = trajectory_metrics(enhanced_xz, fps)
    residual_metrics = scene_residual_metrics(
        global_feet,
        mapped_scene,
        correction,
        windows,
        contact,
        args.contact_threshold,
    )
    step_limit = max(3.0, baseline_trajectory["root_xz_step_max_cm_per_frame"] * 1.5)
    acceleration_limit = baseline_trajectory["root_xz_accel_p95_m_per_s2"] * 1.25

    source_translation = baseline["smpl_params_global"]["transl"]
    p2_y_translation = p2_y["smpl_params_global"]["transl"]
    target_translation = enhanced["smpl_params_global"]["transl"]
    invariants = {
        "top_level_keys_equal": baseline.keys() == enhanced.keys(),
        "global_param_keys_equal": baseline["smpl_params_global"].keys()
        == enhanced["smpl_params_global"].keys(),
        "incam_param_keys_equal": baseline["smpl_params_incam"].keys()
        == enhanced["smpl_params_incam"].keys(),
        "root_y_equal_to_p2_y": bool(torch.equal(p2_y_translation[:, 1], target_translation[:, 1])),
        "body_pose_equal": bool(torch.equal(baseline["smpl_params_global"]["body_pose"], enhanced["smpl_params_global"]["body_pose"])),
        "global_orient_equal": bool(torch.equal(baseline["smpl_params_global"]["global_orient"], enhanced["smpl_params_global"]["global_orient"])),
        "betas_equal": bool(torch.equal(baseline["smpl_params_global"]["betas"], enhanced["smpl_params_global"]["betas"])),
        "incam_equal": tree_equal(baseline["smpl_params_incam"], enhanced["smpl_params_incam"]),
        "K_fullimg_equal": tree_equal(baseline["K_fullimg"], enhanced["K_fullimg"]),
        "net_outputs_equal": tree_equal(baseline["net_outputs"], enhanced["net_outputs"]),
        "finite_root_xyz": bool(torch.isfinite(target_translation).all()),
        "baseline_and_p2_y_xz_equal": bool(torch.equal(source_translation[:, (0, 2)], p2_y_translation[:, (0, 2)])),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"P2-XYZ invariant failed: {invariants}")

    endpoint_residual_improved = (
        residual_metrics["post"]["p2_xyz_median_cm"]
        < residual_metrics["post"]["baseline_median_cm"]
        and residual_metrics["post"]["p2_xyz_p95_cm"]
        < residual_metrics["post"]["baseline_p95_cm"]
    )
    decision = (
        "diagnostic_pass"
        if max_anchor <= args.max_anchor_correction
        and enhanced_trajectory["root_xz_step_max_cm_per_frame"] <= step_limit
        and enhanced_trajectory["root_xz_accel_p95_m_per_s2"] <= acceleration_limit
        and endpoint_residual_improved
        else "guardrail_failed"
    )
    payload = {
        "method": "Human3R P2-XYZ",
        "decision": decision,
        "frames": length,
        "fps": fps,
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "p2_y_result": str(args.p2_y_result.resolve()),
        "human3r_dir": str(args.human3r_dir.resolve()),
        "scene_planes": str(args.scene_planes.resolve()),
        "windows": windows,
        "transition_diagnostics": transition,
        "contact_threshold": args.contact_threshold,
        "scene_alignment": alignment,
        "scene_residual_cm": residual_metrics,
        "baseline_xz": baseline_trajectory,
        "p2_xyz": enhanced_trajectory,
        "max_anchor_correction_m": max_anchor,
        "guardrails": {
            "max_anchor_correction_m": args.max_anchor_correction,
            "root_xz_step_limit_cm_per_frame": step_limit,
            "root_xz_accel_limit_m_per_s2": acceleration_limit,
            "post_floor_median_and_p95_residual_must_improve": True,
        },
        "invariants": invariants,
    }
    torch.save(enhanced, args.output_dir / "p2_xyz_hmr4d_results.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_curves(args.output_dir, baseline_xz, enhanced_xz, correction, fps)
    save_report(args.output_dir, payload)
    if not args.skip_video:
        render_comparison(
            p2_y,
            enhanced,
            args.output_dir / "p2_y_vs_p2_xyz.mp4",
            baseline_label="Human3R P2-Y (root Y only)",
            enhanced_name="p2_xyz",
            enhanced_label="Human3R P2-XYZ (root XYZ)",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
