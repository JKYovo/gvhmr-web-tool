#!/usr/bin/env python3
"""Apply a Human3R multi-surface constraint to GVHMR global root Y only."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT / "wham_v2"))
from common import WINDOWS, foot_height_curve, percentile_abs, window_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--scene-planes", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def foot_landmarks(result: Mapping[str, Any], model: torch.nn.Module) -> np.ndarray:
    params = {key: value[None] for key, value in result["smpl_params_global"].items()}
    _, joints = model(**params)
    return joints[0, :, 17:23].cpu().numpy()


def first_stable(
    condition: np.ndarray, start: int, end: int, stable_frames: int
) -> int | None:
    condition = np.asarray(condition, dtype=np.int8)
    end = min(end, len(condition))
    if end - start < stable_frames:
        return None
    counts = np.convolve(condition[start:end], np.ones(stable_frames, dtype=np.int32), mode="valid")
    matches = np.flatnonzero(counts == stable_frames)
    return int(start + matches[0]) if len(matches) else None


def smoothstep(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 1), dtype=np.float64)
    phase = np.linspace(0.0, 1.0, length)
    return phase * phase * (3.0 - 2.0 * phase)


def build_offset(
    curve: np.ndarray, contact: np.ndarray, box_height: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if contact.shape != (len(curve), 4):
        raise ValueError(
            f"contact must have shape ({len(curve)}, 4), got {contact.shape}"
        )
    levels = {
        name: float(np.median(curve[start:end]))
        for name, (start, end) in WINDOWS.items()
    }
    offsets = {
        "pre": -levels["pre"],
        "top": box_height - levels["top"],
        "post": -levels["post"],
    }

    ascent_start = first_stable(
        curve > levels["pre"] + 0.05,
        WINDOWS["pre"][1],
        WINDOWS["top"][0],
        8,
    )
    if ascent_start is None:
        ascent_start = WINDOWS["pre"][1]
    ascent_end = first_stable(
        curve > levels["top"] - 0.03,
        ascent_start + 1,
        WINDOWS["top"][0],
        30,
    )
    if ascent_end is None:
        ascent_end = WINDOWS["top"][0]

    curve_descent_start = first_stable(
        curve < levels["top"] - 0.05,
        WINDOWS["top"][1],
        WINDOWS["post"][0],
        8,
    )
    if curve_descent_start is None:
        curve_descent_start = WINDOWS["top"][1]
    curve_descent_end = first_stable(
        np.abs(curve - levels["post"]) < 0.03,
        curve_descent_start + 1,
        len(curve),
        20,
    )
    if curve_descent_end is None:
        curve_descent_end = WINDOWS["post"][0]

    # The baseline foot-height curve changes abruptly when the subject lands,
    # so interpolating only between its two detected levels can compress a
    # large absolute correction into a handful of frames.  The network's four
    # static-foot probabilities expose the complete take-off/landing interval.
    # Requiring all four landmarks to be confident deliberately starts the
    # blend before support is lost and finishes it at the first short period in
    # which both feet are again static.  The minimum duration is an additional
    # continuity guard for noisy contact logits.
    all_feet_static = np.min(contact, axis=1)
    contact_descent_start = first_stable(
        all_feet_static < 0.95,
        WINDOWS["top"][1],
        WINDOWS["post"][0],
        12,
    )
    descent_start = (
        contact_descent_start
        if contact_descent_start is not None
        else curve_descent_start
    )
    contact_descent_end = first_stable(
        all_feet_static > 0.95,
        descent_start + 12,
        WINDOWS["post"][0],
        3,
    )
    descent_end = (
        contact_descent_end if contact_descent_end is not None else curve_descent_end
    )
    minimum_descent_frames = 60
    descent_end = max(descent_end, descent_start + minimum_descent_frames - 1)
    descent_end = min(descent_end, WINDOWS["post"][0])

    if not 0 <= ascent_start < ascent_end < descent_start < descent_end < len(curve):
        raise RuntimeError(
            "Invalid detected transition order: "
            f"{ascent_start}, {ascent_end}, {descent_start}, {descent_end}"
        )

    correction = np.full(len(curve), offsets["pre"], dtype=np.float64)
    correction[ascent_start : ascent_end + 1] = offsets["pre"] + (
        offsets["top"] - offsets["pre"]
    ) * smoothstep(ascent_end - ascent_start + 1)
    correction[ascent_end + 1 : descent_start] = offsets["top"]
    correction[descent_start : descent_end + 1] = offsets["top"] + (
        offsets["post"] - offsets["top"]
    ) * smoothstep(descent_end - descent_start + 1)
    correction[descent_end + 1 :] = offsets["post"]

    diagnostics: dict[str, Any] = {
        **{f"baseline_{name}_m": value for name, value in levels.items()},
        **{f"offset_{name}_m": value for name, value in offsets.items()},
        "ascent_start": ascent_start,
        "ascent_end": ascent_end,
        "descent_start": descent_start,
        "descent_end": descent_end,
        "curve_descent_start": curve_descent_start,
        "curve_descent_end": curve_descent_end,
        "contact_descent_start": contact_descent_start,
        "contact_descent_end": contact_descent_end,
        "contact_threshold": 0.95,
        "minimum_descent_frames": minimum_descent_frames,
    }
    return correction, diagnostics


def clone_with_root_y(
    source: Mapping[str, Any], corrected_y: np.ndarray
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    translation = result["smpl_params_global"]["transl"].clone()
    translation[:, 1] = torch.as_tensor(corrected_y, dtype=translation.dtype)
    result["smpl_params_global"]["transl"] = translation
    return result


def tree_equal(left: Any, right: Any) -> bool:
    """Exact equality for result subtrees that P2-Y must not modify."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            tree_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def variant_metrics(
    feet: np.ndarray, root_y: np.ndarray, contact: np.ndarray, fps: float
) -> tuple[dict[str, float], np.ndarray]:
    curve = foot_height_curve(feet)
    window = window_metrics(curve)
    speed = np.zeros(feet.shape[:2], dtype=np.float64)
    speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1)
    left = np.max(contact[:, :2], axis=-1) > 0.8
    right = np.max(contact[:, 2:4], axis=-1) > 0.8
    speed_mask = np.stack([left, left, left, right, right, right], axis=-1)
    selected_speed = speed[speed_mask]
    root_step = np.diff(root_y)
    root_acceleration = np.diff(root_y, n=2) * fps * fps
    metrics = dict(window.__dict__)
    metrics.update(
        {
            "contact_speed_median_mm_per_frame": float(np.median(selected_speed) * 1000.0),
            "contact_speed_p95_mm_per_frame": float(np.percentile(selected_speed, 95) * 1000.0),
            "root_step_max_cm_per_frame": percentile_abs(root_step, 100.0) * 100.0,
            "root_accel_p95_m_per_s2": percentile_abs(root_acceleration, 95.0),
        }
    )
    return metrics, curve


def save_diagnostics(
    output_dir: Path,
    curves: Mapping[str, np.ndarray],
    roots: Mapping[str, np.ndarray],
    correction: np.ndarray,
    contact: np.ndarray,
    fps: float,
    box_height: float,
) -> None:
    frame = np.arange(len(correction))
    time = frame / fps
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(time, curves["baseline"], color="black", label="GVHMR + FootMR")
    axes[0].plot(time, curves["p2_y"], color="tab:blue", label="Human3R P2-Y")
    axes[0].axhline(0.0, color="tab:green", linestyle="--", label="ground")
    axes[0].axhline(box_height, color="tab:orange", linestyle="--", label="box top")
    axes[0].set_ylabel("lowest foot Y (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, roots["baseline"], color="black", label="baseline root Y")
    axes[1].plot(time, roots["p2_y"], color="tab:blue", label="P2-Y root Y")
    axes[1].plot(time, correction, color="tab:red", alpha=0.8, label="applied Y offset")
    axes[1].set_ylabel("root / correction (m)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    for index, label in enumerate(("L ankle", "L foot", "R ankle", "R foot")):
        axes[2].plot(time, contact[:, index], linewidth=0.9, label=label)
    axes[2].set_ylabel("static probability")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(ncol=4)
    axes[2].grid(alpha=0.25)
    for axis in axes:
        for start, end in WINDOWS.values():
            axis.axvspan(start / fps, end / fps, color="gray", alpha=0.06)
    figure.tight_layout()
    figure.savefig(output_dir / "p2_y_curves.png", dpi=160)
    plt.close(figure)

    with (output_dir / "p2_y_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "time_s",
                "baseline_root_y_m",
                "p2_y_root_y_m",
                "root_y_offset_m",
                "baseline_foot_min_y_m",
                "p2_y_foot_min_y_m",
                "contact_mean",
            ]
        )
        for index in frame:
            writer.writerow(
                [
                    int(index),
                    float(time[index]),
                    float(roots["baseline"][index]),
                    float(roots["p2_y"][index]),
                    float(correction[index]),
                    float(curves["baseline"][index]),
                    float(curves["p2_y"][index]),
                    float(contact[index].mean()),
                ]
            )


def save_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    baseline = payload["baseline"]
    enhanced = payload["p2_y"]
    transition = payload["transition_diagnostics"]
    decision = enhanced["decision"]
    report = f"""# Human3R P2-Y 单视频离线实验

## 结论

- 判定：`{decision}`。
- Human3R 检出的箱顶相对地面高度：{payload['box_height_m']:.4f} m。
- P2-Y 只替换 `smpl_params_global.transl[:, 1]`；X/Z、姿态、朝向、betas、incam、相机内参和 `net_outputs` 均精确不变。
- 本结果只通过当前 1060 帧爬箱视频的离线保护条件，尚未证明对其他场景具有泛化能力，也未接入 Web。

## 指标

| 指标 | GVHMR + FootMR | Human3R P2-Y | 保护条件 |
| --- | ---: | ---: | ---: |
| 回地高度残差 | {baseline['floor_return_cm']:.4f} cm | {enhanced['floor_return_cm']:.6f} cm | < 5 cm |
| 箱顶高度误差 | {baseline['top_height_error_cm']:.4f} cm | {enhanced['top_height_error_cm']:.6f} cm | < 5 cm |
| root 最大步长 | {baseline['root_step_max_cm_per_frame']:.4f} cm/frame | {enhanced['root_step_max_cm_per_frame']:.4f} cm/frame | <= {payload['root_step_limit_cm_per_frame']:.4f} |
| root 加速度 P95 | {baseline['root_accel_p95_m_per_s2']:.4f} m/s^2 | {enhanced['root_accel_p95_m_per_s2']:.4f} m/s^2 | <= {payload['root_accel_limit_m_per_s2']:.4f} |
| 接触足速度 P95 | {baseline['contact_speed_p95_mm_per_frame']:.4f} mm/frame | {enhanced['contact_speed_p95_mm_per_frame']:.4f} mm/frame | 诊断项 |

## 过渡区间

- 上箱平滑区间：{transition['ascent_start']}–{transition['ascent_end']}。
- 足高曲线原始下箱区间：{transition['curve_descent_start']}–{transition['curve_descent_end']}。
- 结合四个足部静态概率后的下箱区间：{transition['descent_start']}–{transition['descent_end']}。
- 接触阈值：{transition['contact_threshold']:.2f}；最短下降平滑区间：{transition['minimum_descent_frames']} 帧。

## 产物

- `p2_y_hmr4d_results.pt`：保持原 tensor schema 的候选结果。
- `metrics.json`：完整指标、窗口、过渡诊断和不变量检查。
- `p2_y_curves.csv` / `p2_y_curves.png`：逐帧曲线。
- `baseline_vs_p2_y.mp4`：全局轨迹并排视频；渲染器只显示统一地面，不显示重建箱体几何。

## 限制

- 三个稳定窗口来自该爬箱片段，当前脚本是 benchmark，不是通用在线算法。
- 绝对高度依赖 Human3R 单目重建的尺度与平面检测；箱体遮挡、弱纹理或非静态相机会改变可靠性。
- 目前只修正 Y，不能解决 GVHMR 的 X/Z 漂移。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def render_comparison(
    baseline: Mapping[str, Any],
    enhanced: Mapping[str, Any],
    output_path: Path,
    *,
    baseline_label: str = "GVHMR + FootMR",
    enhanced_name: str = "p2_y",
    enhanced_label: str = "Human3R P2-Y (root Y only)",
) -> None:
    for alias, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)

    from einops import einsum
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.vis.renderer import (
        Renderer,
        get_global_cameras_static,
        get_ground_params_from_points,
    )
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces = make_smplx("smpl").faces
    joint_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    meshes = {}
    for name, result in (("baseline", baseline), (enhanced_name, enhanced)):
        output = smplx(**to_cuda(result["smpl_params_global"]))
        meshes[name] = torch.stack(
            [torch.matmul(smplx2smpl, vertices) for vertices in output.vertices]
        ).cpu()
        del output

    reference = meshes["baseline"].clone()
    offset = einsum(joint_regressor.cpu(), reference[0], "j v, v i -> j i")[0]
    offset[1] = reference[:, :, 1].min()
    reference = reference - offset
    transform = compute_T_ayfz2ay(
        einsum(joint_regressor.cpu(), reference[[0]], "j v, l v i -> l j i"), inverse=True
    )
    for name in meshes:
        meshes[name] = apply_T_on_points(meshes[name] - offset, transform)

    joints = einsum(joint_regressor.cpu(), meshes["baseline"], "j v, l v i -> l j i")
    camera_r, camera_t, lights = get_global_cameras_static(
        meshes["baseline"], beta=2.0, cam_height_degree=20, target_center_height=1.0
    )
    scale, center_x, center_z = get_ground_params_from_points(joints[:, 0], meshes["baseline"])
    panel_width, panel_height = 640, 360
    _, _, intrinsics = create_camera_sensor(panel_width, panel_height, 24)
    renderer = Renderer(panel_width, panel_height, device="cuda", faces=faces, K=intrinsics)
    renderer.set_ground(scale * 1.5, center_x, center_z)
    color = torch.ones(3, device="cuda") * 0.8
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (panel_width * 2, panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    labels = {"baseline": baseline_label, enhanced_name: enhanced_label}
    for frame in range(len(reference)):
        panels = []
        for name in ("baseline", enhanced_name):
            cameras = renderer.create_camera(camera_r[frame], camera_t[frame])
            image = renderer.render_with_ground(
                meshes[name][[frame]].cuda(), color[None], cameras, lights
            )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image,
                labels[name],
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                2,
            )
            panels.append(image)
        writer.write(np.hstack(panels))
    writer.release()


def main() -> None:
    args = parse_args()
    for path in (args.gvhmr_result, args.scene_planes, args.video):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = torch.load(args.gvhmr_result, map_location="cpu")
    scene = json.loads(args.scene_planes.read_text(encoding="utf-8"))
    box_height = float(scene["selected_pair"]["separation_m"])
    length = baseline["smpl_params_global"]["transl"].shape[0]
    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if video_frames != length:
        raise RuntimeError(f"Video/result length mismatch: {video_frames} vs {length}")

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23

    landmark_model = SmplxLiteV437Coco23().eval()
    baseline_feet = foot_landmarks(baseline, landmark_model)
    baseline_curve = foot_height_curve(baseline_feet)
    contact = baseline["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid().cpu().numpy()
    correction, diagnostics = build_offset(baseline_curve, contact, box_height)
    baseline_root_y = baseline["smpl_params_global"]["transl"][:, 1].cpu().numpy()
    corrected_root_y = baseline_root_y + correction
    enhanced = clone_with_root_y(baseline, corrected_root_y)

    enhanced_feet = foot_landmarks(enhanced, landmark_model)
    baseline_metrics, baseline_curve = variant_metrics(
        baseline_feet, baseline_root_y, contact, fps
    )
    enhanced_metrics, enhanced_curve = variant_metrics(
        enhanced_feet, corrected_root_y, contact, fps
    )
    for metrics in (baseline_metrics, enhanced_metrics):
        metrics["top_height_error_cm"] = abs(metrics["top_relative_cm"] / 100.0 - box_height) * 100.0
    step_limit = max(3.0, baseline_metrics["root_step_max_cm_per_frame"] * 1.5)
    acceleration_limit = baseline_metrics["root_accel_p95_m_per_s2"] * 1.25
    decision = (
        "pass"
        if enhanced_metrics["floor_return_cm"] < 5.0
        and enhanced_metrics["top_height_error_cm"] < 5.0
        and enhanced_metrics["root_step_max_cm_per_frame"] <= step_limit
        and enhanced_metrics["root_accel_p95_m_per_s2"] <= acceleration_limit
        else "guardrail_failed"
    )
    enhanced_metrics["decision"] = decision

    source_translation = baseline["smpl_params_global"]["transl"]
    target_translation = enhanced["smpl_params_global"]["transl"]
    invariants = {
        "top_level_keys_equal": baseline.keys() == enhanced.keys(),
        "global_param_keys_equal": (
            baseline["smpl_params_global"].keys()
            == enhanced["smpl_params_global"].keys()
        ),
        "incam_param_keys_equal": (
            baseline["smpl_params_incam"].keys()
            == enhanced["smpl_params_incam"].keys()
        ),
        "root_x_equal": bool(torch.equal(source_translation[:, 0], target_translation[:, 0])),
        "root_z_equal": bool(torch.equal(source_translation[:, 2], target_translation[:, 2])),
        "body_pose_equal": bool(
            torch.equal(
                baseline["smpl_params_global"]["body_pose"],
                enhanced["smpl_params_global"]["body_pose"],
            )
        ),
        "global_orient_equal": bool(
            torch.equal(
                baseline["smpl_params_global"]["global_orient"],
                enhanced["smpl_params_global"]["global_orient"],
            )
        ),
        "betas_equal": bool(
            torch.equal(
                baseline["smpl_params_global"]["betas"],
                enhanced["smpl_params_global"]["betas"],
            )
        ),
        "incam_equal": all(
            torch.equal(baseline["smpl_params_incam"][key], enhanced["smpl_params_incam"][key])
            for key in baseline["smpl_params_incam"]
        ),
        "K_fullimg_equal": tree_equal(baseline["K_fullimg"], enhanced["K_fullimg"]),
        "net_outputs_equal": tree_equal(baseline["net_outputs"], enhanced["net_outputs"]),
        "finite_root_y": bool(np.isfinite(corrected_root_y).all()),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"P2-Y invariant failed: {invariants}")

    torch.save(enhanced, args.output_dir / "p2_y_hmr4d_results.pt")
    payload = {
        "method": "Human3R P2-Y",
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "scene_planes": str(args.scene_planes.resolve()),
        "video": str(args.video.resolve()),
        "fps": fps,
        "frames": length,
        "box_height_m": box_height,
        "windows": {name: list(window) for name, window in WINDOWS.items()},
        "transition_diagnostics": diagnostics,
        "invariants": invariants,
        "baseline": baseline_metrics,
        "p2_y": enhanced_metrics,
        "root_step_limit_cm_per_frame": step_limit,
        "root_accel_limit_m_per_s2": acceleration_limit,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_report(args.output_dir, payload)
    save_diagnostics(
        args.output_dir,
        {"baseline": baseline_curve, "p2_y": enhanced_curve},
        {"baseline": baseline_root_y, "p2_y": corrected_root_y},
        correction,
        contact,
        fps,
        box_height,
    )
    if not args.skip_video:
        render_comparison(baseline, enhanced, args.output_dir / "baseline_vs_p2_y.mp4")
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
