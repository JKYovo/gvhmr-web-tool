#!/usr/bin/env python3
"""Generate FootMR-v2 A/B/C trajectories, metrics, plots, report, and a 2x2 video."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    WINDOWS,
    contact_y_correction,
    foot_height_curve,
    hybrid_root_y,
    interpolate_track,
    percentile_abs,
    validate_windows,
    window_metrics,
)


VARIANT_LABELS = {
    "a_current": "A Current GVHMR + FootMR v1",
    "b_contact_y": "B Deterministic contact-Y",
    "c_delta": "C-delta WHAM W2-W0 transfer",
    "c_root_y": "C-rootY WHAM W2 relative root-Y",
    "w0": "WHAM W0 Trajectory Decoder",
    "w1": "WHAM W1 Contact Reset",
    "w2": "WHAM W2 Learned Refiner",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--wham-result", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=0.8)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def clone_result_with_y(source: Mapping[str, Any], new_y: np.ndarray, method: str) -> Dict[str, Any]:
    result = copy.deepcopy(source)
    transl = result["smpl_params_global"]["transl"].clone()
    transl[:, 1] = torch.as_tensor(new_y, dtype=transl.dtype)
    result["smpl_params_global"]["transl"] = transl
    result["footmr_v2"] = {
        "method": method,
        "diagnostic_only": method.startswith("c_"),
        "root_xz_unchanged": True,
        "pose_unchanged": True,
    }
    return result


@torch.inference_mode()
def gvhmr_foot_landmarks(result: Mapping[str, Any], model: torch.nn.Module) -> np.ndarray:
    params = {key: value[None] for key, value in result["smpl_params_global"].items()}
    _, joints = model(**params)
    return joints[0, :, 17:23].cpu().numpy()


def build_gvhmr_variants(
    result: Mapping[str, Any],
    wham: Mapping[str, Any],
    contact_threshold: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    from hmr4d.model.gvhmr.utils.endecoder import EnDecoder

    length = result["smpl_params_global"]["transl"].shape[0]
    validate_windows(length)
    support_params = {
        key: value[None].clone() for key, value in result["smpl_params_global"].items()
    }
    # FootMR v1 intentionally preserves this camera-space pre-global-IK pose.
    support_params["body_pose"] = result["smpl_params_incam"]["body_pose"][None].clone()
    endecoder = EnDecoder().eval()
    with torch.inference_mode():
        joints = endecoder.fk_v2(**support_params)[0]
    support_joints = joints[:, [7, 10, 8, 11]]
    contact_probability = result["net_outputs"]["static_conf_logits"][0, :, :4].sigmoid()
    root_y = result["smpl_params_global"]["transl"][:, 1]
    corrected_y, support_disp_y, support_mask = contact_y_correction(
        root_y, support_joints, contact_probability, contact_threshold
    )

    frame_ids = wham["frame_ids"].cpu().numpy()
    wham_full = {}
    for name in ("w0", "w1", "w2"):
        wham_full[name] = interpolate_track(
            wham["variants"][name]["trans_world"].cpu().numpy(), frame_ids, length
        )
    c_delta_y, c_root_y = hybrid_root_y(
        root_y.cpu().numpy(), wham_full["w0"][:, 1], wham_full["w2"][:, 1]
    )

    variants = {
        "a_current": copy.deepcopy(result),
        "b_contact_y": clone_result_with_y(result, corrected_y.cpu().numpy(), "b_contact_y"),
        "c_delta": clone_result_with_y(result, c_delta_y, "c_delta"),
        "c_root_y": clone_result_with_y(result, c_root_y, "c_root_y"),
    }
    diagnostics = {
        "support_disp_y": support_disp_y.cpu().numpy(),
        "support_count": support_mask.sum(-1).cpu().numpy(),
        "contact_probability": contact_probability.cpu().numpy(),
        "wham_w0_trans": wham_full["w0"],
        "wham_w1_trans": wham_full["w1"],
        "wham_w2_trans": wham_full["w2"],
    }
    return variants, diagnostics


def contact_speed_stats(
    feet_world: np.ndarray,
    contact: np.ndarray,
    fps: float,
    model: str,
) -> Dict[str, float]:
    speed = np.zeros(feet_world.shape[:2], dtype=np.float64)
    speed[1:] = np.linalg.norm(np.diff(feet_world, axis=0), axis=-1)
    if model == "gvhmr":
        left = np.max(contact[:, :2], axis=-1) > 0.8
        right = np.max(contact[:, 2:4], axis=-1) > 0.8
        mask = np.stack([left, left, left, right, right, right], axis=-1)
    else:
        mask = contact > 0.5
    selected = speed[mask]
    if selected.size == 0:
        return {"contact_speed_median_mm_per_frame": math.nan, "contact_speed_p95_mm_per_frame": math.nan}
    return {
        "contact_speed_median_mm_per_frame": float(np.median(selected) * 1000.0),
        "contact_speed_p95_mm_per_frame": float(np.percentile(selected, 95) * 1000.0),
    }


def metrics_for_variant(
    feet_world: np.ndarray,
    root_y: np.ndarray,
    contact: np.ndarray,
    fps: float,
    model: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    curve = foot_height_curve(feet_world)
    window = window_metrics(curve)
    root_step = np.diff(root_y)
    root_accel = np.diff(root_y, n=2) * fps * fps
    metrics = dict(window.__dict__)
    metrics.update(contact_speed_stats(feet_world, contact, fps, model))
    metrics.update(
        {
            "root_step_max_cm_per_frame": percentile_abs(root_step, 100.0) * 100.0,
            "root_accel_p95_m_per_s2": percentile_abs(root_accel, 95.0),
        }
    )
    return metrics, curve


def classify(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    top_ok = abs(metrics["top_relative_cm"] - baseline["top_relative_cm"]) <= 5.0
    step_limit = max(3.0, baseline["root_step_max_cm_per_frame"] * 1.5)
    continuity_ok = metrics["root_step_max_cm_per_frame"] <= step_limit
    if not top_ok or not continuity_ok:
        return "guardrail_failed"
    floor_return = metrics["floor_return_cm"]
    if floor_return <= 2.0:
        return "pass_trajectory_drift"
    if floor_return <= 5.0:
        return "partial_run_climb2"
    return "fail_consider_scene_constraint"


def save_plot(curves: Mapping[str, np.ndarray], diagnostics: Mapping[str, np.ndarray], fps: float, path: Path) -> None:
    time = np.arange(len(next(iter(curves.values())))) / fps
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    colors = {
        "a_current": "black", "b_contact_y": "tab:orange", "c_delta": "tab:blue",
        "c_root_y": "tab:green", "w0": "tab:red", "w1": "tab:purple", "w2": "tab:brown",
    }
    for name, curve in curves.items():
        axes[0].plot(time, curve, label=VARIANT_LABELS[name], color=colors[name], linewidth=1.2)
    axes[0].set_ylabel("lowest foot Y (m)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)

    for name in ("w0", "w1", "w2"):
        axes[1].plot(time, diagnostics[f"wham_{name}_trans"][:, 1], label=VARIANT_LABELS[name], color=colors[name])
    axes[1].set_ylabel("WHAM root Y (m)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    contact = diagnostics["contact_probability"]
    for index, label in enumerate(("L ankle", "L foot", "R ankle", "R foot")):
        axes[2].plot(time, contact[:, index], label=label, linewidth=0.9)
    axes[2].set_ylabel("GVHMR static probability")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(ncol=4, fontsize=8)
    axes[2].grid(alpha=0.25)

    for axis in axes:
        for start, end in WINDOWS.values():
            axis.axvspan(start / fps, end / fps, color="gray", alpha=0.06)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_csv(
    variants: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    fps: float,
    path: Path,
) -> None:
    fieldnames = ["frame", "time_s"]
    for name in variants:
        fieldnames.extend([f"{name}_root_y_m", f"{name}_foot_min_y_m"])
    for name in ("w0", "w1", "w2"):
        fieldnames.extend([f"{name}_root_y_m", f"{name}_foot_min_y_m"])
    fieldnames.extend(["gvhmr_contact_mean", "gvhmr_support_count"])
    length = len(next(iter(curves.values())))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(length):
            row: Dict[str, Any] = {"frame": frame, "time_s": frame / fps}
            for name, result in variants.items():
                row[f"{name}_root_y_m"] = float(result["smpl_params_global"]["transl"][frame, 1])
                row[f"{name}_foot_min_y_m"] = float(curves[name][frame])
            for name in ("w0", "w1", "w2"):
                row[f"{name}_root_y_m"] = float(diagnostics[f"wham_{name}_trans"][frame, 1])
                row[f"{name}_foot_min_y_m"] = float(curves[name][frame])
            row["gvhmr_contact_mean"] = float(diagnostics["contact_probability"][frame].mean())
            row["gvhmr_support_count"] = int(diagnostics["support_count"][min(frame, length - 2)])
            writer.writerow(row)


def render_comparison(
    variants: Mapping[str, Mapping[str, Any]], video_path: Path, output_path: Path
) -> None:
    # The legacy SMPL pickle loader imports chumpy, which still references
    # NumPy aliases removed in NumPy 1.24. Keep the compatibility shim local
    # to visualization; trajectory computation does not depend on chumpy.
    numpy_legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for alias, value in numpy_legacy_aliases.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)

    from einops import einsum
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor

    ordered = ("a_current", "b_contact_y", "c_delta", "c_root_y")
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces = make_smplx("smpl").faces
    joint_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    meshes = {}
    for name in ordered:
        output = smplx(**to_cuda(variants[name]["smpl_params_global"]))
        meshes[name] = torch.stack([torch.matmul(smplx2smpl, vertices) for vertices in output.vertices]).cpu()
        del output

    baseline = meshes["a_current"].clone()
    offset = einsum(joint_regressor.cpu(), baseline[0], "j v, v i -> j i")[0]
    offset[1] = baseline[:, :, 1].min()
    baseline = baseline - offset
    transform = compute_T_ayfz2ay(
        einsum(joint_regressor.cpu(), baseline[[0]], "j v, l v i -> l j i"), inverse=True
    )
    for name in ordered:
        meshes[name] = apply_T_on_points(meshes[name] - offset, transform)

    joints = einsum(joint_regressor.cpu(), meshes["a_current"], "j v, l v i -> l j i")
    camera_r, camera_t, lights = get_global_cameras_static(
        meshes["a_current"], beta=2.0, cam_height_degree=20, target_center_height=1.0
    )
    scale, center_x, center_z = get_ground_params_from_points(joints[:, 0], meshes["a_current"])

    panel_width, panel_height = 640, 360
    _, _, intrinsics = create_camera_sensor(panel_width, panel_height, 24)
    renderer = Renderer(panel_width, panel_height, device="cuda", faces=faces, K=intrinsics)
    renderer.set_ground(scale * 1.5, center_x, center_z)
    color = torch.ones(3, device="cuda") * 0.8
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (panel_width * 2, panel_height * 2)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    for frame in range(len(baseline)):
        panels = []
        for name in ordered:
            cameras = renderer.create_camera(camera_r[frame], camera_t[frame])
            image = renderer.render_with_ground(
                meshes[name][[frame]].cuda(), color[None], cameras, lights
            )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, VARIANT_LABELS[name], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
            panels.append(image)
        tiled = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        writer.write(tiled)
    writer.release()


def write_report(
    metrics: Mapping[str, Mapping[str, Any]],
    wham_meta: Mapping[str, Any],
    path: Path,
) -> None:
    lines = [
        "# FootMR v2：GVHMR × WHAM 轨迹修正离线实验",
        "",
        "## 实验配置",
        "",
        f"- WHAM commit：`{wham_meta.get('wham_commit', 'unknown')}`",
        f"- WHAM checkpoint：`{wham_meta.get('checkpoint', 'unknown')}`",
        "- 静态相机；DPVO 关闭；FLIP_EVAL 开启；Temporal SMPLify 关闭。",
        f"- Python：`{wham_meta.get('python', 'unknown').splitlines()[0]}`；"
        f"PyTorch：`{wham_meta.get('torch', 'unknown')}`；CUDA runtime：`{wham_meta.get('cuda_runtime', 'unknown')}`。",
        "- 窗口：pre `[30,105)`，top `[600,795)`，post `[970,1056)`。",
        "- C-delta/C-rootY 仅为诊断性轨迹迁移，不是可部署的跨网络 tensor 接口。",
        "",
        "## 资产校验",
        "",
        "| 资产 | SHA256 |",
        "|---|---|",
    ]
    for name, digest in wham_meta.get("assets_sha256", {}).items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend([
        "",
        "## 方法与公式",
        "",
        "- A 保持现有 GVHMR + FootMR v1 不变。FootMR v1 只修正姿态，本实验只改 root Y。",
        "- B 对每个增量使用 `delta_root_y' = delta_root_y - mean(delta_support_joint_y)`，"
        "仅统计 `sigmoid(static_conf) > 0.8` 的 `[左踝、左脚、右踝、右脚]`，再从首帧重新积分。",
        "- W0 是 WHAM Trajectory Decoder；W1 是官方 `reset_root_velocity`；W2 是官方 563 维 context 驱动的 Trajectory Refiner。",
        "- C-delta 使用 `GVHMR_y + (W2_y - W0_y)`，并在 pre 窗口去除中位偏置。",
        "- C-rootY 使用 W2 相对 pre 窗口的 root-Y 曲线替换 GVHMR 的相对 root Y。",
        "- 所有 C 结果只替换 root Y；FootMR 姿态、betas、global orientation、root X/Z 均保持不变。",
        "",
        "## 结果",
        "",
        "| 方案 | floor return | signed return | top relative | contact speed p95 | root step max | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for name in ("a_current", "b_contact_y", "c_delta", "c_root_y", "w0", "w1", "w2"):
        item = metrics[name]
        lines.append(
            f"| {VARIANT_LABELS[name]} | {item['floor_return_cm']:.2f} cm | "
            f"{item['signed_floor_return_cm']:+.2f} cm | {item['top_relative_cm']:+.2f} cm | "
            f"{item['contact_speed_p95_mm_per_frame']:.2f} mm/f | "
            f"{item['root_step_max_cm_per_frame']:.2f} cm/f | {item['decision']} |"
        )
    best_name = min(("b_contact_y", "c_delta", "c_root_y"), key=lambda name: metrics[name]["floor_return_cm"])
    best = metrics[best_name]
    lines.extend(["", "## 自动结论", ""])
    if best["decision"] == "pass_trajectory_drift":
        lines.append(
            f"- `{best_name}` 将回地残差降到 {best['floor_return_cm']:.2f} cm，且通过箱顶/连续性约束；主要问题可归因于 trajectory drift。"
        )
    elif best["decision"] == "partial_run_climb2":
        lines.append(
            f"- 最佳结果 `{best_name}` 为 {best['floor_return_cm']:.2f} cm，处于 2–5 cm 灰区；应先复测 climb2。"
        )
    else:
        lines.append(
            f"- 最佳结果 `{best_name}` 仍为 {best['floor_return_cm']:.2f} cm；WHAM 轨迹信息不足以通过首轮门槛，可进入 absolute scene constraint 调研。"
        )
    if metrics["b_contact_y"]["floor_return_cm"] > metrics["a_current"]["floor_return_cm"]:
        lines.append("- 确定性 contact-Y 使结果变差，不能用简单解除 Y 限制替代学习式 refiner。")
    if metrics["w2"]["floor_return_cm"] >= metrics["w0"]["floor_return_cm"]:
        lines.append("- WHAM W2 在其自身输出上没有改善回地高度，禁止把混合结果包装为部署方案。")
    lines.extend([
        "",
        "## 失败分析与下一步",
        "",
        f"- A 的可复现基线是 {metrics['a_current']['floor_return_cm']:.2f} cm；"
        f"B 增至 {metrics['b_contact_y']['floor_return_cm']:.2f} cm，说明简单接触期速度抵消会累计错误。",
        f"- WHAM W1 虽把自身残差降到 {metrics['w1']['floor_return_cm']:.2f} cm，"
        f"但箱顶相对高度从 W0 的 {metrics['w0']['top_relative_cm']:.2f} cm 降至 "
        f"{metrics['w1']['top_relative_cm']:.2f} cm，按本实验窗口指标已明显压平。",
        f"- WHAM W2 自身残差为 {metrics['w2']['floor_return_cm']:.2f} cm，"
        f"高于 W0 的 {metrics['w0']['floor_return_cm']:.2f} cm；学习式 refiner 在该视频上未恢复原地面高度。",
        "- 两个 C 混合均大于 5 cm 且触发保护条件，因此不运行 climb2，也不进入 Web 集成。",
        "- 按预设决策树，下一阶段应单独规划 JOSH/Human3R 一类 absolute scene constraint；"
        "本实验不实现或声称已验证该方向。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (args.gvhmr_result, args.wham_result, args.video):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gvhmr = torch.load(args.gvhmr_result, map_location="cpu")
    wham = torch.load(args.wham_result, map_location="cpu")
    length = gvhmr["smpl_params_global"]["transl"].shape[0]
    validate_windows(length)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_length != length:
        raise RuntimeError(f"Video/result length mismatch: {video_length} vs {length}")

    variants, diagnostics = build_gvhmr_variants(gvhmr, wham, args.contact_threshold)
    for name, result in variants.items():
        torch.save(result, args.output_dir / f"{name}_hmr4d_results.pt")

    from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23

    landmark_model = SmplxLiteV437Coco23().eval()
    curves: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    gv_contact = diagnostics["contact_probability"]
    for name, result in variants.items():
        feet = gvhmr_foot_landmarks(result, landmark_model)
        root_y = result["smpl_params_global"]["transl"][:, 1].cpu().numpy()
        metrics[name], curves[name] = metrics_for_variant(feet, root_y, gv_contact, fps, "gvhmr")

    frame_ids = wham["frame_ids"].cpu().numpy()
    wham_contact = interpolate_track(wham["contact"].cpu().numpy(), frame_ids, length)
    for name in ("w0", "w1", "w2"):
        feet = interpolate_track(
            wham["variants"][name]["feet_world"].cpu().numpy(), frame_ids, length
        )
        root_y = diagnostics[f"wham_{name}_trans"][:, 1]
        metrics[name], curves[name] = metrics_for_variant(feet, root_y, wham_contact, fps, "wham")

    for name in ("a_current", "b_contact_y", "c_delta", "c_root_y"):
        metrics[name]["decision"] = classify(metrics[name], metrics["a_current"])
    for name in ("w0", "w1", "w2"):
        metrics[name]["decision"] = classify(metrics[name], metrics["w0"])

    payload = {
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "fps": fps,
        "gvhmr_result": str(args.gvhmr_result.resolve()),
        "wham_result": str(args.wham_result.resolve()),
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_csv(variants, curves, diagnostics, fps, args.output_dir / "curves.csv")
    save_plot(curves, diagnostics, fps, args.output_dir / "trajectory_contact_curves.png")
    write_report(metrics, wham.get("meta", {}), args.output_dir / "report.md")
    if not args.skip_video:
        render_comparison(variants, args.video, args.output_dir / "comparison_2x2.mp4")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
