import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


RESULT_PREFIX = "__GVHMR_CORE_RESULT__="
CRF = 23


def _ensure_chumpy_numpy_compat():
    import numpy as np

    # Chumpy 0.70 imports aliases removed from recent NumPy releases.
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _prepare_core(core_root):
    core_root = Path(core_root).expanduser().resolve()
    if not (core_root / "hmr4d" / "__init__.py").is_file():
        raise FileNotFoundError(f"GVHMR core package not found: {core_root / 'hmr4d'}")
    sys.path.insert(0, str(core_root))
    os.chdir(core_root)
    import hmr4d

    imported_root = Path(hmr4d.__file__).resolve().parents[1]
    if imported_root != core_root:
        raise RuntimeError(f"Imported hmr4d from {imported_root}, expected {core_root}")
    return core_root


def _build_cfg(output_dir, *, static_cam, f_mm=None, use_dpvo=False, verbose=False):
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            "video_name=external_core_job",
            f"static_cam={str(bool(static_cam)).lower()}",
            f"use_dpvo={str(bool(use_dpvo)).lower()}",
            f"verbose={str(bool(verbose)).lower()}",
        ]
        if f_mm not in (None, "", 0):
            overrides.append(f"f_mm={int(f_mm)}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    preprocess_dir = output_dir / "preprocess"
    cfg.output_dir = str(output_dir)
    cfg.preprocess_dir = str(preprocess_dir)
    cfg.video_path = str(output_dir / "0_input_video.mp4")
    cfg.paths.bbx = str(preprocess_dir / "bbx.pt")
    cfg.paths.bbx_xyxy_video_overlay = str(preprocess_dir / "bbx_xyxy_video_overlay.mp4")
    cfg.paths.vit_features = str(preprocess_dir / "vit_features.pt")
    cfg.paths.vitpose = str(preprocess_dir / "vitpose.pt")
    cfg.paths.vitpose_video_overlay = str(preprocess_dir / "vitpose_video_overlay.mp4")
    cfg.paths.slam = str(preprocess_dir / "slam_results.pt")
    cfg.paths.hmr4d_results = str(output_dir / "hmr4d_results.pt")
    cfg.paths.incam_video = str(output_dir / "1_incam.mp4")
    cfg.paths.global_video = str(output_dir / "2_global.mp4")
    cfg.paths.incam_global_horiz_video = str(output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4")
    cfg.ckpt_path = str(Path("inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"))
    return cfg


def _prepare_video_copy(source_path, destination_path):
    from hmr4d.utils.video_io_utils import normalize_video_fps

    source_path = Path(source_path).expanduser().resolve()
    destination_path = Path(destination_path).expanduser().resolve()
    if source_path == destination_path:
        return destination_path
    normalize_video_fps(source_path, destination_path, target_fps=30, crf=CRF)
    return destination_path


def _cleanup_preprocess(preprocess_dir):
    preprocess_dir = Path(preprocess_dir)
    bbx_path = preprocess_dir / "bbx.pt"
    preserved_bbx = bbx_path.read_bytes() if bbx_path.is_file() else None
    if preprocess_dir.exists():
        shutil.rmtree(preprocess_dir)
    if preserved_bbx is not None:
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        bbx_path.write_bytes(preserved_bbx)


def _apply_ground_constraint(core_root, output_dir, video_path, result_path, mode):
    """Apply an optional core post-process while preserving the raw tensor."""
    output_dir = Path(output_dir)
    result_path = Path(result_path)
    if mode == "none":
        return {
            "ground_constraint": "none",
            "ground_constraint_status": "not_requested",
        }
    if mode == "human3r":
        raise ValueError("Human3R scene constraint is present in the UI but is not enabled yet.")
    if mode != "flat_y":
        raise ValueError(f"Unsupported ground constraint: {mode}")

    raw_path = output_dir / "hmr4d_results_raw.pt"
    if not raw_path.is_file():
        shutil.copy2(result_path, raw_path)

    constraint_dir = output_dir / "ground_constraint_global_v1_1"
    constraint_dir.mkdir(parents=True, exist_ok=True)
    enhanced_path = constraint_dir / "contact_global_root_hmr4d_results.pt"
    metrics_path = constraint_dir / "metrics.json"
    script = core_root / "tools" / "bench" / "human3r_p2y" / "apply_contact_global_root.py"
    if not script.is_file():
        raise FileNotFoundError(f"Contact global V1.1 postprocessor not found: {script}")

    error = None
    if not enhanced_path.is_file():
        command = [
            sys.executable,
            str(script),
            "--gvhmr-result",
            str(raw_path),
            "--video",
            str(video_path),
            "--output-dir",
            str(constraint_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=core_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            error = completed.stdout.strip() or f"exit code {completed.returncode}"

    decision = None
    metrics = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            decision = metrics.get("decision")
        except (OSError, ValueError):
            decision = None

    if error is None and decision == "diagnostic_pass" and enhanced_path.is_file():
        shutil.copy2(enhanced_path, result_path)
        print("[Ground Constraint] Contact global V1.1 applied; raw tensor preserved.", flush=True)
        return {
            "ground_constraint": "flat_y",
            "ground_constraint_status": "applied",
            "raw_hmr4d_results_path": str(raw_path.resolve()),
            "global_contact_results_path": str(enhanced_path.resolve()),
            "ground_constraint_metrics_path": str(metrics_path.resolve()),
        }

    shutil.copy2(raw_path, result_path)
    reason = error or _ground_constraint_fallback_reason(metrics, decision)
    print(f"[Ground Constraint] Contact global V1.1 fallback to raw result: {reason}", flush=True)
    payload = {
        "ground_constraint": "flat_y",
        "ground_constraint_status": "fallback",
        "ground_constraint_error": reason,
        "ground_constraint_fallback_reason": reason,
        "raw_hmr4d_results_path": str(raw_path.resolve()),
    }
    if metrics_path.is_file():
        payload["ground_constraint_metrics_path"] = str(metrics_path.resolve())
    return payload


def _ground_constraint_fallback_reason(metrics, decision):
    if decision != "guardrail_failed" or not isinstance(metrics, dict):
        return f"保护条件结果：{decision or '指标缺失或损坏'}"
    failed = metrics.get("failed_guardrails")
    if not isinstance(failed, list):
        failed = [
            key
            for key, passed in metrics.get("guardrails", {}).items()
            if passed is False
        ]
    details = metrics.get("guardrail_details", {})
    labels = {
        "horizontal_correction_pass": "水平修正过大",
        "vertical_correction_pass": "垂直修正过大",
        "root_step_pass": "root 单帧跳变过大",
        "root_acceleration_pass": "root 加速度增幅过大",
        "slip_improved": "脚滑指标未同时改善",
        "height_improved": "悬空/穿地指标未同时改善",
    }

    def number(values, key, unit=""):
        value = values.get(key) if isinstance(values, dict) else None
        return f"{value:.4g}{unit}" if isinstance(value, (int, float)) else "?"

    reasons = []
    for key in failed:
        values = details.get(key, {}) if isinstance(details, dict) else {}
        if key == "horizontal_correction_pass":
            suffix = f"（实际 {number(values, 'actual_m', 'm')}，上限 {number(values, 'limit_m', 'm')}）"
        elif key == "vertical_correction_pass":
            suffix = f"（实际 {number(values, 'actual_m', 'm')}，上限 {number(values, 'limit_m', 'm')}）"
        elif key == "root_step_pass":
            suffix = f"（实际 {number(values, 'actual_cm_per_frame', 'cm/帧')}，上限 {number(values, 'limit_cm_per_frame', 'cm/帧')}）"
        elif key == "root_acceleration_pass":
            suffix = f"（P95 实际 {number(values, 'actual_m_per_s2_p95', 'm/s²')}，上限 {number(values, 'limit_m_per_s2_p95', 'm/s²')}）"
        elif key == "slip_improved":
            suffix = (
                f"（接触速度 P95 {number(values, 'anchor_speed_before_mm_per_frame_p95')}→"
                f"{number(values, 'anchor_speed_after_mm_per_frame_p95')}mm/帧；段漂移 P95 "
                f"{number(values, 'endpoint_drift_before_cm_p95')}→"
                f"{number(values, 'endpoint_drift_after_cm_p95')}cm）"
            )
        elif key == "height_improved":
            suffix = (
                f"（支撑高度 P95 {number(values, 'support_height_before_cm_p95')}→"
                f"{number(values, 'support_height_after_cm_p95')}cm；悬空 "
                f"{number(values, 'hover_before_pct')}→{number(values, 'hover_after_pct')}%；穿地 "
                f"{number(values, 'penetration_before_pct')}→{number(values, 'penetration_after_pct')}%）"
            )
        else:
            suffix = ""
        reasons.append(f"{labels.get(key, key)}{suffix}")
    return "；".join(reasons) if reasons else "保护条件未通过（未记录具体失败项）"


def _process(args):
    import hydra
    import torch
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.pylogger import Log
    from tools.demo.demo import load_data_dict, run_preprocess

    cfg = _build_cfg(
        args.output_dir,
        static_cam=args.static_cam,
        f_mm=args.f_mm,
        use_dpvo=args.use_dpvo,
        verbose=args.verbose,
    )
    output_dir = Path(cfg.output_dir)
    preprocess_dir = Path(cfg.preprocess_dir)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    # The enhanced core normalizes the submitted source to 30 FPS from its
    # timestamps before any preprocessing. Passing the source explicitly
    # avoids relabeling every decoded frame and slowing down 60 FPS videos.
    cfg.source_video_path = str(Path(args.video).expanduser().resolve())

    result_path = Path(cfg.paths.hmr4d_results)
    if not result_path.is_file():
        run_preprocess(cfg)
        data = load_data_dict(cfg)
        Log.info("[HMR4D] Predicting with external core")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        prediction = detach_to_cpu(
            model.predict(
                data,
                static_cam=cfg.static_cam,
                no_postproc=cfg.no_postproc,
            )
        )
        torch.save(prediction, result_path)
    else:
        Log.info(f"[HMR4D] Reusing cached result at {result_path}")

    ground_result = _apply_ground_constraint(
        Path(args.core_root).expanduser().resolve(),
        output_dir,
        cfg.video_path,
        result_path,
        args.ground_constraint,
    )

    if not args.save_intermediate:
        _cleanup_preprocess(preprocess_dir)

    return {
        "output_dir": str(output_dir),
        "input_video_path": str(Path(cfg.video_path).resolve()),
        "hmr4d_results_path": str(result_path.resolve()),
        **ground_result,
    }


def _merge_preview_videos(merge_func, input_paths, output_path):
    merge_func([str(path) for path in input_paths], str(output_path))


def _preview(args):
    _ensure_chumpy_numpy_compat()
    import torch
    from hmr4d.utils.video_io_utils import get_video_lwh, merge_videos_horizontal
    from tools.demo.demo import render_global, render_incam

    cfg = _build_cfg(args.output_dir, static_cam=True)
    output_dir = Path(cfg.output_dir)
    result_path = Path(cfg.paths.hmr4d_results)
    video_path = Path(cfg.video_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing inference result: {result_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing processed video: {video_path}")

    bbx_path = Path(cfg.paths.bbx)
    if not bbx_path.is_file():
        bbx_path.parent.mkdir(parents=True, exist_ok=True)
        length = get_video_lwh(video_path)[0]
        torch.save({"bbx_xys": torch.zeros((length, 3))}, bbx_path)

    render_incam(cfg)
    render_global(cfg)
    preview_path = Path(cfg.paths.incam_global_horiz_video)
    if not preview_path.is_file():
        _merge_preview_videos(
            merge_videos_horizontal,
            [cfg.paths.incam_video, cfg.paths.global_video],
            preview_path,
        )
    return {
        "incam_video_path": str(Path(cfg.paths.incam_video).resolve()),
        "global_video_path": str(Path(cfg.paths.global_video).resolve()),
        "preview_video_path": str(preview_path.resolve()),
    }


def _probe(core_root, checkpoint_root):
    import hmr4d
    import torch

    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    core_checkpoints = core_root / "inputs" / "checkpoints"
    required = (
        core_checkpoints / "gvhmr" / "gvhmr_siga24_release.ckpt",
        core_checkpoints / "hmr2" / "epoch=10-step=25000.ckpt",
        core_checkpoints / "vitpose" / "vitpose-h-multi-coco.pth",
        core_checkpoints / "yolo" / "yolov8x.pt",
        core_root / "inputs" / "footmr_assets" / "footmr_checkpoint.ckpt",
        core_root / "inputs" / "footmr_assets" / "vitpose-h-wholebody.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing external core checkpoints:\n" + "\n".join(missing))
    with tempfile.TemporaryDirectory(prefix="gvhmr-core-probe-") as temporary:
        cfg = _build_cfg(Path(temporary) / "probe", static_cam=True)
        config_output_dir = str(Path(cfg.output_dir).resolve())
    return {
        "core_root": str(core_root),
        "hmr4d_path": str(Path(hmr4d.__file__).resolve()),
        "checkpoint_root": str(checkpoint_root),
        "core_checkpoint_root": str(core_checkpoints.resolve()),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_output_dir": config_output_dir,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "process", "preview"):
        command = subparsers.add_parser(name)
        command.add_argument("--core-root", required=True)
        command.add_argument("--checkpoint-root", required=True)

    process = subparsers.choices["process"]
    process.add_argument("--video", required=True)
    process.add_argument("--output-dir", required=True)
    process.add_argument("--static-cam", action="store_true")
    process.add_argument("--f-mm", type=int, default=None)
    process.add_argument("--save-intermediate", action="store_true")
    process.add_argument(
        "--ground-constraint",
        choices=("none", "flat_y", "human3r"),
        default="none",
    )
    process.add_argument("--use-dpvo", action="store_true")
    process.add_argument("--verbose", action="store_true")

    preview = subparsers.choices["preview"]
    preview.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    core_root = _prepare_core(args.core_root)
    if args.command == "probe":
        result = _probe(core_root, args.checkpoint_root)
    elif args.command == "process":
        result = _process(args)
    else:
        result = _preview(args)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
