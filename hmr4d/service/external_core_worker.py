import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


RESULT_PREFIX = "__GVHMR_CORE_RESULT__="
CRF = 23
LONG_VIDEO_THRESHOLD_FRAMES = 1800
LONG_VIDEO_WINDOW_FRAMES = 600
LONG_VIDEO_STRIDE_FRAMES = 480


def _positive_env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _effective_checkpoint_path(configured_path, model):
    configured_path = Path(configured_path).expanduser().resolve()
    if model.pipeline.use_foot_refiner and configured_path.name == "gvhmr_siga24_release.ckpt":
        return Path("inputs/footmr_assets/footmr_checkpoint.ckpt").resolve()
    return configured_path


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
    preview_dir = output_dir / "preview"
    cfg.paths.incam_video = str(preview_dir / "incam.mp4")
    cfg.paths.global_video = str(preview_dir / "global.mp4")
    cfg.paths.incam_global_horiz_video = str(preview_dir / "comparison.mp4")
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
    if preprocess_dir.exists():
        shutil.rmtree(preprocess_dir)


def _run_stage(command, *, cwd, label, env=None):
    """Run one post-process stage while forwarding useful logs to the Web job."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    recent = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        print(f"[{label}] {line}", flush=True)
        recent.append(line)
        if len(recent) > 40:
            recent.pop(0)
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        detail = "\n".join(recent) or f"exit code {return_code}"
        raise RuntimeError(f"{label} failed:\n{detail}")


def _ground_paths(output_dir):
    diagnostics = Path(output_dir) / "diagnostics"
    constraint = diagnostics / "ground_constraint"
    return {
        "diagnostics": diagnostics,
        "raw": diagnostics / "source" / "hmr4d_results_raw.pt",
        "constraint": constraint,
        "contact": constraint / "contact_global_v1_1",
        "gravity": constraint / "gravity_alignment",
        "human3r_scene": constraint / "human3r_scene",
        "work": Path(output_dir) / ".work" / "human3r_reconstruction",
    }


def _run_contact_constraint(core_root, video_path, input_path, contact_dir):
    script = core_root / "tools" / "bench" / "human3r_p2y" / "apply_contact_global_root.py"
    if not script.is_file():
        raise FileNotFoundError(f"Contact global V1.1 postprocessor not found: {script}")
    contact_dir.mkdir(parents=True, exist_ok=True)
    enhanced_path = contact_dir / "contact_global_root_hmr4d_results.pt"
    metrics_path = contact_dir / "metrics.json"
    if not enhanced_path.is_file() or not metrics_path.is_file():
        _run_stage(
            [
                sys.executable,
                str(script),
                "--gvhmr-result",
                str(input_path),
                "--video",
                str(video_path),
                "--output-dir",
                str(contact_dir),
            ],
            cwd=core_root,
            label="Ground Constraint",
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Ground constraint metrics are missing or invalid: {exc}") from exc
    decision = metrics.get("decision")
    if decision not in {"diagnostic_pass", "guardrail_failed"} or not enhanced_path.is_file():
        raise RuntimeError(
            "Contact global V1.1 did not produce a valid constrained result: "
            f"{_ground_constraint_fallback_reason(metrics, decision)}"
        )
    warning = (
        _ground_constraint_fallback_reason(metrics, decision)
        if decision == "guardrail_failed"
        else None
    )
    return enhanced_path, metrics_path, warning


def _run_gravity_alignment(core_root, raw_path, gravity_source, gravity_dir):
    script = core_root / "tools" / "bench" / "human3r_p2y" / "apply_scene_gravity.py"
    if not script.is_file():
        raise FileNotFoundError(f"Scene gravity postprocessor not found: {script}")
    gravity_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = gravity_dir / "gravity_aligned_hmr4d_results.pt"
    metrics_path = gravity_dir / "metrics.json"
    if not aligned_path.is_file() or not metrics_path.is_file():
        _run_stage(
            [
                sys.executable,
                str(script),
                "--gvhmr-result",
                str(raw_path),
                "--ground-plane",
                str(gravity_source),
                "--output-dir",
                str(gravity_dir),
            ],
            cwd=core_root,
            label="Gravity Alignment",
        )
    return aligned_path, metrics_path


def _estimate_standing_gravity(core_root, raw_path, gravity_dir):
    script = core_root / "tools" / "bench" / "human3r_p2y" / "estimate_standing_gravity.py"
    if not script.is_file():
        raise FileNotFoundError(f"Standing gravity calibrator not found: {script}")
    gravity_dir.mkdir(parents=True, exist_ok=True)
    gravity_source = gravity_dir / "standing_gravity.json"
    if not gravity_source.is_file():
        _run_stage(
            [
                sys.executable,
                str(script),
                "--gvhmr-result",
                str(raw_path),
                "--output",
                str(gravity_source),
            ],
            cwd=core_root,
            label="Standing Gravity",
        )
    return gravity_source


def _run_human3r_ground(core_root, video_path, paths, *, save_intermediate):
    human3r_python = Path(
        os.environ.get(
            "HUMAN3R_PYTHON",
            "/home/user-kevien/miniforge3/envs/human3r/bin/python",
        )
    ).expanduser().resolve()
    model_path = Path(
        os.environ.get(
            "HUMAN3R_MODEL_PATH",
            str(core_root / "inputs" / "human3r_assets" / "human3r_672S.pth"),
        )
    ).expanduser().resolve()
    human3r_root = core_root / "third-party" / "Human3R"
    dinov2_root = core_root / "third-party" / "dinov2"
    run_script = core_root / "tools" / "bench" / "human3r_p2y" / "run_human3r_headless.py"
    extract_script = core_root / "tools" / "bench" / "human3r_p2y" / "extract_ground_plane.py"
    body_model_root = core_root / "runtime" / "checkpoints" / "body_models"
    mean_params = core_root / "hmr4d" / "network" / "hmr2" / "configs" / "smpl_mean_params.npz"
    curope_dir = human3r_root / "src" / "croco" / "models" / "curope"
    curope_extension = next(curope_dir.glob("curope*.so"), None)
    for path, label in (
        (human3r_python, "Human3R Python"),
        (model_path, "Human3R checkpoint"),
        (human3r_root / "demo.py", "Human3R submodule"),
        (dinov2_root / "hubconf.py", "DINOv2 submodule"),
        (run_script, "Human3R runner"),
        (extract_script, "Human3R ground extractor"),
        (body_model_root / "smplx" / "SMPLX_NEUTRAL.npz", "SMPL-X neutral model"),
        (mean_params, "SMPL mean parameters"),
        (curope_extension, "Human3R CUDA RoPE extension"),
    ):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    scene_dir = paths["human3r_scene"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    ground_path = scene_dir / "ground_plane.json"
    reconstruction = paths["work"]
    human3r_env = os.environ.copy()
    human3r_env["PYTHONNOUSERSITE"] = "1"
    human3r_env.pop("PYTHONPATH", None)
    if not ground_path.is_file():
        if reconstruction.exists():
            shutil.rmtree(reconstruction)
        reconstruction.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run_stage(
                [
                    str(human3r_python),
                    str(run_script),
                    "--seq_path",
                    str(video_path),
                    "--model_path",
                    str(model_path),
                    "--output_dir",
                    str(reconstruction),
                    "--human3r_root",
                    str(human3r_root),
                    "--dinov2_root",
                    str(dinov2_root),
                    "--body_model_root",
                    str(body_model_root),
                    "--mean_params",
                    str(mean_params),
                    "--size",
                    "512",
                    "--chunk_size",
                    "100",
                    "--use_ttt3r",
                ],
                cwd=core_root,
                env=human3r_env,
                label="Human3R",
            )
            _run_stage(
                [
                    str(human3r_python),
                    str(extract_script),
                    "--human3r-dir",
                    str(reconstruction),
                    "--output-dir",
                    str(scene_dir),
                ],
                cwd=core_root,
                env=human3r_env,
                label="Human3R Ground",
            )
            metadata = reconstruction / "run_metadata.json"
            if metadata.is_file():
                shutil.copy2(metadata, scene_dir / "run_metadata.json")
        finally:
            if not save_intermediate:
                shutil.rmtree(reconstruction, ignore_errors=True)
                try:
                    reconstruction.parent.rmdir()
                except OSError:
                    pass
                median_depth = scene_dir / "median_depth.npy"
                median_depth.unlink(missing_ok=True)
    return ground_path


def _apply_ground_constraint(
    core_root,
    output_dir,
    video_path,
    result_path,
    mode,
    *,
    static_cam=True,
    save_intermediate=False,
):
    """Apply an optional core post-process while preserving the raw tensor."""
    output_dir = Path(output_dir)
    result_path = Path(result_path)
    if mode == "none":
        return {
            "ground_constraint": "none",
            "ground_constraint_status": "not_requested",
        }
    if mode not in {"flat_y", "gravity_flat", "human3r"}:
        raise ValueError(f"Unsupported ground constraint: {mode}")
    if mode in {"gravity_flat", "human3r"} and not static_cam:
        raise ValueError(f"{mode} requires the static camera option")

    paths = _ground_paths(output_dir)
    raw_path = paths["raw"]
    if not raw_path.is_file():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, raw_path)

    contact_input = raw_path
    gravity_source = None
    gravity_metrics = None
    effective_mode = mode
    fallback_reason = None
    if mode == "gravity_flat":
        try:
            gravity_source = _estimate_standing_gravity(
                core_root, raw_path, paths["gravity"]
            )
            contact_input, gravity_metrics = _run_gravity_alignment(
                core_root, raw_path, gravity_source, paths["gravity"]
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            effective_mode = "flat_y"
            fallback_reason = str(exc).strip() or exc.__class__.__name__
            print(
                "[Gravity Alignment] Standing calibration unavailable; "
                f"falling back to automatic flat ground: {fallback_reason}",
                flush=True,
            )
    elif mode == "human3r":
        gravity_source = _run_human3r_ground(
            core_root,
            video_path,
            paths,
            save_intermediate=save_intermediate,
        )
        contact_input, gravity_metrics = _run_gravity_alignment(
            core_root, raw_path, gravity_source, paths["gravity"]
        )

    enhanced_path, metrics_path, warning = _run_contact_constraint(
        core_root, video_path, contact_input, paths["contact"]
    )
    shutil.copy2(enhanced_path, result_path)
    print(
        f"[Ground Constraint] {effective_mode} applied; diagnostics organized under "
        f"{paths['constraint']}",
        flush=True,
    )
    payload = {
        "ground_constraint": mode,
        "ground_constraint_effective_mode": effective_mode,
        "ground_constraint_status": "applied",
        "raw_hmr4d_results_path": str(raw_path.resolve()),
        "global_contact_results_path": str(enhanced_path.resolve()),
        "ground_constraint_metrics_path": str(metrics_path.resolve()),
    }
    if gravity_source is not None:
        payload["gravity_source_path"] = str(Path(gravity_source).resolve())
    if gravity_metrics is not None:
        payload["gravity_alignment_metrics_path"] = str(Path(gravity_metrics).resolve())
    overlay = paths["human3r_scene"] / "ground_overlay.png"
    metadata = paths["human3r_scene"] / "run_metadata.json"
    if overlay.is_file():
        payload["human3r_ground_overlay_path"] = str(overlay.resolve())
    if metadata.is_file():
        payload["human3r_run_metadata_path"] = str(metadata.resolve())
    warnings = [item for item in (fallback_reason, warning) if item]
    if warnings:
        payload["ground_constraint_warning"] = "；".join(warnings)
    if fallback_reason:
        payload["ground_constraint_fallback_reason"] = fallback_reason
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


def _apply_orientation_guard(output_dir, result_path):
    """Repair only strict, isolated root-orientation impulses."""
    import torch
    from hmr4d.utils.orientation_guard import guard_isolated_orientation_jumps

    output_dir = Path(output_dir)
    result_path = Path(result_path)
    guard_dir = output_dir / "diagnostics" / "orientation_guard"
    guard_dir.mkdir(parents=True, exist_ok=True)
    original_path = guard_dir / "original_hmr4d_results.pt"
    guarded_path = guard_dir / "orientation_guard_hmr4d_results.pt"
    metrics_path = guard_dir / "metrics.json"

    # A process retry commonly sees the already-guarded public result. Preserve
    # the original diagnostics instead of replacing them with a false
    # "not-needed" report from the idempotent second pass.
    if original_path.is_file() and guarded_path.is_file() and metrics_path.is_file():
        try:
            current = torch.load(result_path, map_location="cpu", weights_only=False)
            cached = torch.load(guarded_path, map_location="cpu", weights_only=False)
            same_orientation = all(
                torch.equal(
                    current[f"smpl_params_{space}"]["global_orient"],
                    cached[f"smpl_params_{space}"]["global_orient"],
                )
                for space in ("global", "incam")
            )
            if same_orientation:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                return {
                    "orientation_guard_status": "applied",
                    "orientation_guard_original_path": str(original_path.resolve()),
                    "orientation_guard_results_path": str(guarded_path.resolve()),
                    "orientation_guard_metrics_path": str(metrics_path.resolve()),
                    "orientation_guard_detections": int(metrics.get("num_detections", 0)),
                }
        except (KeyError, OSError, RuntimeError, ValueError):
            pass

    prediction = torch.load(result_path, map_location="cpu", weights_only=False)
    guarded, metrics = guard_isolated_orientation_jumps(prediction)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not metrics["triggered"]:
        print(
            "[Orientation Guard] No strict isolated root-orientation impulse detected.",
            flush=True,
        )
        return {
            "orientation_guard_status": "not_needed",
            "orientation_guard_metrics_path": str(metrics_path.resolve()),
            "orientation_guard_detections": 0,
        }

    shutil.copy2(result_path, original_path)
    torch.save(guarded, guarded_path)
    shutil.copy2(guarded_path, result_path)
    frames = ", ".join(
        str(item["boundary_frame"]) for item in metrics["detections"]
    )
    print(
        f"[Orientation Guard] Repaired {metrics['num_detections']} isolated impulse(s) "
        f"at frame boundaries: {frames}",
        flush=True,
    )
    return {
        "orientation_guard_status": "applied",
        "orientation_guard_original_path": str(original_path.resolve()),
        "orientation_guard_results_path": str(guarded_path.resolve()),
        "orientation_guard_metrics_path": str(metrics_path.resolve()),
        "orientation_guard_detections": int(metrics["num_detections"]),
    }


def _process(args):
    import hydra
    import torch
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.long_video import predict_long_video
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
    long_video_result = {}
    if not result_path.is_file():
        run_preprocess(cfg)
        data = load_data_dict(cfg)
        frame_count = int(data["length"])
        threshold = _positive_env_int(
            "GVHMR_LONG_VIDEO_THRESHOLD_FRAMES", LONG_VIDEO_THRESHOLD_FRAMES
        )
        is_long_video = frame_count >= threshold
        if is_long_video:
            # The trained networks already use a 120-frame local mask for any
            # sequence longer than 120 frames. The local implementation is
            # mathematically equivalent to the dense masked implementation,
            # without allocating an L x L attention score tensor.
            cfg.attention_impl = "local"
            cfg.network.attention_impl = "local"
        Log.info("[HMR4D] Predicting with external core")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        effective_checkpoint = _effective_checkpoint_path(cfg.ckpt_path, model)
        if is_long_video:
            Log.info(
                f"[Long Video] Exact full-sequence local attention; {frame_count} frames"
            )
            started = time.perf_counter()
            fallback_reason = None
            try:
                prediction = detach_to_cpu(
                    model.predict(
                        data,
                        static_cam=cfg.static_cam,
                        no_postproc=cfg.no_postproc,
                    )
                )
            except torch.cuda.OutOfMemoryError:
                # Unusually long sequences can still exceed memory in FK/IK,
                # which scale linearly even after attention is bounded. Retry
                # only that case with resumable temporal windows.
                torch.cuda.empty_cache()
                Log.warn(
                    "[Long Video] Full-sequence FK/IK exceeded CUDA memory; "
                    "falling back to resumable network windows"
                )
                fallback_reason = "cuda_out_of_memory_in_full_sequence_fk_or_ik"
                prediction = None
            if prediction is not None:
                elapsed = time.perf_counter() - started
                public_long_dir = output_dir / "diagnostics" / "long_video"
                public_long_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = _checkpoint_identity(effective_checkpoint)
                manifest = {
                    "schema": "long-video-exact-local-v1",
                    "source": str(Path(cfg.video_path).resolve()),
                    "frames": frame_count,
                    "checkpoint": checkpoint,
                    "attention_impl": "local",
                    "attention_window_frames": int(model.pipeline.denoiser3d.max_len),
                    "static_cam": bool(cfg.static_cam),
                    "no_postproc": bool(cfg.no_postproc),
                }
                metrics = {
                    "mode": "exact-full-sequence-local-attention",
                    "frames": frame_count,
                    "network_and_postprocess_elapsed_s": elapsed,
                    "window_fallback": False,
                }
                public_manifest = public_long_dir / "manifest.json"
                public_metrics = public_long_dir / "metrics.json"
                public_manifest.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                public_metrics.write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                long_video_result = {
                    "long_video_mode": metrics["mode"],
                    "long_video_windows": 1,
                    "long_video_reused_windows": 0,
                    "long_video_manifest_path": str(public_manifest.resolve()),
                    "long_video_metrics_path": str(public_metrics.resolve()),
                }

        if is_long_video and prediction is None:
            window_frames = _positive_env_int(
                "GVHMR_LONG_VIDEO_WINDOW_FRAMES", LONG_VIDEO_WINDOW_FRAMES
            )
            stride_frames = _positive_env_int(
                "GVHMR_LONG_VIDEO_STRIDE_FRAMES", LONG_VIDEO_STRIDE_FRAMES
            )
            if not 0 < stride_frames < window_frames:
                raise ValueError(
                    "GVHMR_LONG_VIDEO_STRIDE_FRAMES must be smaller than "
                    "GVHMR_LONG_VIDEO_WINDOW_FRAMES"
                )
            cache_identity = json.dumps(
                {
                    "schema": "shared-preprocess-windowed-network-v1",
                    "checkpoint": _checkpoint_identity(effective_checkpoint),
                    "static_cam": bool(cfg.static_cam),
                    "no_postproc": bool(cfg.no_postproc),
                    "use_foot_refiner": bool(model.pipeline.use_foot_refiner),
                    "num_2d_joints": int(cfg.network.num_2d_joints),
                    "attention_impl": str(cfg.attention_impl),
                    "attention_chunk_size": int(cfg.attention_chunk_size),
                },
                sort_keys=True,
            )
            Log.info(f"[Long Video] Network windows {window_frames}/{stride_frames}")
            prediction, long_metrics = predict_long_video(
                model=model,
                data=data,
                normalized_video=Path(cfg.video_path),
                work_root=output_dir / ".long_video_work",
                static_cam=bool(cfg.static_cam),
                no_postproc=bool(cfg.no_postproc),
                detach_to_cpu=detach_to_cpu,
                window_frames=window_frames,
                stride_frames=stride_frames,
                cache_identity=cache_identity,
                log=lambda message: print(message, flush=True),
            )
            public_long_dir = output_dir / "diagnostics" / "long_video"
            public_long_dir.mkdir(parents=True, exist_ok=True)
            public_manifest = public_long_dir / "manifest.json"
            public_metrics = public_long_dir / "metrics.json"
            shutil.copy2(long_metrics["manifest"], public_manifest)
            shutil.copy2(long_metrics["metrics"], public_metrics)
            long_video_result = {
                "long_video_mode": long_metrics["mode"],
                "long_video_windows": long_metrics["windows"],
                "long_video_reused_windows": long_metrics["reused_windows"],
                "long_video_manifest_path": str(public_manifest.resolve()),
                "long_video_metrics_path": str(public_metrics.resolve()),
            }
            long_metrics["fallback_reason"] = fallback_reason
            public_metrics.write_text(
                json.dumps(long_metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        elif not is_long_video:
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
        public_long_dir = output_dir / "diagnostics" / "long_video"
        public_manifest = public_long_dir / "manifest.json"
        public_metrics = public_long_dir / "metrics.json"
        if public_manifest.is_file() and public_metrics.is_file():
            try:
                cached_metrics = json.loads(public_metrics.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached_metrics = {}
            long_video_result = {
                "long_video_mode": cached_metrics.get("mode", "unknown"),
                "long_video_windows": int(cached_metrics.get("windows", 1)),
                "long_video_reused_windows": int(
                    cached_metrics.get("reused_windows", 0)
                ),
                "long_video_manifest_path": str(public_manifest.resolve()),
                "long_video_metrics_path": str(public_metrics.resolve()),
            }

    ground_result = _apply_ground_constraint(
        Path(args.core_root).expanduser().resolve(),
        output_dir,
        cfg.video_path,
        result_path,
        args.ground_constraint,
        static_cam=bool(args.static_cam),
        save_intermediate=bool(args.save_intermediate),
    )
    orientation_result = _apply_orientation_guard(output_dir, result_path)

    if not args.save_intermediate:
        _cleanup_preprocess(preprocess_dir)

    return {
        "output_dir": str(output_dir),
        "input_video_path": str(Path(cfg.video_path).resolve()),
        "hmr4d_results_path": str(result_path.resolve()),
        **long_video_result,
        **ground_result,
        **orientation_result,
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
    Path(cfg.paths.incam_video).parent.mkdir(parents=True, exist_ok=True)

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
        choices=("none", "flat_y", "gravity_flat", "human3r"),
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
