import re
import shutil
import threading
import time
import json
import math
from pathlib import Path
from queue import Empty, Queue

from hmr4d.api.video_to_data import GVHMRRunner
from hmr4d.service.common import (
    create_dated_output_dir,
    ensure_dir,
    make_batch_id,
    make_job_id,
    terminal_job_states,
    utc_now_iso,
    zip_artifacts,
)
from hmr4d.service.external_core import ExternalCoreRunner


_PERCENT_RE = re.compile(r"(?<!\d)(\d{1,3})%")
GROUND_CONSTRAINTS = {"none", "flat_y", "human3r"}
SONIC_SPEED_MIN = 0.25
SONIC_SPEED_MAX = 1.0
SONIC_SPEED_STEP = 0.05


def _normalize_ground_constraint(value):
    value = str(value or "none").strip().lower()
    if value not in GROUND_CONSTRAINTS:
        raise ValueError(f"Unsupported ground constraint: {value}")
    if value == "human3r":
        raise ValueError("Human3R scene constraint is not enabled yet.")
    return value


def _scale_progress(value, start, end):
    value = max(0, min(100, int(value)))
    return round(start + (end - start) * value / 100)


def _progress_from_log(message, task_kind):
    text = str(message or "")
    lower = text.lower()
    match = _PERCENT_RE.search(text)
    meter_percent = int(match.group(1)) if match else None

    if task_kind == "preview":
        if "rendering incam" in lower:
            percent = _scale_progress(meter_percent, 5, 46) if meter_percent is not None else 5
            return percent, "渲染相机视角", meter_percent is not None
        if "rendering global" in lower:
            percent = _scale_progress(meter_percent, 47, 90) if meter_percent is not None else 47
            return percent, "渲染全局视角", meter_percent is not None
        if "render incam" in lower and "already exists" in lower:
            return 46, "复用相机视角", False
        if "render global" in lower and "already exists" in lower:
            return 90, "复用全局视角", False
        if "merge videos" in lower:
            return 96, "合成预览视频", False
        return None

    if "reusing cached result" in lower:
        return 92, "读取已有动作结果", False
    if "[output dir]" in lower:
        return 3, "初始化任务目录", False
    if "[input]" in lower:
        return 5, "读取输入视频", False
    if "copy video" in lower:
        return 7, "转换输入视频", False
    if meter_percent is not None and re.search(r"\bcopy\b", lower):
        return _scale_progress(meter_percent, 7, 12), "转换输入视频", True
    if "[preprocess] start" in lower:
        return 12, "开始视频预处理", False
    if "yolov8 tracking" in lower or (meter_percent is not None and "tracking" in lower):
        percent = _scale_progress(meter_percent, 12, 29) if meter_percent is not None else 12
        return percent, "跟踪人物", meter_percent is not None
    if "vitpose" in lower:
        percent = _scale_progress(meter_percent, 29, 57) if meter_percent is not None else 57
        return percent, "提取人体关键点", meter_percent is not None
    if "hmr2 feature" in lower or "vit_features" in lower:
        percent = _scale_progress(meter_percent, 57, 82) if meter_percent is not None else 82
        return percent, "提取图像特征", meter_percent is not None
    if "dpvo" in lower or "simplevo" in lower:
        percent = _scale_progress(meter_percent, 82, 90) if meter_percent is not None else 82
        return percent, "估计相机运动", meter_percent is not None
    if "[preprocess] end" in lower:
        return 90, "视频预处理完成", False
    if "[hmr4d] predicting" in lower:
        return 93, "恢复人体动作", False
    if "[long video] window" in lower:
        return 96, "分窗恢复长视频动作", False
    if "loading gvhmr model" in lower or "loading ckpt" in lower:
        return 95, "加载动作模型", False
    if "[hmr4d] elapsed" in lower:
        return 98, "动作恢复完成", False
    if "[ground constraint]" in lower:
        return 99, "应用地面约束", False
    if "[cleanup]" in lower:
        return 99, "整理输出文件", False
    return None


class JobManager:
    def __init__(self, settings, store):
        self.settings = settings
        self.store = store
        self._runner = None
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._sonic_controller = None
        self._sonic_job_id = None
        self._sonic_generation = 0
        self._sonic_lock = threading.RLock()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="gvhmr-job-worker")
        self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        self._queue.put(None)
        with self._sonic_lock:
            if self._sonic_controller is not None:
                self._sonic_controller.close()
            self._sonic_job_id = None
        close_runner = getattr(self._runner, "close", None)
        if callable(close_runner):
            close_runner()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _get_runner(self):
        if self._runner is None:
            if self.settings.core_root is not None:
                self._runner = ExternalCoreRunner(
                    core_root=self.settings.core_root,
                    checkpoint_root=self.settings.checkpoint_root,
                    python_executable=self.settings.core_python,
                )
            else:
                self._runner = GVHMRRunner(checkpoint_root=self.settings.checkpoint_root)
        return self._runner

    def _stage_input_video(self, source_video, output_dir):
        source_video = Path(source_video).expanduser().resolve()
        output_dir = ensure_dir(output_dir)
        staged_path = output_dir / f"submitted_input{source_video.suffix.lower()}"
        if source_video != staged_path:
            shutil.copy2(source_video, staged_path)
        return staged_path

    def _make_output_dir(self, source_video, output_root=None):
        source_video = Path(source_video)
        root = Path(output_root).expanduser().resolve() if output_root else self.settings.output_root
        return create_dated_output_dir(root, source_video.name)

    def _relocate_job_paths(self, job):
        output_dir = Path(job["output_dir"])
        if output_dir.exists():
            return False

        candidate = self.settings.output_root / output_dir.name
        if not candidate.is_dir() or not any(
            (candidate / name).is_file()
            for name in ("job.json", "hmr4d_results.pt", "0_input_video.mp4")
        ):
            return False

        job["output_dir"] = str(candidate)
        input_path = Path(job.get("input_video") or "")
        input_candidates = [
            candidate / input_path.name if input_path.name else None,
            *sorted(candidate.glob("submitted_input.*")),
            candidate / "0_input_video.mp4",
        ]
        for path in input_candidates:
            if path is not None and path.is_file():
                job["input_video"] = str(path)
                break
        job["artifacts"] = {}
        return True

    def submit_job(
        self,
        *,
        video_source,
        static_cam,
        f_mm=None,
        save_intermediate=False,
        generate_preview=False,
        ground_constraint="none",
        batch_id=None,
        display_name=None,
        output_dir=None,
    ):
        video_source = Path(video_source).expanduser().resolve()
        if not video_source.exists():
            raise FileNotFoundError(f"Video not found at {video_source}")
        ground_constraint = _normalize_ground_constraint(ground_constraint)

        job_id = make_job_id()
        custom_output_dir = bool(str(output_dir or "").strip())
        output_name = display_name or video_source.name
        output_path = self._make_output_dir(output_name, output_root=output_dir if custom_output_dir else None)
        staged_input = self._stage_input_video(video_source, output_path)

        job = {
            "job_id": job_id,
            "batch_id": batch_id,
            "status": "queued",
            "task_kind": "process",
            "source_video_path": str(video_source),
            "input_video": str(staged_input),
            "display_name": display_name or video_source.name,
            "submitted_at": utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "updated_at": utc_now_iso(),
            "static_cam": bool(static_cam),
            "f_mm": None if f_mm in (None, "", 0) else int(f_mm),
            "save_intermediate": bool(save_intermediate),
            "generate_preview": bool(generate_preview),
            "ground_constraint": ground_constraint,
            "ground_constraint_status": "pending" if ground_constraint != "none" else "not_requested",
            "ground_constraint_error": None,
            "ground_constraint_fallback_reason": None,
            "ground_constraint_warning": None,
            "preview_status": "not_requested",
            "preview_error_summary": None,
            "progress_percent": 0,
            "progress_stage": "等待 GPU",
            "preview_progress_percent": 0,
            "preview_progress_stage": "尚未生成预览",
            "output_dir": str(output_path),
            "custom_output_dir": custom_output_dir,
            "artifacts": {},
            "error_summary": None,
            "logs": [],
            "cancel_requested": False,
        }
        self.store.create_job(job)
        self._queue.put(job_id)
        if batch_id:
            self.store.update_batch_counts(batch_id)
        return job

    def submit_batch(
        self,
        *,
        video_sources,
        static_cam,
        f_mm=None,
        save_intermediate=False,
        generate_preview=False,
        ground_constraint="none",
        input_dir=None,
        output_dir=None,
        display_names=None,
    ):
        batch_id = make_batch_id()
        batch_dir = ensure_dir(self.settings.batch_root / batch_id)
        total = len(video_sources)
        batch = {
            "batch_id": batch_id,
            "submitted_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "queued",
            "job_ids": [],
            "input_dir": str(input_dir) if input_dir else None,
            "batch_dir": str(batch_dir),
            "total": total,
            "queued": total,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        }
        self.store.create_batch(batch)

        display_names = list(display_names or [])
        for index, video_source in enumerate(video_sources):
            job = self.submit_job(
                video_source=video_source,
                static_cam=static_cam,
                f_mm=f_mm,
                save_intermediate=save_intermediate,
                generate_preview=generate_preview,
                ground_constraint=ground_constraint,
                batch_id=batch_id,
                display_name=display_names[index] if index < len(display_names) else None,
                output_dir=output_dir,
            )
            batch["job_ids"].append(job["job_id"])

        batch["updated_at"] = utc_now_iso()
        self.store.save_batch(batch)
        self.store.update_batch_counts(batch_id)
        return self.store.get_batch(batch_id)

    def list_jobs(self, limit=50, batch_id=None):
        return self.store.list_jobs(limit=limit, batch_id=batch_id)

    def list_batches(self, limit=20):
        return self.store.list_batches(limit=limit)

    def get_job(self, job_id):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        paths_changed = self._relocate_job_paths(job)
        if job.get("status") in terminal_job_states():
            artifacts_before = dict(job.get("artifacts", {}))
            self._build_artifact_map(job)
            if paths_changed or job["artifacts"] != artifacts_before:
                self.store.save_job(job)
        elif paths_changed:
            self.store.save_job(job)
        return job

    def ensure_artifact_bundle(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return None
        return self._finalize_job(job)

    def get_batch(self, batch_id):
        batch = self.store.get_batch(batch_id)
        if batch is None:
            return None
        return self.store.update_batch_counts(batch_id)

    def cancel_job(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] in terminal_job_states():
            return job

        job["cancel_requested"] = True
        job["updated_at"] = utc_now_iso()
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = utc_now_iso()
            job["error_summary"] = "Cancelled before execution."
            self._finalize_job(job)
        else:
            self._append_log(job, "[Control] Cancellation requested. The current stage will finish before stopping.")
            self.store.save_job(job)
        if job.get("batch_id"):
            self.store.update_batch_counts(job["batch_id"])
        return job

    def retry_job(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] not in terminal_job_states():
            raise RuntimeError("Only terminal jobs can be retried.")

        job["status"] = "queued"
        job["task_kind"] = "process"
        job["started_at"] = None
        job["finished_at"] = None
        job["updated_at"] = utc_now_iso()
        job["error_summary"] = None
        job["ground_constraint_status"] = (
            "pending" if job.get("ground_constraint") != "none" else "not_requested"
        )
        job["ground_constraint_error"] = None
        job["ground_constraint_fallback_reason"] = None
        job["ground_constraint_warning"] = None
        job["cancel_requested"] = False
        job["preview_status"] = "not_requested"
        job["preview_error_summary"] = None
        job["progress_percent"] = 0
        job["progress_stage"] = "等待 GPU"
        job["preview_progress_percent"] = 0
        job["preview_progress_stage"] = "尚未生成预览"
        self._append_log(job, "[Control] Retry requested.")
        self.store.save_job(job)
        self._queue.put(job_id)
        if job.get("batch_id"):
            self.store.update_batch_counts(job["batch_id"])
        return job

    def request_preview(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] != "succeeded":
            raise RuntimeError("Preview generation is only available for succeeded jobs.")

        if job.get("preview_status") in {"queued", "running"}:
            return job

        artifacts = self._build_artifact_map(job)
        if artifacts.get("preview_video_path"):
            job["preview_status"] = "succeeded"
            job["preview_error_summary"] = None
            self.store.save_job(job)
            return job

        job["task_kind"] = "preview"
        job["preview_status"] = "queued"
        job["preview_error_summary"] = None
        job["preview_progress_percent"] = 0
        job["preview_progress_stage"] = "等待生成预览"
        job["updated_at"] = utc_now_iso()
        job["cancel_requested"] = False
        job["generate_preview"] = True
        self._append_log(job, "[Control] Preview generation requested.")
        self.store.save_job(job)
        self._queue.put(job_id)
        if job.get("batch_id"):
            self.store.update_batch_counts(job["batch_id"])
        return job

    def send_to_sonic(self, job_id, *, speed=1.0):
        """Prepare and asynchronously stream a completed job to local SONIC."""
        speed = float(speed)
        if not math.isfinite(speed):
            raise ValueError("SONIC speed must be finite.")
        normalized_speed = round(speed, 2)
        step_index = round((normalized_speed - SONIC_SPEED_MIN) / SONIC_SPEED_STEP)
        on_step = math.isclose(
            normalized_speed,
            SONIC_SPEED_MIN + step_index * SONIC_SPEED_STEP,
            abs_tol=1e-9,
        )
        if (
            not math.isclose(speed, normalized_speed, abs_tol=1e-9)
            or normalized_speed < SONIC_SPEED_MIN
            or normalized_speed > SONIC_SPEED_MAX
            or not on_step
        ):
            raise ValueError(
                f"Unsupported SONIC speed {speed:g}; choose {SONIC_SPEED_MIN:g}x to "
                f"{SONIC_SPEED_MAX:g}x in {SONIC_SPEED_STEP:g}x steps."
            )
        speed = normalized_speed
        job = self.get_job(job_id)
        if job is None:
            return None
        if job["status"] != "succeeded":
            raise RuntimeError("Only succeeded jobs can be sent to SONIC.")

        from hmr4d.utils.sonic import (
            PlaybackState,
            SonicPlaybackController,
            SonicReference,
        )
        from tools.sonic.convert_gvhmr import convert, sha256

        output_dir = Path(job["output_dir"])
        source = output_dir / "hmr4d_results.pt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing inference result at {source}")
        speed_tag = f"{speed:.2f}".rstrip("0").rstrip(".").replace(".", "_")
        if speed == 1.0:
            reference_path = output_dir / "sonic_reference.npz"
            metadata_path = output_dir / "sonic_conversion.json"
        else:
            reference_path = output_dir / f"sonic_reference_speed_{speed_tag}.npz"
            metadata_path = output_dir / f"sonic_conversion_speed_{speed_tag}.json"
        effective_source_fps = 30.0 * speed
        source_digest = sha256(source)
        metadata = None
        if reference_path.is_file() and metadata_path.is_file():
            try:
                cached = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    cached.get("source_sha256") == source_digest
                    and math.isclose(
                        float(cached.get("source_fps", 0.0)),
                        effective_source_fps,
                        abs_tol=1e-9,
                    )
                ):
                    metadata = cached
            except (OSError, ValueError, TypeError):
                metadata = None
        reused = metadata is not None
        if metadata is None:
            metadata = convert(
                source,
                reference_path,
                source_fps=effective_source_fps,
            )
            metadata["playback_speed"] = speed
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        import numpy as np

        with np.load(reference_path, allow_pickle=False) as data:
            reference = SonicReference(
                term1_local=data["term1_local"],
                root_quat=data["root_quat"],
                wrist=data["wrist"],
                fps=float(data["fps"]),
            )

        job["sonic_status"] = "preparing"
        job["sonic_error"] = None
        job["sonic_frame"] = 0
        job["sonic_frames"] = reference.frame_count
        job["sonic_speed"] = speed
        job["sonic_updated_at"] = utc_now_iso()
        job.setdefault("artifacts", {}).update(
            {
                "sonic_reference_path": str(reference_path),
                "sonic_metadata_path": str(metadata_path),
            }
        )
        self._append_log(
            job,
            f"[SONIC] Prepared {reference.frame_count} frames at {reference.fps:g} FPS"
            f" / {speed:g}x playback"
            f" ({'cache reused' if reused else 'new reference'}).",
        )
        self._finalize_job(job)

        last_saved_frame = -25
        playback_generation = None

        def callback(state, frame_index, frame_count, message):
            nonlocal last_saved_frame
            if (
                state == PlaybackState.STREAMING
                and frame_index < frame_count
                and frame_index - last_saved_frame < 25
            ):
                return
            current = self.store.get_job(job_id)
            if current is None:
                return
            current["sonic_status"] = state.value
            current["sonic_frame"] = int(frame_index)
            current["sonic_frames"] = int(frame_count)
            current["sonic_updated_at"] = utc_now_iso()
            if state == PlaybackState.ERROR:
                current["sonic_error"] = message or "Unknown SONIC playback error."
                self._append_log(current, f"[SONIC] Playback failed: {current['sonic_error']}")
            elif state == PlaybackState.COMPLETE:
                current["sonic_error"] = None
                self._append_log(current, "[SONIC] Playback complete.")
            elif state == PlaybackState.STOPPED:
                self._append_log(current, "[SONIC] Playback replaced or stopped.")
            self.store.save_job(current)
            last_saved_frame = int(frame_index)
            if state in {PlaybackState.COMPLETE, PlaybackState.STOPPED, PlaybackState.ERROR}:
                if (
                    self._sonic_job_id == job_id
                    and self._sonic_generation == playback_generation
                ):
                    self._sonic_job_id = None

        with self._sonic_lock:
            if self._sonic_controller is None:
                self._sonic_controller = SonicPlaybackController()
            self._sonic_generation += 1
            playback_generation = self._sonic_generation
            self._sonic_job_id = job_id
            self._sonic_controller.run(0, reference, callback)
        return {
            "job_id": job_id,
            "status": "preparing",
            "reference_path": str(reference_path),
            "frames": reference.frame_count,
            "fps": reference.fps,
            "speed": speed,
            "duration_s": (reference.frame_count - 1) / reference.fps,
            "reused": reused,
        }

    def pause_sonic(self, job_id):
        """Stop the active live reference so SONIC falls back to its idle pose."""
        job = self.get_job(job_id)
        if job is None:
            return None
        with self._sonic_lock:
            if self._sonic_job_id != job_id or self._sonic_controller is None:
                raise RuntimeError("This job is not currently streaming to SONIC.")
            stopped = self._sonic_controller.stop(0)
            self._sonic_job_id = None
        if not stopped:
            raise RuntimeError("SONIC streaming has already stopped.")

        current = self.get_job(job_id)
        if current is None:
            return None
        current["sonic_status"] = "paused"
        current["sonic_error"] = None
        current["sonic_updated_at"] = utc_now_iso()
        self._append_log(
            current,
            "[SONIC] Streaming paused; policy will blend back to its idle reference.",
        )
        self.store.save_job(current)
        return {
            "job_id": job_id,
            "status": "paused",
            "fallback": "idle_reference",
            "timeout_s": 0.5,
            "blend_s": 0.4,
        }

    def _append_log(self, job, message):
        job.setdefault("logs", []).append(message)
        job["updated_at"] = utc_now_iso()

    def _log_callback(self, job_id, task_kind=None):
        def callback(message):
            job = self.get_job(job_id)
            if job is None:
                return
            current_task = task_kind or job.get("task_kind", "process")
            progress = _progress_from_log(message, current_task)
            if progress is not None:
                percent, stage, is_meter = progress
                percent_key = "preview_progress_percent" if current_task == "preview" else "progress_percent"
                stage_key = "preview_progress_stage" if current_task == "preview" else "progress_stage"
                previous_percent = int(job.get(percent_key, 0) or 0)
                changed = percent > previous_percent or stage != job.get(stage_key)
                job[percent_key] = max(previous_percent, percent)
                job[stage_key] = stage
                job["updated_at"] = utc_now_iso()
                if not is_meter:
                    self._append_log(job, message)
                if changed or not is_meter:
                    self.store.save_job(job)
                return
            self._append_log(job, message)
            self.store.save_job(job)

        return callback

    def _merge_live_job_fields(self, job):
        latest = self.store.get_job(job["job_id"])
        if latest is None:
            return job
        job["logs"] = latest.get("logs", job.get("logs", []))
        job["cancel_requested"] = latest.get("cancel_requested", job.get("cancel_requested", False))
        for key in (
            "progress_percent",
            "progress_stage",
            "preview_progress_percent",
            "preview_progress_stage",
        ):
            if key in latest:
                job[key] = latest[key]
        return job

    def _ensure_preview_input(self, job, runner):
        output_dir = Path(job["output_dir"])
        normalized_video = output_dir / "0_input_video.mp4"
        if normalized_video.is_file():
            return
        if not (output_dir / "hmr4d_results.pt").is_file():
            raise FileNotFoundError(f"Missing inference result at {output_dir / 'hmr4d_results.pt'}")

        source_candidates = [
            Path(job.get("input_video") or ""),
            Path(job.get("source_video_path") or ""),
        ]
        source_video = next((path for path in source_candidates if path.is_file()), None)
        if source_video is None:
            raise FileNotFoundError(
                f"Missing processed video at {normalized_video}; the original task input is also unavailable."
            )

        self._append_log(job, f"[Preview] Rebuilding processed video from {source_video}")
        self.store.save_job(job)
        result = runner.process_video(
            video_path=source_video,
            output_dir=job["output_dir"],
            static_cam=job["static_cam"],
            f_mm=job["f_mm"],
            save_intermediate=job["save_intermediate"],
            **(
                {"ground_constraint": job.get("ground_constraint", "none")}
                if isinstance(runner, ExternalCoreRunner)
                else {}
            ),
            log_callback=self._log_callback(job["job_id"]),
        )
        self._merge_live_job_fields(job)
        job["artifacts"].update(
            {key: value for key, value in result.items() if key.endswith("_path")}
        )
        if "ground_constraint_status" in result:
            job["ground_constraint_status"] = result["ground_constraint_status"]
        if "ground_constraint_error" in result:
            job["ground_constraint_error"] = result["ground_constraint_error"]
        if "ground_constraint_fallback_reason" in result:
            job["ground_constraint_fallback_reason"] = result[
                "ground_constraint_fallback_reason"
            ]

    def _build_artifact_map(self, job):
        output_dir = Path(job["output_dir"])
        constraint_dir = output_dir / "ground_constraint_global_v1_1"
        constraint_result = constraint_dir / "contact_global_root_hmr4d_results.pt"
        constraint_metrics = constraint_dir / "metrics.json"
        legacy_constraint_dir = output_dir / "ground_constraint_flat_y"
        legacy_constraint_result = legacy_constraint_dir / "contact_floor_y_hmr4d_results.pt"
        if not legacy_constraint_result.is_file():
            legacy_constraint_result = legacy_constraint_dir / "flat_ground_y_hmr4d_results.pt"
        legacy_constraint_metrics = legacy_constraint_dir / "metrics.json"
        if not constraint_metrics.is_file() and legacy_constraint_metrics.is_file():
            constraint_metrics = legacy_constraint_metrics
        existing = {
            key: value
            for key, value in job.get("artifacts", {}).items()
            if value and Path(value).is_file()
        }
        artifact_map = {
            "job_json_path": str(output_dir / "job.json"),
            "input_video_path": str(output_dir / "0_input_video.mp4"),
            "hmr4d_results_path": str(output_dir / "hmr4d_results.pt"),
            "raw_hmr4d_results_path": str(output_dir / "hmr4d_results_raw.pt"),
            "global_contact_results_path": str(constraint_result),
            "flat_ground_y_results_path": str(legacy_constraint_result),
            "ground_constraint_metrics_path": str(constraint_metrics),
            "long_video_manifest_path": str(output_dir / "long_video" / "manifest.json"),
            "long_video_metrics_path": str(output_dir / "long_video" / "metrics.json"),
            "sonic_reference_path": str(output_dir / "sonic_reference.npz"),
            "sonic_metadata_path": str(output_dir / "sonic_conversion.json"),
            "incam_video_path": str(output_dir / "1_incam.mp4"),
            "global_video_path": str(output_dir / "2_global.mp4"),
            "preview_video_path": str(output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4"),
            "artifacts_zip_path": str(output_dir / "artifacts.zip"),
        }
        for key, value in artifact_map.items():
            if key not in existing and Path(value).is_file():
                existing[key] = value
        job["artifacts"] = existing
        return job["artifacts"]

    def _finalize_job(self, job):
        job["artifacts"] = self._build_artifact_map(job)
        self.store.save_job(job)
        output_dir = Path(job["output_dir"])
        files = [
            (output_dir / "job.json", "job.json"),
            (output_dir / "hmr4d_results.pt", "hmr4d_results.pt"),
            (output_dir / "hmr4d_results_raw.pt", "hmr4d_results_raw.pt"),
            (
                output_dir / "ground_constraint_global_v1_1" / "contact_global_root_hmr4d_results.pt",
                "ground_constraint_global_v1_1/contact_global_root_hmr4d_results.pt",
            ),
            (
                output_dir / "ground_constraint_global_v1_1" / "metrics.json",
                "ground_constraint_global_v1_1/metrics.json",
            ),
            (output_dir / "long_video" / "manifest.json", "long_video/manifest.json"),
            (output_dir / "long_video" / "metrics.json", "long_video/metrics.json"),
            (
                output_dir / "ground_constraint_flat_y" / "contact_floor_y_hmr4d_results.pt",
                "ground_constraint_flat_y/contact_floor_y_hmr4d_results.pt",
            ),
            (
                output_dir / "ground_constraint_flat_y" / "flat_ground_y_hmr4d_results.pt",
                "ground_constraint_flat_y/flat_ground_y_hmr4d_results.pt",
            ),
            (
                output_dir / "ground_constraint_flat_y" / "metrics.json",
                "ground_constraint_flat_y/metrics.json",
            ),
            (output_dir / "1_incam.mp4", "1_incam.mp4"),
            (output_dir / "2_global.mp4", "2_global.mp4"),
            (
                Path(job.get("artifacts", {}).get("sonic_reference_path", output_dir / "sonic_reference.npz")),
                Path(job.get("artifacts", {}).get("sonic_reference_path", "sonic_reference.npz")).name,
            ),
            (
                Path(job.get("artifacts", {}).get("sonic_metadata_path", output_dir / "sonic_conversion.json")),
                Path(job.get("artifacts", {}).get("sonic_metadata_path", "sonic_conversion.json")).name,
            ),
            (
                output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4",
                f"{output_dir.name}_3_incam_global_horiz.mp4",
            ),
        ]
        if any(Path(path).is_file() for path, _arcname in files[1:]):
            zip_path = zip_artifacts(output_dir / "artifacts.zip", files)
            job["artifacts"]["artifacts_zip_path"] = str(zip_path)
            self.store.save_job(job)
        if job.get("batch_id"):
            self.store.update_batch_counts(job["batch_id"])
        return job

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except Empty:
                continue

            if job_id is None:
                continue

            job = self.get_job(job_id)
            if job is None or job["status"] == "cancelled":
                continue
            if job.get("cancel_requested") and job["status"] == "queued":
                continue

            task_kind = job.get("task_kind", "process")
            if task_kind == "preview":
                job["preview_status"] = "running"
                job["preview_progress_percent"] = max(1, int(job.get("preview_progress_percent", 0) or 0))
                job["preview_progress_stage"] = "准备生成预览"
            else:
                job["status"] = "running"
                job["started_at"] = job.get("started_at") or utc_now_iso()
                job["progress_percent"] = max(1, int(job.get("progress_percent", 0) or 0))
                job["progress_stage"] = "启动 GVHMR"
            job["updated_at"] = utc_now_iso()
            self._append_log(job, f"[Worker] Started {task_kind} task.")
            self.store.save_job(job)
            if job.get("batch_id"):
                self.store.update_batch_counts(job["batch_id"])

            try:
                runner = self._get_runner()
                if task_kind == "process":
                    result = runner.process_video(
                        video_path=job["input_video"],
                        output_dir=job["output_dir"],
                        static_cam=job["static_cam"],
                        f_mm=job["f_mm"],
                        save_intermediate=job["save_intermediate"],
                        **(
                            {"ground_constraint": job.get("ground_constraint", "none")}
                            if isinstance(runner, ExternalCoreRunner)
                            else {}
                        ),
                        log_callback=self._log_callback(job_id, "process"),
                    )
                    self._merge_live_job_fields(job)
                    job["artifacts"].update(
                        {key: value for key, value in result.items() if key.endswith("_path")}
                    )
                    if "ground_constraint_status" in result:
                        job["ground_constraint_status"] = result["ground_constraint_status"]
                    if "ground_constraint_error" in result:
                        job["ground_constraint_error"] = result["ground_constraint_error"]
                    if "ground_constraint_fallback_reason" in result:
                        job["ground_constraint_fallback_reason"] = result[
                            "ground_constraint_fallback_reason"
                        ]
                    if "ground_constraint_warning" in result:
                        job["ground_constraint_warning"] = result[
                            "ground_constraint_warning"
                        ]
                    if job.get("cancel_requested"):
                        job["status"] = "cancelled"
                        job["finished_at"] = utc_now_iso()
                        job["error_summary"] = "Cancelled after the main processing stage completed."
                    elif job["generate_preview"]:
                        try:
                            job["preview_status"] = "running"
                            job["preview_progress_percent"] = 1
                            job["preview_progress_stage"] = "准备生成预览"
                            self.store.save_job(job)
                            preview = runner.generate_preview(
                                output_dir=job["output_dir"],
                                log_callback=self._log_callback(job_id, "preview"),
                            )
                            self._merge_live_job_fields(job)
                            job["artifacts"].update(preview)
                            job["preview_status"] = "succeeded"
                            job["preview_error_summary"] = None
                            job["preview_progress_percent"] = 100
                            job["preview_progress_stage"] = "预览生成完成"
                        except Exception as preview_exc:
                            preview_error = str(preview_exc).strip() or preview_exc.__class__.__name__
                            job["preview_status"] = "failed"
                            job["preview_error_summary"] = preview_error
                            self._append_log(job, f"[Preview] Failed: {preview_error}")
                        job["status"] = "succeeded"
                        job["finished_at"] = utc_now_iso()
                        job["progress_percent"] = 100
                        job["progress_stage"] = "动作处理完成"
                    else:
                        job["status"] = "succeeded"
                        job["finished_at"] = utc_now_iso()
                        job["progress_percent"] = 100
                        job["progress_stage"] = "动作处理完成"
                else:
                    self._ensure_preview_input(job, runner)
                    preview = runner.generate_preview(
                        output_dir=job["output_dir"],
                        log_callback=self._log_callback(job_id, "preview"),
                    )
                    self._merge_live_job_fields(job)
                    job["artifacts"].update(preview)
                    job["task_kind"] = "process"
                    job["preview_status"] = "succeeded"
                    job["preview_error_summary"] = None
                    job["preview_progress_percent"] = 100
                    job["preview_progress_stage"] = "预览生成完成"
            except Exception as exc:
                self._merge_live_job_fields(job)
                error_summary = str(exc).strip() or exc.__class__.__name__
                if task_kind == "preview":
                    job["task_kind"] = "process"
                    job["preview_status"] = "failed"
                    job["preview_error_summary"] = error_summary
                    job["preview_progress_stage"] = "预览生成失败"
                    self._append_log(job, f"[Preview] Failed: {error_summary}")
                else:
                    job["status"] = "failed"
                    job["finished_at"] = utc_now_iso()
                    job["error_summary"] = error_summary
                    job["progress_stage"] = "动作处理失败"
                    if job.get("ground_constraint") != "none":
                        job["ground_constraint_status"] = "failed"
                        job["ground_constraint_error"] = error_summary
                        job["ground_constraint_fallback_reason"] = None
                        job["ground_constraint_warning"] = None
                    self._append_log(job, f"[Worker] Failed: {error_summary}")

            job["updated_at"] = utc_now_iso()
            self._finalize_job(job)

    def wait_for_job(self, job_id, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job(job_id)
            if job and job["status"] in terminal_job_states():
                return job
            time.sleep(0.5)
        return self.get_job(job_id)
