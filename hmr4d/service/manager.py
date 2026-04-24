import shutil
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from hmr4d.api.video_to_data import GVHMRRunner
from hmr4d.service.common import (
    ensure_dir,
    make_batch_id,
    make_job_id,
    short_id,
    terminal_job_states,
    utc_now_iso,
    zip_artifacts,
)


class JobManager:
    def __init__(self, settings, store):
        self.settings = settings
        self.store = store
        self._runner = None
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="gvhmr-job-worker")
        self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _get_runner(self):
        if self._runner is None:
            self._runner = GVHMRRunner(checkpoint_root=self.settings.checkpoint_root)
        return self._runner

    def _stage_input_video(self, source_video, output_dir):
        source_video = Path(source_video).expanduser().resolve()
        output_dir = ensure_dir(output_dir)
        staged_path = output_dir / f"submitted_input{source_video.suffix.lower()}"
        shutil.copy2(source_video, staged_path)
        return staged_path

    def _make_output_dir(self, source_video, job_id):
        source_video = Path(source_video)
        return ensure_dir(self.settings.output_root / f"{source_video.stem}_{short_id(job_id)}")

    def submit_job(
        self,
        *,
        video_source,
        static_cam,
        f_mm=None,
        save_intermediate=False,
        generate_preview=False,
        batch_id=None,
    ):
        video_source = Path(video_source).expanduser().resolve()
        if not video_source.exists():
            raise FileNotFoundError(f"Video not found at {video_source}")

        job_id = make_job_id()
        output_dir = self._make_output_dir(video_source, job_id)
        staged_input = self._stage_input_video(video_source, output_dir)

        job = {
            "job_id": job_id,
            "batch_id": batch_id,
            "status": "queued",
            "task_kind": "process",
            "source_video_path": str(video_source),
            "input_video": str(staged_input),
            "submitted_at": utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "updated_at": utc_now_iso(),
            "static_cam": bool(static_cam),
            "f_mm": None if f_mm in (None, "", 0) else int(f_mm),
            "save_intermediate": bool(save_intermediate),
            "generate_preview": bool(generate_preview),
            "output_dir": str(output_dir),
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
        input_dir=None,
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

        for video_source in video_sources:
            job = self.submit_job(
                video_source=video_source,
                static_cam=static_cam,
                f_mm=f_mm,
                save_intermediate=save_intermediate,
                generate_preview=generate_preview,
                batch_id=batch_id,
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
        return self.store.get_job(job_id)

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
        job["cancel_requested"] = False
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

        preview_path = Path(job["output_dir"]) / f"{Path(job['output_dir']).name}_3_incam_global_horiz.mp4"
        if preview_path.exists():
            job["artifacts"]["preview_video_path"] = str(preview_path)
            self.store.save_job(job)
            return job

        job["status"] = "queued"
        job["task_kind"] = "preview"
        job["updated_at"] = utc_now_iso()
        job["cancel_requested"] = False
        job["generate_preview"] = True
        self._append_log(job, "[Control] Preview generation requested.")
        self.store.save_job(job)
        self._queue.put(job_id)
        if job.get("batch_id"):
            self.store.update_batch_counts(job["batch_id"])
        return job

    def _append_log(self, job, message):
        job.setdefault("logs", []).append(message)
        job["updated_at"] = utc_now_iso()

    def _log_callback(self, job_id):
        def callback(message):
            job = self.get_job(job_id)
            if job is None:
                return
            self._append_log(job, message)
            self.store.save_job(job)

        return callback

    def _build_artifact_map(self, job):
        output_dir = Path(job["output_dir"])
        artifact_map = {
            "job_json_path": str(output_dir / "job.json"),
            "data_path": str(output_dir / "gvhmr_data.npz"),
            "meta_path": str(output_dir / "gvhmr_meta.json"),
            "hmr4d_results_path": str(output_dir / "hmr4d_results.pt"),
            "incam_video_path": str(output_dir / "1_incam.mp4"),
            "global_video_path": str(output_dir / "2_global.mp4"),
            "preview_video_path": str(output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4"),
            "artifacts_zip_path": str(output_dir / "artifacts.zip"),
        }
        job["artifacts"] = {key: value for key, value in artifact_map.items() if Path(value).exists()}
        return job["artifacts"]

    def _finalize_job(self, job):
        job["artifacts"] = self._build_artifact_map(job)
        self.store.save_job(job)
        artifacts = self._build_artifact_map(job)
        output_dir = Path(job["output_dir"])
        files = [
            (output_dir / "job.json", "job.json"),
            (output_dir / "gvhmr_data.npz", "gvhmr_data.npz"),
            (output_dir / "gvhmr_meta.json", "gvhmr_meta.json"),
            (output_dir / "hmr4d_results.pt", "hmr4d_results.pt"),
            (output_dir / "1_incam.mp4", "1_incam.mp4"),
            (output_dir / "2_global.mp4", "2_global.mp4"),
            (output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4", f"{output_dir.name}_3_incam_global_horiz.mp4"),
        ]
        zip_path = zip_artifacts(output_dir / "artifacts.zip", files)
        artifacts["artifacts_zip_path"] = str(zip_path)
        job["artifacts"] = artifacts
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

            job["status"] = "running"
            job["started_at"] = job.get("started_at") or utc_now_iso()
            job["updated_at"] = utc_now_iso()
            self._append_log(job, f"[Worker] Started {job['task_kind']} task.")
            self.store.save_job(job)
            if job.get("batch_id"):
                self.store.update_batch_counts(job["batch_id"])

            try:
                runner = self._get_runner()
                if job["task_kind"] == "process":
                    result = runner.process_video(
                        video_path=job["input_video"],
                        output_dir=job["output_dir"],
                        static_cam=job["static_cam"],
                        f_mm=job["f_mm"],
                        save_intermediate=job["save_intermediate"],
                        log_callback=self._log_callback(job_id),
                    )
                    job["artifacts"].update(result)
                    if job.get("cancel_requested"):
                        job["status"] = "cancelled"
                        job["finished_at"] = utc_now_iso()
                        job["error_summary"] = "Cancelled after the main processing stage completed."
                    elif job["generate_preview"]:
                        preview = runner.generate_preview(
                            output_dir=job["output_dir"],
                            log_callback=self._log_callback(job_id),
                        )
                        job["artifacts"].update(preview)
                        job["status"] = "succeeded"
                        job["finished_at"] = utc_now_iso()
                    else:
                        job["status"] = "succeeded"
                        job["finished_at"] = utc_now_iso()
                else:
                    preview = runner.generate_preview(
                        output_dir=job["output_dir"],
                        log_callback=self._log_callback(job_id),
                    )
                    job["artifacts"].update(preview)
                    job["task_kind"] = "process"
                    job["status"] = "succeeded"
                    job["finished_at"] = utc_now_iso()
            except Exception as exc:
                job["status"] = "failed"
                job["finished_at"] = utc_now_iso()
                job["error_summary"] = str(exc).strip() or exc.__class__.__name__
                self._append_log(job, f"[Worker] Failed: {job['error_summary']}")

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
