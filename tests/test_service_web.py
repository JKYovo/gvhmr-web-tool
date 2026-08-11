import json
import time
import unittest
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from hmr4d.api.video_to_data import _ensure_chumpy_numpy_compat
from hmr4d.service.common import ServiceSettings, create_dated_output_dir
from hmr4d.service.external_core import ExternalCoreRunner, RESULT_PREFIX
from hmr4d.service.external_core_worker import (
    _apply_ground_constraint,
    _ensure_chumpy_numpy_compat as ensure_external_chumpy_compat,
    _merge_preview_videos,
)
from hmr4d.service.manager import JobManager, _progress_from_log
from hmr4d.service.server import create_gvhmr_app
from hmr4d.service.store import SQLiteJobStore


class FailingPreviewRunner:
    def generate_preview(self, output_dir, log_callback=None):
        raise RuntimeError("preview renderer failed")


class RecoveringPreviewRunner:
    def __init__(self):
        self.process_calls = 0

    def process_video(self, video_path, output_dir, **_kwargs):
        self.process_calls += 1
        output_dir = Path(output_dir)
        normalized = output_dir / "0_input_video.mp4"
        normalized.write_bytes(Path(video_path).read_bytes())
        return {"input_video_path": str(normalized), "hmr4d_results_path": str(output_dir / "hmr4d_results.pt")}

    def generate_preview(self, output_dir, log_callback=None):
        output_dir = Path(output_dir)
        incam = output_dir / "1_incam.mp4"
        global_video = output_dir / "2_global.mp4"
        preview = output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4"
        incam.write_bytes(b"incam")
        global_video.write_bytes(b"global")
        preview.write_bytes(b"preview")
        return {
            "incam_video_path": str(incam),
            "global_video_path": str(global_video),
            "preview_video_path": str(preview),
        }


class ServiceWebTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.settings = ServiceSettings(
            checkpoint_root=root / "checkpoints",
            output_root=root / "runtime" / "jobs",
            batch_root=root / "runtime" / "batches",
            db_path=root / "runtime" / "db" / "jobs.sqlite",
            host="127.0.0.1",
            port=7860,
            sync_assets_on_boot=False,
        )
        self.settings.ensure_runtime_dirs()
        self.store = SQLiteJobStore(self.settings.db_path)
        self.manager = JobManager(self.settings, self.store)
        self.app = create_gvhmr_app(self.manager, self.settings, manage_lifecycle=False)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.manager.shutdown()
        self.temp_dir.cleanup()

    def _drain_submitted_job(self):
        try:
            self.manager._queue.get_nowait()
        except Empty:
            pass

    def _make_succeeded_job(self, name="sample.mp4"):
        source = Path(self.temp_dir.name) / name
        source.write_bytes(b"source-video")
        job = self.manager.submit_job(video_source=source, static_cam=True, display_name=name)
        self._drain_submitted_job()
        output_dir = Path(job["output_dir"])
        (output_dir / "0_input_video.mp4").write_bytes(b"normalized-video")
        (output_dir / "hmr4d_results.pt").write_bytes(b"motion")
        job = self.manager.get_job(job["job_id"])
        job["status"] = "succeeded"
        job["finished_at"] = job["updated_at"]
        self.store.save_job(job)
        return self.manager.get_job(job["job_id"])

    def test_single_upload_is_staged_and_upload_temp_is_removed(self):
        response = self.client.post(
            "/api/jobs/upload",
            files={"file": ("dance.mp4", b"video-bytes", "video/mp4")},
            data={"static_cam": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual(job["display_name"], "dance.mp4")
        self.assertTrue(Path(job["input_video"]).is_file())
        self.assertEqual(Path(job["input_video"]).parent, Path(job["output_dir"]))
        self.assertRegex(Path(job["output_dir"]).name, r"^dance_\d{8}_\d{6}$")
        upload_root = self.settings.output_root.parent / "uploads"
        self.assertFalse(upload_root.exists() and any(upload_root.iterdir()))

    def test_ground_constraint_selection_is_persisted_and_human3r_is_disabled(self):
        response = self.client.post(
            "/api/jobs/upload",
            files={"file": ("dance.mp4", b"video-bytes", "video/mp4")},
            data={"static_cam": "true", "ground_constraint": "flat_y"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["ground_constraint"], "flat_y")
        self.assertEqual(response.json()["ground_constraint_status"], "pending")

        disabled = self.client.post(
            "/api/jobs/upload",
            files={"file": ("scene.mp4", b"video-bytes", "video/mp4")},
            data={"ground_constraint": "human3r"},
        )
        self.assertEqual(disabled.status_code, 400, disabled.text)
        self.assertIn("not enabled", disabled.json()["detail"])

    def test_contact_floor_postprocess_preserves_raw_and_selects_enhanced(self):
        root = Path(self.temp_dir.name) / "flat-core"
        script = root / "tools" / "bench" / "human3r_p2y" / "apply_contact_floor_y.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--gvhmr-result')\n"
            "p.add_argument('--video')\n"
            "p.add_argument('--output-dir')\n"
            "p.add_argument('--smoothing-seconds')\n"
            "p.add_argument('--allow-large-correction', action='store_true')\n"
            "a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
            "(out/'contact_floor_y_hmr4d_results.pt').write_bytes(b'enhanced')\n"
            "(out/'metrics.json').write_text(json.dumps({'decision':'diagnostic_pass'}))\n",
            encoding="utf-8",
        )
        output = root / "output"
        output.mkdir()
        selected = output / "hmr4d_results.pt"
        selected.write_bytes(b"raw")
        video = output / "0_input_video.mp4"
        video.write_bytes(b"video")

        result = _apply_ground_constraint(root, output, video, selected, "flat_y")

        self.assertEqual(result["ground_constraint_status"], "applied")
        self.assertEqual(selected.read_bytes(), b"enhanced")
        self.assertEqual((output / "hmr4d_results_raw.pt").read_bytes(), b"raw")

    def test_output_directory_uses_timestamp_and_avoids_collisions(self):
        root = Path(self.temp_dir.name) / "dated-output"
        now = datetime(2026, 8, 7, 17, 25, 30, tzinfo=timezone.utc)

        first = create_dated_output_dir(root, "climb.mp4", now=now)
        second = create_dated_output_dir(root, "climb.mp4", now=now)

        self.assertEqual(first.name, "climb_20260807_172530")
        self.assertEqual(second.name, "climb_20260807_172530_2")

    def test_numpy_2_chumpy_compatibility_for_preview_renderer(self):
        _ensure_chumpy_numpy_compat()
        import chumpy

        self.assertIsNotNone(chumpy)

    def test_external_preview_worker_applies_numpy_chumpy_compatibility(self):
        ensure_external_chumpy_compat()
        import numpy as np

        for name in ("bool", "int", "float", "complex", "object", "unicode", "str"):
            self.assertIn(name, np.__dict__)

    def test_external_preview_passes_string_paths_to_ffmpeg_merge(self):
        calls = []

        def merge_func(input_paths, output_path):
            calls.append((input_paths, output_path))

        _merge_preview_videos(
            merge_func,
            [Path("incam.mp4"), Path("global.mp4")],
            Path("preview.mp4"),
        )

        self.assertEqual(calls, [(["incam.mp4", "global.mp4"], "preview.mp4")])

    def test_batch_upload_keeps_valid_files_and_reports_invalid_ones(self):
        response = self.client.post(
            "/api/jobs/batch-upload",
            files=[
                ("files", ("good.mp4", b"good-video", "video/mp4")),
                ("files", ("bad.txt", b"not-video", "text/plain")),
            ],
            data={"static_cam": "false"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(len(payload["errors"]), 1)
        self.assertEqual(payload["jobs"][0]["display_name"], "good.mp4")
        self.assertFalse(payload["jobs"][0]["static_cam"])
        self.assertTrue(Path(payload["jobs"][0]["input_video"]).is_file())
        upload_root = self.settings.output_root.parent / "uploads"
        self.assertFalse(upload_root.exists() and any(upload_root.iterdir()))

    def test_artifacts_survive_refresh_and_preview_can_stream_inline(self):
        job = self._make_succeeded_job()
        output_dir = Path(job["output_dir"])
        (output_dir / "1_incam.mp4").write_bytes(b"incam")
        (output_dir / "2_global.mp4").write_bytes(b"global")
        preview_path = output_dir / f"{output_dir.name}_3_incam_global_horiz.mp4"
        preview_path.write_bytes(b"preview")

        job = self.manager.ensure_artifact_bundle(job["job_id"])
        self.assertTrue(Path(job["artifacts"]["artifacts_zip_path"]).is_file())
        self.assertEqual(Path(job["artifacts"]["preview_video_path"]), preview_path)
        with zipfile.ZipFile(job["artifacts"]["artifacts_zip_path"]) as archive:
            self.assertIn("hmr4d_results.pt", archive.namelist())
            self.assertIn(preview_path.name, archive.namelist())

        response = self.client.get(f"/jobs/{job['job_id']}/artifact/preview_video?inline=true")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"preview")
        self.assertNotIn("attachment", response.headers.get("content-disposition", ""))

        reloaded_manager = JobManager(self.settings, SQLiteJobStore(self.settings.db_path))
        reloaded = reloaded_manager.get_job(job["job_id"])
        self.assertEqual(Path(reloaded["artifacts"]["preview_video_path"]), preview_path)

    def test_open_folder_uses_selected_job_output_directory(self):
        job = self._make_succeeded_job("open-folder.mp4")
        with patch(
            "hmr4d.service.server._open_system_directory",
            return_value=job["output_dir"],
        ) as open_directory:
            response = self.client.post(f"/jobs/{job['job_id']}/open-folder")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["path"], job["output_dir"])
        open_directory.assert_called_once_with(job["output_dir"])

    def test_progress_tracks_real_pipeline_stage_percentages(self):
        self.assertEqual(_progress_from_log("[Preprocess] Start!", "process")[:2], (12, "开始视频预处理"))
        self.assertEqual(_progress_from_log("ViTPose: 50%|#####", "process")[:2], (43, "提取人体关键点"))
        self.assertEqual(_progress_from_log("Rendering Global: 50%|#####", "preview")[:2], (68, "渲染全局视角"))

        job = self._make_succeeded_job("progress.mp4")
        job["status"] = "running"
        job["progress_percent"] = 12
        job["progress_stage"] = "开始视频预处理"
        self.store.save_job(job)
        self.manager._log_callback(job["job_id"], "process")("ViTPose: 50%|#####")

        updated = self.manager.get_job(job["job_id"])
        self.assertEqual(updated["progress_percent"], 43)
        self.assertEqual(updated["progress_stage"], "提取人体关键点")
        self.assertNotIn("ViTPose: 50%|#####", updated["logs"])

    def test_missing_legacy_output_path_is_relocated_to_current_runtime(self):
        job = self._make_succeeded_job("legacy.mp4")
        current_output = Path(job["output_dir"])
        job["output_dir"] = f"/app/runtime/jobs/{current_output.name}"
        job["input_video"] = f"/app/runtime/jobs/{current_output.name}/submitted_input.mp4"
        job["artifacts"] = {
            "hmr4d_results_path": f"/app/runtime/jobs/{current_output.name}/hmr4d_results.pt"
        }
        self.store.save_job(job)

        relocated = self.manager.get_job(job["job_id"])
        self.assertEqual(Path(relocated["output_dir"]), current_output)
        self.assertEqual(
            Path(relocated["artifacts"]["hmr4d_results_path"]),
            current_output / "hmr4d_results.pt",
        )

    def test_preview_failure_keeps_main_job_succeeded(self):
        job = self._make_succeeded_job("preview-failure.mp4")
        self.manager._runner = FailingPreviewRunner()
        self.manager.start()
        requested = self.manager.request_preview(job["job_id"])
        self.assertEqual(requested["preview_status"], "queued")

        deadline = time.time() + 3
        updated = None
        while time.time() < deadline:
            updated = self.manager.get_job(job["job_id"])
            if updated.get("preview_status") == "failed":
                break
            time.sleep(0.02)

        self.assertIsNotNone(updated)
        self.assertEqual(updated["status"], "succeeded")
        self.assertEqual(updated["preview_status"], "failed")
        self.assertIn("preview renderer failed", updated["preview_error_summary"])
        self.assertTrue(Path(updated["artifacts"]["hmr4d_results_path"]).is_file())

    def test_old_job_rebuilds_missing_processed_video_before_preview(self):
        job = self._make_succeeded_job("legacy-preview.mp4")
        Path(job["output_dir"], "0_input_video.mp4").unlink()
        runner = RecoveringPreviewRunner()
        self.manager._runner = runner
        self.manager.start()
        self.manager.request_preview(job["job_id"])

        deadline = time.time() + 3
        updated = None
        while time.time() < deadline:
            updated = self.manager.get_job(job["job_id"])
            if updated.get("preview_status") == "succeeded":
                break
            time.sleep(0.02)

        self.assertIsNotNone(updated)
        self.assertEqual(updated["status"], "succeeded")
        self.assertEqual(updated["preview_status"], "succeeded")
        self.assertEqual(runner.process_calls, 1)
        self.assertTrue(Path(updated["artifacts"]["preview_video_path"]).is_file())

    def test_restart_marks_interrupted_preview_without_failing_motion(self):
        job = self._make_succeeded_job("restart.mp4")
        job["preview_status"] = "running"
        job["task_kind"] = "preview"
        self.store.save_job(job)

        recovered_store = SQLiteJobStore(self.settings.db_path)
        recovered = recovered_store.get_job(job["job_id"])
        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["preview_status"], "failed")
        self.assertEqual(recovered["task_kind"], "process")

    def test_gmr_bridge_is_only_advertised_when_callback_is_configured(self):
        standalone = self.client.get("/api/capabilities")
        self.assertEqual(standalone.status_code, 200)
        self.assertFalse(standalone.json()["gmr_bridge_available"])

        bridge_app = create_gvhmr_app(
            self.manager,
            self.settings,
            submit_to_gmr=lambda job_id: {"job_id": f"gmr_{job_id}"},
            manage_lifecycle=False,
        )
        bridge_client = TestClient(bridge_app)
        capabilities = bridge_client.get("/api/capabilities")
        self.assertTrue(capabilities.json()["gmr_bridge_available"])
        response = bridge_client.post("/jobs/example/to-gmr")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], "gmr_example")

    def test_capabilities_report_embedded_backend_by_default(self):
        payload = self.client.get("/api/capabilities").json()
        self.assertEqual(payload["runtime"]["inference_backend"], "embedded")
        self.assertFalse(payload["runtime"]["external_core"]["configured"])
        self.assertEqual(payload["ground_constraints"]["default"], "none")
        options = {item["value"]: item for item in payload["ground_constraints"]["options"]}
        self.assertFalse(options["flat_y"]["enabled"])
        self.assertFalse(options["human3r"]["enabled"])

    def test_external_core_runner_uses_subprocess_protocol(self):
        root = Path(self.temp_dir.name)
        core_root = root / "core"
        (core_root / "hmr4d").mkdir(parents=True)
        (core_root / "hmr4d" / "__init__.py").write_text("", encoding="utf-8")
        (core_root / "tools" / "demo").mkdir(parents=True)
        (core_root / "tools" / "demo" / "demo.py").write_text("", encoding="utf-8")
        (core_root / "inputs" / "checkpoints").mkdir(parents=True)

        fake_python = root / "fake-python"
        fake_python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "print('worker log')\n"
            "command = sys.argv[2]\n"
            "if command == 'process':\n"
            "    output = sys.argv[sys.argv.index('--output-dir') + 1]\n"
            f"    print('{RESULT_PREFIX}' + json.dumps({{'output_dir': output, 'hmr4d_results_path': output + '/hmr4d_results.pt', 'arguments': sys.argv}}))\n"
            "else:\n"
            f"    print('{RESULT_PREFIX}' + json.dumps({{'command': command}}))\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        source = root / "video.mp4"
        source.write_bytes(b"video")
        output_dir = root / "output"
        logs = []

        runner = ExternalCoreRunner(
            core_root=core_root,
            checkpoint_root=core_root / "inputs" / "checkpoints",
            python_executable=fake_python,
        )
        result = runner.process_video(
            source,
            output_dir,
            static_cam=True,
            f_mm=24,
            ground_constraint="flat_y",
            log_callback=logs.append,
        )

        self.assertEqual(Path(result["output_dir"]), output_dir)
        self.assertEqual(Path(result["hmr4d_results_path"]), output_dir / "hmr4d_results.pt")
        self.assertIn("[Core] worker log", logs)
        self.assertIn("--ground-constraint", result["arguments"])
        self.assertIn("flat_y", result["arguments"])

        external_settings = replace(
            self.settings,
            core_root=core_root,
            core_python=str(fake_python),
        )
        external_manager = JobManager(external_settings, self.store)
        self.assertIsInstance(external_manager._get_runner(), ExternalCoreRunner)


if __name__ == "__main__":
    unittest.main()
