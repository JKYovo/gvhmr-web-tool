import json
import os
import signal
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path


RESULT_PREFIX = "__GVHMR_CORE_RESULT__="


def inspect_external_core(core_root, core_python=None):
    if core_root is None:
        return {
            "configured": False,
            "ready": True,
            "backend": "embedded",
            "root": None,
            "python": None,
            "missing": [],
        }

    root = Path(core_root).expanduser().resolve()
    python_value = str(core_python or "python")
    python_path = Path(python_value).expanduser()
    python_exists = python_path.is_file() if python_path.is_absolute() else shutil.which(python_value) is not None
    required = (
        root / "hmr4d" / "__init__.py",
        root / "tools" / "demo" / "demo.py",
        root / "inputs" / "checkpoints",
    )
    missing = [str(path) for path in required if not path.exists()]
    if not python_exists:
        missing.append(f"Python executable: {python_value}")
    return {
        "configured": True,
        "ready": not missing,
        "backend": "external_core",
        "root": str(root),
        "python": python_value,
        "missing": missing,
    }


class ExternalCoreRunner:
    def __init__(self, *, core_root, checkpoint_root, python_executable=None):
        self.core_root = Path(core_root).expanduser().resolve()
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        self.python_executable = str(python_executable or "python")
        self.worker_path = Path(__file__).resolve().with_name("external_core_worker.py")
        self._process_lock = threading.Lock()
        self._active_process = None
        status = inspect_external_core(self.core_root, self.python_executable)
        if not status["ready"]:
            raise RuntimeError("GVHMR external core is not ready:\n" + "\n".join(status["missing"]))

    def _run(self, command, arguments, log_callback=None):
        cmd = [
            self.python_executable,
            str(self.worker_path),
            command,
            "--core-root",
            str(self.core_root),
            "--checkpoint-root",
            str(self.checkpoint_root),
            *arguments,
        ]
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(self.core_root)
        env["GVHMR_CHECKPOINT_ROOT"] = str(self.checkpoint_root)
        recent_output = deque(maxlen=40)
        result = None
        process = subprocess.Popen(
            cmd,
            cwd=self.core_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._process_lock:
            self._active_process = process
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                if line.startswith(RESULT_PREFIX):
                    try:
                        result = json.loads(line[len(RESULT_PREFIX) :])
                    except json.JSONDecodeError as exc:
                        process.wait()
                        raise RuntimeError(f"GVHMR core returned invalid JSON: {exc}") from exc
                    continue
                recent_output.append(line)
                if log_callback:
                    log_callback(f"[Core] {line}")
        finally:
            process.stdout.close()
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None

        return_code = process.wait()
        if return_code != 0:
            detail = "\n".join(recent_output) or f"exit code {return_code}"
            raise RuntimeError(f"GVHMR external core failed:\n{detail}")
        if result is None:
            detail = "\n".join(recent_output)
            raise RuntimeError(f"GVHMR external core returned no result.\n{detail}".rstrip())
        return result

    def close(self):
        with self._process_lock:
            process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            process.wait()

    def probe(self, log_callback=None):
        return self._run("probe", [], log_callback=log_callback)

    def process_video(
        self,
        video_path,
        output_dir,
        static_cam,
        f_mm=None,
        save_intermediate=False,
        *,
        ground_constraint="none",
        use_dpvo=False,
        verbose=False,
        log_callback=None,
    ):
        arguments = [
            "--video",
            str(Path(video_path).expanduser().resolve()),
            "--output-dir",
            str(Path(output_dir).expanduser().resolve()),
            "--ground-constraint",
            str(ground_constraint),
        ]
        if static_cam:
            arguments.append("--static-cam")
        if f_mm not in (None, "", 0):
            arguments.extend(("--f-mm", str(int(f_mm))))
        if save_intermediate:
            arguments.append("--save-intermediate")
        if use_dpvo:
            arguments.append("--use-dpvo")
        if verbose:
            arguments.append("--verbose")
        return self._run("process", arguments, log_callback=log_callback)

    def generate_preview(self, output_dir, log_callback=None):
        return self._run(
            "preview",
            ["--output-dir", str(Path(output_dir).expanduser().resolve())],
            log_callback=log_callback,
        )
