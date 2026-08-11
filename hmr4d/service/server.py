import argparse
import mimetypes
import os
import shutil
import site
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from shutil import copyfileobj
from typing import Annotated


def _sanitize_python_path():
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    user_site_candidates = []
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if isinstance(user_site, str):
        user_site_candidates.append(Path(user_site).resolve())
    elif isinstance(user_site, (list, tuple)):
        user_site_candidates.extend(Path(path).resolve() for path in user_site)

    user_base = os.environ.get("PYTHONUSERBASE")
    if user_base:
        user_site_candidates.append(Path(user_base).expanduser().resolve())
    user_site_candidates.append((Path.home() / ".local").resolve())

    sanitized = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            sanitized.append(entry)
            continue
        if any(candidate == resolved or candidate in resolved.parents for candidate in user_site_candidates):
            continue
        sanitized.append(entry)
    sys.path[:] = sanitized


_sanitize_python_path()

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from hmr4d import os_chdir_to_proj_root
from hmr4d.service.assets import ensure_assets, inspect_assets
from hmr4d.service.common import ServiceSettings, ensure_dir, iter_video_files, make_job_id
from hmr4d.service.external_core import inspect_external_core
from hmr4d.service.manager import JobManager
from hmr4d.service.store import SQLiteJobStore


STATIC_APP_DIR = Path(__file__).resolve().parent / "static_app"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class JobCreateRequest(BaseModel):
    video_source: str
    static_cam: bool = True
    f_mm: int | None = None
    save_intermediate: bool = False
    generate_preview: bool = False
    ground_constraint: str = "none"
    display_name: str | None = None
    output_dir: str | None = None


class BatchCreateRequest(BaseModel):
    video_sources: list[str] = Field(default_factory=list)
    input_dir: str | None = None
    static_cam: bool = True
    f_mm: int | None = None
    save_intermediate: bool = False
    generate_preview: bool = False
    ground_constraint: str = "none"
    output_dir: str | None = None

    @model_validator(mode="after")
    def validate_sources(self):
        if not self.video_sources and not self.input_dir:
            raise ValueError("Either video_sources or input_dir must be provided.")
        return self


def create_components(settings=None):
    settings = settings or ServiceSettings.from_env()
    settings.ensure_runtime_dirs()
    store = SQLiteJobStore(settings.db_path)
    manager = JobManager(settings, store)
    return settings, store, manager


def _safe_stem(value):
    stem = Path(value or "video").stem
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem).strip("_")
    return safe or "video"


def _upload_root(settings):
    return settings.output_root.parent / "uploads"


def _save_upload(upload_file, upload_dir):
    filename = Path(upload_file.filename or "video").name
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {suffix or '(none)'}")
    unique = make_job_id().replace("job_", "")[:6]
    upload_path = upload_dir / f"{_safe_stem(filename)}_{unique}{suffix}"
    with upload_path.open("wb") as file:
        copyfileobj(upload_file.file, file)
    if upload_path.stat().st_size == 0:
        raise ValueError(f"Uploaded file is empty: {filename}")
    return upload_path


def _optional_output_dir(value):
    value = str(value or "").strip()
    return value or None


def _native_pick_directory(settings, initial_path=None):
    workspace_root = settings.output_root.parent
    initial = Path(initial_path).expanduser() if initial_path else workspace_root
    try:
        initial = initial.resolve()
    except OSError:
        initial = workspace_root.resolve()
    if not initial.exists():
        initial = workspace_root.resolve()
    if initial.is_file():
        initial = initial.parent

    picker_env = {
        "DISPLAY": os.environ.get("DISPLAY", ""),
        "XAUTHORITY": os.environ.get("XAUTHORITY", ""),
        "DBUS_SESSION_BUS_ADDRESS": os.environ.get("DBUS_SESSION_BUS_ADDRESS", ""),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "zh_CN.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    picker_env = {key: value for key, value in picker_env.items() if value}
    zenity = shutil.which("zenity")
    if zenity:
        result = subprocess.run(
            [
                zenity,
                "--file-selection",
                "--directory",
                "--title=选择 GVHMR 输出根目录",
                f"--filename={str(initial)}/",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=picker_env,
            check=False,
        )
        if result.returncode == 0:
            selected = result.stdout.strip()
            return str(Path(selected).expanduser().resolve()) if selected else None
        if result.returncode == 1:
            return None
        raise RuntimeError(result.stderr.strip() or "System directory picker failed.")

    raise RuntimeError(result.stderr.strip() or "No native directory picker is available on this server.")


def _open_system_directory(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    if sys.platform == "darwin":
        opener = shutil.which("open")
        command = [opener, str(path)] if opener else None
    elif os.name == "nt":
        command = ["explorer", str(path)]
    else:
        opener = shutil.which("xdg-open")
        command = [opener, str(path)] if opener else None
    if not command:
        raise RuntimeError("No system file manager opener is available on this server.")

    opener_env = {
        "DISPLAY": os.environ.get("DISPLAY", ""),
        "XAUTHORITY": os.environ.get("XAUTHORITY", ""),
        "DBUS_SESSION_BUS_ADDRESS": os.environ.get("DBUS_SESSION_BUS_ADDRESS", ""),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "zh_CN.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    }
    opener_env = {key: value for key, value in opener_env.items() if value}
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=opener_env)
    return str(path)


def _artifact_path(job, artifact_key):
    artifacts = job.get("artifacts", {})
    key_map = {
        "meta": artifacts.get("meta_path"),
        "hmr4d_results": artifacts.get("hmr4d_results_path"),
        "raw_hmr4d_results": artifacts.get("raw_hmr4d_results_path"),
        "flat_ground_y_results": artifacts.get("flat_ground_y_results_path"),
        "ground_constraint_metrics": artifacts.get("ground_constraint_metrics_path"),
        "incam_video": artifacts.get("incam_video_path"),
        "global_video": artifacts.get("global_video_path"),
        "preview_video": artifacts.get("preview_video_path"),
        "zip": artifacts.get("artifacts_zip_path"),
    }
    path = key_map.get(artifact_key)
    if not path:
        return None
    path = Path(path)
    return path if path.exists() else None


def _runtime_capabilities(settings):
    asset_status = inspect_assets(settings.checkpoint_root)
    gpu_available = False
    gpu_name = None
    gpu_error = None
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        gpu_error = str(exc).strip() or exc.__class__.__name__

    core = inspect_external_core(settings.core_root, settings.core_python)
    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_error": gpu_error,
        "assets_ready": asset_status["ready"],
        "missing_assets": [item["label"] for item in asset_status["missing"]],
        "inference_ready": gpu_available and asset_status["ready"] and core["ready"],
        "inference_backend": core["backend"],
        "external_core": core,
    }


def _ground_constraint_capabilities(runtime):
    core = runtime["external_core"]
    root = Path(core["root"]) if core.get("root") else None
    flat_y_available = bool(
        core.get("ready")
        and root is not None
        and (root / "tools" / "bench" / "human3r_p2y" / "apply_contact_floor_y.py").is_file()
    )
    return {
        "default": "flat_y" if flat_y_available else "none",
        "options": [
            {"value": "none", "label": "不启用", "enabled": True},
            {"value": "flat_y", "label": "自动平地约束", "enabled": flat_y_available},
            {
                "value": "human3r",
                "label": "Human3R 场景约束",
                "enabled": False,
                "reason": "尚未启用",
            },
        ],
    }


def create_gvhmr_app(manager, settings, *, submit_to_gmr=None, manage_lifecycle=False):
    lifespan = None
    if manage_lifecycle:
        @asynccontextmanager
        async def lifespan(app):
            os_chdir_to_proj_root()
            if settings.sync_assets_on_boot:
                ensure_assets(settings.checkpoint_root, logger=print)
            manager.start()
            try:
                yield
            finally:
                manager.shutdown()

    app = FastAPI(title="GVHMR Web Service", lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager

    @app.get("/health")
    def health():
        runtime = _runtime_capabilities(settings)
        return {
            "status": "ok",
            "inference_ready": runtime["inference_ready"],
            "checkpoint_root": str(settings.checkpoint_root),
            "output_root": str(settings.output_root),
            "db_path": str(settings.db_path),
            "inference_backend": runtime["inference_backend"],
            "external_core": runtime["external_core"],
        }

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(STATIC_APP_DIR / "index.html")

    @app.get("/api/capabilities")
    def capabilities():
        runtime = _runtime_capabilities(settings)
        return {
            "service": "gvhmr-web",
            "api_version": 1,
            "video_extensions": sorted(VIDEO_EXTENSIONS),
            "defaults": {
                "static_cam": True,
                "f_mm": None,
                "save_intermediate": False,
                "generate_preview": False,
                "ground_constraint": _ground_constraint_capabilities(runtime)["default"],
            },
            "ground_constraints": _ground_constraint_capabilities(runtime),
            "paths": {
                "checkpoint_root": str(settings.checkpoint_root),
                "output_root": str(settings.output_root),
                "db_path": str(settings.db_path),
            },
            "gmr_bridge_available": submit_to_gmr is not None,
            "runtime": runtime,
        }

    @app.get("/api/fs/pick-directory")
    def pick_directory(initial: str | None = None):
        try:
            selected = _native_pick_directory(settings, initial)
            return {"path": selected, "cancelled": selected is None}
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/upload")
    def upload_job(
        file: Annotated[UploadFile, File()],
        static_cam: Annotated[bool, Form()] = True,
        f_mm: Annotated[int | None, Form()] = None,
        save_intermediate: Annotated[bool, Form()] = False,
        generate_preview: Annotated[bool, Form()] = False,
        ground_constraint: Annotated[str, Form()] = "none",
        output_dir: Annotated[str | None, Form()] = None,
    ):
        upload_dir = ensure_dir(_upload_root(settings) / make_job_id().replace("job_", "upload_"))
        try:
            video_path = _save_upload(file, upload_dir)
            return manager.submit_job(
                video_source=video_path,
                static_cam=static_cam,
                f_mm=f_mm,
                save_intermediate=save_intermediate,
                generate_preview=generate_preview,
                ground_constraint=ground_constraint,
                display_name=Path(file.filename or video_path.name).name,
                output_dir=_optional_output_dir(output_dir),
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    @app.post("/api/jobs/batch-upload")
    def batch_upload_jobs(
        files: Annotated[list[UploadFile], File()],
        static_cam: Annotated[bool, Form()] = True,
        f_mm: Annotated[int | None, Form()] = None,
        save_intermediate: Annotated[bool, Form()] = False,
        generate_preview: Annotated[bool, Form()] = False,
        ground_constraint: Annotated[str, Form()] = "none",
        output_dir: Annotated[str | None, Form()] = None,
    ):
        upload_dir = ensure_dir(_upload_root(settings) / make_job_id().replace("job_", "batch_upload_"))
        try:
            video_paths = []
            display_names = []
            errors = []
            for upload_file in files:
                try:
                    video_path = _save_upload(upload_file, upload_dir)
                    video_paths.append(str(video_path))
                    display_names.append(Path(upload_file.filename or video_path.name).name)
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    errors.append({"file": upload_file.filename, "error": str(exc)})
            if not video_paths:
                return {"batch_id": None, "jobs": [], "errors": errors}
            batch = manager.submit_batch(
                video_sources=video_paths,
                static_cam=static_cam,
                f_mm=f_mm,
                save_intermediate=save_intermediate,
                generate_preview=generate_preview,
                ground_constraint=ground_constraint,
                input_dir=None,
                output_dir=_optional_output_dir(output_dir),
                display_names=display_names,
            )
            jobs = [manager.get_job(job_id) for job_id in batch.get("job_ids", [])]
            return {"batch_id": batch["batch_id"], "jobs": [job for job in jobs if job], "errors": errors}
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    @app.post("/jobs")
    def create_job(request: JobCreateRequest):
        try:
            return manager.submit_job(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/jobs")
    def list_jobs(limit: int = 50):
        return manager.list_jobs(limit=limit)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job

    @app.get("/batches")
    def list_batches(limit: int = 20):
        return manager.list_batches(limit=limit)

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: str):
        batch = manager.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
        return batch

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = manager.cancel_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return manager.get_job(job_id)

    @app.post("/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        try:
            job = manager.retry_job(job_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return manager.get_job(job_id)

    @app.post("/jobs/{job_id}/preview")
    def preview_job(job_id: str):
        try:
            job = manager.request_preview(job_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return manager.get_job(job_id)

    @app.post("/jobs/{job_id}/open-folder")
    def open_job_folder(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        try:
            path = _open_system_directory(job["output_dir"])
        except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"path": path}

    @app.get("/jobs/{job_id}/artifacts")
    def download_artifacts(job_id: str):
        job = manager.ensure_artifact_bundle(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        artifact_path = job.get("artifacts", {}).get("artifacts_zip_path")
        if not artifact_path or not Path(artifact_path).exists():
            raise HTTPException(status_code=404, detail=f"No artifact bundle for job: {job_id}")
        return FileResponse(artifact_path, filename=f"{job_id}_artifacts.zip")

    @app.get("/jobs/{job_id}/artifact/{artifact_key}")
    def download_artifact(job_id: str, artifact_key: str, inline: bool = False):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        path = _artifact_path(job, artifact_key)
        if path is None:
            raise HTTPException(status_code=404, detail=f"No artifact {artifact_key} for job: {job_id}")
        if inline:
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return FileResponse(path, media_type=media_type)
        return FileResponse(path, filename=path.name)

    @app.post("/jobs/{job_id}/to-gmr")
    def submit_job_to_gmr(job_id: str):
        if submit_to_gmr is None:
            raise HTTPException(status_code=400, detail="GMR bridge is not configured.")
        try:
            return submit_to_gmr(job_id)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/batches")
    def create_batch(request: BatchCreateRequest):
        video_sources = list(request.video_sources)
        if request.input_dir:
            input_dir = Path(request.input_dir).expanduser().resolve()
            if not input_dir.exists():
                raise HTTPException(status_code=404, detail=f"Directory not found: {input_dir}")
            video_sources.extend(str(path) for path in iter_video_files(input_dir))
        try:
            return manager.submit_batch(
                video_sources=video_sources,
                static_cam=request.static_cam,
                f_mm=request.f_mm,
                save_intermediate=request.save_intermediate,
                generate_preview=request.generate_preview,
                ground_constraint=request.ground_constraint,
                input_dir=request.input_dir,
                output_dir=_optional_output_dir(request.output_dir),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.mount("/static", StaticFiles(directory=STATIC_APP_DIR), name="gvhmr_static")
    return app


def create_app(settings=None):
    settings, store, manager = create_components(settings=settings)
    return create_gvhmr_app(manager, settings, manage_lifecycle=True)


def parse_args():
    settings = ServiceSettings.from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
