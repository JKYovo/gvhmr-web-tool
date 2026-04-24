import argparse
import os
import site
import sys
from contextlib import asynccontextmanager
from pathlib import Path


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

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator

from hmr4d import os_chdir_to_proj_root
from hmr4d.service.assets import ensure_assets
from hmr4d.service.common import ServiceSettings, iter_video_files
from hmr4d.service.manager import JobManager
from hmr4d.service.store import SQLiteJobStore
from hmr4d.service.ui import build_ui


class JobCreateRequest(BaseModel):
    video_source: str
    static_cam: bool = True
    f_mm: int | None = None
    save_intermediate: bool = False
    generate_preview: bool = False


class BatchCreateRequest(BaseModel):
    video_sources: list[str] = Field(default_factory=list)
    input_dir: str | None = None
    static_cam: bool = True
    f_mm: int | None = None
    save_intermediate: bool = False
    generate_preview: bool = False

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


def create_app(settings=None):
    settings, store, manager = create_components(settings=settings)

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
    app.state.store = store
    app.state.manager = manager

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "checkpoint_root": str(settings.checkpoint_root),
            "output_root": str(settings.output_root),
            "db_path": str(settings.db_path),
        }

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui")

    @app.post("/jobs")
    def create_job(request: JobCreateRequest):
        try:
            return manager.submit_job(**request.model_dump())
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

    @app.get("/jobs/{job_id}/artifacts")
    def download_artifacts(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        artifact_path = job.get("artifacts", {}).get("artifacts_zip_path")
        if not artifact_path or not Path(artifact_path).exists():
            raise HTTPException(status_code=404, detail=f"No artifact bundle for job: {job_id}")
        return FileResponse(artifact_path, filename=f"{job_id}_artifacts.zip")

    @app.post("/batches")
    def create_batch(request: BatchCreateRequest):
        video_sources = list(request.video_sources)
        if request.input_dir:
            input_dir = Path(request.input_dir).expanduser().resolve()
            if not input_dir.exists():
                raise HTTPException(status_code=404, detail=f"Directory not found: {input_dir}")
            video_sources.extend(str(path) for path in iter_video_files(input_dir))
        batch = manager.submit_batch(
            video_sources=video_sources,
            static_cam=request.static_cam,
            f_mm=request.f_mm,
            save_intermediate=request.save_intermediate,
            generate_preview=request.generate_preview,
            input_dir=request.input_dir,
        )
        return batch

    ui = build_ui(manager, settings)
    app = gr.mount_gradio_app(app, ui, path="/ui")
    return app


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
