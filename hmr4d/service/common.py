import json
import os
import socket
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from hmr4d import PROJ_ROOT, get_checkpoint_root


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_job_id():
    return f"job_{uuid4().hex[:12]}"


def make_batch_id():
    return f"batch_{uuid4().hex[:12]}"


def short_id(value):
    return value.split("_", 1)[-1][:8]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_dated_output_dir(root, source_name, now=None):
    root = ensure_dir(root)
    stem = Path(source_name or "video").stem.strip() or "video"
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    stamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    base = root / f"{stem}_{stamp}"
    candidate = base
    version = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = root / f"{base.name}_{version}"
            version += 1


def resolve_runtime_path(path_like, *, default):
    value = os.environ.get(path_like) if isinstance(path_like, str) else None
    if value:
        return Path(value).expanduser().resolve()
    return Path(default).expanduser().resolve()


@dataclass(frozen=True)
class ServiceSettings:
    checkpoint_root: Path
    output_root: Path
    batch_root: Path
    db_path: Path
    host: str
    port: int
    sync_assets_on_boot: bool
    core_root: Path | None = None
    core_python: str | None = None

    @classmethod
    def from_env(cls):
        checkpoint_root = Path(
            os.environ.get("GVHMR_CHECKPOINT_ROOT", str(get_checkpoint_root()))
        ).expanduser().resolve()
        output_root = Path(
            os.environ.get("GVHMR_OUTPUT_ROOT", str(PROJ_ROOT / "runtime/jobs"))
        ).expanduser().resolve()
        batch_root = Path(
            os.environ.get("GVHMR_BATCH_ROOT", str(PROJ_ROOT / "runtime/batches"))
        ).expanduser().resolve()
        db_path = Path(
            os.environ.get("GVHMR_DB_PATH", str(PROJ_ROOT / "runtime/db/job_db.sqlite"))
        ).expanduser().resolve()
        host = os.environ.get("GVHMR_HOST", "127.0.0.1")
        port = int(os.environ.get("GVHMR_PORT", "7860"))
        sync_assets_on_boot = os.environ.get("GVHMR_SYNC_ASSETS_ON_BOOT", "0") == "1"
        core_root_value = os.environ.get("GVHMR_CORE_ROOT", "").strip()
        core_root = Path(core_root_value).expanduser().resolve() if core_root_value else None
        core_python = os.environ.get("GVHMR_CORE_PYTHON", "").strip() or sys.executable
        return cls(
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            batch_root=batch_root,
            db_path=db_path,
            host=host,
            port=port,
            sync_assets_on_boot=sync_assets_on_boot,
            core_root=core_root,
            core_python=core_python,
        )

    def ensure_runtime_dirs(self):
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.output_root)
        ensure_dir(self.batch_root)
        ensure_dir(self.db_path.parent)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def zip_artifacts(output_path, files):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    stored_extensions = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".npz", ".pt", ".zip"}
    with zipfile.ZipFile(output_path, "w") as archive:
        for file_path, arcname in files:
            if file_path and Path(file_path).exists():
                compression = (
                    zipfile.ZIP_STORED
                    if Path(file_path).suffix.lower() in stored_extensions
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(file_path, arcname=arcname, compress_type=compression)
    return output_path


def guess_lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def terminal_job_states():
    return {"succeeded", "failed", "cancelled"}


def iter_video_files(folder):
    folder = Path(folder)
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            yield path.resolve()
