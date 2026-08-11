import hashlib
import json
import shutil
import uuid
from pathlib import Path


CACHE_SCHEMA = "p6-preprocess-v1"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class PreprocessContentCache:
    """Content-addressed cache for immutable preprocessing tensors."""

    def __init__(self, root, video_path):
        self.enabled = root not in (None, "", "none", "None")
        self.video_sha256 = None
        self.sequence_dir = None
        if self.enabled:
            self.video_sha256 = sha256_file(video_path)
            self.sequence_dir = Path(root).expanduser() / CACHE_SCHEMA / self.video_sha256

    @staticmethod
    def _options_digest(options):
        encoded = json.dumps(options, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def path_for(self, stage, options):
        if not self.enabled:
            return None
        return self.sequence_dir / f"{stage}-{self._options_digest(options)}.pt"

    def restore(self, stage, options, destination):
        source = self.path_for(stage, options)
        destination = Path(destination)
        if source is None or not source.is_file():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.cache-tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        return True

    def store(self, stage, options, source):
        destination = self.path_for(stage, options)
        source = Path(source)
        if destination is None or not source.is_file():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            return
        temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
        shutil.copy2(source, temporary)
        try:
            temporary.replace(destination)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
