import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class StageProfiler:
    """Small opt-in wall-clock profiler with CUDA synchronization."""

    def __init__(self, output_path=None, enabled=False, metadata=None):
        self.output_path = Path(output_path) if output_path is not None else None
        self.enabled = bool(enabled)
        self.started = time.perf_counter()
        self.metadata = dict(metadata or {})
        self.stages = {}

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        self._sync_cuda()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda()
            elapsed = time.perf_counter() - started
            record = self.stages.setdefault(name, {"seconds": 0.0, "calls": 0})
            record["seconds"] += elapsed
            record["calls"] += 1

    def add_metadata(self, **values):
        if self.enabled:
            self.metadata.update(values)

    def write(self):
        if not self.enabled or self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "stages": self.stages,
            "wall_seconds": time.perf_counter() - self.started,
            "sum_of_stage_seconds_including_nested": sum(item["seconds"] for item in self.stages.values()),
        }
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            payload["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output_path)


class NullProfiler:
    @contextmanager
    def section(self, _name):
        yield
