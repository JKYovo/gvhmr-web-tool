"""CPU/small-GPU regression checks for the P6 inference optimizations."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from hmr4d.network.base_arch.transformer.encoder_rope import RoPEAttention
from hmr4d.api.video_to_data import _prepare_video_copy
from hmr4d.utils.perf import StageProfiler
from hmr4d.utils.preproc.content_cache import PreprocessContentCache
from hmr4d.utils.video_io_utils import get_video_fps_duration, get_video_lwh, get_writer, normalize_video_fps


def make_dense_local_mask(length, window, device):
    mask = torch.ones((length, length), device=device, dtype=torch.bool)
    for index in range(length):
        start = max(0, index - window // 2)
        end = min(length, index + window // 2)
        end = max(window, end)
        start = min(length - window, start)
        mask[index, start:end] = False
    return mask


def test_local_attention():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = RoPEAttention(64, 4, dropout=0.0, attention_chunk_size=64).eval().to(device)
    for length in (121, 257):
        data = torch.randn(1, length, 64, device=device)
        padding = torch.zeros(1, length, dtype=torch.bool, device=device)
        dense_mask = make_dense_local_mask(length, 120, device)
        with torch.inference_mode():
            dense = module(data, dense_mask, padding)
            local = module(data, ("local", 120), padding)
        torch.testing.assert_close(local, dense, rtol=2e-5, atol=2e-6)


def test_video_normalization(temporary_dir):
    source = temporary_dir / "source_60fps.mp4"
    output = temporary_dir / "output_30fps.mp4"
    writer = get_writer(source, fps=60, crf=17)
    for frame_id in range(12):
        frame = np.full((32, 48, 3), frame_id * 10, dtype=np.uint8)
        writer.write_frame(frame)
    writer.close()
    metadata = normalize_video_fps(source, output, target_fps=30, crf=17)
    assert get_video_lwh(source)[0] == 12
    assert get_video_lwh(output)[0] == 6
    fps, duration = get_video_fps_duration(output)
    assert fps == 30
    assert abs(duration - metadata["source_duration"]) <= 0.05


def test_web_replaces_stale_relabelled_video(temporary_dir):
    source = temporary_dir / "web_source_60fps.mp4"
    output = temporary_dir / "0_input_video.mp4"

    source_writer = get_writer(source, fps=60, crf=17)
    stale_writer = get_writer(output, fps=30, crf=17)
    for frame_id in range(60):
        frame = np.full((32, 48, 3), frame_id * 4, dtype=np.uint8)
        source_writer.write_frame(frame)
        # Reproduce the old bug: keep every frame and only label it 30 FPS.
        stale_writer.write_frame(frame)
    source_writer.close()
    stale_writer.close()

    stale_fps, stale_duration = get_video_fps_duration(output)
    assert stale_fps == 30
    assert get_video_lwh(output)[0] == 60
    assert stale_duration > 1.9

    _prepare_video_copy(source, output)

    source_fps, source_duration = get_video_fps_duration(source)
    output_fps, output_duration = get_video_fps_duration(output)
    assert source_fps == 60
    assert output_fps == 30
    assert get_video_lwh(output)[0] == 30
    assert abs(output_duration - source_duration) <= 0.05


def test_content_cache(temporary_dir):
    video = temporary_dir / "video.bin"
    video.write_bytes(b"content-addressed-video")
    source = temporary_dir / "source.pt"
    destination = temporary_dir / "destination.pt"
    value = torch.arange(12)
    torch.save(value, source)
    cache = PreprocessContentCache(temporary_dir / "cache", video)
    options = {"backend": "unit-test", "version": 1}
    cache.store("features", options, source)
    assert cache.restore("features", options, destination)
    assert torch.equal(torch.load(destination), value)
    assert not cache.restore("features", {**options, "version": 2}, destination.with_name("miss.pt"))


def test_profiler(temporary_dir):
    output = temporary_dir / "performance.json"
    profiler = StageProfiler(output, enabled=True, metadata={"test": True})
    with profiler.section("unit"):
        _ = sum(range(10))
    profiler.write()
    payload = json.loads(output.read_text())
    assert payload["metadata"]["test"] is True
    assert payload["stages"]["unit"]["calls"] == 1
    assert payload["wall_seconds"] >= payload["stages"]["unit"]["seconds"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="gvhmr-p6-test-") as directory:
        root = Path(directory)
        test_video_normalization(root)
        test_web_replaces_stale_relabelled_video(root)
        test_content_cache(root)
        test_profiler(root)
    test_local_attention()
    print("P6 optimization checks passed")
