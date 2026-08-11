#!/usr/bin/env python3
"""Run the official Human3R demo without starting the interactive Viser server."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HUMAN3R_ROOT = REPO_ROOT / "third-party" / "Human3R"
DEFAULT_DINOV2_ROOT = REPO_ROOT / "third-party" / "dinov2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--human3r_root", type=Path, default=DEFAULT_HUMAN3R_ROOT)
    parser.add_argument("--dinov2_root", type=Path, default=DEFAULT_DINOV2_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Frames per memory-bounded chunk; one previous frame is overlapped.",
    )
    parser.add_argument("--use_ttt3r", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    return parser.parse_args()


def extract_input_frames(
    seq_path: Path, max_frames: int | None, subsample: int
) -> tuple[list[str], str | None, dict[str, float | int | str]]:
    if subsample < 1:
        raise ValueError("--subsample must be at least 1")
    if seq_path.is_dir():
        paths = sorted(str(path) for path in seq_path.iterdir() if path.is_file())
        source_frames = len(paths) if max_frames is None else min(len(paths), max_frames)
        paths = paths[:max_frames:subsample]
        return paths, None, {
            "source_type": "directory",
            "source_frames": source_frames,
            "selected_frames": len(paths),
        }

    capture = cv2.VideoCapture(str(seq_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {seq_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = total_frames if max_frames is None else min(total_frames, max_frames)
    temp_dir = tempfile.mkdtemp(prefix="human3r_frames_")
    paths: list[str] = []
    frame_idx = 0
    try:
        while frame_idx < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % subsample == 0:
                frame_path = Path(temp_dir) / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
                paths.append(str(frame_path))
            frame_idx += 1
    finally:
        capture.release()

    return paths, temp_dir, {
        "source_type": "video",
        "source_fps": fps,
        "source_frames": total_frames,
        "read_frames": frame_idx,
        "selected_frames": len(paths),
    }


def main() -> None:
    args = parse_args()
    seq_path = args.seq_path.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    human3r_root = args.human3r_root.resolve()
    dinov2_root = args.dinov2_root.resolve()

    for path, label in (
        (seq_path, "input"),
        (model_path, "model"),
        (human3r_root, "Human3R"),
        (dinov2_root, "DINOv2"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = ("depth", "conf", "color", "camera", "smpl", "color_smpl")
    if any((output_dir / name).exists() for name in output_names):
        raise FileExistsError(f"Output directory already contains Human3R results: {output_dir}")

    sys.path.insert(0, str(human3r_root))
    # The official helper assumes checkpoints live inside Human3R/src. Keep
    # assets outside the submodule while preserving its sibling imports.
    sys.path.insert(0, str(human3r_root / "src"))
    os.chdir(human3r_root)
    import demo as human3r_demo  # noqa: PLC0415

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment")

    # Human3R asks torch.hub to resolve DINOv2 from GitHub even when
    # pretrained=False. Route that one repository to the pinned local
    # submodule so inference is reproducible and works without GitHub API.
    torch_hub_load = torch.hub.load

    def load_hub_repo(repo_or_dir, model_name, *hub_args, **hub_kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            return torch_hub_load(
                str(dinov2_root),
                model_name,
                *hub_args,
                source="local",
                **hub_kwargs,
            )
        return torch_hub_load(repo_or_dir, model_name, *hub_args, **hub_kwargs)

    torch.hub.load = load_hub_repo
    human3r_demo.add_path_to_dust3r(str(model_path))
    from src.dust3r.inference import inference_recurrent_lighter  # noqa: PLC0415
    from src.dust3r.model import ARCroco3DStereo  # noqa: PLC0415

    image_paths, temp_dir, source_meta = extract_input_frames(
        seq_path, args.max_frames, args.subsample
    )
    if not image_paths:
        raise RuntimeError(f"No input frames selected from {seq_path}")

    start_time = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"Loading Human3R checkpoint: {model_path}", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(model_path)).to(device).eval()

    image_resolution = getattr(model, "mhmr_img_res", None)
    if args.chunk_size < 2:
        raise ValueError("--chunk_size must be at least 2")
    if args.render_video and len(image_paths) > args.chunk_size:
        raise ValueError("Chunked --render_video is not supported; render after reconstruction")

    inference_seconds = 0.0
    previous_global_pose: np.ndarray | None = None
    chunk_root = output_dir / ".chunks"
    chunk_root.mkdir(parents=True, exist_ok=False)

    try:
        for start in range(0, len(image_paths), args.chunk_size):
            end = min(start + args.chunk_size, len(image_paths))
            overlap = 1 if start else 0
            chunk_paths = image_paths[start - overlap : end]
            print(
                f"Preparing frames {start}-{end - 1} "
                f"({len(chunk_paths)} views including {overlap} overlap)",
                flush=True,
            )
            # Chunk boundaries reproduce the official reset_interval behavior:
            # the prior frame seeds a fresh recurrent state, then is discarded.
            views = human3r_demo.prepare_input(
                img_paths=chunk_paths,
                img_mask=[True] * len(chunk_paths),
                size=args.size,
                revisit=1,
                update=True,
                img_res=image_resolution,
                reset_interval=len(chunk_paths) + 1,
            )

            inference_start = time.time()
            outputs, _ = inference_recurrent_lighter(
                views, model, device, use_ttt3r=args.use_ttt3r
            )
            inference_seconds += time.time() - inference_start

            chunk_dir = chunk_root / f"{start:06d}_{end:06d}"
            print(f"Saving chunk to {chunk_dir}", flush=True)
            human3r_demo.prepare_output(
                outputs,
                str(chunk_dir),
                revisit=1,
                use_pose=True,
                save=True,
                render=args.render,
                render_video=args.render_video,
                img_res=image_resolution,
                subsample=args.subsample,
            )

            local_camera_paths = sorted((chunk_dir / "camera").glob("*.npz"))
            if len(local_camera_paths) != len(chunk_paths):
                raise RuntimeError(
                    f"Chunk camera count mismatch: {len(local_camera_paths)} != {len(chunk_paths)}"
                )
            local_poses = [np.load(path)["pose"] for path in local_camera_paths]
            if start:
                if previous_global_pose is None:
                    raise RuntimeError("Missing previous global camera pose")
                alignment = previous_global_pose @ np.linalg.inv(local_poses[0])
            else:
                alignment = np.eye(4, dtype=np.float32)
            aligned_poses = [alignment @ pose for pose in local_poses]

            first_local = overlap
            for local_idx in range(first_local, len(chunk_paths)):
                global_idx = start + local_idx - overlap
                for name in output_names:
                    source_dir = chunk_dir / name
                    if not source_dir.exists():
                        continue
                    candidates = list(source_dir.glob(f"{local_idx:06d}.*"))
                    if len(candidates) != 1:
                        raise RuntimeError(
                            f"Expected one {name} file for local frame {local_idx}, got {candidates}"
                        )
                    destination_dir = output_dir / name
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    destination = destination_dir / f"{global_idx:06d}{candidates[0].suffix}"
                    if name == "camera":
                        camera_data = np.load(candidates[0])
                        np.savez(
                            destination,
                            pose=aligned_poses[local_idx],
                            intrinsics=camera_data["intrinsics"],
                        )
                    else:
                        candidates[0].replace(destination)

            previous_global_pose = aligned_poses[-1]
            shutil.rmtree(chunk_dir)
            del outputs, views, local_poses, aligned_poses
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(chunk_root, ignore_errors=True)

    metadata = {
        "input": str(seq_path),
        "model": str(model_path),
        "human3r_root": str(human3r_root),
        "dinov2_root": str(dinov2_root),
        "device": device,
        "size": args.size,
        "subsample": args.subsample,
        "reset_interval": args.reset_interval,
        "chunk_size": args.chunk_size,
        "use_ttt3r": args.use_ttt3r,
        "inference_seconds": inference_seconds,
        "seconds_per_selected_frame": inference_seconds / len(image_paths),
        "total_seconds": time.time() - start_time,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        ),
        **source_meta,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
