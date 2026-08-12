#!/usr/bin/env python3
"""Convert one WebTool GVHMR result tensor to a validated SONIC reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hmr4d.utils.sonic import convert_smplx_to_sonic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, destination: Path, source_fps: float = 30.0) -> dict[str, object]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GVHMR result does not exist: {source}")
    result = torch.load(source, map_location="cpu", weights_only=False)
    params = result["smpl_params_global"]
    root = params["global_orient"].detach().cpu().numpy()
    body = params["body_pose"].detach().cpu().numpy()
    if root.ndim != 2 or root.shape[1] != 3 or body.shape != (len(root), 63):
        raise ValueError(f"Invalid GVHMR pose shapes: root={root.shape}, body={body.shape}")
    if not np.isfinite(root).all() or not np.isfinite(body).all():
        raise ValueError("GVHMR rotations contain non-finite values")

    axis_angle = np.concatenate((root[:, None], body.reshape(len(root), 21, 3)), axis=1)
    local_rotations = (
        Rotation.from_rotvec(axis_angle.reshape(-1, 3))
        .as_matrix()
        .reshape(len(root), 22, 3, 3)
        .astype(np.float32)
    )
    reference = convert_smplx_to_sonic(local_rotations, source_fps=source_fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        term1_local=reference.term1_local,
        root_quat=reference.root_quat,
        wrist=reference.wrist,
        fps=np.float32(reference.fps),
        source_path=np.asarray(str(source)),
        stream_mode=np.asarray("gvhmr"),
    )
    with np.load(destination, allow_pickle=False) as saved:
        shapes = {
            "term1_local": list(saved["term1_local"].shape),
            "root_quat": list(saved["root_quat"].shape),
            "wrist": list(saved["wrist"].shape),
        }
        finite = all(np.isfinite(saved[key]).all() for key in ("term1_local", "root_quat", "wrist"))
    expected = {
        "term1_local": [reference.frame_count, 72],
        "root_quat": [reference.frame_count, 4],
        "wrist": [reference.frame_count, 6],
    }
    if shapes != expected or not finite:
        raise RuntimeError(f"SONIC reference validation failed: shapes={shapes}, finite={finite}")
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_frames": len(root),
        "source_fps": float(source_fps),
        "frames": reference.frame_count,
        "fps": reference.fps,
        "duration_s": (reference.frame_count - 1) / reference.fps,
        "finite": finite,
        "shapes": shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Web task hmr4d_results.pt")
    parser.add_argument("destination", type=Path, help="Output sonic_reference.npz")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = convert(args.source, args.destination, args.source_fps)
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
