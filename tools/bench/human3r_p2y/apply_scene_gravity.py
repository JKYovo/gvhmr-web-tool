#!/usr/bin/env python3
"""Align GVHMR global pitch/roll to a static camera-frame gravity vector.

Camera-space SMPL and all local body pose parameters are preserved. Only the
global root orientation is corrected; contact/root optimization runs after it.
The gravity vector may come from Human3R or reliable standing frames.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from scipy.ndimage import gaussian_filter1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--ground-plane", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--smoothing-seconds", type=float, default=0.15)
    return parser.parse_args()


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def rotations_between(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = normalize(source)
    target = normalize(target).expand_as(source)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
    skew = torch.zeros((*source.shape[:-1], 3, 3), dtype=source.dtype)
    skew[..., 0, 1] = -cross[..., 2]
    skew[..., 0, 2] = cross[..., 1]
    skew[..., 1, 0] = cross[..., 2]
    skew[..., 1, 2] = -cross[..., 0]
    skew[..., 2, 0] = -cross[..., 1]
    skew[..., 2, 1] = cross[..., 0]
    sin_sq = (cross * cross).sum(dim=-1)
    factor = ((1.0 - dot) / sin_sq.clamp_min(1e-12))[..., None, None]
    eye = torch.eye(3, dtype=source.dtype).expand_as(skew)
    rotation = eye + skew + skew @ skew * factor
    near_identity = sin_sq < 1e-12
    rotation[near_identity & (dot > 0.0)] = eye[near_identity & (dot > 0.0)]
    if bool((near_identity & (dot < 0.0)).any()):
        raise ValueError("Ground normal is antiparallel to the selected target axis")
    return rotation


def angle_to_y(vectors: torch.Tensor) -> torch.Tensor:
    vectors = normalize(vectors)
    return torch.rad2deg(torch.acos(vectors[..., 1].abs().clamp(0.0, 1.0)))


def summary(values: torch.Tensor) -> dict[str, float]:
    values = values.double()
    return {
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.smoothing_seconds < 0:
        raise ValueError("fps must be positive and smoothing-seconds non-negative")
    for path in (args.gvhmr_result, args.ground_plane):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = torch.load(args.gvhmr_result, map_location="cpu", weights_only=False)
    result = copy.deepcopy(source)
    incam_aa = source["smpl_params_incam"]["global_orient"].float()
    global_aa = source["smpl_params_global"]["global_orient"].float()
    if incam_aa.shape != global_aa.shape or incam_aa.ndim != 2:
        raise ValueError(f"Unexpected root orientation shapes: {incam_aa.shape}, {global_aa.shape}")

    plane_payload = json.loads(args.ground_plane.read_text(encoding="utf-8"))
    plane = plane_payload.get("selected_ground", plane_payload)
    normal_camera = normalize(torch.tensor(plane["normal"], dtype=torch.float32))
    rotation_incam = axis_angle_to_matrix(incam_aa)
    rotation_global = axis_angle_to_matrix(global_aa)
    camera_to_global = rotation_global @ rotation_incam.mT
    normal_global = camera_to_global @ normal_camera
    target_sign = -1.0 if float(normal_global[:, 1].median()) < 0.0 else 1.0
    target = torch.tensor([0.0, target_sign, 0.0], dtype=torch.float32)
    sigma_frames = args.smoothing_seconds * args.fps
    smoothed_np = normal_global.numpy()
    if sigma_frames > 0:
        smoothed_np = gaussian_filter1d(smoothed_np, sigma=sigma_frames, axis=0, mode="nearest")
    smoothed_normal = normalize(torch.from_numpy(smoothed_np).float())
    correction = rotations_between(smoothed_normal, target)
    corrected_rotation = correction @ rotation_global
    result["smpl_params_global"]["global_orient"] = matrix_to_axis_angle(
        corrected_rotation
    ).to(source["smpl_params_global"]["global_orient"].dtype)

    corrected_normal = (corrected_rotation @ rotation_incam.mT) @ normal_camera
    correction_angle = torch.rad2deg(
        torch.acos(
            ((torch.diagonal(correction, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
        )
    )
    output_path = args.output_dir / "gravity_aligned_hmr4d_results.pt"
    torch.save(result, output_path)
    metrics = {
        "method": "static-camera gravity alignment",
        "gravity_source_method": plane_payload.get("method", "unknown"),
        "frames": int(len(global_aa)),
        "fps": float(args.fps),
        "smoothing_seconds": float(args.smoothing_seconds),
        "ground_normal_camera": normal_camera.tolist(),
        "target_global_axis": target.tolist(),
        "plane_residual_median_m": float(plane["residual_median_m"]) if "residual_median_m" in plane else None,
        "plane_residual_p95_m": float(plane["residual_p95_m"]) if "residual_p95_m" in plane else None,
        "gravity_residual_before_deg": summary(angle_to_y(normal_global)),
        "gravity_residual_after_deg": summary(angle_to_y(corrected_normal)),
        "orientation_correction_deg": summary(correction_angle),
        "preservation": {
            "smpl_params_incam_unchanged": True,
            "body_pose_unchanged": True,
            "betas_unchanged": True,
            "global_transl_unchanged": True,
            "net_outputs_unchanged": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
