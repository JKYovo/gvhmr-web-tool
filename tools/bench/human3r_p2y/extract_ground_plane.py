#!/usr/bin/env python3
"""Extract one static ground plane from a Human3R reconstruction."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human3r-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=1.05,
        help="Human3R confidence is exp-parameterized with a minimum near 1.",
    )
    parser.add_argument("--human-mask-threshold", type=float, default=0.1)
    parser.add_argument("--distance-threshold", type=float, default=0.025)
    parser.add_argument("--max-planes", type=int, default=16)
    parser.add_argument("--min-plane-points", type=int, default=120)
    return parser.parse_args()


def normalized_plane(model: np.ndarray) -> tuple[np.ndarray, float]:
    normal = np.asarray(model[:3], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Degenerate plane: {model}")
    normal = normal / norm
    offset = float(model[3] / norm)
    # Keep the closest point on a visible plane in front of the camera.  Plane
    # geometry is sign invariant, but a stable convention simplifies reports.
    if offset > 0:
        normal = -normal
        offset = -offset
    return normal, offset


def plane_score(plane: dict[str, object], width: int, height: int) -> dict[str, float]:
    pixels = np.asarray(plane["pixels"])
    bounds = np.asarray(plane["pixel_bounds"])
    median_v = float(np.median(pixels[:, 1]) / height)
    lower_fraction = float(np.mean(pixels[:, 1] >= 0.55 * height))
    width_coverage = float((bounds[1, 0] - bounds[0, 0] + 1) / width)
    bottom_reach = float((bounds[1, 1] + 1) / height)
    # A floor should occupy the lower image, span a useful horizontal region,
    # and reach toward the bottom.  This deliberately avoids a camera-gravity
    # assumption, which Human3R does not provide.
    location = np.exp(-((median_v - 0.72) / 0.24) ** 2)
    score = (
        float(plane["points"])
        * max(lower_fraction, 0.02)
        * max(width_coverage, 0.05)
        * max(bottom_reach, 0.1)
        * float(location)
    )
    return {
        "score": float(score),
        "median_v_fraction": median_v,
        "lower_image_fraction": lower_fraction,
        "width_coverage": width_coverage,
        "bottom_reach": bottom_reach,
    }


def main() -> None:
    args = parse_args()
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Ground extraction requires Open3D in the Human3R environment"
        ) from error
    if args.frame_stride < 1 or args.pixel_stride < 1:
        raise ValueError("Frame and pixel strides must be positive")
    o3d.utility.random.seed(42)
    root = args.human3r_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_paths = sorted((root / "depth").glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No Human3R depths found in {root}")
    frame_ids = list(range(0, len(depth_paths), args.frame_stride))
    depths: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for frame_id in frame_ids:
        depth = np.load(root / "depth" / f"{frame_id:06d}.npy").astype(np.float32)
        confidence = np.load(root / "conf" / f"{frame_id:06d}.npy")
        smpl = np.load(root / "smpl" / f"{frame_id:06d}.npz", allow_pickle=True)
        human_mask = smpl["msk"]
        if human_mask.ndim == 3:
            human_mask = human_mask[0]
        valid = (
            np.isfinite(depth)
            & (depth > 0.1)
            & (depth < 20.0)
            & (confidence >= args.confidence_threshold)
            & (human_mask < args.human_mask_threshold)
        )
        depths.append(np.where(valid, depth, np.nan))
        intrinsics.append(
            np.load(root / "camera" / f"{frame_id:06d}.npz")["intrinsics"]
        )

    depth_stack = np.stack(depths)
    valid_count = np.isfinite(depth_stack).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_depth = np.nanmedian(depth_stack, axis=0)
    del depth_stack
    required_observations = max(3, int(np.ceil(len(frame_ids) * 0.15)))
    valid_median = np.isfinite(median_depth) & (valid_count >= required_observations)

    height, width = median_depth.shape
    vv, uu = np.mgrid[0:height:args.pixel_stride, 0:width:args.pixel_stride]
    sampled_depth = median_depth[:: args.pixel_stride, :: args.pixel_stride]
    sampled_valid = valid_median[:: args.pixel_stride, :: args.pixel_stride]
    intrinsic = np.median(np.stack(intrinsics), axis=0)
    x = (uu - intrinsic[0, 2]) * sampled_depth / intrinsic[0, 0]
    y = (vv - intrinsic[1, 2]) * sampled_depth / intrinsic[1, 1]
    points_grid = np.stack((x, y, sampled_depth), axis=-1)
    points = points_grid[sampled_valid].astype(np.float64)
    pixels = np.stack((uu, vv), axis=-1)[sampled_valid].astype(np.int32)
    if len(points) < args.min_plane_points:
        raise RuntimeError(f"Too few static scene points: {len(points)}")

    remaining_points = points.copy()
    remaining_pixels = pixels.copy()
    planes: list[dict[str, object]] = []
    for plane_id in range(args.max_planes):
        if len(remaining_points) < args.min_plane_points:
            break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining_points))
        model, inlier_indices = cloud.segment_plane(
            distance_threshold=args.distance_threshold,
            ransac_n=3,
            num_iterations=2000,
        )
        inlier_indices = np.asarray(inlier_indices, dtype=np.int64)
        if len(inlier_indices) < args.min_plane_points:
            break
        normal, offset = normalized_plane(np.asarray(model))
        inlier_points = remaining_points[inlier_indices]
        inlier_pixels = remaining_pixels[inlier_indices]
        residuals = np.abs(inlier_points @ normal + offset)
        plane: dict[str, object] = {
            "id": plane_id,
            "normal": normal.tolist(),
            "offset": offset,
            "points": int(len(inlier_indices)),
            "centroid": np.median(inlier_points, axis=0).tolist(),
            "median_pixel": np.median(inlier_pixels, axis=0).tolist(),
            "pixel_bounds": [
                inlier_pixels.min(axis=0).tolist(),
                inlier_pixels.max(axis=0).tolist(),
            ],
            "residual_median_m": float(np.median(residuals)),
            "residual_p95_m": float(np.percentile(residuals, 95)),
            "pixels": inlier_pixels,
        }
        plane["ground_score"] = plane_score(plane, width, height)
        planes.append(plane)
        keep = np.ones(len(remaining_points), dtype=bool)
        keep[inlier_indices] = False
        remaining_points = remaining_points[keep]
        remaining_pixels = remaining_pixels[keep]

    if not planes:
        raise RuntimeError("No static plane was reconstructed")
    selected = max(planes, key=lambda item: float(item["ground_score"]["score"]))
    quality = selected["ground_score"]
    if (
        float(quality["lower_image_fraction"]) < 0.15
        or float(quality["width_coverage"]) < 0.20
        or float(quality["bottom_reach"]) < 0.70
    ):
        raise RuntimeError(
            "Best plane does not have convincing ground coverage: "
            f"{json.dumps(quality, ensure_ascii=False)}"
        )

    middle_id = frame_ids[len(frame_ids) // 2]
    overlay = cv2.imread(str(root / "color" / f"{middle_id:06d}.png"))
    if overlay is None:
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for u, v in np.asarray(selected["pixels"]):
        cv2.circle(overlay, (int(u), int(v)), args.pixel_stride, (50, 220, 50), -1)
    label = (
        f"ground: plane {selected['id']}, "
        f"residual {100 * float(selected['residual_median_m']):.2f} cm"
    )
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 80, 20), 2)
    cv2.imwrite(str(output_dir / "ground_overlay.png"), overlay)
    np.save(output_dir / "median_depth.npy", median_depth)

    serializable_planes = [
        {key: value for key, value in plane.items() if key != "pixels"}
        for plane in planes
    ]
    selected_id = int(selected["id"])
    payload = {
        "method": "multi_frame_median_depth_ground_only_ransac",
        "human3r_dir": str(root),
        "frames_used": frame_ids,
        "frame_stride": args.frame_stride,
        "pixel_stride": args.pixel_stride,
        "confidence_threshold": args.confidence_threshold,
        "human_mask_threshold": args.human_mask_threshold,
        "required_observations": required_observations,
        "static_points": int(len(points)),
        "image_size": [width, height],
        "median_intrinsics": intrinsic.tolist(),
        "planes": serializable_planes,
        "selected_ground_id": selected_id,
        "selected_ground": serializable_planes[selected_id],
    }
    (output_dir / "ground_plane.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_ground_id": selected_id,
                "selected_ground": payload["selected_ground"],
                "plane_count": len(planes),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
