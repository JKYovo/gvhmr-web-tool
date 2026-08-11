#!/usr/bin/env python3
"""Extract parallel floor and box-top planes from a Human3R reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human3r-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=2.0)
    parser.add_argument("--human-mask-threshold", type=float, default=0.1)
    parser.add_argument("--distance-threshold", type=float, default=0.025)
    parser.add_argument("--max-planes", type=int, default=16)
    parser.add_argument("--min-plane-points", type=int, default=120)
    return parser.parse_args()


def normalized_plane(model: np.ndarray) -> tuple[np.ndarray, float]:
    normal = np.asarray(model[:3], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Degenerate plane: {model}")
    return normal / norm, float(model[3] / norm)


def connected_components(
    pixels: np.ndarray, width: int, height: int, pixel_stride: int
) -> list[dict[str, object]]:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 1
    kernel_size = pixel_stride * 3 + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    connected_mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8))
    _, labels = cv2.connectedComponents(connected_mask)
    pixel_labels = labels[pixels[:, 1], pixels[:, 0]]
    components: list[dict[str, object]] = []
    for label in np.unique(pixel_labels):
        if label == 0:
            continue
        indices = np.flatnonzero(pixel_labels == label)
        if len(indices) < 20:
            continue
        component_pixels = pixels[indices]
        median_pixel = np.median(component_pixels, axis=0)
        center_score = np.exp(-((median_pixel[0] - width / 2) / (0.35 * width)) ** 2)
        vertical_score = np.exp(-((median_pixel[1] - 0.62 * height) / (0.25 * height)) ** 2)
        components.append(
            {
                "indices": indices,
                "points": int(len(indices)),
                "median_pixel": median_pixel.tolist(),
                "pixel_bounds": [
                    component_pixels.min(axis=0).tolist(),
                    component_pixels.max(axis=0).tolist(),
                ],
                "score": float(len(indices) * center_score * vertical_score),
            }
        )
    return sorted(components, key=lambda item: float(item["score"]), reverse=True)


def main() -> None:
    args = parse_args()
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
        intrinsics.append(np.load(root / "camera" / f"{frame_id:06d}.npz")["intrinsics"])

    depth_stack = np.stack(depths)
    valid_count = np.isfinite(depth_stack).sum(axis=0)
    with np.errstate(all="ignore"):
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
        planes.append(
            {
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
                "pixels": inlier_pixels,
                "inlier_points": inlier_points,
            }
        )
        keep = np.ones(len(remaining_points), dtype=bool)
        keep[inlier_indices] = False
        remaining_points = remaining_points[keep]
        remaining_pixels = remaining_pixels[keep]

    candidates: list[dict[str, float | int]] = []
    for i, first in enumerate(planes):
        n_first = np.asarray(first["normal"])
        for j in range(i + 1, len(planes)):
            second = planes[j]
            n_second = np.asarray(second["normal"])
            dot = float(np.dot(n_first, n_second))
            parallel = abs(dot)
            if parallel < np.cos(np.deg2rad(10.0)):
                continue
            first_v = float(first["median_pixel"][1])
            second_v = float(second["median_pixel"][1])
            floor_id, top_id = (i, j) if first_v >= second_v else (j, i)
            floor_plane = planes[floor_id]
            top_plane = planes[top_id]
            floor_normal = np.asarray(floor_plane["normal"])
            floor_offset = float(floor_plane["offset"])
            top_points = np.asarray(top_plane["inlier_points"])
            separation = abs(float(np.median(top_points @ floor_normal + floor_offset)))
            if not 0.20 <= separation <= 1.00:
                continue
            pixel_gap = abs(first_v - second_v)
            support = np.sqrt(float(first["points"]) * float(second["points"]))
            score = support * max(pixel_gap, 5.0) * parallel
            candidates.append(
                {
                    "floor_id": floor_id,
                    "top_id": top_id,
                    "separation_m": separation,
                    "parallel_cosine": parallel,
                    "pixel_gap": pixel_gap,
                    "score": score,
                }
            )
    if not candidates:
        raise RuntimeError("No parallel floor/box-top plane pair found")
    selected = max(candidates, key=lambda item: float(item["score"]))
    floor = planes[int(selected["floor_id"])]
    top = planes[int(selected["top_id"])]
    top_components = connected_components(
        np.asarray(top["pixels"]), width, height, args.pixel_stride
    )
    if not top_components:
        raise RuntimeError("Selected top plane has no connected component")
    top_component = top_components[0]
    floor_normal = np.asarray(floor["normal"])
    floor_offset = float(floor["offset"])
    initial_center = np.asarray(top_component["median_pixel"])
    heights = np.abs(points @ floor_normal + floor_offset)
    search_mask = (
        (np.abs(pixels[:, 0] - initial_center[0]) <= 110)
        & (np.abs(pixels[:, 1] - initial_center[1]) <= 60)
        & (heights >= 0.20)
        & (heights <= 1.00)
    )
    search_indices = np.flatnonzero(search_mask)
    histogram, edges = np.histogram(heights[search_indices], bins=np.arange(0.20, 1.005, 0.005))
    peak_order = np.argsort(histogram)[::-1]
    refined = None
    for peak_index in peak_order:
        peak_height = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
        peak_indices = search_indices[
            np.abs(heights[search_indices] - peak_height) <= args.distance_threshold
        ]
        if len(peak_indices) < 40:
            continue
        components = connected_components(
            pixels[peak_indices], width, height, args.pixel_stride
        )
        for component in components:
            component_indices = peak_indices[np.asarray(component["indices"])]
            component_points = points[component_indices]
            centered = component_points - component_points.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            component_normal = vh[-1]
            parallel = abs(float(np.dot(component_normal, floor_normal)))
            if parallel < np.cos(np.deg2rad(10.0)):
                continue
            refined = {
                "indices": component_indices,
                "component": component,
                "parallel_cosine": parallel,
                "histogram_peak_m": peak_height,
            }
            break
        if refined is not None:
            break
    if refined is None:
        raise RuntimeError("Unable to refine a connected horizontal box-top surface")
    refined_indices = np.asarray(refined["indices"])
    refined_separation = float(np.median(heights[refined_indices]))
    selected["separation_m"] = refined_separation
    selected["refinement"] = {
        "method": "central_height_histogram_connected_plane",
        "histogram_peak_m": refined["histogram_peak_m"],
        "parallel_cosine": refined["parallel_cosine"],
    }
    selected["top_component"] = {
        key: value
        for key, value in refined["component"].items()
        if key != "indices"
    }

    overlay = cv2.imread(str(root / "color" / f"{frame_ids[len(frame_ids) // 2]:06d}.png"))
    if overlay is None:
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
    colors = ((50, 220, 50), (30, 80, 240))
    overlay_planes = (
        np.asarray(floor["pixels"]),
        pixels[refined_indices],
    )
    for plane_pixels, color in zip(overlay_planes, colors):
        for u, v in plane_pixels:
            cv2.circle(overlay, (int(u), int(v)), args.pixel_stride, color, -1)
    cv2.putText(overlay, "floor", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[0], 2)
    cv2.putText(
        overlay,
        f"box top: {float(selected['separation_m']):.3f} m",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colors[1],
        2,
    )
    cv2.imwrite(str(output_dir / "plane_overlay.png"), overlay)
    np.save(output_dir / "median_depth.npy", median_depth)

    serializable_planes = [
        {
            key: value
            for key, value in plane.items()
            if key not in {"pixels", "inlier_points"}
        }
        for plane in planes
    ]
    payload = {
        "human3r_dir": str(root),
        "frames_used": frame_ids,
        "frame_stride": args.frame_stride,
        "pixel_stride": args.pixel_stride,
        "confidence_threshold": args.confidence_threshold,
        "human_mask_threshold": args.human_mask_threshold,
        "required_observations": required_observations,
        "static_points": int(len(points)),
        "planes": serializable_planes,
        "pair_candidates": sorted(candidates, key=lambda item: float(item["score"]), reverse=True),
        "selected_pair": selected,
    }
    (output_dir / "scene_planes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_pair": selected, "planes": serializable_planes}, indent=2))


if __name__ == "__main__":
    main()
