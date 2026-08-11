#!/usr/bin/env python3
"""Evaluate whether saved Human3R people form a usable single-person GMR track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human3r-dir", type=Path, required=True)
    parser.add_argument("--bbox-file", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-threshold", type=float, default=50.0)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    return parser.parse_args()


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append((start, previous, previous - start + 1))
            start = value
        previous = value
    runs.append((start, previous, previous - start + 1))
    return sorted(runs, key=lambda item: item[2], reverse=True)


def main() -> None:
    args = parse_args()
    for path in (args.human3r_dir, args.bbox_file, args.video):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smpl_paths = sorted((args.human3r_dir / "smpl").glob("*.npz"))
    camera_paths = sorted((args.human3r_dir / "camera").glob("*.npz"))
    bbox = torch.load(args.bbox_file, map_location="cpu", weights_only=False)["bbx_xyxy"].numpy()
    if not len(smpl_paths) or not (len(smpl_paths) == len(camera_paths) == len(bbox)):
        raise RuntimeError(
            f"Input length mismatch: smpl={len(smpl_paths)}, camera={len(camera_paths)}, bbox={len(bbox)}"
        )

    distances = np.full(len(smpl_paths), np.inf, dtype=np.float64)
    candidate_counts = np.zeros(len(smpl_paths), dtype=np.int32)
    selected_indices = np.full(len(smpl_paths), -1, dtype=np.int32)
    projected_candidates: list[np.ndarray] = []
    expected_heads = np.empty((len(smpl_paths), 2), dtype=np.float64)
    for frame, (smpl_path, camera_path) in enumerate(zip(smpl_paths, camera_paths)):
        smpl = np.load(smpl_path)
        intrinsics = np.load(camera_path)["intrinsics"]
        translation = smpl["transl"]
        candidate_counts[frame] = len(translation)
        projected = (
            translation[:, :2] / translation[:, 2:3] * intrinsics[(0, 1), (0, 1)]
            + intrinsics[(0, 1), (2, 2)]
            if len(translation)
            else np.empty((0, 2))
        )
        projected_candidates.append(projected)
        x1, y1, x2, y2 = bbox[frame]
        # Human3R translations use a head-centered SMPL-X layer. Ten percent
        # below the detector box top is a stable head-center proxy on this clip.
        expected = np.array([(x1 + x2) * 0.5, y1 + 0.10 * (y2 - y1)]) * 0.4
        expected_heads[frame] = expected
        if len(projected):
            candidate_distance = np.linalg.norm(projected - expected, axis=1)
            selected = int(np.argmin(candidate_distance))
            selected_indices[frame] = selected
            distances[frame] = float(candidate_distance[selected])

    valid = distances <= args.distance_threshold
    invalid_runs = contiguous_runs(~valid)
    longest_gap = invalid_runs[0][2] if invalid_runs else 0
    coverage = float(valid.mean())
    decision = (
        "pass"
        if coverage >= args.min_coverage and longest_gap <= args.max_gap
        else "guardrail_failed"
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != len(smpl_paths):
        raise RuntimeError(f"Video/Human3R length mismatch: {frame_count} vs {len(smpl_paths)}")
    output_size = (640, 360)
    writer = cv2.VideoWriter(
        str(args.output_dir / "human3r_only_tracking_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open Human3R-only diagnostic video writer")
    for frame in range(frame_count):
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read video frame {frame}")
        image = cv2.resize(image, output_size)
        x1, y1, x2, y2 = bbox[frame] * 0.5
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 20), 2)
        expected = expected_heads[frame] * 1.25
        cv2.drawMarker(
            image,
            tuple(expected.astype(int)),
            (255, 180, 20),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        for index, point in enumerate(projected_candidates[frame] * 1.25):
            is_selected = index == selected_indices[frame]
            color = (40, 200, 40) if is_selected and valid[frame] else (30, 30, 230)
            cv2.circle(image, tuple(point.astype(int)), 8 if is_selected else 5, color, -1)
            cv2.putText(
                image,
                str(index),
                tuple((point + np.array([7, -7])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        distance_label = "inf" if not np.isfinite(distances[frame]) else f"{distances[frame]:.1f}px"
        state = "VALID" if valid[frame] else "INVALID"
        color = (40, 200, 40) if valid[frame] else (30, 30, 230)
        cv2.putText(
            image,
            f"Human3R-only target: {state}  distance={distance_label}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )
        writer.write(image)
    writer.release()
    capture.release()

    finite_distance = distances[np.isfinite(distances)]
    payload = {
        "method": "Human3R-only target coverage diagnostic",
        "decision": decision,
        "frames": len(smpl_paths),
        "fps": fps,
        "distance_threshold_human3r_px": args.distance_threshold,
        "minimum_coverage": args.min_coverage,
        "maximum_allowed_gap_frames": args.max_gap,
        "valid_frames": int(valid.sum()),
        "invalid_frames": int((~valid).sum()),
        "coverage": coverage,
        "zero_candidate_frames": int((candidate_counts == 0).sum()),
        "longest_invalid_gap_frames": longest_gap,
        "longest_invalid_gap_seconds": longest_gap / fps,
        "largest_invalid_runs": [list(run) for run in invalid_runs[:10]],
        "candidate_count_distribution": {
            str(count): int((candidate_counts == count).sum())
            for count in np.unique(candidate_counts)
        },
        "distance_median_px": float(np.median(finite_distance)),
        "distance_p90_px": float(np.percentile(finite_distance, 90)),
        "distance_p95_px": float(np.percentile(finite_distance, 95)),
        "gmr_exported": False,
        "gmr_export_reason": (
            "Human3R person coverage failed; long-gap pose/root interpolation is not allowed."
            if decision != "pass"
            else "Track passed coverage checks."
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    time = np.arange(len(distances)) / fps
    plot_distance = np.where(np.isfinite(distances), distances, args.distance_threshold * 6)
    figure, axis = plt.subplots(figsize=(15, 4.5))
    axis.plot(time, plot_distance, color="tab:red", linewidth=0.9)
    axis.axhline(args.distance_threshold, color="black", linestyle="--", label="valid threshold")
    axis.fill_between(time, 0, plot_distance, where=~valid, color="tab:red", alpha=0.18)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("nearest Human3R head distance (px at 512x288)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "tracking_distance.png", dpi=160)
    plt.close(figure)

    report = f"""# Human3R-only 人体轨迹诊断

- 判定：`{decision}`。
- 有效覆盖：{payload['valid_frames']}/{payload['frames']}（{coverage:.2%}），要求至少 {args.min_coverage:.0%}。
- 完全没有人体候选：{payload['zero_candidate_frames']} 帧。
- 最长连续无效段：{longest_gap} 帧（{longest_gap / fps:.2f} 秒），允许最多 {args.max_gap} 帧。
- 最近候选距离中位数/P95：{payload['distance_median_px']:.2f}/{payload['distance_p95_px']:.2f} px。

该诊断已经借用 GVHMR bbox 作为目标位置参考；即使如此仍未通过覆盖保护。直接导出会要求对最长约 {longest_gap / fps:.2f} 秒的人体姿态和 root 做插值，不能视为 Human3R 预测，因此未生成 GMR 输入文件。
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
