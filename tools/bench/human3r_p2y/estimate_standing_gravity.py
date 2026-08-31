#!/usr/bin/env python3
"""Estimate fixed-camera gravity from reliable upright GVHMR frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch3d.transforms import axis_angle_to_matrix

from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-torso-verticality", type=float, default=0.94)
    parser.add_argument("--minimum-bilateral-contact", type=float, default=0.75)
    parser.add_argument("--minimum-frames", type=int, default=15)
    return parser.parse_args()


def contact_confidence(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    while logits.ndim > 2:
        logits = logits[0]
    probabilities = logits.sigmoid()
    if probabilities.shape[-1] < 4:
        raise ValueError(f"Expected at least four contact channels, got {probabilities.shape}")
    values = []
    for start in (0, 2):
        foot = probabilities[:, start : start + 2]
        values.append(0.7 * foot.max(dim=1).values + 0.3 * foot.min(dim=1).values)
    return values[0], values[1]


def longest_segment(mask: torch.Tensor) -> tuple[int, int]:
    best_start = best_stop = start = 0
    active = False
    for index, value in enumerate(mask.tolist() + [False]):
        if value and not active:
            start = index
            active = True
        elif not value and active:
            if index - start > best_stop - best_start:
                best_start, best_stop = start, index
            active = False
    return best_start, best_stop


def main() -> None:
    args = parse_args()
    if not args.gvhmr_result.is_file():
        raise FileNotFoundError(args.gvhmr_result)
    source = torch.load(args.gvhmr_result, map_location="cpu", weights_only=False)
    model = SmplxLiteV437Coco23().eval()
    with torch.inference_mode():
        _, joints = model(**source["smpl_params_global"])
    torso = joints[:, [5, 6]].mean(dim=1) - joints[:, [11, 12]].mean(dim=1)
    verticality = torso[:, 1].abs() / torso.norm(dim=-1).clamp_min(1e-8)
    left, right = contact_confidence(source["net_outputs"]["static_conf_logits"])
    bilateral = torch.minimum(left, right)
    candidates = (verticality >= args.minimum_torso_verticality) & (
        bilateral >= args.minimum_bilateral_contact
    )
    start, stop = longest_segment(candidates)
    if stop - start < args.minimum_frames:
        raise RuntimeError(
            f"No reliable standing calibration segment: longest={stop-start}, required={args.minimum_frames}"
        )
    selected = torch.zeros_like(candidates)
    selected[start:stop] = candidates[start:stop]

    incam = axis_angle_to_matrix(source["smpl_params_incam"]["global_orient"].float())
    global_rotation = axis_angle_to_matrix(
        source["smpl_params_global"]["global_orient"].float()
    )
    camera_to_global = global_rotation @ incam.mT
    global_down = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
    samples = camera_to_global[selected].mT @ global_down
    normal = samples.median(dim=0).values
    normal = normal / normal.norm().clamp_min(1e-8)
    samples = samples / samples.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    deviations = torch.rad2deg(torch.acos((samples @ normal).clamp(-1.0, 1.0)))
    payload = {
        "method": "GVHMR reliable-standing fixed-camera gravity calibration",
        "normal": normal.tolist(),
        "selected_frames": torch.where(selected)[0].tolist(),
        "selected_segment": [int(start), int(stop)],
        "selected_count": int(selected.sum()),
        "thresholds": {
            "minimum_torso_verticality": float(args.minimum_torso_verticality),
            "minimum_bilateral_contact": float(args.minimum_bilateral_contact),
            "minimum_frames": int(args.minimum_frames),
        },
        "selected_statistics": {
            "torso_verticality_median": float(verticality[selected].median()),
            "bilateral_contact_median": float(bilateral[selected].median()),
            "gravity_sample_deviation_median_deg": float(deviations.median()),
            "gravity_sample_deviation_p95_deg": float(torch.quantile(deviations, 0.95)),
            "gravity_sample_deviation_max_deg": float(deviations.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
