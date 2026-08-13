"""Memory-bounded long-video inference with one shared preprocessing pass.

Only the temporal GVHMR/FootMR network is windowed. Detection, pose and HMR2
features are computed for the normalized video once, and the loaded network is
reused for every window. Window predictions are cached for safe resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def window_starts(frame_count: int, window_frames: int, stride_frames: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("Long-video inference requires at least one frame")
    if window_frames < 2 or not 0 < stride_frames < window_frames:
        raise ValueError("Require window_frames >= 2 and 0 < stride_frames < window_frames")
    if frame_count <= window_frames:
        return [0]
    starts = list(range(0, frame_count - window_frames + 1, stride_frames))
    final_start = frame_count - window_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def result_length(result: dict[str, Any]) -> int:
    return int(result["smpl_params_global"]["global_orient"].shape[0])


def validate_result(result: dict[str, Any], expected: int, context: str = "result") -> None:
    for group in ("smpl_params_global", "smpl_params_incam"):
        params = result[group]
        expected_shapes = {
            "global_orient": (expected, 3),
            "body_pose": (expected, 63),
            "betas": (expected, 10),
            "transl": (expected, 3),
        }
        for key, shape in expected_shapes.items():
            value = params[key]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(
                    f"Invalid {context} {group}.{key}: {tuple(value.shape)}, expected {shape}"
                )
    intrinsics = result["K_fullimg"]
    if tuple(intrinsics.shape) != (expected, 3, 3) or not torch.isfinite(intrinsics).all():
        raise ValueError(f"Invalid {context} K_fullimg: {tuple(intrinsics.shape)}")


def slice_data(data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    total = int(data["length"])
    if not 0 <= start < end <= total:
        raise IndexError(f"Invalid data slice {start}:{end} for {total} frames")
    sliced: dict[str, Any] = {"length": torch.tensor(end - start)}
    for key, value in data.items():
        if key == "length":
            continue
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == total:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _y_rotation(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _circular_location(angles: np.ndarray) -> float:
    center = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    residual = np.angle(np.exp(1j * (angles - center)))
    return float(center + np.median(residual))


def fit_yaw_translation(
    reference_orient: np.ndarray,
    reference_transl: np.ndarray,
    moving_orient: np.ndarray,
    moving_transl: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    relative = Rotation.from_rotvec(reference_orient) * Rotation.from_rotvec(
        moving_orient
    ).inv()
    yaw = _circular_location(relative.as_euler("YXZ", degrees=False)[:, 0])
    rotate = _y_rotation(yaw)
    rotated = moving_transl @ rotate.T
    offset = np.median(reference_transl - rotated, axis=0)
    before = np.linalg.norm(reference_transl - moving_transl, axis=1)
    after = np.linalg.norm(reference_transl - (rotated + offset), axis=1)
    return yaw, offset, {
        "root_overlap_rmse_before_m": float(np.sqrt(np.mean(before**2))),
        "root_overlap_rmse_after_m": float(np.sqrt(np.mean(after**2))),
        "yaw_correction_deg": float(np.degrees(yaw)),
        "translation_correction_m": [float(value) for value in offset],
    }


def _transform_global(result: dict[str, Any], yaw: float, offset: np.ndarray) -> None:
    params = result["smpl_params_global"]
    rotate = _y_rotation(yaw)
    transl = params["transl"].numpy().astype(np.float64)
    params["transl"] = torch.from_numpy((transl @ rotate.T + offset).astype(np.float32))
    orient = params["global_orient"].numpy().astype(np.float64)
    transformed = Rotation.from_matrix(rotate) * Rotation.from_rotvec(orient)
    params["global_orient"] = torch.from_numpy(transformed.as_rotvec().astype(np.float32))
    net_global = result.get("net_outputs", {}).get("pred_smpl_params_global")
    if isinstance(net_global, dict):
        net_global["transl"] = params["transl"].unsqueeze(0)
        net_global["global_orient"] = params["global_orient"].unsqueeze(0)


def _window_weight(
    index: int, starts: list[int], lengths: list[int], total: int
) -> np.ndarray:
    start = starts[index]
    length = lengths[index]
    weight = np.ones(length, dtype=np.float64)
    if index > 0:
        overlap = starts[index - 1] + lengths[index - 1] - start
        if overlap > 0:
            phase = np.linspace(0.0, math.pi / 2.0, overlap, endpoint=False)
            weight[:overlap] *= np.sin(phase) ** 2
    if index + 1 < len(starts):
        overlap = start + length - starts[index + 1]
        if overlap > 0:
            phase = np.linspace(0.0, math.pi / 2.0, overlap, endpoint=False)
            weight[-overlap:] *= np.cos(phase) ** 2
    if start == 0:
        weight[0] = 1.0
    if start + length == total:
        weight[-1] = 1.0
    return weight


def _blend_rotvec(values: list[np.ndarray], normalized_weights: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return values[0].astype(np.float32, copy=True)
    if len(values) != 2:
        raise RuntimeError(f"Expected at most two overlapping windows, got {len(values)}")
    # Vectorized quaternion SLERP supports all 21 body joints at once. SciPy's
    # Slerp treats the first dimension as a time sequence and cannot directly
    # interpolate a batch of independent rotation pairs.
    first = Rotation.from_rotvec(values[0]).as_quat()
    second = Rotation.from_rotvec(values[1]).as_quat()
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    fraction = float(normalized_weights[1])
    angle = np.arccos(dot)
    sine = np.sin(angle)
    near = sine < 1.0e-8
    safe_sine = np.where(near, 1.0, sine)
    scale_first = np.where(
        near, 1.0 - fraction, np.sin((1.0 - fraction) * angle) / safe_sine
    )
    scale_second = np.where(
        near, fraction, np.sin(fraction * angle) / safe_sine
    )
    quaternion = scale_first * first + scale_second * second
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return Rotation.from_quat(quaternion).as_rotvec().astype(np.float32)


def _blend_parameter(
    chunks: list[dict[str, Any]],
    starts: list[int],
    weights: list[np.ndarray],
    group: str,
    key: str,
    total: int,
    rotations: bool,
) -> torch.Tensor:
    sample = chunks[0][group][key].numpy()
    output = np.empty((total, *sample.shape[1:]), dtype=np.float32)
    for frame in range(total):
        values: list[np.ndarray] = []
        frame_weights: list[float] = []
        for chunk, start, weight in zip(chunks, starts, weights):
            local = frame - start
            if 0 <= local < len(weight) and weight[local] > 1.0e-12:
                values.append(chunk[group][key][local].numpy())
                frame_weights.append(float(weight[local]))
        if not values:
            raise RuntimeError(f"No window contributes to frame {frame}")
        normalized = np.asarray(frame_weights, dtype=np.float64)
        normalized /= normalized.sum()
        if rotations:
            joints = values[0].size // 3
            reshaped = [value.reshape(joints, 3) for value in values]
            frame_value = _blend_rotvec(reshaped, normalized)
            output[frame] = frame_value.reshape(sample.shape[1:])
        else:
            output[frame] = np.tensordot(normalized, np.stack(values), axes=(0, 0))
    return torch.from_numpy(output)


def _blend_time_tensor(
    values: list[torch.Tensor],
    starts: list[int],
    weights: list[np.ndarray],
    total: int,
) -> torch.Tensor:
    window_lengths = [len(weight) for weight in weights]
    first = values[0]
    time_dim = 1 if first.ndim >= 2 and first.shape[0] == 1 else 0
    moved = [value.movedim(time_dim, 0).cpu() for value in values]
    output = torch.empty((total, *moved[0].shape[1:]), dtype=moved[0].dtype)
    if output.dtype.is_floating_point:
        output.zero_()
        weight_sum = torch.zeros(total, dtype=torch.float64)
        for value, start, weight in zip(moved, starts, weights):
            length = min(len(value), len(weight), total - start)
            current = torch.from_numpy(weight[:length]).to(output.dtype)
            view = (length,) + (1,) * (output.ndim - 1)
            output[start : start + length] += value[:length] * current.view(view)
            weight_sum[start : start + length] += torch.from_numpy(weight[:length])
        if torch.any(weight_sum <= 0):
            raise RuntimeError("A temporal output frame has zero blend weight")
        output /= weight_sum.to(output.dtype).view((total,) + (1,) * (output.ndim - 1))
    else:
        best = torch.full((total,), -1.0, dtype=torch.float64)
        for value, start, weight in zip(moved, starts, weights):
            length = min(len(value), len(weight), total - start)
            candidate = torch.from_numpy(weight[:length])
            mask = candidate > best[start : start + length]
            output[start : start + length][mask] = value[:length][mask]
            best[start : start + length][mask] = candidate[mask]
    return output.movedim(0, time_dim)


def _stitch_tree(
    values: list[Any], starts: list[int], weights: list[np.ndarray], total: int
) -> Any:
    first = values[0]
    if isinstance(first, dict) and all(isinstance(value, dict) for value in values):
        common = set(first)
        for value in values[1:]:
            common &= set(value)
        return {
            key: _stitch_tree([value[key] for value in values], starts, weights, total)
            for key in first
            if key in common
        }
    if torch.is_tensor(first) and all(torch.is_tensor(value) for value in values):
        time_shaped = (
            (first.ndim >= 1 and first.shape[0] == len(weights[0]))
            or (first.ndim >= 2 and first.shape[0] == 1 and first.shape[1] == len(weights[0]))
        )
        compatible = all(
            value.ndim == first.ndim
            and (
                value.shape[1:] == first.shape[1:]
                if first.shape[0] != 1
                else value.shape[0] == 1 and value.shape[2:] == first.shape[2:]
            )
            for value in values
        )
        if time_shaped and compatible:
            return _blend_time_tensor(values, starts, weights, total)
        return first.clone()
    return first


def stitch_predictions(
    chunks: list[dict[str, Any]], starts: list[int], total: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not chunks or len(chunks) != len(starts):
        raise ValueError("Chunk predictions and starts must be non-empty and aligned")
    lengths = [result_length(chunk) for chunk in chunks]
    if any(starts[index] + lengths[index] < starts[index + 1] for index in range(len(starts) - 1)):
        raise ValueError("Long-video windows contain an uncovered gap")
    if any(
        starts[index + 2] < starts[index] + lengths[index]
        for index in range(max(0, len(starts) - 2))
    ):
        raise ValueError("Triple-overlap windows are not supported")

    alignments: list[dict[str, Any]] = []
    for index in range(1, len(chunks)):
        previous_start = starts[index - 1]
        current_start = starts[index]
        overlap_start = current_start
        overlap_end = min(previous_start + lengths[index - 1], current_start + lengths[index])
        if overlap_end <= overlap_start:
            raise ValueError(f"Windows {index - 1} and {index} do not overlap")
        previous_slice = slice(overlap_start - previous_start, overlap_end - previous_start)
        current_slice = slice(0, overlap_end - current_start)
        previous = chunks[index - 1]["smpl_params_global"]
        current = chunks[index]["smpl_params_global"]
        yaw, offset, metrics = fit_yaw_translation(
            previous["global_orient"][previous_slice].numpy(),
            previous["transl"][previous_slice].numpy(),
            current["global_orient"][current_slice].numpy(),
            current["transl"][current_slice].numpy(),
        )
        _transform_global(chunks[index], yaw, offset)
        metrics.update(
            {"chunk": index, "overlap_start": overlap_start, "overlap_end": overlap_end}
        )
        alignments.append(metrics)

    weights = [
        _window_weight(index, starts, lengths, total) for index in range(len(chunks))
    ]
    stitched: dict[str, Any] = {}
    for group in ("smpl_params_global", "smpl_params_incam"):
        stitched[group] = {}
        for key in ("body_pose", "betas", "global_orient", "transl"):
            stitched[group][key] = _blend_parameter(
                chunks,
                starts,
                weights,
                group,
                key,
                total,
                rotations=key in ("body_pose", "global_orient"),
            )
    stitched["K_fullimg"] = _blend_time_tensor(
        [chunk["K_fullimg"] for chunk in chunks], starts, weights, total
    )
    stitched["net_outputs"] = _stitch_tree(
        [chunk["net_outputs"] for chunk in chunks], starts, weights, total
    )
    for group in ("pred_smpl_params_global", "pred_smpl_params_incam"):
        if group in stitched["net_outputs"]:
            source_group = (
                "smpl_params_global" if group.endswith("global") else "smpl_params_incam"
            )
            for key, value in stitched[source_group].items():
                stitched["net_outputs"][group][key] = value.unsqueeze(0)
    return stitched, alignments


def apply_global_postprocess(
    result: dict[str, Any], *, endecoder: Any, static_cam: bool, use_foot_refiner: bool
) -> None:
    from hmr4d.model.gvhmr.utils.postprocess import (
        pp_static_joint,
        pp_static_joint_cam,
        process_ik,
    )

    try:
        device = next(endecoder.parameters()).device
    except StopIteration:
        device = next(endecoder.buffers()).device

    def to_device(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: to_device(item) for key, item in value.items()}
        return value.to(device) if torch.is_tensor(value) else value

    outputs = result["net_outputs"]
    # The window cache deliberately lives on CPU. Only the three structures
    # needed by GVHMR's postprocessor are copied back to the model device;
    # copying pred_context as well wastes memory on long sequences.
    post_outputs = to_device(
        {
            "pred_smpl_params_global": outputs["pred_smpl_params_global"],
            "pred_smpl_params_incam": outputs["pred_smpl_params_incam"],
            "static_conf_logits": outputs["static_conf_logits"],
        }
    )
    with torch.no_grad():
        if static_cam:
            transl = pp_static_joint_cam(post_outputs, endecoder)
        else:
            transl = pp_static_joint(post_outputs, endecoder)
        body_pose = process_ik(post_outputs, endecoder)
    result["smpl_params_global"]["transl"] = transl[0].cpu()
    result["smpl_params_global"]["body_pose"] = body_pose[0].cpu()
    outputs["pred_smpl_params_global"]["transl"] = transl.cpu()
    outputs["pred_smpl_params_global"]["body_pose"] = body_pose.cpu()
    outputs["decode_dict"]["body_pose"] = body_pose.cpu()
    if not use_foot_refiner:
        result["smpl_params_incam"]["body_pose"] = body_pose[0].cpu()
        outputs["pred_smpl_params_incam"]["body_pose"] = body_pose.cpu()


def predict_long_video(
    *,
    model: Any,
    data: dict[str, Any],
    normalized_video: Path,
    work_root: Path,
    static_cam: bool,
    no_postproc: bool,
    detach_to_cpu: Callable[[Any], Any],
    window_frames: int = 600,
    stride_frames: int = 480,
    cache_identity: str = "default",
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total = int(data["length"])
    starts = window_starts(total, window_frames, stride_frames)
    if len(starts) == 1:
        prediction = detach_to_cpu(
            model.predict(data, static_cam=static_cam, no_postproc=no_postproc)
        )
        validate_result(prediction, total)
        return prediction, {"mode": "single", "frames": total, "windows": 1}

    digest = sha256_file(normalized_video)
    identity_digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:12]
    run_root = (
        Path(work_root)
        / f"{digest[:12]}_w{window_frames}_s{stride_frames}_{identity_digest}"
    )
    window_root = run_root / "windows"
    window_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    manifest = {
        "schema": "long-video-network-windows-v1",
        "source": str(Path(normalized_video).resolve()),
        "source_sha256": digest,
        "frames": total,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "starts": starts,
        "cache_identity": cache_identity,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"Long-video resume manifest mismatch: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    chunks: list[dict[str, Any]] = []
    reused = 0
    prediction_elapsed = 0.0
    for index, start in enumerate(starts):
        end = min(total, start + window_frames)
        cache_path = window_root / f"window_{index:03d}_{start}_{end}.pt"
        prediction = None
        if cache_path.is_file():
            try:
                candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
                validate_result(candidate, end - start, str(cache_path))
                prediction = candidate
                reused += 1
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                cache_path.unlink(missing_ok=True)
        log(
            f"[Long Video] Window {index + 1}/{len(starts)} frames {start}:{end}"
            + (" (resume cache)" if prediction is not None else "")
        )
        if prediction is None:
            started = time.perf_counter()
            prediction = detach_to_cpu(
                model.predict(
                    slice_data(data, start, end),
                    static_cam=static_cam,
                    no_postproc=True,
                )
            )
            prediction_elapsed += time.perf_counter() - started
            validate_result(prediction, end - start, f"window {index}")
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(prediction, temporary)
            temporary.replace(cache_path)
        chunks.append(prediction)

    stitch_started = time.perf_counter()
    stitched, alignments = stitch_predictions(chunks, starts, total)
    stitch_elapsed = time.perf_counter() - stitch_started
    postprocess_elapsed = 0.0
    if not no_postproc:
        postprocess_started = time.perf_counter()
        apply_global_postprocess(
            stitched,
            endecoder=model.pipeline.endecoder,
            static_cam=static_cam,
            use_foot_refiner=bool(model.pipeline.use_foot_refiner),
        )
        postprocess_elapsed = time.perf_counter() - postprocess_started
    validate_result(stitched, total, "stitched long-video result")
    metrics = {
        "mode": "shared-preprocess-windowed-network",
        "frames": total,
        "windows": len(starts),
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "overlap_frames": window_frames - stride_frames,
        "reused_windows": reused,
        "network_prediction_elapsed_s": prediction_elapsed,
        "stitch_elapsed_s": stitch_elapsed,
        "global_postprocess_elapsed_s": postprocess_elapsed,
        "alignments": alignments,
        "manifest": str(manifest_path.resolve()),
    }
    metrics_path = run_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics["metrics"] = str(metrics_path.resolve())
    return stitched, metrics
