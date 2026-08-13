import imageio.v3 as iio
import numpy as np
import torch
from pathlib import Path
import shutil
import ffmpeg
from tqdm import tqdm
import cv2
from fractions import Fraction


def _probe_video_stream(video_path):
    probe = ffmpeg.probe(str(video_path))
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError(f"No video stream found in {video_path}")


def _get_video_rotation(video_path):
    stream = _probe_video_stream(video_path)
    for side_data in stream.get("side_data_list") or []:
        if side_data.get("rotation") is not None:
            return int(side_data["rotation"]) % 360
    rotate_tag = (stream.get("tags") or {}).get("rotate")
    return int(float(rotate_tag)) % 360 if rotate_tag is not None else 0


def get_video_lwh(video_path, display_oriented=False):
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    if display_oriented and _get_video_rotation(video_path) in {90, 270}:
        W, H = H, W
    return L, W, H


def get_video_fps_duration(video_path):
    """Return the average video FPS and timeline duration from ffprobe metadata."""
    probe = ffmpeg.probe(str(video_path), select_streams="v:0")
    stream = probe["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    duration = stream.get("duration") or probe.get("format", {}).get("duration")
    duration = float(duration) if duration is not None else None
    return fps, duration


def normalize_video_fps(input_path, output_path, target_fps=30, crf=23):
    """Create a constant-FPS video while preserving its timeline duration.

    Unlike relabeling every decoded frame with a new FPS, the ffmpeg ``fps``
    filter drops or duplicates frames according to timestamps. This keeps a
    60 FPS source at its original speed when normalized to 30 FPS.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("normalize_video_fps requires different input and output paths")

    source_fps, source_duration = get_video_fps_duration(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        output_fps, output_duration = get_video_fps_duration(output_path)
        duration_matches = (
            source_duration is None
            or output_duration is None
            or abs(source_duration - output_duration) <= max(0.1, 1.5 / target_fps)
        )
        if abs(output_fps - target_fps) < 1e-3 and duration_matches:
            return {
                "source_fps": source_fps,
                "source_duration": source_duration,
                "output_fps": output_fps,
                "output_duration": output_duration,
                "reused": True,
            }

    temporary = output_path.with_suffix(output_path.suffix + ".normalize-tmp.mp4")
    temporary.unlink(missing_ok=True)
    video = ffmpeg.input(str(input_path)).video.filter("fps", fps=target_fps, round="near")
    output = ffmpeg.output(
        video,
        str(temporary),
        vcodec="libx264",
        pix_fmt="yuv420p",
        r=target_fps,
        crf=crf,
        an=None,
    )
    ffmpeg.run(output, overwrite_output=True, quiet=True)
    temporary.replace(output_path)
    output_fps, output_duration = get_video_fps_duration(output_path)
    if abs(output_fps - target_fps) >= 1e-3:
        raise RuntimeError(f"Expected {target_fps} FPS, got {output_fps} for {output_path}")
    if source_duration is not None and output_duration is not None:
        if abs(source_duration - output_duration) > max(0.1, 1.5 / target_fps):
            raise RuntimeError(
                f"Video normalization changed duration: {source_duration:.3f}s -> {output_duration:.3f}s"
            )
    return {
        "source_fps": source_fps,
        "source_duration": source_duration,
        "output_fps": output_fps,
        "output_duration": output_duration,
        "reused": False,
    }


def transcode_video_normalized(video_path, out_video_path, fps=30, crf=17):
    """Compatibility entry point used by the embedded Web runner."""
    return normalize_video_fps(video_path, out_video_path, target_fps=fps, crf=crf)


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # If video path not exists, an error will be raised by ffmpegs
    filter_args = []
    should_check_length = False

    # 1. Trim
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(("trim", f"start_frame={start_frame}:end_frame={end_frame}"))

    # 2. Scale
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # Excute then check
    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    Args:
        images: (N, H, W, 3) RGB, uint8
        crf: 17 is visually lossless, 23 is default, +6 results in half the bitrate
    0 is lossless, https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    """remember to .close()"""
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(
        merged_video,
        out_video_path,
        vcodec="libx264",
        preset="veryfast",
        crf=23,
        pix_fmt="yuv420p",
        an=None,
    )
    ffmpeg.run(output, overwrite_output=True, quiet=True)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(
        merged_video,
        out_video_path,
        vcodec="libx264",
        preset="veryfast",
        crf=23,
        pix_fmt="yuv420p",
        an=None,
    )
    ffmpeg.run(output, overwrite_output=True, quiet=True)
