import torch
import imageio.v3 as iio
import json
import os
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np
from pathlib import Path

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm
from hmr4d.utils.video_io_utils import get_video_lwh


MEMMAP_METADATA_SUFFIX = ".json"


def _prepare_crop(img, center, size, img_ds, img_dst_size):
    size_ds = size * img_ds
    factor = float(size_ds / img_dst_size / 2.0)
    if factor > 1.1:
        img = cv2.GaussianBlur(img, (5, 5), (factor - 1) / 2)
    crop, bbx_xys_ds = crop_and_resize(
        img,
        center * img_ds,
        size_ds,
        img_dst_size,
        enlarge_ratio=1.0,
    )
    normalized = ((torch.from_numpy(crop) / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(
        2, 0, 1
    )
    return normalized, torch.from_numpy(bbx_xys_ds).float() / img_ds


def get_batch_memmap(input_path, bbx_xys, output_path, img_ds=0.5, img_dst_size=256):
    """Stream normalized crops into a disk mapping instead of retaining video frames.

    This is used when ViTPose and HMR2 share their crop input on long videos.
    The resulting tensor has the same shape/dtype as :func:`get_batch`, while
    resident memory stays bounded by one decoded frame and one crop.
    """
    input_path = str(input_path)
    output_path = Path(output_path)
    frame_count = get_video_lwh(input_path)[0]
    if len(bbx_xys) != frame_count:
        raise ValueError(
            f"Crop bbox length mismatch: {len(bbx_xys)} boxes for {frame_count} frames"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    mapped = np.memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 3, img_dst_size, img_dst_size),
    )
    scaled_frames = iio.imiter(
        input_path,
        plugin="pyav",
        filter_sequence=[("scale", f"iw*{img_ds}:ih*{img_ds}")],
    )
    resized_boxes = torch.empty((frame_count, 3), dtype=torch.float32)
    decoded = 0
    try:
        for index, img in enumerate(
            tqdm(scaled_frames, total=frame_count, desc="Shared Crop")
        ):
            if index >= frame_count:
                raise RuntimeError(
                    f"Video decoder produced more than {frame_count} frames for {input_path}"
                )
            normalized, resized_box = _prepare_crop(
                img,
                bbx_xys[index, :2].numpy(),
                float(bbx_xys[index, 2]),
                img_ds,
                img_dst_size,
            )
            mapped[index] = normalized.numpy()
            resized_boxes[index] = resized_box
            decoded += 1
    except Exception:
        del mapped
        output_path.unlink(missing_ok=True)
        raise
    if decoded != frame_count:
        del mapped
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Video decoder produced {decoded} frames, expected {frame_count}: {input_path}"
        )
    mapped.flush()
    return torch.from_numpy(mapped), resized_boxes


def open_batch_memmap(output_path):
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + MEMMAP_METADATA_SUFFIX)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in metadata["shape"])
    expected_bytes = int(np.prod(shape)) * np.dtype(metadata["dtype"]).itemsize
    if output_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Shared crop mapping has {output_path.stat().st_size} bytes, expected {expected_bytes}"
        )
    mapped = np.memmap(output_path, mode="r+", dtype=metadata["dtype"], shape=shape)
    return torch.from_numpy(mapped), metadata


def create_batch_memmap(
    input_path,
    bbx_xys,
    output_path,
    *,
    source_sha256,
    img_ds=0.5,
    img_dst_size=256,
):
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + MEMMAP_METADATA_SUFFIX)
    output_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    images, resized_boxes = get_batch_memmap(
        input_path, bbx_xys, output_path, img_ds=img_ds, img_dst_size=img_dst_size
    )
    metadata = {
        "schema": "shared-normalized-crops-v1",
        "source_sha256": source_sha256,
        "shape": list(images.shape),
        "dtype": "float32",
        "img_ds": img_ds,
        "img_dst_size": img_dst_size,
        "bbx_xys": resized_boxes.tolist(),
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    return images, resized_boxes


def get_or_create_batch_memmap(
    input_path,
    bbx_xys,
    output_path,
    *,
    source_sha256,
    img_ds=0.5,
    img_dst_size=256,
):
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + MEMMAP_METADATA_SUFFIX)
    if output_path.is_file() and metadata_path.is_file():
        try:
            images, metadata = open_batch_memmap(output_path)
            expected_shape = (len(bbx_xys), 3, img_dst_size, img_dst_size)
            if (
                metadata.get("schema") == "shared-normalized-crops-v1"
                and metadata.get("source_sha256") == source_sha256
                and tuple(images.shape) == expected_shape
                and float(metadata.get("img_ds")) == float(img_ds)
                and int(metadata.get("img_dst_size")) == int(img_dst_size)
            ):
                boxes = torch.tensor(metadata["bbx_xys"], dtype=torch.float32)
                if tuple(boxes.shape) == (len(bbx_xys), 3):
                    return images, boxes, True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    images, boxes = create_batch_memmap(
        input_path,
        bbx_xys,
        output_path,
        source_sha256=source_sha256,
        img_ds=img_ds,
        img_dst_size=img_dst_size,
    )
    return images, boxes, False


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        normalized, bbx_xys_ds = _prepare_crop(
            imgs[i], gt_center[i].numpy(), float(gt_bbx_size[i]), img_ds, img_dst_size
        )
        imgs_list.append(normalized)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.stack(imgs_list)
    bbx_xys = torch.stack(bbx_xys_ds_list)
    return imgs, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave=True, batch_size=16, inference_dtype="fp32"):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave
        self.batch_size = batch_size
        self.inference_dtype = inference_dtype

    @torch.inference_mode()
    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        batch_size = self.batch_size
        autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.inference_dtype)
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size].cuda(non_blocking=True)

            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().float().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
