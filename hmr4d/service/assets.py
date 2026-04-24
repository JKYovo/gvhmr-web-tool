import argparse
import shutil
import urllib.request
from pathlib import Path

from hmr4d.service.common import ensure_dir


ASSET_MANIFEST = (
    (
        "SMPL body model",
        "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl",
        ("body_models", "smpl", "SMPL_NEUTRAL.pkl"),
    ),
    (
        "SMPL-X body model",
        "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz",
        ("body_models", "smplx", "SMPLX_NEUTRAL.npz"),
    ),
    (
        "GVHMR checkpoint",
        "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt",
        ("gvhmr", "gvhmr_siga24_release.ckpt"),
    ),
    (
        "HMR2 checkpoint",
        "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt",
        ("hmr2", "epoch=10-step=25000.ckpt"),
    ),
    (
        "ViTPose checkpoint",
        "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth",
        ("vitpose", "vitpose-h-multi-coco.pth"),
    ),
    (
        "YOLO checkpoint",
        "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt",
        ("yolo", "yolov8x.pt"),
    ),
)

DPVO_ASSET = (
    "DPVO checkpoint",
    "https://huggingface.co/camenduru/GVHMR/resolve/main/dpvo/dpvo.pth",
    ("dpvo", "dpvo.pth"),
)


def download_url(url, target_path):
    target_path = Path(target_path)
    ensure_dir(target_path.parent)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    tmp_path.replace(target_path)
    return target_path


def ensure_assets(checkpoint_root, *, with_dpvo=False, logger=None):
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    manifest = list(ASSET_MANIFEST)
    if with_dpvo:
        manifest.append(DPVO_ASSET)

    downloaded = []
    for label, url, relative_parts in manifest:
        target_path = checkpoint_root.joinpath(*relative_parts)
        if target_path.exists():
            if logger:
                logger(f"[Assets] Reusing {label}: {target_path}")
            continue
        if logger:
            logger(f"[Assets] Downloading {label}: {target_path}")
        download_url(url, target_path)
        downloaded.append(str(target_path))

    return {
        "checkpoint_root": str(checkpoint_root),
        "downloaded": downloaded,
        "with_dpvo": bool(with_dpvo),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--with-dpvo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = ensure_assets(args.checkpoint_root, with_dpvo=args.with_dpvo, logger=print)
    print(f"[Assets] Ready under {result['checkpoint_root']}")


if __name__ == "__main__":
    main()
