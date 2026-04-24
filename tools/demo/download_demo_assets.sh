#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_DPVO=0

usage() {
    cat <<'EOF'
Usage: bash tools/demo/download_demo_assets.sh [--with-dpvo]

Downloads the minimum checkpoint files needed to run the GVHMR demo.
By default this is enough for:
  python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s

Pass --with-dpvo if you also want inputs/checkpoints/dpvo/dpvo.pth.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-dpvo)
            WITH_DPVO=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

download_file() {
    local url="$1"
    local output="$2"

    if [[ -f "$output" ]]; then
        echo "[skip] $output"
        return 0
    fi

    mkdir -p "$(dirname "$output")"
    echo "[download] $output"

    if command -v aria2c >/dev/null 2>&1; then
        aria2c --console-log-level=error -c -x 8 -s 8 -k 1M "$url" -d "$(dirname "$output")" -o "$(basename "$output")"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$output" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$output"
    else
        echo "Need one of: aria2c, wget, curl" >&2
        exit 1
    fi
}

cd "$ROOT_DIR"

download_file \
    "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl" \
    "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
download_file \
    "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz" \
    "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
download_file \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt" \
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
download_file \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt" \
    "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
download_file \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth" \
    "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
download_file \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt" \
    "inputs/checkpoints/yolo/yolov8x.pt"

if [[ "$WITH_DPVO" -eq 1 ]]; then
    download_file \
        "https://huggingface.co/camenduru/GVHMR/resolve/main/dpvo/dpvo.pth" \
        "inputs/checkpoints/dpvo/dpvo.pth"
fi

echo
echo "Assets are ready under $ROOT_DIR/inputs/checkpoints"
echo "Static-camera demo:"
echo "  python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s"
