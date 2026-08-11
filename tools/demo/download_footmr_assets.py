"""Download and verify the public FootMR inference assets.

Assets are deliberately stored outside ``inputs/checkpoints`` because this
worktree may share that path with another project through a symbolic link.
"""

import argparse
import base64
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


SHARE_TOKEN = "tpLX3F6Mz4FqHaD"
BASE_URL = "https://cloud.tnt.uni-hannover.de/public.php/webdav/checkpoints"
ASSETS = {
    "footmr": {
        "url": f"{BASE_URL}/footmr/footmr_checkpoint.ckpt",
        "filename": "footmr_checkpoint.ckpt",
        "sha256": "2d31d8b5f7079c86dc176472909d4b3c14db801f0bcd9f99571637d0a860407a",
    },
    "vitpose": {
        "url": f"{BASE_URL}/vitpose/vitpose-h-wholebody.pth",
        "filename": "vitpose-h-wholebody.pth",
        "sha256": "dbed01fd5bb221610bf26434ec63426025f76eaca46f6177db71c9771a43316c",
    },
}


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(name, asset, output_root):
    destination = output_root / asset["filename"]
    if destination.is_file() and sha256(destination) == asset["sha256"]:
        print(f"OK    {name}: {destination}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(asset["url"])
    auth = base64.b64encode(f"{SHARE_TOKEN}:".encode()).decode()
    request.add_header("Authorization", f"Basic {auth}")

    print(f"GET   {name}: {asset['url']}")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r      {received / 2**20:.0f}/{total / 2**20:.0f} MiB", end="", flush=True)
        print()
        actual = sha256(partial)
        if actual != asset["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {name}: {actual}")
        os.replace(partial, destination)
        print(f"OK    {name}: {destination}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("inputs/footmr_assets"))
    parser.add_argument("--only", choices=tuple(ASSETS), action="append")
    args = parser.parse_args()
    selected = args.only or list(ASSETS)
    for name in selected:
        download_asset(name, ASSETS[name], args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
