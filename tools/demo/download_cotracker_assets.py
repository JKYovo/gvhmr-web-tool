"""Download and verify the CoTracker3 offline inference checkpoint.

The asset is kept outside ``inputs/checkpoints`` because that directory can be
shared with gvhmr-web-tool through a symbolic link.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inputs/cotracker_assets/scaled_offline.pth"),
    )
    args = parser.parse_args()
    destination = args.output
    if destination.is_file() and sha256(destination) == SHA256:
        print(f"OK    {destination}")
        return
    if destination.parent.is_symlink():
        raise RuntimeError(f"Refusing to write through a symlink: {destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        print(f"GET   {URL}")
        with urllib.request.urlopen(URL) as response, partial.open("wb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
        actual = sha256(partial)
        if actual != SHA256:
            raise RuntimeError(f"SHA256 mismatch: expected {SHA256}, got {actual}")
        os.replace(partial, destination)
        print(f"OK    {destination}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
