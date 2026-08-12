#!/usr/bin/env python3
"""Stream a prepared reference to an already-running local SONIC instance."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

import numpy as np

from hmr4d.utils.sonic import PlaybackState, SonicPlaybackController, SonicReference


def load_reference(path: Path) -> SonicReference:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SONIC reference does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return SonicReference(
            term1_local=data["term1_local"],
            root_quat=data["root_quat"],
            wrist=data["wrist"],
            fps=float(data["fps"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    args = parser.parse_args()
    reference = load_reference(args.reference)
    finished = threading.Event()
    errors: list[str] = []
    last_bucket = [-1]

    def callback(state, frame_index, frame_count, message):
        if state == PlaybackState.STREAMING:
            bucket = int(10 * frame_index / frame_count)
            if bucket != last_bucket[0]:
                print(f"{min(100, bucket * 10)}%", flush=True)
                last_bucket[0] = bucket
        elif state in (PlaybackState.COMPLETE, PlaybackState.STOPPED):
            print(state.value, flush=True)
            finished.set()
        elif state == PlaybackState.ERROR:
            errors.append(message or "unknown SONIC playback error")
            finished.set()

    duration = (reference.frame_count - 1) / reference.fps
    print(f"Streaming {reference.frame_count} frames @ {reference.fps:g} FPS ({duration:.2f}s) -> {args.endpoint}")
    controller = SonicPlaybackController(endpoint=args.endpoint)
    try:
        controller.run(0, reference, callback)
        finished.wait(duration + 10.0)
        if errors:
            raise RuntimeError(errors[0])
        if not finished.is_set():
            raise TimeoutError("SONIC playback did not finish before timeout")
    except KeyboardInterrupt:
        print("Stopping SONIC stream...", flush=True)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
