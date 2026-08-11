import argparse
from pathlib import Path
from tqdm import tqdm
from hmr4d.utils.pylogger import Log
import subprocess
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-d", "--output_root", type=str, default=None)
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--model", choices=("footmr", "gvhmr"), default="footmr")
    parser.add_argument("--use_sapiens", action="store_true")
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--pose_batch_size", type=int, default=16)
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument("--render", choices=("all", "none", "incam", "global"), default="all")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cache_root", type=str, default="outputs/cache")
    args = parser.parse_args()

    folder = Path(args.folder)
    output_root = args.output_root

    # Run demo.py for each .mp4 file
    mp4_paths = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.MP4")))
    Log.info(f"Found {len(mp4_paths)} .mp4 files in {folder}")
    for mp4_path in tqdm(mp4_paths):
        command = [
            "python",
            "tools/demo/demo.py",
            "--video",
            str(mp4_path),
            "--model",
            args.model,
            "--pose_batch_size",
            str(args.pose_batch_size),
            "--feature_batch_size",
            str(args.feature_batch_size),
            "--render",
            args.render,
            "--cache_root",
            args.cache_root,
        ]
        if output_root is not None:
            command += ["--output_root", output_root]
        if args.static_cam:
            command += ["-s"]
        if args.use_sapiens:
            command += ["--use_sapiens"]
        if args.no_postproc:
            command += ["--no_postproc"]
        if args.profile:
            command += ["--profile"]
        Log.info(f"Running: {' '.join(command)}")
        subprocess.run(command, env=dict(os.environ), check=True)
