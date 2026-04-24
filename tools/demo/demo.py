import argparse
import site
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _sanitize_python_path():
    os_env = __import__("os").environ
    os_env.setdefault("PYTHONNOUSERSITE", "1")

    user_site_candidates = []
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if isinstance(user_site, str):
        user_site_candidates.append(Path(user_site).resolve())
    elif isinstance(user_site, (list, tuple)):
        user_site_candidates.extend(Path(path).resolve() for path in user_site)

    user_base = os_env.get("PYTHONUSERBASE")
    if user_base:
        user_site_candidates.append(Path(user_base).expanduser().resolve())
    user_site_candidates.append((Path.home() / ".local").resolve())

    sanitized = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            sanitized.append(entry)
            continue
        if any(candidate == resolved or candidate in resolved.parents for candidate in user_site_candidates):
            continue
        sanitized.append(entry)
    sys.path[:] = sanitized


_sanitize_python_path()

from hmr4d.api.video_to_data import GVHMRRunner, make_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser()
    output_root = args.output_root or "outputs/demo"
    output_dir = make_output_dir(video_path, output_root, add_timestamp=False)

    runner = GVHMRRunner()
    runner.process_video(
        video_path=video_path,
        output_dir=output_dir,
        static_cam=args.static_cam,
        f_mm=args.f_mm,
        save_intermediate=True,
        use_dpvo=args.use_dpvo,
        verbose=args.verbose,
    )
    runner.generate_preview(output_dir)


if __name__ == "__main__":
    main()
