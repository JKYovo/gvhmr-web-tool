import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
import shutil
import uuid
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d import PROJ_ROOT
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_fps_duration,
    normalize_video_fps,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.preproc.content_cache import PreprocessContentCache, file_identity, sha256_file
from hmr4d.utils.preproc.vitfeat_extractor import get_batch, get_or_create_batch_memmap
from hmr4d.network.hmr2 import HMR2A_CKPT

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.perf import NullProfiler, StageProfiler
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange


CRF = 23  # 17 is lossless, every +6 halves the mp4 size
RENDER_SMPLX_BATCH_SIZE = 128


@torch.no_grad()
def smplx_to_smpl_vertices_batched(smplx, smplx2smpl, params, batch_size=RENDER_SMPLX_BATCH_SIZE):
    """Generate SMPL render vertices without putting a whole long sequence on CUDA."""
    lengths = {int(value.shape[0]) for value in params.values() if torch.is_tensor(value)}
    if len(lengths) != 1:
        raise ValueError(f"SMPL-X render parameter lengths do not match: {sorted(lengths)}")
    length = lengths.pop()
    if length < 1 or batch_size < 1:
        raise ValueError(f"Invalid render length/batch size: {length}/{batch_size}")
    chunks = []
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batch = {
            key: value[start:end] if torch.is_tensor(value) else value
            for key, value in params.items()
        }
        vertices = smplx(**to_cuda(batch)).vertices
        converted = torch.stack([torch.matmul(smplx2smpl, frame) for frame in vertices])
        chunks.append(converted.cpu())
        del vertices, converted
    return torch.cat(chunks, dim=0)


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument(
        "--model",
        choices=("footmr", "gvhmr"),
        default="footmr",
        help="FootMR is the enhanced default; use gvhmr for the original baseline.",
    )
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
    parser.add_argument(
        "--use_sapiens",
        action="store_true",
        help="Use optional Sapiens whole-body keypoints for FootMR instead of ViTPose.",
    )
    parser.add_argument(
        "--no_postproc",
        action="store_true",
        help="Disable global contact/IK post-processing, which can suppress fine foot motion.",
    )
    parser.add_argument(
        "--pose_batch_size",
        type=int,
        default=16,
        help="ViTPose inference batch size; reduce this if GPU memory is limited.",
    )
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument(
        "--render",
        choices=("all", "none", "incam", "global"),
        default="all",
        help="Select preview rendering; use none when only the GMR tensor is needed.",
    )
    parser.add_argument("--profile", action="store_true", help="Write synchronized per-stage timings as JSON.")
    parser.add_argument(
        "--shared_preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode and crop once for ViTPose and HMR2.",
    )
    parser.add_argument("--inference_dtype", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--pose_inference_dtype", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--feature_inference_dtype", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument(
        "--pose_flip_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable only for the experimental fast mode; it can affect foot keypoints.",
    )
    parser.add_argument("--attention_impl", choices=("dense", "local"), default="dense")
    parser.add_argument("--attention_chunk_size", type=int, default=128)
    parser.add_argument(
        "--cache_root",
        type=str,
        default="outputs/cache",
        help="Cross-job preprocessing cache; pass none to disable.",
    )
    args = parser.parse_args()
    default_pose_dtype = "fp16" if args.model == "footmr" else "fp32"
    pose_inference_dtype = args.pose_inference_dtype or args.inference_dtype or default_pose_dtype
    feature_inference_dtype = args.feature_inference_dtype or args.inference_dtype or "fp32"
    if args.use_sapiens and args.model != "footmr":
        parser.error("--use_sapiens is only valid with --model footmr")
    if args.pose_batch_size < 1 or args.feature_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.attention_chunk_size < 1:
        parser.error("--attention_chunk_size must be positive")

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        variant = "gvhmr" if args.model == "gvhmr" else (
            "footmr_sapiens" if args.use_sapiens else "footmr_vitpose"
        )
        pose_cache_name = {
            "gvhmr": "kp2d_coco17",
            "footmr_vitpose": "kp2d_coco23_vitpose",
            "footmr_sapiens": "kp2d_coco23_sapiens",
        }[variant]
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
            f"variant={variant}",
            f"pose_cache_name={pose_cache_name}",
            f"use_sapiens={args.use_sapiens}",
            f"no_postproc={args.no_postproc}",
            f"pose_batch_size={args.pose_batch_size}",
            f"feature_batch_size={args.feature_batch_size}",
            f"shared_preprocess={args.shared_preprocess}",
            f"inference_dtype={args.inference_dtype or 'fp32'}",
            f"pose_inference_dtype={pose_inference_dtype}",
            f"feature_inference_dtype={feature_inference_dtype}",
            f"pose_flip_test={args.pose_flip_test}",
            f"attention_impl={args.attention_impl}",
            f"attention_chunk_size={args.attention_chunk_size}",
            f"network.attention_impl={args.attention_impl}",
            f"network.attention_chunk_size={args.attention_chunk_size}",
            f"cache_root={args.cache_root}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        config_name = "demo" if args.model == "footmr" else "demo_gvhmr"
        cfg = compose(config_name=config_name, overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.sequence_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    cfg.source_video_path = str(video_path.resolve())
    cfg.render_mode = args.render
    cfg.profile = args.profile
    return cfg


def ensure_normalized_video(cfg, profiler=None):
    """Normalize to 30 FPS by timestamps, including Web external-worker jobs."""
    profiler = profiler or NullProfiler()
    destination = Path(cfg.video_path).expanduser().resolve()
    configured_source = getattr(cfg, "source_video_path", None)
    source = Path(configured_source).expanduser().resolve() if configured_source else None
    if source is None or not source.is_file():
        # The Web external worker keeps the submitted source beside its output.
        candidate = Path(cfg.output_dir).expanduser().resolve() / "submitted_input.mp4"
        if candidate.is_file():
            source = candidate
    if source is None or not source.is_file() or source == destination:
        return None

    Log.info(f"[Normalize Video] {source} -> {destination} at 30 FPS")
    with profiler.section("video.normalize_30fps"):
        source_fps, source_duration = get_video_fps_duration(source)

        def valid_normalized(path):
            if not path.is_file():
                return None
            output_fps, output_duration = get_video_fps_duration(path)
            duration_matches = (
                source_duration is None
                or output_duration is None
                or abs(source_duration - output_duration) <= max(0.1, 1.5 / 30)
            )
            if abs(output_fps - 30) >= 1e-3 or not duration_matches:
                return None
            return {
                "source_fps": source_fps,
                "source_duration": source_duration,
                "output_fps": output_fps,
                "output_duration": output_duration,
                "reused": True,
            }

        metadata = valid_normalized(destination)
        cache_root = getattr(cfg, "cache_root", None)
        cache_enabled = cache_root not in (None, "", "none", "None")
        normalized_cache = None
        if metadata is None and cache_enabled:
            source_digest = sha256_file(source)
            normalized_cache = Path(cache_root).expanduser() / "normalized-video-v1" / f"{source_digest}.mp4"
            cached_metadata = valid_normalized(normalized_cache)
            if cached_metadata is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.cache-tmp")
                shutil.copy2(normalized_cache, temporary)
                temporary.replace(destination)
                metadata = cached_metadata

        if metadata is None:
            metadata = normalize_video_fps(source, destination, target_fps=30, crf=CRF)
            if cache_enabled:
                if normalized_cache is None:
                    source_digest = sha256_file(source)
                    normalized_cache = Path(cache_root).expanduser() / "normalized-video-v1" / f"{source_digest}.mp4"
                normalized_cache.parent.mkdir(parents=True, exist_ok=True)
                if not normalized_cache.is_file():
                    temporary = normalized_cache.with_suffix(
                        normalized_cache.suffix + f".{uuid.uuid4().hex}.tmp"
                    )
                    shutil.copy2(destination, temporary)
                    temporary.replace(normalized_cache)
    Log.info(
        "[Normalize Video] "
        f"{metadata['source_fps']:.3f} FPS/{metadata['source_duration']:.3f}s -> "
        f"{metadata['output_fps']:.3f} FPS/{metadata['output_duration']:.3f}s"
    )
    return metadata


@torch.no_grad()
def run_preprocess(cfg, profiler=None):
    profiler = profiler or NullProfiler()
    ensure_normalized_video(cfg, profiler)
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose
    expected_length = get_video_lwh(video_path)[0]

    def discard_if_wrong_length(path, tensor_key=None):
        path = Path(path)
        if not path.is_file():
            return
        try:
            value = torch.load(path, map_location="cpu")
            if tensor_key is not None:
                value = value[tensor_key]
            valid = len(value) == expected_length
        except Exception as exc:
            Log.warn(f"Ignoring unreadable cache {path}: {exc}")
            valid = False
        if not valid:
            Log.warn(f"Ignoring cache with the wrong frame count: {path}")
            path.unlink()

    discard_if_wrong_length(paths.bbx, "bbx_xys")
    discard_if_wrong_length(paths.vitpose)
    discard_if_wrong_length(paths.vit_features)
    if not static_cam:
        discard_if_wrong_length(paths.slam)

    with profiler.section("cache.hash_video"):
        content_cache = PreprocessContentCache(getattr(cfg, "cache_root", None), video_path)

    def restore_cache(stage, options, destination):
        if Path(destination).exists():
            return False
        with profiler.section(f"cache.restore.{stage}"):
            restored = content_cache.restore(stage, options, destination)
        if restored:
            Log.info(f"[Cache] Restored {stage} -> {destination}")
        return restored

    def store_cache(stage, options, source):
        with profiler.section(f"cache.store.{stage}"):
            content_cache.store(stage, options, source)

    bbox_cache_options = {"tracker": "yolov8x", "conf": 0.5, "enlarge": 1.2}
    restore_cache("bbx", bbox_cache_options, paths.bbx)
    discard_if_wrong_length(paths.bbx, "bbx_xys")

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        with profiler.section("preprocess.tracker"):
            tracker = Tracker()
            bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
            torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
            del tracker
    else:
        bbx_data = torch.load(paths.bbx)
        bbx_xys = bbx_data["bbx_xys"]
        bbx_xyxy = bbx_data["bbx_xyxy"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    store_cache("bbx", bbox_cache_options, paths.bbx)
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    pose_inference_dtype = str(getattr(cfg, "pose_inference_dtype", getattr(cfg, "inference_dtype", "fp32")))
    feature_inference_dtype = str(
        getattr(cfg, "feature_inference_dtype", getattr(cfg, "inference_dtype", "fp32"))
    )
    pose_flip_test = bool(getattr(cfg, "pose_flip_test", True))
    if cfg.use_sapiens:
        pose_cache_options = {
            "backend": "sapiens",
            "num_joints": int(cfg.num_joints),
        }
    else:
        pose_checkpoint = (
            PROJ_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
            if cfg.num_joints == 17
            else PROJ_ROOT / "inputs/footmr_assets/vitpose-h-wholebody.pth"
        )
        pose_cache_options = {
            "backend": "vitpose",
            "num_joints": int(cfg.num_joints),
            "flip_test": pose_flip_test,
            "inference_dtype": pose_inference_dtype,
            "checkpoint": file_identity(pose_checkpoint),
        }
    feature_cache_options = {
        "backend": "hmr2",
        "inference_dtype": feature_inference_dtype,
        "checkpoint": file_identity(HMR2A_CKPT),
    }
    restore_cache("pose", pose_cache_options, paths.vitpose)
    restore_cache("features", feature_cache_options, paths.vit_features)
    discard_if_wrong_length(paths.vitpose)
    discard_if_wrong_length(paths.vit_features)

    # Get 2D pose. Cache names are backend-specific to prevent COCO17/23 reuse.
    vitpose = None
    if Path(paths.vitpose).exists():
        cached_vitpose = torch.load(paths.vitpose)
        if cached_vitpose.shape[-2] == cfg.num_joints:
            vitpose = cached_vitpose
            Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
        else:
            Log.warn(
                f"Ignoring incompatible pose cache at {paths.vitpose}: "
                f"expected {cfg.num_joints} joints, got {cached_vitpose.shape[-2]}"
            )

    need_pose = vitpose is None
    need_features = not Path(paths.vit_features).exists()
    shared_imgs = None
    shared_bbx_xys = None
    use_shared_crop = (
        not cfg.use_sapiens
        and bool(getattr(cfg, "shared_preprocess", False))
        and (
            (need_pose and need_features)
            or (expected_length >= 1800 and (need_pose or need_features))
        )
    )
    if use_shared_crop:
        with profiler.section("preprocess.shared_decode_crop"):
            if expected_length >= 1800:
                # Keep the large temporary mapping with the job. A failed run
                # can resume it, while a completed pose+feature pass removes it.
                shared_crop_path = Path(cfg.preprocess_dir) / "shared_crops_float32.mmap"
                shared_imgs, shared_bbx_xys, reused_shared_crop = get_or_create_batch_memmap(
                    video_path,
                    bbx_xys,
                    shared_crop_path,
                    source_sha256=content_cache.video_sha256,
                    img_ds=0.5,
                )
                if reused_shared_crop:
                    Log.info(f"[Cache] Restored shared crop -> {shared_crop_path}")
            else:
                shared_crop_path = None
                shared_imgs, shared_bbx_xys = get_batch(video_path, bbx_xys, img_ds=0.5)

    if need_pose:
        if cfg.use_sapiens:
            from hmr4d.utils.preproc.sapiens import SapiensPoseExtractor

            with profiler.section("preprocess.pose_sapiens"):
                vitpose_extractor = SapiensPoseExtractor()
                vitpose = vitpose_extractor.extract(video_path, paths.extracted_frames, bbx_xyxy)
        else:
            with profiler.section("preprocess.pose_vitpose"):
                vitpose_extractor = VitPoseExtractor(
                    number_joints=cfg.num_joints,
                    batch_size=cfg.pose_batch_size,
                    inference_dtype=pose_inference_dtype,
                    flip_test=pose_flip_test,
                )
                pose_input = shared_imgs if shared_imgs is not None else video_path
                pose_bbx = shared_bbx_xys if shared_bbx_xys is not None else bbx_xys
                vitpose = vitpose_extractor.extract(pose_input, pose_bbx)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
        store_cache("pose", pose_cache_options, paths.vitpose)
    else:
        store_cache("pose", pose_cache_options, paths.vitpose)
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco_skeleton_batch(video, vitpose, cfg.num_joints, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if need_features:
        with profiler.section("preprocess.hmr2_features"):
            extractor = Extractor(
                batch_size=int(getattr(cfg, "feature_batch_size", 16)),
                inference_dtype=feature_inference_dtype,
            )
            feature_input = shared_imgs if shared_imgs is not None else video_path
            vit_features = extractor.extract_video_features(feature_input, bbx_xys)
            torch.save(vit_features, paths.vit_features)
            del extractor
        store_cache("features", feature_cache_options, paths.vit_features)
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")
        store_cache("features", feature_cache_options, paths.vit_features)
    if shared_imgs is not None:
        del shared_imgs
    if "shared_crop_path" in locals() and shared_crop_path is not None:
        shared_crop_path.unlink(missing_ok=True)
        shared_crop_path.with_suffix(shared_crop_path.suffix + ".json").unlink(
            missing_ok=True
        )

    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        slam_cache_options = {
            "backend": "dpvo" if cfg.use_dpvo else "simplevo-sift-step8-scale0.5",
            "f_mm": cfg.f_mm,
        }
        restore_cache("slam", slam_cache_options, paths.slam)
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                with profiler.section("preprocess.simple_vo"):
                    simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                    vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                    torch.save(vo_results, paths.slam)
            else:  # DPVO
                from hmr4d.utils.preproc.slam import SLAMModel

                with profiler.section("preprocess.dpvo"):
                    length, width, height = get_video_lwh(cfg.video_path)
                    K_fullimg = estimate_K(width, height)
                    intrinsics = convert_K_to_K4(K_fullimg)
                    slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                    bar = tqdm(total=length, desc="DPVO")
                    while True:
                        ret = slam.track()
                        if ret:
                            bar.update()
                        else:
                            break
                    slam_results = slam.process()  # (L, 7), numpy
                    torch.save(slam_results, paths.slam)
            store_cache("slam", slam_cache_options, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")
            store_cache("slam", slam_cache_options, paths.slam)

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def render_incam(cfg, profiler=None):
    profiler = profiler or NullProfiler()
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    with profiler.section("render.incam_smplx"):
        pred = torch.load(cfg.paths.hmr4d_results)
        smplx = make_smplx("supermotion").cuda()
        smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        faces_smpl = make_smplx("smpl").faces
        pred_c_verts = smplx_to_smpl_vertices_batched(
            smplx, smplx2smpl, pred["smpl_params_incam"]
        )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    with profiler.section("render.incam_frames_encode"):
        for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
            img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

            writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg, profiler=None):
    profiler = profiler or NullProfiler()
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    with profiler.section("render.global_smplx"):
        pred = torch.load(cfg.paths.hmr4d_results)
        smplx = make_smplx("supermotion").cuda()
        smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        faces_smpl = make_smplx("smpl").faces
        J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cpu()
        pred_ay_verts = smplx_to_smpl_vertices_batched(
            smplx, smplx2smpl, pred["smpl_params_global"]
        )

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    with profiler.section("render.global_frames_encode"):
        for i in tqdm(range(render_length), desc=f"Rendering Global"):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            img = renderer.render_with_ground(verts_glob[[i]].cuda(), color[None], cameras, global_lights)
            writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    profiler = StageProfiler(
        paths.performance,
        enabled=cfg.profile,
        metadata={
            "video": cfg.source_video_path,
            "variant": cfg.variant,
            "render_mode": cfg.render_mode,
            "inference_dtype": cfg.inference_dtype,
            "pose_inference_dtype": cfg.pose_inference_dtype,
            "feature_inference_dtype": cfg.feature_inference_dtype,
            "attention_impl": cfg.attention_impl,
        },
    )
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    try:
        # ===== Preprocess and save to disk ===== #
        run_preprocess(cfg, profiler=profiler)
        with profiler.section("data.load"):
            data = load_data_dict(cfg)
        profiler.add_metadata(frames=int(data["length"]), normalized_seconds=float(data["length"] / 30))

        # ===== HMR4D ===== #
        if not Path(paths.hmr4d_results).exists():
            Log.info("[HMR4D] Predicting")
            with profiler.section("hmr4d.model_load"):
                model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
                model.load_pretrained_model(cfg.ckpt_path, strict=model.pipeline.use_foot_refiner)
                model = model.eval().cuda()
            tic = Log.sync_time()
            with profiler.section("hmr4d.predict_total"):
                pred = model.predict(
                    data,
                    static_cam=cfg.static_cam,
                    no_postproc=cfg.no_postproc,
                    profiler=profiler,
                )
                pred = detach_to_cpu(pred)
            data_time = data["length"] / 30
            Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
            with profiler.section("hmr4d.save"):
                torch.save(pred, paths.hmr4d_results)

        # ===== Render ===== #
        if cfg.render_mode in ("all", "incam"):
            render_incam(cfg, profiler=profiler)
        if cfg.render_mode in ("all", "global"):
            render_global(cfg, profiler=profiler)
        if cfg.render_mode == "all" and not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            with profiler.section("render.merge"):
                merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
    finally:
        profiler.write()
