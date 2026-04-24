import json
import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d import CHECKPOINT_ENV_VAR, PROJ_ROOT, os_chdir_to_proj_root, resolve_checkpoint_path
from hmr4d.build_gvhmr import build_gvhmr_demo
from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo.hmr_cam import create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy, convert_K_to_K4
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay, compute_cam_angvel
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_horizontal,
    read_video_np,
    save_video,
    transcode_video_normalized,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points


CRF = 23
PROCESSED_FPS = 30
SCHEMA_VERSION = 1
PRIMARY_PERSON_MODE = "single_primary_track"
_REQUIRED_CHECKPOINTS = (
    ("GVHMR checkpoint", ("gvhmr", "gvhmr_siga24_release.ckpt")),
    ("SMPL body model", ("body_models", "smpl", "SMPL_NEUTRAL.pkl")),
    ("SMPL-X body model", ("body_models", "smplx", "SMPLX_NEUTRAL.npz")),
    ("HMR2 checkpoint", ("hmr2", "epoch=10-step=25000.ckpt")),
    ("ViTPose checkpoint", ("vitpose", "vitpose-h-multi-coco.pth")),
    ("YOLO checkpoint", ("yolo", "yolov8x.pt")),
)


class _CallbackLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__(level=logging.INFO)
        self._callback = callback
        self.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%m/%d %H:%M:%S"))

    def emit(self, record):
        try:
            self._callback(self.format(record))
        except Exception:
            pass


@contextmanager
def _capture_logs(log_callback=None):
    if log_callback is None:
        yield
        return

    handler = _CallbackLogHandler(log_callback)
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def _resolve_path(path_like):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (PROJ_ROOT / path).resolve()
    return path


def _normalize_optional_f_mm(f_mm):
    if f_mm in (None, "", 0):
        return None
    return int(f_mm)


def _video_metadata(video_path, *, display_oriented=False):
    video_path = Path(video_path)
    length, width, height = get_video_lwh(video_path, display_oriented=display_oriented)
    metadata = iio.immeta(video_path, plugin="pyav")
    fps = metadata.get("fps")
    return {
        "num_frames": int(length),
        "width": int(width),
        "height": int(height),
        "fps": float(fps) if fps is not None else None,
    }


def _tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_prediction(pred):
    return {
        "smpl_global_body_pose": _tensor_to_numpy(pred["smpl_params_global"]["body_pose"]),
        "smpl_global_global_orient": _tensor_to_numpy(pred["smpl_params_global"]["global_orient"]),
        "smpl_global_transl": _tensor_to_numpy(pred["smpl_params_global"]["transl"]),
        "smpl_global_betas": _tensor_to_numpy(pred["smpl_params_global"]["betas"]),
        "smpl_incam_body_pose": _tensor_to_numpy(pred["smpl_params_incam"]["body_pose"]),
        "smpl_incam_global_orient": _tensor_to_numpy(pred["smpl_params_incam"]["global_orient"]),
        "smpl_incam_transl": _tensor_to_numpy(pred["smpl_params_incam"]["transl"]),
        "smpl_incam_betas": _tensor_to_numpy(pred["smpl_params_incam"]["betas"]),
        "camera_K_fullimg": _tensor_to_numpy(pred["K_fullimg"]),
    }


def make_output_dir(video_path, output_root, add_timestamp=False):
    video_path = Path(video_path)
    output_root = _resolve_path(output_root)
    suffix = ""
    if add_timestamp:
        suffix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{video_path.stem}{suffix}"


def build_demo_cfg(output_dir, static_cam, f_mm=None, use_dpvo=False, verbose=False):
    output_dir = _resolve_path(output_dir)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo")

    cfg.video_name = output_dir.name
    cfg.output_root = str(output_dir.parent)
    cfg.static_cam = bool(static_cam)
    cfg.verbose = bool(verbose)
    cfg.use_dpvo = bool(use_dpvo)
    cfg.f_mm = _normalize_optional_f_mm(f_mm)
    cfg.ckpt_path = str(resolve_checkpoint_path("gvhmr", "gvhmr_siga24_release.ckpt"))
    OmegaConf.resolve(cfg)
    return cfg


def _prepare_video_copy(source_video_path, target_video_path):
    source_video_path = _resolve_path(source_video_path)
    target_video_path = Path(target_video_path)
    Log.info(f"[Input]: {source_video_path}")
    source_meta = _video_metadata(source_video_path, display_oriented=True)
    Log.info(
        f"(L, W, H) = ({source_meta['num_frames']}, {source_meta['width']}, {source_meta['height']})"
    )

    if source_video_path == target_video_path and target_video_path.exists():
        return source_meta

    should_copy = True
    if target_video_path.exists():
        try:
            target_meta = _video_metadata(target_video_path, display_oriented=False)
            should_copy = (
                source_meta["num_frames"] != target_meta["num_frames"]
                or source_meta["width"] != target_meta["width"]
                or source_meta["height"] != target_meta["height"]
            )
        except Exception:
            should_copy = True

    if should_copy:
        Log.info(f"[Copy Video] {source_video_path} -> {target_video_path}")
        transcode_video_normalized(source_video_path, target_video_path, fps=PROCESSED_FPS, crf=CRF)
    else:
        Log.info(f"[Copy Video] Reusing {target_video_path}")

    return source_meta


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    if not static_cam:
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()
                torch.save(vo_results, paths.slam)
            else:
                from hmr4d.utils.preproc.slam import SLAMModel

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
                slam_results = slam.process()
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    return {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return incam_video_path

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(PROJ_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    video_path = cfg.video_path
    _, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)
    writer = get_writer(incam_video_path, fps=PROCESSED_FPS, crf=CRF)
    try:
        for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc="Rendering Incam"):
            img = renderer.render_mesh(pred_c_verts[i].cuda(), img_raw, [0.8, 0.8, 0.8])
            writer.write_frame(img)
    finally:
        writer.close()
        reader.close()
    return incam_video_path


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return global_video_path

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(PROJ_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(PROJ_ROOT / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        verts = verts.clone()
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    video_path = cfg.video_path
    _, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(global_video_path, fps=PROCESSED_FPS, crf=CRF)
    try:
        for i in tqdm(range(get_video_lwh(video_path)[0]), desc="Rendering Global"):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
            writer.write_frame(img)
    finally:
        writer.close()

    return global_video_path


class GVHMRRunner:
    def __init__(self, checkpoint_root="inputs/checkpoints", device="cuda"):
        os_chdir_to_proj_root()
        self.checkpoint_root = _resolve_path(checkpoint_root)
        self.device = device
        self._model = None
        self._cuda_available = torch.cuda.is_available()

        if self.device != "cuda":
            raise ValueError("GVHMRRunner v1 only supports device='cuda'.")

        os.environ[CHECKPOINT_ENV_VAR] = str(self.checkpoint_root)
        self._validate_checkpoint_root()
        Log.info(f"[Checkpoint Root]: {self.checkpoint_root}")
        if self._cuda_available:
            Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
            Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')
        else:
            Log.warning("[GPU]: CUDA is unavailable. Cached export-only operations can still run, but inference and rendering require CUDA.")

    def _require_cuda(self, task_name):
        if not self._cuda_available:
            raise RuntimeError(
                f"GVHMR requires a CUDA-enabled NVIDIA GPU for {task_name}, but torch.cuda.is_available() is False."
            )

    def _validate_checkpoint_root(self):
        missing = []
        for label, parts in _REQUIRED_CHECKPOINTS:
            candidate = self.checkpoint_root.joinpath(*parts)
            if not candidate.exists():
                missing.append(f"{label}: {candidate}")
        if missing:
            raise FileNotFoundError("Missing required GVHMR assets:\n" + "\n".join(missing))

    def _get_model(self):
        if self._model is None:
            self._require_cuda("model loading")
            Log.info("[HMR4D] Loading GVHMR model")
            self._model = build_gvhmr_demo(self.checkpoint_root).to(self.device)
        return self._model

    def _predict(self, data, static_cam):
        model: DemoPL = self._get_model()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=static_cam)
        pred = detach_to_cpu(pred)
        data_time = int(data["length"]) / PROCESSED_FPS
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        return pred

    def _write_exports(self, cfg, source_video_path):
        pred = torch.load(cfg.paths.hmr4d_results, map_location="cpu")
        flat_pred = _flatten_prediction(pred)
        npz_path = Path(cfg.output_dir) / "gvhmr_data.npz"
        meta_path = Path(cfg.output_dir) / "gvhmr_meta.json"
        np.savez_compressed(npz_path, **flat_pred)

        source_meta = _video_metadata(source_video_path, display_oriented=True)
        processed_meta = _video_metadata(cfg.video_path)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "source_video_path": str(_resolve_path(source_video_path)),
            "source_num_frames": source_meta["num_frames"],
            "source_width": source_meta["width"],
            "source_height": source_meta["height"],
            "source_fps": source_meta["fps"],
            "processed_num_frames": processed_meta["num_frames"],
            "processed_fps": PROCESSED_FPS,
            "static_cam": bool(cfg.static_cam),
            "f_mm": cfg.f_mm,
            "output_dir": str(_resolve_path(cfg.output_dir)),
            "person_mode": PRIMARY_PERSON_MODE,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "output_dir": str(_resolve_path(cfg.output_dir)),
            "data_path": str(npz_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "hmr4d_results_path": str(_resolve_path(cfg.paths.hmr4d_results)),
            "meta": meta,
        }

    def process_video(
        self,
        video_path,
        output_dir,
        static_cam,
        f_mm=None,
        save_intermediate=False,
        *,
        use_dpvo=False,
        verbose=False,
        log_callback=None,
    ):
        with _capture_logs(log_callback):
            source_video_path = _resolve_path(video_path)
            if not source_video_path.exists():
                raise FileNotFoundError(f"Video not found at {source_video_path}")

            cfg = build_demo_cfg(
                output_dir=output_dir,
                static_cam=static_cam,
                f_mm=f_mm,
                use_dpvo=use_dpvo,
                verbose=verbose,
            )

            output_dir = Path(cfg.output_dir)
            preprocess_dir = Path(cfg.preprocess_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            Log.info(f"[Output Dir]: {output_dir}")

            _prepare_video_copy(source_video_path, cfg.video_path)

            if not Path(cfg.paths.hmr4d_results).exists():
                self._require_cuda("video processing")
                run_preprocess(cfg)
                data = load_data_dict(cfg)
                Log.info("[HMR4D] Predicting")
                pred = self._predict(data, static_cam=cfg.static_cam)
                torch.save(pred, cfg.paths.hmr4d_results)
            else:
                Log.info(f"[HMR4D] Reusing cached result at {cfg.paths.hmr4d_results}")

            result = self._write_exports(cfg, source_video_path)

            if not save_intermediate and preprocess_dir.exists():
                shutil.rmtree(preprocess_dir)
                Log.info(f"[Cleanup] Removed preprocess artifacts at {preprocess_dir}")

            return result

    def generate_preview(self, output_dir, log_callback=None):
        with _capture_logs(log_callback):
            cfg = build_demo_cfg(output_dir=output_dir, static_cam=True)
            output_dir = Path(cfg.output_dir)
            result_path = Path(cfg.paths.hmr4d_results)
            video_path = Path(cfg.video_path)

            if not output_dir.exists():
                raise FileNotFoundError(f"Output directory not found at {output_dir}")
            if not result_path.exists():
                raise FileNotFoundError(f"Missing inference result at {result_path}")
            if not video_path.exists():
                raise FileNotFoundError(f"Missing processed video at {video_path}")

            if not Path(cfg.paths.incam_video).exists() or not Path(cfg.paths.global_video).exists():
                self._require_cuda("preview generation")

            render_incam(cfg)
            render_global(cfg)
            if not Path(cfg.paths.incam_global_horiz_video).exists():
                Log.info("[Merge Videos]")
                merge_videos_horizontal(
                    [cfg.paths.incam_video, cfg.paths.global_video],
                    cfg.paths.incam_global_horiz_video,
                )

            return {
                "incam_video_path": str(_resolve_path(cfg.paths.incam_video)),
                "global_video_path": str(_resolve_path(cfg.paths.global_video)),
                "preview_video_path": str(_resolve_path(cfg.paths.incam_global_horiz_video)),
            }
