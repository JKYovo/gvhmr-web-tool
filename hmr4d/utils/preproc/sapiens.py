import glob
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class SapiensPoseExtractor:
    """Optional Sapiens COCO-wholebody extractor used by FootMR."""

    def __init__(
        self,
        checkpoint="inputs/footmr_assets/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth",
        tqdm_leave=True,
    ):
        checkpoint = Path(checkpoint)
        sapiens_root = Path("third-party/sapiens")
        if not sapiens_root.is_dir():
            raise RuntimeError(
                "Sapiens submodule is not initialized. Run: "
                "git submodule update --init third-party/sapiens"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Sapiens checkpoint not found: {checkpoint}. See docs/INSTALL.md."
            )
        try:
            from mmpose.apis import inference_topdown
            from mmpose.apis import init_model as init_pose_estimator
            from mmpose.structures import merge_data_samples
        except ImportError as exc:
            raise RuntimeError(
                "Sapiens Python packages are not installed. Follow the optional "
                "Sapiens section in docs/INSTALL.md."
            ) from exc

        self.inference_topdown = inference_topdown
        self.merge_data_samples = merge_data_samples
        self.pose_estimator = init_pose_estimator(
            "hmr4d/configs/sapiens_2b-210e_coco_wholebody-1024x768.py",
            str(checkpoint),
            override_ckpt_meta=True,
            device="cuda",
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
        )
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, outfolder, bbx_xyxy):
        outfolder = Path(outfolder)
        outfolder.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-qscale:v", "2", str(outfolder / "%06d.jpg")],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        all_frames = sorted(glob.glob(str(outfolder / "*.jpg")))
        if len(all_frames) != len(bbx_xyxy):
            raise RuntimeError(
                f"Sapiens frame count mismatch: {len(all_frames)} images vs {len(bbx_xyxy)} boxes"
            )

        poses_2d = np.zeros((len(all_frames), 23, 3), dtype=np.float32)
        for i, img_path in tqdm(
            enumerate(all_frames), total=len(all_frames), desc="Sapiens", leave=self.tqdm_leave
        ):
            pose_results = self.inference_topdown(self.pose_estimator, img_path, bbx_xyxy[[i]])
            results = self.merge_data_samples(pose_results).get("pred_instances", None)
            keypoints = results["keypoints"][0][:23]
            confidence = results["keypoint_scores"][0][:23]
            poses_2d[i] = np.concatenate((keypoints, confidence[:, None]), axis=1).astype(np.float32)
        return torch.from_numpy(poses_2d)
