import torch
import pytorch_lightning as pl
from pathlib import Path
from hydra.utils import instantiate
from hmr4d import PROJ_ROOT
from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from hmr4d.utils.perf import NullProfiler


class DemoPL(pl.LightningModule):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

    @torch.no_grad()
    def predict(self, data, static_cam=False, no_postproc=False, profiler=None):
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 3)
            "f_imgseq": (F, 3, 256, 256)
        }

        """
        profiler = profiler or NullProfiler()
        with profiler.section("hmr4d.prepare_input"):
            batch = {
                "length": data["length"][None],
                "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
                "bbx_xys": data["bbx_xys"][None],
                "K_fullimg": data["K_fullimg"][None],
                "cam_angvel": data["cam_angvel"][None],
                "f_imgseq": data["f_imgseq"][None],
            }
            if self.pipeline.use_foot_refiner:
                assert data["kp2d"].shape[-2] == 23
                batch["foot_obs"] = batch["obs"][:, :, 15:23].clone()
            batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(
            batch,
            train=False,
            postproc=not no_postproc,
            static_cam=static_cam,
            profiler=profiler,
        )

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    def load_pretrained_model(self, ckpt_path, strict=False):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        ckpt_path = Path(ckpt_path)
        # gvhmr-web-tool's external worker predates FootMR and passes the
        # original checkpoint path explicitly. Keep that protocol compatible
        # while resolving the enhanced model's own combined checkpoint here.
        if self.pipeline.use_foot_refiner and ckpt_path.name == "gvhmr_siga24_release.ckpt":
            ckpt_path = PROJ_ROOT / "inputs/footmr_assets/footmr_checkpoint.ckpt"
            strict = True
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


MainStore.store(name="gvhmr_pl_demo", node=builds(DemoPL, pipeline="${pipeline}"), group="model/gvhmr")
