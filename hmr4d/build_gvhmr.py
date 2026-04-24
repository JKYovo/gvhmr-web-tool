from omegaconf import OmegaConf
from pathlib import Path
from hmr4d import PROJ_ROOT, resolve_checkpoint_path
from hydra.utils import instantiate
from hydra import compose, initialize_config_module
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.configs import register_store_gvhmr


def build_gvhmr_demo(checkpoint_root=None):
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo")
    gvhmr_demo_pl: DemoPL = instantiate(cfg.model, _recursive_=False)
    ckpt_path = (
        Path(checkpoint_root).expanduser().resolve() / "gvhmr/gvhmr_siga24_release.ckpt"
        if checkpoint_root is not None
        else resolve_checkpoint_path("gvhmr", "gvhmr_siga24_release.ckpt")
    )
    gvhmr_demo_pl.load_pretrained_model(ckpt_path)
    return gvhmr_demo_pl.eval()
