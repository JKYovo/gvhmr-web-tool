import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ENV_VAR = "GVHMR_CHECKPOINT_ROOT"
DEFAULT_CHECKPOINT_ROOT = PROJ_ROOT / "inputs/checkpoints"


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)


def get_checkpoint_root():
    checkpoint_root = os.environ.get(CHECKPOINT_ENV_VAR)
    if checkpoint_root:
        return Path(checkpoint_root).expanduser().resolve()
    return DEFAULT_CHECKPOINT_ROOT


def resolve_checkpoint_path(*parts):
    return get_checkpoint_root().joinpath(*parts)
