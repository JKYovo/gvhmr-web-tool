"""Standalone regression checks for the FootMR inference integration."""

import argparse
import gc
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_module


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr4d.configs import register_store_gvhmr  # noqa: E402
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: E402,F401
from hmr4d.network.footmr.foot_transformer import FootEncoderRoPE  # noqa: E402
from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE  # noqa: E402


def run(name, fn, failures):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as exc:
        failures.append(name)
        print(f"FAIL  {name}: {exc}")


def test_joint_parameterization():
    for joints in (17, 23):
        model = NetworkEncoderRoPE(
            num_2d_joints=joints,
            latent_dim=32,
            num_layers=1,
            num_heads=4,
            imgseq_dim=0,
            cam_angvel_dim=0,
        ).eval()
        output = model(
            length=torch.tensor([2]),
            obs=torch.zeros(1, 2, joints, 3),
            f_cliffcam=torch.zeros(1, 2, 3),
        )
        assert output["pred_x"].shape == (1, 2, 151)
        assert torch.isfinite(output["pred_x"]).all()


def test_refiner_shape():
    model = FootEncoderRoPE(latent_dim=32, num_layers=1, num_heads=4).eval()
    output = model(
        length=torch.tensor([2]),
        obs=torch.zeros(1, 2, 8, 3),
        f_cliffcam=torch.zeros(1, 2, 3),
        global_rot6d=torch.zeros(1, 2, 24),
    )
    assert output.shape == (1, 2, 12)
    assert torch.isfinite(output).all()


def synthetic_batch(length=2):
    obs = torch.zeros(1, length, 23, 3)
    obs[..., 2] = 1
    return {
        "length": torch.tensor([length]),
        "obs": obs,
        "foot_obs": obs[:, :, 15:23].clone(),
        "bbx_xys": torch.tensor([[[320.0, 240.0, 400.0]]]).repeat(1, length, 1),
        "K_fullimg": torch.tensor(
            [[[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]]]
        ).repeat(1, length, 1, 1),
        "cam_angvel": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]).repeat(
            1, length, 1
        ),
        "f_imgseq": torch.zeros(1, length, 1024),
    }


def test_cache_routing():
    baseline = compose(config_name="demo_gvhmr", overrides=["video_name=cache_test"])
    footmr = compose(config_name="demo", overrides=["video_name=cache_test"])
    sapiens = compose(
        config_name="demo",
        overrides=[
            "video_name=cache_test",
            "variant=footmr_sapiens",
            "pose_cache_name=kp2d_coco23_sapiens",
            "use_sapiens=True",
        ],
    )
    assert len({baseline.paths.vitpose, footmr.paths.vitpose, sapiens.paths.vitpose}) == 3
    assert len({baseline.output_dir, footmr.output_dir, sapiens.output_dir}) == 3


def test_checkpoints_and_forward():
    footmr = compose(config_name="demo", overrides=["video_name=synthetic"])
    model = hydra.utils.instantiate(footmr.model, _recursive_=False).eval()
    # Exercise the compatibility path used by gvhmr-web-tool's external worker.
    model.load_pretrained_model(
        "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
        strict=False,
    )
    assert sum(p.numel() for p in model.parameters()) == 44_685_900
    assert sum(p.numel() for p in model.pipeline.foot_motion_refiner.parameters()) == 3_636_844

    with torch.no_grad():
        output = model.pipeline.forward(
            synthetic_batch(), train=False, postproc=False, static_cam=True
        )
    before = output["body_pose_before_foot_refine"]
    after = output["pred_smpl_params_incam"]["body_pose"]
    non_ankle = torch.ones(63, dtype=torch.bool)
    non_ankle[18:24] = False
    assert torch.equal(before[..., non_ankle], after[..., non_ankle])
    assert not torch.equal(before[..., 18:24], after[..., 18:24])
    for params in (
        output["pred_smpl_params_incam"],
        output["pred_smpl_params_global"],
    ):
        assert all(torch.isfinite(value).all() for value in params.values())
    del model, output
    gc.collect()

    baseline = compose(config_name="demo_gvhmr", overrides=["video_name=synthetic"])
    model = hydra.utils.instantiate(baseline.model, _recursive_=False).eval()
    state = torch.load(baseline.ckpt_path, "cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    assert not model.pipeline.use_foot_refiner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    register_store_gvhmr()
    failures = []

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        run("17/23-joint network parameterization", test_joint_parameterization, failures)
        run("FootMR residual-refiner contract", test_refiner_shape, failures)
        run("backend-specific cache routing", test_cache_routing, failures)
        if not args.skip_assets:
            run("strict checkpoints + synthetic FootMR forward", test_checkpoints_and_forward, failures)

    if failures:
        print(f"FAILED: {failures}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
