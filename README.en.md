# GVHMR Web Tool

[简体中文 README](README.md)

This repository packages the single-person motion recovery pipeline from [GVHMR](https://github.com/zju3dv/GVHMR) as a deployable local Web tool. Source mode includes GVHMR-Enhanced directly: FootMR ankle refinement is the default, with Contact-aware Global Optimizer V1.1 available as the automatic flat-ground constraint.

![GVHMR Web interface](docs/images/gvhmr-web.png)

## Features

- Single and batch upload for `mp4 / mov / avi / mkv / webm`
- Static-camera mode and optional focal length `f_mm`
- FootMR COCO23 ankle residual refinement with isolated preprocessing caches
- Optional Contact Global V1.1 flat-ground constraint using continuous toe/heel contacts and a sequence-wide root XYZ solve
- Human3R scene constraints remain disabled in the Web UI
- SQLite-backed job history, filtering, cancellation, and retry
- On-demand previews whose failures do not invalidate motion results
- In-page preview playback and separate PT, camera-view, global-view, and ZIP downloads
- Automatic upload cleanup with task inputs and outputs kept together
- Docker-first deployment plus a source-mode workflow for development

## Workflow

1. Upload video and submit GVHMR inference.
2. Follow progress in the job console and retry failed jobs when appropriate.
3. Download `hmr4d_results.pt`, or generate a preview to inspect the recovered motion.

## Quick Start

The supported runtime is Linux x86_64 with an NVIDIA GPU, a working driver, Docker, and NVIDIA Container Toolkit.

```bash
bash doctor.sh
bash start_web.sh
```

Open:

```text
http://127.0.0.1:7860/
```

Check status or stop the service with:

```bash
bash status.sh
bash stop_web.sh
```

The first launch builds the Docker image and downloads model assets, for a total transfer of roughly `16GB to 17GB`. See the [quick-start guide](docs/QUICKSTART.md) and [deployment guide](docs/DEPLOYMENT.md) for details.

Run the integrated enhanced source backend with:

```bash
conda activate gvhmr
python tools/demo/download_footmr_assets.py
bash start_web_source.sh
```

Source mode uses this repository as its core by default. Set `GVHMR_CORE_ROOT` only to select another worktree. The Docker path currently retains the original GVHMR baseline.

## Outputs

Each task is stored under:

```text
runtime/jobs/<video-name>_<short-job-id>/
```

Core artifacts:

- `hmr4d_results.pt`: published result; Global V1.1 when enabled and accepted by its guardrails, otherwise raw FootMR
- `hmr4d_results_raw.pt`: preserved raw FootMR result when the constraint is enabled
- `ground_constraint_global_v1_1/contact_global_root_hmr4d_results.pt`: accepted V1.1 candidate
- `ground_constraint_global_v1_1/metrics.json`: contacts, corrections, guardrails, and final decision
- `job.json`: job summary
- `artifacts.zip`: bundle of currently available outputs

The automatic constraint runs once from the raw FootMR tensor and only changes `smpl_params_global.transl`; the former local-Y postprocessor is no longer chained into new tasks. A V1.1 failure or guardrail rejection falls back directly to `hmr4d_results_raw.pt`. The UI/API value remains `flat_y` for compatibility, but now denotes Global V1.1.

## SONIC integration without Kimodo

This repository now includes the SMPL-X22-to-SONIC conversion and local ZMQ
playback adapter. Both run in the `gvhmr` Conda environment and no longer
require a Kimodo checkout, Kimodo virtual environment, or PEFT. For a
successful job, click **Send to SONIC** in the detail actions. It generates or
reuses the 50 FPS reference from the final `hmr4d_results.pt` and streams it in
the background without replacing the motion result. While streaming, **Pause
SONIC** stops the live reference; the SONIC policy then blends back to its
built-in idle/default reference instead of holding the final motion frame. The
CLI remains available:

```bash
conda run -n gvhmr python tools/sonic/convert_gvhmr.py \
  runtime/jobs/<job>/hmr4d_results.pt \
  runtime/jobs/<job>/sonic_reference.npz \
  --metadata runtime/jobs/<job>/sonic_conversion.json

conda run -n gvhmr python tools/sonic/play_reference.py \
  runtime/jobs/<job>/sonic_reference.npz
```

Conversion is read-only with respect to the Web result. The conversion CLI
does not connect to SONIC, while the Web button does. The 50 FPS
`term1_local`, `root_quat`, `wrist`, and protocol path
were verified element-for-element against the previous Kimodo outputs for
jntm, lly, cxk, qhy, and ydd, with maximum array difference zero. This proves
input compatibility only; it does not address policy tracking or robot joint
limits. The current ZMQ PUB protocol has no acknowledgement, so “streaming
complete” means the Web publisher finished sending—not that SONIC confirmed
receipt. Start SONIC/MuJoCo before clicking the button.

On-demand previews:

- `1_incam.mp4`
- `2_global.mp4`
- `*_3_incam_global_horiz.mp4`

`submitted_input.*`, `0_input_video.mp4`, and `_gvhmr_work/` are task inputs or working files rather than stable interchange formats.

## Documentation

- [Quick start](docs/QUICKSTART.md)
- [Docker and LAN deployment](docs/DEPLOYMENT.md)
- [Source development environment](docs/INSTALL.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Current Scope

- The service processes one primary person track and does not manage multiple identities.
- GVHMR inference requires CUDA; CPU inference is not supported.
- The default Web service is standalone and does not depend on GMR Web.
- `runtime/`, model checkpoints, and Docker images are intentionally excluded from Git.

## Upstream And Citation

This tool is based on the original GVHMR project:

- Project page: https://zju3dv.github.io/gvhmr
- Paper: https://arxiv.org/abs/2409.06662
- Upstream repository: https://github.com/zju3dv/GVHMR

Please cite the original paper when using GVHMR research results:

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

FootMR is used by the default enhanced source backend:

```bibtex
@InProceedings{wehrbein26footmr,
  author    = {Wehrbein, Tom and Rosenhahn, Bodo},
  title     = {Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```
