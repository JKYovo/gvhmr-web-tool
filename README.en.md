# GVHMR Web Tool

[简体中文 README](README.md)

For AI-assisted customer deployment, require the deployment agent to read [README_AI_DEPLOY.md](README_AI_DEPLOY.md) first. Human3R and some included components are subject to non-commercial license terms and therefore are not distributed, installed, or enabled in commercial customer deployments.

This repository packages the single-person motion recovery pipeline from [GVHMR](https://github.com/zju3dv/GVHMR) as a deployable local Web tool. Source mode includes GVHMR-Enhanced directly: FootMR ankle refinement is the default, with Contact-aware Global Optimizer V1.1 available as the automatic flat-ground constraint.

![GVHMR Web interface](docs/images/gvhmr-web.png)

## Features

- Single and batch upload for `mp4 / mov / avi / mkv / webm`
- Static-camera mode and optional focal length `f_mm`
- FootMR COCO23 ankle residual refinement with isolated preprocessing caches
- Four ground modes: disabled, Contact Global V1.1, standing-calibrated gravity plus Global V1.1, and Human3R scene gravity plus Global V1.1
- Human3R compatibility remains a local private add-on; its source, submodules, weights, and compiled artifacts are not distributed in the public customer repository
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

New jobs use a compact, stable layout:

```text
job/
├── 0_input_video.mp4
├── hmr4d_results.pt
├── job.json
├── artifacts.zip
├── preview/
├── exports/
├── diagnostics/
└── .work/
```

`hmr4d_results.pt` is always the published result. Raw tensors, candidates, metrics, and inspection images live under `diagnostics/`; Human3R per-frame reconstruction is temporary and is deleted after ground extraction by default. The ZIP and Web download list expose only final results and concise diagnostics. `gravity_flat` may fall back once to `flat_y` when no reliable standing segment exists and records the reason. `human3r` does not silently fall back. The compatibility API value `flat_y` denotes Global V1.1.

## SONIC integration without Kimodo

This repository now includes the SMPL-X22-to-SONIC conversion and local ZMQ
playback adapter. Both run in the `gvhmr` Conda environment and no longer
require a Kimodo checkout, Kimodo virtual environment, or PEFT. For a
successful job, click **Send to SONIC** in the detail actions. It generates or
reuses the 50 FPS reference from the final `hmr4d_results.pt` and streams it in
the background without replacing the motion result. While streaming, **Pause
SONIC** stops the live reference; the SONIC policy then blends back to its
built-in idle/default reference instead of holding the final motion frame. The
SONIC speed slider covers `0.25x` through `1.00x` in `0.05x` steps. It changes
only the next playback timeline, keeps the output at 50 FPS, and does not send
anything until **Send to SONIC** is clicked. Each speed uses an isolated cache. The
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

- `preview/incam.mp4`
- `preview/global.mp4`
- `preview/comparison.mp4`

`submitted_input.*` and `preprocess/` are temporary and removed after success. `0_input_video.mp4` is the normalized durable input, but is not a stable interchange format.

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

The optional scene-gravity mode uses Human3R:

```bibtex
@article{chen2025human3r,
  title={Human3R: Everyone Everywhere All at Once},
  author={Chen, Yue and Chen, Xingyu and Xue, Yuxuan and Chen, Anpei and Xiu, Yuliang and Gerard, Pons-Moll},
  journal={arXiv preprint arXiv:2510.06219},
  year={2025}
}
```

Human3R and some included components are under non-commercial terms, including CC BY-NC-SA 4.0 and the NAVER Non-Commercial License, so Human3R is not deployed in the public commercial customer package. It must remain disabled unless the customer separately obtains explicit commercial permission and completes legal review. This integration grants no commercial-use rights.
