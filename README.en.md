# GVHMR Web Tool

[简体中文 README](README.md)

This repository turns [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) into a local Web tool for:

- uploading a video
- running GVHMR on a single main person track
- exporting structured motion data as `gvhmr_data.npz` and `gvhmr_meta.json`
- optionally generating preview videos

It keeps the original GVHMR inference pipeline, but adds:

- a reusable Python API in `hmr4d/api/video_to_data.py`
- a local Web UI
- a Docker-first deployment path
- job persistence, batch submission, and downloadable artifacts

## What This Fork Is For

Use this repository when you want a teammate to run GVHMR as a tool instead of as a research demo.

Typical workflow:

1. Start the Web service.
2. Upload one or more videos.
3. Download `NPZ + JSON` results.
4. Optionally download preview videos.

## One-Click Deploy

This is the recommended way to run the project on another machine.

### Supported Runtime

- Linux `x86_64`
- NVIDIA GPU
- NVIDIA driver installed
- Docker installed
- NVIDIA Container Toolkit installed

### Environment Doctor

You can run a standalone environment check first:

```bash
bash doctor.sh
```

On Ubuntu / Debian, you can also ask the scripts to attempt automatic repair:

```bash
bash doctor.sh --fix
```

Current `--fix` coverage:

- install Docker
- start `docker.service`
- install and configure NVIDIA Container Toolkit

For safety, the scripts do not attempt to install an NVIDIA driver. If `nvidia-smi` is unavailable, the doctor exits early and asks the user to install the driver first.

### Start

From the repository root:

```bash
bash start_web.sh
```

To attempt environment repair before launch:

```bash
bash start_web.sh --fix
```

The Web UI will be served at:

```text
http://127.0.0.1:7860/ui
```

### LAN Access

To expose the service to other devices on the same network:

```bash
bash start_web_lan.sh
```

This also supports:

```bash
bash start_web_lan.sh --fix
```

### Status And Stop

```bash
bash status.sh
bash stop_web.sh
```

### First Launch

On the first launch, the scripts will:

- check the host environment
- build the Docker image
- check CUDA visibility inside the container
- create `runtime/checkpoints`, `runtime/jobs`, `runtime/batches`, and `runtime/db`
- download the required model assets into `runtime/checkpoints`

Subsequent launches reuse the same image and checkpoints.

## Local Source Mode

If you are developing the project locally instead of using Docker:

```bash
pip install -r deploy/env/requirements-ui.txt
python tools/app/run_ui.py
```

The source-mode UI is mainly for development. For teammate-facing deployment, prefer `start_web.sh`.

## Outputs

Each job writes results under `runtime/jobs`.

Core outputs:

- `gvhmr_data.npz`
- `gvhmr_meta.json`
- `hmr4d_results.pt`

Optional outputs:

- `1_incam.mp4`
- `2_global.mp4`
- `*_3_incam_global_horiz.mp4`

## Data Format

`gvhmr_data.npz` contains flattened arrays such as:

- `smpl_global_body_pose`
- `smpl_global_global_orient`
- `smpl_global_transl`
- `smpl_global_betas`
- `smpl_incam_body_pose`
- `smpl_incam_global_orient`
- `smpl_incam_transl`
- `smpl_incam_betas`
- `camera_K_fullimg`

`gvhmr_meta.json` contains job metadata such as:

- source video path
- source resolution and fps
- processed frame count
- processed fps
- `static_cam`
- `f_mm`
- output directory
- `person_mode = "single_primary_track"`

## Repository Layout

- `deploy/docker/`
  Dockerfile and Compose configuration.
- `deploy/env/`
  Runtime and development environment dependency files.
- `deploy/scripts/`
  Actual deployment script implementations.
- root `doctor.sh` / `start_web.sh` / `start_web_lan.sh` / `status.sh` / `stop_web.sh`
  Thin convenience wrappers for end users.
- `hmr4d/api/`
  Reusable API layer for `video -> data`.
- `hmr4d/service/`
  Job manager, persistence, service entrypoint, and Web UI.
- `tools/app/run_ui.py`
  Source-mode launcher for local development.
- `start_web.sh`
  One-click local Docker start.
- `start_web_lan.sh`
  One-click LAN-facing Docker start.
- `doctor.sh`
  Environment check and optional auto-fix entrypoint.

## Important Notes

- This project currently targets single-person main-track processing.
- It does not support CPU inference.
- `inputs/`, `outputs/`, and `runtime/` are not tracked in git.
- Docker images and model checkpoints are intentionally not committed to this repository.

## Development

For development environment setup:

```bash
conda env create -f deploy/env/environment-dev.yml
conda activate gvhmr-dev
```

## Upstream Project

This work is based on the original GVHMR project:

- Project page: https://zju3dv.github.io/gvhmr
- Paper: https://arxiv.org/abs/2409.06662
- Upstream repo: https://github.com/zju3dv/GVHMR

## Citation

If you use GVHMR itself in research, please cite the original paper:

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

## Acknowledgement

Thanks to the authors of:

- [WHAM](https://github.com/yohanshin/WHAM)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch)
