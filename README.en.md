# GVHMR Web Tool

[简体中文 README](README.md)

This repository packages the single-person motion recovery pipeline from [GVHMR](https://github.com/zju3dv/GVHMR) as a deployable local Web tool. Users can upload one video or a batch, inspect persistent jobs, download `hmr4d_results.pt`, and generate camera/world-view previews when needed.

![GVHMR Web interface](docs/images/gvhmr-web.png)

## Features

- Single and batch upload for `mp4 / mov / avi / mkv / webm`
- Static-camera mode and optional focal length `f_mm`
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

## Outputs

Each task is stored under:

```text
runtime/jobs/<video-name>_<short-job-id>/
```

Core artifacts:

- `hmr4d_results.pt`: GVHMR motion result
- `job.json`: job summary
- `artifacts.zip`: bundle of currently available outputs

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
