# GVHMR-Enhanced: GVHMR with FootMR ankle refinement

This branch integrates the inference path from
[FootMR](https://github.com/twehrbein/FootMR) into GVHMR. FootMR is the
default demo model; the original GVHMR remains available as a baseline.
Implementation and validation history is maintained in
[docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md).

## FootMR quick start

Download the public FootMR and whole-body ViTPose checkpoints without writing
through a shared `inputs/checkpoints` symlink:

```shell
conda activate gvhmr
python tools/demo/download_footmr_assets.py
```

Run enhanced inference, or explicitly select the original baseline:

```shell
python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s
python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s --model gvhmr
```

FootMR uses ViTPose whole-body by default. `--use_sapiens` enables the
optional, slower Sapiens backend after its dependencies and checkpoint have
been installed. Use `--no_postproc` when global contact/IK post-processing
suppresses fine-grained foot motion. Results are isolated under
`outputs/demo/<video>/<gvhmr|footmr_vitpose|footmr_sapiens>/` while expensive
shared preprocessing stays in `outputs/demo/<video>/preprocess/`.

When launched through the sibling `gvhmr-web-tool` source-mode server, no
editable installation of this worktree is required. The Web external worker
sets `PYTHONPATH` to this core directory and imports it in an isolated process.

### Inference speed and video timing

The demo normalizes the source video to 30 FPS by timestamps **before**
tracking or pose inference. A 60 FPS input is therefore resampled instead of
merely being relabeled as 30 FPS, so its real-time duration and motion speed
are preserved.

When only the tensor for GMR is needed, skip preview rendering:

```shell
python tools/demo/demo.py --video VIDEO --render none --profile
```

FootMR defaults to FP16 for ViTPose and FP32 for HMR2 because this was the
best tested speed/quality tradeoff. Use `--pose_inference_dtype fp32` for the
strict-quality fallback. `--inference_dtype fp16` also changes HMR2 to FP16;
it is faster but is not the default because small feature differences can be
amplified by contact post-processing. Cross-job bbox, pose, HMR2 feature and
normalized-video caches are stored under `outputs/cache`; pass
`--cache_root none` to disable them. With `--profile`, synchronized stage
timings are written to the result directory as `performance.json`.

---

# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates  
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)  
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

## News 🔥

- [2025-03-08] By default not using DPVO. We implemented a SimpleVO, which is more efficient and compatible with GVHMR.
- [2025-03-08] We added a new option `f_mm` to specify the focal length of the fullframe camera in mm.

## Setup

Please see [installation](docs/INSTALL.md) for details.

## Quick Start

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo for GVHMR](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing)

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for GVHMR](https://huggingface.co/spaces/LittleFrog/GVHMR)

### Demo
Demo entries are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static. FootMR is now the default; pass `--model gvhmr` for the original behavior.
We also provide a script `demo_folder.py` to inference a entire folder.
```shell
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### Reproduce
1. **Test**:
To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:
    ```shell
    python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    ```
    To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

2. **Train**:
To train the model, use the following command:
    ```shell
    # The gvhmr_siga24_release.ckpt is trained with 2x4090 for 420 epochs, note that different GPU settings may lead to different results.
    python tools/train.py exp=gvhmr/mixed/mixed
    ```
    During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@InProceedings{wehrbein26footmr,
  author    = {Wehrbein, Tom and Rosenhahn, Bodo},
  title     = {Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2026},
}
```

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

The optional offline CoTracker3 contact-foot experiment uses the upstream
CoTracker code and checkpoint. Cite it when that experiment is used:

```bibtex
@inproceedings{karaev24cotracker3,
  title     = {CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos},
  author    = {Nikita Karaev and Iurii Makarov and Jianyuan Wang and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  booktitle = {Proc. {arXiv:2410.11831}},
  year      = {2024}
}
```

# Acknowledgement

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.

The optional CoTracker3 P5 experiment uses
[CoTracker](https://github.com/facebookresearch/co-tracker) at commit
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`. The majority of upstream
CoTracker is licensed under CC BY-NC 4.0; this optional component and its
checkpoint may only be used under the applicable upstream non-commercial
terms and are not relicensed by this repository.
