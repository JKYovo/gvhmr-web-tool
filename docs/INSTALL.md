# Install

## Environment

```bash
git clone https://github.com/zju3dv/GVHMR
cd GVHMR

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
# to install gvhmr in other repo as editable, try adding "python.analysis.extraPaths": ["path/to/your/package"] to settings.json
```

### FootMR inference assets

FootMR inference needs its combined checkpoint and the COCO-wholebody
ViTPose checkpoint. The downloader verifies their SHA256 hashes and stores
them under `inputs/footmr_assets`, deliberately avoiding
`inputs/checkpoints` because that path may be shared with another worktree.

```bash
conda activate gvhmr
python tools/demo/download_footmr_assets.py
```

Expected files:

```text
inputs/footmr_assets/
├── footmr_checkpoint.ckpt
└── vitpose-h-wholebody.pth
```

### Optional: Sapiens foot keypoints

Sapiens is slower and substantially more memory intensive than ViTPose. It
is not required for the default FootMR path.

```bash
git submodule update --init third-party/sapiens
cd third-party/sapiens/engine && pip install -e . -v --no-build-isolation
cd ../cv && pip install -e . -v --no-build-isolation && pip install -r requirements/optional.txt
cd ../pretrain && pip install -e . -v --no-build-isolation
cd ../pose && pip install -e . -v --no-build-isolation
cd ../../..
```

Download `sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth` from
<https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_host/pose/checkpoints/sapiens_2b>
and place it in `inputs/footmr_assets/`. Then run the demo with
`--use_sapiens`.

### Using gvhmr-web-tool

The sibling Web tool's `start_web_source.sh` discovers this worktree and sets
the external worker's `PYTHONPATH` explicitly. In that mode, this repository
does not need an additional `pip install -e .`; the Web package and algorithm
package remain isolated in separate processes.

### Runtime defaults and performance options

No additional editable install is needed when the source-mode Web launcher is
used. The selected core worktree first resamples an input to 30 FPS according
to timestamps and only then runs bbox tracking, ViTPose, HMR2 and
GVHMR/FootMR. This preserves the duration of 60 FPS sources instead of making
their output play at half speed.

For GMR-only generation, preview rendering can be skipped without changing
the saved tensor:

```bash
python tools/demo/demo.py --video VIDEO --render none --profile
```

The tested FootMR defaults are:

- ViTPose whole-body: FP16, batch size 16.
- HMR2 features: FP32, batch size 16.
- shared video decode/crop: enabled.
- transformer attention: dense.
- cross-job cache: `outputs/cache`.

Use `--pose_inference_dtype fp32` to restore the original FP32 ViTPose path.
`--inference_dtype fp16` enables FP16 for both ViTPose and HMR2, but is not a
quality-preserving default because contact post-processing can amplify the
HMR2 feature difference. Use `--cache_root none` to disable cross-job caches.
`--profile` writes CUDA-synchronized stage timings to `performance.json` in
the result directory.

### Optional: DPVO (not recommended if you want fast inference speed)
```bash
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
pip install -e .
```

## Inputs & Outputs

```bash
mkdir inputs
mkdir outputs
```

**Weights**

```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

**Data**

We provide preprocessed data for training and evaluation.
Note that we do not intend to distribute the original datasets, and you need to download them (annotation, videos, etc.) from the original websites.
*We're unable to provide the original data due to the license restrictions.*
By downloading the preprocessed data, you agree to the original dataset's terms of use and use the data for research purposes only.

You can download them from [Google-Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link). Please place them in the "inputs" folder and execute the following commands:

```bash
cd inputs
# Train
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# Test
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# The folder structure should be like this:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```
