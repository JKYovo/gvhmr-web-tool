# FootMR v2: GVHMR x WHAM offline experiment

This directory deliberately does not modify the FootMR-v1 inference pipeline.
It contains two entry points:

```bash
/home/user-kevien/miniforge3/envs/wham/bin/python tools/bench/wham_v2/run_wham.py \
  --video /path/to/climb.mp4 \
  --output-dir /path/to/experiment/wham

/home/user-kevien/miniforge3/envs/gvhmr/bin/python tools/bench/wham_v2/analyze.py \
  --gvhmr-result /path/to/hmr4d_results.pt \
  --wham-result /path/to/experiment/wham/wham_w0_w1_w2.pt \
  --video /path/to/climb.mp4 \
  --output-dir /path/to/experiment
```

`run_wham.py` requires the official WHAM repository at commit
`2b54f7797391c94876848b905ed875b154c4a295`. It records three trajectories:

- W0: trajectory decoder output;
- W1: deterministic contact-based root-velocity reset;
- W2: learned trajectory refiner output.

`analyze.py` preserves GVHMR/FootMR pose, shape, root orientation, and root X/Z.
The C-delta and C-rootY results are diagnostic transfers only, not a claim that
WHAM's learned refiner accepts GVHMR features.
