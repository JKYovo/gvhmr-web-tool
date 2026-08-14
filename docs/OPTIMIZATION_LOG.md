# GVHMR-Enhanced 优化记录

本文档是 `feature/gvhmr-opt` 分支的持续优化日志。每完成一项算法、推理、性能或工程优化，都必须在此追加一条记录；不能只记录代码变更而省略验证结果。

## 记录规范

每项优化至少记录以下内容：

1. 编号、日期、状态、依据的上游版本或实验基线。
2. 优化目标和明确不在范围内的内容。
3. 关键实现、接口变化、兼容方式和资产位置。
4. 可复现的验证命令、定量结果以及未通过或未执行的检查。
5. 已知风险、回退方式和下一步验收条件。

状态统一使用：`实现中`、`CPU 已验证`、`GPU 已验证`、`实验完成（未采用）`、`已回退`。只有完整视频推理及产物检查通过后，才能标记为 `GPU 已验证`；未达到集成门槛的离线实验必须保留失败结果并标记为 `实验完成（未采用）`。

---

## P0：以 FootMR 构建 GVHMR-Enhanced 推理基线

### 基本信息

- 日期：2026-08-07
- 分支：`feature/gvhmr-opt`
- 状态：`CPU 已验证`，GPU 视频验收待完成
- 上游依据：FootMR commit `9c5b4123b344d74926822c20f182b2af4494dc41`
- 范围：视频推理、命令行 demo、批量 demo、Web source-mode 兼容
- 不包含：FootMR 训练、MOOF/MOYO/RICH 评测栈、论文指标复现

### 优化目标

在保留当前 GVHMR 和已有 SimpleVO 路径的前提下，将 FootMR 作为增强版默认推理模型。FootMR 不重新预测整个人体，而是根据原始人体结果和脚部观测预测左右踝全局旋转残差，降低集成风险。

论文报告的 MOYO 踝关节角误差改进仅作为上游结果引用。本地未运行 MOYO 评测，因此不能将“约 30%”作为本次验收结果。

### 关键实现

- 将 GVHMR 2D 关节输入参数化：原始 GVHMR 使用 COCO17，FootMR 使用 COCO23，两套 checkpoint 均可严格加载。
- 移植 FootMR `FootEncoderRoPE` 和 COCO23 SMPL-X 回归支持。
- FootMR refiner 使用 8 个踝/足 2D 点、左右膝与初始踝全局旋转以及 bounding-box camera 信息。
- refiner 只替换 `body_pose` 中左右踝槽位 `18:24`；其他姿态槽位保持不变。
- refined pose 同时进入 incam/global 结果；默认保留官方全局后处理，`--no_postproc` 可关闭 contact/IK 后处理。
- standalone demo 默认 `--model footmr`，可用 `--model gvhmr` 回退到原始基线。
- 默认使用 ViTPose whole-body；`--use_sapiens` 使用懒加载，依赖或权重缺失时给出明确错误。
- 增加 `--pose_batch_size`，用于控制 12 GB GPU 上的 ViTPose 峰值显存。
- bbox、图像特征和 VO 结果共享；COCO17、FootMR-ViTPose、FootMR-Sapiens 的姿态缓存分别隔离。
- 输出保持 `smpl_params_incam`、`smpl_params_global`、`K_fullimg`、`net_outputs` schema 不变。
- `gvhmr-web-tool` source mode 通过 `PYTHONPATH` 直接加载本仓库，不要求执行 `pip install -e .`。

### 资产与版本

FootMR 资产保存在 `inputs/footmr_assets/`，避免写入当前可能指向其他仓库的 `inputs/checkpoints` 符号链接。

| 资产 | SHA256 |
| --- | --- |
| `footmr_checkpoint.ckpt` | `2d31d8b5f7079c86dc176472909d4b3c14db801f0bcd9f99571637d0a860407a` |
| `vitpose-h-wholebody.pth` | `dbed01fd5bb221610bf26434ec63426025f76eaca46f6177db71c9771a43316c` |

Sapiens 子模块固定为 commit `08dce797f7b40f5b41388f518cac85535c3f5d13`。Sapiens 2B 推理不是 P0 的 GPU 验收阻塞项。

### 接口与缓存

```text
python tools/demo/demo.py --video VIDEO                         # FootMR + ViTPose
python tools/demo/demo.py --video VIDEO --model gvhmr           # GVHMR baseline
python tools/demo/demo.py --video VIDEO --use_sapiens           # 可选 Sapiens
python tools/demo/demo.py --video VIDEO --no_postproc           # 关闭全局后处理
python tools/demo/demo.py --video VIDEO --pose_batch_size 4     # 降低峰值显存
```

standalone 输出目录：

```text
outputs/demo/<video>/
├── preprocess/
├── gvhmr/
├── footmr_vitpose/
└── footmr_sapiens/
```

姿态缓存分别使用 `kp2d_coco17.pt`、`kp2d_coco23_vitpose.pt` 和 `kp2d_coco23_sapiens.pt`。加载缓存时还会检查关节数，不兼容的旧缓存会被忽略并重算。

### 已完成验证

所有 Python 检查均在 `gvhmr` Conda 环境中执行。

| 检查 | 结果 |
| --- | --- |
| `python -m compileall -q hmr4d tools/demo tools/bench/test_footmr_integration.py` | 通过 |
| `python tools/bench/test_footmr_integration.py` | 通过 |
| COCO17/23 主干 shape 与 finite 检查 | 通过 |
| FootMR checkpoint strict load | `missing=[]`，`unexpected=[]` |
| FootMR 总参数量 | `44,685,900` |
| FootMR refiner 参数量 | `3,636,844` |
| 2 帧 CPU 合成输入 incam/global 前向 | shape 正确且均为有限值 |
| 非踝 pose 最大变化 | `0.0` |
| ViTPose whole-body CPU strict load 与前向 | 输出 `(1, 133, 64, 48)`，均为有限值 |
| 两个资产 SHA256 | 与上表一致 |
| Sapiens 缺失权重错误提示 | 通过 |
| Web external worker probe | 实际导入 `gvhmr-core-opt/hmr4d/__init__.py` |
| `git diff --check HEAD` | 通过 |

### 尚未完成的 GPU 验收

本次检查时 RTX 3060 12 GB 仅剩约 2.5 GB 空闲显存，主要被工作区外的两个长期 Python 进程占用。按照“不终止其他 GPU 进程”的约束，以下检查尚未执行：

- 官方 `stepdance.mp4` 的 FootMR-ViTPose 完整推理和渲染。
- `tennis.mp4` 的 GVHMR baseline 回归。
- 缓存隔离、输出帧数和渲染视频的端到端检查。
- baseline/enhanced 并排视频和左右踝变化的人工确认。

在上述项目通过前，本项不能标记为 `GPU 已验证`。

### 回退与兼容

- standalone 使用 `--model gvhmr` 回退到原始 GVHMR。
- FootMR、GVHMR 和 Sapiens 输出目录相互独立，回退不会覆盖其他模型结果。
- Web source mode 仍沿用原输出 tensor schema；旧 worker 传入 GVHMR checkpoint 路径时，FootMR pipeline 会解析到独立的 FootMR checkpoint。
- 本次未修改 `gvhmr-web-tool`。

### 主要涉及文件

- `hmr4d/configs/demo.yaml`
- `hmr4d/configs/demo_gvhmr.yaml`
- `hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py`
- `hmr4d/model/gvhmr/gvhmr_pl_demo.py`
- `hmr4d/network/footmr/foot_transformer.py`
- `hmr4d/utils/preproc/vitpose.py`
- `hmr4d/utils/preproc/sapiens.py`
- `tools/demo/demo.py`
- `tools/demo/demo_folder.py`
- `tools/demo/download_footmr_assets.py`
- `tools/bench/test_footmr_integration.py`

---

## P1：FootMR v2 的 GVHMR × WHAM 轨迹修正离线实验

### 基本信息

- 日期：2026-08-07
- 分支：`feature/gvhmr-opt`
- 状态：`实验完成（未采用）`
- WHAM 依据：commit `2b54f7797391c94876848b905ed875b154c4a295`
- WHAM checkpoint：`wham_vit_bedlam_w_3dpw.pth.tar`
- 输入：Web job `climb_6466b84f`，1060 帧，30 FPS，静态相机
- 范围：离线轨迹诊断、候选 root-Y 修正和 2×2 对比视频
- 不包含：Web 集成、线上推理接口、WHAM 与 GVHMR 的跨网络特征级联合

### 优化目标

验证 WHAM 的 Trajectory Decoder、接触速度重置和 Learned Trajectory Refiner 能否修正当前 GVHMR + FootMR v1 在爬箱动作结束后无法回到原地面高度的问题。

### 实验方案

- A：保持当前 GVHMR + FootMR v1 不变。
- B：根据 GVHMR 足部静态置信度做确定性 contact-Y 速度抵消。
- C-delta：把 WHAM `W2-W0` 的 root-Y 修正量迁移到 GVHMR。
- C-rootY：用 WHAM W2 相对起始窗口的 root-Y 曲线替换 GVHMR 的相对 root Y。
- W0/W1/W2：分别检查 WHAM Trajectory Decoder、官方 `reset_root_velocity` 和 Learned Refiner 自身输出。

C 方案只替换 root Y，FootMR 姿态、betas、global orientation 和 root X/Z 均保持不变。它们是诊断性轨迹迁移，不是可部署的跨网络 tensor 接口。

### 关键结果

| 方案 | 回地高度残差 | 箱顶相对高度 | 接触速度 P95 | 判定 |
| --- | ---: | ---: | ---: | --- |
| A：当前 GVHMR + FootMR v1 | 17.20 cm | 48.30 cm | 3.55 mm/frame | 未通过 |
| B：确定性 contact-Y | 27.31 cm | 50.28 cm | 3.95 mm/frame | 未通过，且比 A 更差 |
| C-delta：WHAM W2-W0 迁移 | 29.24 cm | 56.20 cm | 4.48 mm/frame | 保护条件失败 |
| C-rootY：WHAM W2 root-Y | 32.17 cm | 31.49 cm | 5.55 mm/frame | 保护条件失败 |
| WHAM W0 | 16.74 cm | 19.72 cm | 14.52 mm/frame | 未通过 |
| WHAM W1 | 10.04 cm | 5.80 cm | 9.43 mm/frame | 压平箱顶高度，保护条件失败 |
| WHAM W2 | 28.52 cm | 28.08 cm | 11.79 mm/frame | 保护条件失败 |

### 验证与产物

- WHAM 资产 SHA256、运行版本和配置已写入实验 manifest。
- `tools/bench/wham_v2/test_common.py`：5 项测试通过。
- 2×2 对比视频：1280×720、30 FPS、1060 帧、35.33 秒。
- 完整指标、曲线、候选 tensor 和报告保存在 `/home/user-kevien/gvhmr_pkg/experiments/footmr_v2/climb_6466b84f/`。

主要产物：

- `comparison_2x2.mp4`
- `metrics.json`
- `curves.csv`
- `trajectory_contact_curves.png`
- `report.md`
- `wham/wham_w0_w1_w2.pt`
- `wham/wham_manifest.json`

### 结论与处置

- WHAM v2 离线实验和对比视频已经生成。
- 所有候选方案均未达到预设的 5 cm 回地高度门槛，不能作为增强方案接入 Web。
- B 说明简单接触期 Y 速度抵消会累计误差；W1 虽降低回地残差，但明显压平箱顶高度；W2 在该视频上比 W0 更差。
- 按预设保护条件停止，不继续运行 `climb2`，也不修改当前 GVHMR + FootMR v1 默认路径。
- 下一方向需要单独评估 absolute scene constraint；当前结果不能包装为可部署的 WHAM v2。

### 主要涉及文件

- `tools/bench/wham_v2/common.py`
- `tools/bench/wham_v2/run_wham.py`
- `tools/bench/wham_v2/analyze.py`
- `tools/bench/wham_v2/test_common.py`
- `tools/bench/wham_v2/README.md`

---

## P2-Y：Human3R 场景几何约束的 global root-Y 离线实验

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`GPU 已验证`
- Human3R 依据：commit `402f2b2c7f20514e99cb42e4126c46b4ff75593f`
- DINOv2 依据：commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- 输入：Web job `climb_6466b84f`，1280×720，30 FPS，1060 帧
- 范围：独立 Human3R 环境、完整视频场景重建、地面/箱顶检测、只改 global root Y 的离线候选和并排视频
- 不包含：Web 集成、自动通用动作分段、X/Z 修正、多视频泛化验证、Human3R 训练或论文指标复现

### 优化目标

验证绝对场景几何能否补足 GVHMR 单独依靠人体运动时缺少的垂直参照。Human3R 从视频联合重建相机、人体与场景；本实验只使用其场景深度恢复地面和箱顶的高度差，不用 Human3R 的人体结果替换 GVHMR 姿态。

输出候选只修改 `smpl_params_global["transl"][:, 1]`。global X/Z、body pose、global orientation、betas、全部 incam 参数、`K_fullimg` 和 `net_outputs` 均保持精确不变。

### 环境、资产与关键实现

- 新建独立 Conda 环境 `/home/user-kevien/miniforge3/envs/human3r`，使用 Python 3.11、PyTorch 2.4.1 + CUDA 12.4、TorchVision 0.19.1、NumPy 1.26.4、Transformers 4.46.3、gsplat 1.5.3 和 Open3D 0.19.0；`pip check` 通过。
- Human3R 权重保存在 `inputs/human3r_assets/human3r_672S.pth`，SHA256 为 `84d2a70386473b58b90eef8f78521065ad10908bab647ee58d1196f5018fb778`。
- CUDA RoPE 已在独立环境中编译。SMPL/SMPL-X 使用只读符号链接，没有改写 WebTool 资产。
- headless runner 将 DINOv2 固定到本地子模块，不启动 Viser；完整保存每帧 depth/conf/color/camera/SMPL-X。
- 为控制 12 GB GPU 内存，每 100 帧分块，下一块重叠上一块最后一帧，并据此拼接相机轨迹。
- 场景检测使用多帧中位深度、人体 mask、固定 Open3D 随机种子 42、RANSAC 平面与中心高度直方图连通面重拟合。
- root-Y 修正用 smoothstep 在稳定平面高度间过渡。初版只按足高曲线检测到 929–940 的下箱区间，导致修正集中；最终使用四个足部静态概率，把下降过渡扩展到 878–955 帧，并设置至少 60 帧的连续性保护。
- tensor 元数据写入独立 `metrics.json`，不向 `.pt` 结果增加顶层键，保证输出 schema 不变。

### 完整 Human3R 与场景重建结果

Human3R 完成全部 1060 帧 GPU 推理：

| 项目 | 结果 |
| --- | ---: |
| 推理耗时 | 292.37 s |
| 总耗时 | 391.94 s |
| 每帧推理耗时 | 0.276 s |
| 峰值 CUDA allocated | 4.31 GB |
| depth/conf/color/camera/SMPL-X | 均为 1060 帧 |
| 深度分辨率 | 512×288 |
| 输出体积 | 约 2.6 GB |

分块拼接后的相机帧间位移中位数约 0.96 mm，分块边界最大位移约 7.4 mm。最终检测到箱顶相对地面高度 `0.698077 m`；`plane_overlay.png` 人工检查中，绿色点位于地面，橙色点位于木箱顶面。

### P2-Y 定量结果

| 指标 | GVHMR + FootMR baseline | Human3R P2-Y | 保护条件 |
| --- | ---: | ---: | ---: |
| 回地高度残差 | 17.2035 cm | 0.000006 cm | < 5 cm |
| 箱顶相对高度 | 48.2993 cm | 69.8077 cm | 目标 69.8077 cm |
| 箱顶高度误差 | 21.5084 cm | 0.000008 cm | < 5 cm |
| 接触足速度 P95 | 3.5498 mm/frame | 4.3828 mm/frame | 诊断项 |
| root 最大步长 | 3.5190 cm/frame | 4.1734 cm/frame | <= 5.2785 cm/frame |
| root 加速度 P95 | 2.3659 m/s² | 2.3338 m/s² | <= 2.9574 m/s² |

最终判定为 `pass`。1060 帧候选 tensor 均为有限值，顶层键及 global/incam 子键不变；X/Z、pose、orientation、betas、incam、`K_fullimg` 和 `net_outputs` 的精确相等检查全部通过。CPU helper 单元测试 2 项通过。

并排视频为 1280×360、30 FPS、1060 帧、35.33 秒。上箱、箱顶、下箱及回地关键帧检查未发现初版的垂直单帧跳变；箱顶动作没有被压平。渲染视频只绘制统一地面，没有绘制 Human3R 箱体网格，因此箱顶接触仍需结合 `plane_overlay.png` 和高度曲线判断。

### 产物与复现

实验产物位于：

```text
/home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/
├── human3r/                         # 完整 Human3R 每帧输出
├── scene/
│   ├── scene_planes.json
│   ├── plane_overlay.png
│   └── median_depth.npy
└── p2_y/
    ├── p2_y_hmr4d_results.pt
    ├── metrics.json
    ├── report.md
    ├── p2_y_curves.csv
    ├── p2_y_curves.png
    └── baseline_vs_p2_y.mp4
```

完整安装、命令和保护条件见 `tools/bench/human3r_p2y/README.md`。

### 结论、限制与处置

- 绝对场景高度在当前片段中能修复 GVHMR 的回地 root-Y 漂移，而且没有重现 WHAM 方案压平箱顶高度的问题。
- 当前通过的是单视频离线验收，不代表通用可部署。`pre/top/post` 稳定窗口仍来自该片段的人工标注窗口；在自动表面接触分段和多场景验证完成前，不接入 Web 默认路径。
- 单目 Human3R 的绝对尺度、箱体可见性、纹理和相机运动都会影响平面检测。本方案只改 Y，明确不能解决 X/Z 漂移。
- 回退方式是不使用 `p2_y_hmr4d_results.pt`，继续读取原始 GVHMR + FootMR 结果；Web 当前未改动，无需代码回退。

### 主要涉及文件

- `.gitmodules`
- `third-party/Human3R`
- `third-party/dinov2`
- `tools/bench/human3r_p2y/run_human3r_headless.py`
- `tools/bench/human3r_p2y/extract_scene_planes.py`
- `tools/bench/human3r_p2y/apply_p2y.py`
- `tools/bench/human3r_p2y/test_apply_p2y.py`
- `tools/bench/human3r_p2y/README.md`

---

## P2-XYZ：Human3R 同地面水平终点约束实验

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`实验完成（未采用）`
- 基线：已通过单视频验收的 Human3R P2-Y
- 输入：`climb_6466b84f`，1060 帧，30 FPS
- 范围：在 P2-Y 上增加 global root X/Z 的独立离线候选、连续性检查和 GMR 可读 `.pt`
- 不包含：Web 集成、自动稳定窗口、水平测量真值、多视频泛化验证

### 优化目标

验证 Human3R 场景是否能为 GVHMR 的水平轨迹提供绝对参照，同时保持 P2-Y 高度、全身姿态和 incam 结果不变。

### 方案筛选与实现

首先检查了 Human3R 保存的逐帧 SMPL-X root。该人体分支在约 500–800 帧把箱体或背景误检为唯一人体，投影中心与 GVHMR bbox 相差约 200 像素，因此拒绝直接迁移 Human3R root XYZ。

最终诊断方案只使用静态场景：

- 用 GVHMR incam SMPL-X 计算六个足底点并投影到原视频。
- 将足点缩放到 Human3R 512×288 相机，和 Human3R 地面平面求射线交点。
- 在动作前的地面窗口拟合 Human3R 平面切空间到 GVHMR global X/Z 的无反射刚体变换，不拟合额外尺度。
- 用动作前后同一个物理地面估计水平终点偏差；pre 锚点固定为零。
- 箱顶水平交点逐帧离散 P95 为 31.79 cm，明显大于 pre 的 5.17 cm 和 post 的 7.68 cm，因此拒绝使用箱顶水平观测。中间锚点根据原 GVHMR 水平轨迹完成度 `0.5406` 放置。
- X/Z 锚点沿用 P2-Y 的 251–412 上箱和 878–955 下箱 smoothstep 区间，避免引入新的不连续。
- 输出从 P2-Y 深拷贝，只替换 `smpl_params_global["transl"][:, (0, 2)]`；P2-Y 的 root Y 精确不变。

### 定量结果

场景估计的相对修正锚点：

| 窗口 | X 修正 | Z 修正 |
| --- | ---: | ---: |
| pre | 0.0000 m | 0.0000 m |
| top（按轨迹进度放置） | +0.0916 m | +0.0320 m |
| post | +0.1694 m | +0.0591 m |

连续性与同地面终点检查：

| 指标 | P2-Y / baseline XZ | P2-XYZ | 保护条件 |
| --- | ---: | ---: | ---: |
| root XZ 最大步长 | 4.5834 cm/frame | 4.4816 cm/frame | <= 6.8751 |
| root XZ 步长 P95 | 1.5097 cm/frame | 1.4727 cm/frame | 诊断项 |
| root XZ 加速度 P95 | 4.6636 m/s² | 4.6577 m/s² | <= 5.8295 |
| post 同地面残差 median | 17.2408 cm | 4.6429 cm | 必须改善 |
| post 同地面残差 P95 | 19.6648 cm | 12.4509 cm | 必须改善 |

候选最终水平位移由 `[-2.3879, +0.0797] m` 改为 `[-2.2185, +0.1388] m`。最大 X/Z 修正范数为 17.94 cm，小于 25 cm 保护上限。算法内部判定为 `diagnostic_pass`。

### 验证与不变量

- P2-XYZ helper 单元测试 2 项通过；P2-Y helper 2 项继续通过。
- 输出 1060 帧，所有 root XYZ 有限。
- 顶层 schema、global/incam 参数键保持不变。
- root Y 与 P2-Y 逐元素相等。
- body pose、global orientation、betas、全部 incam、`K_fullimg`、`net_outputs` 均精确不变。
- P2-Y/P2-XYZ 隔离对比视频为 1280×360、30 FPS、1060 帧、35.33 秒；关键帧未发现新增水平瞬移。
- 使用现有 `gmr` Conda 环境调用 `load_gvhmr_pred_file` 和 `get_gvhmr_data_offline_fast` 实测加载成功，得到 1060 帧、30 FPS，人体高度 1.7565 m；无需先转换为 SMPL-X NPZ。

### 结论与处置

- 同一地面前后约束表明 baseline 终点存在约 18 cm 的场景不一致，保守 X/Z 修正能降低回地水平残差，而且没有恶化轨迹连续性。
- 但当前没有水平测量真值，且箱顶水平观测不稳定；`diagnostic_pass` 只表示保护条件通过，不能证明水平坐标更准确。
- 本项暂标记为`实验完成（未采用）`，不替换 P2-Y 推荐结果、不接入 Web。用户可以把候选 `.pt` 交给 GMR 做 A/B 测试。

### 产物与文件

```text
/home/user-kevien/gvhmr_pkg/experiments/footmr_p2_xyz/climb_6466b84f/p2_xyz/
├── p2_xyz_hmr4d_results.pt
├── metrics.json
├── report.md
├── p2_xyz_curves.csv
├── p2_xyz_curves.png
└── p2_y_vs_p2_xyz.mp4
```

主要实现：

- `tools/bench/human3r_p2y/apply_p2xyz.py`
- `tools/bench/human3r_p2y/test_apply_p2xyz.py`
- `tools/bench/human3r_p2y/apply_p2y.py`（参数化对比视频标签）
- `tools/bench/human3r_p2y/README.md`

---

## P3：Human3R-only 人体动作替代实验

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`实验完成（未采用）`
- 输入：`climb_6466b84f` 的完整 Human3R 1060 帧输出
- 范围：评估完全不用 GVHMR/FootMR、直接采用 Human3R SMPL-X 人体轨迹供 GMR 使用的可行性
- 不包含：用 GVHMR 姿态填补 Human3R 缺口、长段人工插值、重新训练 Human3R

### 验证方法

Human3R 输出每帧零到多个 SMPL-X 人体候选，但当前保存格式没有持久化跨帧 `smpl_id`。为避免把背景人物当成目标，本实验使用现有 GVHMR bbox 作为只读评估参考：

- 将 bbox 顶部向下 10% 的位置作为近似头部中心。
- 投影 Human3R 每个候选的 head-centered translation。
- 选择距离 bbox 头部中心最近的候选。
- 在 Human3R 512×288 坐标下使用 50 px 的宽松有效阈值，约等于原视频 125 px。
- 要求有效覆盖至少 95%，最长缺口不超过 5 帧；否则禁止导出 GMR 文件。

这实际上给 Human3R-only 提供了来自 GVHMR bbox 的额外目标提示，因此结果不是对 Human3R 的不利设定。如果连该 oracle-assisted 选择都失败，无提示的纯 Human3R 轨迹不会更可靠。

### 结果

| 指标 | 结果 | 保护条件 |
| --- | ---: | ---: |
| 有效帧 | 592 / 1060 | 至少 1007 / 1060 |
| 有效覆盖率 | 55.85% | >= 95% |
| 无任何人体候选 | 63 帧 | 诊断项 |
| 最长连续无效段 | 173 帧 / 5.77 s | <= 5 帧 |
| 第二长连续无效段 | 116 帧 / 3.87 s | <= 5 帧 |
| 候选距离中位数 | 38.92 px | <= 50 px |
| 候选距离 P95 | 231.23 px | <= 50 px |

主要连续失效区间包括 565–737、311–426、800–858 和 237–293 帧。叠加视频显示箱顶站立阶段的 Human3R 候选集中在画面右侧背景人物附近，与目标人物头部相差约 218–220 px；这与 P2-XYZ 阶段发现的人体 root 误检一致。

### 结论与处置

- 判定为 `guardrail_failed`。
- 当前视频上只用 Human3R 不会比 GVHMR + FootMR 更好，因为人体目标轨迹不完整；场景重建成功不代表人体分支也成功。
- 直接导出必须对最长 5.77 秒的 root 和全身姿态进行插值，这已不再是 Human3R 预测，会制造不可验收的动作，因此明确不生成 Human3R-only GMR 输入。
- 当前保留 Human3R 的合适方式仍是“GVHMR/FootMR 人体 + Human3R 场景约束”。若后续继续尝试，应优先加入目标专用裁剪/跟踪并在推理时保存 `smpl_id`，然后重新运行完整 Human3R，而不是在现有缺失轨迹上补洞。

### 产物与文件

```text
/home/user-kevien/gvhmr_pkg/experiments/human3r_only/climb_6466b84f/
├── human3r_only_tracking_overlay.mp4
├── tracking_distance.png
├── metrics.json
└── report.md
```

主要实现：

- `tools/bench/human3r_p2y/evaluate_human3r_only.py`
- `tools/bench/human3r_p2y/README.md`

---

## P4：ydd 单地面 Human3R Ground-XYZ 实验

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`实验完成（诊断候选）`
- 输入：`/home/user-kevien/视频/dataset/dance_date/ydd.mp4`
- 规范化输入：990 帧、30 FPS、33.00 秒、1920×1080，SHA256 `9cba6e3c969d194192839f7a2d5f7fbe636110ccd32f42474e8d3c1c20c024fc`
- 人体：GVHMR + FootMR ViTPose，固定相机
- 场景：完整 Human3R 672S + TTT3R，512×288、100 帧分块
- 范围：单一木地板的自动 ground-only root XYZ 离线候选及 GMR 可读验证
- 不包含：Web 集成、Human3R 人体融合、多层地面/箱顶、测量标定或 3D ground truth

### 视频输入与 FootMR

原视频是 60 FPS、1980 帧。当前 demo 会按输入帧逐帧处理并以 30 FPS 输出，因此先用真正的时间重采样得到 30 FPS、990 帧输入，避免输出变成 66 秒慢动作。

FootMR 完成全部 990 帧并严格保留结果 schema：

- `smpl_params_global` / `smpl_params_incam` 均为 990 帧。
- `K_fullimg` 为 `(990, 3, 3)`，全部 tensor 为有限值。
- `net_outputs.body_pose_before_foot_refine` 存在。
- incam refiner 对比确认只有 body-pose 槽位 6、7（左右踝）变化；其他 19 个槽位逐元素不变。
- incam/global/横向合并视频均为 990 帧、30 FPS、33 秒。

第一次渲染在模型前向完成后因旧版 chumpy 使用 NumPy 已删除的 `np.int` 等别名而失败。`hmr4d/utils/smplx_utils.py` 现在仅在加载旧 SMPL pickle 前补齐兼容别名；复用已保存结果补渲染，不重跑模型。

### Human3R 与单地面提取

Human3R 完成 990 帧：

| 项目 | 结果 |
| --- | ---: |
| 推理耗时 | 285.32 s |
| 总耗时 | 372.83 s |
| 每帧推理耗时 | 0.288 s |
| 峰值 CUDA allocated | 4.31 GB |
| depth/conf/color/camera/SMPL-X | 均为 990 帧 |

该片段只有木地板，没有箱体或台阶，因此新增 `extract_ground_plane.py`，不再强制寻找 floor + box-top 配对。算法使用多帧中位深度、Human3R 人体 mask、固定随机种子 RANSAC，并用图像下半区占比、水平覆盖、底边覆盖和点数选择地面。

爬箱实验沿用的 confidence `2.0` 在本片段没有足够像素，保护条件第一次正确拒绝了结果。Human3R confidence 是最小值接近 1 的指数参数；统计全部采样帧后将 ground-only 默认阈值设为 `1.05`，仍要求像素至少在 15% 的采样帧中有效。最终地面：

| 指标 | 结果 |
| --- | ---: |
| RANSAC 点数 | 1841 |
| 中值残差 | 0.403 cm |
| P95 残差 | 1.655 cm |
| 图像下半区占比 | 100% |
| 水平覆盖率 | 29.88% |
| 底边覆盖率 | 99.65% |
| 法向量 | `[-0.0238, 0.9979, 0.0598]` |

`ground_overlay.png` 已人工检查，绿色点位于目标人物脚下的木地板，没有选中背景墙面。

### 自动 Ground-XYZ 实现

新增 `apply_ground_xyz.py`，不使用爬箱片段的 `pre/top/post` 人工窗口：

- 从 FootMR incam 结果计算六个脚趾/脚跟点，投影到 Human3R 512×288 图像并与地面求射线交点。
- 使用最初 3 秒的高置信度接触帧拟合 Human3R 地面切空间到 GVHMR global XZ 的无反射刚体变换，不拟合额外尺度。
- 只在静态足概率大于 0.8 的帧观测 XYZ correction；930 帧有观测，剔除 26 个离群帧后保留 904 帧。
- 无接触区间只线性插值 correction，再用 61 帧（约 2 秒）低频窗口平滑；不修改人体姿态。
- 首轮 31 帧平滑虽通过旧保护条件，但接触足速度 P95 增至 7.00 mm/frame。最终改用 61 帧平滑，并新增“不超过 baseline 接触足速度 P95 的 1.25 倍”保护条件。
- 输出只替换 `smpl_params_global.transl`；body pose、global orientation、betas、完整 incam、`K_fullimg` 和 `net_outputs` 必须精确不变。

### 定量结果

| 指标 | FootMR baseline | Ground-XYZ | 结果 |
| --- | ---: | ---: | --- |
| 接触足水平残差 median | 5.023 cm | 3.005 cm | 改善 |
| 接触足水平残差 P95 | 18.096 cm | 9.510 cm | 改善 |
| 接触足离地高度 median | 13.328 cm | 1.021 cm | 改善 |
| 接触足离地高度 P95 | 24.182 cm | 5.252 cm | 改善 |
| root 最大步长 | 5.195 cm/frame | 4.924 cm/frame | 未恶化 |
| root 步长 P95 | 1.964 cm/frame | 1.931 cm/frame | 未恶化 |
| root 加速度 P95 | 10.857 m/s² | 10.966 m/s² | 小幅 +1.0%，低于 13.572 上限 |
| 接触足速度 P95 | 5.680 mm/frame | 6.113 mm/frame | +7.6%，低于 7.100 上限 |

最大水平 correction 范数为 15.96 cm；最大绝对 Y correction 为 22.26 cm。末帧 correction 为 `[+1.70, -8.73, -6.09] cm`。baseline 在约 8–23 秒期间出现约 20 cm 的持续 root-Y 抬升，Ground-XYZ 将接触足重新约束到单一地面；并排接触表中未发现由平滑 correction 引入的姿态改变。

算法判定为 `diagnostic_pass`。该判定仅表示内部场景一致性、连续性、修正幅度和 schema 保护条件通过；没有地面测量真值，不能解释为绝对 XYZ 精度已经证明。

### 验证与 GMR

- ground-only helper 3 项、P2-Y helper 2 项、P2-XYZ helper 2 项，共 7 项 CPU 测试通过。
- 顶层 schema、global/incam 子键、pose、orientation、betas、incam、`K_fullimg`、`net_outputs` 不变量全部通过。
- 正式候选和对比视频均为 990 帧、30 FPS、33 秒。
- 使用现有 `gmr` Conda 环境调用 `load_gvhmr_pred_file` 与 `get_gvhmr_data_offline_fast`，实测得到 990 帧、30 FPS、有限 root 和完整 SMPL-X 关节字典，可直接作为 GMR 输入。
- 未修改 `gvhmr-web-tool`；运行期间没有终止其 GPU 任务。Human3R 曾与 WebTool preview 并行运行，未发生显存溢出。

### 产物

```text
/home/user-kevien/gvhmr_pkg/experiments/human3r_scene_ydd/
├── input/ydd_30fps.mp4
├── gvhmr/ydd_30fps/footmr_vitpose/
│   ├── hmr4d_results.pt
│   ├── 1_incam.mp4
│   ├── 2_global.mp4
│   └── ydd_30fps_3_incam_global_horiz.mp4
├── human3r/
├── scene/
│   ├── ground_plane.json
│   ├── ground_overlay.png
│   └── median_depth.npy
└── ground_xyz/
    ├── ground_xyz_hmr4d_results.pt
    ├── footmr_vs_ground_xyz.mp4
    ├── metrics.json
    ├── report.md
    ├── ground_xyz_curves.csv
    └── ground_xyz_curves.png
```

### 限制与处置

- 当前正式结果仍是独立诊断候选，不覆盖 FootMR baseline，也不接入 Web 默认路径。
- 本方案假设固定相机和单一平坦地面。动态相机、反光/弱纹理地面、多层支撑面需要重新验证。
- Human3R 只负责静态场景，绝不使用其人体 SMPL-X root；之前的 Human3R-only 实验已证明人体目标轨迹不可靠。
- 如人工观看并排视频认为 X/Z 约束改变了真实舞步位置，应回退到 FootMR 的 `hmr4d_results.pt`；两份结果均保留。

### 主要涉及文件

- `hmr4d/utils/smplx_utils.py`
- `tools/bench/human3r_p2y/extract_ground_plane.py`
- `tools/bench/human3r_p2y/apply_ground_xyz.py`
- `tools/bench/human3r_p2y/test_apply_ground_xyz.py`
- `tools/bench/human3r_p2y/README.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P5：CoTracker3 接触足 Ground-XZ 实验

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`实验完成（未采用）`
- 上游：CoTracker 官方仓库 commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`
- 输入：`ydd_30fps.mp4`，990 帧、30 FPS、1920×1080
- 基线：P4 `ground_xyz_hmr4d_results.pt`
- 范围：独立 P5 离线接触足 X/Z 诊断；不接入 WebTool，不替换 FootMR/Human3R，不修改 global Y

### 优化目标

Ground-XYZ 已明显改善地面高度和场景 XZ 残差，但接触足速度 P95 比 FootMR 略高。P5 尝试用 CoTracker3 跟踪接触鞋部像素，把接触期间的相对图像位移转换到 Human3R 地面切空间，仅生成 global root X/Z residual，以降低脚滑和水平漂移。

固定相机下静态地板点本身不移动，不能提供新的 root 约束，因此查询点必须是实际鞋部，而不是脚旁地板纹理。

### 关键实现

- 新增固定子模块 `third-party/CoTracker`，版本为 `82e02e8`；通过 `sys.path` 懒加载，不要求把 CoTracker pip 安装进 `gvhmr` 环境。
- 官方 `scaled_offline.pth` 保存到 `inputs/cotracker_assets/`，不写 `inputs/checkpoints`；实测 SHA256 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`，严格加载成功，参数量 25,385,700。
- 左右足接触概率阈值 0.8，最多合并 3 帧缺口，忽略少于 8 帧的短段；长段按最多 60 帧、8 帧重叠分窗。`ydd` 共 40 个窗口。
- 每只脚使用 COCO23 的大脚趾、小脚趾、脚跟三个投影点。使用 visibility、画面边界、单帧速度和三点一致性过滤；40 个窗口均有可用约束。
- 初始全帧版本直接联合相对约束时会跨窗口累计漂移，因此加入 Ground-XYZ 零修正绝对先验和稀疏时间平滑；只允许替换 `smpl_params_global.transl[:, (0, 2)]`。
- 针对目标人在全帧中较小的问题，增加每窗口固定 128×96 足部 ROI，再由 CoTracker 内部缩放到 384×512。ROI 只决定跟踪取景，不修改人体结果。
- 保护条件要求跟踪覆盖率至少 50%、最大修正不超过 15 cm、root 连续性不恶化、场景 XZ median/P95 不超过 baseline 1.1 倍、跟踪锁定残差改善，且接触足速度 median 不恶化、P95 至少改善 2%。失败不生成正式推荐文件。

### 验证结果

CPU helper 5 项全部通过：接触段合并/分窗、ROI 边界、低可见性与错误点剔除、稀疏相对约束连续求解，以及只修改 global X/Z 的 schema 不变量。

GPU 推理没有终止其他进程。官方 checkpoint 严格加载，模型内部输入分辨率 384×512：

| 项目 | 全帧 | 足部 ROI |
| --- | ---: | ---: |
| 可用窗口 | 40 / 40 | 40 / 40 |
| 接触足帧覆盖率 | 99.38% | 93.78% |
| 峰值 CUDA allocated | 5.31 GiB | 5.21 GiB |
| 跟踪锁定残差 P95（未经强平滑） | 1.22 cm | 2.28 cm |
| Ground-XYZ 跟踪锁定残差 P95 | 6.94 cm | 5.35 cm |

首轮没有绝对先验时最大 X/Z correction 累计到 28.38 cm，场景残差 P95 从 9.51 cm 恶化到 30.08 cm，接触足速度 P95 从 6.29 增至 8.53 mm/frame，判定失败。加入绝对先验后最大修正降到约 7.1 cm，场景残差得到保护，但低时间平滑下足速 P95 仍为 9.65–10.34 mm/frame、root 加速度超限。

复用 ROI 约束在 CPU 扫描时间平滑权重 1、2、5、10、20、50、100 与绝对先验权重 0.5、1、2、5、10、20，共 42 组：

| 指标 | Ground-XYZ | P5 扫描最佳 | 结论 |
| --- | ---: | ---: | --- |
| 接触足速度 median | 1.490 mm/frame | 1.365 mm/frame | 可降低 |
| 接触足速度 P95 | 6.288 mm/frame | 6.396 mm/frame | 仍恶化 1.7% |
| root 最大步长 | 4.924 cm/frame | 4.927 cm/frame | 基本不变 |
| root 加速度 P95 | 10.966 m/s² | 10.950 m/s² | 基本不变 |
| 最大 X/Z correction | 0 | 5.44 cm | 保护范围内 |
| 场景 XZ 残差 median | 3.005 cm | 3.159 cm | +5.1% |
| 场景 XZ 残差 P95 | 9.510 cm | 9.951 cm | +4.6% |

不存在同时改善接触足 median/P95 且保住所有保护条件的组合，因此最终判定 `guardrail_failed`，没有生成 `cotracker_ground_xz_hmr4d_results.pt`。正式推荐仍是 P4 Ground-XYZ。

### 原因分析与边界

- 目标脚在全帧中很小；ROI 虽增加局部特征分辨率，但 FootMR 投影点仍可能落在鞋边或邻近地板。叠加图确认多数点位于脚部附近，也存在边界查询。
- CoTracker 优化的是 2D 点轨迹自洽。把鞋部亚像素噪声经单目射线投影到地面后，误差会随深度被放大；较低的 2D/相对约束残差不能证明 3D root 更准确。
- Ground-XYZ 本身已用 61 帧窗口提取低频趋势，接触足 P95 较低。CoTracker residual 引入的高频量超过它能纠正的漂移量；继续增大平滑只会收敛回 baseline。
- 不应通过放宽速度或场景残差保护来“通过”实验。后续只有在鞋部 segmentation、语义点置信度或多视角/测量真值可用时再继续。

### 许可与产物

CoTracker 上游多数代码和 checkpoint 使用 CC BY-NC 4.0，部分上游目录有单独条款。README、LICENSE 和本日志已明确：P5 只能按适用的上游非商业/署名条款使用，不构成本仓库对 CoTracker 的重新许可。

诊断产物：

```text
/home/user-kevien/gvhmr_pkg/experiments/human3r_scene_ydd/
├── cotracker_ground_xz/          # 无绝对先验首轮，失败
├── cotracker_ground_xz_v2/       # 全帧 + 绝对先验，失败
└── cotracker_ground_xz_crop/     # 足部 ROI，失败
```

辅助核对视频和接触表位于 `cotracker_ground_xz_v2/cotracker_tracks.mp4` 与 `cotracker_tracks_contact_sheet.png`。失败目录中的 `candidate_hmr4d_results.pt` 只用于诊断，不能当作推荐 GMR 输入。

主要实现：

- `tools/bench/human3r_p2y/apply_cotracker_ground_xz.py`
- `tools/bench/human3r_p2y/test_apply_cotracker_ground_xz.py`
- `tools/demo/download_cotracker_assets.py`
- `third-party/CoTracker`

### qhy 跟进验证

用户随后要求在 WebTool 最新生成的 `qhy_20260810_143151` 上复测。只读取 WebTool 结果，输入和 FootMR tensor 复制到独立实验目录后处理；没有修改 WebTool 任务：

- 输入：1507 帧、30 FPS、50.23 秒、1920×1080，SHA256 `537f8c626caa077233b2e55fb42459f944043f77ca013477a1f48ce532a56589`。
- FootMR tensor：1507 帧，包含 `body_pose_before_foot_refine`，SHA256 `cbfcdbf6bc1f96da378a519294336aa71a973c3be1aca61e6efa980f9ae5c18b`。
- 视频是固定相机、单一混凝土地面，人物与鞋部比 `ydd` 更大。FootMR 检出 60 个接触窗口，左右足合计 2670 个接触足帧。

Human3R 完成全部 1507 帧，推理 422.55 秒、总耗时 539.07 秒、峰值 CUDA allocated 4.02 GiB。地面叠加图经人工检查，绿色点覆盖人物脚下混凝土地面而非墙面：

| 地面指标 | qhy |
| --- | ---: |
| 平面点数 | 2008 |
| 中值残差 | 0.310 cm |
| P95 残差 | 1.648 cm |
| 图像下半区占比 | 100% |
| 水平覆盖率 | 31.05% |
| 底边覆盖率 | 99.65% |
| 法向量 | `[-0.0121, 0.9997, -0.0216]` |

默认 2 秒 Ground-XYZ 检测到原 FootMR 在约 36–40 秒出现约 1.1 m 的瞬时 root-X 偏移，并用场景足点将其压回。但默认结果需要最大 93.44 cm X/Z correction，接触足速度 P95 从 6.02 增到 11.23 mm/frame，因此保护条件拒绝。为了区分尖峰和高频 correction，又测试 4、6、10 秒平滑：

| 版本 | 最大 X/Z correction | 接触足速度 P95 | 场景 XZ P95 | 判定 |
| --- | ---: | ---: | ---: | --- |
| FootMR | 0 | 6.025 mm/frame | 67.84 cm | baseline |
| Ground-XYZ 2 s | 93.44 cm | 11.232 mm/frame | 14.22 cm | 失败 |
| Ground-XYZ 4 s | 90.16 cm | 10.373 mm/frame | 16.75 cm | 失败 |
| Ground-XYZ 6 s | 81.84 cm | 9.146 mm/frame | 20.27 cm | 失败 |
| Ground-XYZ 10 s | 60.50 cm | 7.038 mm/frame | 30.27 cm | 失败 |

10 秒版本在连续性和接触足速度方面最保守，但仍超过 35 cm 的水平修正上限，且没有测量真值证明约 1.1 m 的 baseline 偏移应被怎样修正。因此只把它作为 CoTracker 上游诊断候选，不提升为推荐 Ground-XYZ。

CoTracker ROI 在 Ground-XYZ-S10 上完成 60/60 窗口，跟踪覆盖率 91.91%，峰值 CUDA allocated 5.21 GiB：

| 指标 | Ground-XYZ-S10 | + CoTracker 默认保守解 |
| --- | ---: | ---: |
| 跟踪锁定残差 median | 1.437 cm | 0.778 cm |
| 跟踪锁定残差 P95 | 9.152 cm | 4.205 cm |
| 场景 XZ 残差 median | 7.718 cm | 7.622 cm |
| 场景 XZ 残差 P95 | 30.265 cm | 28.520 cm |
| 接触足速度 median | 1.627 mm/frame | 1.629 mm/frame |
| 接触足速度 P95 | 7.059 mm/frame | 8.540 mm/frame |
| root 最大步长 | 4.897 cm/frame | 4.820 cm/frame |
| root 加速度 P95 | 16.201 m/s² | 16.067 m/s² |

CoTracker residual 最大 9.40 cm，schema 与 global Y 不变量全部通过，但接触足速度恶化，判定 `guardrail_failed`。复用 2681 条缓存约束扫描时间平滑 20–1000、绝对先验 0.2–20 共 42 组，最佳足速 P95 仍为 7.196 mm/frame，高于 Ground-S10 的 7.059。

为排除失败来自 Ground-XYZ 前置处理，又把同一批像素轨迹反算为直接作用于原 FootMR 的 residual，扫描 49 组：原始 FootMR 在统一接触 mask 下 P95 为 6.069 mm/frame，最佳 CoTracker 组合为 6.094 mm/frame，仍未改善。不存在通过候选，因此 qhy 复测仍不生成正式 `cotracker_ground_xz_hmr4d_results.pt`。

qhy 诊断产物：

```text
/home/user-kevien/gvhmr_pkg/experiments/human3r_scene_qhy/
├── input/qhy_30fps.mp4
├── gvhmr/hmr4d_results.pt
├── human3r/
├── scene/
│   ├── ground_plane.json
│   └── ground_overlay.png
├── ground_xyz_s10/
│   ├── ground_xyz_hmr4d_results.pt       # 失败候选，不推荐
│   ├── qhy_footmr_vs_ground_s10.mp4
│   └── qhy_footmr_vs_ground_s10_contact_sheet.png
└── cotracker_on_ground_s10/
    ├── candidate_hmr4d_results.pt        # 失败候选，不推荐
    ├── metrics.json
    ├── qhy_cotracker_tracks.mp4
    ├── qhy_ground_s10_vs_cotracker.mp4
    └── qhy_ground_s10_vs_cotracker_contact_sheet.png
```

结论：qhy 的鞋部尺寸和跟踪覆盖优于 ydd，但仍重复出现“2D/相对锁定残差改善、3D 接触足速度不改善”。这加强了 P5 暂不采用的判断。qhy 原始 FootMR 仍是唯一通过既有流程的正式结果；Ground-S10 可用于人工判断 36–40 秒 root-X 尖峰，但不能直接替换正式 GMR 输入。

---

## P6：GVHMR/FootMR 视频推理性能优化

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`GPU 已验证`；P6-4 局部注意力实验未采用
- 实验视频：`qhy.mp4`，原始 60 FPS、1507 帧、25.1167 秒
- 标准输入：按时间戳重采样为 30 FPS、754 帧、25.133 秒
- 范围：P6-0 分阶段测量、P6-1 渲染开关、P6-2 共享解码裁剪、P6-3 AMP/batch、P6-4 局部注意力、P6-5 跨任务缓存
- 不包含：降低输入空间分辨率、替换 YOLO/ViTPose/HMR2 模型、修改 `gvhmr-web-tool`、终止其他 GPU 进程

### 先修复 FPS 时序错误

旧 Web 流程把 60 FPS 的 1507 个输入帧全部写成 30 FPS，得到 50.23 秒输出，动作因此以约一半速度播放。这不是 GVHMR 网络造成的动作变化，而是容器时间轴错误。

现在统一使用 ffmpeg `fps` filter 按时间戳先重采样，再运行 bbox、姿态、特征、VO 和人体网络：

```text
60 FPS / 1507 帧 / 25.1167 s
    -> 时间戳重采样
30 FPS / 754 帧 / 25.133 s
    -> YOLO -> ViTPose -> HMR2 -> GVHMR + FootMR
```

单元测试用 12 帧 60 FPS 合成视频验证输出为 6 帧 30 FPS，时长误差不超过 0.05 秒。该修复只恢复正确动作速度和工作量，单独记账，不计入 P6 模型加速收益。Web external worker 兼容路径会优先读取同任务目录的 `submitted_input.mp4` 并重新建立正确时间轴；没有修改 WebTool 本身。

### P6-0：分阶段 profiler 与瓶颈定位

新增 CUDA 同步 profiler，`--profile` 将结果写入各输出目录的 `performance.json`。754 帧 FP32 冷启动基线为：

| 阶段 | 耗时 |
| --- | ---: |
| 30 FPS 重采样 | 4.37 s |
| YOLOv8x 跟踪 | 35.15 s |
| ViTPose whole-body | 75.30 s |
| HMR2 特征 | 42.83 s |
| GVHMR + FootMR 前向 | 0.70 s |
| 模型加载 | 1.62 s |

结论是主要瓶颈位于 YOLO、ViTPose 和 HMR2，GVHMR + FootMR 网络对 25.1 秒视频的前向不足一秒。最终默认配置的独立冷启动完成 754 帧，端到端 wall time 为 138.83 秒、峰值 CUDA allocated 为 4248.37 MiB；其中重采样 6.36 秒、YOLO 43.25 秒、共享解码裁剪 4.94 秒、ViTPose FP16 35.45 秒、HMR2 FP32 46.52 秒、人体网络 0.67 秒。冷启动阶段受磁盘和 GPU 状态波动影响，因此只用同条件的阶段测试判断单项收益，不用这两次 wall time 计算总加速比。

### P6-1：按需关闭预览渲染

增加 `--render {all,none,incam,global}`，默认 `all` 保持原行为。754 帧预览实测 incam 34.82 秒、global 109.20 秒、合并 9.28 秒，共 153.4 秒。只为 GMR 生成 tensor 时使用 `--render none` 可完全移除这部分工作，`hmr4d_results.pt` 不变。这是本轮最大且严格无损的加速项；WebTool 已有 `generate_preview=false` 路由，无需修改 WebTool。

### P6-2：ViTPose/HMR2 共享解码与裁剪

当姿态和图像特征均未命中缓存时，只解码一次视频并共享人体裁剪。视觉阶段从约 118.13 秒降为 116.03 秒，节省约 2.1 秒（1.8%）。bbox、关键点和 HMR2 features 均 bit-identical，已默认启用，可用 `--no-shared_preprocess` 回退。

### P6-3：混合精度和 batch 扫描

所有推理包装改为 `torch.inference_mode()`，并分别控制姿态和特征精度：

| 实验 | 阶段耗时变化 | 最终质量差异 | 决策 |
| --- | ---: | --- | --- |
| ViTPose FP16 | 71.8 -> 29.6 s | 足点平均 0.0047 px、最大 0.117 px；global translation 最大 1.67 mm；global pose 最大 0.131 度 | FootMR 默认启用 |
| HMR2 FP16 | 40.5 -> 20.5 s | 单独启用时 global translation 最大约 4.48 mm | 不默认 |
| ViTPose + HMR2 FP16 | 视觉阶段进一步缩短 | 后处理放大边界差异，global translation 最大约 2.29 cm | 不默认 |
| 关闭 ViTPose flip-test | 75.3 -> 42.3 s | 足点平均 5.51 px；global pose 最大约 10.4 度；translation 最大约 5.09 cm | 不采用 |

ViTPose FP16 与 HMR2 FP32 的 batch 8/16/32 扫描中，batch 16 接近两者最佳且适合 12 GB GPU，因此保持默认 16。严格质量回退为 `--pose_inference_dtype fp32`；`--inference_dtype fp16` 会同时把 HMR2 改为 FP16，属于非默认快速模式。

### P6-4：局部注意力

增加 dense/local 实验开关和等价性回归，但不改变默认实现：

| 序列长度 | dense | local | 结果 |
| --- | ---: | ---: | --- |
| 754 帧 | 0.1283 s | 0.1460 s | local 慢约 13.8%，输出一致 |
| 1508 帧 | 0.2543 s | 0.2451 s | local 快约 3.6%，有轻微浮点差异 |

正确重采样后人体网络本身不足一秒，局部窗口在当前长度上没有稳定收益，因此默认仍为 `--attention_impl dense`。local 只保留为长序列实验开关。

### P6-5：内容寻址的跨任务缓存

缓存键包含规范化视频 SHA256、checkpoint 文件状态和影响输出的推理参数。bbox 与 HMR2 feature 可跨 GVHMR/FootMR 共享；COCO17、FootMR-ViTPose COCO23 和 FootMR-Sapiens COCO23 姿态按 backend、关节数、精度和 flip-test 隔离。默认缓存目录为 `outputs/cache`，可用 `--cache_root none` 禁用。

实测缓存命中后 bbox、pose、features 文件 SHA256 完全一致，最终 28 组输出 tensor bit-identical；30 FPS 规范化视频恢复从 6.50 秒降至 0.118 秒，其余视觉预处理恢复约为毫秒级。只读 Web external-worker 探针生成 754 帧、30 FPS、25.133 秒结果，tensor 与 standalone ViTPose-FP16/HMR2-FP32 逐元素一致。WebTool worker 仍会先进行一次旧复制，core 随后纠正时间轴，约浪费 5 秒；在“不修改 WebTool”约束下保留该限制。

### 验证命令与回退

```bash
conda activate gvhmr
python tools/bench/test_p6_optimizations.py
python tools/bench/test_footmr_integration.py
python -m compileall -q hmr4d tools/demo tools/bench/test_p6_optimizations.py
git diff --check
```

推荐的 GMR-only 命令：

```bash
python tools/demo/demo.py --video VIDEO --render none --profile
```

各优化均可独立回退：保留预览用 `--render all`；ViTPose 回 FP32 用 `--pose_inference_dtype fp32`；关闭跨任务缓存用 `--cache_root none`；关闭共享裁剪用 `--no-shared_preprocess`；注意力保持默认 dense。FPS 时间重采样属于正确性修复，不提供旧慢动作路径。

### 主要涉及文件

- `hmr4d/utils/video_io_utils.py`
- `hmr4d/utils/perf.py`
- `hmr4d/utils/preproc/content_cache.py`
- `hmr4d/utils/preproc/vitpose.py`
- `hmr4d/utils/preproc/vitfeat_extractor.py`
- `hmr4d/network/base_arch/transformer/encoder_rope.py`
- `hmr4d/network/gvhmr/relative_transformer.py`
- `hmr4d/configs/demo.yaml`
- `hmr4d/configs/demo_gvhmr.yaml`
- `tools/demo/demo.py`
- `tools/demo/demo_folder.py`
- `tools/bench/test_p6_optimizations.py`

---

## P7：无需 Human3R 的自标定单地面 root-Y 约束

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`GPU 已验证`
- 验证输入：P6 正确重采样后的 qhy FootMR 结果，754 帧、30 FPS、25.133 秒
- 范围：单一平地、固定或近似固定相机、只修正 global root-Y、ELF3 GMR 转换和完整视频渲染
- 不包含：箱顶/台阶等多支撑面、root-X/Z 修正、替代 Human3R 的一般场景重建、修改 WebTool 或 GMR WebTool

### 方法

该方法不读取 Human3R 点云。它将 FootMR 的四个静态足概率扩展到左右脚各三个足底标记，使用开头三秒内的高置信度接触帧分别标定六个标记的支撑高度。后续每个接触帧计算“标定高度减当前脚底高度”的中值作为 root-Y 观测，依次执行：

1. 0.5 秒 rolling median 建立局部趋势；
2. median/MAD 离群点拒绝；
3. 对缺少接触观测的区间插值；
4. 默认两秒 Hann 低通平滑；
5. 以标定窗口中值归零，只写入 `smpl_params_global.transl[:, 1]`。

输出保持 global X/Z、body pose、global orientation、betas、incam、相机内参和全部 `net_outputs` 逐元素不变。最大 Y 修正、root 步长、root 加速度、接触足速度和接触高度残差都有保护条件；失败时只生成带 `candidate_` 前缀的诊断文件。

### qhy 结果

| 指标 | FootMR | Flat-ground-Y |
| --- | ---: | ---: |
| 接触脚高度残差 median | 13.957 cm | 1.024 cm |
| 接触脚高度残差 P95 | 24.716 cm | 4.174 cm |
| 接触足速度 median | 2.165 mm/frame | 2.162 mm/frame |
| 接触足速度 P95 | 9.092 mm/frame | 9.017 mm/frame |
| root 最大步长 | 6.437 cm/frame | 6.433 cm/frame |
| root 加速度 P95 | 19.116 m/s² | 19.118 m/s² |

- 754 帧中有 685 帧包含高置信度接触观测，681 帧通过离群点过滤。
- 最大绝对 root-Y 修正为 23.459 cm，末帧修正为 +22.432 cm，低于 25 cm 保护上限。
- global X/Z 与所有非 root-Y tensor 不变量通过，最终判定 `diagnostic_pass`。
- 该结果只证明相对开头支撑高度的一致性改善；没有外部场景或测量真值时，不能证明绝对地面高度，也不能用于箱子、台阶等多高度动作。

### GMR 与播放验证

通过 `gmr` Conda 环境将通过版本转换为 ELF3：

- `robot_motion.pkl`：754 帧，root position、root quaternion 和 29 个 DoF 全部有限。
- `robot_preview.mp4`：640x480、30 FPS、754 帧、25.133 秒，已用 VLC 实际启动播放。
- GMR ground offset 为 0.0891 m；机器人 root 高度范围为 0.9606–1.0386 m。

检查时发现 `gmr-web-tool/gmr_web/runner.py` 的 `TARGET_MOTION_FPS=200` 会在没有实际插帧的情况下把 754 帧标记为 200 FPS，首次视频只有 3.77 秒。该错误版本未用于验收或播放；本次在独立实验目录按源时间轴重新封装为 30 FPS。遵守仓库边界，没有修改 `gmr-web-tool`。在接入 Web 自动链路前，应在 GMR 侧修复“帧数不变但 metadata 设为 200 FPS”的问题。

### 与 Human3R 的同基线对比

为避免此前 1507 帧慢时间轴造成不公平比较，将已有 Human3R 场景的偶数帧 `0,2,...,1506` 对齐到当前 754 帧。缩略图像素核对显示固定偶数映射的均方误差显著低于相邻偏移。三种约束均从同一份 P6 FootMR tensor 开始，并使用相同的 30 FPS、接触 mask 和两秒平滑窗口。

| 指标 | FootMR | 自标定 Flat-Y | Human3R-Y-only | Human3R-XYZ |
| --- | ---: | ---: | ---: | ---: |
| 对 Human3R 地面的高度残差 median | 14.037 cm | 1.035 cm | 1.014 cm | 1.014 cm |
| 对 Human3R 地面的高度残差 P95 | 24.374 cm | 4.141 cm | 4.184 cm | 4.184 cm |
| 最大绝对 Y correction | 0 | 23.459 cm | 23.693 cm | 23.693 cm |
| 最大 X/Z correction | 0 | 0 | 0 | 79.526 cm |
| 接触足速度 P95 | 9.092 mm/frame | 9.017 mm/frame | 9.091 mm/frame | 11.519 mm/frame |

自标定与 Human3R-Y-only 的 Y correction 平均绝对差为 0.137 cm、P95 差 0.385 cm、最大差 0.560 cm、末帧只差 0.025 cm。在 ELF3 上，两者 robot root 平均差 0.167 cm、最大差 0.587 cm，关节角最大差约 `3.0e-5 rad`。关键帧接触表中两者没有可辨识的姿态差异：

| ELF3 指标 | FootMR | 自标定 Flat-Y | Human3R-Y-only |
| --- | ---: | ---: | ---: |
| robot root 高度范围 | 23.425 cm | 7.801 cm | 7.654 cm |
| 末帧相对首帧高度 | -21.871 cm | -4.902 cm | -4.952 cm |

结论：在 qhy 这种单一平地视频上，Human3R 对 Y 没有产生实质增益；自标定方法以更少依赖复现了同等效果。Human3R-XYZ 虽将场景 X/Z 残差 P95 从 39.13 cm 降到 14.62 cm，但需要最大 79.53 cm 水平修正，并使接触足速度超过保护线，最终仍为 `guardrail_failed`，不能作为正式 GMR 输入。推荐 qhy 使用自标定 Flat-Y；Human3R 只保留给箱顶、台阶或需要外部场景几何的片段。

### 产物和主要文件

```text
/home/user-kevien/gvhmr_pkg/experiments/flat_ground_y_qhy_754/
├── flat_ground_y_hmr4d_results.pt
├── metrics.json
├── flat_ground_y_curve.png
├── comparison_flat_vs_human3r.json
├── comparison_flat_vs_human3r.png
├── gmr_comparison_metrics.json
├── gmr_comparison_3way.mp4
├── gmr_comparison_contact_sheet.png
└── gmr_elf3_30fps/
    ├── robot_motion.pkl
    └── robot_preview.mp4
```

- `tools/bench/human3r_p2y/apply_flat_ground_y.py`
- `tools/bench/human3r_p2y/test_apply_flat_ground_y.py`

### 回退方式

不使用 `flat_ground_y_hmr4d_results.pt`，直接把原 FootMR `hmr4d_results.pt` 交给 GMR。该实验不覆盖原结果，也未接入默认 Web 路径。

---

## P8：WebTool 可选地面约束接入

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`CPU 已验证`
- 上游依据或实验基线：P7 Flat-ground-Y、qhy 754 帧 FootMR 结果
- 范围：`gvhmr-web-tool` 单视频/批量任务、external core worker、任务产物与下载、Web 配置界面
- 不包含：启用 Human3R 推理、一般场景重建、多支撑面约束、修改 GMR 转换算法

### 优化目标

将 P7 单平地 root-Y 后处理作为 Web 可选功能接入，默认对新提交的 FootMR 任务启用，同时保留不处理的原始输出。Human3R 场景约束先展示接口位置，但在后端实现和验收完成前不可选择。

### 关键实现

- 单视频和批量表单增加三项互斥选择：`不启用`、默认 `自动平地约束`、禁用的 `Human3R 场景约束`。
- Web capabilities 根据 external core 中 P7 脚本是否存在决定 Flat-ground-Y 是否可用；脚本缺失时自动回退为默认不启用。
- external worker 在 FootMR 推理之后调用 P7 脚本。只有 `decision=diagnostic_pass` 才用增强 tensor 替换任务主结果；保护失败或脚本失败时恢复原结果。
- 启用时将原始 FootMR tensor 保存为 `hmr4d_results_raw.pt`，增强候选和指标分别保存在 `ground_constraint_flat_y/flat_ground_y_hmr4d_results.pt`、`ground_constraint_flat_y/metrics.json`。
- 主 `hmr4d_results.pt` 始终表示当前选中的有效结果，因此 Web 的预览和“转 ELF3”沿用原调用链，不会再额外执行一次地面约束。
- runner 返回的 `ground_constraint_status/error` 作为任务元数据保存，只有 `*_path` 写入 artifacts，避免把错误文本误当文件路径。
- Human3R 在 UI 中可见但 disabled；API 直接提交 `human3r` 会返回 400，避免绕过前端误用未完成后端。

### 接口、配置与资产变化

external core worker 新增：

```text
--ground-constraint {none,flat_y,human3r}
```

任务 JSON 新增 `ground_constraint`、`ground_constraint_status` 和可选的 `ground_constraint_error`。下载列表新增原始 FootMR PT 与地面约束指标 JSON；ZIP 同步包含这些产物。没有新增模型或 Python 依赖，继续使用 `gvhmr` Conda 环境和 external core source mode。

### 验证方法与结果

- Python `compileall`、`node --check`、`git diff --check` 全部通过。
- `python -m unittest discover -s tests -p 'test_service_web.py' -v`：18 项全部通过，覆盖选择持久化、Human3R 拒绝、capabilities、external runner 参数路由，以及 raw/enhanced 保留。
- 使用 qhy 754 帧真实 FootMR tensor 在临时目录调用 Web worker：状态为 `applied`，指标判定为 `diagnostic_pass`；raw、enhanced、metrics 均存在，主结果 SHA256 等于 enhanced 且不同于 raw。
- 验证使用临时目录，没有覆盖 P7 实验产物，也没有运行新的 GPU 人体推理。

### 未完成项和已知风险

- Human3R 只保留不可选入口，尚未定义场景缓存、帧对齐、失败保护和多平面选择协议。
- Flat-ground-Y 只适用于单一平地。箱顶、台阶、跳上不同高度支撑面时应选择“不启用”，不能把当前算法解释为场景几何重建。
- `CPU 已验证` 表示后处理和 Web 链路验证完成；还需用重启后的 Web 新提交一段完整视频，才能把本项标记为 `GPU 已验证`。

### 回退方式

新任务在页面选择“不启用”。对已生成任务可直接下载或使用 `hmr4d_results_raw.pt`；Web 不会覆盖该原始副本。若 P7 脚本不可用，capabilities 自动禁用 Flat-ground-Y 并将默认项切换为“不启用”。

### 主要涉及文件

- core：`tools/bench/human3r_p2y/apply_flat_ground_y.py`
- core：`docs/OPTIMIZATION_LOG.md`
- WebTool：`hmr4d/service/external_core.py`
- WebTool：`hmr4d/service/external_core_worker.py`
- WebTool：`hmr4d/service/manager.py`
- WebTool：`hmr4d/service/server.py`
- WebTool：`hmr4d/service/static_app/{index.html,styles.css,app.js}`
- WebTool：`tests/test_service_web.py`

---

## P9：共享平面最低接触足底 root-Y 候选

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`实验完成（未采用）`
- 上游依据或实验基线：P7 自标定 Flat-ground-Y、cxk 367 帧、ydd 990 帧、qhy 754 帧正确 30 FPS FootMR 结果
- 范围：单一平地、global root-Y、FootMR 接触概率、ELF3 诊断播放
- 不包含：修改 Web 默认算法、root-X/Z、脚踝姿态、Human3R、多支撑平面

### 优化目标

P7 在 cxk 上虽然状态为 `applied`，但最大修正只有 3.89 cm；按旧指标接触高度 median/P95 仅从 2.12/8.11 cm 改善到 1.88/6.75 cm，肉眼仍可见悬空。原因是 P7 为六个足底标记分别保存标定高度，并使用两秒低频平滑，更接近长期漂移校正，不是严格的共享地面接触约束。

### 关键实现

- 将四个 FootMR 静态概率合并为左右脚接触 mask。
- 每只脚使用三个足底标记中的最低点作为 sole height。
- 从开头三秒高置信度接触样本标定一个左右脚共享的地面高度。
- 每个接触帧用“共享地面减当前最低支撑足底”构造单一 root-Y 观测；默认使用 0.5 秒平滑。
- 只修改 `smpl_params_global.transl[:, 1]`，其余 tensor 逐元素保持不变。
- 增加有效性保护：接触足底 median 必须不高于 1 cm、P95 不高于 5 cm；保留最大修正、root 步长、加速度和接触足速度保护。

### 验证方法与结果

CPU 单元检查共 7 项通过，其中新版新增 3 项：左右脚接触合并、足底最低点选择、共享地面慢漂移恢复。

统一使用“接触脚最低足底到共享平面”的绝对残差比较：

| 视频 | 方案 | median | P95 | 最大 Y 修正 | 接触足速度 P95 | 判定 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cxk | raw FootMR | 3.752 cm | 9.554 cm | 0 | 12.042 mm/frame | baseline |
| cxk | P7 Flat-Y 2.0s | 2.515 cm | 6.778 cm | 3.890 cm | 12.214 mm/frame | 通过但效果弱 |
| cxk | P9 shared-floor 0.5s | 0.533 cm | 4.540 cm | 8.051 cm | 12.381 mm/frame | `diagnostic_pass` |
| ydd | raw FootMR | 13.272 cm | 22.706 cm | 0 | 5.680 mm/frame | baseline |
| ydd | P7 Flat-Y 2.0s | 0.429 cm | 1.600 cm | 22.234 cm | 5.625 mm/frame | `diagnostic_pass` |
| ydd | P9 shared-floor 1.0s | 0.248 cm | 1.399 cm | 23.108 cm | 5.606 mm/frame | `diagnostic_pass` |
| qhy | raw FootMR | 14.885 cm | 25.742 cm | 0 | 9.088 mm/frame | baseline |
| qhy | P7 Flat-Y 2.0s | 1.020 cm | 4.675 cm | 23.461 cm | 9.020 mm/frame | `diagnostic_pass` |
| qhy | P9 shared-floor 1.0s | 0.934 cm | 3.648 cm | 25.742 cm | 8.983 mm/frame | `guardrail_failed` |

固定 0.5 秒版本在 ydd 通过，median/P95 为 0.225/1.361 cm；在 qhy 为 0.833/3.822 cm，但最大修正达到 26.095 cm，超过 25 cm 保护线。将 qhy 平滑改为 1.0 秒或 2.0 秒仍分别需要 25.742 cm 和 25.119 cm，因此问题不是单纯由平滑窗口过短引起。

按用户要求，将 qhy 0.5 秒失败候选作为诊断输入送入 GMR，不受保护线阻止：

- ELF3：754 帧、30 FPS，root 和 DoF 全部有限。
- GMR ground offset：0.1080 m。
- robot root 高度范围：0.9496–1.0197 m。
- 已使用 MuJoCo 交互 viewer 循环播放，启用 COM 投影与支撑多边形；该播放不改变失败判定。

### 未完成项和已知风险

- 新版在 cxk、ydd 改善，但 qhy 超出最大修正保护，不能证明对现有 P7 无退化。
- “最低点贴地”优先避免整只脚悬空，但双脚同时接触且高度不一致时只保证更低的脚；另一只脚仍可能悬空。
- 当前接触来自 FootMR logits，没有物理接触力或场景真值。快速抬脚、交叉脚和脚尖动作可能被误判。
- 若后续接入 Web，必须采用“P9 通过则使用，否则回退 P7”的级联，而不能直接替换 P7。

### 回退方式

继续保持 Web 默认调用 P7 `apply_flat_ground_y.py`。P9 只生成独立实验 tensor；qhy 正式结果继续使用 P7 通过版本。

### 主要涉及文件和产物

- `tools/bench/human3r_p2y/apply_contact_floor_y.py`
- `tools/bench/human3r_p2y/test_apply_contact_floor_y.py`
- `/home/user-kevien/gvhmr_pkg/experiments/flat_ground_y_cxk_367/`
- `/home/user-kevien/gvhmr_pkg/experiments/contact_floor_y_regression_ydd_qhy/`

---

## P10：Web 自动平地切换为共享接触足底约束

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：`CPU 已验证`
- 上游依据或实验基线：P9 shared-floor 0.5 秒候选及 cxk/ydd/qhy 回归
- 范围：`gvhmr-web-tool` 自动平地 worker、能力检测、任务产物和兼容读取
- 不包含：Human3R、root-X/Z、完整新视频 GPU 人体推理

### 优化目标

按人工检查结果，用 P9 的左右脚共享地面、最低接触足底观测和 0.5 秒平滑替换 Web 中 P7 的六标记独立高度、2 秒低频 Flat-Y。保留“自动平地约束”接口名和 `flat_y` API value，避免破坏已有调用。

### 关键实现

- Web worker 改为调用 `apply_contact_floor_y.py --smoothing-seconds 0.5 --allow-large-correction`。
- 固定 25 cm 最大修正仅记录为诊断值，不参与 Web 决策；有限值、root 步长、加速度、接触足速度以及 median/P95 有效性仍强制检查。
- 新增强结果名为 `ground_constraint_flat_y/contact_floor_y_hmr4d_results.pt`，主 `hmr4d_results.pt`、原始 `hmr4d_results_raw.pt` 和 metrics 接口不变。
- artifact map 和 ZIP 同时兼容旧 `flat_ground_y_hmr4d_results.pt`，历史任务不会因升级丢失下载入口。
- capabilities 只有检测到新版脚本时才将自动平地设为可用和默认。

### 验证方法与结果

- core Flat-Y/P2-Y 单元检查 8 项通过。
- Web 单元测试 18 项通过，Python compileall、JavaScript 语法和 diff whitespace 检查通过。
- 使用 Web worker 在临时目录真实回归，三者均为 `applied`、主结果 SHA256 等于 shared-floor enhanced、强制保护无失败：

| 视频 | 帧数 | 最大 Y 修正 | 固定幅度阈值 | Web 结果 |
| --- | ---: | ---: | --- | --- |
| cxk | 367 | 8.051 cm | 不强制 | `diagnostic_pass` |
| ydd | 990 | 23.361 cm | 不强制 | `diagnostic_pass` |
| qhy | 754 | 26.095 cm | 不强制 | `diagnostic_pass` |

测试使用已有正确 30 FPS FootMR tensor，不重新运行人体网络，也不覆盖现有任务或实验结果。

### 未完成项和已知风险

- 取消固定 25 cm 上限意味着长视频可接受更大的平滑 Y 修正；其他连续性和效果保护仍会回退，但不能替代场景真值。
- 当前只证明三个本地样本的后处理链路；还需从重启后的 Web 新提交完整视频，才能标记为 `GPU 已验证`。
- 双脚高度观测冲突时仍以更低的接触足为锚，不能保证两只脚同时完全贴地。

### 回退方式

页面选择“不启用”可完全绕过后处理。代码级回退可将 worker 路由恢复为 `apply_flat_ground_y.py`；raw FootMR 始终保存在 `hmr4d_results_raw.pt`。

### 主要涉及文件

- core：`tools/bench/human3r_p2y/apply_contact_floor_y.py`
- core：`tools/bench/human3r_p2y/test_apply_contact_floor_y.py`
- WebTool：`hmr4d/service/external_core_worker.py`
- WebTool：`hmr4d/service/manager.py`
- WebTool：`hmr4d/service/server.py`
- WebTool：`hmr4d/service/static_app/index.html`
- WebTool：`tests/test_service_web.py`

---

## P11：ELF3 机器人侧接触地面约束实验

### 基本信息

- 日期：2026-08-10
- 分支：`feature/gvhmr-opt`
- 状态：三段回归完成（待人工确认）
- 上游依据或实验基线：P10 cxk shared-floor 0.5 秒结果及其 GMR ELF3 输出
- 范围：GMR 重定向完成后的 `robot_motion.pkl` root-Z 独立后处理
- 不包含：修改 `gmr-web-tool` 默认流程、重新运行 GVHMR/FootMR、动力学控制

### 优化目标

解决人体侧自动平地已经生效、但 cxk 经 GMR 重定向后 ELF3 接触脚仍悬空约 8 cm 的问题。约束直接使用最终 MuJoCo ELF3 的 `lf_tc/rf_tc` site，而不是人体所有关节的最低点。

### 关键实现

- 从 FootMR 四路 `static_conf_logits` 合并左右脚高置信接触状态。
- 在 MuJoCo 中对原始 `robot_motion.pkl` 逐帧正向运动学，读取 `lf_tc/rf_tc` 的实际高度。
- 接触帧以最低接触脚构造 3 cm 目标高度，缺失区间插值并做 0.5 秒平滑。
- 只修改 `root_pos[:,2]`；增加任意脚底至少高于 0.5 cm 的硬约束，防止错误接触标签导致非接触脚穿地。
- 输出独立 PKL 和 JSON 指标，不覆盖 GMR 原始结果。

### 接口、配置与资产变化

- 新增 `tools/bench/human3r_p2y/apply_robot_contact_floor_z.py`。
- 实验输出：`experiments/contact_floor_y_regression_ydd_qhy/cxk/gmr_robot_contact_floor_z_0p5s/`。
- 默认目标脚底高度 3 cm、最低安全高度 0.5 cm、接触阈值 0.8、平滑 0.5 秒。

### 验证方法与结果

使用 cxk、ydd、qhy 的 30 FPS FootMR tensor 和现有 ELF3 GMR 输出，在 `gmr` Conda 环境运行 MuJoCo 正向运动学：

| 指标 | GMR 原始 | P11 |
| --- | ---: | ---: |
| 接触帧最低支撑脚中位高度 | 7.888 cm | 3.055 cm |
| 接触帧最低支撑脚 P05 / P95 | 5.484 / 11.686 cm | 1.768 / 5.596 cm |
| 全序列最低脚底高度 | -0.955 cm | 0.500 cm |
| 脚底穿地帧 | 6 | 0 |
| root-Z P95 单帧变化 | 2.126 cm | 2.195 cm |
| root-Z P95 加速度 | 8.903 m/s² | 10.886 m/s² |

修正量中位数为 -4.453 cm，范围 -9.146 至 +1.455 cm；102 帧触发防穿地约束。所有输出有限，帧数和 FPS 不变。

ydd 和 qhy 使用完全相同参数继续回归：

| 视频 | 帧数 | 接触锚点中位高度（前→后） | 穿地帧（前→后） | 修正中位数 | root-Z P95 单帧变化（前→后） |
| --- | ---: | ---: | ---: | ---: | ---: |
| cxk | 367 | 7.888 → 3.055 cm | 6 → 0 | -4.453 cm | 2.126 → 2.195 cm |
| ydd | 990 | -1.689 → 3.000 cm | 976 → 0 | +4.623 cm | 0.922 → 0.936 cm |
| qhy | 754 | 1.024 → 3.000 cm | 167 → 0 | +1.869 cm | 0.729 → 0.786 cm |

三段均保持原帧数和 30 FPS，输出有限；说明修正来自逐帧 ELF3 脚底几何，而非复用 cxk 的固定偏移。

### 未完成项和已知风险

- FootMR 接触标签与 ELF3 最低脚在部分帧冲突，防穿地约束会把接触脚抬高；因此接触脚 P95 仍为 5.596 cm。
- 防穿地硬约束使 root-Z P95 加速度上升约 22.3%，需要 MuJoCo 人工检查是否出现可见竖直抖动。
- 三段数值回归均通过，但 ydd/qhy 仍需 MuJoCo 人工检查动作观感，尤其是防穿地硬约束是否引入可见竖直抖动。

### 回退方式

继续使用原始 `gmr_contact_floor_y_0p5s/robot_motion.pkl` 即可完全绕过 P11；P11 没有修改 Web 或 GMR 代码。

### 主要涉及文件

- `tools/bench/human3r_p2y/apply_robot_contact_floor_z.py`
- `docs/OPTIMIZATION_LOG.md`
- `/home/user-kevien/gvhmr_pkg/experiments/contact_floor_y_regression_ydd_qhy/cxk/gmr_robot_contact_floor_z_0p5s/robot_motion_contact_floor_z.pkl`
- `/home/user-kevien/gvhmr_pkg/experiments/contact_floor_y_regression_ydd_qhy/cxk/gmr_robot_contact_floor_z_0p5s/robot_contact_floor_z_metrics.json`
- `/home/user-kevien/gvhmr_pkg/experiments/contact_floor_y_regression_ydd_qhy/ydd/gmr_robot_contact_floor_z_0p5s/robot_motion_contact_floor_z.pkl`
- `/home/user-kevien/gvhmr_pkg/experiments/contact_floor_y_regression_ydd_qhy/qhy/gmr_robot_contact_floor_z_0p5s/robot_motion_contact_floor_z.pkl`

---

## P12：GMR Web 内置 GVHMR 接触地面约束 pipeline

### 基本信息

- 日期：2026-08-10
- 分支：core `feature/gvhmr-opt`；GMR Web 本地 worktree
- 状态：Web API 已验证
- 上游依据或实验基线：P11 cxk/ydd/qhy ELF3 robot-side 独立实验
- 范围：GMR Web 后处理 pipeline、产物/质量报告/CSV/仿真路由、带手与不带手 ELF3
- 不包含：改变默认 `auto` pipeline、修改通用 gmr-optimizer 算法、训练或 GPU 推理

### 优化目标

将 P11 从独立脚本投入 GMR Web，同时保留原始 `robot_motion.pkl`。用户可在后处理流程选择
`GVHMR 接触地面约束`，通过现有后处理下载、CSV 和 MuJoCo 入口使用修正版。

### 关键实现

- 新增本地 pipeline `gvhmr_contact_floor`，不通过通用 `run_postprocess.sh`。
- 只接受 `gvhmr_smplx` + ELF3；支持 `elf3` 和 `elf3_no_hands`，其他输入在提交/补跑时明确拒绝。
- 从任务暂存的 GVHMR `.pt` 读取 FootMR `static_conf_logits`；若机器人输出帧数不同，以时间端点对齐重采样接触 mask。
- MuJoCo 正向运动学读取最终 `lf_tc/rf_tc`，接触目标使用任务 `ground_clearance`（默认 3 cm），0.5 秒平滑，并保证任一脚底至少 0.5 cm。
- 只修改后处理副本 `root_pos[:,2]`，生成 `motion_gvhmr_contact_floor.pkl`、CSV 和 `quality_gvhmr_contact_floor.json`。
- 原始 `robot_motion.pkl` 不覆盖；“仿真后处理”自动指向新 PKL。
- `auto` 会检查 GVHMR tensor：存在可用 FootMR `static_conf_logits` 时选择新 pipeline；原版 GVHMR 自动回退 `gvhmr_safe`。
- 修正单文件 API 与 capabilities/UI 的默认值不一致：调用方省略 `auto_postprocess` 时现在默认启用；显式传 `false` 仍可关闭。

### 接口、配置与资产变化

- 页面后处理列表新增 `GVHMR 接触地面约束`。
- capabilities 新增 `gvhmr_contact_floor_available=true`。
- 真实运行服务来自 `/home/user-kevien/gvhmr_pkg/gmr-optimizer/gmr_web`；同时将兼容实现同步到公开封装 `/home/user-kevien/gvhmr_pkg/gmr-web-tool/gmr_web`。
- 服务地址 `http://127.0.0.1:7870/`，最终重启后 PID 文件记录 `455044`。

### 验证方法与结果

- `gmr` 环境：新 pipeline 4 项单元测试通过；GMR Web lifecycle 12 项通过。
- Python compileall、JavaScript syntax、Git diff whitespace 检查通过。
- cxk/ydd/qhy 真实 tensor + ELF3 PKL 离线路由回归与 P11 完全一致：接触锚点中位高度分别变为 3.055、3.000、3.000 cm，三段穿地帧均为 0。
- 使用已有 Web 任务 `job_288741d370bc`（`ydd_beta_native.pt`、`elf3_no_hands`）调用真实 POST `/jobs/{id}/postprocess`：
  - 状态 `queued → running → succeeded`；
  - 接触锚点中位高度 `8.371 → 3.001 cm`；
  - 穿地帧 `48 → 0`；
  - 自动生成后处理 PKL、CSV 和质量报告；
  - capabilities 返回新 pipeline 及正确中文说明。
- 用户随后新建的 cxk 任务 `job_9233f05a77c8` 暴露出单文件 API 缺省关闭自动后处理：任务最初只播放 raw，接触锚点中位高度为 12.478 cm。补跑新 pipeline 后为 3.040 cm，并将 MuJoCo 切换到 `motion_gvhmr_contact_floor.pkl`。据此加入上述动态 `auto` 路由和 API 默认值修复。

### 未完成项和已知风险

- pipeline 依赖 FootMR `static_conf_logits`；原版 GVHMR tensor 的 `auto` 会回退 `gvhmr_safe`，显式选择新 pipeline 则给出明确错误。
- FootMR 接触错误时防穿地约束可能抬高标记为接触的另一只脚；质量 JSON 会记录 P05/P95、clamp 帧数和 root-Z 连续性，仍需真机前检查。
- 当前实现只做运动学 root-Z 后处理，不等价于动力学接触优化。

### 回退方式

选择其他 pipeline 或直接使用原始 `robot_motion.pkl` 即可绕过；删除后处理产物不会影响原始 GMR 结果。

### 主要涉及文件

- GMR actual：`gmr_web/gvhmr_contact_floor.py`
- GMR actual：`gmr_web/manager.py`
- GMR actual：`gmr_web/external_backend.py`
- GMR actual：`gmr_web/server.py`
- GMR actual：`gmr_web/static_app/index.html`
- GMR actual：`tests/test_gvhmr_contact_floor.py`
- GMR docs：`README.md`、`docs/technical_report.md`
- GMR public wrapper：`gmr_web/gvhmr_contact_floor.py`、manager/server/UI/tests/README

---

## P13：GVHMR-Enhanced 合入 GVHMR Web 主仓库

### 基本信息

- 日期：2026-08-11
- 分支：`gvhmr-web-tool/main`
- 状态：CPU/服务测试已验证，GPU 视频回归沿用合并前结果
- 上游依据或实验基线：core `feature/gvhmr-opt` 提交 `33c6cfd`
- 范围：FootMR、COCO23、性能缓存、场景/地面实验工具与 Web 源码启动接入
- 不包含：Docker 默认切换 FootMR、启用 Web Human3R 选项、提交模型权重和运行产物

### 优化目标

将 `gvhmr-core-opt` 的增强算法直接并入 `gvhmr-web-tool`，使源码 Web 不再依赖同级 core worktree，同时保留切换其他算法 worktree 的能力。

### 关键实现

- 合并 FootMR 脚踝 refiner、COCO17/23 参数化、Sapiens 懒加载、缓存/批量与 transformer 性能选项。
- 合并 Human3R、DINOv2、CoTracker、Sapiens 固定子模块和实验工具；Human3R 仍不在 Web 中启用。
- `start_web_source.sh` 默认将当前仓库设为 `GVHMR_CORE_ROOT`；显式环境变量仍可覆盖。
- external worker 使用增强 `demo` 配置，并把原始视频路径交给时间戳 30 FPS 重采样流程，避免 60 FPS 输入变慢。
- 保留 Web 的 checkpoint root 路由、旋转视频尺寸处理和原始 GVHMR embedded/Docker baseline。
- 自动平地约束使用 `apply_contact_floor_y.py --smoothing-seconds 0.5 --allow-large-correction`，保留 raw tensor。

### 接口、配置与资产变化

- 源码模式默认模型为 FootMR-ViTPose。
- 新增 `inputs/footmr_assets/` 安全下载入口，权重不进入 Git。
- `GVHMR_CORE_ROOT` 从必需的同级 worktree 切换为可选覆盖项。

### 验证方法与结果

- FootMR：17/23 关节参数化、residual refiner、缓存隔离、严格 checkpoint 与 2 帧合成前向通过。
- P6 性能检查和 SimpleVO 串并行回归通过。
- Human3R/地面约束测试 18/18；移除 WHAM 目录并迁移共享统计函数后再次为 18/18。
- Web 服务测试 18/18；增强 worker self-probe 确认导入当前仓库、资产完整且 CUDA 可用。
- Python compileall、Shell/JavaScript 语法、`git diff --check` 均通过。

### 未完成项和已知风险

- Docker 下载与镜像尚未包含 FootMR 资产，因此 Docker 路径继续使用 GVHMR baseline。
- Web 暂不暴露 Sapiens、模型 baseline 切换和 Human3R 选项。
- WHAM 实验工具按最终方案移除，不进入 Web 推理或发布树；此前实验结论仅作为历史记录保留。
- 合并后没有重新占用 GPU 跑完整视频；GPU 质量结论来自合并前相同算法提交。

### 回退方式

源码启动时显式指定旧 core worktree，或使用 Docker/embedded GVHMR baseline；Git 层可回退本次 merge commit。

### 主要涉及文件

- `start_web_source.sh`
- `hmr4d/service/external_core_worker.py`
- `hmr4d/configs/demo.yaml`
- `hmr4d/model/gvhmr/`
- `hmr4d/network/footmr/`
- `tools/demo/demo.py`
- `tools/bench/human3r_p2y/`

---

## P15：Web 输入视频强制按时间戳重采样为 30 FPS

### 基本信息

- 日期：2026-08-11
- 分支：`gvhmr-web-tool/main`
- 状态：`CPU 已验证`
- 上游依据或实验基线：P14 Metric VDA 测试发现 ydd/qhy 旧任务的 `0_input_video.mp4` 被慢放一倍
- 范围：Web embedded/external-core 视频准备路径、旧错误输入识别和 60→30 FPS 回归
- 不包含：重新生成历史 GVHMR tensor、修改模型算法、GPU 完整推理

### 优化目标

保证模型使用的 `0_input_video.mp4` 是按源视频时间戳真正重采样得到的恒定 30 FPS 视频，而不是保留全部源帧后只将容器帧率改成 30。源视频为 60 FPS 时应丢弃约一半帧，同时保持播放时长和动作速度不变。

### 关键实现

- Web embedded runner 不再仅凭分辨率和帧数判断是否复用 `0_input_video.mp4`，统一交给 `normalize_video_fps` 校验目标帧率和源/目标时长。
- 若目标文件是旧版错误产物（30 FPS 标记、帧数未减少、时长翻倍），会自动重新编码。
- external worker 的兼容视频复制函数也改为同一时间戳重采样入口，避免未来重新启用时恢复旧行为。
- FFmpeg 使用 `fps=30` filter 和 H.264 CFR 输出；不会通过单纯修改 metadata 改变播放速度。

### 接口、配置与资产变化

- Web 页面和 API 参数不变；所有新任务自动使用 30 FPS 模型输入。
- 上传原件 `submitted_input.*` 继续保留原始帧率，供审计和重新处理；真正送入 GVHMR 的文件仍为 `0_input_video.mp4`。
- 无新增依赖或模型资产。

### 验证方法与结果

在 `gvhmr` Conda 环境运行：

```bash
PYTHONPATH=/home/user-kevien/gvhmr_pkg/gvhmr-web-tool \
  /home/user-kevien/miniforge3/bin/conda run -n gvhmr \
  python tools/bench/test_p6_optimizations.py
```

合成回归先构造 60 FPS、60 帧、1.000 秒源视频，再构造旧错误形式的 30 FPS、60 帧、2.000 秒 `0_input_video.mp4`。Web 输入准备完成后结果为 30 FPS、30 帧、1.000 秒；测试输出 `P6 optimization checks passed`。

### 未完成项和已知风险

- 已经由旧输入生成的历史 `hmr4d_results.pt` 不会因本修复自动重算；这类任务必须从原始 `submitted_input.*` 新建任务，不能只重新生成预览。
- 变帧率视频会按显示时间戳采样为恒定 30 FPS，个别输出帧可能复制或丢弃，这是保持真实时间轴的预期行为。
- 本项没有重新占用 GPU 跑完整视频，因为修复位于模型前的 CPU/FFmpeg 输入阶段。

### 回退方式

回退本项三个代码文件即可恢复旧逻辑；不建议回退，因为旧逻辑会在 60 FPS 输入上造成动作慢放。

### 主要涉及文件

- `hmr4d/api/video_to_data.py`
- `hmr4d/service/external_core_worker.py`
- `tools/bench/test_p6_optimizations.py`
- `docs/OPTIMIZATION_LOG.md`

---

## P16：Web 自动平地切换至 Contact Global V1.1

### 基本信息

- 日期：2026-08-11
- 分支：`feature/gvhmr-opt`
- 状态：`CPU 已验证`
- 上游依据或实验基线：core-opt P16.1 Contact-aware Global Optimizer V1.1；cxk/ydd/qhy 三视频离线对比
- 范围：Web source-mode 后处理路由、能力检测、任务产物、下载与失败回退
- 不包含：Human3R 启用、重新运行完整视频 GPU 人体推理、旧历史任务批量重算

### 优化目标

用 marker-aware Contact Global V1.1 完全替代新任务中的 local-Y 0.5 秒后处理。自动平地必须从原始 FootMR tensor 单次求解整段 root XYZ；V1.1 失败或保护条件拒绝时直接恢复 raw，不能再串联或回退到旧 local-Y。

### 关键实现

- Web 仓库内置 `apply_contact_global_root.py`，采用 toe/heel 独立 XZ anchor、FootMR 四路连续接触权重、segment fade 以及按真实 `dt` 归一化的一阶/二阶时间项。
- external worker 的 `flat_y` 路由改为调用 Global V1.1。`flat_y` 仅作为现有页面和 API 的兼容值保留，不再表示旧 root-Y 算法。
- 新任务只创建 `ground_constraint_global_v1_1/`，从 `hmr4d_results_raw.pt` 生成候选。只有 `metrics.json` 的 `decision=diagnostic_pass` 时才覆盖当前 `hmr4d_results.pt`。
- 脚本异常、metrics 缺失或 guardrail 失败时直接把 `hmr4d_results_raw.pt` 恢复为主结果。测试中即使 core 同时存在旧 `apply_contact_floor_y.py`，也不会调用它。
- manager、下载接口和 ZIP 增加 Global V1.1 tensor 与 metrics。旧 `ground_constraint_flat_y` 产物只用于历史任务只读发现和归档兼容，不进入任何新任务计算链路。
- capabilities 只有检测到 `apply_contact_global_root.py` 才启用并默认选择“自动平地约束（Global V1.1）”；Human3R 仍显示但不可选。
- V1.1 只修改 `smpl_params_global.transl`；body pose、global orient、betas、incam、相机和 `net_outputs` 保持不变。

### 接口、配置与资产变化

新任务产物：

```text
hmr4d_results_raw.pt
ground_constraint_global_v1_1/
├── contact_global_root_hmr4d_results.pt
├── metrics.json
└── contact_global_root_curves.png
hmr4d_results.pt
```

任务 artifact 新增 `global_contact_results_path`。页面下载项新增“Global V1.1 结果 PT”。没有新增模型权重或 Python 依赖，继续使用 `gvhmr` Conda 环境。

### 验证方法与结果

```bash
PYTHONPATH=/home/user-kevien/gvhmr_pkg/gvhmr-web-tool \
  /home/user-kevien/miniforge3/bin/conda run -n gvhmr \
  python -m unittest tools.bench.human3r_p2y.test_apply_contact_global_root

PYTHONPATH=/home/user-kevien/gvhmr_pkg/gvhmr-web-tool \
  /home/user-kevien/miniforge3/bin/conda run -n gvhmr \
  python -m unittest discover -s tests -p 'test_service_web.py' -v

PYTHONPATH=/home/user-kevien/gvhmr_pkg/gvhmr-web-tool \
  /home/user-kevien/miniforge3/bin/conda run -n gvhmr \
  python tools/bench/test_p6_optimizations.py
```

- V1.1 算法单测 `5/5` 通过，覆盖 toe/heel 路由、连续权重、contact 精炼、全局 solve 和 24/30/60 FPS 一致性。
- Web 服务测试 `20/20` 通过，覆盖新路由成功、失败恢复 raw、旧脚本不调用、capabilities、新 artifact 和 ZIP。
- P6 输入重采样回归通过：60 FPS、1 秒合成输入输出为 30 FPS、30 帧、1 秒。
- `py_compile`、`node --check hmr4d/service/static_app/app.js` 和 `git diff --check` 通过。
- 使用现有 cxk 367 帧 raw tensor 在临时目录调用真实 Web worker：`decision=diagnostic_pass`，主结果逐字节等于新候选，raw 保留，旧目录未创建。
- 该结果和先前 core-opt V1.1 实验 tensor 的 root translation 最大绝对差为 `4.768e-7 m`、P95 为 `1.192e-7 m`，在 `atol=1e-6` 下完全一致；文件 SHA 不同来自 SciPy LSQR 停止迭代的浮点微差，不能用 SHA 作为跨次求解一致性条件。
- 源码 Web 已重启于 `http://127.0.0.1:7860/`，core 为当前 `gvhmr-web-tool`，健康检查显示 `inference_ready=true`，capabilities 默认启用 Global V1.1。

### Web 布局修正

首次接入时，自动平地卡片的 V1.1 长说明由一行变成两行，触发桌面布局原有的 `.process-panel { overflow: auto; }`，导致“视频处理”卡片出现块内纵向滚动。现已将说明缩短为“默认；整段 root XYZ 接触优化”，并把该卡片设为 `overflow: visible`。桌面首行保证至少 390 px、整个内容区至少 740 px；窗口不足时由页面整体滚动，不裁切表单或在视频处理块内滚动。这只恢复界面布局，不改变 `flat_y` 到 Global V1.1 的后端路由、参数或产物。

### 未完成项和已知风险

- 本次没有提交新视频重新运行 FootMR GPU 推理；Web 接入使用已有 cxk raw tensor验证 CPU 后处理及发布逻辑。完整人体模型 GPU 路径未因本项改变。
- 单一固定平地与 FootMR contact 仍是前提。多层箱顶、台阶或错误接触不能由 V1.1 自动识别，guardrail 也不等于场景真值验证。
- 历史任务不会自动重算；其旧 local-Y 文件仍可读取，但新提交或重试后的运行链路只使用 V1.1。

### 回退方式

页面可选“不启用”，直接发布原始 FootMR。算法异常或保护失败也会自动恢复 `hmr4d_results_raw.pt`；不再提供回退到旧 local-Y 的运行分支。

### 主要涉及文件

- `tools/bench/human3r_p2y/apply_contact_global_root.py`
- `tools/bench/human3r_p2y/test_apply_contact_global_root.py`
- `hmr4d/service/external_core_worker.py`
- `hmr4d/service/manager.py`
- `hmr4d/service/server.py`
- `hmr4d/service/static_app/app.js`
- `hmr4d/service/static_app/index.html`
- `tests/test_service_web.py`
- `README.md`
- `README.en.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P17：WebTool 内置 SONIC 适配层，移除 Kimodo 运行时依赖

### 基本信息

- 日期：2026-08-12
- 仓库：`/home/user-kevien/gvhmr_pkg/gvhmr-web-tool`
- 分支：`main`
- 状态：CPU 已验证，五条真实 Web 任务 bit-exact
- 上游依据：此前实验使用的 `kimodo.integrations.sonic`
- 范围：最终 Web `hmr4d_results.pt` 转 SONIC reference、50 FPS SO(3) 重采样、SONIC v4 ZMQ 协议、本地播放和 Web 显式发送按钮
- 不包含：任务完成后自动推流、修改 SONIC/MuJoCo、真机运行、控制策略优化

### 优化目标

把已经合入正式 GVHMR WebTool 的人体结果直接转换并发送给 SONIC，不再启动废弃的 `gvhmr-core-opt` 或 Kimodo 环境，同时保证 SONIC 收到的 reference 与之前实际验证过的 Kimodo 链路完全一致。

### 关键实现

- 新增 `hmr4d/utils/sonic.py`，保留原 Apache-2.0/NVIDIA 版权声明和旧实现的坐标变换、SMPL-X FK、30→50 FPS Slerp、腕部计算、10 帧 look-ahead、v4 binary message 与播放生命周期。
- 不复制 Kimodo 的 137 MB `SMPLX_NEUTRAL.npz`，只嵌入旧实现实际读取的 `55×3 float32 J` 与 55 个 parent，约 1.8 KB。保持 float32 是 bit-exact 的必要条件；使用原始 float64 会令 `term1_local` 出现最大一个 float32 ULP 的差异。
- 新增通用 CLI：`tools/sonic/convert_gvhmr.py` 接受任意 Web 任务最终发布的 `hmr4d_results.pt`；`tools/sonic/play_reference.py` 接受任意 reference 文件，不再绑定实验动作名。
- 转换只读取 Web 结果并写入显式指定的目标路径，不覆盖任务 tensor、不修改 job.json、不自动连接 SONIC。
- `requirements.txt` 明确加入 SciPy 与 PyZMQ；运行统一使用 `gvhmr` Conda 环境。

### 验证方法与结果

在 WebTool 本仓库用当前 jntm/lly/cxk/qhy/ydd 正式任务的 `hmr4d_results.pt` 重新生成 SONIC reference，并与之前实际播放使用的 Kimodo reference 比较：

| 动作 | Web 帧数 | SONIC 帧数 | term1_local | root_quat | wrist | fps |
| --- | ---: | ---: | --- | --- | --- | --- |
| jntm | 540 | 899 | bit-exact | bit-exact | bit-exact | bit-exact |
| lly | 469 | 781 | bit-exact | bit-exact | bit-exact | bit-exact |
| cxk | 367 | 611 | bit-exact | bit-exact | bit-exact | bit-exact |
| qhy | 1507 | 2511 | bit-exact | bit-exact | bit-exact | bit-exact |
| ydd | 1979 | 3298 | bit-exact | bit-exact | bit-exact | bit-exact |

四组数组在五条动作上的最大绝对差均为 `0`。此外，输入校验、转换 shape、look-ahead/protocol header、假 socket 播放和暂停后不再发包 4 项测试通过；Web 服务测试 `23/23` 通过，其中覆盖按钮 API、首次转换、源 SHA 缓存复用、状态回调、暂停与重复暂停保护、独立下载和 ZIP 收录。30 FPS 输入重采样回归、`py_compile`、`node --check` 与 `git diff --check` 均通过，测试未连接真实 SONIC。当前运行环境为 NumPy 2.2.6；另在隔离临时环境用 requirements 声明的 NumPy 1.23.5 + SciPy 1.15.3 复算 cxk，NPZ SHA256 与四组数组仍完全一致，排除当前机器依赖漂移造成的偶然通过。

正式源码 Web 已通过 `start_web_source.sh` 重启于 `http://127.0.0.1:7860/`，capabilities 显示 `sonic_bridge_available=true`、`inference_ready=true`，core 仍为当前 `gvhmr-web-tool`，自动地面约束仍为 Global V1.1。机器原有 PyZMQ 只位于用户 site-packages，而服务使用 `PYTHONNOUSERSITE=1`，因此另将 `pyzmq 27.1.0` 安装进 `gvhmr` 环境并验证隔离导入成功。

隔离验证结果位于：

```text
/home/user-kevien/gvhmr_pkg/experiments/sonic_webtool_migration_20260812/
├── bit_exact_report.json
└── new/{jntm,lly,cxk,qhy,ydd}/
    ├── sonic_reference.npz
    └── conversion.json
```

### 使用方式

成功任务的详情操作栏会显示“发送到 SONIC”。点击后 Web 从最终发布的 `hmr4d_results.pt` 生成或复用 `sonic_reference.npz`，并由后台线程推流；HTTP 请求不会等待整个动作播放完成。同一 Web 服务只维护一个 publisher，再次发送会停止并替换当前动作。页面复用原 `.detail-actions` 操作栏，没有增加面板、页面高度或块内滚动。

推流处于 `preparing/streaming` 时显示“暂停 SONIC”。暂停会停止 publisher 并停止重复最后一帧；当前 SONIC policy 在 live reference 超过 `0.5s` 未更新后丢弃它，再用 `0.4s` 平滑切换到自身 `stream_reference.npz` 的 idle reference。这里没有发送全零 SMPL/T-pose，也没有直接调用机器人 reset，因此“默认姿态”严格指 SONIC policy 的内置 idle/default reference。暂停后任务记录为 `paused`，仍可再次发送。

### 自动平地回退原因展示

Global V1.1 的 `metrics.json` 新增 `failed_guardrails` 和 `guardrail_details`，不仅记录布尔结果，也记录水平/垂直修正、root 步长、root 加速度的实际值与阈值，以及脚滑、接触段漂移、支撑高度、悬空和穿地的优化前后数值。Web worker 将失败项转换为中文 `ground_constraint_fallback_reason`；任务详情的“错误”和操作提示会直接显示该原因，任务 JSON 同时保留结构化字段。异常、指标缺失或损坏仍显示对应运行原因；历史 metrics 没有数值明细时至少显示失败的 guardrail 名称。没有新增卡片或滚动区域。

任务会记录 `preparing / streaming / complete / stopped / error` 状态和帧进度，生成的 reference 与转换 metadata 可单独下载，也会重建进任务 ZIP。推流进度约每 0.5 秒写入一次 SQLite，终态一定保存；服务重启会把未完成的推流标记为 error。

CLI 仍可独立使用：

```bash
conda run -n gvhmr python tools/sonic/convert_gvhmr.py \
  runtime/jobs/<任务>/hmr4d_results.pt \
  runtime/jobs/<任务>/sonic_reference.npz \
  --metadata runtime/jobs/<任务>/sonic_conversion.json

conda run -n gvhmr python tools/sonic/play_reference.py \
  runtime/jobs/<任务>/sonic_reference.npz
```

### 未完成项和已知风险

- 当前 SONIC v4 使用 ZMQ PUB，没有 ACK。`complete` 只表示 Web publisher 已发送完整段，不能证明 SONIC subscriber 实际收到；用户仍须先启动 SONIC/MuJoCo。
- bit-exact 证明输入 SONIC 的 reference 和协议不变，不证明 SONIC 对 ELF3 的跟踪质量。此前踝关节目标越限问题仍存在，不得直接用于真机。
- 暂停依赖当前 SONIC 配置 `require_live_reference=false`、`live_reference_timeout_s=0.5` 和 `source_blend_seconds=0.4`；如果未来 SONIC 改为强制 live reference，停止推流将不再等价于回默认姿态，需同步调整协议。

### 回退方式

删除本地适配模块和 `tools/sonic` 即可回退；Web 推理、地面约束、任务结果和页面均未依赖该入口，也没有旧结果被覆盖。

### 主要涉及文件

- `hmr4d/utils/sonic.py`
- `hmr4d/service/manager.py`
- `hmr4d/service/server.py`
- `hmr4d/service/store.py`
- `hmr4d/service/static_app/index.html`
- `hmr4d/service/static_app/app.js`
- `tools/sonic/convert_gvhmr.py`
- `tools/sonic/play_reference.py`
- `tools/sonic/test_sonic.py`
- `requirements.txt`
- `tests/test_service_web.py`
- `README.md`
- `README.en.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P24：长视频预览 SMPL-X 分批生成

### 基本信息

- 日期：2026-08-13
- 分支：`main`
- 状态：GPU 验证完成，Web 预览已生成
- 上游依据或实验基线：`xbd` Web job，7162 帧/30 FPS；原预览在 `render_incam` 一次性 LBS 时 OOM
- 范围：incam/global 预览的 SMPL-X→SMPL 顶点生成内存上界
- 不包含：降低渲染分辨率、跳帧、改变 renderer、改变 SMPL/GVHMR 数值结果

### 优化目标

修复长视频在预览开始时将整段 SMPL-X 参数一次性送入 GPU，导致 12 GB RTX 3060 OOM 的问题；保持原帧数、分辨率、视角和编码逻辑。

### 关键实现

- `smplx_to_smpl_vertices_batched()` 默认按 128 帧执行 SMPL-X LBS 和稀疏 SMPL-X→SMPL 顶点转换，逐批移回 CPU 后拼接。
- incam 渲染继续逐帧把顶点送入原 Renderer；global 的 J regressor、对齐和相机轨迹计算留在 CPU，实际渲染时只上传当前帧。
- 参数序列长度不一致、空序列或非法 batch size 会明确失败，不静默截短。

### 验证方法与结果

- 用 xbd 的真实前 5 帧比较原整批路径和新 2 帧小批路径：输出 `[5,6890,3]`，全部 finite；最大绝对误差 `7.15e-7 m`（约 `0.0007 mm`），为 CUDA 批大小的浮点累加差异。
- 原失败位置需要额外申请 `1.12 GiB` 并 OOM；修复后 7162 帧顶点生成和渲染成功，worker 渲染期间显存约 `0.74 GiB`。
- Web job `job_8565848b9d88` 的 preview 状态由 failed 变为 succeeded，错误清空。
- incam、global、并排视频均为 7162 帧、30 FPS、238.734 秒；分辨率分别为 `1160×650`、`1160×650`、`2320×650`。

### 未完成项和已知风险

- 顶点 CPU 缓存仍随视频长度线性增长；7162 帧约在本机内存范围内，更长视频可进一步改为顶点磁盘分块或边生成边渲染。
- 两个视角仍各渲染一次完整视频，修复内存但不降低约 24 分钟的预览耗时。

### 回退方式

恢复 incam/global 中原来一次性 `smplx(**to_cuda(params))` 的两处调用即可；但长视频会重新出现 OOM。

### 主要涉及文件

- `tools/demo/demo.py`
- `docs/OPTIMIZATION_LOG.md`

---

## P25：关闭自动平地的 raw 结果回退

### 基本信息

- 日期：2026-08-13
- 分支：`main`
- 状态：`CPU 已验证`
- 范围：Web 自动平地 Global V1.1 的结果选择和失败语义
- 不包含：修改 Global V1.1 求解器、阈值或历史任务产物

### 变更内容

- `diagnostic_pass` 与 `guardrail_failed` 只要都生成了有效候选 PT 和可解析 metrics，均采用 Global V1.1 候选作为主 `hmr4d_results.pt`。
- guardrail 不通过不再恢复 `hmr4d_results_raw.pt`，而是记录 `ground_constraint_warning`，在任务详情中显示诊断提示。
- 后处理脚本异常、候选缺失、metrics 缺失/损坏或 decision 非法时直接使 Web 任务失败，并明确提示 raw fallback 已关闭。
- `hmr4d_results_raw.pt` 继续保留用于下载和人工对比，但不再自动成为成功任务的发布结果。
- 历史任务不自动改写，既有 `fallback` 状态和文件保持原样。

### 验证方法与结果

在 `gvhmr` 环境运行 Web 服务回归测试，覆盖正常采用、guardrail 警告仍采用候选、脚本异常使任务失败且不调用旧 local-Y，以及任务字段持久化。另执行 Python/JavaScript 语法检查与 `git diff --check`。

### 风险与恢复方式

关闭保护回退意味着过大的 XYZ 修正也会被发布；`ground_constraint_warning` 和 metrics 必须用于人工判断。页面仍可选择“不启用”来完全绕过地面后处理。若要恢复旧行为，可恢复 worker 中只接受 `diagnostic_pass` 并复制 raw 的分支。

### 主要涉及文件

- `hmr4d/service/external_core_worker.py`
- `hmr4d/service/manager.py`
- `hmr4d/service/store.py`
- `hmr4d/service/static_app/app.js`
- `tests/test_service_web.py`
- `docs/OPTIMIZATION_LOG.md`

---

## P26：GVHMR Web → SONIC 真机安全启动文档

### 基本信息

- 日期：2026-08-13
- 分支：`main`
- 状态：文档完成，待真机验证
- 上游依据或实验基线：`/home/user-kevien/kimodo/KIMODO_SONIC_REAL_ROBOT_QUICKSTART.md`
- 范围：GVHMR Web 内置 SONIC publisher 通过 5558 隔离隧道接入 ELF3 真机
- 不包含：修改真机控制器、SONIC policy、`raindrop` overlay 或 Web 推流代码

### 优化目标

把已在 Kimodo 中验证的真机启动、链路隔离、状态切换和急停要求改写为 GVHMR Web 可直接执行的流程，避免再启动 Kimodo 或误占官方 PICO/5557 链路。

### 关键实现

- 明确 Web 本机 5557 → SSH reverse tunnel → 机器人 5558 → `raindrop` SONIC policy 的数据路径。
- 区分 Kimodo 长期占用 5557 和 Web 仅播放期间绑定 5557 的行为：推流前只要求机器人 5558 `LISTEN`，推流过程再验证 `ESTAB`。
- 保留已部署 overlay 所需的 `SONIC_REFERENCE_MODE=kimodo`，并解释它只是隔离模式的历史名称。
- 写入单一硬件控制器、单一 keyboard publisher、ROS Domain 31、Fast DDS root 用户一致性和逐级状态切换检查。
- 区分 Web“暂停 SONIC”与实体急停，加入首次仅限短、慢、小幅动作的验收边界。
- 纳入本地仿真实测的 `SCHED_FIFO/99` 调度检查，但不将仿真结果当作真机验收。

### 验证方法与结果

- 核对 Kimodo 真机 Quickstart 中的机器人 IP、工作区 source 顺序、Domain 31、5558 隧道、真机 launch 和状态键。
- 核对 Web 当前“发送到 SONIC / 暂停 SONIC”语义及 50 FPS reference 输出。
- 本次仅修改 Markdown，未连接或驱动真机，因此真机仍需按文档的首次记录项验收。

### 未完成项和已知风险

- ZMQ PUB 无 ACK，Web 推流完成不等于真机完整执行。
- `SONIC_REFERENCE_MODE=kimodo` 属于机器人端现有部署约定，后续如改名需同步 overlay 和文档。
- 真机的实时调度、关节限位、自碰和支撑稳定性尚未在本次修改中实测。

### 回退方式

删除真机 Quickstart 和 README 导航项即可；本次没有代码或部署状态需要回退。

### 主要涉及文件

- `docs/SONIC_REAL_ROBOT_QUICKSTART.md`
- `README.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P27：Web SONIC 动作速度倍率

### 基本信息

- 日期：2026-08-13
- 分支：`main`
- 状态：CPU 已验证，待 MuJoCo/真机人工确认
- 上游依据或实验基线：新版 qhy 正常速度与旧 qhy 意外半速 SONIC reference 对比
- 范围：Web 任务详情中的 SONIC 输入端动作速度和倍率隔离缓存
- 不包含：修改 GVHMR 推理帧率、SONIC 50 Hz 控制频率、SONIC policy 或机器人控制器

### 优化目标

当正常速度舞蹈超过机器人腿部跟踪带宽时，允许只降低发送给 SONIC 的动作速度，不再通过错误解释视频 FPS 获得偶然的慢放效果，也不修改 GVHMR 结果和预览。

### 关键实现

- 任务详情操作栏增加“SONIC 速度”按钮，在 `1.0× → 0.75× → 0.5×` 三档循环，默认 `1.0×`。
- API `POST /jobs/{job_id}/to-sonic?speed=<倍率>` 仅接受上述三档，其他值明确返回错误。
- 倍率作用于源动作时间轴：`0.75×` 使用等效 22.5 FPS 源时间，`0.5×` 使用等效 15 FPS 源时间，再通过现有旋转 SLERP 输出固定 50 FPS reference。
- `1.0×` 继续使用 `sonic_reference.npz`；慢速档分别使用带倍率的 NPZ/JSON 文件，并同时校验源 PT SHA256 和等效源 FPS，避免缓存混用。
- 当前播放倍率写入任务状态、提示文本和 conversion metadata；重新发送会安全替换当前 Web publisher。

### 接口、配置与资产变化

- 新增可选查询参数 `speed`，默认 `1.0`，保持旧 API 调用兼容。
- 新增任务字段 `sonic_speed`，历史任务缺失时按 `1.0` 展示。
- 慢速缓存示例：`sonic_reference_speed_0_75.npz`、`sonic_conversion_speed_0_75.json`。
- SONIC 输出频率始终为 50 FPS；倍率不会改写 `hmr4d_results.pt`。

### 验证方法与结果

- `python -m unittest discover -s tests -p 'test_service_web.py' -v`：25 项全部通过。
- `tools.sonic.test_sonic`：4 项通过，覆盖协议窗口、转换、播放和暂停。
- JavaScript `node --check` 通过。
- qhy `0.75×` 实际生成 1674 帧、50 FPS、33.46 秒；原 `1.0×` 为 1256 帧、50 FPS、25.1 秒。
- qhy 平均关节变化速率由 `7.61` 降至 `5.73`，P95 由 `15.33` 降至 `11.49`，约为正常速度的 75%。

### 未完成项和已知风险

- 降速能减轻跟踪带宽压力，但不能从算法上消除机器人足端滑动；最终仍需 MuJoCo 与真机人工确认。
- 当前倍率是页面会话选择，不会在不同浏览器或服务重启后保留为默认值；实际播放倍率会写入任务记录。
- 仅提供不高于正常速度的三档，避免误操作加速高动态动作。

### 回退方式

移除速度按钮和 API 查询参数，并恢复 `send_to_sonic(job_id)` 固定按 30→50 FPS 转换即可；已有慢速缓存不会影响默认 reference。

### 主要涉及文件

- `hmr4d/service/server.py`
- `hmr4d/service/manager.py`
- `hmr4d/service/store.py`
- `hmr4d/service/static_app/index.html`
- `hmr4d/service/static_app/app.js`
- `tests/test_service_web.py`
- `README.md`
- `docs/SONIC_REAL_ROBOT_QUICKSTART.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P28：长视频精确推理、内存上界与非模型产物加速

### 基本信息

- 日期：2026-08-13
- 分支：`main`
- 状态：GPU/CPU 回归完成，源码 Web 已接入
- 上游依据或实验基线：xbd Web job `job_8565848b9d88`，7162 帧/30 FPS/238.7 秒
- 范围：长视频模型 OOM、预处理内存峰值、官方 postproc 复杂度、极端长度续跑、ZIP 和预览合并
- 不包含：降低 ViTPose/HMR2 精度、跳帧推理、重新生成 xbd、缩短两个完整视角本身的逐帧渲染

### 优化目标

旧 xbd 需要 15 个独立 600 帧窗口、480 帧步长和 120 帧 overlap，实际重复推理约 26%，并重复加载模型、检测、ViTPose、HMR2 特征和视频编码。目标是优先保持整段 GVHMR/FootMR 数值语义，避免长序列注意力 OOM，并把分窗降为极端情况的自动兜底。

### 关键实现

- ≥1800 帧默认把 GVHMR 与 FootMR 的 attention 从 dense masked 实现切到局部实现。网络原本对超过 120 帧的序列就只允许每个 query 访问局部 120 帧；新实现不再分配 `L×L` score/mask，数学含义不变。
- 整段仍统一计算 sequence betas、global root rollout、FootMR ankle residual 和官方 contact/IK postproc，不发生窗口拼接。短视频继续走原 dense 整段路径。
- 静态相机 postproc 的两个“每帧修改所有未来帧”循环改成等价累计修正：相机修正保留原递推阈值，接触位移用 cumulative sum，复杂度从 O(L²) 降为 O(L)。
- ViTPose 与 HMR2 同时未命中缓存时继续共享相同人体 crop；≥1800 帧改为逐帧解码并写任务目录 float32 mmap，避免同时常驻全视频 RGB 和数 GB crop tensor。两模型完成后清理 mmap；中途失败则保留用于重试。
- 若极端序列仍在整段 FK/IK 触发 CUDA OOM，自动切换共享预处理、单模型常驻的 600/480 网络窗口。窗口按视频 SHA、有效 FootMR checkpoint SHA 和配置隔离，逐窗口原子缓存，可断点续跑；动态 overlap、root yaw/translation 对齐和 rotation SLERP 用于拼接。
- 长视频模式、模型身份、耗时、是否发生窗口 fallback 写入 `long_video/manifest.json` 与 `metrics.json`，并进入 Web artifact 和 ZIP。
- ZIP 对 MP4/PT/NPZ 等已经压缩或大二进制使用 `ZIP_STORED`，JSON/文本仍 Deflate；并排预览的二次编码显式使用 `libx264 veryfast`。二者不改变模型输入或 SMPL tensor。
- 输入视频规范化曾实验 `veryfast`，但因重编码像素变化导致 qhy SMPL 有可测差异，已经撤回，继续使用原输入编码逻辑。

### 验证方法与结果

- 单元/服务：`35/35` 通过，覆盖 dense/local 等价、7162 窗口覆盖、旋转 SLERP、gap/triple overlap 拒绝、缓存身份/续跑、流式 crop 与原内存 crop bit-identical、两个累计 postproc 与旧 suffix 循环等价、ZIP 压缩类型与内容一致。
- qhy 754 帧真实特征 A/B：原 dense 和整段 local 的整个结果树逐 tensor `torch.equal`，incam/global、net_outputs 均逐 bit 一致。local 主网络+FootMR+官方 postproc 为 `0.91 s`，CUDA peak allocated `253 MB`。
- 7162 帧合成真实特征序列：整段 local 主网络+FootMR+root rollout+官方 postproc `2.83 s`，CUDA peak allocated `475.6 MB`；incam/global 全部 finite 且 shape 正确。因此 xbd 长度在 12 GB GPU 上不需要默认分窗。
- 真实旧 qhy raw 与新 O(L) postproc 对比：incam 完全相等；global 最大 root 差 `1.03e-6 m`、最大 pose 差 `0.000203°`，来自累计加法顺序的浮点误差。
- 分窗安全性实验仍保留：qhy 600/480 两窗的 incam body P95 相对整段为 `0.024°`、root translation P95 为 `0.48 mm`，窗口连接处姿态/root step 不高于原序列 P95；但整段重跑非线性 IK 会放大 global 脚踝差异，所以分窗只作为 OOM fallback，不作为常规输出。
- 输入 `veryfast` 实验：qhy 60→30 FPS 保持 754 帧和 25.13 秒，但相对原编码结果 body P95 约 `1.3°`、global root P95 约 `2.3 cm`。该项未采用。
- 非模型产物实测：xbd 的 PT+两个 MP4+JSON 共 135.3 MB 使用直存 ZIP 为 `4.81 s`；qhy 3840×1080、754 帧并排 veryfast 合并为 `13.72 s`，输出仍为 754 帧、30 FPS、25.13 秒。
- P6 FPS/缓存检查通过；Python compileall、JavaScript syntax、`git diff --check` 均通过。

### 性能预期与边界

- 旧 xbd 分块实际处理约 9000 帧；新默认视觉预处理只处理 7162 帧一次，消除约 26% overlap 重复、15 次模型加载和15次分块视频编码。精确总加速仍需下一条真实长视频完成后以 Web metrics 记录，不能由 qhy 外推成固定倍数。
- xbd 历史 15 个窗口产物目录已被清理，只保留最终 7162 帧 tensor 和 metrics，因此没有伪造新旧窗口逐文件 A/B，也没有覆盖历史任务。
- 流式 mmap 降低 RAM 峰值，但会暂用磁盘：float32 crop 每帧约 0.75 MiB，7162 帧约 5.25 GiB；成功后自动删除。任务中断后保留是为了续跑，必要时可人工清理该任务 `preprocess/shared_crops_float32.mmap*`。
- 当前推理结果的精确路径无法跨进程从头续跑视觉模型内部 batch；已有 bbox/pose/features 内容缓存以及极端 OOM fallback 窗口缓存可以续跑。完整“ViTPose/HMR2 每批落盘续跑”仍是后续项。

### 回退方式

将 external worker 的长视频 `cfg.attention_impl/network.attention_impl` 恢复为 dense 即回到旧整段行为，但会重新出现长序列 OOM；也可提高 `GVHMR_LONG_VIDEO_THRESHOLD_FRAMES` 暂时关闭自动 local。删除 `long_video.py` 路由可取消 OOM 分窗兜底，不影响短视频。

### 主要涉及文件

- `hmr4d/service/external_core_worker.py`
- `hmr4d/utils/long_video.py`
- `hmr4d/network/base_arch/transformer/encoder_rope.py`
- `hmr4d/model/gvhmr/utils/postprocess.py`
- `hmr4d/utils/preproc/vitfeat_extractor.py`
- `tools/demo/demo.py`
- `hmr4d/service/common.py`
- `hmr4d/service/manager.py`
- `hmr4d/utils/video_io_utils.py`
- `tests/test_long_video.py`
- `tests/test_service_web.py`

---

## P29：Web 奇数分辨率输入规范化修复

### 基本信息

- 日期：2026-08-13
- 分支：`main`
- 状态：CPU 已验证，真实任务已重试
- 实验基线：Web job `job_282c4d8e3105`，`新宝岛.webm`，VP8、1157×652、87.93 秒
- 范围：Web 输入转恒定 30 FPS MP4 时的 H.264 尺寸兼容与错误诊断
- 不包含：GVHMR/FootMR 模型逻辑、长视频 attention、视觉特征或地面约束算法

### 优化目标

修复宽或高为奇数的 Web 视频在模型运行前规范化失败的问题。原任务的 1157×652 输入可完整解码，但 `libx264 + yuv420p` 要求两个维度均为偶数，FFmpeg 因此报错 `width not divisible by 2`。

### 关键实现

- 规范化前读取经过显示方向旋转后的宽高，仅在任一维度为奇数时增加 FFmpeg `pad`。
- 宽高分别使用 `ceil(iw/2)*2` 和 `ceil(ih/2)*2`，只在右侧或底部补最多一个黑色像素；不缩放、不裁剪原画面。
- 偶数尺寸输入继续使用原编码路径；30 FPS 时间轴、编码参数和模型输入语义不变。
- 捕获 FFmpeg stderr 并写入任务异常，同时删除失败的临时输出，避免页面只显示缺少根因的 `ffmpeg error`。

### 接口、配置与资产变化

- Web/API 参数不变，用户不需要手工转码或重新上传。
- `新宝岛.webm` 的模型输入由 1157×652 自动规范化为 1158×652；时长保持约 87.93 秒。
- 没有新增依赖、权重或缓存格式。

### 验证方法与结果

- 新增 1157×652、60 FPS、1 秒的 FFV1/yuv444p 合成输入回归，规范化结果为 1158×652、30 FPS、30 帧、1 秒。
- `python -m unittest tests.test_long_video -v`：10/10 通过。
- `python -m unittest tests.test_service_web -v`：26/26 通过。
- `git diff --check` 通过。
- 原失败任务直接复用已上传的 WebM 重试；真实任务是否完整推理成功需以 Web job 最终状态为准。

### 未完成项和已知风险

- 补边会使奇数维度增加一个像素，但不会改变已有像素或视频时间轴；相机内参按实际规范化后尺寸计算。
- 本项只解决输入规范化失败，不改变长视频所需的 GPU 时间和显存。

### 回退方式

移除奇数尺寸检测和 `pad` filter 即恢复旧行为，但奇数宽高输入会再次在 H.264 编码阶段失败。

### 主要涉及文件

- `hmr4d/utils/video_io_utils.py`
- `tests/test_long_video.py`
- `README.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P30：Web SONIC 连续速度滑块

### 基本信息

- 日期：2026-08-14
- 分支：`main`
- 状态：CPU 已验证，待 MuJoCo/真机人工确认
- 上游依据或实验基线：P27 的 `1.0× / 0.75× / 0.5×` 三档循环按钮
- 范围：Web SONIC 播放速度选择、API 校验、倍率缓存和真机说明
- 不包含：修改 GVHMR tensor、SONIC 50 Hz 控制频率、尖峰过滤、机器人安全约束或控制策略

### 优化目标

将三档循环按钮改为可直接拖动的速度滑块，允许在真机或 MuJoCo 验证时小步调整动作速度，不必在相差较大的固定档位之间跳转。

### 关键实现

- 任务详情增加 `0.25×～1.00×` 横向滑块，步长 `0.05×`，当前值固定显示两位小数。
- 拖动只改变下一次发送所用的倍率，不会自动连接或推流；仍需显式点击“发送到 SONIC”。
- API 同步接受 16 个离散档位，拒绝非有限值、越界值和不在 `0.05×` 网格上的值。
- 输出仍统一为 50 FPS，通过 SO(3) SLERP 改变动作时间轴，不改写人体结果或预览。
- 每个倍率继续使用隔离的 NPZ/JSON 缓存；倍率字符串规范化后，已有 `0.5×`、`0.75×` 和 `1.0×` 缓存命名保持兼容。
- 真机文档改为首次 `0.5×` 或更低，并按 `0.05×` 小步提高。

### 接口、配置与资产变化

- `POST /jobs/{job_id}/to-sonic?speed=<倍率>` 的范围由三档扩展为 `0.25～1.00`，步长 `0.05`；省略参数仍为 `1.0`。
- 示例新增缓存：`sonic_reference_speed_0_65.npz` 与 `sonic_conversion_speed_0_65.json`。
- 没有新增依赖或模型资产。

### 验证方法与结果

- `python -m unittest discover -s tests -p 'test_service_web.py' -v`：27/27 通过，包含静态页面滑块范围检查。
- `python -m unittest tools.sonic.test_sonic -v`：4/4 通过。
- 服务测试覆盖 `0.65×` API 路由、50 FPS 输出、时长变化、独立缓存首次生成与复用。
- 当前 cxk 真实 PT 离线转换 `0.65×` 得到 939 帧、50 FPS、18.76 秒，三组数组均为有限值；该检查没有打开 ZMQ publisher。
- `0.81×`、`0.2×` 和 `1.05×` 均返回明确的 400 错误。
- `node --check hmr4d/service/static_app/app.js`、Python compile 和 `git diff --check` 通过。
- 本项没有连接 SONIC、MuJoCo 或真机；动态跟踪效果必须单独人工确认。

### 未完成项和已知风险

- 更慢只降低时间变化率，不能修复 GVHMR 单帧姿态尖峰，也不能保证机器人稳定。
- 默认值仍为 `1.00×`，首次真机必须按照 Quickstart 主动降低并使用可靠支撑。
- 当前没有基于 root/关节速度的自动真机安全拦截，结构合法但不可执行的动作仍可能被发送。

### 回退方式

恢复三档按钮及后端 `(0.5, 0.75, 1.0)` 白名单即可；新生成的其他倍率缓存不会影响旧默认 reference。

### 主要涉及文件

- `hmr4d/service/manager.py`
- `hmr4d/service/static_app/index.html`
- `hmr4d/service/static_app/app.js`
- `hmr4d/service/static_app/styles.css`
- `tests/test_service_web.py`
- `README.md`
- `README.en.md`
- `docs/SONIC_REAL_ROBOT_QUICKSTART.md`
- `docs/OPTIMIZATION_LOG.md`

---

## P31：Web Studio 视觉与任务工作区重构

### 基本信息

- 日期：2026-08-14
- 分支：`main`
- 状态：CPU 与浏览器布局已验证
- 上游依据或实验基线：P30 Web 功能与用户提供的界面参考图
- 范围：Web 静态页面视觉、上传交互、任务列表和任务详情布局
- 不包含：修改 API、任务数据库、推理模型、地面约束、预览生成或 SONIC 数据链路

### 优化目标

在保留现有处理能力的前提下，将旧的高密度控制台改成更清晰的浅色 Studio 工作区；桌面端恢复固定一屏工作台和可滚动任务队列，使上传、任务状态与结果预览无需滚动整个页面即可同时看到。

### 关键实现

- 页面改为浅灰背景、白色圆角卡片、蓝色主操作和轻量阴影；桌面端保持“视频处理 / 任务队列”双栏，任务详情位于下方。
- 上传区增加云上传图标、拖放反馈和真实文件拖放处理；点击文件选择与批量上传原逻辑保持不变。
- 地面约束使用三张独立选项卡展示；Human3R 保留“新”标签和禁用状态，不改变后端能力声明。
- 任务条目增加视频占位缩略图、名称、job ID、时间、状态胶囊和进度条；所有匹配任务保留在队列中，队列在固定高度内独立纵向滚动并自动保持选中项可见。
- 桌面端页面固定为一屏，上半区为视频处理与任务队列，下半区为任务详情；详情按参考图使用左侧指标/操作、右侧大预览，下载、路径、日志和 JSON 放在底部折叠栏。
- 1440×800 及更大桌面视口使用固定工作台；宽度不超过 820px 或高度不超过 760px 时回退单栏文档滚动，避免小屏强行压缩导致控件不可操作。
- CSS/JavaScript 静态资源版本更新为 `20260814-studio-workspace-v2`，避免浏览器继续使用旧界面缓存。

### 接口、配置与资产变化

- Web API、表单字段 value、任务 schema、任务产物与运行目录均未改变。
- SONIC 发送、暂停和 `0.25×～1.00×` 速度滑块仍沿用 P30 实现。
- 没有新增前端框架、第三方依赖、模型资产或运行时服务。

### 验证方法与结果

- Headless Chrome 在 1440×800、1440×900 和 1920×1080 下完成实际页面截图检查；页面本身无横向或纵向溢出，视频处理、任务队列和完整任务详情均在一屏内。
- 1440×800 DOM 尺寸检查中，`body.scrollHeight == body.clientHeight == 800`；80 个任务的队列为 `scrollHeight > clientHeight`，确认只有任务队列内部滚动。
- `python -m unittest discover -s tests -p 'test_service_web.py' -v`：27/27 通过。
- `python -m unittest tools.sonic.test_sonic -v`：4/4 通过。
- `node --check hmr4d/service/static_app/app.js` 与 `git diff --check` 通过。
- `gvhmr-web-opt.service` 重启后 `/api/capabilities` 正常返回，推理、自动平地和 SONIC 能力均保持可用。

### 未完成项和已知风险

- Headless Chrome 只能确认页面布局与静态交互结构，已有任务的实际视频解码和人工视觉偏好仍应在常用浏览器中确认。
- 固定一屏布局以桌面工作台为目标；低于 821px 宽或 761px 高时会回退页面滚动，以保证小屏控件仍可用。

### 回退方式

回退本项三个静态页面文件即可恢复旧界面；后端数据和任务产物无需迁移。

### 主要涉及文件

- `hmr4d/service/static_app/index.html`
- `hmr4d/service/static_app/app.js`
- `hmr4d/service/static_app/styles.css`
- `README.md`
- `docs/OPTIMIZATION_LOG.md`

---

## 后续优化记录模板

复制以下小节并追加到本文档，不能覆盖历史记录。

```markdown
## P<N>：优化名称

### 基本信息

- 日期：YYYY-MM-DD
- 分支：
- 状态：实现中 / CPU 已验证 / GPU 已验证 / 实验完成（未采用） / 已回退
- 上游依据或实验基线：
- 范围：
- 不包含：

### 优化目标

### 关键实现

### 接口、配置与资产变化

### 验证方法与结果

### 未完成项和已知风险

### 回退方式

### 主要涉及文件
```
