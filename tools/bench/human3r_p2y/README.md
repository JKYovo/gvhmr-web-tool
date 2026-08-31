# Human3R P2-Y 离线实验

该目录包含 Human3R 场景实验和 Web 使用的 headless 推理、地面提取及重力校正脚本。Web 的 `human3r` 模式使用场景地面法向校正 GVHMR 重力，再执行 Contact Global V1.1；旧 P2-Y/P2-XYZ 脚本仍作为离线 benchmark 保留。

## 固定版本与环境

- Human3R：`402f2b2c7f20514e99cb42e4126c46b4ff75593f`
- DINOv2：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- 独立环境：`/home/user-kevien/miniforge3/envs/human3r`
- Python 3.11、PyTorch 2.4.1、TorchVision 0.19.1、CUDA 12.4
- NumPy 1.26.4、Transformers 4.46.3、Viser 0.2.23、gsplat 1.5.3、Open3D 0.19.0

本机已完成的安装流程如下。Human3R 与 GVHMR 依赖不共用环境。

```bash
conda create -n human3r python=3.11 cmake
conda install -n human3r pytorch=2.4.1 torchvision=0.19.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda run -n human3r pip install -r third-party/Human3R/requirements.txt
conda install -n human3r 'llvm-openmp<16'
conda run -n human3r pip install gsplat==1.5.3 open3d==0.19.0

cd third-party/Human3R/src/croco/models/curope
conda run -n human3r python setup.py build_ext --inplace
```

最后应运行 `conda run -n human3r pip check`。本机环境该检查为通过状态。

## 资产

实验使用：

```text
inputs/human3r_assets/human3r_672S.pth
SHA256 84d2a70386473b58b90eef8f78521065ad10908bab647ee58d1196f5018fb778
```

当前 headless 推理通过参数复用 WebTool 已有的 `runtime/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` 和 `hmr4d/network/hmr2/configs/smpl_mean_params.npz`，不会向 Human3R 子模块复制人体资产。训练/评测脚本额外需要的 SMPL、joint regressor 和 `smplx2smpl.pkl` 不属于 Web 推理依赖。

`curope*.so` 是必需运行资产，不能依赖官方的慢速 PyTorch 回退；该回退在特殊位置 token 上可能越界。应按上节用 Human3R 环境和 CUDA Toolkit 编译。

## 三阶段运行

以下路径对应已经完成的 `climb_6466b84f` 实验；新实验应使用新的输出目录。Human3R headless 脚本发现目标目录已有结果时会拒绝覆盖。

### 1. 完整 Human3R 重建

```bash
conda run --no-capture-output -n human3r \
python tools/bench/human3r_p2y/run_human3r_headless.py \
  --seq_path /home/user-kevien/gvhmr_pkg/gvhmr-web-tool/runtime/jobs/climb_6466b84f/0_input_video.mp4 \
  --model_path inputs/human3r_assets/human3r_672S.pth \
  --output_dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/human3r \
  --size 512 \
  --chunk_size 100 \
  --use_ttt3r
```

该脚本将 DINOv2 路由到固定的本地子模块，避免 `torch.hub` 再访问 GitHub；每 100 帧分块，并用一帧重叠对齐相机轨迹。它保存全部帧的 depth、confidence、color、camera 和 SMPL-X，不启动 Viser。

### 2. 重建地面与箱顶

```bash
conda run --no-capture-output -n human3r \
python tools/bench/human3r_p2y/extract_scene_planes.py \
  --human3r-dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/human3r \
  --output-dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/scene
```

脚本用多帧中位深度、Human3R 人体 mask、固定随机种子 Open3D RANSAC 和中心连通水平面拟合生成 `scene_planes.json` 与 `plane_overlay.png`。必须人工确认叠加点确实分别位于地面和箱顶。

### 3. 仅替换 GVHMR global root Y

```bash
conda run --no-capture-output -n gvhmr \
python tools/bench/human3r_p2y/apply_p2y.py \
  --gvhmr-result /home/user-kevien/gvhmr_pkg/experiments/footmr_v2/climb_6466b84f/a_current_hmr4d_results.pt \
  --scene-planes /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/scene/scene_planes.json \
  --video /home/user-kevien/gvhmr_pkg/gvhmr-web-tool/runtime/jobs/climb_6466b84f/0_input_video.mp4 \
  --output-dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/p2_y
```

先加 `--skip-video` 检查指标，可避免在保护条件失败时浪费渲染时间。通过后去掉该参数生成并排视频。

## 保护条件与边界

- 回地高度残差 `< 5 cm`。
- 箱顶高度误差 `< 5 cm`。
- root 单帧最大步长不超过基线的 1.5 倍（且至少允许 3 cm/frame）。
- root 加速度 P95 不超过基线的 1.25 倍。
- 顶层 schema、global/incam 参数键、X/Z、姿态、朝向、betas、`K_fullimg` 与 `net_outputs` 必须精确不变。

当前 `pre/top/post` 稳定窗口来自同目录的 `surface_metrics.py`，只适用于这段已标注的爬箱视频。接入通用推理前，必须把稳定表面时间段改为自动检测并在更多视频上验证。该方案不改 X/Z，因此不能解决水平漂移。

## P2-XYZ 水平漂移诊断候选

`apply_p2xyz.py` 在已通过的 P2-Y 上再增加保守 X/Z 修正。Human3R 的人体分支在本视频约 500–800 帧存在背景误检，因此脚本不会迁移 Human3R SMPL-X root；它只把 GVHMR 足点投影到 Human3R 的静态场景平面，在动作前后的同一地面上估计水平终点漂移。

```bash
conda run --no-capture-output -n gvhmr \
python tools/bench/human3r_p2y/apply_p2xyz.py \
  --gvhmr-result /home/user-kevien/gvhmr_pkg/experiments/footmr_v2/climb_6466b84f/a_current_hmr4d_results.pt \
  --p2-y-result /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/p2_y/p2_y_hmr4d_results.pt \
  --p2-y-metrics /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/p2_y/metrics.json \
  --human3r-dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/human3r \
  --scene-planes /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/scene/scene_planes.json \
  --video /home/user-kevien/gvhmr_pkg/gvhmr-web-tool/runtime/jobs/climb_6466b84f/0_input_video.mp4 \
  --output-dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_xyz/climb_6466b84f/p2_xyz
```

当前片段的最终修正约为 X `+16.94 cm`、Z `+5.91 cm`。它通过 schema、有限值、root 水平步长、加速度和回地场景残差保护，但没有水平 3D ground truth。箱顶水平交点的逐帧离散 P95 达到约 31.8 cm，因此箱顶观测只用于 Y，高度不可靠的水平观测不会写入 X/Z。

P2-XYZ 输出是可供 GMR 直接读取的 `p2_xyz_hmr4d_results.pt`，但当前状态仍是诊断候选；在更多相机与场景验证前，P2-Y 是更保守的推荐结果。

## 单地面视频：自动 Ground-XYZ

没有箱体或台阶的视频不能使用爬箱实验的 `pre/top/post` 人工窗口。此时先提取一个覆盖图像下方的静态地面，再从 FootMR 的高置信度足部接触帧自动估计 root XYZ 的低频漂移。Human3R 的人体 SMPL-X 仍不参与融合。

```bash
conda run --no-capture-output -n human3r \
python tools/bench/human3r_p2y/extract_ground_plane.py \
  --human3r-dir /path/to/experiment/human3r \
  --output-dir /path/to/experiment/scene

conda run --no-capture-output -n gvhmr \
python tools/bench/human3r_p2y/apply_ground_xyz.py \
  --gvhmr-result /path/to/footmr/hmr4d_results.pt \
  --human3r-dir /path/to/experiment/human3r \
  --ground-plane /path/to/experiment/scene/ground_plane.json \
  --video /path/to/input_30fps.mp4 \
  --output-dir /path/to/experiment/ground_xyz
```

处理流程如下：

1. Human3R 多帧中位深度去掉人体区域，Open3D RANSAC 提取多个静态平面；ground-only 默认 confidence 阈值为 1.05（Human3R confidence 最小值接近 1），仍要求足够的跨帧重复观测。
2. 用平面点数、图像下半区占比、水平覆盖率和底边覆盖率选择单一地面；必须人工检查 `ground_overlay.png`。
3. 将 FootMR 的六个脚趾/脚跟点投影到 Human3R 地面，使用最初约 3 秒的高置信度接触帧把地面切空间刚体对齐到 GVHMR global XZ。
4. 只在高置信度接触帧观测 XYZ correction，剔除离群点并使用约 2 秒窗口提取低频趋势；无接触区间只插值 correction，不修改姿态。
5. 输出 `ground_xyz_hmr4d_results.pt`，只允许 `smpl_params_global.transl` 变化，并检查 schema、有限值、root 步长、加速度、接触足速度和场景残差。

该方法目前仍是假设固定相机、单一平坦地面的离线诊断候选。单目 Human3R 没有测量标定或 3D ground truth，保护条件通过不能解释成绝对 XYZ 已验证。原视频若不是 30 FPS，应先正确重采样；不能只复制 60 FPS 帧再用 30 FPS 写出，否则时长会翻倍。

## CoTracker3 接触足 X/Z 实验（P5，当前未采用）

CoTracker3 不替换 GVHMR/FootMR，也不跟踪固定相机下没有信息量的静态地板点。实验从 FootMR 的左右足接触概率检测接触段，用 COCO23 的大脚趾、小脚趾和脚跟投影作为查询，跟踪实际鞋部像素相对运动，再对 Ground-XYZ 的 global root X/Z 建立相对约束。Ground-XYZ 的 Y、pose、orientation、betas、incam、`K_fullimg` 和 `net_outputs` 必须逐元素不变。

CoTracker 固定为 `third-party/CoTracker` commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`。不需要安装进 Conda 环境；脚本只在 GPU 跟踪阶段从固定子模块懒加载。先下载并校验权重：

```bash
conda run --no-capture-output -n gvhmr \
python tools/demo/download_cotracker_assets.py
```

权重保存到 `inputs/cotracker_assets/scaled_offline.pth`，SHA256 为 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`，不会写入可能链接到 WebTool 的 `inputs/checkpoints`。

```bash
conda run --no-capture-output -n gvhmr \
python tools/bench/human3r_p2y/apply_cotracker_ground_xz.py \
  --ground-xyz-result /path/to/ground_xyz_hmr4d_results.pt \
  --ground-xyz-metrics /path/to/ground_xyz/metrics.json \
  --video /path/to/input_30fps.mp4 \
  --output-dir /path/to/cotracker_ground_xz \
  --skip-video
```

实现与保护如下：

1. 合并最多 3 帧的接触缺口，忽略少于 8 帧的段；长段切成最多 60 帧、重叠 8 帧的 offline 窗口，避免一次处理整段视频。
2. 默认围绕每只脚建立固定 128×96 ROI 后交给 CoTracker3，避免全帧缩放后鞋部过小；低可见性、越界、速度突变和三点不一致轨迹会被剔除。
3. 稀疏最小二乘联合求解各窗口的相对 X/Z 约束，并以 Ground-XYZ 零修正作为绝对先验，防止窗口间累计漂移。默认强时间平滑来自 `ydd` 的保守扫描，不表示通用最优参数。
4. 只有跟踪覆盖率、最大修正、root 步长/加速度、场景 XZ 残差、跟踪锁定残差和接触足速度全部通过时，才生成 `cotracker_ground_xz_hmr4d_results.pt`。失败时只保留明确命名的 `candidate_hmr4d_results.pt` 和诊断文件，正式流程继续使用 Ground-XYZ。

在 `ydd` 上，局部 ROI 跟踪覆盖 93.78% 的接触足帧。未经强平滑的跟踪锁定残差 P95 从 5.35 cm 降到 2.28 cm，但接触足速度 P95 从 6.29 增到 9.65 mm/frame，因此拒绝。随后对缓存约束扫描时间平滑权重 1–100 和绝对先验权重 0.5–20；最佳接触足速度 P95 仍为 6.40 mm/frame，没有优于 6.29 mm/frame baseline。当前不会生成推荐 tensor，也不接入 WebTool。

该结果说明 CoTracker 的 2D 自洽不等于 3D root 更准确：单目地面射线转换会放大亚像素噪声，查询点也可能落在鞋边或邻近地板。若后续继续，应先加入鞋部 mask/语义关键点和带不确定度的多视角或测量真值评估，而不是放宽本实验保护条件。

许可边界：CoTracker 上游多数代码为 CC BY-NC 4.0，且部分目录有单独条款。该 P5 实验及权重只能在符合上游非商业和署名要求时使用，不能视为被本仓库重新许可。

## Human3R-only 人体轨迹检查

Human3R 同时输出人体 SMPL-X，但不能默认认为其单人轨迹完整。以下命令使用已有 bbox 作为目标位置参考，检查每帧最近的 Human3R 头部中心；bbox 只用于评估，不会把 GVHMR 姿态混入 Human3R。

```bash
conda run --no-capture-output -n human3r \
python tools/bench/human3r_p2y/evaluate_human3r_only.py \
  --human3r-dir /home/user-kevien/gvhmr_pkg/experiments/footmr_p2_y/climb_6466b84f/human3r \
  --bbox-file /home/user-kevien/gvhmr_pkg/gvhmr-web-tool/runtime/jobs/climb_6466b84f/preprocess/bbx.pt \
  --video /home/user-kevien/gvhmr_pkg/gvhmr-web-tool/runtime/jobs/climb_6466b84f/0_input_video.mp4 \
  --output-dir /home/user-kevien/gvhmr_pkg/experiments/human3r_only/climb_6466b84f
```

当前视频在宽松的 50 Human3R 像素阈值下只有 55.85% 有效覆盖，最长连续丢失 173 帧，且有 63 帧完全没有人体候选。保护条件要求至少 95% 覆盖且最长缺口不超过 5 帧，因此拒绝导出 Human3R-only GMR 文件。`human3r_only_tracking_overlay.mp4` 用绿色显示有效最近候选、红色显示误检或丢失。
