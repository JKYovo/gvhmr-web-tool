# GVHMR Web 工具

[English README](README.en.md)

客户服务器由 AI 辅助部署时，先让部署代理完整阅读 [README_AI_DEPLOY.md](README_AI_DEPLOY.md)。公开客户版不分发或安装 Human3R/DINOv2；Human3R 是本机私有可选扩展。

把 [GVHMR](https://github.com/zju3dv/GVHMR) 的单人视频人体动作恢复流程封装成一个可部署的本地 Web 工具。源码模式已经内置 GVHMR-Enhanced：默认使用 FootMR 做脚踝细化，并将 Contact-aware Global Optimizer V1.1 作为自动平地约束，不再要求旁路安装 `gvhmr-core-opt`。

![GVHMR Web 界面](docs/images/gvhmr-web.png)

## 主要功能

- 新版 Web Studio 使用一屏双栏任务工作区和拖放上传；任务队列显示按需缓存的视频封面并独立滚动，任务详情固定显示在工作区下方
- 单视频与批量上传，支持 `mp4 / mov / avi / mkv / webm`；输入统一按时间戳转为 30 FPS，奇数宽高会无损补齐至偶数
- 静态相机和可选焦距 `f_mm`
- FootMR COCO23 脚踝残差细化，以及隔离的预处理缓存
- 四档地面处理：不启用、Contact Global V1.1 自动平地、站立帧重力校正 + 自动平地、Human3R 场景重力 + 自动平地
- 默认严格检查 global/incam 同步出现的孤立极端朝向跳变，只重分配短窗口 root 转速
- Human3R 接口仅作为本机私有扩展保留；公开客户仓库不分发其源码、子模块、权重或编译产物
- SQLite 任务持久化、状态筛选、取消和失败重试
- 长视频整段精确 local attention、内存受控共享裁剪和极端 OOM 断点续跑
- 推理完成后按需生成预览，预览失败不会破坏人体动作结果
- 页面内播放预览；下载列表默认只显示最终 PT、SONIC 数据、对比预览、Human3R 地面检查图和精简 ZIP
- Web 可一键启动/关闭本机 MuJoCo，并把同一任务发送到隔离的仿真 SONIC；仿真和真机链路由后端强制互斥
- 上传临时文件自动清理，任务输入和结果统一保存在任务目录
- Docker 优先的一键部署，以及供开发者使用的源码模式

## 使用流程

1. 上传视频并提交 GVHMR 推理。
2. 在右侧任务控制台查看状态，失败任务可以重试。
3. 下载 `hmr4d_results.pt`，或点击“生成预览”检查恢复效果。

## 快速启动

运行环境需要 Linux x86_64、NVIDIA GPU、可用驱动、Docker 和 NVIDIA Container Toolkit。

```bash
bash doctor.sh
bash start_web.sh
```

浏览器访问：

```text
http://127.0.0.1:7860/
```

查看状态或停止服务：

```bash
bash status.sh
bash stop_web.sh
```

第一次启动会构建 Docker 镜像并下载模型，总下载量约 `16GB ~ 17GB`。完整说明见 [快速开始](docs/QUICKSTART.md) 和 [部署说明](docs/DEPLOYMENT.md)。

要运行当前内置的 GVHMR-Enhanced，请使用已有的 `gvhmr` Conda 环境：

```bash
conda activate gvhmr
python tools/demo/download_footmr_assets.py
bash start_web_source.sh
```

源码启动默认以当前仓库作为算法 core；只有需要切换其他 worktree 时才设置 `GVHMR_CORE_ROOT`。Docker 启动路径目前仍保留原始 GVHMR baseline。

## 长视频优化

源码 Web 对不少于 1800 帧的视频自动启用整段精确 local attention。GVHMR 和
FootMR 原本在长序列上就使用 120 帧局部注意力范围，新实现跳过无用的完整
`L×L` 注意力矩阵，但仍然整段计算 betas、FootMR 脚踝残差、global root rollout
和官方 contact/IK 后处理。因此正常长视频不切块、不拼接，也不会引入分块边界跳变。

长视频预处理会把 ViTPose 和 HMR2 共用的人体 crop 逐帧写入任务目录的临时
磁盘映射，避免全视频图像和数 GB float crop 同时驻留内存；成功提取 pose 和
features 后自动删除。静态相机后处理中的未来帧重复更新也已改成数学等价的
线性累计计算。只有极端序列仍在整段 FK/IK 触发 CUDA OOM 时，才自动回退到
600 帧窗口、480 帧步长的缓存续跑模式。

当前验证结果：

- qhy 754 帧的原 dense 与新 local 输出树逐 tensor、逐 bit 一致。
- 7162 帧真实特征合成序列的完整主网络、FootMR、root rollout 和官方后处理耗时
  `2.83 s`，CUDA peak allocated 为 `475.6 MB`；这不是包含 YOLO、ViTPose、
  HMR2 和视频编码的端到端总耗时。
- 旧 xbd 分块流程实际处理约 9000 帧；新默认路径只预处理原始 7162 帧一次，
  消除约 26% overlap 重复和反复加载模型、切片编码的开销。完整端到端加速倍率
  需要由下一条真实长视频的任务 metrics 测量，不能由上述主网络耗时直接外推。
- 曾测试加快输入 H.264 转码，但 qhy 的 body P95 变化约 `1.3°`、global root P95
  变化约 `2.3 cm`，因此没有采用。当前不会用降低输入质量换速度。

长视频任务会额外保存 `diagnostics/long_video/manifest.json` 和 `diagnostics/long_video/metrics.json`，记录
实际模式、模型身份、耗时以及是否发生 OOM 窗口回退。完整实验和风险边界见
[优化记录 P28](docs/OPTIMIZATION_LOG.md#p28长视频精确推理内存上界与非模型产物加速)。

## 输出内容

每个任务默认保存到：

```text
runtime/jobs/<视频名>_<任务短 ID>/
```

新任务使用固定目录结构，任务根目录只保留用户真正需要识别的入口文件：

```text
任务目录/
├── 0_input_video.mp4
├── hmr4d_results.pt
├── job.json
├── artifacts.zip
├── preview/                  # 按需生成
├── exports/                  # 按需生成 SONIC 文件
├── diagnostics/              # raw、指标和检查图
└── .work/                    # 仅保留中间文件时使用
```

`hmr4d_results.pt` 始终是最终发布结果。原始 FootMR tensor、接触优化候选、重力指标和朝向检查统一移入 `diagnostics/`；Human3R 的逐帧 `depth/conf/color/camera/smpl` 只在 `.work/` 临时存在，默认在提取地面后删除。`artifacts.zip` 不再打包 raw/candidate 或单独的 incam/global 调试视频，只包含最终结果和少量摘要。

自动平地从原始 FootMR tensor 单次运行，只修改 `smpl_params_global.transl`。`gravity_flat` 先用可靠站立帧估计重力；找不到可靠站立段时会记录原因并只回退一次 `flat_y`。`human3r` 从场景地面估计重力，不回退到其他模式。保护项未通过但候选完整时仍发布约束结果并显示诊断警告；脚本异常、候选缺失或 metrics 损坏会使任务失败。页面/API 为兼容现有调用仍使用 `flat_y` 表示 Global V1.1。

孤立朝向保护位于地面约束之后，只在 global 单帧转角至少 `80°`、incam 同一边界至少 `75°`，且峰值相对邻域高度显著、附近没有第二个同级峰值时触发。修复使用端点固定的 SO(3) 短窗口平滑，只修改 `global_orient`；正常转身、`body_pose`、`transl` 和 `betas` 不参与平滑。未触发的任务只保存 metrics，不复制额外 tensor。

## SONIC 接入（无需 Kimodo）

本仓库已经内置 GVHMR SMPL-X22 到 SONIC reference 的转换和本地 ZMQ
播放适配层，统一使用 `gvhmr` Conda 环境，不需要 Kimodo 仓库、Kimodo
venv 或 PEFT。成功任务可在详情操作栏选择“发送到仿真”或“发送到真机”；按钮会从
最终发布的 `hmr4d_results.pt` 生成或复用 50 FPS reference，并在后台推流，
不会阻塞页面，也不会覆盖人体结果。推流期间可点击“暂停 SONIC”：Web 会停止
live reference，SONIC policy 随后按自身配置平滑回到内置 idle/default reference，
而不是保持动作最后一帧。推流还受 SONIC 模式硬联锁：接收端只有在
`sonic_teleop` 内才订阅；切到 `normal` 或其他模式会立即使当前发送会话永久失效，
再次进入 SONIC 后也不会续播，必须由用户重新点击发送。也可使用 CLI：

```bash
conda run -n gvhmr python tools/sonic/convert_gvhmr.py \
  runtime/jobs/<任务目录>/hmr4d_results.pt \
  runtime/jobs/<任务目录>/sonic_reference.npz \
  --metadata runtime/jobs/<任务目录>/sonic_conversion.json
```

SONIC/MuJoCo 已经启动时可通过仿真端口 5557 播放：

```bash
conda run -n gvhmr python tools/sonic/play_reference.py \
  runtime/jobs/<任务目录>/sonic_reference.npz
```

转换只读取 `hmr4d_results.pt`，不会覆盖 Web 任务结果。页面按钮会连接本机
SONIC，CLI 转换命令本身不会连接。
输出固定为 50 FPS 的 `term1_local`、`root_quat` 和 `wrist`。该适配层已对
jntm、lly、cxk、qhy、ydd 与旧 Kimodo 输入逐元素验证，数组最大差为 0。
这只保证输入 SONIC 不变，不代表控制策略已解决踝关节跟踪或机器人限位问题。
Web 使用兼容 SUB 的 ZMQ XPUB 检查订阅生命周期：没有 `sonic_teleop` 订阅时零帧
拒绝发送，播放中退订时立即终止；它仍没有逐帧执行 ACK，因此页面“推流完成”只能
证明发送端完成，不能证明机器人实际执行了每一帧。该联锁同时要求机器人
`raindrop` overlay 部署新版 SONIC `policy.py/state.py`，旧接收端不能视为已修复。
Web 的“启动仿真”只会启动本机 MuJoCo 和仿真控制器，固定使用
ROS Domain 73、`ROS_LOCALHOST_ONLY=1` 和本机 5557。真机使用独立的本机 5559；
Web 不包含 SSH 或真机 launch 命令，也不会启动真机控制器。检测到任一真机隧道时
后端拒绝启动/发送仿真，仿真未完全关闭时后端拒绝发送真机。

任务详情中的“SONIC 速度”滑块支持 `0.25×～1.00×`，步长为 `0.05×`。
倍率只调整下一次发送给 SONIC 的动作时间轴，各档均通过旋转 SLERP 重新采样并保持
50 FPS 控制输入；不会改写 GVHMR tensor、预览视频或原始动作速度。不同倍率使用
独立 reference 缓存，拖动滑块不会自动推流，适合在机器人腿部跟不上正常速度时逐级降低速度验证。

按需生成的预览：

- `preview/incam.mp4`
- `preview/global.mp4`
- `preview/comparison.mp4`

`submitted_input.*` 和 `preprocess/` 只在任务执行期间使用，成功后删除；`0_input_video.mp4` 是规范化后的持久输入，但不是稳定的数据交换格式。

## 文档

- [快速开始](docs/QUICKSTART.md)
- [Web → MuJoCo/SONIC 仿真安全流程](docs/SONIC_SIMULATION.md)
- [GVHMR Web → ELF3 SONIC 真机 Quickstart](docs/SONIC_REAL_ROBOT_QUICKSTART.md)
- [Docker 与局域网部署](docs/DEPLOYMENT.md)
- [源码开发环境](docs/INSTALL.md)
- [常见问题](docs/TROUBLESHOOTING.md)

## 适用范围

- 当前只处理单人主轨，不提供多人身份管理。
- GVHMR 推理要求 CUDA，不支持 CPU 推理。
- 默认 Web 服务是独立工具，不依赖 GMR Web。
- `runtime/`、模型权重和 Docker 镜像不提交到 Git。

## 上游与引用

本工具基于原始 GVHMR 项目：

- 项目主页：https://zju3dv.github.io/gvhmr
- 论文：https://arxiv.org/abs/2409.06662
- 上游仓库：https://github.com/zju3dv/GVHMR

使用 GVHMR 研究成果时，请引用原论文：

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

源码模式默认启用的脚踝细化来自 FootMR：

```bibtex
@InProceedings{wehrbein26footmr,
  author    = {Wehrbein, Tom and Rosenhahn, Bodo},
  title     = {Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

可选场景重力约束来自 Human3R：

```bibtex
@article{chen2025human3r,
  title={Human3R: Everyone Everywhere All at Once},
  author={Chen, Yue and Chen, Xingyu and Xue, Yuxuan and Chen, Anpei and Xiu, Yuliang and Gerard, Pons-Moll},
  journal={arXiv preprint arXiv:2510.06219},
  year={2025}
}
```

Human3R 及其部分上游组件采用非商业许可证（包括 CC BY-NC-SA 4.0 和 NAVER Non-Commercial License）。启用该可选项前必须检查 `third-party/Human3R/LICENSE` 和 `NOTICE.txt`；本仓库集成不授予商业使用权。
