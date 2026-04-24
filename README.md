# GVHMR Web 工具

[English README](README.en.md)

这个仓库在原始 [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) 的基础上，做了一个更适合团队内部使用的本地 Web 工具，用于：

- 上传视频
- 运行 GVHMR 单人主轨推理
- 导出结构化人体运动数据 `gvhmr_data.npz` 和 `gvhmr_meta.json`
- 按需生成预览视频

它保留了原始 GVHMR 的推理链路，同时新增了：

- 可复用的 Python API：`hmr4d/api/video_to_data.py`
- 本地 Web UI
- Docker 优先的一键部署方式
- 任务持久化、批量提交和结果打包下载

## 这个仓库适合做什么

如果你希望把 GVHMR 当成“工具”而不是“研究 demo”来使用，这个仓库就是为这个场景准备的。

典型使用流程：

1. 启动 Web 服务
2. 上传一个或多个视频
3. 下载 `NPZ + JSON` 结果
4. 按需下载预览视频

## 一键部署

这是目前最推荐的运行方式。

### 支持的运行环境

- Linux `x86_64`
- NVIDIA GPU
- 已安装 NVIDIA 驱动
- 已安装 Docker
- 已安装 NVIDIA Container Toolkit

### 启动

在仓库根目录执行：

```bash
bash start_web.sh
```

默认访问地址：

```text
http://127.0.0.1:7860/ui
```

### 局域网访问

如果希望同一局域网内的其他设备访问：

```bash
bash start_web_lan.sh
```

### 查看状态与停止服务

```bash
bash status.sh
bash stop_web.sh
```

### 首次启动会做什么

首次启动时，脚本会自动：

- 构建 Docker 镜像
- 检查容器内 CUDA 是否可用
- 创建 `runtime/checkpoints`、`runtime/jobs`、`runtime/batches`、`runtime/db`
- 把必需模型下载到 `runtime/checkpoints`

后续再次启动时，会直接复用已有镜像和权重。

## 本地源码模式

如果你是在本地做开发，而不是走 Docker 部署：

```bash
pip install -r deploy/env/requirements-ui.txt
python tools/app/run_ui.py
```

源码模式主要用于开发调试。给同事或其他使用者部署时，优先推荐 `start_web.sh`。

## 输出内容

每个任务的结果会写到 `runtime/jobs` 下。

核心输出：

- `gvhmr_data.npz`
- `gvhmr_meta.json`
- `hmr4d_results.pt`

可选输出：

- `1_incam.mp4`
- `2_global.mp4`
- `*_3_incam_global_horiz.mp4`

## 数据格式

`gvhmr_data.npz` 中会包含类似这些扁平字段：

- `smpl_global_body_pose`
- `smpl_global_global_orient`
- `smpl_global_transl`
- `smpl_global_betas`
- `smpl_incam_body_pose`
- `smpl_incam_global_orient`
- `smpl_incam_transl`
- `smpl_incam_betas`
- `camera_K_fullimg`

`gvhmr_meta.json` 里会记录任务元信息，例如：

- 源视频路径
- 源视频分辨率和 fps
- 处理后帧数
- 处理后 fps
- `static_cam`
- `f_mm`
- 输出目录
- `person_mode = "single_primary_track"`

## 仓库结构

- `deploy/docker/`
  Dockerfile 和 Compose 配置
- `deploy/env/`
  运行时和开发环境依赖文件
- `deploy/scripts/`
  真正的部署脚本实现
- 根目录 `start_web.sh` / `start_web_lan.sh` / `status.sh` / `stop_web.sh`
  面向使用者的薄包装入口
- `hmr4d/api/`
  视频转结构化数据的 API 层
- `hmr4d/service/`
  任务管理、持久化、服务入口和 Web UI
- `tools/app/run_ui.py`
  本地源码模式启动入口
- `start_web.sh`
  本机 Docker 一键启动
- `start_web_lan.sh`
  局域网可访问的一键启动

## 重要说明

- 当前版本只面向单人主轨处理
- 不支持 CPU 推理
- `inputs/`、`outputs/`、`runtime/` 不纳入 git 版本管理
- Docker 镜像和模型权重不会直接提交到仓库

## 开发环境

如果你要做开发：

```bash
conda env create -f deploy/env/environment-dev.yml
conda activate gvhmr-dev
```

## 上游项目

本仓库基于原始 GVHMR 项目构建：

- 项目主页：https://zju3dv.github.io/gvhmr
- 论文：https://arxiv.org/abs/2409.06662
- 上游仓库：https://github.com/zju3dv/GVHMR

## 引用

如果你在研究中使用的是 GVHMR 本身，请引用原论文：

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

## 致谢

感谢以下项目作者的工作：

- [WHAM](https://github.com/yohanshin/WHAM)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch)
