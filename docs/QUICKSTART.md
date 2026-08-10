# 快速开始

## 1. 确认运行环境

需要：

- Linux x86_64
- NVIDIA GPU 和可用驱动
- Docker
- NVIDIA Container Toolkit
- 首次部署约 `16GB ~ 17GB` 下载空间

克隆工具仓库：

```bash
git clone https://github.com/JKYovo/gvhmr-web-tool.git
cd gvhmr-web-tool
```

运行自检：

```bash
bash doctor.sh
```

Ubuntu 或 Debian 上可以显式允许脚本尝试安装 Docker 和 NVIDIA Container Toolkit：

```bash
bash doctor.sh --fix
```

脚本不会安装 NVIDIA 显卡驱动。

## 2. 启动服务

```bash
bash start_web.sh
```

首次启动会依次检查环境、构建镜像、验证容器 CUDA、下载模型并启动服务。后续启动会复用镜像和权重。

浏览器访问：

```text
http://127.0.0.1:7860/
```

## 3. 处理视频

1. 选择单视频或批量模式。
2. 固定机位保留“静态相机”，移动相机时关闭。
3. 焦距未知时让 `f_mm` 保持自动。
4. 提交任务，在右侧任务控制台查看状态。
5. 完成后下载 PT；需要视觉检查时再生成预览。

结果位于：

```text
runtime/jobs/<视频名>_<任务短 ID>/
```

## 管理服务

```bash
bash status.sh
bash stop_web.sh
```

环境或推理出错时参见 [常见问题](TROUBLESHOOTING.md)。
