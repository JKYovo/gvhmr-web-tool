# 源码开发环境

GVHMR-Enhanced 当前通过源码模式运行；Docker 路径暂时保留原始 GVHMR baseline。

## 创建环境

```bash
conda activate gvhmr
```

该环境使用 Python 3.10、PyTorch 2.3 和 CUDA 12.1 对应依赖。创建完成后确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出中的 CUDA 状态必须是 `True`。

## 准备模型

下载运行所需模型到 `inputs/checkpoints`：

```bash
python -m hmr4d.service.assets --checkpoint-root inputs/checkpoints
```

再下载 FootMR 与 whole-body ViTPose 权重。下载器会校验 SHA256，并保存到独立的 `inputs/footmr_assets/`：

```bash
python tools/demo/download_footmr_assets.py
```

## Human3R 边界

公开客户仓库不分发、不安装 Human3R、DINOv2、Human3R 权重或编译扩展。客户部署不要执行递归 submodule 初始化；缺少 Human3R 时对应 Web 选项禁用属于预期行为。AI 代理必须按 [README_AI_DEPLOY.md](../README_AI_DEPLOY.md) 部署。

如果模型已经由 Docker 下载到 `runtime/checkpoints`，也可以通过环境变量复用：

```bash
export GVHMR_CHECKPOINT_ROOT="$PWD/runtime/checkpoints"
```

## 启动源码服务

```bash
python -m hmr4d.service.server --host 127.0.0.1 --port 7860
```

使用源码启动脚本运行内置 GVHMR-Enhanced：

```bash
bash start_web_source.sh
```

脚本默认把当前 `gvhmr-web-tool` 仓库作为算法 core，不需要额外安装 `gvhmr-core-opt`。如需测试其他 worktree，可以显式指定：

```bash
GVHMR_CORE_ROOT=/path/to/gvhmr-core-opt bash start_web_source.sh
```

停止源码服务：

```bash
bash stop_web_source.sh
```

增强 backend 在独立 Python 进程中运行。Web 继续管理任务、日志和预览；默认推理为 FootMR-ViTPose，输入先按时间戳重采样至 30 FPS，再进行 bbox、姿态、图像特征和 VO 预处理。

也可以使用包装入口：

```bash
python tools/app/run_ui.py --host 127.0.0.1 --port 7860
```

源码模式默认使用：

```text
inputs/checkpoints
runtime/jobs
runtime/batches
runtime/db/job_db.sqlite
```

## 开发检查

```bash
python -m unittest discover -s tests -v
node --check hmr4d/service/static_app/app.js
python -m compileall hmr4d/service
```

这些测试使用临时目录和假预览 runner，不会执行完整 GPU 推理。
