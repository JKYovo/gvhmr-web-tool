# 源码开发环境

日常使用推荐 Docker。只有需要修改 GVHMR 推理、服务端或前端代码时，才需要源码环境。

## 创建环境

```bash
conda env create -f deploy/env/environment-dev.yml
conda activate gvhmr-dev
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

如果模型已经由 Docker 下载到 `runtime/checkpoints`，也可以通过环境变量复用：

```bash
export GVHMR_CHECKPOINT_ROOT="$PWD/runtime/checkpoints"
```

## 启动源码服务

```bash
python -m hmr4d.service.server --host 127.0.0.1 --port 7860
```

需要让 Web 调用独立 GVHMR 算法 worktree 时，使用源码启动脚本：

```bash
bash start_web_source.sh
```

脚本会自动发现同级目录 `../gvhmr-core-opt`，并将其作为外部推理 backend。也可以显式指定：

```bash
GVHMR_CORE_ROOT=/path/to/gvhmr-core-opt bash start_web_source.sh
```

停止源码服务：

```bash
bash stop_web_source.sh
```

外部 backend 在独立 Python 进程中运行。Web 继续管理任务、日志和预览，算法导入固定来自 `GVHMR_CORE_ROOT`，不会与 Web 仓库内的 `hmr4d` 混用。

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
