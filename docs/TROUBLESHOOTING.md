# 常见问题

## 页面显示 GPU 不可用

先检查宿主机：

```bash
nvidia-smi
```

再检查容器：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

宿主机正常但容器失败时，通常是 NVIDIA Container Toolkit 未安装或 Docker runtime 未配置。

## 页面显示模型缺失

检查：

```bash
find runtime/checkpoints -type f | sort
```

重新运行启动脚本会复用已有文件并补齐缺失模型：

```bash
bash start_web.sh
```

不要删除下载中的 `*.part` 文件，除非确认下载已经中断且不会恢复。

## Docker 没有权限

```bash
docker info
```

如果当前用户不能访问 Docker，可以将用户加入 docker 组并重新登录，或在 Ubuntu/Debian 上运行：

```bash
bash doctor.sh --fix
```

## 端口 7860 已被占用

查看占用：

```bash
ss -ltnp | grep ':7860'
```

改用其他端口：

```bash
GVHMR_PORT=7861 bash start_web.sh
```

## 任务失败

先展开页面底部的“任务日志”，再检查容器日志：

```bash
docker logs --tail 200 gvhmr-web
```

任务失败后原输入仍保存在任务目录，可以直接点击“重试任务”。

## 预览失败但 PT 已经生成

预览是独立的后续渲染。页面显示预览失败时，`hmr4d_results.pt` 仍然有效，可以正常下载。确认 GPU 显存和 FFmpeg 后点击“重新生成预览”。

## 页面中的 `/app/runtime/jobs` 是什么

这是 Docker 容器路径，对应宿主机仓库中的 `runtime/jobs`。新版服务会在 Docker 与源码模式切换时修正可识别的旧任务路径。

## 磁盘空间持续增长

上传临时目录会在任务创建后自动清理，长期数据主要位于：

```text
runtime/jobs
runtime/checkpoints
```

可以删除不再需要的任务目录，但同时保留的 SQLite 历史记录仍会显示该任务；建议先停止服务并备份 `runtime/db`。

## 页面样式没有更新

先强制刷新浏览器缓存。Docker 部署更新了静态文件后，需要重新构建镜像：

```bash
unset GVHMR_SKIP_BUILD
bash start_web.sh
```
