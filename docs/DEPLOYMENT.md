# Docker 与局域网部署

## 商业部署许可证边界

Human3R 仓库及其所含部分组件受 CC BY-NC-SA 4.0、NAVER Non-Commercial License 等非商业条款约束，不允许直接用于本项目的客户商业部署。公开客户版因此不安装、不分发也不启用 Human3R/DINOv2；页面中 Human3R 不可用是预期状态。只有客户另行取得明确商业授权并完成法务确认后才能重新评估，部署人员和 AI 代理不得自行绕过该限制。

## 本机模式

```bash
bash start_web.sh
```

容器端口默认只绑定到 `127.0.0.1:7860`。运行目录映射如下：

| 宿主机 | 容器 |
| --- | --- |
| `runtime/checkpoints` | `/app/runtime/checkpoints` |
| `runtime/jobs` | `/app/runtime/jobs` |
| `runtime/batches` | `/app/runtime/batches` |
| `runtime/db` | `/app/runtime/db` |

Web 页面始终把新任务写入默认任务目录，避免让容器误用未挂载的宿主机路径。

## 局域网模式

```bash
bash start_web_lan.sh
```

脚本会把端口公开到局域网，并打印访问地址。当前 GVHMR 服务没有账号系统，只应在可信局域网中使用；不要直接暴露到公网。

## 首次构建

首次启动通常包含：

- Docker 运行时和 Python 依赖，约 `11GB`
- GVHMR、HMR2、ViTPose、YOLO 和身体模型，约 `5.6GB`

网络较快时通常需要 30 到 60 分钟，普通网络可能需要 45 到 90 分钟。构建和模型下载均可复用。

## 常用环境变量

```bash
export GVHMR_PORT=7860
export GVHMR_IMAGE_NAME=gvhmr-web:latest
export GVHMR_CONTAINER_NAME=gvhmr-web
```

已有正确镜像并希望跳过构建检查时：

```bash
export GVHMR_SKIP_BUILD=1
bash start_web.sh
```

源码模式还支持：

```bash
export GVHMR_CHECKPOINT_ROOT=/path/to/checkpoints
export GVHMR_OUTPUT_ROOT=/path/to/runtime/jobs
export GVHMR_BATCH_ROOT=/path/to/runtime/batches
export GVHMR_DB_PATH=/path/to/runtime/db/job_db.sqlite
```

这些自定义路径必须对运行服务的用户可读写。Docker 模式应同时提供对应 volume 映射。

## 更新代码

代码更新后重新构建镜像：

```bash
git pull
unset GVHMR_SKIP_BUILD
bash start_web.sh
```

`runtime/` 不会因镜像重建而删除。停止服务使用：

```bash
bash stop_web.sh
```
