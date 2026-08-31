# GVHMR Web 客户部署说明（供 AI 代理读取）

本文是自动化部署代理的执行契约。目标是在客户 Linux GPU 服务器上部署本仓库的公开版本，并交付一个可验证、可停止、不会误连真机的 GVHMR Web 服务。

## 0. 强制边界

部署代理必须遵守以下规则：

1. Human3R 仓库及其所含部分组件受 CC BY-NC-SA 4.0、NAVER Non-Commercial License 等非商业条款约束，不允许直接用于本项目的客户商业部署。因此禁止下载、初始化、复制、上传或启用 Human3R、DINOv2 和 `human3r_672S.pth`。公开仓库不交付这些组件；Web 中 Human3R 选项显示不可用属于正常状态。只有客户另行取得明确的商业授权并完成法务确认后，才能重新评估，AI 不得自行判断为可商用。
2. 不索取或记录 Hugging Face token、GitHub token、SSH 私钥、机器人密钥等秘密。若私有仓库认证确有需要，让客户在自己的终端完成登录。
3. 不执行 `git submodule update --init --recursive`。默认 FootMR + SimpleVO 路径不需要 Human3R 子模块。
4. 不删除或覆盖已有的 `runtime/`、任务、数据库和 checkpoint。升级前必须备份，所有新部署先使用独立目录。
5. 不通过 Web 启动真机，不创建真机 SSH 隧道，不修改机器人主控。客户部署默认只开放视频推理；SONIC/MuJoCo 需另行按安全文档授权。
6. Web 没有账号系统。不得把 7860 端口直接暴露到公网；优先绑定 `127.0.0.1`，通过客户自己的 VPN、SSH 隧道或带认证的 HTTPS 反向代理访问。

## 1. 部署前先收集证据

在修改服务器前读取并向客户报告：

```bash
uname -a
cat /etc/os-release
nvidia-smi
df -h
free -h
docker --version || true
conda --version || true
ss -ltnp | grep ':7860 ' || true
```

最低建议：Linux x86_64、可用 NVIDIA 驱动、12 GB 显存、50 GB 可用磁盘、16 GB 内存。若 GPU、磁盘或驱动不满足，不要猜测修复；报告证据并暂停。

还需向客户确认：

- 安装目录；默认建议 `/opt/gvhmr-web-tool` 或客户用户目录下的独立目录。
- 仅本机访问、VPN/SSH 隧道访问，还是已有带认证的 HTTPS 反向代理。
- 使用增强源码模式还是 Docker baseline。客户需要 FootMR 和优化后的地面约束时必须选择“增强源码模式”。

## 2. 推荐部署：增强源码模式

以下命令都从仓库根目录执行。克隆时禁止递归子模块：

```bash
git clone https://github.com/JKYovo/gvhmr-web-tool.git
cd gvhmr-web-tool
git status --short --branch
```

创建独立环境：

```bash
conda env create -n gvhmr -f deploy/env/environment-dev.yml
conda run -n gvhmr python -m pip install -e .
```

准备公开推理资产：

```bash
conda run -n gvhmr python -m hmr4d.service.assets \
  --checkpoint-root inputs/checkpoints
conda run -n gvhmr python tools/demo/download_footmr_assets.py
```

不要执行任何包含 `Human3R`、`dinov2`、`HUMAN3R_PYTHON` 或 `HUMAN3R_MODEL_PATH` 的安装命令。其未部署原因是非商业许可证限制，不是遗漏、下载失败或环境故障。

启动：

```bash
bash start_web_source.sh
```

脚本默认绑定 `127.0.0.1`，使用当前仓库作为算法 core，并将 PID 和日志写到 `runtime/`。不要为了远程访问直接改成 `0.0.0.0`。

## 3. Docker baseline（仅在客户明确接受时）

Docker 路径方便部署，但当前保留原始 GVHMR baseline，不包含源码模式的 FootMR 默认推理和全部优化。只有客户明确接受该差异时才执行：

```bash
bash doctor.sh
bash start_web.sh
```

若客户要求增强版，不得用 Docker baseline 冒充验收通过。

## 4. 必须完成的验收

服务启动后执行：

```bash
curl -fsS http://127.0.0.1:7860/health
curl -fsS http://127.0.0.1:7860/api/capabilities
bash status_web_source.sh
```

增强源码模式的验收条件：

- `health.status` 为 `ok`。
- `runtime.inference_backend` 为 `external_core`，并且 `runtime.inference_ready` 为 `true`。
- `flat_y` 和 `gravity_flat` 为 enabled。
- `human3r` 为 disabled；这是许可证要求下的公开客户版预期结果，不得为了消除该提示私自安装 Human3R。
- 页面可打开，上传区明确显示输入会规范化为 30 FPS。
- 使用客户授权的短视频完成一次测试；最终任务目录至少包含 `0_input_video.mp4`、`hmr4d_results.pt` 和 `job.json`。
- 成功任务不应残留 `submitted_input.*` 或 `preprocess/`；调试文件应位于 `diagnostics/`。

不要使用客户隐私视频做公开演示，不要上传任务结果到外部服务。

## 5. 远程访问

最小风险方案是让服务继续监听本机，然后从管理员电脑建立 SSH 隧道：

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

浏览器访问 `http://127.0.0.1:7860/`。如果客户已有 Nginx/Caddy 和身份认证，可反向代理到 `127.0.0.1:7860`；AI 只能在客户明确授权后修改该配置，并必须保留 HTTPS、认证和访问日志策略。

## 6. 运维、更新与回滚

查看状态和日志：

```bash
bash status_web_source.sh
tail -n 200 runtime/gvhmr_web.log
```

正常停止：

```bash
bash stop_web_source.sh
```

更新前：

1. 确认没有 `queued/running` 任务和 SONIC 推流。
2. 备份 `runtime/db/`；客户要求保留结果时同时备份 `runtime/jobs/`。
3. 记录当前 `git rev-parse HEAD`。
4. `git pull --ff-only`，重新运行测试和能力检查后再启动。

更新失败时回到记录的已知可用 commit。禁止使用 `git reset --hard` 清理客户工作区；应在新 worktree 或新目录验证后切换。

## 7. AI 最终交付报告模板

部署完成后只报告可验证事实：

```text
仓库 commit：
部署模式：增强源码 / Docker baseline
服务地址：
监听范围：127.0.0.1 / 已认证反向代理
GPU 与驱动：
checkpoint 检查：通过 / 未通过
capabilities：external_core=；flat_y=；gravity_flat=；human3r=disabled（预期）
短视频验收任务：
日志位置：
停止命令：
未完成项或风险：
```

不得把“服务进程已启动”等同于推理验收通过，也不得把 Human3R 缺失报告为公开版部署故障。
