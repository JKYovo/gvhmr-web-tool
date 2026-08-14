# GVHMR Web → ELF3 SONIC 真机 Quickstart

> 本文仅用于已经在 MuJoCo 通过人工检查的短动作。真机首次测试必须有现场人员、可靠支撑和可随时操作的实体急停。

## 0. 数据链路与边界

GVHMR Web 已经内置 SMPL-X22 到 SONIC reference 的转换和 50 FPS ZMQ 推流，真机测试不需要启动 Kimodo，也不需要 Kimodo 的 Python 环境。

```text
GVHMR Web 最终 hmr4d_results.pt
  → 本机 Web 转换为 50 FPS SONIC reference
  → 本机 tcp://127.0.0.1:5557
  → SSH reverse tunnel
  → 机器人 127.0.0.1:5558
  → raindrop overlay 的 SONIC policy
  → ELF3 真机
```

机器人端环境变量仍为 `SONIC_REFERENCE_MODE=kimodo`。这是已部署的 5558 隔离模式的历史名称，不表示需要运行 Kimodo；不要自行改成 `gvhmr`。

本流程不覆盖同事使用的官方 PICO/5557 环境。机器人上的改动位于独立 overlay `/home/bxi/raindrop`，不要在 `/home/bxi/bxi_ws/bxi_rl_controller_ros2_example` 中修改 SONIC plugin、应用 stash 或重新构建。

Web 使用 ZMQ PUB，协议没有 ACK。页面显示“推流完成”只代表 Web 发完，不能单独证明机器人收到并执行了每一帧。

## 1. 真机上电前的必要条件

- 首次动作仅使用 5～15 秒、慢速、小幅、双脚支撑的 reference。
- 该动作已在相同 SONIC policy 与 MuJoCo 中完整播放，无超限、摔倒、明显自碰或脚步跟踪失败。
- 机器人可靠支撑，双脚离地或减载，周围无人和障碍物。
- 现场人员手持实体急停，并清楚断电和退回 `zero_torque` 的流程。
- 不在真机上首次尝试跳跃、单脚、快速转身、深蹲、大幅挥臂或长视频。

任何一项不满足时不要继续。

## 2. 检查重复真机控制器

在机器人 SSH 终端中检查：

```bash
pgrep -af 'hardware_elf3|example_demo_hw|bxi_example_py_elf3_demo'
pgrep -af 'remote_controller.*keyboard|remote_controller_keyboard'
```

如果已有真机控制器，先由启动它的 APP、服务或终端按官方流程正常退出。机器人未可靠支撑、未进入安全状态时，不要直接 `kill -9` 控制器。

同一时刻只能有：

- 一个 `example_demo_hw.launch.py` 真机控制器；
- 一个 keyboard remote controller；
- 一个 Web SONIC publisher。

ROS daemon 可能缓存已退出节点，真实检查时使用：

```bash
ROS2CLI_NO_DAEMON=1 ros2 node list
ROS2CLI_NO_DAEMON=1 ros2 topic list -t
```

## 3. 启动本机 GVHMR Web

在本机使用 `gvhmr` Conda 环境和当前 Web 仓库：

```bash
cd /home/user-kevien/gvhmr_pkg/gvhmr-web-tool
conda activate gvhmr
bash start_web_source.sh
```

打开 <http://127.0.0.1:7860>，确认要测试的任务已成功，且 MuJoCo 人工检查使用的正是该任务最终发布的 `hmr4d_results.pt`。此时不要点击“发送到 SONIC”。

Web 已经运行时不要重复启动第二个实例。

## 4. 建立本机 5557 → 机器人 5558 隔离隧道

在本机另开终端，建立 SSH reverse tunnel 并保持终端运行：

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=2 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:5558:127.0.0.1:5557 \
  bxi@192.168.88.172
```

密码验证后终端持续空白是正常现象。不要把远端监听改成 `0.0.0.0`，也不要建立旧的远端 5557 隧道。

在机器人端检查：

```bash
ss -ltn | grep ':5558 '
```

应看到只在回环地址上的 5558 `LISTEN`。

## 5. 启动真机 SONIC 控制器

确认实体急停、可靠支撑与无重复控制器后，在机器人 SSH 终端执行：

```bash
sudo -i
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
source /home/bxi/raindrop/install/setup.bash
cd /home/bxi/raindrop
export ROS_DOMAIN_ID=31
export ROS_LOCALHOST_ONLY=0
export SONIC_REFERENCE_MODE=kimodo
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

`/home/bxi/raindrop/install/setup.bash` 必须最后 source。带 `_hw` 的 launch 是真机；不带 `_hw` 的 `example_demo.launch.py` 是仿真，不要混用。

另开一个机器人 root 只读终端，确认环境已传入启动进程：

```bash
sonic_launch_pid=$(pgrep -n -f '/opt/ros/humble/bin/ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py')
tr '\0' '\n' < "/proc/${sonic_launch_pid}/environ" |
  grep -E '^(SONIC_REFERENCE_MODE|ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY)='
```

必须看到 `SONIC_REFERENCE_MODE=kimodo` 和 `ROS_DOMAIN_ID=31`。

控制器会自动执行两阶段 hardware reset，但真机启动后仍处于 `zero_torque` 且不动是预期行为。不要在这个 launch 终端输入状态按键。

## 6. 启动唯一的键盘节点

这台主控的 Fast DDS 共享内存按用户权限隔离。真机控制器由 root 运行时，键盘节点和 ROS 只读检查也使用 root。

另开机器人 SSH 终端：

```bash
sudo -i
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
cd /home/bxi/bxi_ws/bxi_rl_controller_ros2_example
source install/setup.bash
export ROS_DOMAIN_ID=31
export ROS_LOCALHOST_ONLY=0
ros2 launch remote_controller remote_controller_keyboard.launch.py
```

等到日志显示 `input candidate ready; accepting commands: keyboard`。键盘事件只从这个终端输入。

## 7. 验证硬件链并逐级切换状态

在机器人 root 只读终端中执行：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
export ROS_DOMAIN_ID=31
export ROS_LOCALHOST_ONLY=0
ROS2CLI_NO_DAEMON=1 ros2 topic info -v /motion_commands
ROS2CLI_NO_DAEMON=1 ros2 topic info -v /hardware/actuator_states
ROS2CLI_NO_DAEMON=1 ros2 topic info -v /hardware/actuators_cmds
timeout 5 env ROS2CLI_NO_DAEMON=1 \
  ros2 topic echo --once /hardware/state_machine_info
```

继续前必须满足：

- `/motion_commands` 只有一个键盘 publisher 和一个控制器 subscriber；
- `/hardware/actuator_states` 和 `/hardware/actuators_cmds` 均存在；
- 可以读到 `/hardware/state_machine_info`。

然后每次只按一次键，每次都等只读终端确认 `current` 状态更新：

| 键盘终端按键 | 仅在当前状态下有效的转换 |
| --- | --- |
| `!`（`Shift+1`） | `zero_torque → pd_brake`；会施加关节 PD，机器人可能立即动作 |
| `1` | `pd_brake → normal` |
| `@`（`Shift+2`） | `normal → sonic_teleop` |
| `1` | `sonic_teleop → normal` |

不要连续重复按键。如果键盘日志有事件而状态不变，先查当前 route 和 `/motion_commands` 端点，不要乱按尝试。

## 8. 推流前最后检查

进入 `sonic_teleop` 后先不发动作，让机器人在 SONIC 内置 idle reference 下稳定至少 10 秒。

在机器人端确认 SSH 隧道正在回环地址监听 5558：

```bash
ss -ltnp | grep ':5558 '
```

推流前只看到 `LISTEN` 是正常的。Kimodo 会长期占用本机 5557，但 GVHMR Web 仅在点击“发送到 SONIC”后的播放期间绑定 5557；因此不能照搬 Kimodo 流程，在推流前强制要求 `ESTAB`。

- 5558 没有 `LISTEN`：检查 SSH 隧道终端。
- 控制器环境不是 `SONIC_REFERENCE_MODE=kimodo`：当前可能仍是官方 PICO/5557 链路，不要发送 Web 动作。
- 发现多个硬件控制器或 keyboard publisher：不要继续。

## 9. 首次发送 Web 动作

1. 现场人员握住实体急停，其他人员退出机器人可达范围。
2. 确认机器人仍受支撑，不让其承重行走。
3. 在 Web 中选择已通过 MuJoCo 的短动作。
4. 首次测试先将“SONIC 速度”滑块设为 `0.5×` 或更低；确认稳定后再按 `0.05×` 小步提高，不能直接跳到 `1.0×`。倍率只调整动作时间轴，所有档位都保持 50 FPS 控制输入；拖动滑块本身不会发送动作。
5. 在机器人只读终端预先运行 `watch -n 0.1 "ss -tnp | grep ':5558 ' || true"`。
6. 点击“发送到 SONIC”。播放期间应看到 5558 `ESTAB`，同时观察最初 1～2 秒的关节方向、限位和跟踪。
7. 如果始终没有 `ESTAB`，立即点击“暂停 SONIC”，检查隧道、overlay 和模式，不要反复点击发送。
8. 任何超限、抽动、跟踪失稳、转向错误或支撑异常，立即点击“暂停 SONIC”；来不及时直接使用实体急停。

“暂停 SONIC”会停止 Web live reference。SONIC policy 在 reference 超时后平滑回到自身 idle/default reference；它不等同于实体急停，也不会切断扭矩。

## 10. 正常停止顺序

1. 在 Web 点击“暂停 SONIC”，等待机器人稳定回到 idle。
2. 在键盘终端按一次 `1`，确认 `sonic_teleop → normal`。
3. 按 BXI 官方真机流程退回安全状态并停止真机控制器。
4. 最后关闭 SSH 5558 隧道和 Web。

不要先关闭控制器、键盘节点或隧道，再期待 Web 暂停能使机器人回到 idle。

## 11. 常见问题

### Web 显示完成，真机没有动

依次检查：

```bash
# 本机 Web 播放时应监听 5557
ss -ltnp | grep ':5557 '

# 机器人端应同时有 5558 LISTEN 和 ESTAB
ss -ltnp | grep ':5558 '
ss -tnp | grep ':5558 '
```

再确认当前状态是 `sonic_teleop`，并且 policy 连接的是 5558 而非 5557。

### 动作延迟、丢步或突然冲一下

先查控制日志中的周期统计和调度策略。本地仿真已验证：控制线程未获得 `SCHED_FIFO/99` 时，时序异常约为 12%～13%；授权后降至 0.61%，且无跳过周期。真机上仍应实际确认控制线程策略，不要假设 root 启动必然成功。

```bash
controller_pid=$(pgrep -n -f '/bxi_example_py_elf3_demo')
ps -L -p "$controller_pid" -o pid,tid,psr,cls,rtprio,pri,pcpu,stat,comm
```

控制线程应显示 `FF`/`SCHED_FIFO` 和 `RTPRIO 99`，启动日志不应有 `Operation not permitted`。

### 状态键无效

确认按键输入在 keyboard launch 的那个终端，并且 `/motion_commands` 只有一个 publisher。在 `zero_torque` 直接按 `1` 会被忽略，必须按状态图逐级转换。

### 如何恢复同事使用的 PICO/5557 模式

由现场人员安全停止当前真机控制器，重新打开终端，只 source 官方工作区：

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
```

不要 source `/home/bxi/raindrop/install/setup.bash`，也不要设置 `SONIC_REFERENCE_MODE`。

## 12. 首次真机记录项

每次真机试验至少保存：

- Web `job_id`、`sonic_reference.npz` SHA256、帧数和 FPS；
- SONIC 动作速度倍率；
- 机器人 SONIC/overlay commit 或部署版本；
- 启动时间和控制器日志；
- `SCHED_FIFO/99` 验证结果和控制周期 P99/跳过数；
- 是否触发急停、限位、自碰、支撑失稳或跟踪异常；
- 现场视频和最终“通过 / 不通过”结论。
