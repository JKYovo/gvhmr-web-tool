"""Safety-scoped lifecycle manager for the local ELF3 MuJoCo simulation.

This module intentionally contains no SSH command and no hardware launch name.
It can only start the local MuJoCo node and the simulation controller.
"""

from __future__ import annotations

import os
import resource
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

import yaml

from hmr4d.service.common import ensure_dir, utc_now_iso


SIMULATION_SONIC_ENDPOINT = "tcp://127.0.0.1:5557"
REAL_ROBOT_SONIC_ENDPOINT = "tcp://127.0.0.1:5559"
LEGACY_REAL_TUNNEL_FORWARD = "127.0.0.1:5558:127.0.0.1:5557"
ISOLATED_REAL_TUNNEL_FORWARD = "127.0.0.1:5558:127.0.0.1:5559"

_SIMULATION_EXECUTABLES = (
    "elf3_dof29_sim.launch.py",
    "example_demo.launch.py",
    "bxi_example_py_elf3_demo",
)


def iter_process_commands(proc_root=Path("/proc")):
    """Return process command lines without depending on ps/pgrep output formats."""
    commands = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return commands
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command:
            commands.append((int(entry.name), command))
    return commands


def inspect_process_conflicts(commands=None):
    commands = list(commands if commands is not None else iter_process_commands())
    current_pid = os.getpid()
    simulation = []
    legacy_tunnels = []
    isolated_tunnels = []
    for pid, command in commands:
        if pid == current_pid:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        executable_names = {Path(token).name for token in tokens}
        is_simulation = (
            any(name in executable_names for name in _SIMULATION_EXECUTABLES)
            or any(token.endswith("/mujoco/simulation") for token in tokens)
            or any(token == "__node:=simulation_mujoco" for token in tokens)
        )
        is_ssh = any(Path(token).name == "ssh" for token in tokens)
        if is_simulation:
            simulation.append({"pid": pid, "command": command})
        if is_ssh and any(LEGACY_REAL_TUNNEL_FORWARD in token for token in tokens):
            legacy_tunnels.append({"pid": pid, "command": command})
        if is_ssh and any(ISOLATED_REAL_TUNNEL_FORWARD in token for token in tokens):
            isolated_tunnels.append({"pid": pid, "command": command})
    return {
        "simulation_processes": simulation,
        "legacy_real_tunnels": legacy_tunnels,
        "isolated_real_tunnels": isolated_tunnels,
    }


class SimulationManager:
    """Start and stop only the local Domain-isolated MuJoCo stack."""

    def __init__(self, settings):
        self.settings = settings
        self.workspace = Path(settings.simulation_workspace)
        self.bxi_setup = Path(settings.simulation_bxi_setup)
        self.mujoco_library_dir = Path(settings.simulation_mujoco_library_dir)
        self.install_setup = self.workspace / "install" / "setup.bash"
        self.ros_domain_id = int(settings.simulation_ros_domain_id)
        self.runtime_dir = ensure_dir(settings.output_root.parent / "simulation")
        self._lock = threading.RLock()
        self._state = "stopped"
        self._error = None
        self._updated_at = utc_now_iso()
        self._processes = {}
        self._logs = {}
        self._worker = None
        self._cancel = threading.Event()

    @property
    def available(self):
        return self.availability_error is None

    @property
    def availability_error(self):
        required = (
            (self.workspace.is_dir(), f"仿真工作区 {self.workspace}"),
            (self.bxi_setup.is_file(), f"BXI setup {self.bxi_setup}"),
            (self.install_setup.is_file(), f"ELF3 setup {self.install_setup}"),
            (
                self.mujoco_library_dir.is_dir(),
                f"MuJoCo 运行库目录 {self.mujoco_library_dir}",
            ),
            (
                Path("/opt/ros/humble/setup.bash").is_file(),
                "ROS Humble setup /opt/ros/humble/setup.bash",
            ),
        )
        missing = [description for exists, description in required if not exists]
        return "缺少" + "、".join(missing) if missing else None

    def status(self):
        with self._lock:
            self._refresh_process_state_locked()
            conflicts = inspect_process_conflicts()
            managed_pids = {
                process.pid for process in self._processes.values()
            }
            external_simulation = [
                item
                for item in conflicts["simulation_processes"]
                if item["pid"] not in managed_pids
                and self._process_group_id(item["pid"]) not in managed_pids
            ]
            realtime_limit = resource.getrlimit(resource.RLIMIT_RTPRIO)[0]
            realtime_available = realtime_limit == resource.RLIM_INFINITY or realtime_limit > 0
            return {
                "available": self.available,
                "availability_error": self.availability_error,
                "state": self._state if self.available else "unavailable",
                "error": self._error,
                "updated_at": self._updated_at,
                "ros_domain_id": self.ros_domain_id,
                "mujoco_library_dir": str(self.mujoco_library_dir),
                "simulation_endpoint": SIMULATION_SONIC_ENDPOINT,
                "real_robot_endpoint": REAL_ROBOT_SONIC_ENDPOINT,
                "managed_pids": {
                    name: process.pid
                    for name, process in self._processes.items()
                    if process.poll() is None
                },
                "external_simulation_processes": external_simulation,
                "legacy_real_tunnel_active": bool(conflicts["legacy_real_tunnels"]),
                "isolated_real_tunnel_active": bool(conflicts["isolated_real_tunnels"]),
                "realtime_scheduling_available": realtime_available,
                "warnings": [] if realtime_available else [
                    "当前服务没有实时调度权限；MuJoCo 可用于动作预检，但不能视为严格的真机时序验收。"
                ],
                "log_paths": {
                    name: str(self.runtime_dir / f"{name}.log")
                    for name in ("mujoco", "controller")
                },
            }

    def start(self):
        with self._lock:
            if not self.available:
                raise RuntimeError(
                    f"本机仿真环境不可用：{self.availability_error}。"
                )
            self._refresh_process_state_locked()
            if self._state in {"starting", "ready"}:
                return self.status()
            if self._state == "stopping":
                raise RuntimeError("仿真正在停止，请等待完成后重试。")
            conflicts = inspect_process_conflicts()
            if conflicts["legacy_real_tunnels"] or conflicts["isolated_real_tunnels"]:
                raise RuntimeError(
                    "检测到真机 SSH 隔离隧道；安全策略禁止在真机链路存在时启动仿真。"
                )
            if conflicts["simulation_processes"]:
                raise RuntimeError(
                    "检测到 Web 之外启动的 MuJoCo/ELF3 仿真进程；请先正常关闭，避免重复控制器。"
                )
            self._cancel.clear()
            self._state = "starting"
            self._error = None
            self._updated_at = utc_now_iso()
            self._worker = threading.Thread(
                target=self._start_worker,
                daemon=True,
                name="gvhmr-simulation-start",
            )
            self._worker.start()
            return self.status()

    def stop(self):
        with self._lock:
            self._refresh_process_state_locked()
            if self._state == "stopped" and not self._processes:
                return self.status()
            if self._state == "stopping":
                return self.status()
            self._cancel.set()
            self._state = "stopping"
            self._updated_at = utc_now_iso()
            self._worker = threading.Thread(
                target=self._stop_worker,
                daemon=True,
                name="gvhmr-simulation-stop",
            )
            self._worker.start()
            return self.status()

    def shutdown(self):
        self._cancel.set()
        self._stop_worker()

    def assert_can_stream(self, target, *, confirm_real=False):
        target = str(target or "").strip().lower()
        status = self.status()
        if status["legacy_real_tunnel_active"]:
            raise RuntimeError(
                "检测到旧 5557→5558 真机隧道。它可能把仿真指令转发到真机，必须先关闭。"
            )
        if target == "simulation":
            if status["isolated_real_tunnel_active"]:
                raise RuntimeError("真机隔离隧道仍在运行；关闭后才能向仿真发送动作。")
            if status["state"] != "ready":
                raise RuntimeError("MuJoCo/SONIC 仿真尚未就绪，请先点击“启动仿真”。")
            return SIMULATION_SONIC_ENDPOINT
        if target == "real":
            if not confirm_real:
                raise RuntimeError("发送到真机必须显式确认；Web 不会启动真机控制器。")
            if status["state"] != "stopped" or status["external_simulation_processes"]:
                raise RuntimeError("检测到仿真正在运行；必须完全关闭仿真后才能发送到真机。")
            if not status["isolated_real_tunnel_active"]:
                raise RuntimeError(
                    "未检测到 5559→机器人5558 的隔离隧道；Web 不会自动建立真机连接。"
                )
            return REAL_ROBOT_SONIC_ENDPOINT
        raise ValueError("SONIC target must be explicitly set to simulation or real.")

    def _start_worker(self):
        try:
            self._spawn("mujoco", "ros2 launch bxi_example_py_elf3 elf3_dof29_sim.launch.py")
            self._wait_for(
                lambda: self._ros_node_exists("/simulation_mujoco"),
                timeout=25,
                description="MuJoCo 节点",
            )
            self._spawn(
                "controller",
                "ros2 launch bxi_example_py_elf3 example_demo.launch.py start_sim:=false",
            )
            self._wait_for(
                lambda: "normal" in self._read_state().lower(),
                timeout=35,
                description="ELF3 normal 状态",
            )
            self._ros_output(
                "ros2 topic pub --times 2 --rate 5 --qos-reliability best_effort "
                "/motion_commands communication/msg/MotionCommands '{btn_10: 9}'",
                timeout=8,
            )
            self._wait_for(
                lambda: "sonic" in self._read_state().lower(),
                timeout=15,
                description="SONIC 仿真状态",
            )
            with self._lock:
                if self._cancel.is_set():
                    raise RuntimeError("仿真启动已取消。")
                self._state = "ready"
                self._error = None
                self._updated_at = utc_now_iso()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._state = "error"
                self._updated_at = utc_now_iso()
            self._stop_processes()

    def _stop_worker(self):
        try:
            if self._process_alive("controller"):
                try:
                    self._ros_output(
                        "ros2 topic pub --times 2 --rate 5 --qos-reliability best_effort "
                        "/motion_commands communication/msg/MotionCommands '{btn_1: 1}'",
                        timeout=5,
                    )
                    time.sleep(0.5)
                except Exception:
                    pass
            self._stop_processes()
        finally:
            with self._lock:
                self._state = "stopped"
                self._error = None
                self._updated_at = utc_now_iso()
                self._cancel.clear()

    def _spawn(self, name, command):
        log_path = self.runtime_dir / f"{name}.log"
        log = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            self._bash_command(command),
            cwd=self.workspace,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=self._base_env(),
        )
        with self._lock:
            self._processes[name] = process
            self._logs[name] = log
            self._updated_at = utc_now_iso()

    def _bash_command(self, command):
        setup = " && ".join(
            f"source {shlex.quote(str(path))}"
            for path in (
                Path("/opt/ros/humble/setup.bash"),
                self.bxi_setup,
                self.install_setup,
            )
        )
        script = (
            f"{setup} && export ROS_DOMAIN_ID={self.ros_domain_id} "
            "ROS_LOCALHOST_ONLY=1 ROS2CLI_NO_DAEMON=1 && exec "
            f"{command}"
        )
        return ["/bin/bash", "--noprofile", "--norc", "-c", script]

    def _base_env(self):
        env = os.environ.copy()
        # The Web process deliberately hides user site-packages, while the
        # installed ELF3 controller currently obtains onnxruntime from the
        # host user's Python 3.10 site.  Do not leak Web Python isolation or
        # its repository PYTHONPATH into the separately sourced ROS process.
        env.pop("PYTHONNOUSERSITE", None)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        env["ROS_LOCALHOST_ONLY"] = "1"
        env["ROS2CLI_NO_DAEMON"] = "1"
        if self.mujoco_library_dir.is_dir():
            current_library_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                part
                for part in (str(self.mujoco_library_dir), current_library_path)
                if part
            )
        return env

    def _ros_output(self, command, timeout=4):
        result = subprocess.run(
            self._bash_command(command),
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=self._base_env(),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout.strip() or f"ROS command failed: {command}")
        return result.stdout

    def _read_state(self):
        try:
            output = self._ros_output(
                "timeout 3 ros2 topic echo --no-daemon --once --full-length "
                "/simulation/state_machine_info std_msgs/msg/String",
                timeout=5,
            )
            message = next(yaml.safe_load_all(output), None)
            payload = message.get("data") if isinstance(message, dict) else None
            snapshot = yaml.safe_load(payload) if isinstance(payload, str) else None
            current = snapshot.get("current") if isinstance(snapshot, dict) else None
            state = current.get("name") if isinstance(current, dict) else None
            return str(state or "")
        except (RuntimeError, ValueError, yaml.YAMLError, subprocess.TimeoutExpired):
            return ""

    def _ros_node_exists(self, node_name):
        try:
            output = self._ros_output(
                "ros2 node list --no-daemon --spin-time 0.5",
                timeout=3,
            )
        except (RuntimeError, subprocess.TimeoutExpired):
            return False
        return str(node_name) in output.splitlines()

    def _wait_for(self, predicate, *, timeout, description):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise RuntimeError("仿真启动已取消。")
            for name, process in list(self._processes.items()):
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"{name} 进程提前退出，exit code={code}。")
            if predicate():
                return
            time.sleep(0.5)
        raise RuntimeError(f"等待{description}超时。")

    def _process_alive(self, name):
        process = self._processes.get(name)
        return process is not None and process.poll() is None

    def _stop_processes(self):
        for name in ("controller", "mujoco"):
            process = self._processes.get(name)
            if process is None:
                continue
            process_group = process.pid
            self._signal_process_group(process_group, signal.SIGINT)
            if not self._wait_process_group(process_group, timeout=6):
                self._signal_process_group(process_group, signal.SIGTERM)
                if not self._wait_process_group(process_group, timeout=3):
                    self._signal_process_group(process_group, signal.SIGKILL)
                    self._wait_process_group(process_group, timeout=2)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        with self._lock:
            self._processes.clear()
            for log in self._logs.values():
                try:
                    log.close()
                except OSError:
                    pass
            self._logs.clear()

    def _refresh_process_state_locked(self):
        if self._state not in {"starting", "ready"}:
            return
        exited = [
            f"{name}: exit {process.poll()}"
            for name, process in self._processes.items()
            if process.poll() is not None
        ]
        if exited:
            self._state = "error"
            self._error = "仿真进程意外退出（" + ", ".join(exited) + "）。"
            self._updated_at = utc_now_iso()
            self._stop_processes()

    @staticmethod
    def _process_group_id(pid):
        try:
            return os.getpgid(int(pid))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _signal_process_group(process_group, signal_number):
        try:
            os.killpg(process_group, signal_number)
        except ProcessLookupError:
            pass

    @classmethod
    def _wait_process_group(cls, process_group, *, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False
