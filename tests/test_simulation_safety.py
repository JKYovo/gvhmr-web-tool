import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hmr4d.service.common import ServiceSettings
from hmr4d.service.simulation import (
    ISOLATED_REAL_TUNNEL_FORWARD,
    LEGACY_REAL_TUNNEL_FORWARD,
    REAL_ROBOT_SONIC_ENDPOINT,
    SIMULATION_SONIC_ENDPOINT,
    SimulationManager,
    inspect_process_conflicts,
)


def empty_conflicts():
    return {
        "simulation_processes": [],
        "legacy_real_tunnels": [],
        "isolated_real_tunnels": [],
    }


class SimulationSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        root = Path(self.temp_dir.name)
        workspace = root / "simulation_ws"
        (workspace / "install").mkdir(parents=True)
        (workspace / "install" / "setup.bash").write_text("", encoding="utf-8")
        bxi_setup = root / "bxi_setup.bash"
        bxi_setup.write_text("", encoding="utf-8")
        mujoco_library_dir = root / "mujoco" / "bin"
        mujoco_library_dir.mkdir(parents=True)
        settings = ServiceSettings(
            checkpoint_root=root / "checkpoints",
            output_root=root / "runtime" / "jobs",
            batch_root=root / "runtime" / "batches",
            db_path=root / "runtime" / "db" / "jobs.sqlite",
            host="127.0.0.1",
            port=7860,
            sync_assets_on_boot=False,
            simulation_workspace=workspace,
            simulation_bxi_setup=bxi_setup,
            simulation_mujoco_library_dir=mujoco_library_dir,
            simulation_ros_domain_id=73,
        )
        settings.ensure_runtime_dirs()
        self.manager = SimulationManager(settings)

    def tearDown(self):
        self.manager.shutdown()
        self.temp_dir.cleanup()

    def test_process_inspection_distinguishes_simulation_and_two_tunnels(self):
        commands = [
            (101, "ros2 launch bxi_example_py_elf3 example_demo.launch.py start_sim:=false"),
            (102, f"ssh -NT -R {LEGACY_REAL_TUNNEL_FORWARD} bxi@robot"),
            (103, f"ssh -NT -R {ISOLATED_REAL_TUNNEL_FORWARD} bxi@robot"),
        ]
        result = inspect_process_conflicts(commands)
        self.assertEqual([item["pid"] for item in result["simulation_processes"]], [101])
        self.assertEqual([item["pid"] for item in result["legacy_real_tunnels"]], [102])
        self.assertEqual([item["pid"] for item in result["isolated_real_tunnels"]], [103])

    def test_process_inspection_ignores_diagnostic_commands_that_only_mention_markers(self):
        command = (
            "bash -lc ps -ef | rg "
            "'elf3_dof29_sim|example_demo|simulation_mujoco|555[79]'"
        )
        result = inspect_process_conflicts([(104, command)])
        self.assertEqual(result, empty_conflicts())

    def test_simulation_stream_requires_ready_and_no_real_tunnel(self):
        with patch("hmr4d.service.simulation.inspect_process_conflicts", return_value=empty_conflicts()):
            with self.assertRaisesRegex(RuntimeError, "尚未就绪"):
                self.manager.assert_can_stream("simulation")
            self.manager._state = "ready"
            self.assertEqual(
                self.manager.assert_can_stream("simulation"),
                SIMULATION_SONIC_ENDPOINT,
            )

        conflicts = empty_conflicts()
        conflicts["isolated_real_tunnels"] = [{"pid": 5, "command": "ssh"}]
        with patch("hmr4d.service.simulation.inspect_process_conflicts", return_value=conflicts):
            with self.assertRaisesRegex(RuntimeError, "真机隔离隧道"):
                self.manager.assert_can_stream("simulation")

    def test_real_stream_requires_confirmation_isolated_tunnel_and_stopped_sim(self):
        conflicts = empty_conflicts()
        conflicts["isolated_real_tunnels"] = [{"pid": 6, "command": "ssh"}]
        with patch("hmr4d.service.simulation.inspect_process_conflicts", return_value=conflicts):
            with self.assertRaisesRegex(RuntimeError, "显式确认"):
                self.manager.assert_can_stream("real", confirm_real=False)
            self.assertEqual(
                self.manager.assert_can_stream("real", confirm_real=True),
                REAL_ROBOT_SONIC_ENDPOINT,
            )
            self.manager._state = "ready"
            with self.assertRaisesRegex(RuntimeError, "仿真正在运行"):
                self.manager.assert_can_stream("real", confirm_real=True)

    def test_legacy_tunnel_blocks_both_targets(self):
        conflicts = empty_conflicts()
        conflicts["legacy_real_tunnels"] = [{"pid": 7, "command": "ssh"}]
        with patch("hmr4d.service.simulation.inspect_process_conflicts", return_value=conflicts):
            self.manager._state = "ready"
            with self.assertRaisesRegex(RuntimeError, "旧 5557"):
                self.manager.assert_can_stream("simulation")
            self.manager._state = "stopped"
            with self.assertRaisesRegex(RuntimeError, "旧 5557"):
                self.manager.assert_can_stream("real", confirm_real=True)

    def test_local_launch_command_is_domain_and_loopback_isolated(self):
        command = self.manager._bash_command(
            "ros2 launch bxi_example_py_elf3 example_demo.launch.py start_sim:=false"
        )[-1]
        self.assertIn("ROS_DOMAIN_ID=73", command)
        self.assertIn("ROS_LOCALHOST_ONLY=1", command)
        self.assertIn("example_demo.launch.py", command)
        self.assertNotIn("example_demo_hw.launch.py", command)
        self.assertNotIn("ssh ", command)
        self.assertTrue(
            self.manager._base_env()["LD_LIBRARY_PATH"].startswith(
                str(self.manager.mujoco_library_dir)
            )
        )

    def test_state_reader_uses_full_ros_string_and_returns_exact_active_state(self):
        snapshot = (
            "data: '{\"graph\": {\"states\": [\"normal\", \"sonic_teleop\"]}, "
            "\"current\": {\"id\": 1876263228, "
            "\"name\": \"com.bxi.basic_actions/normal\"}}'\n---\n"
        )
        with patch.object(self.manager, "_ros_output", return_value=snapshot) as ros_output:
            self.assertEqual(
                self.manager._read_state(),
                "com.bxi.basic_actions/normal",
            )
        self.assertIn("--full-length", ros_output.call_args.args[0])
        self.assertIn("--no-daemon", ros_output.call_args.args[0])

    def test_transient_node_discovery_timeout_is_not_fatal(self):
        with patch.object(
            self.manager,
            "_ros_output",
            side_effect=subprocess.TimeoutExpired("ros2 node list", 3),
        ):
            self.assertFalse(self.manager._ros_node_exists("/simulation_mujoco"))


if __name__ == "__main__":
    unittest.main()
