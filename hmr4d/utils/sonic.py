# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert GVHMR SMPL-X motions and stream them to an ELF3 SONIC policy.

This module is derived from ``kimodo.integrations.sonic``.  It is kept local
so GVHMR-to-SONIC conversion and playback do not require a Kimodo checkout or
Python environment.  The protocol, coordinate transforms, interpolation and
minimal SMPL-X rest skeleton are intentionally unchanged for exact backward
compatibility.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import json
import math
import os
import threading
import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp


SONIC_TARGET_FPS = 50.0
SONIC_WINDOW = 10
SONIC_TOPIC = "smpl_ref"
SONIC_PROTOCOL_VERSION = 4
HEADER_SIZE = 1280
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5557"

_SMPLX_JOINT_COUNT = 22
_SONIC_OUTPUT_JOINTS = np.concatenate((np.arange(22), np.array([39, 54])))
_SMPL_BASE_INVERSE_WXYZ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
_Y_UP_TO_Z_UP = Rotation.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64))

# Minimal data extracted without numeric conversion from Kimodo's
# assets/skeletons/smplx22/SMPLX_NEUTRAL.npz.  Embedding only J and the parent
# row avoids copying the 137 MB body model while preserving old SONIC output.
_REST_JOINTS_F32_LE = (
    "lq9MO63rs771NEU8/CJ7PV9q477uy2S8yFl2vR0f6b6U9Ra8OAq9OTJQd77MR3+8qJXtPSurUr/x"
    "Xr+8o7fVvX5UUb8FTdW82bIgPFCX4L3yTLC8ipeUPQntnL/SP2K9zCS2vfo8nb+jWz29G4PHuhk6"
    "a70a8uI7+1/1PX9ZpL99+4A91dACvki0pL8kIpU90j1gvBCm3D2oQcq8QKw3PbZn4Twne5q52pdJ"
    "vdVy3DxrJNS7VPI2PAqBiT4bM2u72wQoPgqUrj3dEYG8GnAbvu66pD3w0Jy80h7WPhuDVjxEcm69"
    "K4zYvrr8Mz050Tq9nZErPwC+FD1vkni9ExYsvwFsIT3Xlnm9XJufu+BBiT5kmxW8QG4BPWMlnz5e"
    "wH49324BvV0lnz5/v34926dFP6RP4jxLTym9o9VNP6579Dyc+hy9I5tTPzU74TxmwRy9F5NHP+ml"
    "9TxQcIS9O2lPPypD/DxXrYy9UW5VP4bO6zxeGpW9thVBP85gsjyd4tW9mQtFP7gbqTzEde69lwVJ"
    "P994mzxZtwK+tINEP1qQ3TyySbS9zLFLP5q76TyREr+9c4lRP1rJ3TwFms29ufg1PxM4ljx7qw+9"
    "2lM6P4I2njz/byW83JQ/P0LkZzz+yLE72adFv49R4jzPTim9ndVNv8989Dy5+Ry9HptTv8094Tyi"
    "vxy9FpNHv8Ko9TwEcYS9OmlPvypG/Dwiroy9U25Vv0zQ6zwXG5W9xRVBv+xhsjx649W9pAtFv80b"
    "qTxIdu69pAVJv5l4mzxutwK+vINEvymT3Tz4SrS9zbFLv9i96TzOE7+9eIlRv4vK3Txzms29oPg1"
    "v74zljxAqQ+9v1M6v8UynjxzaSW85ZQ/v6zmZzxDybE7"
)
_PARENTS_I64_LE = (
    "//////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAAEAAAAAAAAAAUA"
    "AAAAAAAABgAAAAAAAAAHAAAAAAAAAAgAAAAAAAAACQAAAAAAAAAJAAAAAAAAAAkAAAAAAAAADAAAAAAAAAANAAAA"
    "AAAAAA4AAAAAAAAAEAAAAAAAAAARAAAAAAAAABIAAAAAAAAAEwAAAAAAAAAPAAAAAAAAAA8AAAAAAAAADwAAAAAA"
    "AAAUAAAAAAAAABkAAAAAAAAAGgAAAAAAAAAUAAAAAAAAABwAAAAAAAAAHQAAAAAAAAAUAAAAAAAAAB8AAAAAAAAA"
    "IAAAAAAAAAAUAAAAAAAAACIAAAAAAAAAIwAAAAAAAAAUAAAAAAAAACUAAAAAAAAAJgAAAAAAAAAVAAAAAAAAACgA"
    "AAAAAAAAKQAAAAAAAAAVAAAAAAAAACsAAAAAAAAALAAAAAAAAAAVAAAAAAAAAC4AAAAAAAAALwAAAAAAAAAVAAAA"
    "AAAAADEAAAAAAAAAMgAAAAAAAAAVAAAAAAAAADQAAAAAAAAANQAAAAAAAAA="
)

FloatArray = NDArray[np.float32]


def _minimal_smplx_joint_info() -> tuple[np.ndarray, np.ndarray]:
    joints = np.frombuffer(base64.b64decode(_REST_JOINTS_F32_LE), dtype="<f4").reshape(55, 3)
    parents = np.frombuffer(base64.b64decode(_PARENTS_I64_LE), dtype="<i8").copy()
    if joints.shape != (55, 3) or parents.shape != (55,):
        raise RuntimeError("Embedded SONIC SMPL-X skeleton is corrupt")
    return joints, parents


@dataclass(frozen=True)
class SonicReference:
    term1_local: FloatArray
    root_quat: FloatArray
    wrist: FloatArray
    fps: float = SONIC_TARGET_FPS

    def __post_init__(self) -> None:
        term1 = np.asarray(self.term1_local)
        root = np.asarray(self.root_quat)
        wrist = np.asarray(self.wrist)
        if term1.ndim != 2 or term1.shape[1] != 72:
            raise ValueError(f"term1_local must have shape [T, 72], got {term1.shape}")
        frame_count = term1.shape[0]
        if frame_count < 1:
            raise ValueError("SONIC reference must contain at least one frame")
        if root.shape != (frame_count, 4):
            raise ValueError(f"root_quat must have shape {(frame_count, 4)}, got {root.shape}")
        if wrist.shape != (frame_count, 6):
            raise ValueError(f"wrist must have shape {(frame_count, 6)}, got {wrist.shape}")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {self.fps}")
        if not all(np.isfinite(array).all() for array in (term1, root, wrist)):
            raise ValueError("SONIC reference contains non-finite values")
        if np.any(np.linalg.norm(root, axis=1) <= 1.0e-6):
            raise ValueError("SONIC reference contains an invalid root quaternion")

    @property
    def frame_count(self) -> int:
        return int(self.term1_local.shape[0])


class PlaybackState(str, Enum):
    PREPARING = "preparing"
    STREAMING = "streaming"
    COMPLETE = "complete"
    STOPPED = "stopped"
    ERROR = "error"


PlaybackCallback = Callable[[PlaybackState, int, int, str | None], None]


def _validate_local_rotations(local_rot_mats: np.ndarray) -> np.ndarray:
    rotations = np.asarray(local_rot_mats)
    expected_tail = (_SMPLX_JOINT_COUNT, 3, 3)
    if rotations.ndim != 4 or rotations.shape[1:] != expected_tail:
        raise ValueError(f"SMPL-X local rotations must have shape [T, 22, 3, 3], got {rotations.shape}")
    if rotations.shape[0] < 1 or not np.isfinite(rotations).all():
        raise ValueError("SMPL-X local rotations must contain finite frames")
    return rotations.astype(np.float64, copy=False)


def _resample_rotations(local_rot_mats: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"source_fps must be positive and finite, got {source_fps}")
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive and finite, got {target_fps}")
    frame_count = local_rot_mats.shape[0]
    if frame_count == 1:
        return local_rot_mats.copy()
    duration = (frame_count - 1) / float(source_fps)
    target_count = max(2, int(math.floor(duration * target_fps + 0.5)) + 1)
    source_times = np.arange(frame_count, dtype=np.float64) / float(source_fps)
    target_times = np.linspace(0.0, duration, target_count, dtype=np.float64)
    result = np.empty((target_count, _SMPLX_JOINT_COUNT, 3, 3), dtype=np.float64)
    for joint_index in range(_SMPLX_JOINT_COUNT):
        source_rotations = Rotation.from_matrix(local_rot_mats[:, joint_index])
        result[:, joint_index] = Slerp(source_times, source_rotations)(target_times).as_matrix()
    return result


def _compute_sonic_body_reference(body_pose: np.ndarray, root_y_up: Rotation) -> tuple[np.ndarray, np.ndarray]:
    frame_count = body_pose.shape[0]
    root_z_up = _Y_UP_TO_Z_UP * root_y_up
    root_z_up_rotvec = root_z_up.as_rotvec()
    rest_joints, parents = _minimal_smplx_joint_info()
    zeros = np.zeros((frame_count, 99), dtype=np.float64)
    full_pose = np.concatenate((root_z_up_rotvec, body_pose.reshape(frame_count, 63), zeros), axis=-1)
    rotation_matrices = Rotation.from_rotvec(full_pose.reshape(-1, 3)).as_matrix().reshape(frame_count, 55, 3, 3)
    relative_joints = np.broadcast_to(rest_joints, (frame_count, *rest_joints.shape)).astype(np.float64).copy()
    relative_joints[:, 1:] -= rest_joints[parents[1:]]
    local_transforms = np.zeros((frame_count, 55, 4, 4), dtype=np.float64)
    local_transforms[:, :, :3, :3] = rotation_matrices
    local_transforms[:, :, :3, 3] = relative_joints
    local_transforms[:, :, 3, 3] = 1.0
    global_transforms = np.empty_like(local_transforms)
    global_transforms[:, 0] = local_transforms[:, 0]
    for joint_index in range(1, len(parents)):
        global_transforms[:, joint_index] = np.matmul(global_transforms[:, parents[joint_index]], local_transforms[:, joint_index])
    joints = np.take(global_transforms[:, :, :3, 3], _SONIC_OUTPUT_JOINTS, axis=1)
    base_inverse = Rotation.from_quat(_SMPL_BASE_INVERSE_WXYZ, scalar_first=True)
    root_robot = root_z_up * base_inverse
    local_joints = np.einsum("tij,tkj->tki", root_robot.inv().as_matrix(), joints)
    return local_joints.reshape(frame_count, 72).astype(np.float32), root_robot.as_quat(scalar_first=True).astype(np.float32)


def _decompose_swing(rotation_axis_angle: np.ndarray) -> Rotation:
    quaternions = Rotation.from_rotvec(np.asarray(rotation_axis_angle, dtype=np.float64)).as_quat(scalar_first=True)
    twist = np.zeros_like(quaternions)
    twist[:, 0] = quaternions[:, 0]
    twist[:, 2] = quaternions[:, 2]
    norms = np.linalg.norm(twist, axis=1, keepdims=True)
    degenerate = norms[:, 0] < 1.0e-12
    twist[~degenerate] /= norms[~degenerate]
    twist[degenerate] = np.array([1.0, 0.0, 0.0, 0.0])
    twist_inverse = twist * np.array([1.0, -1.0, -1.0, -1.0])
    return Rotation.from_quat(twist_inverse, scalar_first=True) * Rotation.from_quat(quaternions, scalar_first=True)


def _compute_wrist_reference(body_pose: np.ndarray) -> np.ndarray:
    left_elbow_swing = _decompose_swing(body_pose[:, 17]).as_euler("XYZ", degrees=False)
    right_elbow_swing = _decompose_swing(body_pose[:, 18]).as_euler("XYZ", degrees=False)
    left_wrist = Rotation.from_rotvec(body_pose[:, 19]).as_euler("XYZ", degrees=False)
    right_wrist = Rotation.from_rotvec(body_pose[:, 20]).as_euler("XYZ", degrees=False)
    wrist = np.empty((body_pose.shape[0], 6), dtype=np.float32)
    wrist[:, 0] = left_elbow_swing[:, 0] + left_wrist[:, 0]
    wrist[:, 1] = left_wrist[:, 1]
    wrist[:, 2] = left_elbow_swing[:, 2] + left_wrist[:, 2]
    wrist[:, 3] = -(right_elbow_swing[:, 0] + right_wrist[:, 0])
    wrist[:, 4] = -right_wrist[:, 1]
    wrist[:, 5] = right_elbow_swing[:, 2] + right_wrist[:, 2]
    return wrist


def convert_smplx_to_sonic(local_rot_mats: np.ndarray, source_fps: float, target_fps: float = SONIC_TARGET_FPS) -> SonicReference:
    local_rotations = _validate_local_rotations(local_rot_mats)
    resampled = _resample_rotations(local_rotations, source_fps, target_fps)
    root_y_up = Rotation.from_matrix(resampled[:, 0])
    body_pose = Rotation.from_matrix(resampled[:, 1:].reshape(-1, 3, 3)).as_rotvec().reshape(resampled.shape[0], 21, 3)
    term1_local, root_quat = _compute_sonic_body_reference(body_pose, root_y_up)
    wrist = _compute_wrist_reference(body_pose)
    return SonicReference(
        term1_local=np.ascontiguousarray(term1_local, dtype=np.float32),
        root_quat=np.ascontiguousarray(root_quat, dtype=np.float32),
        wrist=np.ascontiguousarray(wrist, dtype=np.float32),
        fps=float(target_fps),
    )


def build_smpl_ref_fields(reference: SonicReference, frame_index: int) -> dict[str, np.ndarray]:
    start = int(frame_index)
    if start < 0 or start >= reference.frame_count:
        raise IndexError(f"frame_index {frame_index} is outside [0, {reference.frame_count - 1}]")
    indices = np.minimum(np.arange(start, start + SONIC_WINDOW), reference.frame_count - 1)
    return {
        "term1_local": np.ascontiguousarray(reference.term1_local[indices], dtype=np.float32),
        "root_quat": np.ascontiguousarray(reference.root_quat[indices], dtype=np.float32),
        "wrist": np.ascontiguousarray(reference.wrist[indices], dtype=np.float32),
        "frame_index": np.array([start], dtype=np.int64),
        "source_ready": np.array([True], dtype=bool),
        "source_stream_mode": np.array([1], dtype=np.int32),
        "source_calibration_ready": np.array([True], dtype=bool),
    }


def _dtype_name(value: np.ndarray) -> tuple[str, np.ndarray]:
    names = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64", np.dtype(np.int32): "i32", np.dtype(np.int64): "i64", np.dtype(np.uint8): "u8", np.dtype(bool): "bool"}
    name = names.get(value.dtype)
    return (name, value) if name is not None else ("f32", value.astype(np.float32))


def pack_smpl_ref_message(fields: dict[str, np.ndarray]) -> bytes:
    header_fields: list[dict[str, object]] = []
    binary_data: list[bytes] = []
    for name, raw_value in fields.items():
        if not isinstance(raw_value, np.ndarray):
            continue
        dtype_name, value = _dtype_name(raw_value)
        value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        header_fields.append({"name": name, "dtype": dtype_name, "shape": list(value.shape)})
        binary_data.append(value.tobytes())
    header = {"v": SONIC_PROTOCOL_VERSION, "endian": "le", "count": 1, "fields": header_fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"SONIC header is too large: {len(header_json)} > {HEADER_SIZE}")
    return SONIC_TOPIC.encode("utf-8") + header_json.ljust(HEADER_SIZE, b"\x00") + b"".join(binary_data)


class SonicPlaybackController:
    def __init__(
        self,
        endpoint: str | None = None,
        *,
        pre_roll_seconds: float = 0.2,
        final_hold_seconds: float = 0.2,
        subscriber_timeout_seconds: float = 2.0,
        socket_factory: Callable[[], object] | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("SONIC_SMPL_REF_ENDPOINT", DEFAULT_ENDPOINT)
        self.pre_roll_seconds = float(pre_roll_seconds)
        self.final_hold_seconds = float(final_hold_seconds)
        self.subscriber_timeout_seconds = float(subscriber_timeout_seconds)
        if not math.isfinite(self.subscriber_timeout_seconds) or self.subscriber_timeout_seconds <= 0.0:
            raise ValueError("subscriber_timeout_seconds must be positive and finite")
        self._socket_factory = socket_factory
        self._lock = threading.Lock()
        self._owner_client_id: int | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def run(self, client_id: int, reference: SonicReference, callback: PlaybackCallback | None = None) -> None:
        with self._lock:
            active = self._thread is not None and self._thread.is_alive()
            owner = self._owner_client_id
        if active and owner != client_id:
            raise RuntimeError("Another browser session is currently streaming to SONIC")
        if active:
            self.stop(client_id)
        stop_event = threading.Event()
        thread = threading.Thread(target=self._play, args=(int(client_id), reference, stop_event, callback), name=f"sonic-playback-{client_id}", daemon=True)
        with self._lock:
            self._owner_client_id, self._stop_event, self._thread = int(client_id), stop_event, thread
        thread.start()

    def stop(self, client_id: int | None = None) -> bool:
        with self._lock:
            if self._thread is None or (client_id is not None and self._owner_client_id != client_id):
                return False
            thread, stop_event = self._thread, self._stop_event
        if stop_event is not None:
            stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=2.0)
        return True

    def close(self) -> None:
        self.stop(None)

    def _make_socket(self) -> object:
        if self._socket_factory is not None:
            return self._socket_factory()
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for SONIC playback") from exc
        # XPUB is wire-compatible with SUB but exposes subscription lifecycle.
        # The receiver exists only while the robot is in sonic_teleop, so an
        # unsubscribe is a hard mode interlock rather than a recoverable gap.
        socket = zmq.Context.instance().socket(zmq.XPUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        return socket

    @staticmethod
    def _read_subscription_events(socket: object, connected: bool) -> tuple[bool, bool]:
        import zmq

        disconnected = False
        topic = SONIC_TOPIC.encode("utf-8")
        while socket.poll(timeout=0, flags=zmq.POLLIN):
            try:
                event = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if not event or event[1:] != topic:
                continue
            if event[0] == 1:
                connected = True
            elif event[0] == 0:
                connected = False
                disconnected = True
        return connected, disconnected

    def _wait_for_subscriber(self, socket: object, stop_event: threading.Event) -> bool:
        deadline = time.monotonic() + self.subscriber_timeout_seconds
        connected = False
        while time.monotonic() < deadline:
            connected, _ = self._read_subscription_events(socket, connected)
            if connected:
                return True
            if stop_event.wait(0.01):
                return False
        raise RuntimeError(
            "SONIC 接收端未连接；只有处于 sonic_teleop 模式时才能开始推流。"
        )

    def _assert_subscriber_active(self, socket: object, connected: bool) -> bool:
        connected, disconnected = self._read_subscription_events(socket, connected)
        if disconnected or not connected:
            raise RuntimeError(
                "SONIC 接收端已退出 sonic_teleop；当前推流已安全终止，重新进入后必须重新发送。"
            )
        return connected

    @staticmethod
    def _notify(callback: PlaybackCallback | None, state: PlaybackState, frame_index: int, frame_count: int, message: str | None = None) -> None:
        if callback is not None:
            try:
                callback(state, frame_index, frame_count, message)
            except Exception:
                pass

    @staticmethod
    def _wait_until(stop_event: threading.Event, deadline: float) -> bool:
        return stop_event.wait(max(0.0, deadline - time.monotonic()))

    def _play(self, client_id: int, reference: SonicReference, stop_event: threading.Event, callback: PlaybackCallback | None) -> None:
        socket = None
        terminal_state = PlaybackState.STOPPED
        try:
            self._notify(callback, PlaybackState.PREPARING, 0, reference.frame_count)
            socket = self._make_socket()
            socket.bind(self.endpoint)
            subscriber_connected = self._wait_for_subscriber(socket, stop_event)
            if not subscriber_connected:
                return
            interval = 1.0 / reference.fps
            first_message = pack_smpl_ref_message(build_smpl_ref_fields(reference, 0))
            deadline = time.monotonic()
            for _ in range(max(1, int(math.ceil(self.pre_roll_seconds * reference.fps)))):
                if stop_event.is_set():
                    return
                subscriber_connected = self._assert_subscriber_active(
                    socket, subscriber_connected
                )
                socket.send(first_message)
                deadline += interval
                if self._wait_until(stop_event, deadline):
                    return
            deadline = time.monotonic()
            for frame_index in range(reference.frame_count):
                if stop_event.is_set():
                    return
                subscriber_connected = self._assert_subscriber_active(
                    socket, subscriber_connected
                )
                socket.send(pack_smpl_ref_message(build_smpl_ref_fields(reference, frame_index)))
                if frame_index == 0 or (frame_index + 1) % 5 == 0 or frame_index + 1 == reference.frame_count:
                    self._notify(callback, PlaybackState.STREAMING, frame_index + 1, reference.frame_count)
                deadline += interval
                if self._wait_until(stop_event, deadline):
                    return
            final_message = pack_smpl_ref_message(build_smpl_ref_fields(reference, reference.frame_count - 1))
            for _ in range(max(0, int(math.ceil(self.final_hold_seconds * reference.fps)))):
                if stop_event.is_set():
                    return
                subscriber_connected = self._assert_subscriber_active(
                    socket, subscriber_connected
                )
                socket.send(final_message)
                deadline += interval
                if self._wait_until(stop_event, deadline):
                    return
            terminal_state = PlaybackState.COMPLETE
        except Exception as exc:
            terminal_state = PlaybackState.ERROR
            self._notify(callback, PlaybackState.ERROR, 0, reference.frame_count, str(exc))
        finally:
            if socket is not None:
                try:
                    socket.close(linger=0)
                except TypeError:
                    socket.close()
            if terminal_state != PlaybackState.ERROR:
                self._notify(callback, terminal_state, reference.frame_count if terminal_state == PlaybackState.COMPLETE else 0, reference.frame_count)
            with self._lock:
                if self._stop_event is stop_event:
                    self._owner_client_id = self._stop_event = self._thread = None
