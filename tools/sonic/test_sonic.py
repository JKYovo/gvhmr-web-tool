# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the WebTool-local SONIC adapter; no real socket is opened."""

from __future__ import annotations

import json
import threading
import unittest
from collections import deque

import numpy as np

from hmr4d.utils.sonic import (
    HEADER_SIZE,
    PlaybackState,
    SonicPlaybackController,
    SonicReference,
    build_smpl_ref_fields,
    convert_smplx_to_sonic,
    pack_smpl_ref_message,
)


def identity_motion(frames: int) -> np.ndarray:
    return np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 22, 3, 3)).copy()


def reference(frames: int = 3) -> SonicReference:
    return SonicReference(
        term1_local=np.arange(frames * 72, dtype=np.float32).reshape(frames, 72),
        root_quat=np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (frames, 1)),
        wrist=np.zeros((frames, 6), dtype=np.float32),
        fps=500.0,
    )


class FakeSocket:
    def __init__(self, *, subscribed=True):
        self.endpoint = None
        self.messages = []
        self.closed = False
        self.options = []
        self.events = deque([b"\x01smpl_ref"] if subscribed else [])

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def bind(self, endpoint):
        self.endpoint = endpoint

    def send(self, message):
        self.messages.append(message)

    def poll(self, timeout=0, flags=0):
        return int(bool(self.events))

    def recv(self, flags=0):
        if not self.events:
            import zmq

            raise zmq.Again()
        return self.events.popleft()

    def disconnect_subscriber(self):
        self.events.append(b"\x00smpl_ref")

    def close(self, linger=0):
        self.closed = True


class SonicTests(unittest.TestCase):
    def test_conversion_shape_and_validation(self):
        converted = convert_smplx_to_sonic(identity_motion(4), 30.0)
        self.assertEqual(converted.term1_local.shape, (6, 72))
        self.assertEqual(converted.root_quat.shape, (6, 4))
        self.assertEqual(converted.wrist.shape, (6, 6))
        with self.assertRaises(ValueError):
            convert_smplx_to_sonic(np.zeros((3, 21, 3, 3)), 30.0)

    def test_protocol_window_and_header(self):
        fields = build_smpl_ref_fields(reference(), 2)
        self.assertEqual(fields["term1_local"].shape, (10, 72))
        message = pack_smpl_ref_message(fields)
        offset = len(b"smpl_ref")
        header = json.loads(message[offset : offset + HEADER_SIZE].rstrip(b"\0"))
        self.assertEqual(header["v"], 4)
        self.assertEqual(header["endian"], "le")

    def test_fake_socket_playback(self):
        socket = FakeSocket()
        done = threading.Event()
        states = []

        def callback(state, frame_index, frame_count, message):
            states.append(state)
            if state in (PlaybackState.COMPLETE, PlaybackState.ERROR):
                done.set()

        controller = SonicPlaybackController(
            endpoint="tcp://test:5557",
            pre_roll_seconds=0,
            final_hold_seconds=0,
            socket_factory=lambda: socket,
        )
        controller.run(0, reference(), callback)
        self.assertTrue(done.wait(2))
        controller.close()
        self.assertEqual(socket.endpoint, "tcp://test:5557")
        self.assertEqual(len(socket.messages), 4)
        self.assertTrue(socket.closed)
        self.assertEqual(states[-1], PlaybackState.COMPLETE)

    def test_stop_ends_live_reference_without_repeating_last_frame(self):
        socket = FakeSocket()
        streaming = threading.Event()
        stopped = threading.Event()
        states = []

        def callback(state, frame_index, frame_count, message):
            states.append(state)
            if state == PlaybackState.STREAMING:
                streaming.set()
            elif state == PlaybackState.STOPPED:
                stopped.set()

        controller = SonicPlaybackController(
            endpoint="tcp://test:5557",
            pre_roll_seconds=0,
            final_hold_seconds=0,
            socket_factory=lambda: socket,
        )
        controller.run(0, reference(500), callback)
        self.assertTrue(streaming.wait(2))
        self.assertTrue(controller.stop(0))
        self.assertTrue(stopped.wait(2))
        sent_after_stop = len(socket.messages)
        threading.Event().wait(0.02)
        self.assertEqual(len(socket.messages), sent_after_stop)
        self.assertTrue(socket.closed)
        self.assertEqual(states[-1], PlaybackState.STOPPED)

    def test_playback_requires_sonic_subscriber(self):
        socket = FakeSocket(subscribed=False)
        done = threading.Event()
        errors = []

        def callback(state, _frame_index, _frame_count, message):
            if state == PlaybackState.ERROR:
                errors.append(message)
                done.set()

        controller = SonicPlaybackController(
            endpoint="tcp://test:5557",
            subscriber_timeout_seconds=0.02,
            socket_factory=lambda: socket,
        )
        controller.run(0, reference(), callback)
        self.assertTrue(done.wait(2))
        self.assertEqual(socket.messages, [])
        self.assertIn("sonic_teleop", errors[-1])

    def test_unsubscribe_terminates_current_stream_and_cannot_resume(self):
        socket = FakeSocket()
        streaming = threading.Event()
        failed = threading.Event()
        errors = []

        def callback(state, _frame_index, _frame_count, message):
            if state == PlaybackState.STREAMING:
                streaming.set()
            elif state == PlaybackState.ERROR:
                errors.append(message)
                failed.set()

        controller = SonicPlaybackController(
            endpoint="tcp://test:5557",
            pre_roll_seconds=0,
            final_hold_seconds=0,
            socket_factory=lambda: socket,
        )
        controller.run(0, reference(500), callback)
        self.assertTrue(streaming.wait(2))
        socket.disconnect_subscriber()
        self.assertTrue(failed.wait(2))
        sent_after_disconnect = len(socket.messages)
        threading.Event().wait(0.02)
        self.assertEqual(len(socket.messages), sent_after_disconnect)
        self.assertIn("安全终止", errors[-1])


if __name__ == "__main__":
    unittest.main()
