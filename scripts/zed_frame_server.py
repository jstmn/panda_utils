"""Publish the newest colour + registered depth frame of each camera over ZMQ, for processes that can't run rclpy.

ROS 2 Humble's rclpy is built for Python 3.10, so Python 3.12 code (real2sim-toolkit's live tracker, Mpc-4-Manipulation)
reads the cameras through this server instead. Each camera gets its own ZMQ PUB socket. A frame is published once its
colour image and its depth image with the same header stamp have both arrived; frames are never queued, so a
subscriber that sets ZMQ_CONFLATE always reads the newest one.

Wire format, version "zed_frame_v1": one ZMQ message per frame (single part, so ZMQ_CONFLATE works), laid out as
    [uint32 little-endian header length][header, UTF-8 JSON][rgb bytes][depth bytes]
header keys:
    format       "zed_frame_v1"
    camera       camera id, e.g. "zed2i"
    stamp_ns     ROS header stamp shared by the colour and depth image, host wall clock ns since the Unix epoch
    height, width
    K            colour camera matrix from camera_info, 9 floats, row-major
    rgb_dtype    "uint8"; rgb is (height, width, 3) RGB
    depth_dtype  "uint16"; depth is (height, width) millimetres on the colour grid, 0 = no return
real2sim-toolkit's r2st.zmq_frames decodes this format.

Example usage:

# in every new terminal:
source ~/anaconda3/etc/profile.d/conda.sh; conda activate deoxys
source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash

# Terminal 1 - ZED cameras
CAMS="zed zed2i zedm" bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh

# Terminal 2
python scripts/zed_frame_server.py --cameras zed2i=tcp://127.0.0.1:5600
"""

from __future__ import annotations

import argparse
import json
import struct
import time

import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from panda_utils.ros2_cameras import (
    CAMERA_LAYOUTS,
    LAYOUTS,
    decode_color,
    decode_depth,
    make_qos,
    stamp_to_ns,
)

FRAME_FORMAT = "zed_frame_v1"
STATUS_PERIOD_S = 5.0
FIRST_FRAME_TIMEOUT_S = 20.0


def encode_frame(camera: str, stamp_ns: int, rgb: np.ndarray, depth_mm: np.ndarray, K: np.ndarray) -> bytes:
    """One `zed_frame_v1` message (see the module docstring)."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb {rgb.dtype} {rgb.shape}"
    assert depth_mm.dtype == np.uint16, f"depth dtype {depth_mm.dtype}, expected uint16 mm"
    assert depth_mm.shape == rgb.shape[:2], f"depth {depth_mm.shape} != rgb {rgb.shape[:2]}"
    assert K.shape == (3, 3), f"K shape {K.shape}"
    header = json.dumps(
        {
            "format": FRAME_FORMAT,
            "camera": camera,
            "stamp_ns": int(stamp_ns),
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "K": [float(v) for v in K.reshape(-1)],
            "rgb_dtype": "uint8",
            "depth_dtype": "uint16",
        }
    ).encode("utf-8")
    return b"".join(
        [
            struct.pack("<I", len(header)),
            header,
            np.ascontiguousarray(rgb).tobytes(),
            np.ascontiguousarray(depth_mm).tobytes(),
        ]
    )


def parse_camera_endpoint(spec: str) -> tuple[str, str]:
    """'zed2i=tcp://127.0.0.1:5600' -> ('zed2i', 'tcp://127.0.0.1:5600')."""
    cam_id, sep, endpoint = spec.partition("=")
    assert sep == "=" and cam_id and endpoint, f"expected camera=endpoint, got {spec!r}"
    assert cam_id in CAMERA_LAYOUTS, f"unknown camera {cam_id!r}; known: {sorted(CAMERA_LAYOUTS)}"
    return cam_id, endpoint


class FramePublisher(Node):
    """Subscribes to colour, registered depth and colour camera_info, and publishes stamp-matched pairs."""

    def __init__(self, endpoints: dict[str, str], best_effort: bool, context: zmq.Context):
        super().__init__("zed_frame_server")
        self._sockets: dict[str, zmq.Socket] = {}
        self._color: dict[str, tuple[int, np.ndarray] | None] = {}
        self._depth: dict[str, tuple[int, np.ndarray] | None] = {}
        self._K: dict[str, np.ndarray | None] = {}
        self._last_published_ns: dict[str, int | None] = {}
        self.n_published: dict[str, int] = {}
        qos = make_qos(best_effort)
        self._subscriptions = []
        for cam_id, endpoint in endpoints.items():
            layout_name = CAMERA_LAYOUTS[cam_id]
            assert layout_name == "zed", f"{cam_id} uses the {layout_name!r} layout; this server is for ZED cameras"
            layout = LAYOUTS[layout_name]
            ns = layout.template.format(cam=cam_id).rstrip("/")
            socket = context.socket(zmq.PUB)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.bind(endpoint)
            self._sockets[cam_id] = socket
            self._color[cam_id] = None
            self._depth[cam_id] = None
            self._K[cam_id] = None
            self._last_published_ns[cam_id] = None
            self.n_published[cam_id] = 0
            self._subscriptions += [
                self.create_subscription(Image, f"{ns}/{layout.color}", lambda m, c=cam_id: self._color_cb(m, c), qos),
                self.create_subscription(Image, f"{ns}/{layout.depth}", lambda m, c=cam_id: self._depth_cb(m, c), qos),
                self.create_subscription(
                    CameraInfo, f"{ns}/{layout.color_info}", lambda m, c=cam_id: self._info_cb(m, c), qos
                ),
            ]
            print(f"{cam_id}: {ns}/{{{layout.color}, {layout.depth}, {layout.color_info}}} -> {endpoint}")

    def _color_cb(self, msg: Image, cam_id: str) -> None:
        self._color[cam_id] = (stamp_to_ns(msg), decode_color(msg))
        self._publish_if_paired(cam_id)

    def _depth_cb(self, msg: Image, cam_id: str) -> None:
        self._depth[cam_id] = (stamp_to_ns(msg), decode_depth(msg))
        self._publish_if_paired(cam_id)

    def _info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self._K[cam_id] is None:
            self._K[cam_id] = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def _publish_if_paired(self, cam_id: str) -> None:
        color, depth, K = self._color[cam_id], self._depth[cam_id], self._K[cam_id]
        if color is None or depth is None or K is None or color[0] != depth[0]:
            return
        if color[0] == self._last_published_ns[cam_id]:
            return
        self._sockets[cam_id].send(encode_frame(cam_id, color[0], color[1], depth[1], K))
        self._last_published_ns[cam_id] = color[0]
        self.n_published[cam_id] += 1

    def status(self) -> str:
        parts = []
        for cam_id in self._sockets:
            color, depth = self._color[cam_id], self._depth[cam_id]
            color_ns = "none" if color is None else str(color[0])
            depth_ns = "none" if depth is None else str(depth[0])
            parts.append(
                f"{cam_id}: published={self.n_published[cam_id]} K={'yes' if self._K[cam_id] is not None else 'no'} "
                f"last colour stamp={color_ns} last depth stamp={depth_ns}"
            )
        return " | ".join(parts)


def main(args: argparse.Namespace) -> None:
    endpoints = dict(parse_camera_endpoint(spec) for spec in args.cameras)
    assert len(endpoints) == len(args.cameras), f"a camera is listed twice in {args.cameras}"
    context = zmq.Context()
    rclpy.init(args=None)
    node = FramePublisher(endpoints, args.qos_best_effort, context)
    t_start = time.monotonic()
    t_status = t_start
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        missing = [cam_id for cam_id, n in node.n_published.items() if n == 0]
        assert not missing or now - t_start < FIRST_FRAME_TIMEOUT_S, (
            f"no stamp-matched colour + depth frame within {FIRST_FRAME_TIMEOUT_S:g} s for {missing}. "
            f"Are the cameras launched, and do the topics exist (ros2 topic list)? If topics exist but no data "
            f"arrives, retry with --qos-best-effort. Status: {node.status()}"
        )
        if now - t_status >= STATUS_PERIOD_S:
            print(node.status(), flush=True)
            t_status = now


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish the newest ZED colour + depth frames over ZMQ.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="camera=endpoint pairs, e.g. zed2i=tcp://127.0.0.1:5600. One PUB socket per camera.",
    )
    parser.add_argument(
        "--qos-best-effort", action="store_true", help="Subscribe BEST_EFFORT (for BEST_EFFORT publishers)."
    )
    main(parser.parse_args())
