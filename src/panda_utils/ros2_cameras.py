"""ROS 2 camera topics, decoding, and a latest-frame subscriber, shared by the ROS 2 scripts.

Covers this rig's launchers: ZED via ~/Desktop/ros2_ws/zed_tools/launch_all.sh (/zed_multi/<cam>/...) and RealSense
via realsense2_camera (/camera/<cam>/...). rgb_subscriber_ros2_zed.py uses the stock single-camera ZED convention
(/zed_<id>/zed_node/...), which does not match this rig.
"""

from __future__ import annotations

import time
from collections import namedtuple

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

# Topic layouts, one per camera vendor. ZED and realsense2_camera agree on message types and
# on nothing else: the names differ, and so does which topic carries colour-registered depth.
#
#   ZED        zed_tools/launch_all.sh -> namespace:=zed_multi, camera_name:=<zed|zed2i|zedm>.
#              `depth/depth_registered` is 32FC1 metres, already on the left-rgb grid.
#   RealSense  ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera \
#                  camera_name:=d435 align_depth.enable:=true -> /camera/d435/...
#              `aligned_depth_to_color/image_raw` is 16UC1 millimetres on the colour grid.
#              The raw `depth/image_rect_raw` is on the DEPTH sensor's grid and must not be
#              used for depth: real2sim-toolkit's merged layout, and decode_depth below, both
#              assume depth and colour share rays. align_depth.enable:=true is not optional.
#
# decode_depth handles both encodings already, so a layout is the only thing that varies.
CameraLayout = namedtuple("CameraLayout", "template color depth color_info depth_info")

LAYOUTS = {
    "zed": CameraLayout(
        template="/zed_multi/{cam}",
        color="rgb/image_rect_color",
        depth="depth/depth_registered",
        color_info="rgb/camera_info",
        depth_info="depth/camera_info",
    ),
    "realsense": CameraLayout(
        template="/camera/{cam}",
        color="color/image_raw",
        depth="aligned_depth_to_color/image_raw",
        color_info="color/camera_info",
        depth_info="aligned_depth_to_color/camera_info",
    ),
}
DEFAULT_LAYOUT = "zed"


def parse_camera_spec(spec: str) -> tuple[str, str]:
    """'zed2i' -> ('zed2i', 'zed');  'd435=realsense' -> ('d435', 'realsense')."""
    cam_id, _, layout = spec.partition("=")
    layout = layout or DEFAULT_LAYOUT
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r} in {spec!r}; known: {sorted(LAYOUTS)}")
    return cam_id, layout


def make_qos(best_effort: bool) -> QoSProfile:
    """QoS for ZED image topics.

    The wrapper defaults to RELIABLE. A RELIABLE subscriber never receives from a
    BEST_EFFORT publisher, which shows up as "no data" with no error at all - hence the
    --qos-best-effort escape hatch.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )


def decode_color(msg: Image) -> np.ndarray:
    """ZED colour Image -> HxWx3 uint8 RGB."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)
        img = img[:, :, [2, 1, 0]] if msg.encoding == "bgra8" else img[:, :, :3]
    elif msg.encoding in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    else:
        raise ValueError(f"unsupported colour encoding: {msg.encoding}")
    return np.ascontiguousarray(img)


def decode_depth(msg: Image) -> np.ndarray:
    """ZED depth Image -> HxW uint16 millimetres.

    ZED publishes 32FC1 metres by default; OpenNI mode publishes 16UC1 millimetres.
    Both are normalised to uint16 mm so the HDF5 matches the RealSense pipeline.
    """
    if msg.encoding in ("32FC1", "TYPE_32FC1"):
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        # NaN/inf mark invalid returns in ZED depth; store as 0 rather than wrapping.
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


class CameraSubscriber(Node):
    """Keeps the latest RGB frame and the colour intrinsics of each camera. Spin it from a background thread."""

    def __init__(self, camera_layouts: dict[str, str], best_effort: bool = False, node_name: str = "camera_subscriber"):
        super().__init__(node_name)
        self.camera_ids = list(camera_layouts)
        self._latest: dict[str, np.ndarray | None] = {c: None for c in self.camera_ids}
        self._K: dict[str, np.ndarray | None] = {c: None for c in self.camera_ids}
        qos = make_qos(best_effort)
        self._subscribers = []
        for cam_id, layout_name in camera_layouts.items():
            layout = LAYOUTS[layout_name]
            ns = layout.template.format(cam=cam_id).rstrip("/")
            self._subscribers += [
                self.create_subscription(Image, f"{ns}/{layout.color}", lambda m, c=cam_id: self._color_cb(m, c), qos),
                self.create_subscription(
                    CameraInfo, f"{ns}/{layout.color_info}", lambda m, c=cam_id: self._info_cb(m, c), qos
                ),
            ]

    def _color_cb(self, msg: Image, cam_id: str) -> None:
        try:
            self._latest[cam_id] = decode_color(msg)
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")

    def _info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self._K[cam_id] is None:
            self._K[cam_id] = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def latest_rgb(self, cam_id: str) -> np.ndarray | None:
        return self._latest[cam_id]

    def intrinsics(self, cam_id: str) -> np.ndarray | None:
        """3x3 colour camera matrix, or None before the first camera_info arrives."""
        return self._K[cam_id]

    def wait_for_frames(self, timeout_s: float = 20.0) -> list[str]:
        """Block until every camera has sent a frame and its camera_info. Returns the cameras still missing."""
        deadline = time.time() + timeout_s
        while True:
            missing = [c for c in self.camera_ids if self._latest[c] is None or self._K[c] is None]
            if not missing or time.time() > deadline:
                return missing
            time.sleep(0.2)
