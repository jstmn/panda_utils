"""ROS 2 subscriber for Stereolabs ZED camera RGB/depth streams.

Topic names follow the ZED ROS 2 Wrapper v5.1.0+ convention:
  /zed_<id>/zed_node/rgb/color/rect/image
  /zed_<id>/zed_node/depth/depth_registered
See: https://docs.stereolabs.com/docs/integrations/ros-2/zed-stereo-node

RGB/depth use sensor_msgs/Image (not zed_msgs). zed_msgs covers custom
interfaces only: https://docs.stereolabs.com/docs/integrations/ros-2/zed-msgs
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


# Matches ZED ROS 2 Wrapper defaults (RELIABLE / KEEP_LAST / VOLATILE).
_ZED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class RGBSubscriberRos2Zed:
    """Collects ZED color/depth images over ROS 2 (drop-in style API vs RGBSubscriber)."""

    def __init__(self, camera_ids: list[str], node_name: str = "rgb_subscriber_ros2_zed"):
        for cam_id in camera_ids:
            assert "_" not in cam_id, f"Camera id cannot contain underscore: {cam_id}"

        self._camera_ids = camera_ids
        self._camera_data: dict[str, dict] = {}
        for cam_id in camera_ids:
            self._camera_data[cam_id] = {
                "latest_color": None,
                "latest_depth": None,
                "depth_camera_info": None,
                "color_camera_info": None,
            }

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = Node(node_name)
        self._subscribers = []

        for cam_id in camera_ids:
            ns = f"/zed_{cam_id}/zed_node"
            self._subscribers.append(
                self._node.create_subscription(
                    Image,
                    f"{ns}/depth/depth_registered",
                    lambda msg, s=cam_id: self._depth_callback(msg, s),
                    _ZED_QOS,
                )
            )
            self._subscribers.append(
                self._node.create_subscription(
                    Image,
                    f"{ns}/rgb/color/rect/image",
                    lambda msg, s=cam_id: self._color_callback(msg, s),
                    _ZED_QOS,
                )
            )
            self._subscribers.append(
                self._node.create_subscription(
                    CameraInfo,
                    f"{ns}/depth/camera_info",
                    lambda msg, s=cam_id: self._depth_info_callback(msg, s),
                    _ZED_QOS,
                )
            )
            self._subscribers.append(
                self._node.create_subscription(
                    CameraInfo,
                    f"{ns}/rgb/color/rect/camera_info",
                    lambda msg, s=cam_id: self._color_info_callback(msg, s),
                    _ZED_QOS,
                )
            )

        self._stop_spin = False
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        while rclpy.ok() and not self._stop_spin:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _depth_info_callback(self, msg: CameraInfo, cam_id: str) -> None:
        if self._camera_data[cam_id]["depth_camera_info"] is None:
            self._camera_data[cam_id]["depth_camera_info"] = msg

    def _color_info_callback(self, msg: CameraInfo, cam_id: str) -> None:
        if self._camera_data[cam_id]["color_camera_info"] is None:
            self._camera_data[cam_id]["color_camera_info"] = msg

    def _depth_callback(self, msg: Image, cam_id: str) -> None:
        # Default ZED depth is 32FC1 (meters). OpenNI mode publishes 16UC1 (mm).
        if msg.encoding in ("32FC1", "TYPE_32FC1"):
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            # Convert meters -> mm uint16 to match RealSense / pointcloud_utils pipeline.
            depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            self._camera_data[cam_id]["latest_depth"] = depth_mm
        elif msg.encoding in ("16UC1", "mono16"):
            self._camera_data[cam_id]["latest_depth"] = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width
            )
        else:
            self._node.get_logger().warn(
                f"Unsupported depth encoding '{msg.encoding}' on cam '{cam_id}'; skipping frame"
            )

    def _color_callback(self, msg: Image, cam_id: str) -> None:
        # ZED typically publishes bgra8; also handle rgb8/bgr8.
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("bgra8", "rgba8"):
            img = buf.reshape(msg.height, msg.width, 4)
            if msg.encoding == "bgra8":
                img = img[:, :, [2, 1, 0]]  # BGRA -> RGB (drop alpha)
            else:
                img = img[:, :, :3]  # RGBA -> RGB
        elif msg.encoding in ("bgr8", "rgb8"):
            img = buf.reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                img = img[:, :, ::-1].copy()  # BGR -> RGB
        else:
            img = buf.reshape(msg.height, msg.width, -1)
            self._node.get_logger().warn(
                f"Unexpected color encoding '{msg.encoding}' on cam '{cam_id}'; storing raw channels"
            )
        self._camera_data[cam_id]["latest_color"] = img

    def latest_rgb(self, cam_id: str) -> np.ndarray | None:
        return self._camera_data[cam_id]["latest_color"]

    def latest_depth(self, cam_id: str) -> np.ndarray | None:
        return self._camera_data[cam_id]["latest_depth"]

    def destroy(self) -> None:
        self._stop_spin = True
        self._spin_thread.join(timeout=1.0)
        self._node.destroy_node()
