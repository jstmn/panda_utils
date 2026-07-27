""" Example usage:

# Setup environment
source /opt/ros/humble/setup.bash
cd /home/res1/Desktop/ros2_ws
source install/setup.bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate panda_utils

cd /home/res1/Desktop/panda_utils
python scripts/zed_subscriber.py

# Todo:
1. remove rospy - this is a ros1 thing but we are using ros2. use whatever the ros2 equivalent is.
2. debug the subscriber - we should get images and depth from each of the three cameras.
3.

Run command:
Terminal 1:
bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh

if the above command is not working, try manually launching the clean.sh script
bash ~/Desktop/ros2_ws/zed_tools/clean.sh

Terminal 2:
python3 ~/Desktop/panda_utils/scripts/zed_subscriber.py --topic-prefixes zed=/zed_multi/zed zed2i=/zed_multi/zed2i zedm=/zed_multi/zedm
"""

# if __name__ == "__main__":
#     rgb_subscriber = RGBSubscriberRos2Zed(camera_ids=["north", "south"])
#     while not rospy.is_shutdown():
#         rgb = rgb_subscriber.latest_rgb()
#         depth = rgb_subscriber.latest_depth()
#         print(rgb.shape, depth.shape)
#         rospy.sleep(0.1)

import argparse
import time

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class ZedCameraStream:
    """Holds subscriptions and latest frames for a single ZED camera."""

    def __init__(self, node: Node, cam_id: str, topic_prefix: str, bridge: CvBridge,
                 rgb_window_pos: tuple[int, int], depth_window_pos: tuple[int, int]):
        self.cam_id = cam_id
        self.topic_prefix = "/" + topic_prefix.strip("/")
        self.bridge = bridge
        self.latest_rgb = None
        self.latest_depth = None
        self.rgb_frames = 0
        self.depth_frames = 0

        self.rgb_window_name = f"{cam_id} RGB"
        self.depth_window_name = f"{cam_id} Depth (0-5 m)"

        cv2.namedWindow(self.rgb_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.rgb_window_name, *rgb_window_pos)
        cv2.namedWindow(self.depth_window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.depth_window_name, *depth_window_pos)

        self._subscriptions = [
            node.create_subscription(
                Image,
                f"{self.topic_prefix}/rgb/image_rect_color",
                self._rgb_callback,
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                Image,
                f"{self.topic_prefix}/depth/depth_registered",
                self._depth_callback,
                qos_profile_sensor_data,
            ),
        ]

        node.get_logger().info(
            f"[{cam_id}] Subscribing to:\n"
            f"  {self.topic_prefix}/rgb/image_rect_color\n"
            f"  {self.topic_prefix}/depth/depth_registered"
        )

    def _rgb_callback(self, msg: Image) -> None:
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.rgb_frames += 1

    def _depth_callback(self, msg: Image) -> None:
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_frames += 1

    def show_images(self) -> None:
        if self.latest_rgb is not None:
            cv2.imshow(self.rgb_window_name, self.latest_rgb)

        if self.latest_depth is not None:
            depth_is_mm = self.latest_depth.dtype == np.uint16
            depth = np.asarray(self.latest_depth, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0)
            depth_display = np.zeros(depth.shape, dtype=np.uint8)
            if np.any(valid):
                max_display_depth = 5000.0 if depth_is_mm else 5.0
                clipped = np.clip(depth, 0, max_display_depth)
                depth_display = (clipped * (255.0 / max_display_depth)).astype(np.uint8)
                depth_display[~valid] = 0
            depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_TURBO)
            cv2.imshow(self.depth_window_name, depth_color)


class ZedSubscriber(Node):
    def __init__(self, camera_prefixes: dict[str, str]):
        super().__init__("zed_subscriber")
        bridge = CvBridge()
        window_w, window_h = 500, 300
        self.streams = {}
        for i, (cam_id, prefix) in enumerate(camera_prefixes.items()):
            rgb_pos = (0, i * window_h)
            depth_pos = (window_w, i * window_h)
            self.streams[cam_id] = ZedCameraStream(
                self, cam_id, prefix, bridge, rgb_pos, depth_pos
            )

    def show_images(self) -> None:
        for stream in self.streams.values():
            stream.show_images()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic-prefixes",
        nargs="+",
        default=[
            "zed=/zed_multi/zed",
            "zed2i=/zed_multi/zed2i",
            "zedm=/zed_multi/zedm",
        ],
        help=(
            "camera_id=topic_prefix pairs. Default matches the "
            "zed_multi_camera launch: zed=/zed_multi/zed etc."
        ),
    )
    args = parser.parse_args()

    camera_prefixes = {}
    for pair in args.topic_prefixes:
        cam_id, _, prefix = pair.partition("=")
        camera_prefixes[cam_id] = prefix

    rclpy.init(args=None)
    node = ZedSubscriber(camera_prefixes=camera_prefixes)
    last_log_time = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.show_images()

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            now = time.monotonic()
            if now - last_log_time >= 2.0:
                for cam_id, stream in node.streams.items():
                    rgb_shape = stream.latest_rgb.shape if stream.latest_rgb is not None else "waiting"
                    depth_shape = stream.latest_depth.shape if stream.latest_depth is not None else "waiting"
                    node.get_logger().info(
                        f"[{cam_id}] rgb={rgb_shape} ({stream.rgb_frames} frames), "
                        f"depth={depth_shape} ({stream.depth_frames} frames)"
                    )
                last_log_time = now

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()