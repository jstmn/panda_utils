import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np

class RGBSubscriber:
    """Collects camera images and joint states during demonstration."""

    def __init__(self, camera_ids: list):
        for cam_id in camera_ids:
            assert "_" not in cam_id, f"Camera id cannot contain underscore: {cam_id}"

        self._camera_ids = camera_ids

        # Storage
        self._camera_data = {}
        for cam_id in camera_ids:
            self._camera_data[cam_id] = {
                "latest_color": None,
                "latest_depth": None,
                "depth_camera_info": None,
                "color_camera_info": None,
            }

        # ROS subscribers
        self._subscribers = []

        for cam_id in camera_ids:
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/depth/image_rect_raw",
                    Image,
                    lambda msg, s=cam_id: self._depth_callback(msg, s),
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/color/image_raw", Image, lambda msg, s=cam_id: self._color_callback(msg, s)
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/depth/camera_info",
                    CameraInfo,
                    lambda msg, s=cam_id: self._depth_info_callback(msg, s),
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/color/camera_info",
                    CameraInfo,
                    lambda msg, s=cam_id: self._color_info_callback(msg, s),
                )
            )

    def _depth_info_callback(self, msg, cam_id):
        if self._camera_data[cam_id]["depth_camera_info"] is None:
            self._camera_data[cam_id]["depth_camera_info"] = msg

    def _color_info_callback(self, msg, cam_id):
        if self._camera_data[cam_id]["color_camera_info"] is None:
            self._camera_data[cam_id]["color_camera_info"] = msg

    def _depth_callback(self, msg, cam_id):
        self._camera_data[cam_id]["latest_depth"] = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def _color_callback(self, msg, cam_id):
        self._camera_data[cam_id]["latest_color"] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)


    def latest_rgb(self, cam_id: str) -> np.ndarray:
        return self._camera_data[cam_id]["latest_color"]
