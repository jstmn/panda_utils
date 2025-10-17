import argparse
import rospy
from sensor_msgs.msg import JointState, Image
import numpy as np

from panda_utils.pointcloud_utils import (
    crop_pointcloud_to_workspace,
    pointcloud_from_rgbd,
    visualize_pointcloud,
    align_depth_to_color,
)

# $ rostopic list
# ...
# /panda/joint_states
# /realsense_north/color/image_raw
# /realsense_north/depth/color/points
# /realsense_north/depth/image_rect_raw
# /realsense_north/joint_states
# /realsense_south/color/image_raw
# /realsense_south/depth/color/points
# /realsense_south/depth/image_rect_raw
# /realsense_south/joint_states
# ...


class DemonstrationLogger:
    def __init__(self, camera_namespace: str, rate: int = 20):

        assert camera_namespace in {"realsense_north", "realsense_south"}
        self._camera_namespace = camera_namespace
        self.rate = rospy.Rate(rate)

        # Create a subscribers
        self.depth_sub = rospy.Subscriber(f"/{self._camera_namespace}/depth/image_rect_raw", Image, self.depth_callback)
        self.color_sub = rospy.Subscriber(f"/{self._camera_namespace}/color/image_raw", Image, self.color_callback)
        self.joint_states_sub = rospy.Subscriber("/panda/joint_states", JointState, self.joint_states_callback)
        # rospy.on_shutdown(self.save_data)

        self._depth_img_np: np.ndarray | None = None
        self._color_img_np: np.ndarray | None = None

    def depth_callback(self, msg: Image):
        self._depth_img_np = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def color_callback(self, msg: Image):
        self._color_img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def joint_states_callback(self, msg: JointState):
        q = np.array([msg.position for msg in self._joint_states_msgs])
        dq = np.array([msg.velocity for msg in self._joint_states_msgs])

    def run(self):
        counter = 0
        empty_counter = 0
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self._depth_img_np is None or self._color_img_np is None:
                print(self._depth_img_np is None, self._color_img_np is None)
                continue

            depth_image = self._depth_img_np.copy()
            color_image = self._color_img_np.copy()
            depth_image[depth_image < 1] = 5000  # everything less than 1mm away is discarded.
            depth_aligned = align_depth_to_color(depth_image)
            assert depth_aligned.shape[0:2] == (
                720,
                1280,
            ), f"Depth image should have shape (720, 1280), got {depth_aligned.shape[0:2]}"
            pointcloud, pointcloud_np = pointcloud_from_rgbd(color_image, depth_aligned)
            assert pointcloud_np.shape[1] == 3, f"Pointcloud should have shape (N, 3), got {pointcloud_np.shape}"
            assert len(pointcloud_np) > 0, f"Pointcloud should have at least one point, got {len(pointcloud_np)}"
            pointcloud_cropped = crop_pointcloud_to_workspace(pointcloud)
            pcd_cropped_np = pointcloud_cropped.point.positions.numpy()
            assert pcd_cropped_np.shape[1] == 3, f"Pointcloud should have shape (N, 3), got {pcd_cropped_np.shape}"
            assert len(pcd_cropped_np) > 0, f"Pointcloud should have at least one point, got {len(pcd_cropped_np)}"
            assert (
                pcd_cropped_np.shape[0] < pointcloud_np.shape[0]
            ), f"Cropped pointcloud should have fewer points, got {pcd_cropped_np.shape[0]} and {pointcloud_np.shape[0]}"
            visualize_pointcloud(pointcloud_cropped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_namespace", type=str)
    args, _ = parser.parse_known_args()
    logger = DemonstrationLogger(args.camera_namespace)
    logger.run()


if __name__ == "__main__":
    rospy.init_node("realsense_subscriber", anonymous=False)
    main()
