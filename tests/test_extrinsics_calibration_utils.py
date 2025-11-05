import pytest
import numpy as np
import h5py

import open3d as o3d
import cv2
from panda_utils.utils import print_variable_recursively
from panda_utils.extrinsics_calibration_utils import (
    get_simulated_pointcloud,
    create_maniskill_env,
    visualize_pointcloud,
)
from panda_utils.pointcloud_utils import pointcloud_from_rgbd, align_depth_to_color, crop_pointcloud_to_workspace


"""
pytest tests/test_extrinsics_calibration_utils.py --capture=no
"""


def test_create_maniskill_env():
    env = create_maniskill_env()
    assert env is not None
    obs, _ = env.reset()
    env.close()


def test_get_simulated_pointcloud():
    env = create_maniskill_env()
    joint_states = np.zeros((1, 9))
    pc = get_simulated_pointcloud(env, joint_states)
    assert pc is not None
    assert isinstance(pc, o3d.t.geometry.PointCloud)
    env.close()


@pytest.fixture
def h5_data():
    """Load test data from h5 file

    h5ls tests/data/extrinsics_test_data.h5
        camera__north_color_image Dataset {7, 720, 1280, 3}
        camera__north_depth_image Dataset {7, 480, 640}
        joint_states_dq          Dataset {7, 9}
        joint_states_q           Dataset {7, 9}
    """
    with h5py.File("tests/data/extrinsics_test_data.h5", "r") as f:
        data = {
            "color_image": f["camera__north_color_image"][0],
            "depth_image": f["camera__north_depth_image"][0],
            "joint_states_q": f["joint_states_q"][0],
        }
    return data

def test_create_pointcloud_from_hdf5(h5_data):
    print_variable_recursively(h5_data)
    color_image = h5_data["color_image"]  # (720, 1280, 3)
    depth_image = h5_data["depth_image"]

    cv2.imshow("Color Image", color_image)
    cv2.imshow("Depth Image", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    depth_image[depth_image < 1] = 5000
    depth_aligned = align_depth_to_color(depth_image)
    pointcloud, pointcloud_np = pointcloud_from_rgbd(color_image, depth_aligned)
    # visualize_pointcloud(pointcloud, name="Pointcloud")
