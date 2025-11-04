import argparse
import h5py
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.utils.visualization import ViserRenderer
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose
from mani_skill2_real2sim.utils.common import convert_observation_to_space
from sapien import Pose

from panda_utils.types import CameraIntrinsics, CameraExtrinsics
from panda_utils.constants import REALSENSE_DEPTH_INTRINSICS, REALSENSE_COLOR_INTRINSICS
from panda_utils.pointcloud_utils import pointcloud_from_rgbd, align_depth_to_color


""" calibrate_extrinsics.py

This script will estimate the transformation between the base link of the robot and the camera by aligning a 
pointcloud collected from the camera with a pointcloud collected from ManiSkill using the robot model.

The logic is as follows:
1. A empty ManiSkill environment is created, with only the robot model. There are no other objects in the scene. 
    The scene contains three cameras from different viewpoints to ensure good coverage of the robot.
2. For each hdf5 file, the following steps are performed:
    a. The robots configuration is extracted
    b. The maniskill robot is set to this joint angle and a simulated pointcloud is collected
    c. The pointcloud is created from the rgb and depth images
    d. Mesh registration is performed to align the recorded pointcloud with the virtual pointcloud
    e. The transformation is returned
3. The transformations are averaged to get the final transformation.

The expected format for hdf5 files is as follows. Note that only the first index of N is used.
/rgb           [N, H, W, 3] uint8
/depth         [N, H, W] uint16
/joint_states  [N, 7] float64

# Example usage:
python scripts/calibrate_extrinsics.py \
    --hdf5_paths data/extrinsics_nov3/11-03_12_02_50__south-1/data.hdf5 \
        data/extrinsics_nov3/11-03_12_04_27__south-2/data.hdf5 \
        data/extrinsics_nov3/11-03_12_10_13__south/data.hdf5
"""





def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera extrinsics by aligning recorded pointclouds with simulated ones"
    )
    parser.add_argument(
        "--hdf5_paths",
        nargs="+",
        required=True,
        help="Paths to hdf5 files containing RGB, depth, and joint_states",
    )
    args = parser.parse_args()
    
    # Validate hdf5 files exist
    for hdf5_path in args.hdf5_paths:
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    print(f"Calibrating extrinsics using {len(args.hdf5_paths)} hdf5 files...")
    
    # Create ManiSkill environment
    print("Creating ManiSkill environment...")
    env = create_maniskill_env()
    
    # Estimate extrinsics from each hdf5 file
    transforms = []
    for i, hdf5_path in enumerate(args.hdf5_paths):
        print(f"Processing {hdf5_path} ({i+1}/{len(args.hdf5_paths)})...")
        try:
            transform = estimate_extrinsics_from_hdf5(hdf5_path, env)
            transforms.append(transform)
            print(f"  Estimated transformation:\n{transform}")
        except Exception as e:
            print(f"  Error processing {hdf5_path}: {e}")
            continue
    
    if len(transforms) == 0:
        raise RuntimeError("Failed to estimate any transformations")
    
    # Average transformations
    print("\nAveraging transformations...")
    avg_transform = average_transformations(transforms)
    
    print(f"\nFinal averaged transformation:\n{avg_transform}")
    
    # Create CameraExtrinsics object
    extrinsics = CameraExtrinsics(
        matrix=avg_transform,
        parent_frame="panda_link0",
        child_frame="camera_depth_optical_frame",
    )
    
    # Output result
    print("\nExtrinsics matrix (use this in your code):")
    print(f"matrix = np.array({repr(avg_transform)})")
    env.close()


if __name__ == "__main__":
    main()


