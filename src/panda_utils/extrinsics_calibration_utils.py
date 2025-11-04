import h5py
import numpy as np
import open3d as o3d
from typing import List
import gymnasium as gym
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Panda
from panda_utils.types import CameraIntrinsics, CameraExtrinsics
from panda_utils.constants import REALSENSE_DEPTH_INTRINSICS, REALSENSE_COLOR_INTRINSICS
from panda_utils.pointcloud_utils import pointcloud_from_rgbd, align_depth_to_color


@register_env("EmptyPandaEnv-v1", max_episode_steps=50)
class EmptyPandaEnv(BaseEnv):

    agent: Panda

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="panda", **kwargs)

    @property
    def _default_sensor_configs(self):
        xy_offset = 0.6
        z_offset = 0.5
        target = [0, 0, 0.5]
        pose_center = sapien_utils.look_at(eye=[xy_offset, 0, z_offset], target=target)
        pose_left = sapien_utils.look_at(eye=[0.0, -xy_offset, z_offset], target=target)
        pose_right = sapien_utils.look_at(eye=[0.0, xy_offset, z_offset], target=target)
        cams = [
            CameraConfig(
                uid=uid,
                pose=pose,
                width=512,
                height=512,
                fov=np.deg2rad(80.0),
                near=0.01,
                far=100,
                shader_pack="default",
            ) for uid, pose in zip(["camera_center", "camera_left", "camera_right"], [pose_center, pose_left, pose_right])
        ]
        return cams

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.5, 0.5], target=[0, 0, 0.5])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))


def create_maniskill_env():
    """Create a ManiSkill environment with only the Panda robot and three cameras."""
    # Create environment - empty scene with just the robot
    env = gym.make(
        "EmptyPandaEnv-v1",
        num_envs=1,
        render_mode="rgbd",
        control_mode="pd_joint_pos",
        obs_mode="pointcloud",
    )
    env.reset()
    return env


def get_simulated_pointcloud(env: EmptyPandaEnv, joint_states: np.ndarray) -> o3d.t.geometry.PointCloud:
    """
    Set the robot to the given joint configuration and collect a simulated pointcloud. Note that 
    obs contains:        
        - 'pointcloud'
            - 'xyzw': [n_envs, npoints, 4]
            - 'rgb': [n_envs, npoints, 3]
            - 'segmentation': [n_envs, npoints, 1]
    
    Args:
        env: ManiSkill environment
        joint_states: [7] array of joint angles in radians
        
    Returns:
        Combined pointcloud from multiple camera viewpoints in robot base frame
    """
    assert joint_states.shape == (1, 9), f"Should be (1, 9) - (n_envs=1, ndof=9). 7 rotational joints, 2 gripper joints."
    # Set robot joint positions
    env.unwrapped.agent.robot.set_qpos(joint_states)
    obs = env.get_obs()
    
    # Extract pointcloud from observation
    # Format: 'xyzw': [n_envs, npoints, 4], 'rgb': [n_envs, npoints, 3]
    if "pointcloud" not in obs:
        raise ValueError("Observation does not contain 'pointcloud' key")
    
    pc_dict = obs["pointcloud"]
    if not isinstance(pc_dict, dict):
        raise ValueError(f"Expected pointcloud to be a dict, got {type(pc_dict)}")
    
    # Extract points and colors
    # Shape: [n_envs, npoints, 4] -> for single env, index [0] to get [npoints, 4]
    xyzw = pc_dict["xyzw"][0]  # [npoints, 4] - take first (and only) environment
    
    # Convert torch tensor to numpy array if needed (ManiSkill returns torch tensors)
    if isinstance(xyzw, torch.Tensor):
        xyzw = xyzw.cpu().numpy()
    
    points = xyzw[:, :3]  # Extract xyz coordinates
    colors = pc_dict["rgb"][0].cpu().numpy() / 255.0  # [npoints, 3]

    # Create Open3D pointcloud
    pc = o3d.t.geometry.PointCloud()
    pc.point.positions = o3d.core.Tensor(points, dtype=o3d.core.float32)
    pc.point.colors = o3d.core.Tensor(colors, dtype=o3d.core.float32)
    return pc


def load_hdf5_data(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load RGB, depth, and joint_states from an hdf5 file.
    
    Args:
        hdf5_path: Path to hdf5 file
        
    Returns:
        rgb_image: [H, W, 3] uint8
        depth_image: [H, W] uint16
        joint_states: [7] float64
    """
    with h5py.File(hdf5_path, "r") as f:
        rgb = f["rgb"][0]  # Use first frame
        depth = f["depth"][0]  # Use first frame
        joint_states = f["joint_states"][0]  # Use first frame
        
    return rgb, depth, joint_states


def register_pointclouds(
    source_pc: o3d.t.geometry.PointCloud,
    target_pc: o3d.t.geometry.PointCloud,
    initial_transform: np.ndarray = None,
) -> np.ndarray:
    """
    Perform ICP registration to align source pointcloud to target pointcloud.
    
    Args:
        source_pc: Source pointcloud (recorded from camera)
        target_pc: Target pointcloud (simulated from ManiSkill)
        initial_transform: [4, 4] initial transformation matrix (optional)
        
    Returns:
        transformation: [4, 4] transformation matrix from source to target frame
    """
    # Convert to legacy format for ICP
    source_legacy = source_pc.to_legacy()
    target_legacy = target_pc.to_legacy()
    
    # Downsample for faster registration
    source_down = source_legacy.voxel_down_sample(voxel_size=0.01)
    target_down = target_legacy.voxel_down_sample(voxel_size=0.01)
    
    # Estimate normals
    source_down.estimate_normals()
    target_down.estimate_normals()
    
    # Initial transformation
    if initial_transform is None:
        initial_transform = np.eye(4)
    
    # Perform ICP registration
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance=0.1,
        init=initial_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    
    return result.transformation


def estimate_extrinsics_from_hdf5(
    hdf5_path: str,
    env,
    camera_intrinsics: CameraIntrinsics = REALSENSE_COLOR_INTRINSICS,
) -> np.ndarray:
    """
    Estimate camera extrinsics from a single hdf5 file.
    
    Args:
        hdf5_path: Path to hdf5 file
        env: ManiSkill environment
        camera_intrinsics: Camera intrinsics
        
    Returns:
        transformation: [4, 4] transformation matrix from camera frame to robot base frame
    """
    # Load data
    rgb_image, depth_image, joint_states = load_hdf5_data(hdf5_path)
    
    # Align depth to RGB if dimensions don't match
    # (assuming RGB is color camera resolution and depth is depth camera resolution)
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        depth_image = align_depth_to_color(
            depth_image,
            depth_intrinsics=REALSENSE_DEPTH_INTRINSICS,
            color_intrinsics=camera_intrinsics,
        )
    
    # Create pointcloud from RGB-D (without extrinsics transformation)
    # We'll create it in camera frame, then transform it
    # Use identity transform to get pointcloud in camera frame
    identity_extrinsics = CameraExtrinsics(
        matrix=np.eye(4),
        parent_frame="camera_depth_optical_frame",
        child_frame="camera_depth_optical_frame",
    )
    pc_camera, _ = pointcloud_from_rgbd(
        rgb_image,
        depth_image,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=identity_extrinsics,
    )
    
    # Get simulated pointcloud in robot base frame
    pc_robot = get_simulated_pointcloud(env, joint_states)
    
    # Perform ICP registration
    # The transformation will map from camera frame to robot base frame
    transform = register_pointclouds(pc_camera, pc_robot)
    
    return transform


def average_transformations(transforms: List[np.ndarray]) -> np.ndarray:
    """
    Average multiple 4x4 transformation matrices.
    
    For rotation, we average in SO(3) using the matrix logarithm/exponential.
    For translation, we simply average.
    
    Args:
        transforms: List of [4, 4] transformation matrices
        
    Returns:
        averaged_transform: [4, 4] averaged transformation matrix
    """
    if len(transforms) == 0:
        raise ValueError("No transforms to average")
    
    if len(transforms) == 1:
        return transforms[0]
    
    # Extract rotations and translations
    rotations = [t[:3, :3] for t in transforms]
    translations = [t[:3, 3] for t in transforms]
    
    # Average translation
    avg_translation = np.mean(translations, axis=0)
    
    # Average rotation matrices directly and re-orthonormalize
    # This is a simplified approach - for better results, use quaternion averaging
    # or proper SO(3) averaging (e.g., using scipy.spatial.transform.Rotation)
    avg_rotation = np.mean(rotations, axis=0)
    
    # Re-orthonormalize to ensure valid rotation matrix
    U, _, Vt = np.linalg.svd(avg_rotation)
    avg_rotation = U @ Vt
    # Ensure proper rotation (det = 1)
    if np.linalg.det(avg_rotation) < 0:
        U[:, -1] *= -1
        avg_rotation = U @ Vt
    
    # Construct averaged transformation
    avg_transform = np.eye(4)
    avg_transform[:3, :3] = avg_rotation
    avg_transform[:3, 3] = avg_translation
    
    return avg_transform



def visualize_pointcloud(
    pointcloud: o3d.t.geometry.PointCloud,
    name: str = "",
) -> None:
    """
    Visualize a pointcloud.
    """
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Pointcloud Visualization: {name}")
    vis.add_geometry(pointcloud.to_legacy())
    vis.add_geometry(origin_frame)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])  # Black background
    vis.run()
    vis.destroy_window()
