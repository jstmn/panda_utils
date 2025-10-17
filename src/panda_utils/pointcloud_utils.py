from time import time

import numpy as np
import open3d as o3d
import cv2

from panda_utils.types import (
    CameraIntrinsics,
    CameraExtrinsics,
    WorkspaceBounds,
)
from panda_utils.constants import REALSENSE_DEPTH_INTRINSICS, REALSENSE_COLOR_INTRINSICS, DEPTH_MAX_M, WORKSPACE_BOUNDS


def align_rgb_to_depth(
    rgb_image: np.ndarray,
    depth_intrinsics: CameraIntrinsics = REALSENSE_DEPTH_INTRINSICS,
    color_intrinsics: CameraIntrinsics = REALSENSE_COLOR_INTRINSICS,
) -> np.ndarray:
    """
    Resample RGB image to match the depth camera FOV and resolution.

    Parameters
    ----------
    rgb_image : np.ndarray (H, W, 3)
        RGB image
    depth_intrinsics : CameraIntrinsics
        Intrinsics of the depth camera
    color_intrinsics : CameraIntrinsics
        Intrinsics of the color camera

    Returns
    -------
    aligned_rgb : np.ndarray (Hd, Wd, 3)
        RGB image resampled to depth camera FOV
    """
    Hd, Wd = depth_intrinsics.height, depth_intrinsics.width
    fx_d, fy_d, cx_d, cy_d = depth_intrinsics.values
    fx_c, fy_c, cx_c, cy_c = color_intrinsics.values

    # Step 1: create depth pixel grid
    u_d, v_d = np.meshgrid(np.arange(Wd), np.arange(Hd))

    # Step 2: compute corresponding color coordinates in color image
    u_c = ((u_d - cx_d) * (fx_c / fx_d)) + cx_c
    v_c = ((v_d - cy_d) * (fy_c / fy_d)) + cy_c

    # Step 3: sample RGB values with interpolation for each channel
    aligned_rgb = np.zeros((Hd, Wd, 3), dtype=rgb_image.dtype)
    for channel in range(3):
        aligned_rgb[:, :, channel] = cv2.remap(
            rgb_image[:, :, channel],
            u_c.astype(np.float32),
            v_c.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return aligned_rgb


def align_rgb_to_color(
    rgb_image: np.ndarray,
    depth_intrinsics: CameraIntrinsics = REALSENSE_DEPTH_INTRINSICS,
    color_intrinsics: CameraIntrinsics = REALSENSE_COLOR_INTRINSICS,
) -> np.ndarray:
    """
    Resample RGB image from depth camera resolution to color camera resolution.

    Parameters
    ----------
    rgb_image : np.ndarray (Hd, Wd, 3)
        RGB image at depth camera resolution
    depth_intrinsics : CameraIntrinsics
        Intrinsics of the depth camera
    color_intrinsics : CameraIntrinsics
        Intrinsics of the color camera

    Returns
    -------
    aligned_rgb : np.ndarray (Hc, Wc, 3)
        RGB image resampled to color camera FOV
    """
    Hc, Wc = color_intrinsics.height, color_intrinsics.width
    fx_d, fy_d, cx_d, cy_d = depth_intrinsics.values
    fx_c, fy_c, cx_c, cy_c = color_intrinsics.values

    # Step 1: create color pixel grid
    u_c, v_c = np.meshgrid(np.arange(Wc), np.arange(Hc))

    # Step 2: compute corresponding depth coordinates in depth image
    u_d = ((u_c - cx_c) * (fx_d / fx_c)) + cx_d
    v_d = ((v_c - cy_c) * (fy_d / fy_c)) + cy_d

    # Step 3: sample RGB values with interpolation for each channel
    aligned_rgb = np.zeros((Hc, Wc, 3), dtype=rgb_image.dtype)
    for channel in range(3):
        aligned_rgb[:, :, channel] = cv2.remap(
            rgb_image[:, :, channel],
            u_d.astype(np.float32),
            v_d.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return aligned_rgb


def align_depth_to_color(
    depth_image: np.ndarray,
    depth_intrinsics: CameraIntrinsics = REALSENSE_DEPTH_INTRINSICS,
    color_intrinsics: CameraIntrinsics = REALSENSE_COLOR_INTRINSICS,
) -> np.ndarray:
    """
    Resample depth image to match the color camera FOV and resolution.

    Parameters
    ----------
    depth_image : np.ndarray (H, W)
        Depth values in metric units (or scaled)
    depth_intr : CameraIntrinsics
        Intrinsics of the depth camera
    color_intr : CameraIntrinsics
        Intrinsics of the color camera

    Returns
    -------
    aligned_depth : np.ndarray (Hc, Wc)
        Depth values resampled to color camera FOV
    """
    Hc, Wc = color_intrinsics.height, color_intrinsics.width
    fx_d, fy_d, cx_d, cy_d = depth_intrinsics.values
    fx_c, fy_c, cx_c, cy_c = color_intrinsics.values
    # Step 1: create color pixel grid
    u_c, v_c = np.meshgrid(np.arange(Wc), np.arange(Hc))
    # Step 2: compute corresponding depth coordinates in depth image
    u_d = ((u_c - cx_c) * (fx_d / fx_c)) + cx_d
    v_d = ((v_c - cy_c) * (fy_d / fy_c)) + cy_d
    # Step 3: sample depth values with interpolation
    aligned_depth = cv2.remap(
        depth_image,
        u_d.astype(np.float32),
        v_d.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned_depth


def pointcloud_from_rgbd(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: CameraIntrinsics = REALSENSE_COLOR_INTRINSICS,
    camera_extrinsics: CameraExtrinsics = CAMERA_EXTRINSICS,
    depth_max_m: float = DEPTH_MAX_M,
    remove_outliers: bool = False,
) -> tuple[o3d.t.geometry.PointCloud, np.ndarray]:
    """
    Create a pointcloud from an RGB-D image.

    Args:
        rgb_image (np.ndarray): [H, W, 3] RGB image of type uint8
        depth_image (np.ndarray): [H, W] Depth image of type uint16, in mm
        camera_intrinsics (np.ndarray): [3, 3] Camera intrinsics
        camera_extrinsics (np.ndarray): [4, 4] Camera extrinsics, in panda_link0 frame

    Returns:
        np.ndarray: [N, 3] Pointcloud
    """
    # Check input shapes and types
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, f"RGB image should be [H, W, 3], got {rgb_image.shape}"
    assert rgb_image.dtype == np.uint8, f"RGB image should be uint8, got {rgb_image.dtype}"
    assert depth_image.ndim == 2, f"Depth image should be [H, W], got {depth_image.shape}"
    assert depth_image.dtype == np.uint16, f"Depth image should be uint16, got {depth_image.dtype}"
    assert isinstance(
        camera_intrinsics, CameraIntrinsics
    ), f"Camera intrinsics should be CameraIntrinsics, got {type(camera_intrinsics)}"
    assert isinstance(
        camera_extrinsics, CameraExtrinsics
    ), f"Camera extrinsics should be CameraExtrinsics, got {type(camera_extrinsics)}"
    assert rgb_image.shape[:2] == depth_image.shape, "RGB and depth images should have same H, W dimensions"

    # Create RGBD image
    rgb_o3d = o3d.t.geometry.Image(rgb_image)
    depth_o3d = o3d.t.geometry.Image(depth_image)
    rgbd_image = o3d.t.geometry.RGBDImage(rgb_o3d, depth_o3d)

    # Create pointcloud from RGBD image
    pointcloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics=camera_intrinsics.intrinsic_matrix_open3d,
        depth_scale=1000.0,
        depth_max=depth_max_m,
        stride=1,
        with_normals=False,
    )
    pointcloud = pointcloud.transform(camera_extrinsics.matrix_open3d)
    if remove_outliers:
        assert False, "This feature works, but is really slow. You probably don't want to use it."
        t0 = time()
        pointcloud, _ = pointcloud.remove_statistical_outliers(
            nb_neighbors=15, std_ratio=1.75
        )  # remove points that are outliers based on statistical analysis
        print(f"{time() - t0:.5f} seconds to remove outliers")
    return pointcloud, pointcloud.point.positions.numpy()


def crop_pointcloud_to_workspace(
    pointcloud: o3d.t.geometry.PointCloud,
    workspace_bounds: WorkspaceBounds = WORKSPACE_BOUNDS,
    z_padding_m: float = 0.03,
) -> o3d.t.geometry.PointCloud:
    """
    Crop a pointcloud to the workspace bounds.
    """
    return pointcloud.crop(workspace_bounds.open3d_aabb(z_padding_m=z_padding_m))


def visualize_pointcloud(
    pointcloud: o3d.t.geometry.PointCloud,
    name: str = "",
) -> None:
    """
    Visualize a pointcloud.
    """
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [pointcloud.to_legacy(), origin_frame], window_name=f"Pointcloud Visualization: {name}"
    )
