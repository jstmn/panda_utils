from time import time

import numpy as np
import open3d as o3d
import cv2

from mpcm.constants import (
    DEPTH_MAX_M,
    CameraIntrinsics,
    CameraExtrinsics,
    WorkspaceBounds,
)


def align_rgb_to_depth(
    rgb_image: np.ndarray,
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Resample RGB image to match the depth camera FOV and resolution.

    Note, for getting a pointcloud from the realsense, you should probably do:
        depth_intrinsics = REALSENSE_DEPTH_INTRINSICS,
        color_intrinsics = REALSENSE_COLOR_INTRINSICS,

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
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Resample RGB image from depth camera resolution to color camera resolution.

    Note, for getting a pointcloud from the realsense, you should probably do:
        depth_intrinsics = REALSENSE_DEPTH_INTRINSICS,
        color_intrinsics = REALSENSE_COLOR_INTRINSICS,

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
    depth_image: np.ndarray[np.uint16],
    depth_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Resample depth image to match the color camera FOV and resolution.

    Recommended:
        depth_intrinsics = REALSENSE_DEPTH_INTRINSICS
        color_intrinsics = REALSENSE_COLOR_INTRINSICS

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
    Hd, Wd = depth_intrinsics.height, depth_intrinsics.width
    if Hc == Hd and Wc == Wd:
        return depth_image

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


def num_points(pointcloud: o3d.t.geometry.PointCloud) -> int:
    """
    Get the size of a pointcloud.
    """
    return len(pointcloud.point.positions)


def run_random_down_sample_known_output_size(
    pointcloud: o3d.t.geometry.PointCloud, output_n_points: int
) -> o3d.t.geometry.PointCloud:
    """
    Randomly downsample a pointcloud to the given number of points.
    """
    assert (
        num_points(pointcloud) >= output_n_points
    ), f"Pointcloud should have at least {output_n_points} points, got {num_points(pointcloud)}"
    rand_indices = np.random.permutation(num_points(pointcloud))[:output_n_points].astype(np.int64)
    return pointcloud.select_by_index(
        o3d.core.Tensor(rand_indices, dtype=o3d.core.Dtype.Int64, device=pointcloud.point.positions.device),
        invert=False,
    )


def pointcloud_from_rgbd(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: CameraIntrinsics,
    camera_extrinsics: CameraExtrinsics,
    crop_to: WorkspaceBounds | None = None,
    depth_max_m: float = DEPTH_MAX_M,
    do_farthest_point_down_sample: bool = False,
    output_n_points: int = 1024,
    # stride: int = 5,
    verbose: bool = False,
    print_prefix: str = "  ",
) -> tuple[o3d.t.geometry.PointCloud, np.ndarray]:
    """
    Create a pointcloud from an RGB-D image.

    Recommended:
        camera_intrinsics = REALSENSE_COLOR_INTRINSICS
        camera_extrinsics = CAMERA_EXTRINSICS

    Args:
        rgb_image (np.ndarray): [H, W, 3] RGB image of type uint8
        depth_image (np.ndarray): [H, W] Depth image of type uint16, in mm
        camera_intrinsics (CameraIntrinsics): Camera intrinsics
        camera_extrinsics (CameraExtrinsics): Camera extrinsics

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
    assert (
        rgb_image.shape[:2] == depth_image.shape
    ), f"RGB and depth images should have same H, W dimensions, got {rgb_image.shape=} and {depth_image.shape=}"
    t_start = time()

    # Create RGBD image
    rgb_o3d = o3d.t.geometry.Image(rgb_image)
    depth_o3d = o3d.t.geometry.Image(depth_image)
    rgbd_image = o3d.t.geometry.RGBDImage(rgb_o3d, depth_o3d)
    rgbd_image = rgbd_image.to(o3d.core.Device("CUDA:0"))

    # Create pointcloud from RGBD image
    t0 = time()
    pointcloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics=camera_intrinsics.intrinsic_matrix_open3d,
        depth_scale=1000.0,
        depth_max=depth_max_m,
        stride=1,  # This is the number of pixels to skip between points
        with_normals=False,
    )
    pointcloud = pointcloud.to(o3d.core.Device("CUDA:0"))
    # total time:    0.10224 s on cuda:0 vs 0.14682 s on cpu

    if verbose:
        print(
            f"{print_prefix}create_from_rgbd_image()\t took {time() - t0:.5f} seconds \t new pcd size: {num_points(pointcloud)}"
        )
    pointcloud = pointcloud.transform(camera_extrinsics.matrix_open3d)

    if crop_to:
        pointcloud = pointcloud.crop(crop_to.open3d_aabb().to(pointcloud.point.positions.device))

    if do_farthest_point_down_sample:
        t0 = time()
        sampling_ratio = (1.1 * output_n_points) / num_points(pointcloud)
        pointcloud = pointcloud.random_down_sample(sampling_ratio)
        if verbose:
            print(
                f"{print_prefix}random_down_sample()\t took {time() - t0:.5f} seconds \t new pcd size: {num_points(pointcloud)}"
            )

        assert (
            num_points(pointcloud) > output_n_points
        ), f"Pointcloud should have at least {output_n_points} points, got {num_points(pointcloud)}"
        t0 = time()
        pointcloud = pointcloud.farthest_point_down_sample(num_samples=output_n_points)
        if verbose:
            print(
                f"{print_prefix}farthest_point_down_sample()\t took {time() - t0:.5f} seconds \t new pcd size: {num_points(pointcloud)}"
            )
    else:
        t0 = time()
        pointcloud = run_random_down_sample_known_output_size(pointcloud, output_n_points)
        if verbose:
            print(
                f"{print_prefix}random_down_sample()\t took {time() - t0:.5f} seconds \t new pcd size: {num_points(pointcloud)}"
            )

    # if do_remove_outliers:
    #     # assert False, "This feature works, but is really slow. You probably don't want to use it."
    #     t0 = time()
    #     pointcloud, _ = pointcloud.remove_statistical_outliers(
    #         nb_neighbors=15, std_ratio=1.75
    #     )  # remove points that are outliers based on statistical analysis
    #     if verbose:
    #         print(f"{print_prefix}remove_statistical_outliers()\t took {time() - t0:.5f} seconds \t new pcd size: {num_points(pointcloud)}")

    assert (
        num_points(pointcloud) == output_n_points
    ), f"Pointcloud should have {output_n_points} points, got {num_points(pointcloud)}"

    if verbose:
        print(f"{print_prefix}total time: \t {time() - t_start:.5f} s")

    return pointcloud, pointcloud.point.positions.cpu().numpy()


def crop_pointcloud_to_workspace(
    pointcloud: o3d.t.geometry.PointCloud,
    workspace_bounds: WorkspaceBounds,
    z_padding_m: float = 0.03,
) -> o3d.t.geometry.PointCloud:
    """
    Crop a pointcloud to the workspace bounds.

    Consider using 'workspace_bounds = WORKSPACE_BOUNDS'
    """
    return pointcloud.crop(workspace_bounds.open3d_aabb(z_padding_m=z_padding_m).to(pointcloud.point.positions.device))


def visualize_pointcloud(
    pointcloud: o3d.t.geometry.PointCloud,
    name: str = "",
) -> None:
    """
    Visualize a pointcloud.
    """
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    coord_axis_frames = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=origin)
        for origin in [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ]
    o3d.visualization.draw_geometries(
        [pointcloud.to_legacy(), origin_frame] + coord_axis_frames, window_name=f"Pointcloud Visualization: {name}"
    )
