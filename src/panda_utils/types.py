from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class PandaLimits:
    # Joint space
    q_min: np.ndarray
    q_max: np.ndarray
    q_vel_max: np.ndarray
    q_acc_max: np.ndarray
    q_jerk_max: np.ndarray
    tau_max: np.ndarray
    tau_dot_max: np.ndarray

    # End effector space
    translation_vel_max: np.ndarray
    rotation_vel_max: np.ndarray
    translation_acc_max: np.ndarray
    rotation_acc_max: np.ndarray
    translation_jerk_max: np.ndarray
    rotation_jerk_max: np.ndarray

    def __post_init__(self):
        for x in [
            self.q_min,
            self.q_max,
            self.q_vel_max,
            self.q_acc_max,
            self.q_jerk_max,
            self.tau_max,
            self.tau_dot_max,
            self.translation_vel_max,
            self.rotation_vel_max,
            self.translation_acc_max,
            self.rotation_acc_max,
            self.translation_jerk_max,
            self.rotation_jerk_max,
        ]:
            assert x.shape == (7,), f"Expected shape (7,), got {x.shape}"


@dataclass
class CameraExtrinsics:
    matrix: np.ndarray
    parent_frame: str
    child_frame: str

    @property
    def matrix_open3d(self) -> o3d.core.Tensor:
        return o3d.core.Tensor(self.matrix, dtype=o3d.core.float32)

    @property
    def matrix_open3d_inv(self) -> o3d.core.Tensor:
        return o3d.core.Tensor(np.linalg.inv(self.matrix), dtype=o3d.core.float32)


@dataclass
class CameraIntrinsics:
    """Intrinsics of Realsense D435i"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def values(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def intrinsic_matrix_np(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @property
    def intrinsic_matrix_open3d(self) -> o3d.core.Tensor:
        return o3d.core.Tensor(self.intrinsic_matrix_np, dtype=o3d.core.float32)


@dataclass
class WorkspaceBounds:
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    min_z_m: float
    max_z_m: float

    @property
    def extents(self) -> tuple[float, float, float]:
        """Return the extents (dimensions) of the workspace as (x, y, z)."""
        return (
            self.max_x_m - self.min_x_m,
            self.max_y_m - self.min_y_m,
            self.max_z_m - self.min_z_m,
        )

    @property
    def center(self) -> tuple[float, float, float]:
        """Return the center of the workspace as (x, y, z)."""
        return (
            (self.max_x_m + self.min_x_m) / 2,
            (self.max_y_m + self.min_y_m) / 2,
            (self.max_z_m + self.min_z_m) / 2,
        )

    def open3d_aabb(self, z_padding_m: float = 0.0) -> o3d.t.geometry.AxisAlignedBoundingBox:
        """Return an Open3D AxisAlignedBoundingBox for this workspace."""
        min_bound = o3d.core.Tensor([self.min_x_m, self.min_y_m, self.min_z_m - z_padding_m], dtype=o3d.core.float32)
        max_bound = o3d.core.Tensor([self.max_x_m, self.max_y_m, self.max_z_m + z_padding_m], dtype=o3d.core.float32)
        return o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
