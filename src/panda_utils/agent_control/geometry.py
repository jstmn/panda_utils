"""Pose math for applying model-chosen deltas, and drawing the end effector frame onto camera images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from panda_utils.types import CameraExtrinsics

CONTROL_FRAMES = ("world", "ee")

# Extrinsics `parent_frame` values. r2st_handeye.py writes "robot_base" (static camera, matrix = T_base_cam). A
# wrist camera's calibration is T_ee_cam, marked with "end_effector".
STATIC_PARENT_FRAME = "robot_base"
WRIST_PARENT_FRAME = "end_effector"

# RGB colours for the x, y, z axes - the usual red/green/blue convention, also stated in the prompt.
AXIS_COLORS_RGB = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


def clamp_delta(
    translation_m: np.ndarray, rotvec_rad: np.ndarray, max_translation_m: float, max_rotation_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a step down (direction preserved) so neither its translation nor its rotation exceeds the limit."""
    translation_m = np.asarray(translation_m, dtype=float)
    rotvec_rad = np.asarray(rotvec_rad, dtype=float)
    t_norm = np.linalg.norm(translation_m)
    r_norm = np.linalg.norm(rotvec_rad)
    if t_norm > max_translation_m:
        translation_m = translation_m * (max_translation_m / t_norm)
    if r_norm > max_rotation_rad:
        rotvec_rad = rotvec_rad * (max_rotation_rad / r_norm)
    return translation_m, rotvec_rad


def delta_to_target(T_base_ee: np.ndarray, translation_m: np.ndarray, rotvec_rad: np.ndarray, frame: str) -> np.ndarray:
    """Target end effector pose [4, 4] in the base frame after applying a delta.

    "world": translate along base axes, rotate about base axes through the end effector origin (R_delta @ R).
    "ee": translate along the current end effector axes, rotate about them (R @ R_delta).
    """
    assert frame in CONTROL_FRAMES, f"frame must be one of {CONTROL_FRAMES}, got {frame!r}"
    R = T_base_ee[:3, :3]
    R_delta = Rotation.from_rotvec(rotvec_rad).as_matrix()
    T_target = np.eye(4)
    if frame == "world":
        T_target[:3, 3] = T_base_ee[:3, 3] + translation_m
        T_target[:3, :3] = R_delta @ R
    else:
        T_target[:3, 3] = T_base_ee[:3, 3] + R @ translation_m
        T_target[:3, :3] = R @ R_delta
    return T_target


def pose_error(T_target: np.ndarray, T_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position error [3] m, rotation error as a base-frame rotation vector [3] rad) from current to target."""
    return T_target[:3, 3] - T_current[:3, 3], Rotation.from_matrix(T_target[:3, :3] @ T_current[:3, :3].T).as_rotvec()


def pose_to_text(T_base_ee: np.ndarray) -> str:
    """Human readable pose for the prompt: position in cm, orientation as static-xyz euler angles in degrees."""
    xyz_cm = T_base_ee[:3, 3] * 100.0
    rpy_deg = Rotation.from_matrix(T_base_ee[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"position (cm) x={xyz_cm[0]:.1f} y={xyz_cm[1]:.1f} z={xyz_cm[2]:.1f}; "
        f"orientation (deg, roll/pitch/yaw about world x/y/z) {rpy_deg[0]:.1f} / {rpy_deg[1]:.1f} / {rpy_deg[2]:.1f}"
    )


def load_extrinsics(path: str | Path) -> CameraExtrinsics:
    """Read an extrinsics YAML in the r2st_handeye.py schema (a 4x4 `matrix`, `parent_frame`, `child_frame`)."""
    data = yaml.safe_load(Path(path).expanduser().read_text())
    matrix = np.asarray(data["matrix"], dtype=float)
    assert matrix.shape == (4, 4), f"{path}: expected a 4x4 `matrix`, got shape {matrix.shape}"
    parent = data.get("parent_frame", STATIC_PARENT_FRAME)
    assert parent in (STATIC_PARENT_FRAME, WRIST_PARENT_FRAME), (
        f"{path}: parent_frame must be '{STATIC_PARENT_FRAME}' (static camera) or '{WRIST_PARENT_FRAME}' "
        f"(wrist camera), got {parent!r}"
    )
    return CameraExtrinsics(matrix=matrix, parent_frame=parent, child_frame=data.get("child_frame", ""))


def T_cam_ee(extrinsics: CameraExtrinsics, T_base_ee: np.ndarray) -> np.ndarray:
    """End effector pose in the camera's optical frame."""
    if extrinsics.parent_frame == WRIST_PARENT_FRAME:
        return np.linalg.inv(extrinsics.matrix)
    return np.linalg.inv(extrinsics.matrix) @ T_base_ee


def draw_frame(image_rgb: np.ndarray, T_cam_obj: np.ndarray, K: np.ndarray, axis_length_m: float = 0.08) -> np.ndarray:
    """Return a copy of `image_rgb` with the frame `T_cam_obj` drawn as x/y/z arrows (red/green/blue).

    Assumes a rectified image (no distortion), as published by the ZED `image_rect_color` topics. Nothing is drawn
    if the frame's origin is behind the camera.
    """
    out = image_rgb.copy()
    if T_cam_obj[2, 3] <= 1e-3:
        return out
    points_obj = np.vstack([np.zeros(3), np.eye(3) * axis_length_m])
    rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
    pixels, _ = cv2.projectPoints(points_obj, rvec, T_cam_obj[:3, 3], K, np.zeros(5))
    pixels = np.round(pixels.reshape(-1, 2)).astype(int)
    origin = tuple(int(v) for v in pixels[0])
    thickness = max(2, round(out.shape[0] / 200))
    # Draw z first so x and y stay visible when the gripper points at the camera.
    for axis in (2, 1, 0):
        tip = tuple(int(v) for v in pixels[axis + 1])
        cv2.arrowedLine(out, origin, tip, AXIS_COLORS_RGB[axis], thickness, cv2.LINE_AA, tipLength=0.2)
    cv2.circle(out, origin, thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return out
