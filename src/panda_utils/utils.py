from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from termcolor import cprint

if TYPE_CHECKING:
    from deoxys.franka_interface import FrankaInterface


def cm_to_m(cm: float) -> float:
    return cm / 100.0


def to_public_dict(obj):
    """Recursively convert an object to a dict, skipping private attributes."""
    # Base cases: primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_public_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_public_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_public_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if hasattr(obj, "__slots__"):
        return {
            slot: to_public_dict(getattr(obj, slot))
            for slot in obj.__slots__
            if hasattr(obj, slot) and not slot.startswith("_")
        }
    # Fallback to dir()
    result = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        if callable(value):
            continue
        result[attr] = to_public_dict(value)
    return result


def wait_for_deoxys_ready(robot_interface: FrankaInterface, rate_hz: float = 50.0, max_failures: int = 100) -> bool:
    """Block until deoxys is streaming arm and gripper state. ROS-agnostic, so usable from ROS 1 and ROS 2 scripts.

    Returns:
        bool: True once both are available, False after `max_failures` consecutive polls without them.
    """
    for n_failures in range(max_failures):
        time.sleep(1.0 / rate_hz)
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q
        if arm_q is not None and last_gripper_q is not None:
            return True
        if n_failures % 10 == 0:
            cprint(
                f"Arm q or gripper angle isn't available. Is deoxys running on the RT pc? last_gripper_q: "
                f"{last_gripper_q}, arm_q={arm_q}",
                "yellow",
            )
    cprint(f"Failed to get arm q or gripper angle for {max_failures} consecutive polls.", "red")
    return False


def save_combined_rgbs(color_images: list[np.ndarray], camera_ids: list[str], save_filepath: Path):
    """Save combined RGB images to a single image (tiled horizontally).

    Args:
        color_images: List of color images.
        camera_ids: List of camera ids.
        save_filepath: Path to save the combined RGB image.
    """
    assert len(color_images) == len(camera_ids), "Number of color images must be the same as the number of camera ids"
    height, width = color_images[0].shape[:2]
    combined_rgb = np.zeros((height, width * len(camera_ids), 3), dtype=np.uint8)

    for idx, (color_image, camera_id) in enumerate(zip(color_images, camera_ids)):
        assert color_image.shape == (
            height,
            width,
            3,
        ), f"Color images must have the same shape, got {color_image.shape} for camera {camera_id}"
        combined_rgb[:, idx * width : (idx + 1) * width, :] = color_image

    # Convert RGB to BGR for OpenCV
    combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

    # Add camera labels to each section
    for idx, camera_id in enumerate(camera_ids):
        text = f"Camera: {camera_id}"
        x_offset = idx * width + 10
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            combined_bgr,
            (x_offset - 5, y_offset - text_height - 5),
            (x_offset + text_width + 5, y_offset + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(combined_bgr, text, (x_offset, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(str(save_filepath), combined_bgr)
