import rospy
from deoxys.franka_interface import FrankaInterface
import numpy as np
import cv2
from pathlib import Path


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


def wait_for_deoxys_ready(robot_interface: FrankaInterface) -> bool:
    rate = rospy.Rate(50)

    failed_counter = 0
    while not rospy.is_shutdown():
        rate.sleep()
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q

        # Check if the arm q or gripper angle is available
        if arm_q is None or last_gripper_q is None:
            if failed_counter % 10 == 0:
                rospy.logwarn(
                    f"Arm q or gripper angle isn't available. Is deoxys running on the RT pc? last_gripper_q: {last_gripper_q}, arm_q={arm_q}"
                )
            failed_counter += 1
            if failed_counter > 100:
                rospy.logerr("Failed to get arm q or gripper angle for 100 consecutive times. Shutting down.")
                rospy.signal_shutdown("Failed to get arm q or gripper angle for 100 consecutive times")
                return False
            continue

        return True

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
