"""SpaceMouse teleop for the Franka Panda, publishing joint states over ROS 2.

Requires:
  - deoxys interfaces running on the NUC (see README_SETUP.md)
  - ROS 2 Humble sourced: ``source /opt/ros/humble/setup.bash``
  - SpaceMouse readable (udev rule / correct vendor/product IDs)

# Example usage:
python scripts/run_teleop.py --vendor-id 1133 --product-id 50726
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from rclpy.node import Node
from sensor_msgs.msg import JointState
from termcolor import cprint

# panda_utils.constants pulls in ROS 1 (rospy via panda_utils.utils). Same fallback as
# scripts/log_demonstration.py. Keep in sync with panda_utils/constants.py.
try:
    from panda_utils.constants import SPACE_MOUSE_PRODUCT_ID, SPACE_MOUSE_VENDOR_ID
except Exception:  # noqa: BLE001 - any import error means the ROS 1 chain is unavailable
    SPACE_MOUSE_VENDOR_ID = 1133  # 0x046D
    SPACE_MOUSE_PRODUCT_ID = 50726  # 0xC626

JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]
PUBLISH_HZ = 50.0


def wait_for_deoxys_ready(robot_interface: FrankaInterface, timeout_s: float = 10.0) -> bool:
    """ROS-free wait until deoxys streams arm and gripper state."""
    deadline = time.time() + timeout_s
    failed_counter = 0
    while time.time() < deadline and rclpy.ok():
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q
        if arm_q is not None and last_gripper_q is not None:
            return True

        if failed_counter % 10 == 0:
            cprint(
                "Arm q or gripper angle isn't available. "
                f"Is deoxys running on the RT pc? last_gripper_q={last_gripper_q}, arm_q={arm_q}",
                "yellow",
            )
        failed_counter += 1
        time.sleep(1.0 / PUBLISH_HZ)

    cprint(
        f"Failed to get arm q or gripper angle within {timeout_s:.0f}s.",
        "red",
    )
    return False


def joint_publish_thread_target(node: Node, robot_interface: FrankaInterface) -> None:
    """Publish joint states to /panda/joint_states at PUBLISH_HZ."""
    if not wait_for_deoxys_ready(robot_interface):
        return

    pub = node.create_publisher(JointState, "/panda/joint_states", 10)
    period = 1.0 / PUBLISH_HZ
    np.set_printoptions(linewidth=200)

    counter = 0
    while rclpy.ok():
        start = time.monotonic()
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q
        if arm_q is None or last_gripper_q is None:
            time.sleep(period)
            continue

        gripper_angle = float(last_gripper_q) / 2.0
        js_msg = JointState()
        js_msg.header.stamp = node.get_clock().now().to_msg()
        js_msg.name = JOINT_NAMES
        assert isinstance(
            gripper_angle, float
        ), f"Gripper angle is not a float: {type(gripper_angle)=}, {gripper_angle=}"
        js_msg.position = arm_q.tolist() + [gripper_angle, gripper_angle]
        js_msg.velocity = robot_interface.last_dq.tolist() + [0.0, 0.0]
        js_msg.effort = [0.0] * 9
        pub.publish(js_msg)

        if counter % 200 == 0:
            node.get_logger().info(f"Arm q: {arm_q}")
        counter += 1

        elapsed = time.monotonic() - start
        time.sleep(max(0.0, period - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str)
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--vendor-id", type=int, default=SPACE_MOUSE_VENDOR_ID)
    parser.add_argument("--product-id", type=int, default=SPACE_MOUSE_PRODUCT_ID)
    args = parser.parse_args()

    rclpy.init(args=None)
    node = Node("panda_teleop")

    project_dir = Path(__file__).parent.parent
    if args.interface_cfg is None:
        args.interface_cfg = f"{project_dir}/configs/charmander.yml"

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    robot_interface._state_buffer = []

    joint_publish_thread = threading.Thread(
        target=joint_publish_thread_target,
        args=(node, robot_interface),
        daemon=True,
    )
    joint_publish_thread.start()

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    try:
        while rclpy.ok():
            # Keep the node clock/logging responsive while the main thread teleops.
            rclpy.spin_once(node, timeout_sec=0.0)
            action, grasp = input2action(
                device=device,
                controller_type=controller_type,
            )
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            time.sleep(1.0 / PUBLISH_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        cprint("Done, shutting down spacemouse control", "green")

        robot_interface.control(
            controller_type=controller_type,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
            controller_cfg=controller_cfg,
            termination=True,
        )
        robot_interface.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
