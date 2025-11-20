#!/home/jstm/Projects/mpcm2/deoxys_venv/bin/python3
import argparse
import rospy
from sensor_msgs.msg import JointState
from deoxys.franka_interface import FrankaInterface
import numpy as np
import time
import threading
from termcolor import cprint
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse


def joint_publish_thread_target(robot_interface: FrankaInterface):
    pub = rospy.Publisher('panda/joint_states', JointState, queue_size=10)
    joint_names = [
        'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
        'panda_joint5', 'panda_joint6', 'panda_joint7',
        'panda_finger_joint1', 'panda_finger_joint2'
    ]
    rate = rospy.Rate(50)
    np.set_printoptions(linewidth=200)

    counter = 0
    failed_counter = 0
    while not rospy.is_shutdown():
        rate.sleep()
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q

        # Check if the arm q or gripper angle is available
        if arm_q is None or last_gripper_q is None:
            if failed_counter % 10 == 0:
                rospy.logwarn(f"Arm q or gripper angle isn't available. Is deoxys running on the RT pc? last_gripper_q: {last_gripper_q}, arm_q={arm_q}")
            failed_counter += 1
            if failed_counter > 100:
                rospy.logerr("Failed to get arm q or gripper angle for 100 consecutive times. Shutting down.")
                rospy.signal_shutdown("Failed to get arm q or gripper angle for 100 consecutive times")
                return
            continue
        failed_counter = 0

        gripper_angle = float(last_gripper_q) / 2.0
        js_msg = JointState()
        js_msg.header.stamp = rospy.Time.now()
        js_msg.name = joint_names
        assert isinstance(gripper_angle, float), f"Gripper angle is not a float: {type(gripper_angle)=}, {gripper_angle=}"
        js_msg.position = arm_q.tolist() + [gripper_angle, gripper_angle]
        js_msg.velocity = robot_interface.last_dq.tolist() + [0.0, 0.0]
        js_msg.effort = [0.0] * 9
        pub.publish(js_msg)

        if counter % 200 == 0:
            rospy.loginfo(f"Arm q: {arm_q}")
        counter += 1




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    args = parser.parse_args()
    rospy.init_node('panda_teleop', anonymous=False)

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    robot_interface._state_buffer = []

    joint_publish_thread = threading.Thread(target=joint_publish_thread_target, args=(robot_interface,))
    joint_publish_thread.start()

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    while not rospy.is_shutdown():
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        time.sleep(1/50.0)

    cprint("Done, shutting down spacemouse control", "green")

    robot_interface.control(
        controller_type=controller_type,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=controller_cfg,
        termination=True,
    )

    robot_interface.close()


if __name__ == "__main__":
    main()
