import argparse
from time import sleep
from typing import Callable
import select
import sys
from pathlib import Path
from datetime import datetime

import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
import numpy as np
import h5py
import json
import threading
from termcolor import cprint
from panda_utils.constants import DEFAULT_DEOXYS_INTERFACE_CFG


from panda_utils.deoxys_controller import RESET_JOINT_POSITIONS
from panda_utils.utils import wait_for_deoxys_ready, to_public_dict, save_combined_rgbs




if __name__ == "__main__":
    rospy.init_node("reset_panda", anonymous=True)
    robot_interface = FrankaInterface(DEFAULT_DEOXYS_INTERFACE_CFG, use_visualizer=False)
    wait_for_deoxys_ready(robot_interface)
    reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)
    reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)