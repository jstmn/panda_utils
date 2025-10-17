import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from time import time

import rospy
from sensor_msgs.msg import JointState
from deoxys.franka_interface import FrankaInterface
import numpy as np
from termcolor import cprint
from tf.transformations import euler_from_matrix
import matplotlib
from jrl.robots import Panda
import threading
import string
import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deoxys.utils.yaml_config import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to

from panda_utils.deoxys_controller import wait_for_robot_ready
from panda_utils.constants import PANDA_LIMITS


PI_ON_4 = np.pi / 4
PI_ON_2 = np.pi / 2

should_close = False
np.set_printoptions(linewidth=200, precision=4, suppress=True)


@dataclass
class Recording:
    times: np.ndarray
    q_pos: np.ndarray
    q_vel: np.ndarray
    q_des_pos: np.ndarray
    q_des_pos_user: np.ndarray
    q_des_vel: np.ndarray
    q_des_acc: np.ndarray
    tau: np.ndarray
    tau_vel: np.ndarray
    ee_xyz: np.ndarray
    ee_xyz_des: np.ndarray
    desired_xyz_set: bool

    @property
    def ee_x(self):
        return self.ee_xyz[:, 0]

    @property
    def ee_y(self):
        return self.ee_xyz[:, 1]

    @property
    def ee_z(self):
        return self.ee_xyz[:, 2]

    @property
    def ee_x_des(self):
        return self.ee_xyz_des[:, 0]

    @property
    def ee_y_des(self):
        return self.ee_xyz_des[:, 1]

    @property
    def ee_z_des(self):
        return self.ee_xyz_des[:, 2]

    @property
    def ee_x_dot(self):
        return (self.ee_x[1:] - self.ee_x[:-1]) / (self.times[1:] - self.times[:-1])

    @property
    def ee_y_dot(self):
        return (self.ee_y[1:] - self.ee_y[:-1]) / (self.times[1:] - self.times[:-1])

    @property
    def ee_z_dot(self):
        return (self.ee_z[1:] - self.ee_z[:-1]) / (self.times[1:] - self.times[:-1])

    def __str__(self):
        s = "Recording {"
        s += f"  times:     {self.times.shape},\n"
        s += f"  q_pos:     {self.q_pos.shape},\n"
        s += f"  q_vel:     {self.q_vel.shape},\n"
        s += f"  q_des_pos: {self.q_des_pos.shape},\n"
        s += f"  q_des_pos_user: {self.q_des_pos_user.shape},\n"
        s += f"  q_des_vel: {self.q_des_vel.shape},\n"
        s += f"  q_des_acc: {self.q_des_acc.shape},\n"
        s += f"  tau:       {self.tau.shape},\n"
        s += f"  tau_vel:   {self.tau_vel.shape},\n"
        s += f"  ee_x:      {self.ee_x.shape},\n"
        s += f"  ee_y:      {self.ee_y.shape},\n"
        s += f"  ee_z:      {self.ee_z.shape},\n"
        s += f"  ee_x_des:  {self.ee_x_des.shape},\n"
        s += f"  ee_y_des:  {self.ee_y_des.shape},\n"
        s += f"  ee_z_des:  {self.ee_z_des.shape},\n"
        s += f"  final xyz error, cm:  {100*(self.ee_xyz_des[-1] - self.ee_xyz[-1])},\n"
        s += "}\n"
        return s


def plot_results(recording: Recording, name: str):

    # Plot the data
    n_cols = 5
    fig, axs = plt.subplots(7, n_cols, figsize=(25, 20))
    print(recording)
    fig.suptitle(name, fontweight="bold")

    for i in range(7):
        q_pos_i = np.rad2deg(recording.q_pos[:, i])
        q_vel_i = np.rad2deg(recording.q_vel[:, i])
        q_des_pos_i = np.rad2deg(recording.q_des_pos[:, i])
        q_d_vel_i = np.rad2deg(recording.q_des_vel[:, i])
        q_des_pos_user_i = np.rad2deg(recording.q_des_pos_user[:, i])

        axs[i, 0].plot(recording.times, q_pos_i, label="measured")
        axs[i, 0].plot(recording.times, q_des_pos_i, label="desired")
        axs[i, 0].plot(recording.times, q_des_pos_user_i, label="desired, user")
        axs[i, 1].plot(recording.times, q_vel_i, label="measured")
        axs[i, 1].plot(recording.times, q_d_vel_i, label="desired")

        # Set y limits with 0.5 deg, 0.5 deg/s buffer
        pos_min = min(np.min(q_pos_i), np.min(q_des_pos_i), np.min(q_des_pos_user_i))
        pos_max = max(np.max(q_pos_i), np.max(q_des_pos_i), np.max(q_des_pos_user_i))
        axs[i, 0].set_ylim(pos_min - 0.5, pos_max + 0.5)
        vel_min = min(np.min(q_vel_i), np.min(q_d_vel_i))
        vel_max = max(np.max(q_vel_i), np.max(q_d_vel_i))
        axs[i, 1].set_ylim(vel_min - 0.5, vel_max + 0.5)
        axs[i, 1].axhline(
            y=np.rad2deg(PANDA_LIMITS.q_vel_max[i]), color="k", linestyle="dashed", alpha=0.7, label="max"
        )
        axs[i, 1].axhline(
            y=-np.rad2deg(PANDA_LIMITS.q_vel_max[i]), color="k", linestyle="dashed", alpha=0.7, label="min"
        )

        # Tau column
        axs[i, 3].plot(recording.times, recording.tau[:, i], label="tau")
        axs[i, 3].axhline(y=PANDA_LIMITS.tau_max[i], color="k", linestyle="dashed", alpha=0.7, label="max")
        axs[i, 3].axhline(y=-PANDA_LIMITS.tau_max[i], color="k", linestyle="dashed", alpha=0.7, label="min")

        # Tau dot column
        axs[i, 4].plot(recording.times, recording.tau_vel[:, i], label="tau_dot")
        axs[i, 4].axhline(y=PANDA_LIMITS.tau_dot_max[i], color="k", linestyle="dashed", alpha=0.7, label="max")
        axs[i, 4].axhline(y=-PANDA_LIMITS.tau_dot_max[i], color="k", linestyle="dashed", alpha=0.7, label="min")

        axs[i, 0].set_ylabel(f"Joint {i+1} [deg]")
        axs[i, 3].set_ylabel(f"Joint {i+1} [Nm]")
        axs[i, 4].set_ylabel(f"Joint {i+1} [Nm/s]")
        if i == 0:
            for j in range(n_cols):
                axs[i, j].set_xlabel("Time [s]")
            axs[i, 0].set_title("Q Position [deg]")
            axs[i, 1].set_title("Q Velocities [deg/s]")
            axs[i, 2].set_title("End Effector Position & Velocity")
            axs[i, 3].set_title("Tau [Nm]")
            axs[i, 4].set_title("Tau Dot [Nm/s]")

    # Plot end effector position in the third column (rows 0-2)
    padding = 1.0  # 1 cm
    x_min = min(np.min(100 * recording.ee_x), np.min(100 * recording.ee_x_des))
    x_max = max(np.max(100 * recording.ee_x), np.max(100 * recording.ee_x_des))
    y_min = min(np.min(100 * recording.ee_y), np.min(100 * recording.ee_y_des))
    y_max = max(np.max(100 * recording.ee_y), np.max(100 * recording.ee_y_des))
    z_min = min(np.min(100 * recording.ee_z), np.min(100 * recording.ee_z_des))
    z_max = max(np.max(100 * recording.ee_z), np.max(100 * recording.ee_z_des))
    axs[0, 2].set_ylim(x_min - padding, x_max + padding)
    axs[1, 2].set_ylim(y_min - padding, y_max + padding)
    axs[2, 2].set_ylim(z_min - padding, z_max + padding)

    # Plot end effector position
    axs[0, 2].plot(recording.times, 100 * recording.ee_x, label="Measured")
    axs[1, 2].plot(recording.times, 100 * recording.ee_y, label="Measured")
    axs[2, 2].plot(recording.times, 100 * recording.ee_z, label="Measured")
    axs[0, 2].set_ylabel("X [cm]")
    axs[1, 2].set_ylabel("Y [cm]")
    axs[2, 2].set_ylabel("Z [cm]")
    if recording.desired_xyz_set:
        axs[0, 2].plot(recording.times, 100 * recording.ee_x_des, label="Desired")
        axs[1, 2].plot(recording.times, 100 * recording.ee_y_des, label="Desired")
        axs[2, 2].plot(recording.times, 100 * recording.ee_z_des, label="Desired")

    # Plot end effector velocity in the third column (rows 3-5)
    axs[3, 2].plot(recording.times[1:], 100 * recording.ee_x_dot, label="Measured")
    axs[3, 2].set_ylabel("X Vel [cm/s]")
    axs[4, 2].plot(recording.times[1:], 100 * recording.ee_y_dot, label="Measured")
    axs[4, 2].set_ylabel("Y Vel [cm/s]")
    axs[5, 2].plot(recording.times[1:], 100 * recording.ee_z_dot, label="Measured")
    axs[5, 2].set_ylabel("Z Vel [cm/s]")
    # Plot end effector velocity limits
    axs[3, 2].axhline(
        y=100 * PANDA_LIMITS.translation_vel_max[0], color="k", linestyle="dashed", alpha=0.7, label="max"
    )
    axs[3, 2].axhline(
        y=-100 * PANDA_LIMITS.translation_vel_max[0], color="k", linestyle="dashed", alpha=0.7, label="min"
    )
    axs[4, 2].axhline(
        y=100 * PANDA_LIMITS.translation_vel_max[1], color="k", linestyle="dashed", alpha=0.7, label="max"
    )
    axs[4, 2].axhline(
        y=-100 * PANDA_LIMITS.translation_vel_max[1], color="k", linestyle="dashed", alpha=0.7, label="min"
    )
    axs[5, 2].axhline(
        y=100 * PANDA_LIMITS.translation_vel_max[2], color="k", linestyle="dashed", alpha=0.7, label="max"
    )
    axs[5, 2].axhline(
        y=-100 * PANDA_LIMITS.translation_vel_max[2], color="k", linestyle="dashed", alpha=0.7, label="min"
    )

    # Clear unused subplot in third column row 6
    axs[6, 2].set_visible(False)

    for ax in axs.flatten():
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
        ax.minorticks_on()
        ax.legend()
        ax.yaxis.get_major_formatter().set_useOffset(False)

    lst = [random.choice(string.ascii_letters + string.digits) for n in range(6)]
    sid = "".join(lst).lower()

    plt.tight_layout()
    save_filepath = f"data/deoxys_tuning/{name}/{datetime.now().strftime('%d__%H:%M:%S')} - {sid}.png"
    Path(f"data/deoxys_tuning/{name}").mkdir(parents=True, exist_ok=True)
    plt.savefig(save_filepath)
    print(f"Saved plot to '{save_filepath}'")
    print(f"xdg-open '{save_filepath}'")
    print()


def joint_publish_thread_target(robot_interface: FrankaInterface):
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    pub = rospy.Publisher("panda/joint_states", JointState, queue_size=10)
    joint_names = [
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
    rate = rospy.Rate(60)

    counter = 0
    while not rospy.is_shutdown():

        if should_close:
            cprint(f"[joint_publish_thread_target] {should_close=} -> exiting", "green")
            break

        rate.sleep()
        # try:
        # Publish robot joint states
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q
        if arm_q is None or last_gripper_q is None:
            rospy.logwarn("Arm q_pos or gripper angle isn't available")
            continue

        gripper_angle = float(last_gripper_q) / 2.0
        js_msg = JointState()
        js_msg.header.stamp = rospy.Time.now()
        js_msg.name = joint_names
        assert isinstance(
            gripper_angle, float
        ), f"Gripper angle is not a float: {type(gripper_angle)=}, {gripper_angle=}"
        js_msg.position = arm_q.tolist() + [gripper_angle, gripper_angle]
        js_msg.velocity = robot_interface.last_dq.tolist() + [0.0, 0.0]
        js_msg.effort = [0.0] * 9
        pub.publish(js_msg)

        if counter % 300 == 0:
            rospy.loginfo(f"Arm q_pos: {arm_q}")
        counter += 1


class DeoxysControllerTester:

    def __init__(self, franka_interface: FrankaInterface):
        self.franka_interface = franka_interface
        self.jrl_panda = Panda()

        self.should_start_recording = False
        self.should_end_recording = False
        self.recording_name = ""
        self.recording_time_max = 0.0
        self.desired_xyz = np.zeros(3)
        self.desired_q_pos_user = np.zeros(7)
        self.recording_thread = threading.Thread(target=self.recording_thread_target)
        self.recording_thread.start()

    def recording_thread_target(self):
        rate_hz = 500
        rate = rospy.Rate(rate_hz)
        global should_close
        am_recording = False
        t_recording_start = time()
        i = 0

        while not should_close:
            rate.sleep()
            if i > 0 and i % rate_hz == 0:
                print(f"Recording rate: {i / (time() - t_recording_start):.1f} Hz")

            if self.should_start_recording:
                self.should_start_recording = False
                am_recording = True
                t_recording_start = time()
                self.set_control_records(int(self.recording_time_max * 1.1 * rate_hz))
                i = 0
                print("[recording_thread_target()] Started recording")

            if self.should_end_recording:
                self.should_end_recording = False
                am_recording = False
                print("[recording_thread_target()] Ended recording")
                desired_xyz_is_set = np.max(np.abs(self.ee_xyz_des[0:i] - self.ee_xyz[0:i])) > 0.001

                recording = Recording(
                    times=self.times[0:i],
                    q_pos=self.q_pos[0:i],
                    q_vel=self.q_vel[0:i],
                    q_des_pos=self.q_des_pos[0:i],
                    q_des_pos_user=self.q_des_pos_user[0:i],
                    q_des_vel=self.q_des_vel[0:i],
                    q_des_acc=self.q_des_acc[0:i],
                    tau=self.torques[0:i],
                    tau_vel=self.d_torques[0:i],
                    ee_xyz=self.ee_xyz[0:i],
                    ee_xyz_des=self.ee_xyz_des[0:i],
                    desired_xyz_set=desired_xyz_is_set,
                )
                plot_results(recording, self.recording_name)

            if am_recording:
                self.log_current_state(i)
                i += 1

    def set_control_records(self, n: int):
        self.t0 = time()
        self.times = np.zeros(n)
        self.torques = np.zeros((n, 7))
        self.d_torques = np.zeros((n, 7))
        self.q_pos = np.zeros((n, 7))
        self.q_vel = np.zeros((n, 7))
        self.q_des_pos = np.zeros((n, 7))
        self.q_des_vel = np.zeros((n, 7))
        self.q_des_acc = np.zeros((n, 7))
        self.q_des_pos_user = np.zeros((n, 7))
        self.ee_xyz = np.zeros((n, 3))
        self.ee_xyz_des = np.zeros((n, 3))

    def log_current_state(self, i: int):
        ee_pos = self.franka_interface.last_eef_pose[:3, 3]
        self.times[i] = time() - self.t0
        self.torques[i] = self.franka_interface.last_joint_torques
        self.d_torques[i] = self.franka_interface.last_d_joint_torques
        self.q_pos[i] = self.franka_interface.last_q
        self.q_vel[i] = self.franka_interface.last_dq
        self.q_des_pos[i] = self.franka_interface.last_q_d
        self.q_des_pos_user[i] = self.desired_q_pos_user
        self.q_des_vel[i] = self.franka_interface.last_qd_d
        self.q_des_acc[i] = self.franka_interface.last_qdd_d
        self.ee_xyz[i] = ee_pos
        self.ee_xyz_des[i] = self.desired_xyz

    def start_recording(self, name: str, time_max: float):
        self.should_start_recording = True
        self.recording_name = name
        self.recording_time_max = time_max

    def stop_recording(self):
        self.should_end_recording = True
        global should_close
        should_close = True

    def current_jacobian(self):
        qpos_current = self.franka_interface.last_q
        assert qpos_current is not None, "qpos_current is None"
        J_rotation_first = self.jrl_panda.jacobian_np(qpos_current)
        J_position_first = np.concatenate([J_rotation_first[3:6, :], J_rotation_first[0:3, :]], axis=0)
        return J_position_first

    def __del__(self):
        self.stop_recording()
        self.franka_interface.close()
        cprint("[DeoxysControllerTester] done", "green")

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    def CARTESIAN_VELOCITY_2(self, plot_results: bool = False):
        controller_type = "JOINT_IMPEDANCE"
        controller_cfg = YamlConfig("configs/joint-impedance-controller.yml").as_easydict()
        # Warm start. This first control step takes ~1s.
        self.franka_interface.control(
            controller_type=controller_type,
            action=self.franka_interface.last_q.tolist() + [-1.0],
            controller_cfg=controller_cfg,
        )

        # Main loop
        # Tmax = 3.0
        Tmax = 2.0
        # Tmax = 0.5
        ee_pose0 = self.franka_interface.last_eef_pose
        # ee_vel_desired = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        ee_vel_desired = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        n_steps = 0
        t0 = time()
        global should_close
        q_target = self.franka_interface.last_q

        self.start_recording("CARTESIAN_VELOCITY_2", Tmax)

        while True:
            t_elapsed = time() - t0
            if t_elapsed > Tmax:
                break

            if should_close:
                break

            if t_elapsed < Tmax:
                controller_dt = t_elapsed / n_steps if n_steps > 0 else 0.0
                ee_pose_desired = ee_pose0.copy()
                ee_pose_desired[:3, 3] = ee_pose_desired[:3, 3] + ee_vel_desired[0:3] * t_elapsed
                # ee_pose_desired[:3, :3] = ee_pose0[:3, :3] # TODO: integrate rotation

                # Compute IK using Levenberg-Marquardt
                qpos_current = self.franka_interface.last_q
                ee_pose_current = self.franka_interface.last_eef_pose
                pose_errors_pos = ee_pose_desired[:3, 3] - ee_pose_current[:3, 3]
                error_rot_R = ee_pose_desired[:3, :3] @ ee_pose_current[:3, :3].T  # note: .T equals .inv() bc of SO(3)
                error_rot_euler = euler_from_matrix(error_rot_R)
                pose_error = np.concatenate([pose_errors_pos, error_rot_euler])

                # Run Levenberg-Marquardt
                lambd = 0.0001
                alpha = 1.0
                J = self.current_jacobian()  # [6, 7]
                J_T = J.T  # [7, 6]
                eye = np.eye(self.jrl_panda.ndof)
                lfs_A = J_T @ J + lambd * eye  # [7, 7]
                rhs_B = J_T @ pose_error  # [7]
                delta_q = np.linalg.solve(lfs_A, rhs_B)  # [7]
                q_target = qpos_current + alpha * delta_q

            # Send action
            self.franka_interface.control(
                controller_type=controller_type,
                action=q_target.tolist() + [-1.0],
                controller_cfg=controller_cfg,
            )

            # Debugging, timing
            q_error = np.rad2deg(np.max(np.abs(q_target - qpos_current)))
            # print(f"{n_steps=} dt={controller_dt:.5f} s    {time() - t0:.4f} elapsed  -  pos, current: {100*ee_pose_current[:3, 3]} (cm)  -  pos, desired: {100*ee_pose_desired[:3, 3]} (cm)  -  q_error: {q_error:.5f} (deg)")
            print(
                f"{n_steps=} dt={controller_dt:.5f} s    {time() - t0:.4f} elapsed  -  error_pos: {100*pose_errors_pos} (cm)  -  error_rot: {np.rad2deg(error_rot_euler)} (deg)  -  q_error: {q_error:.5f} (deg)"
            )
            # print(f"{n_steps=} dt={controller_dt:.5f} s    {time() - t0:.4f} elapsed  -  error_pos: {100*pose_errors_pos} (cm)  -  error_rot: {np.rad2deg(error_rot_euler)} (deg)")
            n_steps += 1
            self.desired_xyz = ee_pose_desired[:3, 3]
            self.desired_q_pos_user = q_target

        print(f"{self.franka_interface.last_q=}")
        print(f"[CARTESIAN_VELOCITY_2] Finished after {time() - t0:.4f} seconds. Ave Hz: {n_steps / (time() - t0)}")
        self.stop_recording()

    def CARTESIAN_VELOCITY(self, plot_results: bool = False):
        controller_type = "CARTESIAN_VELOCITY"
        controller_cfg = get_default_controller_config(controller_type=controller_type)

        print(f"{controller_cfg=}")

        # Hypothesis: the torque thershold is being reached causing the arm controller to crash.
        # Actual issue: discontinuities in the joint acceleration. "cartesian_motion_generator_joint_acceleration_discontinuity: true"

        # Warm start
        self.franka_interface.control(
            controller_type=controller_type,
            action=[0, 0, 0.0, 0, 0, 0] + [-1.0],
            controller_cfg=controller_cfg,
        )

        #
        tnow = time()
        t0 = tnow
        time_max = 5
        t_elapsed = 0.0

        self.start_recording("CARTESIAN_VELOCITY", time_max)
        counter = 0

        while t_elapsed < time_max:
            counter += 1
            if counter % 25 == 0:
                print(f"[CARTESIAN_VELOCITY] control rate: {counter / (time() - t0):.2f} Hz")
            t_elapsed = time() - t0
            # vx = 0.05 * (1 - np.abs(np.cos(np.pi * i / 100))) # <- WORKING, from the deoxys example.

            #
            # v_z = 0.05*t_elapsed/time_max
            v_z = 0.05 * t_elapsed / time_max  # working with tmax=5
            action = [0, 0.0, v_z, 0, 0, 0] + [-1]
            # action = [0, 0.0, 0.001, 0, 0, 0] + [-1]

            self.franka_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            error = str(self.franka_interface._state_buffer[-1].current_errors)
            if len(error) > 0:
                cprint(f"ERROR: {error}", "red")
                break

        print(self.franka_interface.last_q)
        print(f"[CARTESIAN_VELOCITY] finished after {time() - t0:.4f} seconds")
        self.stop_recording()

    def OSC_POSITION(self, plot_results: bool = False):
        controller_type = "OSC_POSITION"
        controller_cfg = get_default_controller_config(controller_type=controller_type)
        global should_close
        # ERROR: cartesian_motion_generator_joint_acceleration_discontinuity: true
        n = 100
        self.set_control_records(n)
        tnow = time()
        for i in range(n):
            if should_close:
                break
            tnow = time()
            error = str(self.franka_interface._state_buffer[-1].current_errors)
            if len(error) > 0:
                print(error)

            action = [0, 0, 0.25] + [0.0, 0.0, 0.0] + [-1.0]
            self.franka_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            self.log_current_state(i)

        if plot_results:
            self._plot_results(name="OSC_POSITION", imax=i, plot_torques=False)

    def JOINT_POSITION(self, plot_results: bool = False):
        controller_type = "JOINT_POSITION"
        controller_cfg = get_default_controller_config(controller_type=controller_type)
        scale = 0.2
        target_joint_positions = [
            e + np.clip(np.random.randn() * scale, -scale, scale) for e in self.franka_interface.last_q
        ]
        action = target_joint_positions + [-1.0]

        self.start_recording("JOINT_POSITION", 10.0)

        while True:
            error = np.max(
                np.abs(np.array(self.franka_interface._state_buffer[-1].q_pos) - np.array(target_joint_positions))
            )
            print(f"error: {error}")
            if error < 1e-3:
                break
            self.franka_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        self.stop_recording()

    def JOINT_POSITION_INTERPOLATION_TEST(self, plot_results: bool = False):
        controller_type = "JOINT_POSITION"
        controller_cfg = get_default_controller_config(controller_type=controller_type)
        q0 = self.franka_interface.last_q
        q_target = np.array([0.094, -0.336, -0.02, -2.212, -0.013, 1.905, 0.848])
        action = q_target + [-1.0]
        tmax = 0.5
        t_start = time()
        self.start_recording("JOINT_POSITION_INTERPOLATION_TEST", tmax)

        while time() - t_start < tmax:
            t_elapsed = time() - t_start
            s = t_elapsed / tmax
            action = q_target * (1 - s) + q0 * s
            self.franka_interface.control(
                controller_type=controller_type,
                action=action.tolist() + [-1.0],
                controller_cfg=controller_cfg,
            )
        self.stop_recording()

    def OSC_POSE(self):
        global should_close
        controller_type = "OSC_POSE"
        controller_cfg = get_default_controller_config(controller_type=controller_type)
        # ERROR: cartesian_motion_generator_joint_acceleration_discontinuity: true
        n = 300
        self.set_control_records(n)

        tnow = time()
        t0 = tnow
        action = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0] + [-1])  # works pretty well
        # action = np.array([0.1, 0.0, 0.15, 0.0, 0.0, 0.0] + [-1]) # no progress made
        # action = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1]) # no progress made
        # action = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1])
        # action = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0] + [-1])
        # frozen_dimensions=[ True,  True, False,  True,  True,  True], active_dimensions=[False, False,  True, False, False, False]
        # action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1] + [-1])
        frozen_dimensions = np.abs(action[0:6]) < 0.001
        active_dimensions = np.logical_not(frozen_dimensions)

        initial_pose = self.franka_interface.last_eef_pose
        pose_history = []

        print(f"{frozen_dimensions=}")
        print(f"{active_dimensions=}")
        print(f"{initial_pose=}")

        for i in range(n):
            if should_close:
                cprint(f"[OSC_POSE_control_testing] {should_close=} -> exiting", "green")
                break

            current_pose = self.franka_interface.last_eef_pose
            current_pos = current_pose[:3, 3]
            pos_error = current_pos - initial_pose[:3, 3]
            # Compute rotational error using tf transformations (scipy.spatial.transform)

            # Get rotation matrices
            R_current = current_pose[:3, :3]
            R_initial = initial_pose[:3, :3]
            rot_error_R = R_current @ R_initial.T
            rot_error_euler = np.array(euler_from_matrix(rot_error_R))

            # Create updated action
            error_correction = np.concatenate([pos_error, rot_error_euler])
            error_correction[active_dimensions] = 0.0
            # ^ Set the active dimensions to 0. This means only the dimensions which shouldn't be moving will be corrected.
            updated_action = action.copy()
            #
            updated_action[0:3] -= 5.0 * error_correction[0:3]
            updated_action[3:6] -= 0.5 * error_correction[3:6]

            print(
                f"pos_error (cm): {100*pos_error}     rot_error_euler, deg: {np.rad2deg(rot_error_euler)}     action: {action}     updated_action: {updated_action}     error_correction: {error_correction}"
            )
            self.franka_interface.control(
                controller_type=controller_type,
                action=updated_action,
                controller_cfg=controller_cfg,
            )
            pose_history.append(self.franka_interface.last_eef_pose)

            # Logging
            system_error = str(self.franka_interface._state_buffer[-1].current_errors)
            if len(system_error) > 0:
                print(system_error)
            self.times[i] = time() - t0

        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        pose_history = np.array(pose_history)
        pose_history_euler = np.array(
            [np.array(euler_from_matrix(pose_history[i, :3, :3])) % (2 * np.pi) for i in range(len(pose_history))]
        )

        for i in range(3):
            # xyz
            val_initial = 100 * pose_history[0, i, 3]
            val_final = 100 * pose_history[-1, i, 3]
            vals_t = 100 * pose_history[:, i, 3]
            axs[i, 0].plot(self.times[0 : len(pose_history)], vals_t, label="Measured")
            axs[i, 0].axhline(val_initial, color="green", linestyle="--", linewidth=1, label="Initial")
            axs[i, 0].set_ylabel(f"{'XYZ'[i]} [cm]")
            assert (val_initial > 0 and val_final > 0) or (
                val_initial < 0 and val_final < 0
            ), f"{val_initial=}, {val_final=}"
            if val_initial > 0.0:
                axs[i, 0].set_ylim(0, max(val_initial, val_final) * 1.1)
            else:
                axs[i, 0].set_ylim(min(val_initial, val_final) * 1.1, 0)

            # Rotation
            vals_r = np.rad2deg(pose_history_euler[:, i])
            axs[i, 1].plot(self.times[0 : len(pose_history)], vals_r, label="Measured")
            axs[i, 1].axhline(vals_r[0], color="green", linestyle="--", linewidth=1, label="Initial, 1")
            axs[i, 1].set_ylabel(f"{['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'][i]}")
            minv = np.min(vals_r)
            maxv = np.max(vals_r)
            if minv > 0:
                axs[i, 1].set_ylim(0, maxv * 1.1)
            else:
                axs[i, 1].set_ylim(minv * 1.1, 0)

        for ax in axs.flatten():
            ax.set_xlabel("Time [s]")
            ax.grid(which="major", linestyle="-", linewidth=0.75, alpha=0.8)
            ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
            ax.minorticks_on()
            ax.legend()

        Path("data/deoxys_tuning").mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"data/deoxys_tuning/OSC_POSE__+z__EE_pose__{datetime.now().strftime('%d_%H:%M:%S')}.png")
        plt.close()
        should_close = True
        return self.times, None, None, None, None, None, None, None


def main(deoxys_interface_cfg: str, dont_plot: bool, method: str):
    """Testing the different control types.

    OSC_POSE: controls the end effector in the world frame. Note that there is drift in the end effector rotation and
            position, so that the EE pose will change along dimensions which were not specified in the action.


    Args:
        task (str): The ManiSkill task to run.
        num_wm_envs (int): The number of world model environments to use. These will be used to decide an action for the
                            'real' environment


    """
    global should_close

    def shutdown_handler():
        global should_close
        cprint("Shutting down...", "green")
        should_close = True

    rospy.init_node("franka_mpc_node", anonymous=True)
    rospy.on_shutdown(shutdown_handler)
    franka_interface = FrankaInterface(deoxys_interface_cfg, use_visualizer=False)
    controller = DeoxysControllerTester(franka_interface)  # noqa: F841
    # __init__ takes a few seconds so we warmstart it here.
    wait_for_robot_ready(franka_interface)

    # reset_joint_positions = [  0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4 ]
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]
    reset_joints_to(franka_interface, reset_joint_positions)

    # Start the joint state publisher thread
    # joint_publish_thread = threading.Thread(target=joint_publish_thread_target, args=(franka_interface,))
    # joint_publish_thread.start()
    eval(f"controller.{method}(plot_results=not dont_plot)")


""" 
source /opt/ros/noetic/setup.bash; source ${ROS_WS}/devel/setup.bash; conda activate hardware_env; cd ${PANDA_UTILS_DIR}

# Run
python scripts/deoxys_tuning.py \
    --deoxys-interface-cfg configs/charmander.yml \
    --dont-plot --method CARTESIAN_VELOCITY_2
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--deoxys-interface-cfg", type=str, required=True)
    parser.add_argument("--dont-plot", action="store_true")
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    main(args.deoxys_interface_cfg, args.dont_plot, args.method)
