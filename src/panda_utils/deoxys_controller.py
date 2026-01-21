from typing import Callable
from time import time, sleep
from pathlib import Path
import threading
import os
from datetime import datetime
from trimesh.primitives import Box
import numpy as np
from termcolor import cprint
from tf.transformations import euler_from_matrix
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d

matplotlib.use("Agg")
from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
from jrl2.visualization import visualize_scene
from jrl2.robots import Panda as Jrl2Panda
from jrl2.math_utils import get_translated_pose
from jrl.robots import Panda
from deoxys.utils.yaml_config import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from panda_utils.constants import WORKSPACE_BOUNDS


PI_ON_4 = np.pi / 4
PI_ON_2 = np.pi / 2
VALID_QPOS_LIMIT_PADDING = 0.005
TABLE_PADDING = 0.03

RESET_JOINT_POSITIONS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

should_close = False
np.set_printoptions(linewidth=200, precision=4, suppress=True)


def wait_for_robot_ready(franka_interface: FrankaInterface):
    sleep(0.25)
    for i in range(10):
        if len(franka_interface._state_buffer) > 0:
            break
        if should_close:
            break
        cprint("Waiting for the robot to be ready", "yellow")
        sleep(1.0)
        if i == 3:
            s = (
                "Robot is not ready. Check:\n  1. the franka computer on? 2. deoxys is running on the RT PC "
                "(auto_arm.sh and auto_gripper.sh)? 3. is the E-stop released?"
            )
            cprint(s, "red")
            exit(1)
    cprint("Robot is ready", "green")


def rotate_matrix_by_angular_velocity(matrix: np.ndarray, omega: np.ndarray, t: float) -> np.ndarray:
    """Rotate matrix by angular velocity over time t. Note that t can be fairly large - 3 seconds for example.

    Args:
        matrix (np.ndarray): Initial rotation matrix [3, 3]
        omega (np.ndarray): Angular velocity [roll, pitch, yaw]
        t (float): Integration duration [s]

    Returns:
        np.ndarray: Rotated rotation matrix [3, 3]
    """
    assert (
        abs(np.linalg.det(matrix) - 1.0) < 1e-4
    ), f"matrix is not a rotation matrix, det(matrix) - 1.0: {np.linalg.det(matrix) - 1.0}"
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-8:
        return matrix.copy()

    # skew-symmetric cross-product matrix, then Rodrigues' formula for incremental rotation
    theta = omega_norm * t
    axis = omega / omega_norm
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    assert (
        abs(np.linalg.det(R_delta) - 1.0) < 1e-4
    ), f"R_delta is not a rotation matrix, det(R_delta) - 1.0: {np.linalg.det(R_delta) - 1.0}"
    return matrix @ R_delta


def _get_floor_box() -> Box:
    width_x = WORKSPACE_BOUNDS.extents[0]
    width_y = WORKSPACE_BOUNDS.extents[1]
    height = 0.02
    extents = np.array([width_x, width_y, height])
    center = np.zeros(3)
    center[0] = WORKSPACE_BOUNDS.center[0]
    center[1] = WORKSPACE_BOUNDS.center[1]
    center[2] = WORKSPACE_BOUNDS.min_z_m - height / 2 + TABLE_PADDING
    return Box(extents=extents, transform=get_translated_pose(center))


class DeoxysController:

    GRIPPER_OPEN_ANGLE = -1.0
    GRIPPER_CLOSED_ANGLE = 1.0

    def __init__(self, franka_interface: FrankaInterface, launch_viser: bool = True, viser_use_visual: bool = True):
        self._franka_interface = franka_interface
        wait_for_robot_ready(franka_interface)
        self._jrl_panda = Panda()
        self._jrl2_panda = Jrl2Panda()
        self._viser_use_visual = viser_use_visual
        self._collision_checker = SingleSceneCollisionChecker(self._jrl2_panda, use_visual=viser_use_visual)
        self._table_name = self._collision_checker.add_box(_get_floor_box())
        self._collision_checker.ignore_pair((self._table_name, "panda_link0::link0.stl"))
        self._collision_checker.ignore_pair((self._table_name, "panda_link0::link0.stl__col"))
        self._collision_checker.ignore_pair((self._table_name, "panda_link0::link0.dae"))

        # EE velocity control config
        self.ee_vel_control_controller_type = "JOINT_IMPEDANCE"
        self.ee_vel_control_config = YamlConfig("configs/joint-impedance-controller.yml").as_easydict()
        self.joint_impedance_control_config = YamlConfig("configs/joint-impedance-controller.yml").as_easydict()

        # Visualize collisions
        self._visualize_collision_scene_thread = None
        if launch_viser:
            self.start_viser()

    def start_viser(self):
        assert self._visualize_collision_scene_thread is None, "Viser thread already started"
        self._visualize_collision_scene_thread = threading.Thread(target=self._visualize_collision_scene)
        self._visualize_collision_scene_thread.start()

    def __del__(self):
        if hasattr(self, "_visualize_collision_scene_thread") and self._visualize_collision_scene_thread is not None:
            self._visualize_collision_scene_thread.join()
        self._franka_interface.close()

    def q_to_q_dict(self, qpos: np.ndarray, gripper_width: float) -> dict:
        assert qpos.shape == (7,), f"Expected shape (7,), got {qpos.shape}"
        return {
            "panda_joint1": qpos[0],
            "panda_joint2": qpos[1],
            "panda_joint3": qpos[2],
            "panda_joint4": qpos[3],
            "panda_joint5": qpos[4],
            "panda_joint6": qpos[5],
            "panda_joint7": qpos[6],
            "panda_finger_joint2": gripper_width / 2.0,
            "panda_finger_joint1": gripper_width / 2.0,
        }

    def _visualize_collision_scene(self):
        def get_q_dict():
            return self.q_to_q_dict(self._franka_interface.last_q, self._franka_interface.last_gripper_q)

        def get_pointclouds() -> dict[str, o3d.t.geometry.PointCloud]:
            return dict()

        visualize_scene(
            self._jrl2_panda,
            self._collision_checker,
            get_q_dict=get_q_dict,
            get_pointclouds=get_pointclouds,
            q_range_padding=VALID_QPOS_LIMIT_PADDING,
            use_visual=self._viser_use_visual,
            visualize_collisions=True,
        )

    def config_is_safe(self, qpos: np.ndarray) -> tuple[bool, list[tuple[str, str]]]:
        """Check if the configuration is safe."""
        is_colliding, contact_pairs, _ = self._collision_checker.check_collisions(
            self.q_to_q_dict(qpos, self._franka_interface.last_gripper_q), q_range_padding=VALID_QPOS_LIMIT_PADDING
        )
        return not is_colliding, contact_pairs

    def current_jacobian(self):
        qpos_current = self._franka_interface.last_q
        assert qpos_current is not None, "qpos_current is None"
        J_rotation_first = self._jrl_panda.jacobian_np(qpos_current)
        J_position_first = np.concatenate([J_rotation_first[3:6, :], J_rotation_first[0:3, :]], axis=0)
        return J_position_first

    def end_effector_velocity_control(
        self, twist: np.ndarray, tmax: float, should_stop: Callable[[], bool], gripper_width: float
    ):
        """Control the end effector velocity to a desired twist.

        Args:
            twist (np.ndarray): [6] - [x, y, z, rx, ry, rz] (m, rad)
            tmax (float): [s]
            should_stop (callable): [bool] - whether to stop the control loop
            gripper_width (float): [m] - gripper width
        """
        assert isinstance(twist, np.ndarray)
        assert twist.shape == (6,)

        # Warm start. This first control step takes ~1s. The 1s delay only happens when switching from a different
        # controller type. It also happens the first time the controller is used.
        t0_warm_start = time()
        self._franka_interface.control(
            controller_type=self.ee_vel_control_controller_type,
            action=self._franka_interface.last_q.tolist() + [gripper_width],
            controller_cfg=self.ee_vel_control_config,
        )
        print(f"end_effector_velocity_control() | Warm start took {time() - t0_warm_start:.5f} s")

        # Main loop
        ee_pose0 = self._franka_interface.last_eef_pose
        t0 = time()

        print(f"end_effector_velocity_control() | Starting with {twist=}")
        while True:
            t_elapsed = time() - t0
            if t_elapsed > tmax:
                break

            if should_stop():
                break

            # Get current state
            qpos_current = self._franka_interface.last_q
            ee_pose_current = self._franka_interface.last_eef_pose

            is_safe, contact_pairs = self.config_is_safe(qpos_current)
            if not is_safe:
                cprint("end_effector_velocity_control() | Configuration is not safe, exiting", "red")
                cprint(f"{contact_pairs=}", "red")
                return

            # Compute desired state
            ee_pose_desired = ee_pose0.copy()
            ee_pose_desired[:3, 3] = ee_pose_desired[:3, 3] + twist[0:3] * t_elapsed
            ee_pose_desired[:3, :3] = rotate_matrix_by_angular_velocity(ee_pose0[:3, :3], twist[3:6], t_elapsed)

            # Compute IK using Levenberg-Marquardt
            pose_errors_pos = ee_pose_desired[:3, 3] - ee_pose_current[:3, 3]
            error_rot_R = ee_pose_desired[:3, :3] @ ee_pose_current[:3, :3].T  # note: .T equals .inv() bc of SO(3)
            error_rot_euler = euler_from_matrix(error_rot_R)
            pose_error = np.concatenate([pose_errors_pos, error_rot_euler])

            # Run Levenberg-Marquardt
            lambd = 0.0001
            alpha = 1.0
            J = self.current_jacobian()  # [6, 7]
            J_T = J.T  # [7, 6]
            eye = np.eye(self._jrl_panda.ndof)
            lfs_A = J_T @ J + lambd * eye  # [7, 7]
            rhs_B = J_T @ pose_error  # [7]
            delta_q = np.linalg.solve(lfs_A, rhs_B)  # [7]
            q_target = qpos_current + alpha * delta_q

            # Send action
            self._franka_interface.control(
                controller_type=self.ee_vel_control_controller_type,
                action=q_target.tolist() + [gripper_width],
                controller_cfg=self.ee_vel_control_config,
            )
        print("end_effector_velocity_control() | Finished")

    def joint_position_control(self, q_pos_targets: np.ndarray, tmax: float, should_stop: Callable[[], bool]):
        """Control the robot to track the provided joint positions.

        Args:
            q_pos_targets (np.ndarray): [N, 9] - joint positions (rad) + gripper width
            tmax (float): [s]
            should_stop (callable): [bool] - whether to stop the control loop
            gripper_width (float): [m] - gripper width
        """
        assert isinstance(q_pos_targets, np.ndarray)
        assert q_pos_targets.shape[1] == 9

        # Main loop
        t0 = time()

        print(f"joint_position_control() | Starting with {q_pos_targets=}")
        for i in range(len(q_pos_targets)):
            t_elapsed = time() - t0
            if t_elapsed > tmax:
                break

            sleep(0.05)

            if should_stop():
                break

            # Get current state
            qpos_target_arm = q_pos_targets[i, :7]
            qpos_target_gripper = q_pos_targets[i, 7]

            is_safe, contact_pairs = self.config_is_safe(qpos_target_arm)
            if not is_safe:
                cprint("joint_position_control() | Configuration is not safe, exiting", "red")
                cprint(f"{contact_pairs=}", "red")
                return

            # Send action
            self._franka_interface.control(
                controller_type="JOINT_IMPEDANCE",
                action=qpos_target_arm.tolist() + [qpos_target_gripper],
                controller_cfg=self.joint_impedance_control_config,
            )

            print(
                f"joint_position_control() | {i=} / {len(q_pos_targets)=} \t {qpos_target_arm=} \t {qpos_target_gripper=}"
            )
        print("joint_position_control() | Finished")



    def joint_velocity_control(self, q_vel_targets: np.ndarray, tmax: float, should_stop: Callable[[], bool], gripper_width: float, save_plot: bool = False):
        """Control the robot to track the provided joint velocities.

        Args:
            q_vel_targets (np.ndarray): [N, 9] - joint velocities (rad/s) + gripper width
            tmax (float): [s]
            should_stop (callable): [bool] - whether to stop the control loop
            gripper_width (float): [rad] - gripper position
        """
        assert isinstance(q_vel_targets, np.ndarray)
        assert q_vel_targets.shape[1] == 7
        t_delay = 0.0

        # Main loop
        t0 = time()
        measured_frequency = 15.0 # an approximate

        scale = 2.0
        print(f"joint_velocity_control() | Starting with {q_vel_targets=}")
        for i in range(len(q_vel_targets)):
            t_elapsed = time() - t0
            print(f" {i} | {t_elapsed=} \t {tmax=}")
            if t_elapsed > tmax:
                break

            sleep(t_delay)

            if should_stop():
                print(f" {i} | should_stop()")
                break

            qvel_desired = q_vel_targets[i, :]
            dt = 1.0 / measured_frequency
            qpos_current = self._franka_interface.last_q
            qpos_target = qpos_current + (scale*qvel_desired * dt)

            # Get current state
            is_safe, contact_pairs = self.config_is_safe(qpos_target)
            if not is_safe:
                cprint("joint_position_control() | Configuration is not safe, exiting", "red")
                cprint(f"{contact_pairs=}", "red")
                return

            # Send action
            self._franka_interface.control(
                controller_type="JOINT_IMPEDANCE",
                action=qpos_target.tolist() + [gripper_width],
                controller_cfg=self.joint_impedance_control_config,
            )
            measured_frequency = (i+1) / t_elapsed
            print(f" {i} | {measured_frequency=}")

        print("joint_velocity_control() | Finished")



    def joint_velocity_control_PID(self, q_vel_targets: np.ndarray, tmax: float, should_stop: Callable[[], bool], gripper_width: float, save_plot: bool = False):
        """Control the robot to track the provided joint velocities.

        Args:
            q_vel_targets (np.ndarray): [N, 9] - joint velocities (rad/s) + gripper width
            tmax (float): [s]
            should_stop (callable): [bool] - whether to stop the control loop
            gripper_width (float): [rad] - gripper position
        """
        assert isinstance(q_vel_targets, np.ndarray)
        assert q_vel_targets.shape[1] == 7
        t_delay = 0.0

        # Main loop
        t0 = time()
        measured_frequency = 15.0 # an approximate

        if save_plot:
            qpos_history = []
            control_signal_history = []
            qvel_current_history = []
            qvel_desired_history = []
            t_elapsed_history = []

        scale = 0.25
        Kp = scale * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        Kd = scale * 1 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        Ki = scale * 0.001*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        qvel_error_integral = np.zeros(7)
        qvel_error_previous = np.zeros(7)

        print(f"joint_velocity_control() | Starting with {q_vel_targets=}")
        for i in range(len(q_vel_targets)):
            t_elapsed = time() - t0
            print(f" {i} | {t_elapsed=} \t {tmax=}")
            if t_elapsed > tmax:
                break

            sleep(t_delay)

            if should_stop():
                print(f" {i} | should_stop()")
                break

            qvel_desired = q_vel_targets[i, :]
            qvel_current = self._franka_interface.last_dq
            dt = 1.0 / measured_frequency

            # 
            qvel_error_now = qvel_desired - qvel_current
            qvel_error_integral += qvel_error_now * dt
            qvel_error_derivative = (qvel_error_now - qvel_error_previous) / dt

            # current_vel = self._franka_interface.last_dq
            control_signal = Kp * qvel_error_now + Ki * qvel_error_integral + Kd * qvel_error_derivative


            print(f" {i} | \n  qvel_error_now:\t{qvel_error_now}\n  qvel_error_integral:\t{qvel_error_integral}\n  qvel_error_derivative:\t{qvel_error_derivative}\n  control_signal:\t{control_signal}")

            # print(f" {i} | \n{qpos_current=} \t \n{desired_qvel=} \t \n{q_delta=} \t \n{qpos_target=}")

            # Get current state
            qpos_current = self._franka_interface.last_q
            qpos_target = qpos_current + control_signal
            is_safe, contact_pairs = self.config_is_safe(qpos_target)
            if not is_safe:
                cprint("joint_position_control() | Configuration is not safe, exiting", "red")
                cprint(f"{contact_pairs=}", "red")
                return

            if save_plot:
                qpos_history.append(qpos_current.copy())
                control_signal_history.append(control_signal.copy())
                qvel_current_history.append(qvel_current.copy())
                qvel_desired_history.append(qvel_desired.copy())
                t_elapsed_history.append(t_elapsed)

            # Send action
            self._franka_interface.control(
                controller_type="JOINT_IMPEDANCE",
                action=qpos_target.tolist() + [gripper_width],
                controller_cfg=self.joint_impedance_control_config,
            )
            qvel_error_previous = qvel_error_now.copy()
            measured_frequency = (i+1) / t_elapsed
            print(f" {i} | {measured_frequency=}")


        if save_plot and t_elapsed_history:
            qpos_history = np.array(qpos_history)
            control_signal_history = np.array(control_signal_history)
            qvel_current_history = np.array(qvel_current_history)
            qvel_desired_history = np.array(qvel_desired_history)
            t_elapsed_history = np.array(t_elapsed_history)

            qpos_history_deg = np.rad2deg(qpos_history)
            control_signal_deg = np.rad2deg(control_signal_history)
            qvel_current_deg = np.rad2deg(qvel_current_history)
            qvel_desired_deg = np.rad2deg(qvel_desired_history)

            fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 18), sharex=True)
            for joint_idx in range(7):
                axes[joint_idx, 0].plot(t_elapsed_history, qpos_history_deg[:, joint_idx])
                axes[joint_idx, 0].set_ylabel(f"q{joint_idx}")

                axes[joint_idx, 1].plot(
                    t_elapsed_history,
                    control_signal_deg[:, joint_idx],
                    label="control",
                )
                axes[joint_idx, 1].plot(
                    t_elapsed_history,
                    qvel_current_deg[:, joint_idx],
                    label="current_vel",
                )
                axes[joint_idx, 1].plot(
                    t_elapsed_history,
                    qvel_desired_deg[:, joint_idx],
                    label="target_vel",
                )
                axes[joint_idx, 0].minorticks_on()
                axes[joint_idx, 1].minorticks_on()
                axes[joint_idx, 0].grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.6)
                axes[joint_idx, 0].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
                axes[joint_idx, 1].grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.6)
                axes[joint_idx, 1].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

            axes[0, 0].set_title("Measured joint configuration")
            axes[0, 1].set_title("Control + joint velocities")
            axes[0, 1].legend(loc="upper right")
            axes[-1, 0].set_xlabel("time (s)")
            axes[-1, 1].set_xlabel("time (s)")
            plt.tight_layout()
            save_filepath = f"joint_velocity_control_plot__{datetime.now().strftime('%d_%H:%M:%S')}.png"
            plt.savefig(save_filepath)
            plt.close()
            os.system(f"xdg-open '{save_filepath}'")

        print("joint_velocity_control() | Finished")



"""
# Terminal 1
source /opt/ros/noetic/setup.bash; roscore


# Terminal 2
cd ${panda_utils_DIR}; source /opt/ros/noetic/setup.bash; source hardware_venv/bin/activate;
python panda_utils/deoxys_controller.py
"""

if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent
    franka_interface = FrankaInterface(f"{project_dir}/configs/charmander.yml")
    controller = DeoxysController(franka_interface, launch_viser=True, viser_use_visual=False)

    reset_joints_to(franka_interface, RESET_JOINT_POSITIONS)

    def should_stop():
        return False

    # ee_twist = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    # ee_twist = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    ee_twist = np.array([0.0, 0.0, -0.025, 0.0, 0.0, 0.0])
    # ee_twist = np.array([0.0, 0.0, 0.05, -0.05, -0.05, 0.05])
    tmax = 8.0
    controller.end_effector_velocity_control(
        ee_twist, tmax, should_stop, gripper_width=DeoxysController.GRIPPER_CLOSED_ANGLE
    )
    print("Controller finished. Sleeping until keyboard interrupt.")
    while True:
        try:
            sleep(10.0)
        except KeyboardInterrupt:
            cprint("Keyboard interrupt, exiting", "red")
            reset_joints_to(franka_interface, RESET_JOINT_POSITIONS)
            break
