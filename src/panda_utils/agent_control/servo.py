"""Closed-loop servo to an absolute end effector pose using deoxys OSC_POSE delta goals."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config

from panda_utils.agent_control.geometry import pose_error

CONTROLLER_TYPE = "OSC_POSE"

# deoxys gripper convention: -1 fully open, +1 fully closed.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0


@dataclass
class ServoResult:
    reached: bool
    position_error_m: float
    rotation_error_rad: float
    duration_s: float


class EEPoseServo:
    """Sends capped OSC_POSE delta goals toward a target pose until it is within tolerance or the timeout expires.

    One-shot OSC goals under-track on this rig (see configs/osc-pose-tuned.yml), so the remaining error is
    re-measured and re-sent every control tick. FrankaInterface.control() paces the loop at the interface config's
    POLICY_RATE. deoxys applies a delta goal's rotation in the base frame (R_goal = R_delta @ R_current), which is
    what pose_error() returns.
    """

    def __init__(
        self,
        robot: FrankaInterface,
        max_step_m: float = 0.02,
        max_step_rad: float = np.deg2rad(8.0),
        position_tolerance_m: float = 0.005,
        rotation_tolerance_rad: float = np.deg2rad(2.0),
        timeout_s: float = 6.0,
        gripper_settle_s: float = 1.0,
    ):
        self._robot = robot
        self._cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
        self._max_step_m = max_step_m
        self._max_step_rad = max_step_rad
        self._position_tolerance_m = position_tolerance_m
        self._rotation_tolerance_rad = rotation_tolerance_rad
        self._timeout_s = timeout_s
        self._gripper_settle_s = gripper_settle_s
        self._last_gripper: float | None = None

    def _send(self, step_m: np.ndarray, step_rotvec: np.ndarray, gripper: float, termination: bool = False) -> None:
        # control() multiplies by action_scale, so divide it back out to command metres / radians.
        action = np.concatenate(
            [
                step_m / self._cfg.action_scale.translation,
                step_rotvec / self._cfg.action_scale.rotation,
                [gripper],
            ]
        )
        self._robot.control(
            controller_type=CONTROLLER_TYPE, action=action, controller_cfg=self._cfg, termination=termination
        )

    def move_to(
        self, T_target: np.ndarray, gripper: float, should_stop: Callable[[], bool] = lambda: False
    ) -> ServoResult:
        """Servo to `T_target` [4, 4] (base frame), then apply `gripper` and hold while it settles."""
        t0 = time.time()
        reached = False
        while not should_stop():
            e_pos, e_rot = pose_error(T_target, self._robot.last_eef_pose)
            pos_norm, rot_norm = np.linalg.norm(e_pos), np.linalg.norm(e_rot)
            if pos_norm < self._position_tolerance_m and rot_norm < self._rotation_tolerance_rad:
                reached = True
                break
            if time.time() - t0 > self._timeout_s:
                break
            step_m = e_pos if pos_norm <= self._max_step_m else e_pos * (self._max_step_m / pos_norm)
            step_rot = e_rot if rot_norm <= self._max_step_rad else e_rot * (self._max_step_rad / rot_norm)
            # Gripper keeps its previous command during the motion; the new one is applied once there.
            self._send(step_m, step_rot, self._last_gripper if self._last_gripper is not None else gripper)

        if gripper != self._last_gripper and not should_stop():
            self.hold(gripper, self._gripper_settle_s, should_stop)
        self._last_gripper = gripper

        e_pos, e_rot = pose_error(T_target, self._robot.last_eef_pose)
        return ServoResult(reached, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot)), time.time() - t0)

    def hold(self, gripper: float, duration_s: float, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Hold the current pose while commanding `gripper`."""
        t_end = time.time() + duration_s
        while time.time() < t_end and not should_stop():
            self._send(np.zeros(3), np.zeros(3), gripper)
        self._last_gripper = gripper

    def terminate(self) -> None:
        """Send deoxys' termination command so the NUC-side controller stops tracking the last goal."""
        self._send(
            np.zeros(3),
            np.zeros(3),
            self._last_gripper if self._last_gripper is not None else GRIPPER_OPEN,
            termination=True,
        )
