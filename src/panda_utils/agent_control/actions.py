"""The model's action, and how it is executed: the structured end effector pose delta the model returns, the pose math
that turns it into a target pose, and the servo that drives the arm there, checking for self and table collisions.

Execution uses OSC_POSE_PRECISE from our deoxys fork (deoxys/franka_interface/osc_pose_precise.py): OSC_POSE's base
frame delta action, tracked with per-call IK steps through the joint impedance controller; deoxys' own OSC_POSE barely
moves this arm for goals of a few cm. deoxys and jrl2 are imported only by EEPoseServo / CollisionChecker, so the
schema and pose math stay importable without them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from deoxys.franka_interface import FrankaInterface

    from panda_utils.types import WorkspaceBounds

CONTROL_FRAMES = ("world", "ee")
CONTROLLER_TYPE = "OSC_POSE_PRECISE"

# deoxys gripper convention: -1 fully open, +1 fully closed.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
# The table: a 2 cm slab spanning WORKSPACE_BOUNDS in x and y, with its top TABLE_PADDING_M above the workspace floor.
# DeoxysController uses 1.5 cm; lowered to 0 here (table top at the workspace floor) because at 1.5 cm and then 0.5 cm,
# fingers resting just above the real table read as in collision.
TABLE_THICKNESS_M = 0.02
TABLE_PADDING_M = 0.0
TABLE_NAME = "table"
# Measured joint angles can be slightly outside the URDF limits (as in DeoxysController).
JOINT_LIMIT_PADDING_RAD = 0.005
# Longest controller rollout simulated when checking a move before executing it.
PREDICT_MAX_TICKS = 300
# Agent moves take larger steps per control() call than the fork's default (osc-pose-precise-controller.yml, 1 cm /
# 0.05 rad, which teleop keeps). Each command leads the measured pose by at most one step, so the step also sets how
# hard the joint impedance controller pulls: at 1 cm, measured moves topped out around 5-6 cm/s.
ACTION_SCALE_MULTIPLIER = 2.0


# -- The model's action ----------------------------------------------------------------------------------------------

# No ge/le bounds on the numeric fields: the SDK strips them from the schema sent to the API and then validates
# client-side, so an over-limit step would raise instead of being clamped. Limits are stated in the prompt and
# enforced by clamp_delta.


class EEPoseDelta(BaseModel):
    """Relative end effector motion, expressed in the control frame chosen for the run (world or end effector)."""

    reasoning: str = Field(description="One or two sentences: what you see, and why this motion.")
    dx_cm: float = Field(description="Translation along x, in centimeters, in the action frame the prompt describes.")
    dy_cm: float = Field(description="Translation along y, in centimeters, in the action frame the prompt describes.")
    dz_cm: float = Field(description="Translation along z, in centimeters, in the action frame the prompt describes.")
    droll_deg: float = Field(description="Rotation about x, in degrees, in the action frame the prompt describes.")
    dpitch_deg: float = Field(description="Rotation about y, in degrees, in the action frame the prompt describes.")
    dyaw_deg: float = Field(description="Rotation about z, in degrees, in the action frame the prompt describes.")
    gripper: Literal["open", "closed"] = Field(description="Gripper command, held after this motion.")
    task_complete: bool = Field(description="True only if the task is already done in the current images.")

    @property
    def translation_m(self) -> np.ndarray:
        return np.array([self.dx_cm, self.dy_cm, self.dz_cm]) / 100.0

    @property
    def rotvec_rad(self) -> np.ndarray:
        """Rotation as a rotation vector. For the small steps used here this matches roll/pitch/yaw to first order,
        without depending on the order they are applied in."""
        return np.deg2rad([self.droll_deg, self.dpitch_deg, self.dyaw_deg])


# -- Pose math -------------------------------------------------------------------------------------------------------


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
    """(position error [3] m, rotation error as a base frame rotation vector [3] rad) from current to target."""
    return T_target[:3, 3] - T_current[:3, 3], Rotation.from_matrix(T_target[:3, :3] @ T_current[:3, :3].T).as_rotvec()


def pose_to_text(T_base_ee: np.ndarray) -> str:
    """Human readable world frame pose for the prompt: position in cm, orientation as roll / pitch / yaw in degrees
    (static-xyz euler angles: about world x, then y, then z)."""
    xyz_cm = T_base_ee[:3, 3] * 100.0
    rpy_deg = Rotation.from_matrix(T_base_ee[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"position x={xyz_cm[0]:.1f}, y={xyz_cm[1]:.1f}, z={xyz_cm[2]:.1f} cm; "
        f"roll / pitch / yaw = {rpy_deg[0]:.1f} / {rpy_deg[1]:.1f} / {rpy_deg[2]:.1f} deg"
    )


# -- Execution -------------------------------------------------------------------------------------------------------


class CollisionChecker:
    """Self collisions, and collisions with the table, for a jrl2 model of the arm (collision geometry)."""

    def __init__(self, robot_name: str, workspace_bounds: WorkspaceBounds):
        from jrl2.collision_detection_single_scene import SingleSceneCollisionChecker
        from jrl2.math_utils import get_translated_pose
        from jrl2.robots import get_robot_by_name
        from trimesh.primitives import Box

        self.robot = get_robot_by_name(robot_name)
        self._checker = SingleSceneCollisionChecker(self.robot, use_visual=False)
        wb = workspace_bounds
        center = np.array([wb.center[0], wb.center[1], wb.min_z_m + TABLE_PADDING_M - TABLE_THICKNESS_M / 2])
        table = Box(extents=[wb.extents[0], wb.extents[1], TABLE_THICKNESS_M], transform=get_translated_pose(center))
        self._table_geometry_name = self._checker.add_box(table)
        # The base is bolted to the table.
        for geometry in self.robot.collision_geometry_names:
            if geometry.startswith("panda_link0::"):
                self._checker.ignore_pair((self._table_geometry_name, geometry))
        joints = {joint.name: joint for joint in self.robot.actuated_joints}
        self._finger_limits = {
            name: (joints[name].limit.lower, joints[name].limit.upper)
            for name in self.robot.actuated_joint_names
            if name not in ARM_JOINT_NAMES
        }

    def _q_dict(self, q: np.ndarray, gripper_width_m: float | None) -> dict[str, float]:
        finger = 0.04 if gripper_width_m is None else gripper_width_m / 2.0
        q_dict = dict(zip(ARM_JOINT_NAMES, np.asarray(q, dtype=float)))
        q_dict.update({name: float(np.clip(finger, lo, hi)) for name, (lo, hi) in self._finger_limits.items()})
        return q_dict

    def colliding_pairs(self, q: np.ndarray, gripper_width_m: float | None) -> list[tuple[str, str]]:
        """Colliding geometry pairs at joint positions `q` [7], with the table named "table". Empty if none."""
        try:
            _, pairs, _ = self._checker.check_collisions(
                self._q_dict(q, gripper_width_m), q_range_padding=JOINT_LIMIT_PADDING_RAD
            )
        except AssertionError as exc:  # jrl2 asserts the configuration is within the joint limits
            return [("joint limits", str(exc))]
        rename = {self._table_geometry_name: TABLE_NAME}
        return sorted(tuple(rename.get(name, name) for name in pair) for pair in pairs)

    def fk_tcp(self, q: np.ndarray) -> np.ndarray:
        """Pose [4, 4] of panda_hand_tcp in the base frame."""
        return self.robot.get_all_link_poses_non_batched(self._q_dict(q, None))["panda_hand_tcp"]


@dataclass
class ServoResult:
    reached: bool
    position_error_m: float
    rotation_error_rad: float
    duration_s: float
    # Set when collision checking stopped the move: "current" (the arm is already in collision), "predicted" (the
    # simulated move collides, so it was not started) or "during" (the next control step would collide).
    blocked: str = ""
    collision_pairs: list[tuple[str, str]] = field(default_factory=list)


class EEPoseServo:
    """Closed-loop pose tracking on OSC_POSE_PRECISE: each tick re-measures the remaining error to the target and
    sends it as the action; deoxys limits each part to one action_scale per control() call.

    Every move is collision checked: it is first simulated with the controller's own IK step (perfect tracking from
    the current state) and not started if the arm is in collision now or anywhere along the simulated path. While
    moving, the joint target each command will produce is recomputed from the latest state and checked before it is
    sent. The timeout and the reported duration count from when the arm starts moving, after that check.

    The controller's action_scale (largest step per call) is multiplied by `action_scale_multiplier` for these moves.
    """

    def __init__(
        self,
        robot: FrankaInterface,
        collision_checker: CollisionChecker,
        position_tolerance_m: float = 0.005,
        rotation_tolerance_rad: float = np.deg2rad(1.0),
        timeout_s: float = 1.5,
        gripper_settle_s: float = 1.0,
        action_scale_multiplier: float = ACTION_SCALE_MULTIPLIER,
    ):
        from deoxys.franka_interface.osc_pose_precise import (
            PandaJacobian,
            osc_pose_precise_joint_target,
        )
        from deoxys.utils.config_utils import get_default_controller_config

        self._robot = robot
        # A fresh copy from the YAML, so this doesn't change the default for anyone else (e.g. teleop)
        self._cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
        self._cfg.action_scale.translation *= action_scale_multiplier
        self._cfg.action_scale.rotation *= action_scale_multiplier
        self._collision_checker = collision_checker
        self._kinematics = PandaJacobian()
        self._joint_target = osc_pose_precise_joint_target  # the fork's IK step, to predict what a command will do
        self._position_tolerance_m = position_tolerance_m
        self._rotation_tolerance_rad = rotation_tolerance_rad
        self._timeout_s = timeout_s
        self._gripper_settle_s = gripper_settle_s
        self._last_gripper: float | None = None

    def _action(self, translation_m: np.ndarray, rotvec_rad: np.ndarray, gripper: float) -> np.ndarray:
        return np.concatenate(
            [
                translation_m / self._cfg.action_scale.translation,
                rotvec_rad / self._cfg.action_scale.rotation,
                [gripper],
            ]
        )

    def _send(self, translation_m: np.ndarray, rotvec_rad: np.ndarray, gripper: float, termination: bool = False):
        self._robot.control(
            controller_type=CONTROLLER_TYPE,
            action=self._action(translation_m, rotvec_rad, gripper),
            controller_cfg=self._cfg,
            termination=termination,
        )

    def _within_tolerance(self, e_pos: np.ndarray, e_rot: np.ndarray) -> bool:
        return (
            np.linalg.norm(e_pos) < self._position_tolerance_m and np.linalg.norm(e_rot) < self._rotation_tolerance_rad
        )

    def predict_path(self, T_target: np.ndarray, q: np.ndarray, T_ee: np.ndarray) -> list[np.ndarray]:
        """Joint positions the controller would command, tick by tick, moving from (q, T_ee) to `T_target` if the arm
        tracked every command exactly."""
        # Measured O_T_EE vs the model's panda_hand_tcp (they differ by the end effector configured in Franka Desk)
        tcp_T_ee = np.linalg.inv(self._collision_checker.fk_tcp(q)) @ T_ee
        path = []
        for _ in range(PREDICT_MAX_TICKS):
            e_pos, e_rot = pose_error(T_target, T_ee)
            if self._within_tolerance(e_pos, e_rot):
                break
            q = self._joint_target(self._action(e_pos, e_rot, 0.0), q, T_ee, self._cfg, self._kinematics)
            T_ee = self._collision_checker.fk_tcp(q) @ tcp_T_ee
            path.append(q)
        return path

    def _blocked(self, T_target: np.ndarray, t0: float, blocked: str, pairs: list[tuple[str, str]]) -> ServoResult:
        e_pos, e_rot = pose_error(T_target, self._robot.last_eef_pose)
        return ServoResult(
            False, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot)), time.time() - t0, blocked, pairs
        )

    def move_to(
        self, T_target: np.ndarray, gripper: float, should_stop: Callable[[], bool] = lambda: False
    ) -> ServoResult:
        """Track `T_target` [4, 4] (base frame) until within tolerance, `should_stop()`, or the timeout. Then apply
        `gripper` and hold while it settles (not counted in the reported duration)."""
        reached = False
        # The gripper keeps its previous command during the motion; the new one is applied once there.
        gripper_during = self._last_gripper if self._last_gripper is not None else gripper
        checker = self._collision_checker
        q, T_ee, width = np.asarray(self._robot.last_q), np.asarray(self._robot.last_eef_pose), self._gripper_width()
        pairs = checker.colliding_pairs(q, width)
        if pairs:
            return self._blocked(T_target, time.time(), "current", pairs)
        for q_predicted in self.predict_path(T_target, q, T_ee):
            pairs = checker.colliding_pairs(q_predicted, width)
            if pairs:
                return self._blocked(T_target, time.time(), "predicted", pairs)

        t0 = time.time()
        while not should_stop():
            q, T_ee = np.asarray(self._robot.last_q), np.asarray(self._robot.last_eef_pose)
            e_pos, e_rot = pose_error(T_target, T_ee)
            if self._within_tolerance(e_pos, e_rot):
                reached = True
                break
            if time.time() - t0 > self._timeout_s:
                break
            q_next = self._joint_target(
                self._action(e_pos, e_rot, gripper_during), q, T_ee, self._cfg, self._kinematics
            )
            pairs = checker.colliding_pairs(q_next, self._gripper_width())
            if pairs:
                return self._blocked(T_target, t0, "during", pairs)
            self._send(e_pos, e_rot, gripper_during)
        duration_s = time.time() - t0

        if gripper != self._last_gripper and not should_stop():
            self.hold(gripper, self._gripper_settle_s, should_stop)
        self._last_gripper = gripper

        e_pos, e_rot = pose_error(T_target, self._robot.last_eef_pose)
        return ServoResult(reached, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot)), duration_s)

    def _gripper_width(self) -> float | None:
        width = self._robot.last_gripper_q
        return None if width is None else float(width)

    def hold(self, gripper: float, duration_s: float, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Hold the current pose while commanding `gripper`. The first command after a controller switch takes ~1 s,
        so this also serves as a warm start."""
        t_end = time.time() + duration_s
        while time.time() < t_end and not should_stop():
            self._send(np.zeros(3), np.zeros(3), gripper)
        self._last_gripper = gripper

    def terminate(self) -> None:
        """Send deoxys' termination command so the NUC-side controller stops tracking the last goal."""
        gripper = self._last_gripper if self._last_gripper is not None else GRIPPER_OPEN
        self._send(np.zeros(3), np.zeros(3), gripper, termination=True)
