"""Standalone Franka manipulation script (no ROS dependency).

This is the base motion layer for the data collection pipeline: connect to deoxys,
verify the robot is actually streaming state, and run guarded joint / Cartesian moves.

Deliberately ROS-free. `panda_utils.utils` imports `rospy` and `panda_utils.deoxys_controller`
imports `tf.transformations`, both ROS1-only, and this machine has ROS2 Humble. The few
constants needed are duplicated below rather than importing that chain.

Control path (nothing here is sent over ssh):
    this PC  --ZMQ 5555/5556 (arm), 5557/5558 (gripper)-->  NUC
    NUC      --libfranka/FCI 1 kHz-->                       robot control box
ssh is only used to *launch* bin/franka-interface + bin/gripper-interface on the NUC.

Usage:
    python scripts/franka_manipulation.py --check          # preflight only, never commands motion
    python scripts/franka_manipulation.py --demo home
    python scripts/franka_manipulation.py --demo square
    python scripts/franka_manipulation.py --demo pick_place
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
from termcolor import cprint

from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.yaml_config import YamlConfig

PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_INTERFACE_CFG = PROJECT_DIR / "configs" / "charmander.yml"

# Mirrors panda_utils.deoxys_controller.RESET_JOINT_POSITIONS.
RESET_JOINT_POSITIONS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

# Official Panda joint limits (rad). Mirrors panda_utils.constants.PANDA_LIMITS.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# Refuse configurations this close to a hard limit; libfranka will fault before the limit anyway.
Q_LIMIT_MARGIN_RAD = 0.05

# Mirrors panda_utils.constants.WORKSPACE_BOUNDS (metres, robot base frame).
WS_MIN = np.array([-0.21, -0.3825, -0.01])
WS_MAX = np.array([0.99, 0.4275, 0.75])

# deoxys gripper convention: -1 fully open, +1 fully closed.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# osc-pose-controller.yml: action_scale.translation = 0.05, so |action|=1 -> 5 cm target offset.
OSC_TRANSLATION_SCALE_M = 0.05
CONTROL_HZ = 20.0  # matches CONTROL.POLICY_RATE in charmander.yml


class RobotNotReadyError(RuntimeError):
    """Raised when deoxys is not streaming robot state."""


def _tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def preflight(interface_cfg: Path, verbose: bool = True, require_gripper: bool = True) -> bool:
    """Check config and NUC reachability before opening a control connection.

    Returns True if everything needed to command the robot looks present. Never commands motion.

    Set require_gripper False on a rig with no gripper hardware; the gripper port is then
    reported but not treated as a failure.
    """
    ok = True

    if not interface_cfg.is_file():
        cprint(f"[FAIL] interface config not found: {interface_cfg}", "red")
        return False
    cfg = YamlConfig(str(interface_cfg)).as_easydict()

    nuc_ip = cfg["NUC"]["IP"]
    robot_ip = cfg["ROBOT"]["IP"]
    if verbose:
        cprint(f"[info] config      : {interface_cfg}", "cyan")
        cprint(f"[info] NUC         : {nuc_ip}", "cyan")
        cprint(f"[info] robot (FCI) : {robot_ip}", "cyan")

    # The NUC runs franka-interface; without it nothing downstream works.
    if _tcp_open(nuc_ip, 22):
        cprint(f"[ OK ] NUC {nuc_ip} reachable (ssh open)", "green")
    else:
        cprint(f"[FAIL] NUC {nuc_ip} not reachable on port 22.", "red")
        cprint("       Fix NUC.IP in the config, or power on / re-cable the NUC.", "red")
        ok = False

    # Deoxys publishes robot state from the NUC on PUB_PORT. If franka-interface is not
    # running, the port is closed even when the NUC itself is up.
    pub_port = cfg["NUC"]["PUB_PORT"]
    if _tcp_open(nuc_ip, pub_port):
        cprint(f"[ OK ] deoxys arm state port {nuc_ip}:{pub_port} open", "green")
    else:
        cprint(f"[WARN] deoxys arm state port {nuc_ip}:{pub_port} closed.", "yellow")
        cprint("       Start franka-interface on the NUC (see README_SETUP.md).", "yellow")
        ok = False

    # The gripper runs as a separate process on the NUC. wait_until_ready() blocks on
    # last_gripper_q, so a missing gripper-interface stalls every motion call - check it
    # here rather than letting the script time out with a vaguer error.
    gripper_port = cfg["NUC"]["GRIPPER_PUB_PORT"]
    if not require_gripper:
        cprint(f"[skip] gripper state port {nuc_ip}:{gripper_port} not checked (--no-gripper)",
               "cyan")
    elif _tcp_open(nuc_ip, gripper_port):
        cprint(f"[ OK ] deoxys gripper state port {nuc_ip}:{gripper_port} open", "green")
    else:
        cprint(f"[WARN] deoxys gripper state port {nuc_ip}:{gripper_port} closed.", "yellow")
        cprint("       Start gripper-interface on the NUC (auto_scripts/auto_gripper.sh),", "yellow")
        cprint("       or pass --no-gripper if this rig has no gripper hardware.", "yellow")
        ok = False

    return ok


class PandaArm:
    """Guarded wrapper around deoxys FrankaInterface.

    Every motion call validates robot state first and clamps against joint/workspace limits,
    so a bad target is refused in Python rather than faulting libfranka on the NUC.
    """

    def __init__(self, interface_cfg: Path = DEFAULT_INTERFACE_CFG, control_hz: float = CONTROL_HZ,
                 has_gripper: bool = True):
        self._cfg_path = interface_cfg
        self._dt = 1.0 / control_hz
        self._stop_requested = False
        self._has_gripper = has_gripper
        self.robot = FrankaInterface(str(interface_cfg), use_visualizer=False)
        self._osc_cfg = get_default_controller_config(controller_type="OSC_POSE")
        self._joint_cfg = get_default_controller_config(controller_type="JOINT_POSITION")

    # -- lifecycle ---------------------------------------------------------

    def wait_until_ready(self, timeout_s: float = 10.0) -> None:
        """Block until deoxys reports arm state (and gripper state, if a gripper is fitted).

        ROS-free equivalent of panda_utils.utils.wait_for_deoxys_ready. With has_gripper=False
        only arm state is required - last_gripper_q stays None forever with no gripper-interface
        running, so waiting on it would always time out.
        """
        deadline = time.time() + timeout_s
        warned = False
        while time.time() < deadline:
            arm_ready = self.robot.last_q is not None
            gripper_ready = (not self._has_gripper) or self.robot.last_gripper_q is not None
            if arm_ready and gripper_ready:
                cprint(f"[ OK ] deoxys streaming state. q={np.round(self.robot.last_q, 3)}", "green")
                if not self._has_gripper:
                    cprint("[info] gripper disabled (--no-gripper)", "cyan")
                return
            if not warned:
                cprint("[info] waiting for robot state from the NUC...", "cyan")
                warned = True
            time.sleep(0.1)

        needed = "arm" if not self._has_gripper else "arm and gripper"
        raise RobotNotReadyError(
            f"No {needed} state after {timeout_s:.0f}s "
            f"(last_q={self.robot.last_q}, last_gripper_q={self.robot.last_gripper_q}).\n"
            "Checklist:\n"
            "  1. Joints unlocked in Franka Desk and the E-stop released.\n"
            "  2. FCI activated in Desk.\n"
            "  3. bin/franka-interface running on the NUC"
            f"{'' if not self._has_gripper else ' (and bin/gripper-interface)'}.\n"
            "  4. NUC.IP in the interface config matches the real NUC."
        )

    def request_stop(self, *_args) -> None:
        """SIGINT handler: finish the current step, then unwind cleanly."""
        cprint("\n[info] stop requested, finishing current step...", "yellow")
        self._stop_requested = True

    @property
    def stopped(self) -> bool:
        return self._stop_requested

    def shutdown(self) -> None:
        """Send a zero-action termination command and close the connection.

        Without the terminating command the NUC-side controller keeps running its last
        target, so this matters even on the error path.
        """
        try:
            self.robot.control(
                controller_type="OSC_POSE",
                action=[0.0] * 6 + [GRIPPER_OPEN],
                controller_cfg=self._osc_cfg,
                termination=True,
            )
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask the original error
            cprint(f"[WARN] termination command failed: {exc}", "yellow")
        finally:
            self.robot.close()
            cprint("[ OK ] robot interface closed", "green")

    # -- safety ------------------------------------------------------------

    @staticmethod
    def validate_joints(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"expected 7 joint values, got shape {q.shape}")
        lo, hi = Q_MIN + Q_LIMIT_MARGIN_RAD, Q_MAX - Q_LIMIT_MARGIN_RAD
        bad = np.where((q < lo) | (q > hi))[0]
        if bad.size:
            details = ", ".join(f"j{i + 1}={q[i]:.3f} not in [{lo[i]:.3f}, {hi[i]:.3f}]" for i in bad)
            raise ValueError(f"joint target outside safe limits: {details}")
        return q

    @staticmethod
    def in_workspace(xyz: np.ndarray) -> bool:
        return bool(np.all(xyz >= WS_MIN) and np.all(xyz <= WS_MAX))

    def ee_position(self) -> np.ndarray:
        """Current end-effector position (m) in the robot base frame."""
        return np.asarray(self.robot.last_eef_pose)[:3, 3]

    # -- motion primitives -------------------------------------------------

    def home(self, gripper_open: bool = True) -> None:
        """Move to the shared reset configuration used by the rest of panda_utils."""
        cprint("[info] homing...", "cyan")
        reset_joints_to(self.robot, RESET_JOINT_POSITIONS, gripper_open=gripper_open)
        cprint(f"[ OK ] home reached. q={np.round(self.robot.last_q, 3)}", "green")

    def move_to_joints(self, q_target, gripper: float = GRIPPER_OPEN, timeout_s: float = 7.0,
                       tolerance_rad: float = 1e-2) -> bool:
        """Servo to an absolute joint configuration. Returns True if the target was reached."""
        q_target = self.validate_joints(q_target)
        action = q_target.tolist() + [gripper]
        deadline = time.time() + timeout_s

        while time.time() < deadline and not self._stop_requested:
            self.robot.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=self._joint_cfg,
            )
            if np.max(np.abs(self.robot.last_q - q_target)) < tolerance_rad:
                return True
            time.sleep(self._dt)

        if self._stop_requested:
            return False
        cprint(f"[WARN] joint target not reached within {timeout_s:.1f}s "
               f"(max err {np.max(np.abs(self.robot.last_q - q_target)):.4f} rad)", "yellow")
        return False

    def move_delta(self, dxyz, drot=(0.0, 0.0, 0.0), gripper: float = GRIPPER_OPEN,
                   steps: int = 1) -> None:
        """Apply a Cartesian delta in the base frame using OSC_POSE.

        Args:
            dxyz: (3,) translation in metres, applied per step.
            drot: (3,) axis-angle rotation delta, in normalised OSC units.
            gripper: GRIPPER_OPEN or GRIPPER_CLOSE.
            steps: number of control cycles to hold this delta.

        The OSC action is normalised: 1.0 corresponds to OSC_TRANSLATION_SCALE_M metres.
        """
        dxyz = np.asarray(dxyz, dtype=float)
        drot = np.asarray(drot, dtype=float)
        if dxyz.shape != (3,) or drot.shape != (3,):
            raise ValueError(f"dxyz and drot must be shape (3,), got {dxyz.shape} and {drot.shape}")

        action_xyz = np.clip(dxyz / OSC_TRANSLATION_SCALE_M, -1.0, 1.0)
        action_rot = np.clip(drot, -1.0, 1.0)
        action = action_xyz.tolist() + action_rot.tolist() + [gripper]

        for _ in range(steps):
            if self._stop_requested:
                return
            # Predict where this step sends us and refuse to leave the workspace.
            predicted = self.ee_position() + dxyz
            if not self.in_workspace(predicted):
                cprint(f"[WARN] step would exit workspace at {np.round(predicted, 3)}, skipping", "yellow")
                return
            self.robot.control(
                controller_type="OSC_POSE",
                action=action,
                controller_cfg=self._osc_cfg,
            )
            time.sleep(self._dt)

    def move_to_ee_position(self, target_xyz, tolerance_m: float = 0.01,
                            timeout_s: float = 15.0, max_step_m: float = 0.02,
                            gripper: float = GRIPPER_OPEN) -> bool:
        """Proportionally servo the end effector to a target position, orientation held.

        Returns True if the target was reached within tolerance.
        """
        target_xyz = np.asarray(target_xyz, dtype=float)
        if target_xyz.shape != (3,):
            raise ValueError(f"target_xyz must be shape (3,), got {target_xyz.shape}")
        if not self.in_workspace(target_xyz):
            raise ValueError(
                f"target {np.round(target_xyz, 3)} outside workspace bounds "
                f"{np.round(WS_MIN, 3)}..{np.round(WS_MAX, 3)}"
            )

        deadline = time.time() + timeout_s
        while time.time() < deadline and not self._stop_requested:
            error = target_xyz - self.ee_position()
            distance = float(np.linalg.norm(error))
            if distance < tolerance_m:
                cprint(f"[ OK ] reached {np.round(target_xyz, 3)} (err {distance * 1000:.1f} mm)", "green")
                return True
            # Cap per-step motion so a large error cannot produce a lurch.
            step = error if distance <= max_step_m else error / distance * max_step_m
            self.move_delta(step, gripper=gripper, steps=1)

        if self._stop_requested:
            return False
        cprint(f"[WARN] EE target not reached within {timeout_s:.1f}s", "yellow")
        return False

    def set_gripper(self, closed: bool, settle_s: float = 1.0) -> None:
        """Open or close the gripper and hold pose while it settles. No-op with no gripper."""
        if not self._has_gripper:
            cprint("[skip] gripper command ignored (--no-gripper)", "cyan")
            return
        action = GRIPPER_CLOSE if closed else GRIPPER_OPEN
        cprint(f"[info] gripper {'closing' if closed else 'opening'}...", "cyan")
        steps = max(1, int(settle_s / self._dt))
        for _ in range(steps):
            if self._stop_requested:
                return
            self.robot.control(
                controller_type="OSC_POSE",
                action=[0.0] * 6 + [action],
                controller_cfg=self._osc_cfg,
            )
            time.sleep(self._dt)


# -- demos -----------------------------------------------------------------


def demo_home(arm: PandaArm) -> None:
    arm.home()


def demo_square(arm: PandaArm, side_m: float = 0.10) -> None:
    """Trace a square in the XY plane. Good smoke test for Cartesian control."""
    arm.home()
    start = arm.ee_position()
    cprint(f"[info] tracing {side_m * 100:.0f} cm square from {np.round(start, 3)}", "cyan")

    corners = [
        start + np.array([side_m, 0.0, 0.0]),
        start + np.array([side_m, side_m, 0.0]),
        start + np.array([0.0, side_m, 0.0]),
        start,
    ]
    for i, corner in enumerate(corners, 1):
        if arm.stopped:
            return
        cprint(f"[info] corner {i}/4 -> {np.round(corner, 3)}", "cyan")
        arm.move_to_ee_position(corner)


def demo_pick_place(arm: PandaArm, pick_xy=(0.45, -0.10), place_xy=(0.45, 0.10),
                    grasp_z: float = 0.05, hover_z: float = 0.20) -> None:
    """Hover -> descend -> grasp -> lift -> translate -> release -> retreat.

    Coordinates are placeholders; measure the real object pose before running this
    on hardware. Verify with --demo square first.
    """
    if not arm._has_gripper:
        cprint("[FAIL] pick_place needs a gripper. Use --demo home or --demo square.", "red")
        return
    arm.home()
    pick = np.array([pick_xy[0], pick_xy[1], grasp_z])
    place = np.array([place_xy[0], place_xy[1], grasp_z])
    pick_hover = np.array([pick_xy[0], pick_xy[1], hover_z])
    place_hover = np.array([place_xy[0], place_xy[1], hover_z])

    stages = [
        ("approach pick", lambda: arm.move_to_ee_position(pick_hover)),
        ("open gripper", lambda: arm.set_gripper(closed=False)),
        ("descend", lambda: arm.move_to_ee_position(pick, max_step_m=0.01)),
        ("grasp", lambda: arm.set_gripper(closed=True)),
        ("lift", lambda: arm.move_to_ee_position(pick_hover, gripper=GRIPPER_CLOSE, max_step_m=0.01)),
        ("transfer", lambda: arm.move_to_ee_position(place_hover, gripper=GRIPPER_CLOSE)),
        ("lower", lambda: arm.move_to_ee_position(place, gripper=GRIPPER_CLOSE, max_step_m=0.01)),
        ("release", lambda: arm.set_gripper(closed=False)),
        ("retreat", lambda: arm.move_to_ee_position(place_hover)),
    ]
    for name, action in stages:
        if arm.stopped:
            cprint("[info] stopped mid-sequence", "yellow")
            return
        cprint(f"[stage] {name}", "magenta")
        action()

    arm.home()


DEMOS = {"home": demo_home, "square": demo_square, "pick_place": demo_pick_place}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface-cfg", type=Path, default=DEFAULT_INTERFACE_CFG,
                        help="deoxys interface config (default: configs/charmander.yml)")
    parser.add_argument("--check", action="store_true",
                        help="run preflight checks only, never command motion")
    parser.add_argument("--demo", choices=sorted(DEMOS), default="home",
                        help="motion sequence to run (default: home)")
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="connect without checking NUC reachability first")
    parser.add_argument("--no-gripper", action="store_true",
                        help="rig has no gripper hardware: skip the gripper port check, "
                             "do not wait for gripper state, ignore gripper commands")
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    preflight_ok = preflight(args.interface_cfg, require_gripper=not args.no_gripper)
    if args.check:
        cprint("\npreflight: " + ("PASS" if preflight_ok else "FAIL"),
               "green" if preflight_ok else "red")
        return 0 if preflight_ok else 1
    if not preflight_ok and not args.skip_preflight:
        cprint("\nPreflight failed; refusing to command the robot. "
               "Re-run with --skip-preflight to override.", "red")
        return 1

    arm = PandaArm(args.interface_cfg, control_hz=args.control_hz,
                   has_gripper=not args.no_gripper)
    signal.signal(signal.SIGINT, arm.request_stop)

    try:
        arm.wait_until_ready()
        cprint(f"[info] running demo: {args.demo}", "magenta")
        DEMOS[args.demo](arm)
        cprint("[ OK ] demo complete", "green")
        return 0
    except RobotNotReadyError as exc:
        cprint(f"\n[FAIL] {exc}", "red")
        return 1
    except (ValueError, RuntimeError) as exc:
        cprint(f"\n[FAIL] {exc}", "red")
        return 1
    finally:
        arm.shutdown()


if __name__ == "__main__":
    sys.exit(main())
