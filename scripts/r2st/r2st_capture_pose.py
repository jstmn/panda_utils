#!/usr/bin/env python3
"""Drive the arm with the SpaceMouse and capture joint configurations to paste into a script.

RESET_JOINT_POSITIONS is inlined in half a dozen scripts here -- rm_bc_inference.py,
log_demonstration.py, act_clip_inference.py, franka_manipulation.py -- and every one of them
carries a different hand-tuned list with older ones commented out above it. This is the tool that
produces those numbers: teleop to a pose you like, press ENTER, and it prints the joint vector
formatted the way those files already write it, along with where the end effector ended up.

Nothing is written anywhere unless --out is given. The arm only moves while you push the
SpaceMouse.

Usage:
    source /opt/ros/humble/setup.bash
    source ~/anaconda3/bin/activate rmimic
    cd ~/Desktop/panda_utils

    python scripts/r2st/r2st_capture_pose.py

    # capture without teleop -- for hand-guiding the arm by its own free-drive button
    python scripts/r2st/r2st_capture_pose.py --no-teleop

    # keep every capture
    python scripts/r2st/r2st_capture_pose.py --out ~/Desktop/data/reset_poses.txt

ENTER captures, 'r' returns to the pose the script started from, Ctrl-C exits and reprints
everything captured.
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np
from termcolor import cprint

from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config

np.set_printoptions(precision=8, suppress=True, linewidth=200)

# panda_utils.constants needs open3d, which the rmimic env does not have; rm_bc_inference.py
# inlines RESET_JOINT_POSITIONS for the same reason.
SPACE_MOUSE_VENDOR_ID = 1133
SPACE_MOUSE_PRODUCT_ID = 50726
CONTROLLER_TYPE = "OSC_POSE"
RATE_HZ = 50.0


def wait_for_deoxys_ready(robot: FrankaInterface, timeout_s: float = 10.0) -> bool:
    cprint("⏳ Waiting for deoxys state ...", "cyan")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if robot.last_q is not None and robot.last_gripper_q is not None:
            cprint("✅ Robot interface ready", "green")
            return True
        time.sleep(0.1)
    cprint("❌ No joint/gripper state from deoxys -- are the NUC binaries running?", "red")
    return False


def key_pressed() -> str | None:
    """Non-blocking single key, as rm_bc_inference.enter_is_pressed does it."""
    if select.select([sys.stdin], [], [], 0.0)[0] == [sys.stdin]:
        return sys.stdin.read(1)
    return None


def format_capture(q: np.ndarray, gripper: float, eef) -> str:
    body = ", ".join(f"{v: .8f}" for v in q)
    out = ["RESET_JOINT_POSITIONS = [", f"    {body},", "]"]
    if eef is not None:
        pos, rot = eef
        rpy = np.degrees([np.arctan2(rot[2, 1], rot[2, 2]),
                          -np.arcsin(np.clip(rot[2, 0], -1, 1)),
                          np.arctan2(rot[1, 0], rot[0, 0])])
        out.append(f"# eef pos {np.round(pos, 4)}  rpy_deg {np.round(rpy, 1)}  "
                   f"gripper {gripper:.4f}")
    return "\n".join(out)


def eef_pose(robot: FrankaInterface):
    T = robot.last_eef_pose
    if T is None:
        return None
    T = np.asarray(T)
    return T[:3, 3].astype(float), T[:3, :3].astype(float)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-teleop", action="store_true",
                   help="do not drive the arm; just read poses, for hand-guiding it instead")
    p.add_argument("--out", type=str, default=None,
                   help="append captures here as well as printing them")
    p.add_argument("--interface-cfg", type=str, default=None)
    p.add_argument("--vendor-id", type=int, default=SPACE_MOUSE_VENDOR_ID)
    p.add_argument("--product-id", type=int, default=SPACE_MOUSE_PRODUCT_ID)
    cli = p.parse_args()

    cfg = cli.interface_cfg or str(Path(__file__).resolve().parents[1] / "configs" / "charmander.yml")
    robot = FrankaInterface(cfg, use_visualizer=False)
    if not wait_for_deoxys_ready(robot):
        robot.close()
        return 1

    home_q = np.asarray(robot.last_q).copy()
    device = controller_cfg = None
    if not cli.no_teleop:
        from deoxys.utils.input_utils import input2action
        from deoxys.utils.io_devices import SpaceMouse
        try:
            device = SpaceMouse(vendor_id=cli.vendor_id, product_id=cli.product_id)
            device.start_control()
        except Exception as exc:  # noqa: BLE001
            cprint(f"❌ SpaceMouse {cli.vendor_id}/{cli.product_id}: {exc!r}", "red")
            cprint("   --no-teleop captures poses without it.", "yellow")
            robot.close()
            return 1
        controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
        robot._state_buffer = []

    cprint("\nENTER = capture    r = back to the starting pose    Ctrl-C = done", "yellow")
    cprint(f"teleop: {'OFF (read only)' if cli.no_teleop else 'ON -- the arm moves with the SpaceMouse'}",
           "red" if not cli.no_teleop else "white")
    cprint(f"start q: {home_q}", "white")

    captures: list[str] = []
    try:
        while True:
            if not cli.no_teleop:
                action, _ = input2action(device=device, controller_type=CONTROLLER_TYPE)
                if action is None:          # SpaceMouse reset button quits input2action
                    break
                robot.control(controller_type=CONTROLLER_TYPE, action=action,
                              controller_cfg=controller_cfg)
            k = key_pressed()
            if k == "\n":
                q = robot.last_q
                g = robot.last_gripper_q
                if q is None:
                    cprint("no joint state right now -- try again", "yellow")
                else:
                    text = format_capture(np.asarray(q), float(g or 0.0), eef_pose(robot))
                    captures.append(text)
                    cprint(f"\n--- capture {len(captures)} ---", "green")
                    print(text)
                    if cli.out:
                        with open(cli.out, "a") as f:
                            f.write(text + "\n\n")
            elif k == "r":
                cprint("returning to the starting pose ...", "yellow")
                reset_joints_to(robot, home_q.tolist(), gripper_open=True)
            time.sleep(1.0 / RATE_HZ)
    except KeyboardInterrupt:
        cprint("\n⏹️  done", "yellow")
    finally:
        if not cli.no_teleop and controller_cfg is not None:
            robot.control(controller_type=CONTROLLER_TYPE,
                          action=[0.0] * 6 + [1.0], controller_cfg=controller_cfg,
                          termination=True)
        robot.close()

    if captures:
        cprint(f"\n=== {len(captures)} capture(s) ===", "green")
        for i, c in enumerate(captures, 1):
            cprint(f"--- {i} ---", "cyan")
            print(c)
        if cli.out:
            cprint(f"\nappended to {cli.out}", "green")
        cprint("\nPaste one into the script that needs it -- rm_bc_inference.py:96 for inference, "
               "log_demonstration.py:89 for recording. They do not share a copy.", "white")
    else:
        cprint("nothing captured", "yellow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
