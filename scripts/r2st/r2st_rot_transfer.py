#!/usr/bin/env python3
"""Measure what a commanded OSC delta actually does to the wrist.

Two measurements, both open-loop, because a policy rollout cannot make either of them. A
14.4 Hz diffusion policy commands 0.001-0.005 rad per step, and three rollouts recorded on
2026-09-13 with the translation transfer curve off (Kp.rotation 250 and 500, time_fraction 0.3
and 0.6) all froze identically -- 0.5 cm of travel, 0.4 deg of rotation, eef position std the
same to two decimals in every run. Nothing can be told apart at that amplitude. So command a
size you choose and read back what the arm did, the way rm_bc_inference.py's translation table
(_TRANSFER_ACHIEVED_MM) was built.

  --mode rot    command a rotation about one hand axis, measure the rotation it produces.
                Builds the rotation transfer table.

  --mode trans  command a pure translation offset along one base axis with the rotation
                command at zero, measure the rotation it produces anyway. Builds the coupling
                table: the arm pivots about joint 6, 23 cm behind the fingertips, so a position
                goal turns the hand whether or not anything asked it to. Measured on a rollout,
                this coupling costs -0.0007 rad/step of wrist pitch -- -23 deg over a run, which
                is most of what the policy's commanded lift is fighting.

Each trial returns to the reset pose first, so trials do not accumulate, and amplitude 0.0 is
worth leaving in the sweep as a drift control: whatever it reports is the noise floor for
everything else.

Usage:
    source /opt/ros/humble/setup.bash
    source ~/anaconda3/bin/activate rmimic
    cd ~/Desktop/panda_utils

    # look first: prints the plan, does not move
    python scripts/r2st/r2st_rot_transfer.py

    # the rotation table, dense where the policy actually operates, 3 repeats each
    python scripts/r2st/r2st_rot_transfer.py --execute --axis y --repeats 3
    python scripts/r2st/r2st_rot_transfer.py --execute --axis x --repeats 3
    python scripts/r2st/r2st_rot_transfer.py --execute --axis z --repeats 3

    # the coupling table: how much rotation each translation axis drags in
    python scripts/r2st/r2st_rot_transfer.py --execute --mode trans --axis z --repeats 3

    # does a controller knob move either curve?
    python scripts/r2st/r2st_rot_transfer.py --execute --axis y \
        --controller-cfg configs/osc-pose-tuned.yml --interp-time-fraction 0.6
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from termcolor import cprint

from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config

np.set_printoptions(precision=5, suppress=True, linewidth=200)

CONTROLLER_TYPE = "OSC_POSE"
# Same pose rm_bc_inference.py starts from: OSC authority is configuration dependent, so the
# measurement has to be taken where the policy runs.
RESET_JOINT_POSITIONS = [
    0.09162008, -0.19826458, -0.01990020, -2.47322699, -0.01307074, 2.30396583, 0.84809397,
]
AXES = {"x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),   # what joint 6 turns about: FK gives [0.063, 0.998, 0]
        "z": np.array([0.0, 0.0, 1.0])}
# Dense across the range the policy's commands land in once inflated, plus a drift control at 0.
DEFAULT_ROT_AMPS = [0.0, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
# The points the translation table was built at, so the coupling table lines up with it.
DEFAULT_TRANS_AMPS = [0.0, 0.005, 0.010, 0.020, 0.040]
WS_MIN = np.array([0.20, -0.40, -0.05])
WS_MAX = np.array([0.80, 0.40, 0.60])


class Rate:
    """rm_bc_inference.Rate -- rospy.Rate replacement."""

    def __init__(self, hz: float):
        self._period = 1.0 / float(hz)
        self._next = time.perf_counter() + self._period

    def sleep(self) -> None:
        now = time.perf_counter()
        delay = self._next - now
        if delay > 0.0:
            time.sleep(delay)
            self._next += self._period
        else:
            self._next = now + self._period


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


def eef_pose(robot: FrankaInterface):
    T = robot.last_eef_pose
    if T is None:
        return None
    T = np.asarray(T)
    return T[:3, 3].astype(float), T[:3, :3].astype(float)


def axis_angle(R: np.ndarray) -> np.ndarray:
    tr = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(tr))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2.0 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def run_trial(robot, controller_cfg, scales, mode, axis, amp, steps, rate_hz, execute,
              settle=0):
    """One amplitude, once. Returns per-step achieved rotation (gripper frame) and eef drift.

    `settle` steps are commanded but dropped from the average. reset_joints_to returns before the
    arm has fully stopped, and in a 10-step trial that tail is a fifth of the measurement: the
    2026-09-13 coupling sweep read -0.00021 rad/step at zero amplitude where the longer rotation
    sweeps read -0.00001, which is settling, not coupling.
    """
    tr_scale, rot_scale = scales
    rate = Rate(rate_hz)
    start = eef_pose(robot)
    if start is None:
        raise RuntimeError("no eef pose from deoxys")
    p0, R0 = start
    prev_R, per_step, left_ws = R0, [], False
    for k in range(steps + settle):
        cur = eef_pose(robot)
        if cur is None:
            rate.sleep()
            continue
        p, R = cur
        if np.any(p < WS_MIN) or np.any(p > WS_MAX):
            left_ws = True
            break
        if mode == "rot":
            # Gripper-frame axis-angle -> base frame, exactly as rm_bc_inference's control loop.
            dpos = np.zeros(3)
            drot = R @ (AXES[axis] * amp)
        else:
            dpos = AXES[axis] * amp          # base-frame position goal offset
            drot = np.zeros(3)               # nothing asks the hand to turn
        if execute:
            cmd = np.concatenate([dpos / tr_scale, drot / rot_scale, [-1.0]])
            robot.control(controller_type=CONTROLLER_TYPE, action=cmd,
                          controller_cfg=controller_cfg)
        if k >= settle:
            per_step.append(axis_angle(prev_R.T @ R))
        prev_R = R
        rate.sleep()
        if k == settle - 1:                      # measurement starts here
            p0, R0 = eef_pose(robot) or (p0, R0)
    end = eef_pose(robot)
    p1, R1 = end if end is not None else (p0, R0)
    per_step = np.array(per_step) if per_step else np.zeros((1, 3))
    return dict(per_step=per_step.mean(0), net=axis_angle(R0.T @ R1),
                drift_mm=float(np.linalg.norm(p1 - p0) * 1000.0), left_ws=left_ws)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true",
                   help="actually send the commands; without it nothing moves and the plan is "
                        "printed instead")
    p.add_argument("--mode", choices=("rot", "trans"), default="rot",
                   help="rot: command rotation, measure rotation. trans: command translation "
                        "with zero rotation, measure the rotation it drags in anyway")
    p.add_argument("--axis", choices=("x", "y", "z"), default="y",
                   help="hand-frame rotation axis in rot mode (y is the wrist pitch joint 6 "
                        "turns about), base-frame translation axis in trans mode")
    p.add_argument("--amplitudes", type=float, nargs="*", default=None,
                   help="rad/step in rot mode, metres of goal offset in trans mode "
                        f"(defaults: {DEFAULT_ROT_AMPS} / {DEFAULT_TRANS_AMPS})")
    p.add_argument("--repeats", type=int, default=1,
                   help="times to repeat each amplitude; the spread across repeats is the only "
                        "honest error bar this measurement has")
    p.add_argument("--steps", type=int, default=None,
                   help="steps per trial (default 20 in rot mode, 10 in trans mode, where the "
                        "arm actually travels)")
    p.add_argument("--settle-steps", "--settle_steps", type=int, default=0,
                   help="steps to command but drop before measuring, so the tail of the reset "
                        "move does not land in the average (try 5 for short trials)")
    p.add_argument("--rate", type=float, default=14.4, help="control rate, matching the policy")
    p.add_argument("--interface-cfg", type=str, default=None)
    p.add_argument("--controller-cfg", "--controller_cfg", type=str, default=None,
                   help="OSC yaml to use instead of the deoxys default")
    p.add_argument("--kp-rotation", "--kp_rotation", type=float, default=None)
    p.add_argument("--kp-translation", "--kp_translation", type=float, default=None)
    p.add_argument("--interp-time-fraction", "--interp_time_fraction", type=float, default=None)
    p.add_argument("--max-rot-amp", type=float, default=0.15,
                   help="refuse rotation amplitudes above this, rad/step (default 0.15 ~ 8.6 deg)")
    p.add_argument("--max-trans-amp", type=float, default=0.06,
                   help="refuse translation amplitudes above this, m (default 0.06, the "
                        "--max-offset rm_bc_inference caps its own goals at)")
    p.add_argument("--json-out", type=str, default=None,
                   help="write the measured table here, for building the curve from")
    cli = p.parse_args()

    amps = cli.amplitudes if cli.amplitudes is not None else (
        DEFAULT_ROT_AMPS if cli.mode == "rot" else DEFAULT_TRANS_AMPS)
    steps = cli.steps if cli.steps is not None else (20 if cli.mode == "rot" else 10)
    cap = cli.max_rot_amp if cli.mode == "rot" else cli.max_trans_amp
    unit = "rad/step" if cli.mode == "rot" else "m"
    over = [a for a in amps if abs(a) > cap]
    if over:
        cprint(f"❌ amplitudes {over} exceed the {cli.mode} cap of {cap} {unit}", "red")
        return 1

    if cli.controller_cfg:
        from deoxys.utils.yaml_config import YamlConfig
        controller_cfg = YamlConfig(cli.controller_cfg).as_easydict()
        if controller_cfg.get("controller_type") != CONTROLLER_TYPE:
            cprint(f"❌ {cli.controller_cfg} is a {controller_cfg.get('controller_type')} config",
                   "red")
            return 1
    else:
        controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
    for val, path in ((cli.kp_rotation, ("Kp", "rotation")),
                      (cli.kp_translation, ("Kp", "translation")),
                      (cli.interp_time_fraction, ("traj_interpolator_cfg", "time_fraction"))):
        if val is not None:
            controller_cfg[path[0]][path[1]] = val

    cprint(f"mode {cli.mode}   axis {cli.axis}   {len(amps)} amplitudes x {cli.repeats} repeats "
           f"x {steps} steps @ {cli.rate} Hz", "white")
    cprint(f"Kp translation {controller_cfg.Kp.translation}  rotation {controller_cfg.Kp.rotation}"
           f"   time_fraction {controller_cfg.traj_interpolator_cfg.time_fraction}", "white")
    cprint(f"amplitudes ({unit}): {amps}", "white")
    if not cli.execute:
        cprint("DRY RUN - pass --execute to move the arm", "yellow")

    cfg = cli.interface_cfg or str(
        (__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "charmander.yml"))
    robot = FrankaInterface(cfg, use_visualizer=False)
    if not wait_for_deoxys_ready(robot):
        robot.close()
        return 1

    scales = (float(controller_cfg.action_scale.translation),
              float(controller_cfg.action_scale.rotation))
    rows = []
    try:
        for amp in amps:
            trials = []
            for r in range(cli.repeats):
                if cli.execute:
                    cprint(f"\nreset, then {amp} {unit} on {cli.axis} for {steps} steps "
                           f"({r+1}/{cli.repeats})", "yellow")
                    reset_joints_to(robot, RESET_JOINT_POSITIONS, gripper_open=True)
                    time.sleep(0.5)
                t = run_trial(robot, controller_cfg, scales, cli.mode, cli.axis, amp, steps,
                              cli.rate, cli.execute, cli.settle_steps)
                if t["left_ws"]:
                    cprint("  left the workspace box -- trial cut short", "red")
                trials.append(t)
            per = np.array([t["per_step"] for t in trials])
            net = np.array([t["net"] for t in trials])
            row = dict(amp=amp,
                       per_step_mean=per.mean(0).tolist(), per_step_std=per.std(0).tolist(),
                       net_deg_mean=np.degrees(net.mean(0)).tolist(),
                       drift_mm=float(np.mean([t["drift_mm"] for t in trials])))
            rows.append(row)
            m, s = per.mean(0), per.std(0)
            cprint(f"  {amp:.4f} {unit} -> achieved rotation/step "
                   f"x {m[0]:+.5f}±{s[0]:.5f}  y {m[1]:+.5f}±{s[1]:.5f}  z {m[2]:+.5f}±{s[2]:.5f}"
                   f"   eef drift {row['drift_mm']:.1f} mm", "cyan")
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    finally:
        if cli.execute:
            reset_joints_to(robot, RESET_JOINT_POSITIONS, gripper_open=True)
        robot.close()

    if rows:
        cprint(f"\n=== {cli.mode} sweep, axis {cli.axis} ===", "green")
        cprint(f"{'commanded':>12s} {'achieved rot/step on ' + cli.axis:>26s} {'ratio':>8s} "
               f"{'net deg':>9s} {'drift mm':>9s}", "white")
        i = "xyz".index(cli.axis)
        for r in rows:
            a = r["per_step_mean"][i]
            ratio = f"{a/r['amp']*100:7.1f}%" if r["amp"] else "       -"
            cprint(f"{r['amp']:12.4f} {a:+16.5f}±{r['per_step_std'][i]:.5f} {ratio} "
                   f"{r['net_deg_mean'][i]:9.2f} {r['drift_mm']:9.1f}", "white")
        if cli.mode == "trans":
            cprint("\nAll three rotation axes, since a position goal does not pick one:", "white")
            for r in rows:
                cprint(f"  {r['amp']:.4f} m -> {np.array(r['per_step_mean'])} rad/step", "white")
    if cli.json_out and rows:
        with open(cli.json_out, "w") as f:
            json.dump(dict(mode=cli.mode, axis=cli.axis, rate=cli.rate, steps=steps,
                           repeats=cli.repeats, settle_steps=cli.settle_steps,
                           kp_translation=controller_cfg.Kp.translation,
                           kp_rotation=controller_cfg.Kp.rotation,
                           time_fraction=controller_cfg.traj_interpolator_cfg.time_fraction,
                           rows=rows), f, indent=1)
        cprint(f"wrote {cli.json_out}", "green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
