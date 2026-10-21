"""Agent control of the Panda: a Claude model looks at the cameras and picks end effector pose deltas.

Each step: grab the latest frame from every camera (optionally with the end effector frame drawn on it), send the
last --history_length observations and the task to the model, get back one structured EEPoseDelta, clamp it, and
servo the arm there with deoxys OSC_POSE. Deltas are in the robot base frame (--frame world) or the gripper's own
frame (--frame ee). By default every step waits for ENTER before the robot moves; --auto skips that.

Safety: steps are capped at --max_translation_cm / --max_rotation_deg and targets are clamped to
panda_utils.constants.WORKSPACE_BOUNDS (with z >= --min_ee_height_cm). There is NO collision checking against
objects or the arm itself. Keep a hand on the E-stop.

Cameras must already be running (bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh), and deoxys on the NUC (see
README_SETUP.md). The Anthropic client reads ANTHROPIC_API_KEY (or an `ant auth login` profile).

# Usage:

python scripts/agent_control.py --model claude-opus-5-5 --task "pick up the red cube" --camera_ids zed zedm

# EE frame control, 3 observations of history, EE frame drawn on the calibrated static camera
python scripts/agent_control.py --model claude-opus-5-5 --task "push the peg upright" --camera_ids zed2i zedm \
    --frame ee --history_length 3 --draw_ee_frame --extrinsics zed2i=~/Desktop/data/calib/zed2i.yaml

--extrinsics files use the r2st_handeye.py schema: a 4x4 `matrix` with `parent_frame: robot_base` (static camera,
T_base_cam) or `parent_frame: end_effector` (wrist camera, T_ee_cam), in the camera's optical frame. Cameras
without one are sent undrawn.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import anthropic
import cv2
import numpy as np
import rclpy
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from termcolor import cprint

from panda_utils.agent_control.agent import (
    AgentRefusalError,
    Observation,
    PoseDeltaAgent,
)
from panda_utils.agent_control.geometry import (
    CONTROL_FRAMES,
    T_cam_ee,
    clamp_delta,
    delta_to_target,
    draw_frame,
    load_extrinsics,
)
from panda_utils.agent_control.servo import GRIPPER_CLOSED, GRIPPER_OPEN, EEPoseServo
from panda_utils.constants import RESET_JOINT_POSITIONS, WORKSPACE_BOUNDS
from panda_utils.ros2_cameras import CameraSubscriber, parse_camera_spec
from panda_utils.utils import wait_for_deoxys_ready

PROJECT_DIR = Path(__file__).parent.parent
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Let the cameras deliver frames taken after the arm stopped, before the next observation.
OBSERVATION_SETTLE_S = 0.3


def parse_extrinsics(specs: list[str]) -> dict:
    out = {}
    for spec in specs:
        cam_id, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"--extrinsics expects <camera>=<path>, got {spec!r}")
        out[cam_id] = load_extrinsics(path)
    return out


def take_observation(cams: CameraSubscriber, robot: FrankaInterface, extrinsics: dict, draw: bool) -> Observation:
    T_base_ee = np.array(robot.last_eef_pose)
    images = {}
    for cam_id in cams.camera_ids:
        image = cams.latest_rgb(cam_id)
        if draw and cam_id in extrinsics:
            image = draw_frame(image, T_cam_ee(extrinsics[cam_id], T_base_ee), cams.intrinsics(cam_id))
        images[cam_id] = image
    gripper = robot.last_gripper_q
    return Observation(images=images, T_base_ee=T_base_ee, gripper_width_m=None if gripper is None else float(gripper))


def ask_operator() -> str:
    """'' to execute, 's' to skip this step, 'q' to quit."""
    while True:
        reply = input("   ENTER = execute, s = skip, q = quit > ").strip().lower()
        if reply in ("", "s", "q"):
            return reply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Claude model id, e.g. claude-opus-5-5")
    parser.add_argument(
        "--history_length",
        type=int,
        default=1,
        help="observations sent per call, including the current one (default 1)",
    )
    parser.add_argument(
        "--camera_ids",
        type=str,
        nargs="+",
        required=True,
        help="cameras as launched, optionally '<name>=<layout>' (zed | realsense), e.g. zedm d435=realsense",
    )
    parser.add_argument("--task", type=str, required=True, help="task description given to the model")
    parser.add_argument(
        "--frame",
        type=str,
        choices=CONTROL_FRAMES,
        default="world",
        help="frame the deltas are expressed in: robot base ('world') or gripper ('ee')",
    )
    parser.add_argument(
        "--draw_ee_frame",
        action="store_true",
        help="draw the end effector frame on images from cameras with --extrinsics",
    )
    parser.add_argument("--extrinsics", type=str, nargs="*", default=[], help="<camera>=<extrinsics yaml>, ...")
    parser.add_argument(
        "--effort",
        type=str,
        choices=EFFORT_LEVELS,
        default=None,
        help="output_config.effort; omitted by default, so the model's own default applies",
    )
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--max_translation_cm", type=float, default=5.0, help="per-step translation cap")
    parser.add_argument("--max_rotation_deg", type=float, default=15.0, help="per-step rotation cap")
    parser.add_argument(
        "--min_ee_height_cm", type=float, default=1.0, help="lowest end effector z allowed, in the base frame"
    )
    parser.add_argument("--auto", action="store_true", help="execute steps without waiting for ENTER")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_DIR / "logs" / "agent_control"))
    parser.add_argument("--interface-cfg", type=str, default=str(PROJECT_DIR / "configs" / "charmander.yml"))
    parser.add_argument("--no-reset", action="store_true", help="start from the current pose instead of homing")
    parser.add_argument(
        "--qos-best-effort", action="store_true", help="use BEST_EFFORT QoS if topics exist but no frames arrive"
    )
    args = parser.parse_args()

    if args.history_length < 1:
        parser.error("--history_length must be >= 1")
    try:
        camera_layouts = dict(parse_camera_spec(spec) for spec in args.camera_ids)
        extrinsics = parse_extrinsics(args.extrinsics)
    except (ValueError, AssertionError, OSError) as exc:
        parser.error(str(exc))
    unknown = set(extrinsics) - set(camera_layouts)
    if unknown:
        parser.error(f"--extrinsics given for cameras not in --camera_ids: {sorted(unknown)}")
    if args.draw_ee_frame:
        undrawn = [c for c in camera_layouts if c not in extrinsics]
        if len(undrawn) == len(camera_layouts):
            parser.error("--draw_ee_frame needs --extrinsics for at least one camera")
        if undrawn:
            cprint(f"⚠️  No extrinsics for {undrawn}: their images are sent without the EE frame drawn.", "yellow")

    ws_lo = np.array(
        [
            WORKSPACE_BOUNDS.min_x_m,
            WORKSPACE_BOUNDS.min_y_m,
            max(WORKSPACE_BOUNDS.min_z_m, args.min_ee_height_cm / 100.0),
        ]
    )
    ws_hi = np.array([WORKSPACE_BOUNDS.max_x_m, WORKSPACE_BOUNDS.max_y_m, WORKSPACE_BOUNDS.max_z_m])
    max_translation_m = args.max_translation_cm / 100.0
    max_rotation_rad = np.deg2rad(args.max_rotation_deg)

    agent = PoseDeltaAgent(
        model=args.model,
        task=args.task,
        frame=args.frame,
        history_length=args.history_length,
        max_translation_cm=args.max_translation_cm,
        max_rotation_deg=args.max_rotation_deg,
        draw_ee_frame=args.draw_ee_frame,
        effort=args.effort,
    )

    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.task).strip("-")[:40] or "task"
    run_dir = Path(args.output_dir).expanduser() / f"{slug}__{datetime.now().strftime('%m-%d-%Y__%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({**vars(args), "system_prompt": agent.system_prompt}, indent=2))

    cprint("\n" + "=" * 60, "cyan")
    cprint("🤖 Agent control", "cyan", attrs=["bold"])
    cprint(f"Model    : {args.model}" + (f" (effort {args.effort})" if args.effort else ""), "white")
    cprint(f"Task     : {args.task}", "white")
    cprint(f"Frame    : {args.frame}   history: {args.history_length}   draw EE frame: {args.draw_ee_frame}", "white")
    cprint(f"Cameras  : {list(camera_layouts)}", "white")
    cprint(
        f"Limits   : {args.max_translation_cm:g} cm, {args.max_rotation_deg:g} deg per step; "
        f"workspace {np.round(ws_lo, 3)} .. {np.round(ws_hi, 3)} m",
        "white",
    )
    cprint(f"Logs     : {run_dir}", "white")
    cprint(
        f"Mode     : {'AUTO - steps execute without confirmation' if args.auto else 'confirm each step'}",
        "yellow" if args.auto else "white",
    )
    cprint("=" * 60 + "\n", "cyan")

    rclpy.init(args=None)
    cams = CameraSubscriber(camera_layouts, best_effort=args.qos_best_effort)
    running = True

    def spin_target() -> None:
        while running and rclpy.ok():
            rclpy.spin_once(cams, timeout_sec=0.05)

    threading.Thread(target=spin_target, daemon=True).start()

    robot = FrankaInterface(args.interface_cfg, use_visualizer=False)
    servo = None
    exit_code = 0
    try:
        if not wait_for_deoxys_ready(robot):
            cprint("❌ Deoxys is not streaming robot state. Is franka-interface running on the NUC?", "red")
            return 1
        missing = cams.wait_for_frames()
        if missing:
            cprint(
                f"❌ No frames / camera_info from {missing}. Check the cameras are up, or try --qos-best-effort.", "red"
            )
            return 1
        cprint("✅ Robot and cameras ready", "green")

        if not args.no_reset:
            cprint("🔄 Homing...", "yellow")
            reset_joints_to(robot, RESET_JOINT_POSITIONS, gripper_open=True)
        servo = EEPoseServo(robot)
        # The first command after a controller switch takes ~1 s; take that hit here rather than mid-step.
        servo.hold(GRIPPER_OPEN if (robot.last_gripper_q or 0.0) > 0.04 else GRIPPER_CLOSED, 1.0)

        for step in range(args.max_steps):
            time.sleep(OBSERVATION_SETTLE_S)
            obs = take_observation(cams, robot, extrinsics, args.draw_ee_frame)
            step_dir = run_dir / f"step_{step:03d}"
            step_dir.mkdir()
            for cam_id, image in obs.images.items():
                cv2.imwrite(str(step_dir / f"{cam_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            cprint(f"\n── step {step} ── asking {args.model}...", "cyan")
            t0 = time.time()
            action, response = agent.act(obs)
            latency_s = time.time() - t0
            usage = response.usage
            cprint(f"   {action.reasoning}", "white")
            cprint(
                f"   d=({action.dx_cm:+.1f}, {action.dy_cm:+.1f}, {action.dz_cm:+.1f}) cm  "
                f"rot=({action.droll_deg:+.1f}, {action.dpitch_deg:+.1f}, {action.dyaw_deg:+.1f}) deg  "
                f"gripper={action.gripper}  done={action.task_complete}   "
                f"[{latency_s:.1f} s, {usage.input_tokens} in / {usage.output_tokens} out]",
                "magenta",
            )

            record = {
                "step": step,
                "model": response.model,
                "request_id": response._request_id,
                "latency_s": latency_s,
                "usage": usage.model_dump(),
                "action": action.model_dump(),
                "T_base_ee": obs.T_base_ee.tolist(),
                "gripper_width_m": obs.gripper_width_m,
            }

            if action.task_complete:
                cprint("✅ Model reports the task complete.", "green")
                (step_dir / "step.json").write_text(json.dumps(record, indent=2))
                break

            translation_m, rotvec_rad = clamp_delta(
                action.translation_m, action.rotvec_rad, max_translation_m, max_rotation_rad
            )
            T_target = delta_to_target(obs.T_base_ee, translation_m, rotvec_rad, args.frame)
            unclamped_xyz = T_target[:3, 3].copy()
            T_target[:3, 3] = np.clip(unclamped_xyz, ws_lo, ws_hi)
            notes = []
            if not (np.allclose(translation_m, action.translation_m) and np.allclose(rotvec_rad, action.rotvec_rad)):
                notes.append("The step was scaled down to the per-step limit.")
            if not np.allclose(unclamped_xyz, T_target[:3, 3]):
                notes.append("The target was clamped to the workspace boundary.")
            obs.action = action.model_copy(
                update={
                    "dx_cm": translation_m[0] * 100,
                    "dy_cm": translation_m[1] * 100,
                    "dz_cm": translation_m[2] * 100,
                    "droll_deg": np.rad2deg(rotvec_rad[0]),
                    "dpitch_deg": np.rad2deg(rotvec_rad[1]),
                    "dyaw_deg": np.rad2deg(rotvec_rad[2]),
                }
            )
            for note in notes:
                cprint(f"   ⚠️  {note}", "yellow")
            record["executed"] = obs.action.model_dump()
            record["T_target"] = T_target.tolist()

            reply = "" if args.auto else ask_operator()
            if reply == "q":
                (step_dir / "step.json").write_text(json.dumps({**record, "operator": "quit"}, indent=2))
                break
            if reply == "s":
                notes.append("The operator skipped this step; the robot did not move.")
                record["operator"] = "skipped"
            else:
                result = servo.move_to(T_target, GRIPPER_CLOSED if action.gripper == "closed" else GRIPPER_OPEN)
                record["servo"] = vars(result)
                if not result.reached:
                    notes.append(
                        f"The arm stopped {result.position_error_m * 100:.1f} cm / "
                        f"{np.rad2deg(result.rotation_error_rad):.1f} deg short of the target."
                    )
                    cprint(f"   ⚠️  {notes[-1]}", "yellow")
            obs.outcome = " ".join(notes)
            (step_dir / "step.json").write_text(json.dumps(record, indent=2))
        else:
            cprint(f"\n⏹️  Reached --max_steps ({args.max_steps}) without the model reporting completion.", "yellow")
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    except AgentRefusalError as exc:
        cprint(f"❌ {exc}", "red")
        exit_code = 1
    except anthropic.APIError as exc:
        # The SDK has already retried rate limits, 5xx and connection errors by the time this is raised.
        cprint(f"❌ Claude API error ({type(exc).__name__}): {exc}", "red")
        exit_code = 1
    finally:
        running = False
        if servo is not None:
            try:
                servo.terminate()
            except Exception as exc:  # noqa: BLE001 - never mask the original failure during cleanup
                cprint(f"termination command failed: {exc}", "yellow")
        try:
            robot.close()
        except Exception:  # noqa: BLE001, S110 - never mask the original failure during cleanup
            pass
        cams.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cprint(f"📁 Logs in {run_dir}", "white")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
