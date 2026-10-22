"""Agent control of the Panda: a Claude or OpenAI model looks at the cameras and picks end effector pose deltas.

Each step: grab the latest frame from every camera, send the last --history_length observations and the task
description to the model, get back one structured EEPoseDelta, clamp it, and track that pose with deoxys'
OSC_POSE_PRECISE (our fork: per-tick IK steps through JOINT_IMPEDANCE). Deltas are in the robot base frame
(--action-reference-frame world) or the gripper's own frame (--action-reference-frame ee). Steps execute without confirmation unless --require-confirmation-before-action is
passed, in which case each one waits for ENTER before the robot moves.

A browser viewer (viser, --viser-port) shows the cameras, the observation history and request sent to the model, its
response, and the robot with its current and desired end effector poses. Its reset button homes the arm and clears the
observation history, stopping the current step first.

Safety: steps are capped at --max_translation_cm / --max_rotation_deg and targets are clamped to
panda_utils.constants.WORKSPACE_BOUNDS (with z >= --min_ee_height_cm). There is NO collision checking against
objects or the arm itself. Keep a hand on the E-stop.

Cameras must already be running (bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh), and deoxys on the NUC (see
README_SETUP.md). Claude models (claude-*) use ANTHROPIC_API_KEY (or an `ant auth login` profile); OpenAI models
(gpt-*, o<N>*) require OPENAI_API_KEY to be set in the shell environment.

### Usage:
python scripts/agent_control.py \
    --task-description "pickup the red apple and place it in the grey bowl" \
    --model dream-1 \
    --action-reference-frame ee \
    --camera_ids zed2i zedm \
    --effort medium \
    --history_length 3

    --model gpt-6-astra \
    --task-description "grasp the red cube and raise it 15 cm above the table" \
    --camera_ids zed zed2i zedm \
    --camera_ids zedm zed zed2i \
    --model gpt-6.1-sol \
    --action-reference-frame world \
    --camera_ids zed zedm \

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
import openai
import rclpy
from deoxys.franka_interface import FrankaInterface
from termcolor import cprint

from panda_utils.agent_control.actions import (
    CONTROL_FRAMES,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    TABLE_PADDING_M,
    CollisionChecker,
    EEPoseServo,
    clamp_delta,
    delta_to_target,
)
from panda_utils.agent_control.agent import (
    AgentRefusalError,
    Observation,
    PoseDeltaAgent,
    format_cost,
)
from panda_utils.agent_control.video import RunVideoRecorder
from panda_utils.agent_control.viewer import (
    DEFAULT_VIEWER_ROBOT,
    VIEWER_ROBOTS,
    AgentViewer,
)
from panda_utils.constants import RESET_JOINT_POSITIONS, WORKSPACE_BOUNDS
from panda_utils.ros2_cameras import CAMERA_LAYOUTS, CameraSubscriber
from panda_utils.utils import (
    RobotNotMovingError,
    reset_joints_verified,
    wait_for_deoxys_ready,
)

PROJECT_DIR = Path(__file__).parent.parent
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Let the cameras deliver frames taken after the arm stopped, before the next observation.
OBSERVATION_SETTLE_S = 0.2
# How long to wait at startup for every camera's first frame and camera_info.
CAMERA_TIMEOUT_S = 5.0


def take_observation(cams: CameraSubscriber, robot: FrankaInterface) -> Observation:
    T_base_ee = np.array(robot.last_eef_pose)
    images = {cam_id: cams.latest_rgb(cam_id) for cam_id in cams.camera_ids}
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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="claude-* (Anthropic) or gpt-* / o<N>* (OpenAI, needs OPENAI_API_KEY), e.g. claude-opus-5-5",
    )
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
        choices=sorted(CAMERA_LAYOUTS),
        help="cameras as launched",
    )
    parser.add_argument("--task-description", type=str, required=True, help="task description given to the model")
    parser.add_argument(
        "--action-reference-frame",
        type=str,
        choices=CONTROL_FRAMES,
        default="world",
        help="frame the deltas are expressed in: robot base ('world') or gripper ('ee')",
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=EFFORT_LEVELS,
        default=None,
        help="output_config.effort; omitted by default, so the model's own default applies",
    )
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--max_translation_cm", type=float, default=10.0, help="per-step translation cap")
    parser.add_argument("--max_rotation_deg", type=float, default=20.0, help="per-step rotation cap")
    parser.add_argument(
        "--min_ee_height_cm", type=float, default=1.0, help="lowest end effector z allowed, in the base frame"
    )
    parser.add_argument(
        "--require-confirmation-before-action",
        action="store_true",
        help="wait for ENTER (or s = skip, q = quit) before executing each step",
    )
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_DIR / "logs" / "agent_control"))
    parser.add_argument("--interface-cfg", type=str, default=str(PROJECT_DIR / "configs" / "charmander.yml"))
    parser.add_argument("--no-reset", action="store_true", help="start from the current pose instead of homing")
    parser.add_argument("--viser-port", type=int, default=8080, help="port for the browser viewer")
    parser.add_argument(
        "--robot-model",
        type=str,
        choices=VIEWER_ROBOTS,
        default=DEFAULT_VIEWER_ROBOT,
        help="jrl2 model of the arm: drawn in the viewer, and used for self and table collision checking",
    )
    args = parser.parse_args()

    if args.history_length < 1:
        parser.error("--history_length must be >= 1")
    camera_layouts = {cam_id: CAMERA_LAYOUTS[cam_id] for cam_id in args.camera_ids}

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

    try:
        agent = PoseDeltaAgent(
            model=args.model,
            task=args.task_description,
            frame=args.action_reference_frame,
            history_length=args.history_length,
            max_translation_cm=args.max_translation_cm,
            max_rotation_deg=args.max_rotation_deg,
            effort=args.effort,
        )
    except ValueError as exc:
        parser.error(str(exc))

    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.task_description).strip("-")[:40] or "task"
    run_dir = Path(args.output_dir).expanduser() / f"{slug}__{datetime.now().strftime('%m-%d-%Y__%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({**vars(args), "system_prompt": agent.system_prompt}, indent=2))

    cprint("\n" + "=" * 60, "cyan")
    cprint("🤖 Agent control", "cyan", attrs=["bold"])
    cprint(f"Model    : {args.model}" + (f" (effort {args.effort})" if args.effort else ""), "white")
    cprint(f"Task     : {args.task_description}", "white")
    cprint(f"Frame    : {args.action_reference_frame}   history: {args.history_length}", "white")
    cprint(f"Cameras  : {list(camera_layouts)}", "white")
    cprint(
        f"Limits   : {args.max_translation_cm:g} cm, {args.max_rotation_deg:g} deg per step; "
        f"workspace {np.round(ws_lo, 3)} .. {np.round(ws_hi, 3)} m",
        "white",
    )
    cprint(f"Logs     : {run_dir}", "white")
    confirm = args.require_confirmation_before_action
    cprint(
        f"Mode     : {'confirm each step' if confirm else 'steps execute without confirmation'}",
        "white" if confirm else "yellow",
    )
    viewer = AgentViewer(
        list(camera_layouts),
        args.history_length,
        agent.system_prompt,
        port=args.viser_port,
        robot_name=args.robot_model,
    )
    cprint(f"Viewer   : {viewer.url}", "white")
    collision_checker = CollisionChecker(args.robot_model, WORKSPACE_BOUNDS)
    cprint(
        f"Collision: self + table ({args.robot_model}); table top at z = "
        f"{100 * (WORKSPACE_BOUNDS.min_z_m + TABLE_PADDING_M):.1f} cm",
        "white",
    )
    cprint("=" * 60 + "\n", "cyan")

    rclpy.init(args=None)
    cams = CameraSubscriber(camera_layouts)
    running = True

    def spin_target() -> None:
        while running and rclpy.ok():
            rclpy.spin_once(cams, timeout_sec=0.05)

    threading.Thread(target=spin_target, daemon=True).start()

    robot = FrankaInterface(args.interface_cfg, use_visualizer=False)

    def robot_state() -> tuple[np.ndarray, float | None, np.ndarray] | None:
        q, T_base_ee, gripper = robot.last_q, robot.last_eef_pose, robot.last_gripper_q
        if q is None or T_base_ee is None:
            return None
        return np.asarray(q), None if gripper is None else float(gripper), np.asarray(T_base_ee)

    viewer.start_live(robot_state, cams.latest_rgb)
    servo = None
    recorder = None
    exit_code = 0
    run_cost_usd = 0.0  # estimated; steps with an unpriced model add nothing
    n_actions_executed = 0  # steps where the arm moved
    status = "stopped"
    try:
        if not wait_for_deoxys_ready(robot):
            cprint("❌ Deoxys is not streaming robot state. Is franka-interface running on the NUC?", "red")
            return 1
        missing = cams.wait_for_frames(timeout_s=CAMERA_TIMEOUT_S)
        if missing:
            cprint(
                f"❌ No frames / camera_info from {missing} within {CAMERA_TIMEOUT_S:g} s. Are the cameras up?", "red"
            )
            return 1
        cprint("✅ Robot and cameras ready", "green")

        if not args.no_reset:
            cprint("🔄 Homing...", "yellow")
            viewer.set_status("homing")
            reset_joints_verified(robot, RESET_JOINT_POSITIONS, gripper_open=True)
        servo = EEPoseServo(robot, collision_checker=collision_checker)
        # The first command after a controller switch takes ~1 s; take that hit here rather than mid-step.
        servo.hold(GRIPPER_OPEN if (robot.last_gripper_q or 0.0) > 0.04 else GRIPPER_CLOSED, 1.0)

        def reset_episode() -> None:
            """Home the arm and start the model over with an empty observation history (viewer reset button)."""
            cprint("\n🔄 Reset requested from the viewer: homing and clearing the observation history", "yellow")
            viewer.set_status("resetting: homing")
            reset_joints_verified(robot, RESET_JOINT_POSITIONS, gripper_open=True)
            servo.hold(GRIPPER_OPEN, 1.0)  # warm start the joint impedance controller again after the homing move
            agent.history.clear()
            viewer.finish_reset()

        reset_before_step = False
        for step in range(args.max_steps):
            if viewer.reset_requested.is_set():
                reset_episode()
                reset_before_step = True
            time.sleep(OBSERVATION_SETTLE_S)
            obs = take_observation(cams, robot)
            step_json = run_dir / f"step_{step:03d}.json"
            for cam_id, image in obs.images.items():
                cv2.imwrite(str(run_dir / f"step_{step:03d}__{cam_id}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            cprint(f"\n── step {step} ── asking {args.model}...", "cyan")
            viewer.set_status(f"step {step}: waiting for {args.model}")
            t0 = time.time()
            if recorder is None:
                # The run clock (video overlay, viewer charts) and the video start at the first API call, after homing.
                viewer.start_clock(t0)
                recorder = RunVideoRecorder(run_dir / "run.mp4", cams.camera_ids, cams.latest_rgb, t0=t0)
                recorder.start()
                cprint(f"🎥 Recording {recorder.path}", "white")
            recorder.set_step(step)
            reply = agent.act(obs, on_request=viewer.show_request)
            action = reply.action
            latency_s = time.time() - t0
            if reply.cost_usd is not None:
                run_cost_usd += reply.cost_usd
            viewer.record_cost(run_cost_usd)
            recorder.set_cost(run_cost_usd)
            cprint(f"   {action.reasoning}", "white")
            cprint(
                f"   d=({action.dx_cm:+.1f}, {action.dy_cm:+.1f}, {action.dz_cm:+.1f}) cm  "
                f"rot=({action.droll_deg:+.1f}, {action.dpitch_deg:+.1f}, {action.dyaw_deg:+.1f}) deg  "
                f"gripper={action.gripper}  done={action.task_complete}   "
                f"[{latency_s:.1f} s, {reply.input_tokens} in / {reply.output_tokens} out, "
                f"{format_cost(reply.cost_usd)}; run ${run_cost_usd:.4f}]",
                "magenta",
            )

            record = {
                "step": step,
                "model": reply.model,
                "request_id": reply.request_id,
                "latency_s": latency_s,
                "usage": reply.usage,
                "cost_usd": reply.cost_usd,
                "run_cost_usd": run_cost_usd,
                "action": action.model_dump(),
                "T_base_ee": obs.T_base_ee.tolist(),
                "gripper_width_m": obs.gripper_width_m,
                "reset_before_step": reset_before_step,
            }
            reset_before_step = False

            if viewer.reset_requested.is_set():
                # This answer was planned from the pre-reset scene; don't act on it.
                cprint("   Reset requested while waiting for the model: discarding this step.", "yellow")
                step_json.write_text(json.dumps({**record, "reset_requested": True}, indent=2))
                continue

            if action.task_complete:
                cprint("✅ Model reports the task complete.", "green")
                viewer.show_response(step, reply, latency_s, None, [], None, run_cost_usd)
                status = "task reported complete"
                step_json.write_text(json.dumps(record, indent=2))
                break

            translation_m, rotvec_rad = clamp_delta(
                action.translation_m, action.rotvec_rad, max_translation_m, max_rotation_rad
            )
            T_target = delta_to_target(obs.T_base_ee, translation_m, rotvec_rad, args.action_reference_frame)
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
            viewer.show_response(step, reply, latency_s, obs.action, notes, T_target, run_cost_usd)

            if confirm:
                viewer.set_status(f"step {step}: waiting for ENTER in the terminal")
            choice = ask_operator() if confirm else ""
            if choice == "q":
                status = "quit by operator"
                step_json.write_text(json.dumps({**record, "operator": "quit"}, indent=2))
                break
            if viewer.reset_requested.is_set():
                record["discarded"] = "reset requested"
                viewer.show_outcome("Not executed: reset requested.")
            elif choice == "s":
                notes.append("The operator skipped this step; the robot did not move.")
                record["operator"] = "skipped"
                viewer.show_outcome("Skipped by the operator.")
            else:
                viewer.set_status(f"step {step}: executing")
                result = servo.move_to(
                    T_target,
                    GRIPPER_CLOSED if action.gripper == "closed" else GRIPPER_OPEN,
                    should_stop=viewer.reset_requested.is_set,
                )
                record["servo"] = vars(result)
                if result.blocked not in ("current", "predicted"):  # those were refused before the arm moved
                    n_actions_executed += 1
                    viewer.record_actions(n_actions_executed)
                pairs_text = ", ".join(f"{a} / {b}" for a, b in result.collision_pairs)
                cprint(
                    f"   ⏱  {'reached the target' if result.reached else 'stopped'} after {result.duration_s:.2f} s, "
                    f"{1000 * result.position_error_m:.1f} mm / {np.rad2deg(result.rotation_error_rad):.1f} deg "
                    "from it",
                    "white",
                )
                if viewer.reset_requested.is_set():
                    record["interrupted"] = "reset requested"
                    viewer.show_outcome("Motion stopped: reset requested.")
                elif result.blocked:
                    notes.append(
                        {
                            "current": f"Not executed: the arm is already in collision ({pairs_text}).",
                            "predicted": f"Not executed: this motion would collide ({pairs_text}); the robot did not move.",
                            "during": f"Motion stopped part way: the next step would have collided ({pairs_text}).",
                        }[result.blocked]
                    )
                    cprint(f"   ⛔ {notes[-1]}", "red")
                    viewer.show_outcome(notes[-1])
                elif not result.reached:
                    notes.append(
                        f"The arm stopped {result.position_error_m * 100:.1f} cm / "
                        f"{np.rad2deg(result.rotation_error_rad):.1f} deg short of the target."
                    )
                    cprint(f"   ⚠️  {notes[-1]}", "yellow")
                    viewer.show_outcome(notes[-1])
                else:
                    viewer.show_outcome(f"Reached the target in {result.duration_s:.1f} s.")
            obs.outcome = " ".join(notes)
            step_json.write_text(json.dumps(record, indent=2))
        else:
            cprint(f"\n⏹️  Reached --max_steps ({args.max_steps}) without the model reporting completion.", "yellow")
            status = f"reached --max_steps ({args.max_steps})"
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
        status = "interrupted"
    except RobotNotMovingError as exc:
        cprint(f"❌ {exc}", "red")
        status = f"robot not moving: {exc}"
        exit_code = 1
    except AgentRefusalError as exc:
        cprint(f"❌ {exc}", "red")
        status = f"error: {exc}"
        exit_code = 1
    except (anthropic.APIError, openai.APIError) as exc:
        # The SDK has already retried rate limits, 5xx and connection errors by the time this is raised.
        cprint(f"❌ {agent.provider} API error ({type(exc).__name__}): {exc}", "red")
        status = f"{agent.provider} API error: {exc}"
        exit_code = 1
    finally:
        running = False
        if recorder is not None:
            recorder.stop()  # before the camera node goes away; finalizes the file on every exit path
            cprint(f"🎥 Video saved to {recorder.path}", "white")
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
        cprint(f"💰 Estimated model cost this run: ${run_cost_usd:.4f}", "white")
        viewer.disable_reset()
        viewer.set_status(f"{status}. Robot stopped.")
        if status != "interrupted":
            try:
                input(f"Viewer still up at {viewer.url}. Press ENTER to close it.")
            except (KeyboardInterrupt, EOFError):
                pass
        viewer.stop()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
