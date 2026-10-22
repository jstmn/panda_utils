"""Browser viewer for agent control, built on viser.

Three standalone panels (movable, dockable windows) plus the 3D scene:
    Cameras   live feeds, and the observation history exactly as sent to the model
    Prompt    the text of the latest request (images shown as placeholders) and the system prompt
    Response  the model's reasoning and action, what was executed after clamping, and how execution went
    3D scene  the Panda at its live joint state, the world (robot base) frame, the current end effector pose, and the
              desired next end effector pose (the model's step after clamping, i.e. what the arm is sent to)
Main panel: status, and a reset button that homes the robot and clears the observation history.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import numpy as np
import viser
import viser.transforms as vtf
from jrl2.robots import NAME_TO_ROBOT, get_robot_by_name
from viser.extras import ViserUrdf

from panda_utils.agent_control.actions import EEPoseDelta, pose_to_text
from panda_utils.agent_control.agent import ModelReply, Observation, format_cost

ROBOT_UPDATE_HZ = 20.0
LIVE_IMAGE_EVERY = 4  # live camera images update every Nth robot update, i.e. 5 Hz
ROBOT_COLOR_RGBA = (0.7, 0.72, 0.75, 0.6)
CURRENT_EE_ORIGIN_COLOR = (236, 236, 0)
DESIRED_EE_ORIGIN_COLOR = (220, 0, 220)
BLANK_IMAGE = np.zeros((376, 672, 3), dtype=np.uint8)
# Jrl2 models of this arm. The live joint angles come from the Franka, so only Panda variants make sense.
VIEWER_ROBOTS = sorted(name for name in NAME_TO_ROBOT if name.startswith("panda"))
DEFAULT_VIEWER_ROBOT = "panda_zed_wrist_cam__panda_gripper"
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAME = "panda_finger_joint1"  # each finger moves half the gripper width


def _format_action(action: EEPoseDelta) -> str:
    return (
        f"d = ({action.dx_cm:+.1f}, {action.dy_cm:+.1f}, {action.dz_cm:+.1f}) cm, "
        f"rot = ({action.droll_deg:+.1f}, {action.dpitch_deg:+.1f}, {action.dyaw_deg:+.1f}) deg, "
        f"gripper {action.gripper}"
    )


def _set_pose(handle, T: np.ndarray) -> None:
    handle.position = T[:3, 3]
    handle.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz


class AgentViewer:
    def __init__(
        self,
        camera_ids: Sequence[str],
        history_length: int,
        system_prompt: str,
        port: int = 8080,
        robot_name: str = DEFAULT_VIEWER_ROBOT,
    ):
        self.server = viser.ViserServer(port=port)
        self._camera_ids = list(camera_ids)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._response_md = ""
        # Set by the reset button (viser runs it in a worker thread). The control loop polls it, does the reset at a
        # safe point, then calls finish_reset().
        self.reset_requested = threading.Event()

        # -- 3D scene ------------------------------------------------------
        scene = self.server.scene
        scene.set_up_direction("+z")
        scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1)
        assert robot_name in VIEWER_ROBOTS, f"robot_name must be one of {VIEWER_ROBOTS}, got {robot_name!r}"
        # Jrl2 has no public accessor for the robot's URDF; _yourdfpy_model is the loaded model, meshes included.
        urdf = get_robot_by_name(robot_name)._yourdfpy_model
        expected = set(ARM_JOINT_NAMES + [FINGER_JOINT_NAME])
        assert set(urdf.actuated_joint_names) == expected, f"{robot_name} joints: {urdf.actuated_joint_names}"
        self._joint_names = list(urdf.actuated_joint_names)
        # Translucent, so the end effector frames (between the fingers) stay visible through the hand.
        self._urdf = ViserUrdf(self.server, urdf, root_node_name="/panda", mesh_color_override=ROBOT_COLOR_RGBA)
        scene.add_frame("/world", axes_length=0.15, axes_radius=0.005)
        scene.add_label("/world/label", "world", position=(0.0, 0.0, -0.03))
        self._ee_frame = scene.add_frame(
            "/ee", axes_length=0.05, axes_radius=0.0025, origin_color=CURRENT_EE_ORIGIN_COLOR
        )
        scene.add_label("/ee/label", "EE", position=(0.0, 0.0, -0.02))
        self._desired_frame = scene.add_frame(
            "/ee_desired", axes_length=0.05, axes_radius=0.0025, origin_color=DESIRED_EE_ORIGIN_COLOR, visible=False
        )
        scene.add_label("/ee_desired/label", "EE desired", position=(0.0, 0.0, -0.02))

        # Also the "Reset View" target in the browser.
        self.server.initial_camera.position = (1.3, -0.9, 0.8)
        self.server.initial_camera.look_at = (0.4, 0.0, 0.25)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)

        # -- Main control panel --------------------------------------------------
        self._reset_button = self.server.gui.add_button(
            "Reset robot + clear history",
            icon=viser.Icon.HOME,
            hint="Home the robot to its start pose and clear the observation history sent to the model",
        )

        self._reset_button.on_click(lambda _event: self.request_reset())

        self._status = self.server.gui.add_markdown("**Status:** starting")

        # -- Cameras panel ---------------------------------------------------
        gui = self.server.gui
        cameras = gui.add_panel()
        with cameras.add_tab("Live", viser.Icon.CAMERA):
            self._live = {c: gui.add_image(BLANK_IMAGE, label=c) for c in self._camera_ids}
        with cameras.add_tab("Sent to model", viser.Icon.HISTORY):
            self._history_rows = []
            for _ in range(history_length):
                text = gui.add_markdown("", visible=False)
                images = {c: gui.add_image(BLANK_IMAGE, label=c, visible=False) for c in self._camera_ids}
                self._history_rows.append((text, images))
        cameras.dock_left()
        cameras.set_width(460)

        # -- Response + prompt panels ----------------------------------------
        response = gui.add_panel()
        with response.add_tab("Response", viser.Icon.MESSAGE_CHATBOT):
            self._response = gui.add_markdown("_No response yet._")
        prompt = gui.add_panel()
        with prompt.add_tab("Request", viser.Icon.MESSAGE):
            self._request = gui.add_markdown("_No request yet._")
        with prompt.add_tab("System prompt", viser.Icon.ROBOT):
            gui.add_markdown(f"```\n{system_prompt}\n```")
        response.dock_right()
        response.set_width(420)
        prompt.dock_below(response)

    @property
    def url(self) -> str:
        return f"http://localhost:{self.server.get_port()}"

    # -- live updates -----------------------------------------------------------

    def start_live(
        self,
        get_robot_state: Callable[[], tuple[np.ndarray, float | None, np.ndarray] | None],
        get_image: Callable[[str], np.ndarray | None],
    ) -> None:
        """Mirror the robot and cameras in a background thread.

        get_robot_state() returns (q [7], gripper width m or None, T_base_ee [4, 4]), or None if unavailable.
        """

        def loop() -> None:
            tick = 0
            while not self._stop.wait(1.0 / ROBOT_UPDATE_HZ):
                state = get_robot_state()
                if state is not None:
                    q, gripper_width_m, T_base_ee = state
                    joints = dict(zip(ARM_JOINT_NAMES, q))
                    joints[FINGER_JOINT_NAME] = 0.0 if gripper_width_m is None else gripper_width_m / 2.0
                    self._urdf.update_cfg(np.array([joints[name] for name in self._joint_names]))
                    _set_pose(self._ee_frame, T_base_ee)
                if tick % LIVE_IMAGE_EVERY == 0:
                    for cam_id, handle in self._live.items():
                        image = get_image(cam_id)
                        if image is not None:
                            handle.image = image
                tick += 1

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.server.stop()

    # -- per-step updates ---------------------------------------------------------

    def set_status(self, text: str) -> None:
        self._status.content = f"**Status:** {text}"

    def show_request(self, history: Sequence[Observation], parts: Sequence[tuple[str, str]]) -> None:
        """Show the observations and text about to be sent to the model, oldest first."""
        n = len(history)
        for i, (text, images) in enumerate(self._history_rows):
            visible = i < n
            text.visible = visible
            for handle in images.values():
                handle.visible = visible
            if not visible:
                continue
            obs = history[i]
            label = "**Current observation**" if i == n - 1 else f"**{n - 1 - i} step(s) ago**"
            lines = [label, f"Gripper {pose_to_text(obs.T_base_ee)}"]
            if obs.action is not None:
                lines.append(f"Then: {_format_action(obs.action)}. {obs.outcome}".strip())
            text.content = "  \n".join(lines)
            for cam_id, handle in images.items():
                if cam_id in obs.images:
                    handle.image = obs.images[cam_id]
        self._request.content = "\n\n".join(value if kind == "text" else "`[image]`" for kind, value in parts)

    def show_response(
        self,
        step: int,
        reply: ModelReply,
        latency_s: float,
        executed: EEPoseDelta | None,
        notes: Sequence[str],
        T_desired: np.ndarray | None,
        run_cost_usd: float,
    ) -> None:
        """Show the model's step, and draw the desired next end effector pose (the target after clamping). None, e.g.
        when the model reports the task complete, hides it."""
        action = reply.action
        blocks = [
            f"### Step {step}",
            (
                f"_{reply.model}, {latency_s:.1f} s, {reply.input_tokens} in / {reply.output_tokens} out, "
                f"{format_cost(reply.cost_usd)} (run ${run_cost_usd:.4f})_"
            ),
            f"**Reasoning:** {action.reasoning}",
            f"**Requested:** {_format_action(action)}",
        ]
        if executed is not None:
            blocks.append(f"**Executed:** {_format_action(executed)}")
        if action.task_complete:
            blocks.append("**Task reported complete.**")
        if notes:
            blocks.append("\n".join(f"- {note}" for note in notes))
        self._response_md = "\n\n".join(blocks)
        self._response.content = self._response_md

        if T_desired is None:
            self._desired_frame.visible = False
            return
        _set_pose(self._desired_frame, T_desired)
        self._desired_frame.visible = True

    def show_outcome(self, outcome: str) -> None:
        """Append how execution went to the latest response."""
        if outcome:
            self._response.content = f"{self._response_md}\n\n**Outcome:** {outcome}"

    def request_reset(self) -> None:
        """What the reset button does: flag the control loop, which stops the current step and resets."""
        if not self.reset_requested.is_set():
            self._reset_button.disabled = True
            self.reset_requested.set()
            self.set_status("reset requested; waiting for the current step to stop")

    def disable_reset(self) -> None:
        """For after the control loop has ended, when nothing would act on a reset."""
        self._reset_button.disabled = True

    def finish_reset(self) -> None:
        """Clear the history, request, response and desired pose shown, and re-enable the reset button."""
        for text, images in self._history_rows:
            text.visible = False
            for handle in images.values():
                handle.visible = False
        self._request.content = "_No request yet._"
        self._response_md = ""
        self._response.content = "_No response yet._"
        self._desired_frame.visible = False
        self.reset_requested.clear()
        self._reset_button.disabled = False
