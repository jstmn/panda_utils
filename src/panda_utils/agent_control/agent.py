"""Claude-driven end effector control: images in, one structured end effector pose delta out per call."""

from __future__ import annotations

import base64
import json
from collections import deque
from dataclasses import dataclass

import anthropic
import cv2
import numpy as np
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage

from panda_utils.agent_control.geometry import pose_to_text
from panda_utils.agent_control.schema import EEPoseDelta

# Models that accept server-side refusal fallbacks in the "default" form on the Claude API. A declined request is
# re-run on Anthropic's recommended fallback model inside the same call rather than coming back as a refusal.
SERVER_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

JPEG_QUALITY = 90

FRAME_DESCRIPTIONS = {
    "world": (
        "the robot base frame: +x points forward away from the robot base, +y to the robot's left, +z up. "
        "Rotations are about these fixed axes, through the gripper."
    ),
    "ee": (
        "the end effector frame, attached to the gripper: +z points out of the gripper toward the fingertips, and "
        "the fingers open and close along y. Translations and rotations are along and about the gripper's current "
        "axes, so they move with it."
    ),
}


class AgentRefusalError(RuntimeError):
    """The model (and its fallback, if any) declined the request."""


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # camera id -> RGB, already annotated if the EE frame is drawn
    T_base_ee: np.ndarray  # [4, 4] end effector pose in the robot base frame
    gripper_width_m: float | None
    action: EEPoseDelta | None = None  # the step taken after this observation, as executed (after clamping)
    outcome: str = ""  # what happened when the step was executed, e.g. "target clamped to the workspace"


def build_system_prompt(
    task: str,
    frame: str,
    max_translation_cm: float,
    max_rotation_deg: float,
    history_length: int,
    draw_ee_frame: bool,
) -> str:
    lines = [
        (
            "You control a Franka Panda arm with a parallel-jaw gripper. Each turn you see camera images of the scene "
            "and choose the next small motion of the gripper, which is executed before you see the next images."
        ),
        "",
        f"Task: {task}",
        "",
        f"Motions are relative to the gripper's current pose and expressed in {FRAME_DESCRIPTIONS[frame]}",
        (
            f"One step moves at most {max_translation_cm:g} cm and rotates at most {max_rotation_deg:g} degrees; larger "
            "requests are scaled down, keeping their direction. Targets outside the robot's workspace are clamped to its "
            "edge. Prefer several small, deliberate steps near objects over one large one."
        ),
        "The gripper command is applied after the motion, and stays as set until you change it.",
    ]
    if history_length > 1:
        lines.append(
            f"You are shown up to the {history_length} most recent observations, oldest first, each followed by the "
            "step that was taken after it. Use them to judge whether previous steps had the effect you intended."
        )
    if draw_ee_frame:
        lines.append(
            "The gripper's end effector frame is drawn on the images: red = x, green = y, blue = z (pointing toward "
            "the fingertips). Use it to relate image directions to the axes you command."
        )
    lines.append("Set task_complete to true only once the images show the task is done; that step is not executed.")
    return "\n".join(lines)


def encode_jpeg_b64(image_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok, "JPEG encoding failed"
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


class PoseDeltaAgent:
    """Asks a Claude model for the next end effector pose delta given the last `history_length` observations."""

    def __init__(
        self,
        model: str,
        task: str,
        frame: str,
        history_length: int,
        max_translation_cm: float,
        max_rotation_deg: float,
        draw_ee_frame: bool,
        effort: str | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        assert history_length >= 1, f"history_length must be >= 1, got {history_length}"
        self.model = model
        self.effort = effort
        self.history: deque[Observation] = deque(maxlen=history_length)
        self.system_prompt = build_system_prompt(
            task, frame, max_translation_cm, max_rotation_deg, history_length, draw_ee_frame
        )
        self._client = client or anthropic.Anthropic()

    def _observation_blocks(self, obs: Observation, label: str) -> list[dict]:
        gripper = "unknown" if obs.gripper_width_m is None else f"{obs.gripper_width_m * 100:.1f} cm open"
        blocks = [{"type": "text", "text": f"{label}\nGripper: {pose_to_text(obs.T_base_ee)}; width {gripper}."}]
        for cam_id, image in obs.images.items():
            blocks.append({"type": "text", "text": f"Camera '{cam_id}':"})
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_jpeg_b64(image)},
                }
            )
        if obs.action is not None:
            step = json.dumps(obs.action.model_dump())
            blocks.append({"type": "text", "text": f"Step taken after this observation: {step}. {obs.outcome}".strip()})
        return blocks

    def build_content(self) -> list[dict]:
        content = []
        n = len(self.history)
        for i, obs in enumerate(self.history):
            label = "Current observation:" if i == n - 1 else f"Observation from {n - 1 - i} step(s) ago:"
            content += self._observation_blocks(obs, label)
        content.append({"type": "text", "text": "Choose the next step."})
        return content

    def act(self, obs: Observation) -> tuple[EEPoseDelta, ParsedBetaMessage[EEPoseDelta]]:
        """Add `obs` to the history and return the model's next step along with the raw response, for logging.

        Fill in `obs.action` / `obs.outcome` with what was actually executed so the next call reports it.
        """
        self.history.append(obs)
        kwargs = {}
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        if self.model in SERVER_FALLBACK_MODELS:
            kwargs["betas"] = [SERVER_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": self.build_content()}],
            output_format=EEPoseDelta,
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise AgentRefusalError(f"{self.model} declined the request: {response.stop_details}")
        if response.parsed_output is None:
            raise RuntimeError(f"No structured output from {self.model} (stop_reason={response.stop_reason})")
        return response.parsed_output, response
