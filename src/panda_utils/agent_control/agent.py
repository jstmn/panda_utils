"""Model-driven end effector control: images in, one structured end effector pose delta out per call.

Claude models (claude-*) are called through the Anthropic SDK, OpenAI models (gpt-*, o<N>*, chatgpt-*) through the
OpenAI SDK. Both get the same prompt and observation history and return the same EEPoseDelta schema.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
import cv2
import numpy as np
import openai

from panda_utils.agent_control.actions import EEPoseDelta, pose_to_text

ANTHROPIC = "anthropic"
OPENAI = "openai"
OPENAI_MODEL_PATTERN = re.compile(r"^(gpt-|chatgpt-|o\d)")
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Models that accept server-side refusal fallbacks in the "default" form on the Claude API. A declined request is
# re-run on Anthropic's recommended fallback model inside the same call rather than coming back as a refusal.
SERVER_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

MAX_OUTPUT_TOKENS = 16000

# USD per 1M tokens: (input, output, cache read, cache write). Neither API returns a cost, so it is estimated from the
# token counts. Prices change: check these before trusting totals.
#   Claude: Anthropic's model table as of 2026-10-06. Cache writes at 1.25x input (5 minute TTL); Haiku 5.5's cache
#   read price assumed at 0.1x input.
#   OpenAI: developers.openai.com/api/docs/pricing as of 2026-10-21, standard tier; cache writes at the 30 minute rate.
# Haiku 5.5 costs more for prompts over 100K tokens, and OpenAI's long context rate starts above 272K; prompts here are
# far smaller than either.
PRICES_PER_MTOK = {
    "claude-fable-5-1": (10.00, 50.00, 0.25, 12.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "gpt-6.1-sol": (2.00, 10.00, 0.10, 2.50),
    "gpt-6-sol": (2.00, 10.00, 0.20, 2.50),
    "gpt-6-astra": (10.00, 50.00, 1.00, 12.50),
    "gpt-6-luna": (0.10, 0.50, 0.01, 0.125),
}
JPEG_QUALITY = 90

# Checked against the real arm: at the home pose the reported orientation is roll/pitch/yaw ~ (180, -2, 1) deg, with the
# gripper's +z along world -z, +y along world -y and +x along world +x.
COORDINATES_TEXT = (
    "Coordinates. The world frame is fixed to the robot's base: +x points forward, away from the base; +y points to "
    "the robot's left; +z points up. Every gripper pose you are given is in the world frame: the position (cm) of the "
    "gripper's center point, midway between the fingertips, and its orientation as roll / pitch / yaw (deg), "
    "rotations about the world x, y and z axes applied in that order. Roll = pitch = yaw = 0 would point the fingers "
    "straight up; roll = 180 points them straight down.\n"
    "The gripper frame is attached to that center point and moves with the gripper: +z points out of the gripper toward "
    "the fingertips (the direction it reaches), +y is the line the two fingers open and close along, and +x is "
    "perpendicular to both (right-handed). In the home pose the gripper points straight down: its +x is world +x, "
    "+y is world -y (the robot's right) and +z is world -z."
)
ACTION_FRAME_TEXT = {
    "world": (
        "Your action moves the gripper relative to where it is now, in the world frame: dx / dy / dz shift it along "
        "world x / y / z, and droll / dpitch / dyaw rotate it about world x / y / z, pivoting about the gripper's center "
        "point (a pure rotation does not move that point). For example dz = -2 lowers the gripper 2 cm."
    ),
    "ee": (
        "Your action moves the gripper relative to where it is now, in the gripper frame as it is at that moment: "
        "dx / dy / dz shift it along the gripper's own x / y / z axes, and droll / dpitch / dyaw rotate it about those "
        "axes, pivoting about the gripper's center point. For example dz = +2 moves the gripper 2 cm in the direction "
        "the fingertips point, whichever way that is."
    ),
}


class AgentRefusalError(RuntimeError):
    """The model (and its fallback, if any) declined the request."""


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # camera id -> RGB
    T_base_ee: np.ndarray  # [4, 4] end effector pose in the robot base frame
    gripper_width_m: float | None
    action: EEPoseDelta | None = None  # the step taken after this observation, as executed (after clamping)
    outcome: str = ""  # what happened when the step was executed, e.g. "target clamped to the workspace"


@dataclass
class ModelReply:
    """One call's result, the same shape for every provider."""

    action: EEPoseDelta
    model: str  # as reported by the API, which may differ from the requested one (e.g. after a fallback)
    request_id: str | None
    input_tokens: int  # all input, including cache reads and writes
    output_tokens: int  # including any reasoning / thinking tokens
    cache_read_tokens: int
    cache_write_tokens: int
    usage: dict  # the provider's raw usage object
    cost_usd: float | None  # estimated from PRICES_PER_MTOK; None if the model has no entry


def format_cost(cost_usd: float | None) -> str:
    return "cost unknown (model not in PRICES_PER_MTOK)" if cost_usd is None else f"${cost_usd:.4f}"


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_write_tokens: int
) -> float | None:
    """Estimated USD for one call. `model` may be a dated snapshot of a priced model (OpenAI reports e.g.
    gpt-6.1-sol-<date>), so the longest PRICES_PER_MTOK name it starts with is used."""
    matches = [name for name in PRICES_PER_MTOK if model.startswith(name)]
    if not matches:
        return None
    price_in, price_out, price_cache_read, price_cache_write = PRICES_PER_MTOK[max(matches, key=len)]
    uncached = input_tokens - cache_read_tokens - cache_write_tokens
    return (
        uncached * price_in
        + output_tokens * price_out
        + cache_read_tokens * price_cache_read
        + cache_write_tokens * price_cache_write
    ) / 1e6


def provider_for_model(model: str) -> str:
    if model.startswith("claude-"):
        return ANTHROPIC
    if OPENAI_MODEL_PATTERN.match(model):
        return OPENAI
    raise ValueError(
        f"Unrecognised model {model!r}: expected a Claude model (claude-*) or an OpenAI model (gpt-*, o<N>*)"
    )


def build_system_prompt(
    task: str,
    frame: str,
    max_translation_cm: float,
    max_rotation_deg: float,
    history_length: int,
) -> str:
    lines = [
        (
            "You control a Franka Panda arm with a parallel-jaw gripper. Each turn you see camera images of the scene "
            "and choose the next small motion of the gripper, which is executed before you see the next images."
        ),
        "",
        f"Task: {task}",
        "",
        COORDINATES_TEXT,
        "",
        ACTION_FRAME_TEXT[frame],
        (
            f"One step moves at most {max_translation_cm:g} cm and rotates at most {max_rotation_deg:g} degrees; larger "
            "requests are scaled down, keeping their direction. Targets outside the robot's workspace are clamped to its "
            "edge" #. Prefer several small, deliberate steps near objects over one large one.
        ),
        "The gripper command is applied after the motion, and stays as set until you change it.",
    ]
    if history_length > 1:
        lines.append(
            f"You are shown up to the {history_length} most recent observations, oldest first, each followed by the "
            "step that was taken after it. Use them to judge whether previous steps had the effect you intended."
        )
    lines.append("Set task_complete to true only once the images show the task is done; that step is not executed.")
    return "\n".join(lines)


def encode_jpeg_b64(image_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok, "JPEG encoding failed"
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


class PoseDeltaAgent:
    """Asks a model for the next end effector pose delta given the last `history_length` observations."""

    def __init__(
        self,
        model: str,
        task: str,
        frame: str,
        history_length: int,
        max_translation_cm: float,
        max_rotation_deg: float,
        effort: str | None = None,
        client: anthropic.Anthropic | openai.OpenAI | None = None,
    ):
        """Raises ValueError for an unrecognised model, or an OpenAI model without OPENAI_API_KEY set."""
        assert history_length >= 1, f"history_length must be >= 1, got {history_length}"
        self.model = model
        self.provider = provider_for_model(model)
        self.effort = effort
        self.history: deque[Observation] = deque(maxlen=history_length)
        self.system_prompt = build_system_prompt(task, frame, max_translation_cm, max_rotation_deg, history_length)
        if self.provider == OPENAI and client is None and not os.environ.get(OPENAI_API_KEY_ENV):
            raise ValueError(f"{model} is an OpenAI model: set {OPENAI_API_KEY_ENV} in the shell environment")
        if client is None:
            client = openai.OpenAI() if self.provider == OPENAI else anthropic.Anthropic()
        self._client = client

    def _observation_parts(self, obs: Observation, label: str) -> list[tuple[str, str]]:
        """("text", str) / ("image", base64 jpeg) parts, converted to each provider's block format at call time."""
        gripper = "unknown" if obs.gripper_width_m is None else f"{obs.gripper_width_m * 100:.1f} cm open"
        parts = [("text", f"{label}\nGripper (world frame): {pose_to_text(obs.T_base_ee)}; fingers {gripper}.")]
        for cam_id, image in obs.images.items():
            parts += [("text", f"Camera '{cam_id}':"), ("image", encode_jpeg_b64(image))]
        if obs.action is not None:
            step = json.dumps(obs.action.model_dump())
            parts.append(("text", f"Step taken after this observation: {step}. {obs.outcome}".strip()))
        return parts

    def build_parts(self) -> list[tuple[str, str]]:
        parts = []
        n = len(self.history)
        for i, obs in enumerate(self.history):
            label = "Current observation:" if i == n - 1 else f"Observation from {n - 1 - i} step(s) ago:"
            parts += self._observation_parts(obs, label)
        parts.append(("text", "Choose the next step."))
        return parts

    def act(
        self,
        obs: Observation,
        on_request: Callable[[list[Observation], list[tuple[str, str]]], None] | None = None,
    ) -> ModelReply:
        """Add `obs` to the history and return the model's next step.

        `on_request(history, parts)` is called with exactly what is about to be sent, before the (slow) API call.
        Fill in `obs.action` / `obs.outcome` with what was actually executed so the next call reports it.
        """
        self.history.append(obs)
        parts = self.build_parts()
        if on_request is not None:
            on_request(list(self.history), parts)
        if self.provider == OPENAI:
            return self._act_openai(parts)
        return self._act_anthropic(parts)

    def _act_anthropic(self, parts: list[tuple[str, str]]) -> ModelReply:
        content = [
            (
                {"type": "text", "text": value}
                if kind == "text"
                else {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": value}}
            )
            for kind, value in parts
        ]
        kwargs = {}
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        if self.model in SERVER_FALLBACK_MODELS:
            kwargs["betas"] = [SERVER_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}],
            output_format=EEPoseDelta,
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise AgentRefusalError(f"{self.model} declined the request: {response.stop_details}")
        if response.parsed_output is None:
            raise RuntimeError(f"No structured output from {self.model} (stop_reason={response.stop_reason})")
        # Anthropic reports cache reads and writes separately from input_tokens. Priced at the model that answered,
        # which after a server-side fallback is the fallback model.
        usage = response.usage
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        input_tokens = usage.input_tokens + cache_read + cache_write
        return ModelReply(
            action=response.parsed_output,
            model=response.model,
            request_id=response._request_id,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            usage=usage.model_dump(),
            cost_usd=estimate_cost_usd(response.model, input_tokens, usage.output_tokens, cache_read, cache_write),
        )

    def _act_openai(self, parts: list[tuple[str, str]]) -> ModelReply:
        content = [
            (
                {"type": "input_text", "text": value}
                if kind == "text"
                else {"type": "input_image", "image_url": f"data:image/jpeg;base64,{value}", "detail": "auto"}
            )
            for kind, value in parts
        ]
        kwargs = {}
        if self.effort is not None:
            kwargs["reasoning"] = {"effort": self.effort}
        response = self._client.responses.parse(
            model=self.model,
            instructions=self.system_prompt,
            input=[{"role": "user", "content": content}],
            text_format=EEPoseDelta,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            **kwargs,
        )
        refusals = [
            block.refusal
            for item in response.output
            if item.type == "message"
            for block in item.content
            if block.type == "refusal"
        ]
        if refusals:
            raise AgentRefusalError(f"{self.model} declined the request: {' '.join(refusals)}")
        if response.output_parsed is None:
            raise RuntimeError(f"No structured output from {self.model} (status={response.status})")
        # OpenAI's input_tokens already include cached tokens and cache writes.
        usage = response.usage
        details = usage.input_tokens_details
        cache_read = (details.cached_tokens or 0) if details is not None else 0
        cache_write = (getattr(details, "cache_write_tokens", None) or 0) if details is not None else 0
        return ModelReply(
            action=response.output_parsed,
            model=response.model,
            request_id=getattr(response, "_request_id", None),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            usage=usage.model_dump(),
            cost_usd=estimate_cost_usd(
                response.model, usage.input_tokens, usage.output_tokens, cache_read, cache_write
            ),
        )
