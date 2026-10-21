"""Structured action the model returns each step."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

# No ge/le bounds on the numeric fields: the SDK strips them from the schema sent to the API and then validates
# client-side, so an over-limit step would raise instead of being clamped. Limits are stated in the prompt and
# enforced by geometry.clamp_delta.


class EEPoseDelta(BaseModel):
    """Relative end effector motion, expressed in the control frame chosen for the run (world or end effector)."""

    reasoning: str = Field(description="One or two sentences: what you see, and why this motion.")
    dx_cm: float = Field(description="Translation along the control frame's x axis, in centimeters.")
    dy_cm: float = Field(description="Translation along the control frame's y axis, in centimeters.")
    dz_cm: float = Field(description="Translation along the control frame's z axis, in centimeters.")
    droll_deg: float = Field(description="Rotation about the control frame's x axis, in degrees.")
    dpitch_deg: float = Field(description="Rotation about the control frame's y axis, in degrees.")
    dyaw_deg: float = Field(description="Rotation about the control frame's z axis, in degrees.")
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
