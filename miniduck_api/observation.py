from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import PolicyConfig
from .contracts import Command, RobotState


class ObservationBuilder:
    """Builds the exact 64-element observation used in training."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.step_index = 0
        self.action_history = [np.zeros(config.action_size, dtype=np.float32) for _ in range(3)]

    def reset(self) -> None:
        self.step_index = 0
        for action in self.action_history:
            action.fill(0.0)

    def push_action(self, action: Sequence[float]) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (self.config.action_size,):
            raise ValueError(f"action must have shape ({self.config.action_size},)")
        self.action_history = [value.copy(), self.action_history[0], self.action_history[1]]
        self.step_index += 1

    def build(self, state: RobotState, command: Command) -> np.ndarray:
        scales = self.config.obs_scales
        if (
            abs(command.vx) < 0.02
            and abs(command.vy) < 0.02
            and abs(command.yaw_rate) < 0.1
        ):
            # Training pins zero-command emergency stops to the nominal stand
            # phase; deployment must build the same observation.
            self.step_index = 0
        command_scale = np.asarray(scales["command"], dtype=np.float32)
        q_error = np.asarray(state.joints.position) - np.asarray(self.config.default_actuator)
        angle = 2.0 * math.pi * self.step_index / self.config.gait_phase_period_steps
        obs = np.concatenate([
            np.asarray(state.imu.gyro) * float(scales["ang_vel"]),
            np.asarray(state.imu.gravity) * float(scales["gravity"]),
            np.asarray(state.imu.acceleration) * float(scales["accel"]),
            np.asarray([command.vx, command.vy, command.yaw_rate]) * command_scale,
            q_error * float(scales["dof_pos"]),
            np.asarray(state.joints.velocity) * float(scales["dof_vel"]),
            *self.action_history,
            np.asarray([math.cos(angle), math.sin(angle)]),
        ]).astype(np.float32)
        if obs.shape != (self.config.obs_size,):
            raise RuntimeError(f"built observation has invalid shape {obs.shape}")
        return obs
