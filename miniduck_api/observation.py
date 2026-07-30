from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import PolicyConfig
from .contracts import Command, RobotState, SkillMode


class ObservationBuilder:
    """Builds the exact 64-element observation used in training."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.step_index = 0
        self.heading_error_rad = 0.0
        self.line_active = False
        self.stationary_steps = 0
        self.action_history = [np.zeros(config.action_size, dtype=np.float32) for _ in range(3)]

    def reset(self) -> None:
        self.step_index = 0
        self.heading_error_rad = 0.0
        self.line_active = False
        self.stationary_steps = 0
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
        stationary = (
            abs(command.vx) < 0.02
            and abs(command.vy) < 0.02
            and abs(command.yaw_rate) < 0.1
        )
        pure_sagittal = (
            abs(command.vx) >= 0.02
            and abs(command.vy) < 0.02
            and abs(command.yaw_rate) < 0.1
        )
        effective_yaw_rate = command.yaw_rate
        if pure_sagittal and not self.line_active:
            self.heading_error_rad = 0.0
            self.line_active = True
            self.stationary_steps = 0
        if self.line_active and (pure_sagittal or stationary):
            self.heading_error_rad += float(state.imu.gyro[2]) * self.config.policy_dt
            effective_yaw_rate = float(np.clip(
                -self.config.heading_hold_kp * self.heading_error_rad
                - self.config.heading_hold_kd * float(state.imu.gyro[2]),
                -self.config.heading_hold_max_yaw_rate,
                self.config.heading_hold_max_yaw_rate,
            ))
            if stationary:
                self.stationary_steps += 1
                if self.stationary_steps * self.config.policy_dt >= self.config.heading_hold_stop_s:
                    self.line_active = False
            else:
                self.stationary_steps = 0
        skill = command.skill
        if skill == SkillMode.AUTO:
            skill = SkillMode.STAND if stationary else SkillMode.LOCOMOTION
        if skill != SkillMode.LOCOMOTION:
            # Training pins zero-command emergency stops to the nominal stand
            # phase; deployment must build the same observation.
            self.step_index = 0
        command_scale = np.asarray(scales["command"], dtype=np.float32)
        q_error = np.asarray(state.joints.position) - np.asarray(self.config.default_actuator)
        angle = 2.0 * math.pi * self.step_index / self.config.gait_phase_period_steps
        if skill == SkillMode.SQUAT:
            target_height = (
                self.config.squat_body_height_m
                if command.body_height_m is None
                else float(command.body_height_m)
            )
            span = max(
                self.config.nominal_body_height_m - self.config.squat_body_height_m,
                1.0e-6,
            )
            depth = np.clip(
                (self.config.nominal_body_height_m - target_height) / span,
                0.0,
                1.0,
            )
            skill_features = np.asarray([1.0 - 2.0 * depth, 1.0])
        elif skill == SkillMode.STAND:
            skill_features = np.asarray([1.0, 0.0])
        else:
            skill_features = np.asarray([math.cos(angle), math.sin(angle)])
        obs = np.concatenate([
            np.asarray(state.imu.gyro) * float(scales["ang_vel"]),
            np.asarray(state.imu.gravity) * float(scales["gravity"]),
            np.asarray(state.imu.acceleration) * float(scales["accel"]),
            np.asarray([command.vx, command.vy, effective_yaw_rate]) * command_scale,
            q_error * float(scales["dof_pos"]),
            np.asarray(state.joints.velocity) * float(scales["dof_vel"]),
            *self.action_history,
            skill_features,
        ]).astype(np.float32)
        if obs.shape != (self.config.obs_size,):
            raise RuntimeError(f"built observation has invalid shape {obs.shape}")
        return obs
