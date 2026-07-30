from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import numpy as np

from .config import PolicyConfig
from .contracts import Command, RobotState, RobotTransport
from .observation import ObservationBuilder


@dataclass(frozen=True)
class AgentOutput:
    observation: np.ndarray
    action: np.ndarray
    joint_targets: np.ndarray


class MiniDuckAgent:
    """One sim-to-real agent: state -> observation -> PPO policy -> joint targets.

    `policy` only needs infer(observation) and joint_targets(action), so tests and
    simulator adapters can replace ONNX without changing the agent.
    """

    def __init__(self, policy, config: PolicyConfig):
        self.policy = policy
        self.config = config
        self.observations = ObservationBuilder(config)
        self.previous_joint_targets = np.asarray(
            config.default_actuator, dtype=np.float32
        )

    def reset(self) -> None:
        self.observations.reset()
        self.previous_joint_targets = np.asarray(
            self.config.default_actuator, dtype=np.float32
        )

    def act(self, state: RobotState, command: Command) -> AgentOutput:
        self._validate_state(state)
        safe_command = self._limit_command(command)
        observation = self.observations.build(state, safe_command)
        action = np.asarray(self.policy.infer(observation), dtype=np.float32)
        if action.shape != (self.config.action_size,) or not np.isfinite(action).all():
            raise RuntimeError("policy returned an invalid action")
        targets = np.asarray(self.policy.joint_targets(action), dtype=np.float32)
        if targets.shape != (self.config.action_size,) or not np.isfinite(targets).all():
            raise RuntimeError("policy returned invalid joint targets")
        max_delta = self.config.nominal_motor_velocity * self.config.policy_dt
        targets = np.clip(
            targets,
            self.previous_joint_targets - max_delta,
            self.previous_joint_targets + max_delta,
        ).astype(np.float32)
        self.previous_joint_targets = targets.copy()
        self.observations.push_action(action)
        return AgentOutput(observation, action, targets)

    def _limit_command(self, command: Command) -> Command:
        values = np.asarray(
            [command.vx, command.vy, command.yaw_rate], dtype=np.float32
        )
        if not np.isfinite(values).all():
            raise RuntimeError("command contains NaN or infinity")
        names = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        limited = [
            float(np.clip(value, *self.config.command_ranges[name]))
            for value, name in zip(values, names)
        ]
        return Command(*limited)

    @staticmethod
    def _validate_state(state: RobotState) -> None:
        values: Sequence[float] = (
            *state.imu.gyro, *state.imu.gravity, *state.imu.acceleration,
            *state.joints.position, *state.joints.velocity,
        )
        if not math.isfinite(state.timestamp_s) or not all(math.isfinite(v) for v in values):
            raise RuntimeError("robot state contains NaN or infinity")

    def run(
        self,
        transport: RobotTransport,
        command_source: Callable[[], Command],
        should_stop: Callable[[], bool],
        max_state_age_s: float = 0.10,
    ) -> None:
        """Run at metadata policy rate; any failure triggers the hardware e-stop."""
        next_tick = time.monotonic()
        self.reset()
        try:
            while not should_stop():
                state = transport.read_state()
                if time.monotonic() - state.timestamp_s > max_state_age_s:
                    raise TimeoutError("robot state is stale")
                output = self.act(state, command_source())
                transport.write_joint_targets(output.joint_targets)
                next_tick += self.config.policy_dt
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.monotonic()
        finally:
            transport.disable_torque()
