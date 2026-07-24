from __future__ import annotations

import time
from typing import Protocol, Sequence

import numpy as np

from miniduck_api.contracts import ImuSample, JointState, RobotState

from .config import HardwareConfig


class ServoBus(Protocol):
    """Minimal boundary to be implemented with the actual servo SDK."""

    def read_positions_rad(self, servo_ids: Sequence[int]) -> Sequence[float]: ...
    def read_velocities_rad_s(self, servo_ids: Sequence[int]) -> Sequence[float]: ...
    def write_positions_rad(
        self, servo_ids: Sequence[int], targets_rad: Sequence[float]
    ) -> None: ...
    def disable_torque(self, servo_ids: Sequence[int]) -> None: ...


class ImuSource(Protocol):
    """IMU values must already use the robot body frame and SI units."""

    def read_imu(self) -> ImuSample: ...


class MiniDuckHardwareAdapter:
    """Convert between policy joint coordinates and physical servo coordinates."""

    def __init__(
        self,
        bus: ServoBus,
        imu: ImuSource,
        config: HardwareConfig,
        *,
        require_calibrated: bool = True,
    ):
        if require_calibrated:
            config.require_calibrated()
        self.bus = bus
        self.imu = imu
        self.config = config
        self._last_targets: np.ndarray | None = None

    def _servo_to_policy(self, values: Sequence[float], *, velocity: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise RuntimeError("servo bus returned invalid joint values")
        result = np.empty(10, dtype=np.float64)
        for index, joint in enumerate(self.config.joints):
            offset = 0.0 if velocity else joint.zero_offset_rad
            result[index] = joint.direction * (values[index] - offset)
        return result

    def _policy_to_servo(self, targets: Sequence[float]) -> np.ndarray:
        policy_targets = np.asarray(targets, dtype=np.float64)
        if policy_targets.shape != (10,) or not np.isfinite(policy_targets).all():
            raise ValueError("joint targets must contain 10 finite values")

        limited = np.empty(10, dtype=np.float64)
        for index, joint in enumerate(self.config.joints):
            limited[index] = np.clip(
                policy_targets[index], joint.min_rad, joint.max_rad
            )

        if self._last_targets is not None:
            delta = np.clip(
                limited - self._last_targets,
                -self.config.max_target_step_rad,
                self.config.max_target_step_rad,
            )
            limited = self._last_targets + delta
        self._last_targets = limited

        servo_targets = np.empty(10, dtype=np.float64)
        for index, joint in enumerate(self.config.joints):
            servo_targets[index] = (
                joint.zero_offset_rad + joint.direction * limited[index]
            )
        return servo_targets

    def read_state(self) -> RobotState:
        ids = self.config.servo_ids
        positions = self._servo_to_policy(
            self.bus.read_positions_rad(ids), velocity=False
        )
        velocities = self._servo_to_policy(
            self.bus.read_velocities_rad_s(ids), velocity=True
        )
        return RobotState(
            timestamp_s=time.monotonic(),
            imu=self.imu.read_imu(),
            joints=JointState(positions, velocities),
        )

    def write_joint_targets(self, targets_rad: Sequence[float]) -> None:
        targets = self._policy_to_servo(targets_rad)
        self.bus.write_positions_rad(self.config.servo_ids, targets)

    def disable_torque(self) -> None:
        self.bus.disable_torque(self.config.servo_ids)
        self._last_targets = None

