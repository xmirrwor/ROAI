from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


JOINT_COUNT = 10


class SkillMode(str, Enum):
    """High-level skill selector encoded in the final two policy inputs."""

    AUTO = "auto"
    LOCOMOTION = "locomotion"
    STAND = "stand"
    SQUAT = "squat"
    RECOVERY = "recovery"
    OBSTACLE = "obstacle"


def _vector(name: str, values: Sequence[float], size: int) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(result)}")
    return result


@dataclass(frozen=True)
class Command:
    """Velocity plus an optional high-level posture command."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    skill: SkillMode = SkillMode.AUTO
    body_height_m: float | None = None
    obstacle_distance_m: float | None = None
    obstacle_height_m: float | None = None


@dataclass(frozen=True)
class ImuSample:
    gyro: tuple[float, float, float]
    gravity: tuple[float, float, float]
    acceleration: tuple[float, float, float]

    def __init__(self, gyro: Sequence[float], gravity: Sequence[float], acceleration: Sequence[float]):
        object.__setattr__(self, "gyro", _vector("gyro", gyro, 3))
        object.__setattr__(self, "gravity", _vector("gravity", gravity, 3))
        object.__setattr__(self, "acceleration", _vector("acceleration", acceleration, 3))


@dataclass(frozen=True)
class JointState:
    position: tuple[float, ...]
    velocity: tuple[float, ...]

    def __init__(self, position: Sequence[float], velocity: Sequence[float]):
        object.__setattr__(self, "position", _vector("position", position, JOINT_COUNT))
        object.__setattr__(self, "velocity", _vector("velocity", velocity, JOINT_COUNT))


@dataclass(frozen=True)
class RobotState:
    timestamp_s: float
    imu: ImuSample
    joints: JointState


class RobotTransport(Protocol):
    """Hardware teammate implements this adapter; policy code stays unchanged."""

    def read_state(self) -> RobotState: ...
    def write_joint_targets(self, targets_rad: Sequence[float]) -> None: ...
    def disable_torque(self) -> None: ...
