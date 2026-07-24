from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class JointCalibration:
    name: str
    servo_id: int
    zero_offset_rad: float
    direction: int
    min_rad: float
    max_rad: float
    calibrated: bool

    def validate(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError(f"{self.name}: direction must be -1 or 1")
        if self.min_rad >= self.max_rad:
            raise ValueError(f"{self.name}: min_rad must be less than max_rad")


@dataclass(frozen=True)
class HardwareConfig:
    policy_dt_s: float
    max_state_age_s: float
    max_target_step_rad: float
    joints: tuple[JointCalibration, ...]

    @classmethod
    def load(cls, path: str | Path) -> "HardwareConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        joints = tuple(JointCalibration(**item) for item in raw["joints"])
        cfg = cls(
            policy_dt_s=float(raw["policy_dt_s"]),
            max_state_age_s=float(raw["safety"]["max_state_age_s"]),
            max_target_step_rad=float(raw["safety"]["max_target_step_rad"]),
            joints=joints,
        )
        cfg.validate()
        return cfg

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def servo_ids(self) -> tuple[int, ...]:
        return tuple(joint.servo_id for joint in self.joints)

    def validate(self) -> None:
        if len(self.joints) != 10:
            raise ValueError(f"expected 10 joints, got {len(self.joints)}")
        if len(set(self.joint_names)) != len(self.joints):
            raise ValueError("joint names must be unique")
        if len(set(self.servo_ids)) != len(self.joints):
            raise ValueError("servo IDs must be unique")
        if self.policy_dt_s <= 0 or self.max_state_age_s <= 0:
            raise ValueError("timing values must be positive")
        if self.max_target_step_rad <= 0:
            raise ValueError("max_target_step_rad must be positive")
        for joint in self.joints:
            joint.validate()

    def require_calibrated(self) -> None:
        missing = [joint.name for joint in self.joints if not joint.calibrated]
        if missing:
            raise RuntimeError(
                "real robot is not calibrated: " + ", ".join(missing)
            )

