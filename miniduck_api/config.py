from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyConfig:
    joint_names: tuple[str, ...]
    default_actuator: tuple[float, ...]
    action_scale: float
    policy_dt: float
    gait_phase_period_steps: int
    obs_scales: dict
    command_ranges: dict
    nominal_motor_velocity: float
    nominal_body_height_m: float = 0.143
    squat_body_height_m: float = 0.132
    heading_hold_kp: float = 2.5
    heading_hold_kd: float = 0.30
    heading_hold_max_yaw_rate: float = 0.45
    heading_hold_stop_s: float = 2.0
    obs_size: int = 64
    action_size: int = 10

    @classmethod
    def load(cls, path: str | Path) -> "PolicyConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(
            joint_names=tuple(raw["joint_names"]),
            default_actuator=tuple(float(x) for x in raw["default_actuator"]),
            action_scale=float(raw["action_scale"]),
            policy_dt=float(raw["policy_dt"]),
            gait_phase_period_steps=int(raw["gait_phase_period_steps"]),
            obs_scales=raw["obs_scales"],
            command_ranges=raw["command_ranges"],
            nominal_motor_velocity=float(raw["nominal_motor_velocity"]),
            nominal_body_height_m=float(raw.get("nominal_body_height_m", 0.143)),
            squat_body_height_m=float(raw.get("squat_body_height_m", 0.132)),
            heading_hold_kp=float(raw.get("heading_hold_kp", 2.5)),
            heading_hold_kd=float(raw.get("heading_hold_kd", 0.30)),
            heading_hold_max_yaw_rate=float(raw.get("heading_hold_max_yaw_rate", 0.45)),
            heading_hold_stop_s=float(raw.get("heading_hold_stop_s", 2.0)),
            obs_size=int(raw.get("obs_size", 64)),
            action_size=int(raw.get("action_size", 10)),
        )
        if len(cfg.joint_names) != cfg.action_size or len(cfg.default_actuator) != cfg.action_size:
            raise ValueError("metadata joint/action dimensions are inconsistent")
        return cfg
