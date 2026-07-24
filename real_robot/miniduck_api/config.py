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
            obs_size=int(raw.get("obs_size", 64)),
            action_size=int(raw.get("action_size", 10)),
        )
        if len(cfg.joint_names) != cfg.action_size or len(cfg.default_actuator) != cfg.action_size:
            raise ValueError("metadata joint/action dimensions are inconsistent")
        return cfg

