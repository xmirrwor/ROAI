from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ActionStage:
    index: int
    key: str
    objective: str
    implemented: bool
    prerequisites: tuple[str, ...]


ACTION_STAGES = (
    ActionStage(
        0,
        "emergency_stop_stand",
        "Stop from a low-speed probe and settle into the nominal standing pose.",
        True,
        ("flat_ground", "velocity_command"),
    ),
    ActionStage(
        1,
        "squat",
        "Track a commanded body-height profile while keeping both feet stable.",
        False,
        ("height_command", "per_env_height_target"),
    ),
    ActionStage(
        2,
        "action_switch",
        "Switch between stand, squat and locomotion without losing balance.",
        False,
        ("skill_conditioning", "transition_sampler"),
    ),
    ActionStage(
        3,
        "fall_recovery",
        "Recover from sampled prone and supine poses before timeout.",
        False,
        ("recovery_reset_distribution", "recovery_termination_rule"),
    ),
    ActionStage(
        4,
        "diagonal_motion",
        "Track simultaneous forward and lateral velocity commands.",
        False,
        ("combined_command_sampler",),
    ),
    ActionStage(
        5,
        "obstacle_crossing",
        "Cross height-varied obstacles without body collision or foot trapping.",
        False,
        ("obstacle_terrain", "height_observation", "clearance_reward"),
    ),
    ActionStage(
        6,
        "ball_kick",
        "Approach and kick a simulated ball toward a target.",
        False,
        ("ball_actor", "ball_observation", "kick_target_reward"),
    ),
)


def _validate_stages(stages: Sequence[ActionStage]) -> None:
    if not stages:
        raise ValueError("action curriculum must contain at least one stage")
    expected = list(range(len(stages)))
    actual = [stage.index for stage in stages]
    if actual != expected:
        raise ValueError(f"action stage indexes must be contiguous: {actual}")
    keys = [stage.key for stage in stages]
    if len(set(keys)) != len(keys):
        raise ValueError(f"action stage keys must be unique: {keys}")


_validate_stages(ACTION_STAGES)


def planned_stage_for_step(
    step: int,
    start_step: int,
    stage_steps: Sequence[int],
    stages: Sequence[ActionStage] = ACTION_STAGES,
) -> Optional[ActionStage]:
    """Return the roadmap stage at a simulator step without capability gating."""
    _validate_stages(stages)
    if len(stage_steps) != len(stages):
        raise ValueError("stage_steps must contain one duration per action stage")
    if any(duration <= 0 for duration in stage_steps):
        raise ValueError("all action stage durations must be positive")
    if step < start_step:
        return None

    relative_step = step - start_step
    for stage, duration in zip(stages, stage_steps):
        if relative_step < duration:
            return stage
        relative_step -= duration
    return stages[-1]


def active_stage_for_step(
    step: int,
    start_step: int,
    stage_steps: Sequence[int],
    max_implemented_stage: int,
) -> Optional[ActionStage]:
    """Clamp the roadmap to stages whose environment support is implemented."""
    if not 0 <= max_implemented_stage < len(ACTION_STAGES):
        raise ValueError("max_implemented_stage is outside ACTION_STAGES")
    planned = planned_stage_for_step(step, start_step, stage_steps)
    if planned is None:
        return None
    return ACTION_STAGES[min(planned.index, max_implemented_stage)]
