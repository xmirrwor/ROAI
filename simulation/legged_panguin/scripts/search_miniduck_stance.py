#!/usr/bin/env python3
"""Search for a robust slanted MiniDuck stance in Isaac Gym."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import isaacgym  # Must be imported before torch.
import torch
from isaacgym import gymtorch

from legged_panguin import LEGGED_GYM_ROOT_DIR
from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.utils import get_args, task_registry


PARAMETER_NAMES = ("pitch", "height", "hip_roll", "hip_pitch", "knee", "ankle")
BOUNDS = torch.tensor(
    [
        [math.radians(20.0), math.radians(52.0)],
        [0.105, 0.240],
        [0.0, 0.20],
        [-1.15, 0.45],
        [0.30, 1.50],
        [-1.50, 0.30],
    ],
    dtype=torch.float,
)
MUJOCO_SEEDS = torch.tensor(
    [
        [0.39687673, 0.15828534, 0.01790700, 0.11744084, 1.06108880, -0.76527804],
        [0.62500000, 0.13000000, 0.00100000, -0.12110000, 1.50000000, -0.82390000],
        [0.50000000, 0.13430000, 0.00100000, -0.26630000, 1.50000000, -0.79420000],
    ],
    dtype=torch.float,
)


def quat_pitch_xyzw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(-1)
    return torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))


def make_joint_targets(env, population: torch.Tensor) -> torch.Tensor:
    targets = torch.zeros(
        population.shape[0],
        env.num_dof,
        device=env.device,
        dtype=torch.float,
    )
    indices = {name: i for i, name in enumerate(env.dof_names)}
    roll = population[:, 2]
    hip_pitch = population[:, 3]
    knee = population[:, 4]
    ankle = population[:, 5]

    targets[:, indices["left_hip_roll"]] = roll
    targets[:, indices["left_hip_pitch"]] = hip_pitch
    targets[:, indices["left_knee"]] = knee
    targets[:, indices["left_ankle"]] = ankle
    targets[:, indices["right_hip_roll"]] = -roll
    targets[:, indices["right_hip_pitch"]] = -hip_pitch
    targets[:, indices["right_knee"]] = knee
    targets[:, indices["right_ankle"]] = ankle
    return targets


def install_population(env, population: torch.Tensor, joint_targets: torch.Tensor) -> None:
    pitch = population[:, 0]
    env.root_states.zero_()
    env.root_states[:, :3] = env.env_origins
    env.root_states[:, 2] += population[:, 1]
    env.root_states[:, 4] = torch.sin(0.5 * pitch)
    env.root_states[:, 6] = torch.cos(0.5 * pitch)
    env.dof_pos[:] = joint_targets
    env.dof_vel.zero_()

    env.gym.set_actor_root_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
    )
    env.gym.set_dof_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
    )

    env.episode_length_buf.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    env.actions.zero_()
    env.last_actions.zero_()
    env.last_dof_vel.zero_()
    env.feet_air_time.zero_()
    env.last_contacts.zero_()


def evaluate_population(env, population: torch.Tensor, steps: int) -> dict[str, torch.Tensor]:
    joint_targets = make_joint_targets(env, population)
    install_population(env, population, joint_targets)
    actions = (joint_targets - env.default_dof_pos) / env.cfg.control.action_scale

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    lifetime = torch.zeros(env.num_envs, device=env.device)
    max_pitch_error = torch.zeros(env.num_envs, device=env.device)
    max_xy_drift = torch.zeros(env.num_envs, device=env.device)
    torque_cost = torch.zeros(env.num_envs, device=env.device)
    support_steps = torch.zeros(env.num_envs, device=env.device)
    start_xy = env.root_states[:, :2].clone()

    for _ in range(steps):
        _, _, _, dones, _ = env.step(actions)
        active = alive.float()
        lifetime += active

        pitch_error = torch.abs(quat_pitch_xyzw(env.base_quat) - population[:, 0])
        xy_drift = torch.norm(env.root_states[:, :2] - start_xy, dim=1)
        max_pitch_error = torch.maximum(max_pitch_error, pitch_error * active)
        max_xy_drift = torch.maximum(max_xy_drift, xy_drift * active)
        torque_cost += torch.mean(torch.square(env.torques), dim=1) * active
        foot_contact = torch.any(
            env.contact_forces[:, env.feet_indices, 2] > 1.0,
            dim=1,
        )
        support_steps += (foot_contact & alive).float()
        alive &= ~dones.bool()

    support_ratio = support_steps / torch.clamp(lifetime, min=1.0)
    mean_torque_cost = torque_cost / torch.clamp(lifetime, min=1.0)
    stability_penalty = (
        35.0 * max_pitch_error
        + 20.0 * max_xy_drift
        + 0.02 * mean_torque_cost
        + 4.0 * (1.0 - support_ratio)
    )
    score = lifetime - stability_penalty
    score += alive.float() * steps
    return {
        "score": score,
        "alive": alive,
        "lifetime": lifetime,
        "max_pitch_error": max_pitch_error,
        "max_xy_drift": max_xy_drift,
        "mean_torque_cost": mean_torque_cost,
        "support_ratio": support_ratio,
    }


def initial_population(count: int, device: torch.device) -> torch.Tensor:
    bounds = BOUNDS.to(device)
    population = bounds[:, 0] + torch.rand(count, len(PARAMETER_NAMES), device=device) * (
        bounds[:, 1] - bounds[:, 0]
    )
    seeds = MUJOCO_SEEDS.to(device)
    seed_count = count // 2
    parent_ids = torch.arange(seed_count, device=device) % len(seeds)
    sigma = torch.tensor([0.12, 0.025, 0.035, 0.22, 0.18, 0.20], device=device)
    population[:seed_count] = seeds[parent_ids] + torch.randn(
        seed_count,
        len(PARAMETER_NAMES),
        device=device,
    ) * sigma
    return torch.maximum(torch.minimum(population, bounds[:, 1]), bounds[:, 0])


def evolve_population(
    population: torch.Tensor,
    score: torch.Tensor,
    generation: int,
    generations: int,
) -> torch.Tensor:
    bounds = BOUNDS.to(population.device)
    elite_count = min(64, population.shape[0])
    elite_ids = torch.topk(score, elite_count).indices
    elites = population[elite_ids]

    next_population = elites[
        torch.randint(elite_count, (population.shape[0],), device=population.device)
    ].clone()
    start_sigma = torch.tensor(
        [0.10, 0.020, 0.025, 0.18, 0.16, 0.18],
        device=population.device,
    )
    end_sigma = torch.tensor(
        [0.008, 0.0015, 0.002, 0.015, 0.015, 0.015],
        device=population.device,
    )
    progress = generation / max(generations - 1, 1)
    sigma = start_sigma * (1.0 - progress) + end_sigma * progress
    next_population += torch.randn_like(next_population) * sigma
    next_population[:elite_count] = elites
    return torch.maximum(torch.minimum(next_population, bounds[:, 1]), bounds[:, 0])


def as_record(population: torch.Tensor, metrics: dict[str, torch.Tensor], index: int) -> dict:
    values = population[index].detach().cpu().tolist()
    return {
        **dict(zip(PARAMETER_NAMES, values)),
        "pitch_degrees": math.degrees(values[0]),
        "score": float(metrics["score"][index].item()),
        "survived": bool(metrics["alive"][index].item()),
        "lifetime_steps": int(metrics["lifetime"][index].item()),
        "max_pitch_error_degrees": math.degrees(
            float(metrics["max_pitch_error"][index].item())
        ),
        "max_xy_drift_m": float(metrics["max_xy_drift"][index].item()),
        "mean_squared_torque": float(metrics["mean_torque_cost"][index].item()),
        "support_ratio": float(metrics["support_ratio"][index].item()),
    }


def search(args) -> None:
    torch.manual_seed(17)
    args.num_envs = args.num_envs or 2048
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.control.use_action_scale_curriculum = False
    env_cfg.init_state.reset_xy_range = 0.0
    env_cfg.init_state.reset_yaw_range = 0.0
    env_cfg.init_state.reset_roll_range = 0.0
    env_cfg.init_state.reset_pitch_range = 0.0
    env_cfg.init_state.reset_lin_vel_range = 0.0
    env_cfg.init_state.reset_ang_vel_range = 0.0
    env_cfg.init_state.reset_joint_scale_range = [1.0, 1.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    generations = 12
    evaluation_steps = 150
    population = initial_population(env.num_envs, env.device)
    best_history = []

    for generation in range(generations):
        metrics = evaluate_population(env, population, evaluation_steps)
        best_index = int(torch.argmax(metrics["score"]).item())
        best = as_record(population, metrics, best_index)
        best_history.append(best)
        survivors = int(metrics["alive"].sum().item())
        print(
            f"generation={generation:02d} survivors={survivors:4d}/{env.num_envs} "
            f"score={best['score']:.3f} pitch={best['pitch_degrees']:.3f}deg "
            f"height={best['height']:.5f} drift={best['max_xy_drift_m']:.5f}m"
        )
        population = evolve_population(
            population,
            metrics["score"],
            generation,
            generations,
        )

    final_metrics = evaluate_population(env, population, 500)
    final_index = int(torch.argmax(final_metrics["score"]).item())
    best = as_record(population, final_metrics, final_index)
    best["joint_angles"] = {
        "left_hip_yaw": 0.0,
        "left_hip_roll": best["hip_roll"],
        "left_hip_pitch": best["hip_pitch"],
        "left_knee": best["knee"],
        "left_ankle": best["ankle"],
        "right_hip_yaw": 0.0,
        "right_hip_roll": -best["hip_roll"],
        "right_hip_pitch": -best["hip_pitch"],
        "right_knee": best["knee"],
        "right_ankle": best["ankle"],
    }
    half_pitch = 0.5 * best["pitch"]
    best["base_quaternion_xyzw"] = [
        0.0,
        math.sin(half_pitch),
        0.0,
        math.cos(half_pitch),
    ]

    output_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "stance_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"miniduck_stance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "num_envs": env.num_envs,
                "generations": generations,
                "evaluation_steps": evaluation_steps,
                "dt": env.dt,
                "best": best,
                "history": best_history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"BEST_STANCE={json.dumps(best, sort_keys=True)}")
    print(f"Saved search report to {output_path}")


if __name__ == "__main__":
    search(get_args())
