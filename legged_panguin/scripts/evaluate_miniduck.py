"""Headless 2 m straight-line benchmark for a trained MiniDuck checkpoint."""

import argparse
import json
import math
import os
import sys

import isaacgym  # Must be imported before torch.
import torch

from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.scripts.play_miniduck import _latest_checkpoint, _load_policy_checkpoint
from legged_panguin.utils import get_args, task_registry


def _benchmark_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--distance_m", type=float, default=2.0)
    parser.add_argument("--speed_mps", type=float, default=0.10)
    parser.add_argument("--timeout_s", type=float, default=40.0)
    parser.add_argument("--output_json", type=str, default="benchmark_result.json")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _yaw_xyzw(quat):
    x, y, z, w = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def evaluate(args, bench):
    if bench.distance_m <= 0 or bench.speed_mps <= 0 or bench.timeout_s <= 0:
        raise ValueError("distance, speed and timeout must be positive")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 64
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_dof_properties = False
    env_cfg.domain_rand.curriculum_push_robots = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_action_delay = 0
    env_cfg.domain_rand.max_imu_delay = 0
    env_cfg.commands.resampling_time = bench.timeout_s + 1.0

    train_cfg.runner.resume = False
    train_cfg.runner.load_run = -1 if args.load_run in (None, "-1", -1) else args.load_run
    train_cfg.runner.checkpoint = -1
    checkpoint = _latest_checkpoint(train_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    iteration = _load_policy_checkpoint(runner, checkpoint, env.device)
    env.common_step_counter = iteration * runner.num_steps_per_env
    policy = runner.get_inference_policy(device=env.device)

    count = env.num_envs
    start_xy = env.root_states[:, :2].clone()
    start_yaw = _yaw_xyzw(env.root_states[:, 3:7]).clone()
    finished = torch.zeros(count, dtype=torch.bool, device=env.device)
    fallen = torch.zeros_like(finished)
    finish_time = torch.full((count,), bench.timeout_s, device=env.device)
    final_forward = torch.zeros(count, device=env.device)
    final_lateral = torch.zeros(count, device=env.device)
    final_yaw_error = torch.zeros(count, device=env.device)
    obs = env.get_observations()
    max_steps = math.ceil(bench.timeout_s / env.dt)

    for step in range(max_steps):
        active = ~(finished | fallen)
        env.commands[:, 0] = bench.speed_mps
        env.commands[:, 1:3] = 0.0
        previous_state = env.root_states.clone()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        state = env.root_states
        delta = state[:, :2] - start_xy
        c, s = torch.cos(start_yaw), torch.sin(start_yaw)
        forward = delta[:, 0] * c + delta[:, 1] * s
        lateral = -delta[:, 0] * s + delta[:, 1] * c
        yaw_error = torch.atan2(
            torch.sin(_yaw_xyzw(state[:, 3:7]) - start_yaw),
            torch.cos(_yaw_xyzw(state[:, 3:7]) - start_yaw),
        )
        reached = active & (forward >= bench.distance_m)
        finished |= reached
        finish_time[reached] = (step + 1) * env.dt
        final_forward[reached] = forward[reached]
        final_lateral[reached] = lateral[reached]
        final_yaw_error[reached] = yaw_error[reached]

        new_falls = active & dones.bool() & ~reached
        if new_falls.any():
            old_delta = previous_state[:, :2] - start_xy
            final_forward[new_falls] = old_delta[new_falls, 0] * c[new_falls] + old_delta[new_falls, 1] * s[new_falls]
            final_lateral[new_falls] = -old_delta[new_falls, 0] * s[new_falls] + old_delta[new_falls, 1] * c[new_falls]
            old_yaw = _yaw_xyzw(previous_state[:, 3:7]) - start_yaw
            final_yaw_error[new_falls] = torch.atan2(torch.sin(old_yaw[new_falls]), torch.cos(old_yaw[new_falls]))
        fallen |= new_falls
        if bool((finished | fallen).all()):
            break

    timed_out = ~(finished | fallen)
    if timed_out.any():
        delta = env.root_states[:, :2] - start_xy
        final_forward[timed_out] = (delta[:, 0] * torch.cos(start_yaw) + delta[:, 1] * torch.sin(start_yaw))[timed_out]
        final_lateral[timed_out] = (-delta[:, 0] * torch.sin(start_yaw) + delta[:, 1] * torch.cos(start_yaw))[timed_out]
        yaw = _yaw_xyzw(env.root_states[:, 3:7]) - start_yaw
        final_yaw_error[timed_out] = torch.atan2(torch.sin(yaw[timed_out]), torch.cos(yaw[timed_out]))

    completed = int(finished.sum().item())
    result = {
        "checkpoint": os.path.abspath(checkpoint),
        "num_trials": count,
        "distance_m": bench.distance_m,
        "command_speed_mps": bench.speed_mps,
        "completion_rate": completed / count,
        "completed": completed,
        "falls": int(fallen.sum().item()),
        "timeouts": int(timed_out.sum().item()),
        "mean_finish_time_s": float(finish_time[finished].mean().item()) if completed else None,
        "mean_finish_speed_mps": float((bench.distance_m / finish_time[finished]).mean().item()) if completed else None,
        "mean_abs_lateral_drift_m": float(final_lateral.abs().mean().item()),
        "max_abs_lateral_drift_m": float(final_lateral.abs().max().item()),
        "mean_abs_heading_error_deg": float(torch.rad2deg(final_yaw_error.abs()).mean().item()),
    }
    with open(bench.output_json, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    benchmark = _benchmark_args()
    evaluate(get_args(), benchmark)
