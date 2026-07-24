"""Eight-command benchmark for MiniDuck flexibility, stability, and accuracy."""

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


COMMANDS = (
    ("forward", (0.10, 0.0, 0.0)),
    ("backward", (-0.06, 0.0, 0.0)),
    ("forward_turn_left", (0.08, 0.0, 0.45)),
    ("forward_turn_right", (0.08, 0.0, -0.45)),
    ("left", (0.0, 0.16, 0.0)),
    ("right", (0.0, -0.16, 0.0)),
    ("turn_left", (0.0, 0.0, 0.80)),
    ("turn_right", (0.0, 0.0, -0.80)),
)


def _benchmark_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--output_json", type=str, default="multiaxis_result.json")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _yaw_xyzw(quat):
    x, y, z, w = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_pitch_xyzw(quat):
    x, y, z, w = quat.unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, torch.asin(pitch_term)


def _mean(values):
    return float(torch.mean(values).item()) if values.numel() else None


def evaluate(args, bench):
    if bench.duration_s <= 0:
        raise ValueError("duration_s must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    count = args.num_envs or 64
    if count < len(COMMANDS) or count % len(COMMANDS) != 0:
        raise ValueError(
            f"num_envs must be a positive multiple of {len(COMMANDS)}; got {count}"
        )

    env_cfg.env.num_envs = count
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
    env_cfg.commands.resampling_time = bench.duration_s + 1.0

    train_cfg.runner.resume = False
    train_cfg.runner.load_run = (
        -1 if args.load_run in (None, "-1", -1) else args.load_run
    )
    train_cfg.runner.checkpoint = -1
    checkpoint = _latest_checkpoint(train_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    iteration = _load_policy_checkpoint(runner, checkpoint, env.device)
    env.common_step_counter = iteration * runner.num_steps_per_env
    policy = runner.get_inference_policy(device=env.device)

    command_table = torch.tensor(
        [values for _, values in COMMANDS], dtype=torch.float32, device=env.device
    )
    command_ids = torch.arange(count, device=env.device) % len(COMMANDS)
    assigned_commands = command_table[command_ids]

    start_xy = env.root_states[:, :2].clone()
    start_z = env.root_states[:, 2].clone()
    start_yaw = _yaw_xyzw(env.root_states[:, 3:7]).clone()
    start_roll, start_pitch = _roll_pitch_xyzw(env.root_states[:, 3:7])
    fallen = torch.zeros(count, dtype=torch.bool, device=env.device)
    last_xy = start_xy.clone()
    last_yaw = start_yaw.clone()
    fall_time = torch.full((count,), bench.duration_s, device=env.device)

    squared_linear_error = torch.zeros(count, device=env.device)
    squared_yaw_error = torch.zeros(count, device=env.device)
    squared_roll_pitch = torch.zeros(count, device=env.device)
    squared_height_error = torch.zeros(count, device=env.device)
    summed_actual_velocity = torch.zeros(count, 3, device=env.device)
    samples = torch.zeros(count, device=env.device)

    obs = env.get_observations()
    max_steps = math.ceil(bench.duration_s / env.dt)
    for step in range(max_steps):
        active = ~fallen
        env.commands[:, :3] = assigned_commands
        previous_state = env.root_states.clone()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        valid = active & ~dones.bool()
        if valid.any():
            linear_error = env.base_lin_vel[:, :2] - assigned_commands[:, :2]
            yaw_error = env.base_ang_vel[:, 2] - assigned_commands[:, 2]
            roll, pitch = _roll_pitch_xyzw(env.root_states[:, 3:7])
            squared_linear_error[valid] += torch.sum(
                torch.square(linear_error[valid]), dim=1
            )
            squared_yaw_error[valid] += torch.square(yaw_error[valid])
            squared_roll_pitch[valid] += (
                torch.square(roll[valid] - start_roll[valid])
                + torch.square(pitch[valid] - start_pitch[valid])
            )
            squared_height_error[valid] += torch.square(
                env.root_states[valid, 2] - start_z[valid]
            )
            summed_actual_velocity[valid, :2] += env.base_lin_vel[valid, :2]
            summed_actual_velocity[valid, 2] += env.base_ang_vel[valid, 2]
            samples[valid] += 1.0
            last_xy[valid] = env.root_states[valid, :2]
            last_yaw[valid] = _yaw_xyzw(env.root_states[valid, 3:7])

        new_falls = active & dones.bool()
        if new_falls.any():
            last_xy[new_falls] = previous_state[new_falls, :2]
            last_yaw[new_falls] = _yaw_xyzw(previous_state[new_falls, 3:7])
            fall_time[new_falls] = step * env.dt
        fallen |= new_falls
        if bool(fallen.all()):
            break

    elapsed = torch.where(
        fallen, fall_time, torch.full_like(fall_time, bench.duration_s)
    )
    desired_x = torch.zeros(count, device=env.device)
    desired_y = torch.zeros(count, device=env.device)
    desired_yaw = torch.zeros(count, device=env.device)
    integration_steps = torch.ceil(elapsed / env.dt).to(torch.long)
    for step in range(max_steps):
        active = step < integration_steps
        if not bool(active.any()):
            break
        vx, vy, yaw_rate = assigned_commands.unbind(dim=1)
        desired_x[active] += (
            torch.cos(desired_yaw[active]) * vx[active]
            - torch.sin(desired_yaw[active]) * vy[active]
        ) * env.dt
        desired_y[active] += (
            torch.sin(desired_yaw[active]) * vx[active]
            + torch.cos(desired_yaw[active]) * vy[active]
        ) * env.dt
        desired_yaw[active] += yaw_rate[active] * env.dt

    delta = last_xy - start_xy
    c, s = torch.cos(start_yaw), torch.sin(start_yaw)
    actual_x = delta[:, 0] * c + delta[:, 1] * s
    actual_y = -delta[:, 0] * s + delta[:, 1] * c
    actual_yaw = torch.atan2(
        torch.sin(last_yaw - start_yaw), torch.cos(last_yaw - start_yaw)
    )
    desired_yaw = torch.atan2(torch.sin(desired_yaw), torch.cos(desired_yaw))
    position_error = torch.sqrt(
        torch.square(actual_x - desired_x) + torch.square(actual_y - desired_y)
    )
    heading_error = torch.atan2(
        torch.sin(actual_yaw - desired_yaw), torch.cos(actual_yaw - desired_yaw)
    ).abs()

    safe_samples = torch.clamp(samples, min=1.0)
    linear_rmse = torch.sqrt(squared_linear_error / safe_samples)
    yaw_rmse = torch.sqrt(squared_yaw_error / safe_samples)
    roll_pitch_rms = torch.sqrt(squared_roll_pitch / safe_samples)
    height_rmse = torch.sqrt(squared_height_error / safe_samples)
    mean_actual_velocity = summed_actual_velocity / safe_samples[:, None]

    command_results = {}
    for command_index, (name, values) in enumerate(COMMANDS):
        mask = command_ids == command_index
        survivors = mask & ~fallen
        command_results[name] = {
            "command": {
                "vx_mps": values[0],
                "vy_mps": values[1],
                "yaw_rate_rps": values[2],
            },
            "num_trials": int(mask.sum().item()),
            "survival_rate": float((~fallen[mask]).float().mean().item()),
            "falls": int(fallen[mask].sum().item()),
            "mean_linear_velocity_tracking_rmse_mps": _mean(linear_rmse[mask]),
            "mean_yaw_rate_tracking_rmse_rps": _mean(yaw_rmse[mask]),
            "mean_actual_vx_mps": _mean(mean_actual_velocity[mask, 0]),
            "mean_actual_vy_mps": _mean(mean_actual_velocity[mask, 1]),
            "mean_actual_yaw_rate_rps": _mean(mean_actual_velocity[mask, 2]),
            "mean_roll_pitch_deviation_rms_deg": _mean(
                torch.rad2deg(roll_pitch_rms[mask])
            ),
            "mean_height_rmse_m": _mean(height_rmse[mask]),
            "mean_endpoint_position_error_m_survivors": _mean(
                position_error[survivors]
            ),
            "mean_endpoint_heading_error_deg_survivors": _mean(
                torch.rad2deg(heading_error[survivors])
            ),
        }

    survivors = ~fallen
    result = {
        "checkpoint": os.path.abspath(checkpoint),
        "duration_s": bench.duration_s,
        "num_trials": count,
        "trials_per_command": count // len(COMMANDS),
        "overall": {
            "survival_rate": float(survivors.float().mean().item()),
            "falls": int(fallen.sum().item()),
            "mean_linear_velocity_tracking_rmse_mps": _mean(linear_rmse),
            "mean_yaw_rate_tracking_rmse_rps": _mean(yaw_rmse),
            "mean_roll_pitch_deviation_rms_deg": _mean(
                torch.rad2deg(roll_pitch_rms)
            ),
            "mean_height_rmse_m": _mean(height_rmse),
            "mean_endpoint_position_error_m_survivors": _mean(
                position_error[survivors]
            ),
            "mean_endpoint_heading_error_deg_survivors": _mean(
                torch.rad2deg(heading_error[survivors])
            ),
        },
        "commands": command_results,
    }
    with open(bench.output_json, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    benchmark = _benchmark_args()
    evaluate(get_args(), benchmark)
