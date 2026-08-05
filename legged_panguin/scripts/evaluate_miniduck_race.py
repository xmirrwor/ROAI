"""Evaluate a MiniDuck checkpoint on a straight 2 m race course."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import isaacgym  # Must be imported before torch.
import torch

from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.scripts.play_miniduck import _load_policy_checkpoint
from legged_panguin.utils import get_args, task_registry


def _race_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--motion",
        choices=("forward", "diagonal", "lateral"),
        required=True,
    )
    parser.add_argument("--command_speed_mps", type=float, required=True)
    parser.add_argument("--direction_sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--distance_m", type=float, default=2.0)
    parser.add_argument("--timeout_s", type=float, default=30.0)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", default="evaluation/race_2m")
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--randomized", action="store_true")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _yaw_xyzw(quat):
    x, y, z, w = quat.unbind(dim=1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _command_for_motion(motion, speed, direction_sign, device):
    speed *= direction_sign
    if motion == "forward":
        values = (speed, 0.0)
    elif motion == "diagonal":
        component = speed / math.sqrt(2.0)
        values = (component, component)
    else:
        values = (0.0, speed)
    return torch.tensor(values, device=device)


def _percentile(values, quantile=0.95):
    return float(torch.quantile(values, quantile).item())


def evaluate(args, race):
    if race.command_speed_mps <= 0.0:
        raise ValueError("command_speed_mps must be positive")
    if race.distance_m <= 0.0 or race.timeout_s <= 0.0:
        raise ValueError("distance_m and timeout_s must be positive")

    checkpoint = os.path.abspath(race.checkpoint_path)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 64
    env_cfg.env.episode_length_s = race.timeout_s + 2.0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.scene.obstacle_enabled = False
    env_cfg.scene.ball_enabled = False
    env_cfg.skill_curriculum.forced_stage = (
        0 if race.motion == "forward" else 4
    )
    env_cfg.commands.resampling_time = race.timeout_s + 2.0
    env_cfg.noise.add_noise = race.randomized
    env_cfg.domain_rand.curriculum_push_robots = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_action_delay = 0
    env_cfg.domain_rand.max_imu_delay = 0
    if not race.randomized:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_link_mass = False
        env_cfg.domain_rand.randomize_dof_properties = False
        env_cfg.domain_rand.motor_strength_range = [1.0, 1.0]
        env_cfg.domain_rand.kp_factor_range = [1.0, 1.0]
        env_cfg.domain_rand.kd_factor_range = [1.0, 1.0]
        env_cfg.domain_rand.motor_velocity_range = [5.24, 5.24]
        env_cfg.domain_rand.joint_target_offset = 0.0
        env_cfg.domain_rand.encoder_offset = 0.0
        env_cfg.domain_rand.gyro_bias = 0.0
        env_cfg.domain_rand.gravity_bias = 0.0
        env_cfg.domain_rand.accel_bias = 0.0

    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    iteration = _load_policy_checkpoint(runner, checkpoint, env.device)
    env.common_step_counter = iteration * runner.num_steps_per_env
    env.reset_idx(
        torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    )
    env.compute_observations()
    policy = runner.get_inference_policy(device=env.device)

    count = env.num_envs
    command = _command_for_motion(
        race.motion,
        race.command_speed_mps,
        race.direction_sign,
        env.device,
    )
    command_batch = command.unsqueeze(0).expand(count, -1)
    start_xy = env.root_states[:, :2].clone()
    start_yaw = _yaw_xyzw(env.root_states[:, 3:7]).clone()
    body_angle = math.atan2(float(command[1]), float(command[0]))
    path_heading = start_yaw + body_angle
    path_unit = torch.stack(
        (torch.cos(path_heading), torch.sin(path_heading)),
        dim=1,
    )
    perpendicular = torch.stack((-path_unit[:, 1], path_unit[:, 0]), dim=1)

    env.command_start_xy[:] = start_xy
    env.command_heading[:] = start_yaw
    env.line_reference_active[:] = True
    env.skill_mode[:] = env.SKILL_LOCOMOTION
    finished = torch.zeros(count, dtype=torch.bool, device=env.device)
    fallen = torch.zeros_like(finished)
    finish_time = torch.full(
        (count,),
        race.timeout_s,
        device=env.device,
    )
    final_progress = torch.zeros(count, device=env.device)
    final_cross_track = torch.zeros(count, device=env.device)
    final_heading_error = torch.zeros(count, device=env.device)
    max_cross_track = torch.zeros(count, device=env.device)
    max_tilt = torch.zeros(count, device=env.device)
    last_valid_xy = start_xy.clone()
    path_length = torch.zeros(count, device=env.device)
    trajectory_xy = [start_xy.clone()]
    obs = env.get_observations()
    max_steps = math.ceil(race.timeout_s / env.dt)

    for step in range(max_steps):
        active = ~(finished | fallen)
        env.commands[:, :3] = 0.0
        env.commands[:, :2] = command_batch
        if race.motion == "forward":
            env._apply_straight_heading_hold()
        else:
            yaw = _yaw_xyzw(env.root_states[:, 3:7])
            heading_error = torch.atan2(
                torch.sin(yaw - start_yaw),
                torch.cos(yaw - start_yaw),
            )
            correction = (
                -env.cfg.skill_curriculum.heading_hold_kp * heading_error
                - env.cfg.skill_curriculum.heading_hold_kd
                * env.base_ang_vel[:, 2]
            )
            env.commands[:, 2] = torch.clamp(
                correction,
                -env.cfg.skill_curriculum.heading_hold_max_yaw_rate,
                env.cfg.skill_curriculum.heading_hold_max_yaw_rate,
            )
        obs[:, 9:12] = env._command_observation()
        obs[:, 62:64] = env._skill_observation()
        previous_state = env.root_states.clone()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        state = env.root_states
        done_now = active & dones.bool()
        valid_xy = torch.where(
            done_now.unsqueeze(1),
            previous_state[:, :2],
            state[:, :2],
        )
        path_length += torch.where(
            active,
            torch.norm(valid_xy - last_valid_xy, dim=1),
            torch.zeros_like(path_length),
        )
        last_valid_xy = torch.where(
            active.unsqueeze(1),
            valid_xy,
            last_valid_xy,
        )
        trajectory_xy.append(last_valid_xy.clone())
        delta = last_valid_xy - start_xy
        progress = torch.sum(delta * path_unit, dim=1)
        cross_track = torch.sum(delta * perpendicular, dim=1)
        yaw = _yaw_xyzw(state[:, 3:7])
        previous_yaw = _yaw_xyzw(previous_state[:, 3:7])
        yaw = torch.where(done_now, previous_yaw, yaw)
        heading_error = torch.atan2(
            torch.sin(yaw - start_yaw),
            torch.cos(yaw - start_yaw),
        )
        roll, pitch, _ = env._quat_to_euler_xyz(state[:, 3:7]) if hasattr(env, "_quat_to_euler_xyz") else (None, None, None)
        if roll is None:
            gravity_tilt = torch.acos(
                torch.clamp(-env.projected_gravity[:, 2], -1.0, 1.0)
            )
        else:
            gravity_tilt = torch.maximum(torch.abs(roll), torch.abs(pitch))
        max_cross_track = torch.where(
            active,
            torch.maximum(max_cross_track, torch.abs(cross_track)),
            max_cross_track,
        )
        max_tilt = torch.where(
            active,
            torch.maximum(max_tilt, gravity_tilt),
            max_tilt,
        )

        net_distance = torch.norm(delta, dim=1)
        reached = active & (net_distance >= race.distance_m)
        finished |= reached
        finish_time[reached] = (step + 1) * env.dt
        final_progress[reached] = progress[reached]
        final_cross_track[reached] = cross_track[reached]
        final_heading_error[reached] = heading_error[reached]

        new_falls = done_now & ~reached
        if new_falls.any():
            old_delta = previous_state[:, :2] - start_xy
            final_progress[new_falls] = torch.sum(
                old_delta[new_falls] * path_unit[new_falls],
                dim=1,
            )
            final_cross_track[new_falls] = torch.sum(
                old_delta[new_falls] * perpendicular[new_falls],
                dim=1,
            )
            old_yaw = _yaw_xyzw(previous_state[:, 3:7])
            final_heading_error[new_falls] = torch.atan2(
                torch.sin(old_yaw[new_falls] - start_yaw[new_falls]),
                torch.cos(old_yaw[new_falls] - start_yaw[new_falls]),
            )
        fallen |= new_falls
        if bool((finished | fallen).all()):
            break

    timed_out = ~(finished | fallen)
    if timed_out.any():
        delta = last_valid_xy - start_xy
        final_progress[timed_out] = torch.sum(
            delta[timed_out] * path_unit[timed_out],
            dim=1,
        )
        final_cross_track[timed_out] = torch.sum(
            delta[timed_out] * perpendicular[timed_out],
            dim=1,
        )
        yaw = _yaw_xyzw(env.root_states[:, 3:7])
        final_heading_error[timed_out] = torch.atan2(
            torch.sin(yaw[timed_out] - start_yaw[timed_out]),
            torch.cos(yaw[timed_out] - start_yaw[timed_out]),
        )

    final_delta = last_valid_xy - start_xy
    net_distance = torch.norm(final_delta, dim=1)
    chord_unit = final_delta / torch.clamp(
        net_distance.unsqueeze(1),
        min=1.0e-6,
    )
    chord_perpendicular = torch.stack(
        (-chord_unit[:, 1], chord_unit[:, 0]),
        dim=1,
    )
    relative_trajectory = torch.stack(trajectory_xy) - start_xy.unsqueeze(0)
    chord_deviation = torch.max(
        torch.abs(
            torch.sum(
                relative_trajectory * chord_perpendicular.unsqueeze(0),
                dim=2,
            )
        ),
        dim=0,
    ).values
    path_efficiency = net_distance / torch.clamp(path_length, min=1.0e-6)

    completed = int(finished.sum().item())
    completion_rate = completed / count
    fall_rate = float(fallen.float().mean().item())
    mean_finish_time = (
        float(finish_time[finished].mean().item()) if completed else None
    )
    mean_finish_speed = (
        float((race.distance_m / finish_time[finished]).mean().item())
        if completed
        else 0.0
    )
    mean_cross = float(max_cross_track.mean().item())
    mean_chord_deviation = float(chord_deviation.mean().item())
    race_score = (
        1000.0 * completion_rate
        + 100.0 * mean_finish_speed
        - 200.0 * fall_rate
        - 10.0 * mean_chord_deviation
    )
    acceptance = {
        "completion_rate_at_least_85pct": completion_rate >= 0.85,
        "fall_rate_at_most_5pct": fall_rate <= 0.05,
        "mean_chord_deviation_at_most_0p10m": mean_chord_deviation <= 0.10,
        "p05_path_efficiency_at_least_0p90": _percentile(
            path_efficiency,
            0.05,
        )
        >= 0.90,
        "p95_heading_error_at_most_15deg": math.degrees(
            _percentile(torch.abs(final_heading_error))
        )
        <= 15.0,
    }
    result = {
        "run_label": race.run_label
        or f"{race.motion}_{iteration}_{race.command_speed_mps:.3f}",
        "motion": race.motion,
        "checkpoint": checkpoint,
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "randomized": race.randomized,
        "distance_m": race.distance_m,
        "timeout_s": race.timeout_s,
        "command_speed_mps": race.command_speed_mps,
        "direction_sign": race.direction_sign,
        "command_vx_mps": float(command[0]),
        "command_vy_mps": float(command[1]),
        "completed": completed,
        "completion_rate": completion_rate,
        "falls": int(fallen.sum().item()),
        "fall_rate": fall_rate,
        "timeouts": int(timed_out.sum().item()),
        "mean_finish_time_s": mean_finish_time,
        "p95_finish_time_s": (
            _percentile(finish_time[finished]) if completed else None
        ),
        "mean_finish_speed_mps": mean_finish_speed,
        "mean_net_distance_m": float(net_distance.mean().item()),
        "mean_path_length_m": float(path_length.mean().item()),
        "mean_path_efficiency": float(path_efficiency.mean().item()),
        "p05_path_efficiency": _percentile(path_efficiency, 0.05),
        "mean_chord_deviation_m": mean_chord_deviation,
        "p95_chord_deviation_m": _percentile(chord_deviation),
        "mean_command_axis_progress_m": float(final_progress.mean().item()),
        "mean_command_axis_max_cross_track_m": mean_cross,
        "p95_command_axis_max_cross_track_m": _percentile(max_cross_track),
        "p95_heading_error_deg": math.degrees(
            _percentile(torch.abs(final_heading_error))
        ),
        "p95_max_tilt_deg": math.degrees(_percentile(max_tilt)),
        "race_score": race_score,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
    }
    output = Path(race.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{result['run_label']}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output={output_path}")


if __name__ == "__main__":
    race_args = _race_args()
    evaluate(get_args(), race_args)
