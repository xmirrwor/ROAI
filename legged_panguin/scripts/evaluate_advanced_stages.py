"""Evaluate advanced MiniDuck skills with raw, unsmoothed trajectories."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import isaacgym  # Must be imported before torch.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from legged_panguin import LEGGED_GYM_ROOT_DIR
from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.scripts.play_miniduck import _load_policy_checkpoint
from legged_panguin.utils import get_args, task_registry
from legged_panguin.utils.helpers import get_load_path


def _extra_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--protocol",
        choices=("fall_recovery", "diagonal_motion", "obstacle_crossing", "ball_kick"),
        required=True,
    )
    parser.add_argument("--duration_s", type=float, default=5.0)
    parser.add_argument("--output_dir", default="evaluation/advanced_stages")
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--randomized", action="store_true")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _yaw(quat):
    x, y, z, w = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _percentile(value, q=0.95):
    return float(torch.quantile(value, q).item())


def _make_env(args, cfg):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 128
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 4 if cfg.protocol == "obstacle_crossing" else 1
    env_cfg.terrain.curriculum = False
    stage_by_protocol = {
        "fall_recovery": 3,
        "diagonal_motion": 4,
        "obstacle_crossing": 5,
        "ball_kick": 6,
    }
    env_cfg.skill_curriculum.forced_stage = stage_by_protocol[cfg.protocol]
    if cfg.protocol == "obstacle_crossing":
        env_cfg.terrain.mesh_type = "heightfield"
        env_cfg.terrain.obstacle_course = True
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.scene.obstacle_enabled = True
    elif cfg.protocol == "ball_kick":
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.obstacle_course = False
        env_cfg.scene.ball_enabled = True
    env_cfg.noise.add_noise = cfg.randomized
    env_cfg.domain_rand.curriculum_push_robots = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_action_delay = 0
    env_cfg.domain_rand.max_imu_delay = 0
    env_cfg.commands.resampling_time = 1000.0
    if not cfg.randomized:
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
    train_cfg.runner.load_run = -1 if args.load_run in (None, "-1", -1) else args.load_run
    root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    checkpoint = get_load_path(
        root,
        load_run=train_cfg.runner.load_run,
        checkpoint=-1 if args.checkpoint in (None, -1) else args.checkpoint,
    )
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    iteration = _load_policy_checkpoint(runner, checkpoint, env.device)
    env.common_step_counter = iteration * runner.num_steps_per_env
    env.reset_idx(torch.arange(env.num_envs, device=env.device, dtype=torch.long))
    env.compute_observations()
    return env, runner.get_inference_policy(device=env.device), checkpoint, iteration


def _write_outputs(cfg, result, trajectory, trials, panels):
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_randomized" if cfg.randomized else "_deterministic"
    stem = result["run_label"] + suffix
    metrics = output / f"{stem}_metrics.json"
    trajectory_path = output / f"{stem}_trajectory.csv"
    trials_path = output / f"{stem}_trials.csv"
    figure = output / f"{stem}_trajectory.png"
    metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for path, rows in ((trajectory_path, trajectory), (trials_path, trials)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        time_s = [row["time_s"] for row in trajectory]
        for ax, (key, ylabel, title), color in zip(axes.flat, panels, colors):
            ax.plot(time_s, [row[key] for row in trajectory], color=color)
            ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
            ax.grid(True, color="#D9D9D9", linewidth=0.6)
        fig.suptitle(
            f"MiniDuck {result['protocol_name']} (n={result['num_trials']}, "
            f"pass={result['acceptance_passed']})"
        )
        fig.savefig(figure, dpi=180, facecolor="white")
        plt.close(fig)
    print(json.dumps(result, indent=2))
    print(f"metrics={metrics}\ntrajectory={trajectory_path}\ntrials={trials_path}\nfigure={figure}")


def _evaluate_recovery(args, cfg, env, policy, checkpoint, iteration):
    count = env.num_envs
    timeout_steps = max(1, round(cfg.duration_s / env.dt))
    target = env.target_projected_gravity.expand_as(env.projected_gravity)
    initial_alignment = torch.sum(env.projected_gravity * target, dim=1).clone()
    initial_pitch_sign = torch.sign(env.projected_gravity[:, 0]).clone()
    success = torch.zeros(count, dtype=torch.bool, device=env.device)
    finished = torch.zeros_like(success)
    success_time = torch.full((count,), float("nan"), device=env.device)
    max_height = env.root_states[:, 2].clone()
    max_alignment = initial_alignment.clone()
    trajectory = []
    obs = env.get_observations()
    for step in range(timeout_steps):
        active = ~finished
        env.commands[:, :3] = 0.0
        env.skill_mode[:] = env.SKILL_RECOVERY
        obs[:, 9:12] = 0.0
        obs[:, 62:64] = env._skill_observation()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        alignment = torch.sum(env.projected_gravity * target, dim=1)
        height = env.root_states[:, 2]
        max_height = torch.where(active, torch.maximum(max_height, height), max_height)
        max_alignment = torch.where(active, torch.maximum(max_alignment, alignment), max_alignment)
        just_finished = active & dones.bool()
        just_success = just_finished & env.time_out_buf.bool()
        success |= just_success
        success_time[just_success] = (step + 1) * env.dt
        finished |= just_finished
        trajectory.append({
            "time_s": round((step + 1) * env.dt, 6),
            "mean_alignment": float(alignment[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_height_m": float(height[active].mean().item()) if torch.any(active) else float("nan"),
            "success_fraction": float(success.float().mean().item()),
            "active_fraction": float(active.float().mean().item()),
        })
        if torch.all(finished):
            break
    trials = [{
        "trial": index,
        "pose": "prone" if initial_pitch_sign[index] >= 0 else "supine",
        "initial_alignment": float(initial_alignment[index].item()),
        "max_alignment": float(max_alignment[index].item()),
        "max_height_m": float(max_height[index].item()),
        "success": bool(success[index].item()),
        "success_time_s": float(success_time[index].item()),
    } for index in range(count)]
    rate = float(success.float().mean().item())
    valid_times = success_time[success]
    result = {
        "run_label": cfg.run_label or f"fall_recovery_{iteration}",
        "protocol_name": "fall_recovery",
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "randomized": cfg.randomized,
        "success_rate": rate,
        "mean_success_time_s": float(valid_times.mean().item()) if len(valid_times) else None,
        "acceptance": {"success_rate_at_least_50pct": rate >= 0.50},
        "acceptance_passed": rate >= 0.50,
        "provenance": {"aggregation": "raw per-step means and per-trial outcomes; no smoothing", "failed_trials": "retained"},
    }
    _write_outputs(cfg, result, trajectory, trials, (
        ("mean_alignment", "Dot product", "Upright alignment"),
        ("mean_height_m", "Height (m)", "Base height"),
        ("success_fraction", "Fraction", "Cumulative recovery success"),
        ("active_fraction", "Fraction", "Trials still active"),
    ))


def _evaluate_diagonal(args, cfg, env, policy, checkpoint, iteration):
    count = env.num_envs
    commands = torch.tensor(
        ((0.10, 0.08), (0.10, -0.08), (-0.08, 0.08), (-0.08, -0.08)),
        device=env.device,
    )
    env.commands[:, :2] = commands[torch.arange(count, device=env.device) % 4]
    env.commands[:, 2] = 0.0
    env.skill_mode[:] = env.SKILL_LOCOMOTION
    start_xy = env.root_states[:, :2].clone()
    start_yaw = _yaw(env.root_states[:, 3:7]).clone()
    env.command_start_xy[:] = start_xy
    env.command_heading[:] = start_yaw
    env.line_reference_active[:] = True
    commanded = env.commands[:, :2].clone()
    unit = commanded / torch.norm(commanded, dim=1, keepdim=True)
    perpendicular = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
    fallen = torch.zeros(count, dtype=torch.bool, device=env.device)
    max_cross = torch.zeros(count, device=env.device)
    max_heading = torch.zeros(count, device=env.device)
    steps = max(1, round(cfg.duration_s / env.dt))
    trajectory = []
    obs = env.get_observations()
    for step in range(steps):
        env.commands[:, :2] = commanded
        env._apply_diagonal_heading_hold()
        obs[:, 9:12] = env._command_observation()
        obs[:, 62:64] = env._skill_observation()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        fallen |= dones.bool()
        delta = env.root_states[:, :2] - start_xy
        progress = torch.sum(delta * unit, dim=1)
        cross = torch.abs(torch.sum(delta * perpendicular, dim=1))
        heading = torch.abs(torch.atan2(
            torch.sin(_yaw(env.root_states[:, 3:7]) - start_yaw),
            torch.cos(_yaw(env.root_states[:, 3:7]) - start_yaw),
        ))
        max_cross = torch.maximum(max_cross, cross)
        max_heading = torch.maximum(max_heading, heading)
        active = ~fallen
        directed_speed = torch.sum(env.base_lin_vel[:, :2] * unit, dim=1)
        trajectory.append({
            "time_s": round((step + 1) * env.dt, 6),
            "mean_progress_m": float(progress[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_directed_speed_mps": float(directed_speed[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_cross_track_m": float(cross[active].mean().item()) if torch.any(active) else float("nan"),
            "fall_fraction": float(fallen.float().mean().item()),
        })
    delta = env.root_states[:, :2] - start_xy
    progress = torch.sum(delta * unit, dim=1)
    mean_speed = progress / cfg.duration_s
    trials = [{
        "trial": index,
        "command_vx_mps": float(commanded[index, 0].item()),
        "command_vy_mps": float(commanded[index, 1].item()),
        "progress_m": float(progress[index].item()),
        "mean_directed_speed_mps": float(mean_speed[index].item()),
        "max_cross_track_m": float(max_cross[index].item()),
        "max_heading_deg": math.degrees(float(max_heading[index].item())),
        "fell": bool(fallen[index].item()),
    } for index in range(count)]
    fall_rate = float(fallen.float().mean().item())
    speed = float(mean_speed[~fallen].mean().item()) if torch.any(~fallen) else 0.0
    acceptance = {
        "fall_rate_at_most_15pct": fall_rate <= 0.15,
        "mean_directed_speed_at_least_0p04": speed >= 0.04,
        "p95_cross_track_at_most_0p12": _percentile(max_cross) <= 0.12,
        "p95_heading_at_most_20deg": math.degrees(_percentile(max_heading)) <= 20.0,
    }
    result = {
        "run_label": cfg.run_label or f"diagonal_motion_{iteration}",
        "protocol_name": "diagonal_motion",
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "duration_s": cfg.duration_s,
        "randomized": cfg.randomized,
        "fall_rate": fall_rate,
        "mean_directed_speed_mps": speed,
        "p95_max_cross_track_m": _percentile(max_cross),
        "p95_max_heading_deg": math.degrees(_percentile(max_heading)),
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "provenance": {"aggregation": "raw per-step means and per-trial outcomes; no smoothing", "failed_trials": "retained"},
    }
    _write_outputs(cfg, result, trajectory, trials, (
        ("mean_progress_m", "Progress (m)", "Command-axis displacement"),
        ("mean_directed_speed_mps", "Speed (m/s)", "Directed velocity"),
        ("mean_cross_track_m", "Deviation (m)", "Path deviation"),
        ("fall_fraction", "Fraction", "Falls"),
    ))


def _evaluate_obstacle(args, cfg, env, policy, checkpoint, iteration):
    count = env.num_envs
    obstacle_cfg = env.cfg.skill_curriculum
    duration_s = max(cfg.duration_s, obstacle_cfg.obstacle_timeout_s)
    steps = max(1, round(duration_s / env.dt))
    command = torch.full((count,), 0.12, device=env.device)
    env.commands[:, :3] = 0.0
    env.commands[:, 0] = command
    env.skill_mode[:] = env.SKILL_OBSTACLE
    env.obstacle_task_active[:] = True
    env.obstacle_elapsed_steps[:] = 0
    env.obstacle_success_latched[:] = False
    start_x = env.obstacle_world_x - obstacle_cfg.obstacle_approach_distance_m
    success = torch.zeros(count, dtype=torch.bool, device=env.device)
    fallen = torch.zeros_like(success)
    finished = torch.zeros_like(success)
    success_time = torch.full((count,), float("nan"), device=env.device)
    max_progress = torch.zeros(count, device=env.device)
    max_clearance = torch.full((count,), -1.0, device=env.device)
    body_collision = torch.zeros_like(success)
    trajectory = []
    obs = env.get_observations()
    target_gravity = env.target_projected_gravity.expand_as(env.projected_gravity)
    with torch.no_grad():
        initial_action = policy(obs.detach())
    initial_diagnostics = {
        "mean_base_height_m": float(env.root_states[:, 2].mean().item()),
        "mean_origin_height_m": float(env.env_origins[:, 2].mean().item()),
        "mean_gravity_alignment": float(torch.sum(
            env.projected_gravity * target_gravity, dim=1
        ).mean().item()),
        "mean_obstacle_feature_0": float(obs[:, 62].mean().item()),
        "mean_obstacle_feature_1": float(obs[:, 63].mean().item()),
        "max_abs_initial_action": float(torch.max(torch.abs(initial_action)).item()),
    }
    print("Obstacle initial diagnostics:")
    print(json.dumps(initial_diagnostics, indent=2))
    for step in range(steps):
        active = ~finished
        env.commands[:, :3] = 0.0
        env.commands[:, 0] = command
        env.skill_mode[:] = env.SKILL_OBSTACLE
        env.obstacle_task_active[:] = active
        env._apply_straight_heading_hold()
        obs[:, 9:12] = env.commands[:, :3] * env.commands_scale
        obs[:, 62:64] = env._skill_observation()
        foot_height = env.rigid_body_state[:, env.feet_indices, 2]
        clearance = torch.max(foot_height, dim=1).values - (
            env.env_origins[:, 2] + env.obstacle_heights
        )
        max_clearance = torch.where(
            active, torch.maximum(max_clearance, clearance), max_clearance
        )
        body_force = torch.norm(
            env.contact_forces[:, env.obstacle_body_indices], dim=-1
        )
        body_collision |= active & torch.any(body_force > 5.0, dim=1)
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        progress = env.obstacle_progress_latched.clone()
        max_progress = torch.where(
            active, torch.maximum(max_progress, progress), max_progress
        )
        just_finished = active & dones.bool()
        just_success = just_finished & env.obstacle_success_latched.bool()
        success |= just_success
        success_time[just_success] = (step + 1) * env.dt
        # Before the common timeout, a non-success termination is a fall.
        if step + 1 < steps:
            fallen |= just_finished & ~just_success
        finished |= just_finished
        still_active = ~finished
        trajectory.append({
            "time_s": round((step + 1) * env.dt, 6),
            "mean_progress_m": float(max_progress[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_max_foot_clearance_m": float(max_clearance[active].mean().item()) if torch.any(active) else float("nan"),
            "success_fraction": float(success.float().mean().item()),
            "fall_fraction": float(fallen.float().mean().item()),
        })
        if not torch.any(still_active):
            break
    trials = [{
        "trial": index,
        "obstacle_height_m": float(env.obstacle_heights[index].item()),
        "max_progress_m": float(max_progress[index].item()),
        "max_foot_clearance_m": float(max_clearance[index].item()),
        "body_collision": bool(body_collision[index].item()),
        "success": bool(success[index].item()),
        "fell": bool(fallen[index].item()),
        "success_time_s": float(success_time[index].item()),
    } for index in range(count)]
    rate = float(success.float().mean().item())
    fall_rate = float(fallen.float().mean().item())
    collision_rate = float(body_collision.float().mean().item())
    valid_times = success_time[success]
    mean_progress = float(max_progress.mean().item())
    acceptance = {
        "success_rate_at_least_25pct": rate >= 0.25,
        "fall_rate_at_most_50pct": fall_rate <= 0.50,
        "mean_progress_at_least_0p30m": mean_progress >= 0.30,
    }
    result = {
        "run_label": cfg.run_label or f"obstacle_crossing_{iteration}",
        "protocol_name": "obstacle_crossing",
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "duration_s": duration_s,
        "randomized": cfg.randomized,
        "success_rate": rate,
        "fall_rate": fall_rate,
        "body_collision_rate": collision_rate,
        "mean_max_progress_m": mean_progress,
        "mean_max_foot_clearance_m": float(max_clearance.mean().item()),
        "mean_success_time_s": float(valid_times.mean().item()) if len(valid_times) else None,
        "initial_diagnostics": initial_diagnostics,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "provenance": {
            "aggregation": "raw per-step means and per-trial outcomes; no smoothing",
            "failed_trials": "retained",
            "obstacle": "fixed transverse ridge; four deterministic height groups",
        },
    }
    _write_outputs(cfg, result, trajectory, trials, (
        ("mean_progress_m", "Progress (m)", "Maximum forward progress"),
        ("mean_max_foot_clearance_m", "Clearance (m)", "Maximum foot clearance"),
        ("success_fraction", "Fraction", "Cumulative crossing success"),
        ("fall_fraction", "Fraction", "Falls"),
    ))


def _evaluate_ball(args, cfg, env, policy, checkpoint, iteration):
    count = env.num_envs
    ball_cfg = env.cfg.skill_curriculum
    steps = max(1, round(max(cfg.duration_s, ball_cfg.ball_timeout_s) / env.dt))
    success = torch.zeros(count, dtype=torch.bool, device=env.device)
    fallen = torch.zeros_like(success)
    finished = torch.zeros_like(success)
    max_progress = torch.zeros(count, device=env.device)
    max_speed = torch.zeros(count, device=env.device)
    closest_distance = torch.full((count,), float("inf"), device=env.device)
    closest_foot_distance = torch.full((count,), float("inf"), device=env.device)
    start_x = env.ball_start_x.clone()
    initial_foot_offset = (
        env.rigid_body_state[:, env.feet_indices, :2]
        - env.root_states[:, None, :2]
    ).clone()
    trajectory = []
    obs = env.get_observations()
    for step in range(steps):
        active = ~finished
        env.commands[:, :3] = 0.0
        env.commands[:, 0] = ball_cfg.ball_approach_speed_mps
        env.skill_mode[:] = env.SKILL_KICK
        env.ball_task_active[:] = active
        env._apply_straight_heading_hold()
        obs[:, 9:12] = env._command_observation()
        obs[:, 62:64] = env._skill_observation()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        progress = env.ball_progress_latched.clone()
        speed = torch.clamp(env.ball_root_states[:, 7], min=0.0)
        relative = env._ball_relative_body()
        distance = torch.norm(relative[:, :2], dim=1)
        foot_distance = torch.min(torch.norm(
            env.rigid_body_state[:, env.feet_indices, :2]
            - env.ball_root_states[:, None, :2],
            dim=-1,
        ), dim=1).values
        closest_distance = torch.where(active, torch.minimum(closest_distance, distance), closest_distance)
        closest_foot_distance = torch.where(active, torch.minimum(closest_foot_distance, foot_distance), closest_foot_distance)
        max_progress = torch.where(active, torch.maximum(max_progress, progress), max_progress)
        max_speed = torch.where(active, torch.maximum(max_speed, speed), max_speed)
        just_finished = active & dones.bool()
        just_success = just_finished & env.ball_success_latched.bool()
        success |= just_success
        if step + 1 < steps:
            fallen |= just_finished & ~just_success
        finished |= just_finished
        trajectory.append({
            "time_s": round((step + 1) * env.dt, 6),
            "mean_ball_progress_m": float(max_progress[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_ball_speed_mps": float(speed[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_robot_ball_distance_m": float(distance[active].mean().item()) if torch.any(active) else float("nan"),
            "mean_closest_robot_ball_distance_m": float(closest_distance[active].mean().item()) if torch.any(active) else float("nan"),
            "success_fraction": float(success.float().mean().item()),
            "fall_fraction": float(fallen.float().mean().item()),
        })
        if torch.all(finished):
            break
    trials = [{
        "trial": index,
        "initial_ball_x_m": float(start_x[index].item()),
        "max_ball_progress_m": float(max_progress[index].item()),
        "max_ball_speed_mps": float(max_speed[index].item()),
        "closest_robot_ball_distance_m": float(closest_distance[index].item()),
        "closest_foot_ball_distance_m": float(closest_foot_distance[index].item()),
        "success": bool(success[index].item()),
        "fell": bool(fallen[index].item()),
    } for index in range(count)]
    rate = float(success.float().mean().item())
    fall_rate = float(fallen.float().mean().item())
    mean_progress = float(max_progress.mean().item())
    phase = ball_cfg.ball_phase
    required_rate = 0.50 if phase == "approach" else 0.25
    acceptance = {
        f"success_rate_at_least_{int(required_rate * 100)}pct": rate >= required_rate,
        "fall_rate_at_most_25pct": fall_rate <= 0.25,
    }
    if phase != "approach":
        acceptance["mean_progress_reaches_goal_line"] = (
            mean_progress >= ball_cfg.ball_goal_distance_m
        )
    result = {
        "run_label": cfg.run_label or f"ball_{phase}_{iteration}",
        "protocol_name": f"ball_kick_{phase}",
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "randomized": cfg.randomized,
        "success_rate": rate,
        "goal_success_rate": rate if phase != "approach" else None,
        "fall_rate": fall_rate,
        "mean_max_ball_progress_m": mean_progress,
        "mean_max_ball_speed_mps": float(max_speed.mean().item()),
        "mean_closest_robot_ball_distance_m": float(closest_distance.mean().item()),
        "median_closest_robot_ball_distance_m": _percentile(closest_distance, 0.50),
        "median_closest_foot_ball_distance_m": _percentile(closest_foot_distance, 0.50),
        "goal_line_distance_m": (
            ball_cfg.ball_goal_distance_m if phase != "approach" else None
        ),
        "goal_width_m": (
            env.cfg.scene.goal_width_m if phase != "approach" else None
        ),
        "initial_left_foot_offset_xy_m": [
            float(initial_foot_offset[:, 0, axis].mean().item()) for axis in range(2)
        ],
        "initial_right_foot_offset_xy_m": [
            float(initial_foot_offset[:, 1, axis].mean().item()) for axis in range(2)
        ],
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "provenance": {"aggregation": "raw per-step means and per-trial outcomes; no smoothing", "failed_trials": "retained"},
    }
    panels = (
        (
            ("mean_robot_ball_distance_m", "Distance (m)", "Current robot-ball distance"),
            ("mean_closest_robot_ball_distance_m", "Distance (m)", "Closest robot-ball distance"),
            ("success_fraction", "Fraction", "Cumulative success"),
            ("fall_fraction", "Fraction", "Falls"),
        )
        if phase == "approach"
        else (
            ("mean_ball_progress_m", "Progress (m)", "Ball forward displacement"),
            ("mean_ball_speed_mps", "Speed (m/s)", "Ball forward speed"),
            ("success_fraction", "Fraction", "Cumulative success"),
            ("fall_fraction", "Fraction", "Falls"),
        )
    )
    _write_outputs(cfg, result, trajectory, trials, panels)


if __name__ == "__main__":
    config = _extra_args()
    base_args = get_args()
    environment, inference_policy, checkpoint_path, checkpoint_iteration = _make_env(base_args, config)
    if config.protocol == "fall_recovery":
        _evaluate_recovery(base_args, config, environment, inference_policy, checkpoint_path, checkpoint_iteration)
    elif config.protocol == "diagonal_motion":
        _evaluate_diagonal(base_args, config, environment, inference_policy, checkpoint_path, checkpoint_iteration)
    elif config.protocol == "obstacle_crossing":
        _evaluate_obstacle(base_args, config, environment, inference_policy, checkpoint_path, checkpoint_iteration)
    else:
        _evaluate_ball(base_args, config, environment, inference_policy, checkpoint_path, checkpoint_iteration)
