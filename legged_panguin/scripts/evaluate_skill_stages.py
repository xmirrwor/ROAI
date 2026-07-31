"""Evaluate MiniDuck squat and action-switch checkpoints with raw trajectories."""

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
from legged_panguin.scripts.play_miniduck import (
    _load_policy_checkpoint,
    _symmetric_policy_action,
)
from legged_panguin.utils import get_args, task_registry
from legged_panguin.utils.helpers import get_load_path


def _evaluation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", choices=("squat", "action_switch"), required=True)
    parser.add_argument("--segment_s", type=float, default=2.0)
    parser.add_argument("--stationary_s", type=float, default=2.0)
    parser.add_argument("--forward_speed", type=float, default=0.08)
    parser.add_argument("--backward_speed", type=float, default=-0.06)
    parser.add_argument("--output_dir", default="evaluation/skill_stages")
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--symmetric_inference", action="store_true")
    parser.add_argument("--symmetry_blend", type=float, default=0.0)
    parser.add_argument("--cross_track_heading_kp", type=float, default=None)
    parser.add_argument("--heading_hold_kp", type=float, default=None)
    parser.add_argument("--heading_hold_kd", type=float, default=None)
    parser.add_argument("--heading_hold_max", type=float, default=None)
    parser.add_argument("--line_hold_max", type=float, default=None)
    parser.add_argument("--locomotion_checkpoint_path", default=None)
    parser.add_argument("--locomotion_blend", type=float, default=0.0)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _yaw_xyzw(quat):
    x, y, z, w = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _percentile(value, q=0.95):
    return float(torch.quantile(value, q).item())


def _checkpoint_path(train_cfg, args):
    root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    checkpoint = -1 if args.checkpoint in (None, -1) else args.checkpoint
    return get_load_path(root, load_run=train_cfg.runner.load_run, checkpoint=checkpoint)


def _protocol(name, segment_s, forward_speed, backward_speed, stationary_s):
    if name == "squat":
        return (
            ("stand_1", 0, 0.0, "nominal", segment_s),
            ("squat_1", 2, 0.0, "squat", segment_s),
            ("stand_2", 0, 0.0, "nominal", segment_s),
            ("squat_2", 2, 0.0, "squat", segment_s),
            ("stand_3", 0, 0.0, "nominal", segment_s),
        )
    return (
        ("stand_1", 0, 0.0, "nominal", stationary_s),
        ("forward", 1, forward_speed, "nominal", segment_s),
        ("stop_1", 0, 0.0, "nominal", stationary_s),
        ("squat", 2, 0.0, "squat", stationary_s),
        ("stand_2", 0, 0.0, "nominal", stationary_s),
        ("backward", 1, backward_speed, "nominal", segment_s),
        ("stop_2", 0, 0.0, "nominal", stationary_s),
    )


def _plot(path, rows, result, boundaries):
    time_s = [row["time_s"] for row in rows]
    colors = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "gray": "#555555"}
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
        panels = (
            ("target_height_m", "mean_height_m", "Body height (m)", "Height tracking"),
            ("command_vx_mps", "mean_vx_mps", "Forward velocity (m/s)", "Velocity tracking"),
            (None, "mean_abs_lateral_m", "Absolute lateral deviation (m)", "Straight-line position"),
            (None, "mean_abs_heading_deg", "Absolute heading error (deg)", "No-turn behavior"),
            (None, "double_contact_fraction", "Fraction", "Double-foot support"),
            (None, "mean_action_rms", "Policy action RMS", "Action effort"),
        )
        for ax, (target_key, value_key, ylabel, title) in zip(axes.flat, panels):
            if target_key:
                ax.plot(time_s, [r[target_key] for r in rows], color=colors["gray"], linestyle=":", label="target")
            ax.plot(time_s, [r[value_key] for r in rows], color=colors["blue"], label="mean")
            for boundary in boundaries:
                ax.axvline(boundary, color="#999999", linewidth=0.7, linestyle="--")
            ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
            ax.grid(True, color="#D9D9D9", linewidth=0.6)
            ax.legend()
        fig.suptitle(
            f"MiniDuck {result['protocol_name']} evaluation "
            f"(n={result['num_trials']}, pass={result['acceptance_passed']})"
        )
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)


def evaluate(args, cfg):
    if not 0.0 <= cfg.symmetry_blend <= 1.0:
        raise ValueError("symmetry_blend must be between 0 and 1")
    if not 0.0 <= cfg.locomotion_blend <= 1.0:
        raise ValueError("locomotion_blend must be between 0 and 1")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 128
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = cfg.randomized
    env_cfg.domain_rand.curriculum_push_robots = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.max_action_delay = 0
    env_cfg.domain_rand.max_imu_delay = 0
    # The evaluator owns the command/skill timeline. Disable the environment's
    # periodic sampler so it cannot overwrite a protocol transition mid-step.
    env_cfg.commands.resampling_time = 1000.0
    if cfg.cross_track_heading_kp is not None:
        env_cfg.skill_curriculum.cross_track_heading_kp = (
            cfg.cross_track_heading_kp
        )
    if cfg.heading_hold_kp is not None:
        env_cfg.skill_curriculum.heading_hold_kp = cfg.heading_hold_kp
    if cfg.heading_hold_kd is not None:
        env_cfg.skill_curriculum.heading_hold_kd = cfg.heading_hold_kd
    if cfg.heading_hold_max is not None:
        env_cfg.skill_curriculum.heading_hold_max_yaw_rate = cfg.heading_hold_max
    if cfg.line_hold_max is not None:
        env_cfg.skill_curriculum.line_hold_max_lateral_mps = cfg.line_hold_max
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
    checkpoint = _checkpoint_path(train_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    iteration = _load_policy_checkpoint(runner, checkpoint, env.device)
    env.common_step_counter = iteration * runner.num_steps_per_env
    policy = runner.get_inference_policy(device=env.device)
    locomotion_policy = None
    if cfg.locomotion_checkpoint_path:
        locomotion_runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            log_root=None,
        )
        _load_policy_checkpoint(
            locomotion_runner,
            cfg.locomotion_checkpoint_path,
            env.device,
        )
        locomotion_policy = locomotion_runner.get_inference_policy(device=env.device)

    phases = _protocol(
        cfg.protocol,
        cfg.segment_s,
        cfg.forward_speed,
        cfg.backward_speed,
        cfg.stationary_s,
    )
    phase_steps = [max(1, round(item[4] / env.dt)) for item in phases]
    total_steps = sum(phase_steps)
    nominal = env.cfg.skill_curriculum.nominal_body_height_m
    squat = env.cfg.skill_curriculum.squat_body_height_m
    count = env.num_envs
    start_xy = env.root_states[:, :2].clone()
    start_yaw = _yaw_xyzw(env.root_states[:, 3:7]).clone()
    env.command_start_xy[:] = start_xy
    env.command_heading[:] = start_yaw
    env.line_reference_active[:] = True
    fallen = torch.zeros(count, dtype=torch.bool, device=env.device)
    max_lateral = torch.zeros(count, device=env.device)
    max_heading = torch.zeros(count, device=env.device)
    previous_action = torch.zeros(count, env.num_actions, device=env.device)
    phase_sums = {
        name: {key: torch.zeros(count, device=env.device) for key in ("height", "height_error", "directed_speed", "linear_speed", "double_contact")}
        for name, *_ in phases
    }
    phase_counts = {name: 0 for name, *_ in phases}
    phase_joint_sums = {
        name: torch.zeros(count, env.num_dof, device=env.device)
        for name, *_ in phases
    }
    phase_action_sums = {
        name: torch.zeros(count, env.num_actions, device=env.device)
        for name, *_ in phases
    }
    phase_displacement = {
        name: torch.zeros(count, device=env.device) for name, *_ in phases
    }
    rows = []
    phase_start_xy = start_xy.clone()
    obs = env.get_observations()
    phase_index = 0
    applied_phase_index = -1
    phase_start = 0
    boundaries = []

    for step in range(total_steps):
        while step - phase_start >= phase_steps[phase_index]:
            phase_start += phase_steps[phase_index]
            boundaries.append(phase_start * env.dt)
            phase_index += 1
        name, mode, vx, height_key, _ = phases[phase_index]
        target_height = squat if height_key == "squat" else nominal
        env.commands[:, :3] = 0.0
        env.commands[:, 0] = vx
        env.skill_mode[:] = mode
        if phase_index != applied_phase_index:
            phase_start_xy = env.root_states[:, :2].clone()
            env._schedule_skill_height(
                torch.arange(env.num_envs, device=env.device), target_height
            )
            applied_phase_index = phase_index
        if mode == env.SKILL_LOCOMOTION and not torch.any(env.line_reference_active):
            env.command_start_xy[:] = env.root_states[:, :2]
            env.command_heading[:] = _yaw_xyzw(env.root_states[:, 3:7])
            env.line_reference_active[:] = True
        env._apply_straight_heading_hold()
        obs[:, 9:12] = env.commands[:, :3] * env.commands_scale
        obs[:, 62:64] = env._skill_observation()
        with torch.no_grad():
            direct_actions = policy(obs.detach())
            symmetry_blend = 1.0 if cfg.symmetric_inference else cfg.symmetry_blend
            if symmetry_blend > 0.0:
                symmetric_actions = _symmetric_policy_action(
                    policy, obs.detach(), direct_action=direct_actions
                )
                actions = torch.lerp(
                    direct_actions, symmetric_actions, symmetry_blend
                )
            else:
                actions = direct_actions
            if locomotion_policy is not None and mode == env.SKILL_LOCOMOTION:
                expert_actions = locomotion_policy(obs.detach())
                actions = torch.lerp(actions, expert_actions, cfg.locomotion_blend)
        obs, _, _, dones, _ = env.step(actions.detach())
        fallen |= dones.bool()
        active = ~fallen
        yaw = _yaw_xyzw(env.root_states[:, 3:7])
        heading = torch.atan2(torch.sin(yaw - start_yaw), torch.cos(yaw - start_yaw)).abs()
        delta_xy = env.root_states[:, :2] - start_xy
        lateral = torch.abs(-delta_xy[:, 0] * torch.sin(start_yaw) + delta_xy[:, 1] * torch.cos(start_yaw))
        max_lateral = torch.maximum(max_lateral, lateral)
        max_heading = torch.maximum(max_heading, heading)
        height = env.root_states[:, 2]
        height_error = torch.abs(height - target_height)
        linear_speed = torch.norm(env.base_lin_vel, dim=1)
        directed_speed = (1.0 if vx >= 0.0 else -1.0) * env.base_lin_vel[:, 0] if vx != 0.0 else torch.zeros_like(linear_speed)
        contacts = (torch.sum((env.contact_forces[:, env.feet_indices, 2] > 1.0).float(), dim=1) == 2).float()
        action_rms = torch.sqrt(torch.mean(torch.square(actions), dim=1))
        action_rate = torch.sqrt(torch.mean(torch.square(actions - previous_action), dim=1))
        previous_action = actions
        phase_delta = env.root_states[:, :2] - phase_start_xy
        phase_forward = (
            phase_delta[:, 0] * torch.cos(start_yaw)
            + phase_delta[:, 1] * torch.sin(start_yaw)
        )
        if vx != 0.0:
            phase_displacement[name] = torch.sign(
                torch.tensor(vx, device=env.device)
            ) * phase_forward

        # The last 0.5 s of each segment measures achieved behavior, not transient response.
        window_start = phase_steps[phase_index] - max(1, round(0.5 / env.dt))
        if step - phase_start >= window_start:
            values = phase_sums[name]
            values["height"] += height
            values["height_error"] += height_error
            values["directed_speed"] += directed_speed
            values["linear_speed"] += linear_speed
            values["double_contact"] += contacts
            phase_joint_sums[name] += env.dof_pos
            phase_action_sums[name] += actions
            phase_counts[name] += 1

        rows.append({
            "time_s": round((step + 1) * env.dt, 6),
            "phase": name,
            "target_height_m": target_height,
            "mean_height_m": float(height[active].mean().item()),
            "command_vx_mps": vx,
            "mean_vx_mps": float(env.base_lin_vel[active, 0].mean().item()),
            "mean_linear_speed_mps": float(linear_speed[active].mean().item()),
            "mean_abs_lateral_m": float(lateral[active].mean().item()),
            "mean_abs_heading_deg": float(torch.rad2deg(heading[active]).mean().item()),
            "double_contact_fraction": float(contacts[active].mean().item()),
            "mean_action_rms": float(action_rms[active].mean().item()),
            "mean_action_rate": float(action_rate[active].mean().item()),
            "fall_fraction": float(fallen.float().mean().item()),
        })

    summaries = {}
    for name, *_ in phases:
        divisor = max(phase_counts[name], 1)
        summaries[name] = {
            key: value / divisor for key, value in phase_sums[name].items()
        }
    stand_names = [name for name, *_ in phases if name.startswith("stand") or name.startswith("stop")]
    squat_names = [name for name, *_ in phases if name.startswith("squat")]
    move_names = [name for name, *_ in phases if name in ("forward", "backward")]
    stand_height_error = torch.stack([summaries[name]["height_error"] for name in stand_names]).max(dim=0).values
    squat_height_error = torch.stack([summaries[name]["height_error"] for name in squat_names]).max(dim=0).values
    squat_height = torch.stack([summaries[name]["height"] for name in squat_names]).mean(dim=0)
    double_support = torch.stack([summaries[name]["double_contact"] for name in stand_names + squat_names]).mean(dim=0)
    if move_names:
        directed_speed = torch.stack([summaries[name]["directed_speed"] for name in move_names]).mean(dim=0)
    else:
        directed_speed = torch.zeros(count, device=env.device)
    stop_names = [name for name, *_ in phases if name.startswith("stop")]
    final_stop_speed = (
        torch.stack([summaries[name]["linear_speed"] for name in stop_names]).max(dim=0).values
        if stop_names else summaries[stand_names[-1]]["linear_speed"]
    )

    acceptance = {
        "fall_rate": float(fallen.float().mean().item()) <= 0.02,
        "p95_stand_height_error": _percentile(stand_height_error) <= 0.008,
        "p95_squat_height_error": _percentile(squat_height_error) <= 0.006,
        "mean_squat_depth": float((nominal - squat_height).mean().item()) >= 0.0075,
        "mean_double_support": float(double_support.mean().item()) >= 0.90,
        "p95_lateral_deviation": _percentile(max_lateral) <= 0.04,
        "p95_heading_error": _percentile(max_heading) <= 0.10,
    }
    if cfg.protocol == "action_switch":
        acceptance.update({
            "mean_directed_speed": float(directed_speed.mean().item()) >= 0.02,
            "p95_stop_speed": _percentile(final_stop_speed) <= 0.035,
        })
    result = {
        "run_label": cfg.run_label or f"{cfg.protocol}_{iteration}",
        "protocol_name": cfg.protocol,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "segment_s": cfg.segment_s,
        "stationary_s": cfg.stationary_s,
        "forward_command_mps": cfg.forward_speed,
        "backward_command_mps": cfg.backward_speed,
        "randomized": cfg.randomized,
        "symmetric_inference": cfg.symmetric_inference,
        "symmetry_blend": 1.0 if cfg.symmetric_inference else cfg.symmetry_blend,
        "locomotion_checkpoint": (
            os.path.abspath(cfg.locomotion_checkpoint_path)
            if cfg.locomotion_checkpoint_path else None
        ),
        "locomotion_blend": cfg.locomotion_blend,
        "fall_rate": float(fallen.float().mean().item()),
        "mean_squat_depth_m": float((nominal - squat_height).mean().item()),
        "p95_stand_height_error_m": _percentile(stand_height_error),
        "p95_squat_height_error_m": _percentile(squat_height_error),
        "mean_double_support_fraction": float(double_support.mean().item()),
        "mean_directed_speed_mps": float(directed_speed.mean().item()),
        "p95_stop_speed_mps": _percentile(final_stop_speed),
        "p95_max_lateral_m": _percentile(max_lateral),
        "p95_max_heading_deg": math.degrees(_percentile(max_heading)),
        "phase_mean_joint_positions_rad": {
            name: torch.mean(phase_joint_sums[name] / max(phase_counts[name], 1), dim=0)
            .detach()
            .cpu()
            .tolist()
            for name, *_ in phases
        },
        "phase_mean_policy_actions": {
            name: torch.mean(phase_action_sums[name] / max(phase_counts[name], 1), dim=0)
            .detach()
            .cpu()
            .tolist()
            for name, *_ in phases
        },
        "phase_terminal_metrics": {
            name: {
                **{
                    key: float(torch.mean(value / max(phase_counts[name], 1)).item())
                    for key, value in phase_sums[name].items()
                },
                "mean_directed_displacement_m": float(
                    torch.mean(phase_displacement[name]).item()
                ),
            }
            for name, *_ in phases
        },
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "provenance": {
            "aggregation": "per-step means and per-trial 95th percentiles; no smoothing",
            "terminal_window_s": 0.5,
            "missing_policy": "falls retained as failures",
            "seed": args.seed,
        },
    }

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = result["run_label"] + ("_randomized" if cfg.randomized else "_deterministic")
    metrics_path = output / f"{stem}_metrics.json"
    csv_path = output / f"{stem}_trajectory.csv"
    figure_path = output / f"{stem}_trajectory.png"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(figure_path, rows, result, boundaries)
    print(json.dumps(result, indent=2))
    print(f"metrics={metrics_path}")
    print(f"trajectory={csv_path}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    evaluation_cfg = _evaluation_args()
    evaluate(get_args(), evaluation_cfg)
