"""Evaluate a MiniDuck move-to-stop transition and preserve its trajectories."""

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


def _evaluation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--move_s", type=float, default=2.0)
    parser.add_argument("--stop_s", type=float, default=6.0)
    parser.add_argument("--speed_mps", type=float, default=0.08)
    parser.add_argument("--settle_linear_mps", type=float, default=0.035)
    parser.add_argument("--settle_angular_rps", type=float, default=0.25)
    parser.add_argument("--settle_tilt_deg", type=float, default=12.0)
    parser.add_argument("--settle_hold_s", type=float, default=0.50)
    parser.add_argument("--output_dir", type=str, default="evaluation/emergency_stop")
    parser.add_argument("--run_label", type=str, default="emergency_stop")
    parser.add_argument("--randomized", action="store_true")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _yaw_xyzw(quat):
    x, y, z, w = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _mean(values, mask):
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else None


def _percentile(values, mask, q):
    selected = values[mask]
    return float(torch.quantile(selected, q).item()) if selected.numel() else None


def _checkpoint_path(train_cfg, args):
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    checkpoint = -1 if args.checkpoint in (None, -1) else args.checkpoint
    return get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=checkpoint,
    )


def _write_trajectory(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_trials(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_trajectory(path, rows, metrics, move_s):
    time_s = [row["time_s"] for row in rows]
    command = [row["command_abs_vx_mps"] for row in rows]
    colors = {
        "blue": "#0072B2",
        "orange": "#D55E00",
        "green": "#009E73",
        "purple": "#CC79A7",
        "gray": "#555555",
    }
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), layout="constrained")
        ax = axes[0, 0]
        ax.plot(time_s, command, color=colors["gray"], linestyle=":", label="command |vx|")
        ax.plot(time_s, [r["mean_linear_speed_mps"] for r in rows], color=colors["blue"], label="mean speed")
        ax.plot(time_s, [r["p95_linear_speed_mps"] for r in rows], color=colors["orange"], linestyle="--", label="95th percentile")
        ax.set(ylabel="Linear speed (m/s)", title="Body speed")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(time_s, [r["mean_angular_speed_rps"] for r in rows], color=colors["green"], label="mean")
        ax.plot(time_s, [r["p95_angular_speed_rps"] for r in rows], color=colors["purple"], linestyle="--", label="95th percentile")
        ax.axhline(metrics["thresholds"]["angular_speed_rps"], color=colors["gray"], linestyle=":", label="settled threshold")
        ax.set(ylabel="Angular speed (rad/s)", title="Body rotation")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(time_s, [r["mean_tilt_deg"] for r in rows], color=colors["blue"], label="mean")
        ax.plot(time_s, [r["p95_tilt_deg"] for r in rows], color=colors["orange"], linestyle="--", label="95th percentile")
        ax.axhline(metrics["thresholds"]["tilt_deg"], color=colors["gray"], linestyle=":", label="settled threshold")
        ax.set(xlabel="Time (s)", ylabel="Tilt from nominal (deg)", title="Body attitude")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(time_s, [r["settled_fraction"] for r in rows], color=colors["green"], label="settled fraction")
        ax.plot(time_s, [r["fall_fraction"] for r in rows], color=colors["orange"], linestyle="--", label="fall fraction")
        ax.set(xlabel="Time (s)", ylabel="Fraction of trials", ylim=(-0.02, 1.02), title="Outcome over time")
        ax.legend()

        for ax in axes.flat:
            ax.axvline(move_s, color="#000000", linewidth=1.0, linestyle="-.")
            ax.axvspan(move_s, time_s[-1], color="#E6E6E6", alpha=0.25)
            ax.grid(True, color="#D9D9D9", linewidth=0.6)
        fig.suptitle(
            f"MiniDuck emergency stop: {metrics['run_label']} "
            f"(n={metrics['num_trials']}, success={metrics['settle_success_rate']:.1%})"
        )
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)


def evaluate(args, cfg):
    if min(cfg.move_s, cfg.stop_s, cfg.speed_mps, cfg.settle_hold_s) <= 0:
        raise ValueError("move, stop, speed and settle hold values must be positive")
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
    env_cfg.commands.resampling_time = cfg.move_s + cfg.stop_s + 1.0
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
    train_cfg.runner.load_run = (
        -1 if args.load_run in (None, "-1", -1) else args.load_run
    )
    train_cfg.runner.checkpoint = -1
    checkpoint = _checkpoint_path(train_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    iteration = _load_policy_checkpoint(runner, checkpoint, env.device)
    env.common_step_counter = iteration * runner.num_steps_per_env
    policy = runner.get_inference_policy(device=env.device)

    count = env.num_envs
    directions = torch.ones(count, device=env.device)
    directions[count // 2:] = -1.0
    move_command = directions * cfg.speed_mps
    start_xy = env.root_states[:, :2].clone()
    start_yaw = _yaw_xyzw(env.root_states[:, 3:7]).clone()
    stop_xy = torch.zeros_like(start_xy)
    fallen = torch.zeros(count, dtype=torch.bool, device=env.device)
    settle_counts = torch.zeros(count, dtype=torch.long, device=env.device)
    settle_steps = torch.full((count,), -1, dtype=torch.long, device=env.device)
    max_post_stop_tilt = torch.zeros(count, device=env.device)
    max_post_stop_drift = torch.zeros(count, device=env.device)
    final_linear = torch.full((count,), float("nan"), device=env.device)
    final_angular = torch.full((count,), float("nan"), device=env.device)
    final_tilt = torch.full((count,), float("nan"), device=env.device)
    pre_stop_speed_sum = torch.zeros(count, device=env.device)
    pre_stop_speed_count = 0
    rows = []

    move_steps = max(1, round(cfg.move_s / env.dt))
    stop_steps = max(1, round(cfg.stop_s / env.dt))
    hold_steps = max(1, round(cfg.settle_hold_s / env.dt))
    pre_stop_window = max(1, round(0.5 / env.dt))
    total_steps = move_steps + stop_steps
    env.commands[:, 0] = move_command
    env.commands[:, 1:3] = 0.0
    env.compute_observations()
    obs = env.get_observations()

    for step in range(total_steps):
        stopping = step >= move_steps
        if step == move_steps:
            stop_xy[:] = env.root_states[:, :2]
        env.commands[:, 0] = 0.0 if stopping else move_command
        env.commands[:, 1:3] = 0.0
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        alive_before = ~fallen
        new_falls = alive_before & dones.bool()
        fallen |= new_falls
        active = ~fallen
        linear_speed = torch.norm(env.base_lin_vel, dim=1)
        angular_speed = torch.norm(env.base_ang_vel, dim=1)
        target_gravity = env.target_projected_gravity.expand_as(env.projected_gravity)
        alignment = torch.clamp(
            torch.sum(env.projected_gravity * target_gravity, dim=1), -1.0, 1.0
        )
        tilt_deg = torch.rad2deg(torch.acos(alignment))
        directed_speed = directions * env.base_lin_vel[:, 0]

        if move_steps - pre_stop_window <= step < move_steps:
            pre_stop_speed_sum[active] += directed_speed[active]
            pre_stop_speed_count += 1

        if stopping:
            drift = torch.norm(env.root_states[:, :2] - stop_xy, dim=1)
            max_post_stop_tilt = torch.maximum(max_post_stop_tilt, tilt_deg)
            max_post_stop_drift = torch.maximum(max_post_stop_drift, drift)
            stable_now = (
                active
                & (linear_speed < cfg.settle_linear_mps)
                & (angular_speed < cfg.settle_angular_rps)
                & (tilt_deg < cfg.settle_tilt_deg)
            )
            settle_counts = torch.where(
                stable_now, settle_counts + 1, torch.zeros_like(settle_counts)
            )
            new_settled = (settle_steps < 0) & (settle_counts >= hold_steps)
            settle_steps[new_settled] = max(0, step - move_steps - hold_steps + 1)

        final_linear[active] = linear_speed[active]
        final_angular[active] = angular_speed[active]
        final_tilt[active] = tilt_deg[active]
        settled = (settle_steps >= 0) & active
        rows.append({
            "time_s": round((step + 1) * env.dt, 6),
            "phase": "stop" if stopping else "move",
            "command_abs_vx_mps": 0.0 if stopping else cfg.speed_mps,
            "mean_linear_speed_mps": _mean(linear_speed, active),
            "p95_linear_speed_mps": _percentile(linear_speed, active, 0.95),
            "mean_angular_speed_rps": _mean(angular_speed, active),
            "p95_angular_speed_rps": _percentile(angular_speed, active, 0.95),
            "mean_tilt_deg": _mean(tilt_deg, active),
            "p95_tilt_deg": _percentile(tilt_deg, active, 0.95),
            "mean_height_m": _mean(env.root_states[:, 2], active),
            "settled_fraction": float(settled.float().mean().item()),
            "fall_fraction": float(fallen.float().mean().item()),
        })

    success = (~fallen) & (settle_steps >= 0)
    settle_time_s = settle_steps.float() * env.dt
    valid_final = ~fallen
    pre_stop_speed = pre_stop_speed_sum / max(pre_stop_speed_count, 1)
    result = {
        "run_label": cfg.run_label,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_iteration": iteration,
        "num_trials": count,
        "randomized": cfg.randomized,
        "protocol": {
            "move_s": cfg.move_s,
            "stop_s": cfg.stop_s,
            "command_speed_abs_mps": cfg.speed_mps,
            "directions": "half forward, half backward",
            "settle_hold_s": cfg.settle_hold_s,
        },
        "thresholds": {
            "linear_speed_mps": cfg.settle_linear_mps,
            "angular_speed_rps": cfg.settle_angular_rps,
            "tilt_deg": cfg.settle_tilt_deg,
        },
        "mean_pre_stop_directed_speed_mps": float(pre_stop_speed.mean().item()),
        "fall_rate": float(fallen.float().mean().item()),
        "settle_success_rate": float(success.float().mean().item()),
        "mean_settle_time_s": _mean(settle_time_s, success),
        "p95_settle_time_s": _percentile(settle_time_s, success, 0.95),
        "mean_final_linear_speed_mps": _mean(final_linear, valid_final),
        "p95_final_linear_speed_mps": _percentile(final_linear, valid_final, 0.95),
        "mean_final_angular_speed_rps": _mean(final_angular, valid_final),
        "p95_final_angular_speed_rps": _percentile(final_angular, valid_final, 0.95),
        "mean_final_tilt_deg": _mean(final_tilt, valid_final),
        "p95_max_post_stop_tilt_deg": _percentile(max_post_stop_tilt, valid_final, 0.95),
        "mean_max_post_stop_drift_m": _mean(max_post_stop_drift, valid_final),
        "acceptance": {
            "pre_stop_motion": float(pre_stop_speed.mean().item()) >= 0.02,
            "fall_rate": float(fallen.float().mean().item()) <= 0.02,
            "settle_success_rate": float(success.float().mean().item()) >= 0.95,
            "p95_settle_time_s": (
                _percentile(settle_time_s, success, 0.95) is not None
                and _percentile(settle_time_s, success, 0.95) <= 2.5
            ),
            "p95_final_linear_speed": (
                _percentile(final_linear, valid_final, 0.95) is not None
                and _percentile(final_linear, valid_final, 0.95) <= cfg.settle_linear_mps
            ),
            "p95_final_angular_speed": (
                _percentile(final_angular, valid_final, 0.95) is not None
                and _percentile(final_angular, valid_final, 0.95) <= cfg.settle_angular_rps
            ),
        },
        "provenance": {
            "aggregation": "per-step mean and 95th percentile; no smoothing",
            "missing_policy": "post-fall samples excluded and each fall retained in trial outcomes",
            "seed": args.seed,
        },
    }
    result["acceptance_passed"] = all(result["acceptance"].values())

    trial_rows = []
    for index in range(count):
        trial_rows.append({
            "trial": index,
            "direction": "forward" if directions[index].item() > 0 else "backward",
            "fell": bool(fallen[index].item()),
            "settled": bool(success[index].item()),
            "settle_time_s": (
                float(settle_time_s[index].item()) if success[index] else ""
            ),
            "pre_stop_directed_speed_mps": float(pre_stop_speed[index].item()),
            "final_linear_speed_mps": (
                float(final_linear[index].item()) if valid_final[index] else ""
            ),
            "final_angular_speed_rps": (
                float(final_angular[index].item()) if valid_final[index] else ""
            ),
            "final_tilt_deg": (
                float(final_tilt[index].item()) if valid_final[index] else ""
            ),
            "max_post_stop_tilt_deg": (
                float(max_post_stop_tilt[index].item()) if valid_final[index] else ""
            ),
            "max_post_stop_drift_m": (
                float(max_post_stop_drift[index].item()) if valid_final[index] else ""
            ),
        })

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.run_label + ("_randomized" if cfg.randomized else "_deterministic")
    metrics_path = output_dir / f"{stem}_metrics.json"
    trajectory_path = output_dir / f"{stem}_trajectory.csv"
    trials_path = output_dir / f"{stem}_trials.csv"
    figure_path = output_dir / f"{stem}_trajectory.png"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_trajectory(trajectory_path, rows)
    _write_trials(trials_path, trial_rows)
    _plot_trajectory(figure_path, rows, result, cfg.move_s)
    print(json.dumps(result, indent=2))
    print(f"metrics={metrics_path}")
    print(f"trajectory={trajectory_path}")
    print(f"trials={trials_path}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    evaluation_cfg = _evaluation_args()
    evaluate(get_args(), evaluation_cfg)
