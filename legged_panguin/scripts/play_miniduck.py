# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
import time

import isaacgym  # Must be imported before torch.
import torch
from isaacgym.torch_utils import euler_from_quat

from legged_panguin import LEGGED_GYM_ROOT_DIR
from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.utils import get_args, task_registry
from legged_panguin.utils.helpers import get_load_path
from legged_panguin.envs.miniduck.miniduck_config import (
    JOINT_MIRROR_PERMUTATION,
    JOINT_MIRROR_SIGNS,
    SYMMETRY_OBS_PERMUTATION,
    SYMMETRY_OBS_SIGNS,
)


LOCOMOTION_COMMANDS = (
    ("forward", (0.10, 0.0, 0.0)),
    ("backward", (-0.06, 0.0, 0.0)),
    ("forward_turn_left", (0.08, 0.0, 0.45)),
    ("forward_turn_right", (0.08, 0.0, -0.45)),
    ("left", (0.0, 0.16, 0.0)),
    ("right", (0.0, -0.16, 0.0)),
    ("turn_left", (0.0, 0.0, 0.80)),
    ("turn_right", (0.0, 0.0, -0.80)),
)

EMERGENCY_STOP_SEQUENCE = (
    ("stable_stand", (0.0, 0.0, 0.0), 2.0),
    ("forward_probe", (0.08, 0.0, 0.0), 2.0),
    ("emergency_stop", (0.0, 0.0, 0.0), 2.0),
    ("backward_probe", (-0.06, 0.0, 0.0), 2.0),
    ("emergency_stop", (0.0, 0.0, 0.0), 2.0),
)

SQUAT_SEQUENCE = (
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("squat", (0.0, 0.0, 0.0), 2.0, 2, "squat"),
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
)

ACTION_SWITCH_SEQUENCE = (
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("forward", (0.08, 0.0, 0.0), 2.0, 1, "nominal"),
    ("stop", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("squat", (0.0, 0.0, 0.0), 2.0, 2, "squat"),
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("backward", (-0.06, 0.0, 0.0), 2.0, 1, "nominal"),
    ("stop", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
)


def _demo_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--demo",
        choices=("emergency_stop", "locomotion", "squat", "action_switch"),
        default="emergency_stop",
    )
    parser.add_argument("--symmetric_inference", action="store_true")
    parser.add_argument("--symmetry_blend", type=float, default=0.0)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


def _checkpoint_path(train_cfg, args):
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    return get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=(
            -1 if args.checkpoint in (None, -1) else args.checkpoint
        ),
    )


def _load_policy_checkpoint(runner, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"]
    target_state = runner.alg.actor_critic.state_dict()
    for key, value in list(state.items()):
        target_value = target_state.get(key)
        if target_value is None or value.shape == target_value.shape:
            continue
        can_expand_input = (
            key in ("actor.0.weight", "critic.0.weight")
            and value.ndim == 2
            and target_value.ndim == 2
            and value.shape[0] == target_value.shape[0]
            and value.shape[1] < target_value.shape[1]
        )
        if not can_expand_input:
            raise RuntimeError(
                f"Checkpoint tensor shape mismatch for {key}: "
                f"{tuple(value.shape)} -> {tuple(target_value.shape)}"
            )
        expanded = target_value.clone()
        expanded[:, : value.shape[1]] = value
        expanded[:, value.shape[1] :] = 0.0
        state[key] = expanded
    runner.alg.actor_critic.load_state_dict(state)
    runner.current_learning_iteration = checkpoint["iter"]
    return checkpoint["iter"]


def _symmetric_policy_action(policy, obs, direct_action=None):
    """Average the policy with its left-right mirror-equivariant prediction."""
    obs_permutation = torch.as_tensor(
        SYMMETRY_OBS_PERMUTATION, dtype=torch.long, device=obs.device
    )
    obs_signs = torch.as_tensor(
        SYMMETRY_OBS_SIGNS, dtype=obs.dtype, device=obs.device
    )
    action_permutation = torch.as_tensor(
        JOINT_MIRROR_PERMUTATION, dtype=torch.long, device=obs.device
    )
    action_signs = torch.as_tensor(
        JOINT_MIRROR_SIGNS, dtype=obs.dtype, device=obs.device
    )
    action = policy(obs) if direct_action is None else direct_action
    mirrored_action = policy(obs[:, obs_permutation] * obs_signs)
    unmirrored_action = mirrored_action[:, action_permutation] * action_signs
    return 0.5 * (action + unmirrored_action)


def play(args, demo):
    if not 0.0 <= demo.symmetry_blend <= 1.0:
        raise ValueError("symmetry_blend must be between 0 and 1")
    # This viewer is intentionally separate from the 4096-environment trainer.
    args.num_envs = 1
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_dof_properties = False
    env_cfg.domain_rand.curriculum_push_robots = False
    env_cfg.domain_rand.max_action_delay = 0
    env_cfg.domain_rand.max_imu_delay = 0
    env_cfg.domain_rand.motor_strength_range = [1.0, 1.0]
    env_cfg.domain_rand.kp_factor_range = [1.0, 1.0]
    env_cfg.domain_rand.kd_factor_range = [1.0, 1.0]
    env_cfg.domain_rand.motor_velocity_range = [5.24, 5.24]
    env_cfg.domain_rand.joint_target_offset = 0.0
    env_cfg.domain_rand.encoder_offset = 0.0
    env_cfg.domain_rand.gyro_bias = 0.0
    env_cfg.domain_rand.gravity_bias = 0.0
    env_cfg.domain_rand.accel_bias = 0.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 1000.0
    env_cfg.viewer.pos = [0.45, -0.45, 0.35]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.13]

    train_cfg.runner.resume = False
    train_cfg.runner.load_run = (
        -1 if args.load_run is None or str(args.load_run) == "-1" else args.load_run
    )
    train_cfg.runner.checkpoint = -1

    initial_checkpoint = _checkpoint_path(train_cfg, args)
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    checkpoint_iteration = _load_policy_checkpoint(
        runner,
        initial_checkpoint,
        env.device,
    )
    env.common_step_counter = checkpoint_iteration * runner.num_steps_per_env
    policy = runner.get_inference_policy(device=env.device)
    loaded_checkpoint = initial_checkpoint
    print(f"Visualizing one MiniDuck from {loaded_checkpoint}")

    obs = env.get_observations()
    if demo.demo == "emergency_stop":
        sequence = tuple((name, command, duration, 1 if command[0] else 0, "nominal") for name, command, duration in EMERGENCY_STOP_SEQUENCE)
    elif demo.demo == "locomotion":
        sequence = tuple((name, command, 5.0, 1, "nominal") for name, command in LOCOMOTION_COMMANDS)
    elif demo.demo == "squat":
        sequence = SQUAT_SEQUENCE
    else:
        sequence = ACTION_SWITCH_SEQUENCE
    sequence_steps = [max(1, round(item[2] / env.dt)) for item in sequence]
    cycle_steps = sum(sequence_steps)
    reload_steps = max(1, int(2.0 / env.dt))
    step = 0
    previous_command_name = None

    while True:
        cycle_step = step % cycle_steps
        command_index = 0
        while cycle_step >= sequence_steps[command_index]:
            cycle_step -= sequence_steps[command_index]
            command_index += 1
        command_name, command, _, skill_mode, height_key = sequence[command_index]
        target_height = (
            env.cfg.skill_curriculum.squat_body_height_m
            if height_key == "squat"
            else env.cfg.skill_curriculum.nominal_body_height_m
        )
        if command_name != previous_command_name:
            print(f"Demo phase: {command_name}; command={command}")
            env.skill_mode[:] = skill_mode
            env._schedule_skill_height(
                torch.arange(env.num_envs, device=env.device), target_height
            )
            if demo.demo in ("emergency_stop", "squat", "action_switch") and not torch.any(env.line_reference_active):
                _, _, yaw = euler_from_quat(env.base_quat)
                env.command_heading[:] = yaw
                env.command_start_xy[:] = env.root_states[:, :2]
                env.line_reference_active[:] = True
            previous_command_name = command_name
        env.commands[0, :3] = torch.tensor(command, device=env.device)
        env.skill_mode[:] = skill_mode
        env._apply_straight_heading_hold()
        obs[:, 9:12] = env.commands[:, :3] * env.commands_scale
        obs[:, 62:64] = env._skill_observation()

        with torch.no_grad():
            direct_actions = policy(obs.detach())
            symmetry_blend = (
                1.0 if demo.symmetric_inference else demo.symmetry_blend
            )
            if symmetry_blend > 0.0:
                symmetric_actions = _symmetric_policy_action(
                    policy, obs.detach(), direct_action=direct_actions
                )
                actions = torch.lerp(
                    direct_actions, symmetric_actions, symmetry_blend
                )
            else:
                actions = direct_actions
        obs, _, _, dones, _ = env.step(actions.detach())

        if dones[0]:
            env.commands[0, :3] = torch.tensor(command, device=env.device)
            env.line_reference_active[:] = False
            previous_command_name = None

        robot_position = env.root_states[0, :3].detach().cpu().tolist()
        env.set_camera(
            [
                robot_position[0] + 0.45,
                robot_position[1] - 0.45,
                robot_position[2] + 0.20,
            ],
            [
                robot_position[0],
                robot_position[1],
                robot_position[2],
            ],
        )

        if step % reload_steps == 0:
            latest_checkpoint = _checkpoint_path(train_cfg, args)
            if latest_checkpoint != loaded_checkpoint:
                # Give the trainer time to finish flushing a newly-created file.
                time.sleep(0.2)
                try:
                    checkpoint_iteration = _load_policy_checkpoint(
                        runner,
                        latest_checkpoint,
                        env.device,
                    )
                except (EOFError, OSError, RuntimeError):
                    pass
                else:
                    env.common_step_counter = (
                        checkpoint_iteration * runner.num_steps_per_env
                    )
                    loaded_checkpoint = latest_checkpoint
                    print(
                        f"Loaded {os.path.basename(loaded_checkpoint)}; "
                        f"command={command_name}"
                    )
        step += 1


if __name__ == "__main__":
    demo_args = _demo_args()
    play(get_args(), demo_args)
