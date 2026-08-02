# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import os
import time

import isaacgym  # Must be imported before torch.
import torch

from legged_panguin import LEGGED_GYM_ROOT_DIR
from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.utils import get_args, task_registry
from legged_panguin.utils.helpers import get_load_path


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


def _latest_checkpoint(train_cfg):
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    return get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=-1,
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


def play(args):
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
    env_cfg.viewer.pos = [0.45, -0.45, 0.35]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.13]

    train_cfg.runner.resume = False
    train_cfg.runner.load_run = (
        -1 if args.load_run is None or str(args.load_run) == "-1" else args.load_run
    )
    train_cfg.runner.checkpoint = -1

    initial_checkpoint = _latest_checkpoint(train_cfg)
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
    single_leg_task = args.task == "miniduck_single_leg_physics"
    print(f"Visualizing one MiniDuck from {loaded_checkpoint}")
    if single_leg_task:
        print("Single-leg physics task: preserving the trained support command.")

    obs = env.get_observations()
    command_steps = max(1, int(5.0 / env.dt))
    reload_steps = max(1, int(2.0 / env.dt))
    step = 0

    while True:
        if single_leg_task:
            command_name = "left_support"
            command = env.commands[0, :3].detach().clone()
        else:
            command_index = (step // command_steps) % len(COMMANDS)
            command_name, command = COMMANDS[command_index]
            env.commands[0, :3] = torch.tensor(command, device=env.device)

        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        if dones[0] and not single_leg_task:
            env.commands[0, :3] = torch.tensor(command, device=env.device)

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
            latest_checkpoint = _latest_checkpoint(train_cfg)
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
    play(get_args())
