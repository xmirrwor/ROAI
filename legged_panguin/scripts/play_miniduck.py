# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
import time

import isaacgym  # Must be imported before torch.
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import euler_from_quat, quat_from_euler_xyz

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
    ("forward", (0.12, 0.0, 0.0)),
    ("backward", (-0.08, 0.0, 0.0)),
    ("forward_turn_left", (0.08, 0.0, 0.45)),
    ("forward_turn_right", (0.08, 0.0, -0.45)),
    ("left", (0.0, 0.16, 0.0)),
    ("right", (0.0, -0.16, 0.0)),
    ("turn_left", (0.0, 0.0, 0.80)),
    ("turn_right", (0.0, 0.0, -0.80)),
)

EMERGENCY_STOP_SEQUENCE = (
    ("stable_stand", (0.0, 0.0, 0.0), 2.0),
    ("forward_probe", (0.10, 0.0, 0.0), 5.0),
    ("emergency_stop", (0.0, 0.0, 0.0), 2.0),
    ("backward_probe", (-0.08, 0.0, 0.0), 5.0),
    ("emergency_stop", (0.0, 0.0, 0.0), 2.0),
)

SQUAT_SEQUENCE = (
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("squat", (0.0, 0.0, 0.0), 2.0, 2, "squat"),
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
)

ACTION_SWITCH_SEQUENCE = (
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("forward", (0.12, 0.0, 0.0), 5.0, 1, "nominal"),
    ("stop", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("squat", (0.0, 0.0, 0.0), 2.0, 2, "squat"),
    ("stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
    ("backward", (-0.08, 0.0, 0.0), 5.0, 1, "nominal"),
    ("stop", (0.0, 0.0, 0.0), 2.0, 0, "nominal"),
)

DIAGONAL_SEQUENCE = (
    ("forward_left", (0.10, 0.08, 0.0), 5.0, 1, "nominal"),
    ("forward_right", (0.10, -0.08, 0.0), 5.0, 1, "nominal"),
    ("backward_left", (-0.08, 0.08, 0.0), 5.0, 1, "nominal"),
    ("backward_right", (-0.08, -0.08, 0.0), 5.0, 1, "nominal"),
)

OBSTACLE_SEQUENCE = (
    ("obstacle_approach", (0.12, 0.0, 0.0), 7.0, 4, "nominal"),
)

BALL_SEQUENCE = (
    ("ball_kick", (0.10, 0.0, 0.0), 6.0, 5, "nominal"),
)

# name, command, seconds, skill, height, expert, event, effective stage
FULL_SEQUENCE = (
    ("stand", (0.0, 0.0, 0.0), 1.5, 0, "nominal", "stage5", "flat_reset", "action_switch"),
    ("forward", (0.10, 0.0, 0.0), 5.0, 1, "nominal", "stage7", None, "action_switch"),
    ("emergency_stop", (0.0, 0.0, 0.0), 2.0, 0, "nominal", "stage7", None, "action_switch"),
    ("backward", (-0.08, 0.0, 0.0), 5.0, 1, "nominal", "stage7", None, "action_switch"),
    ("emergency_stop", (0.0, 0.0, 0.0), 2.0, 0, "nominal", "stage7", None, "action_switch"),
    ("squat", (0.0, 0.0, 0.0), 2.0, 2, "squat", "stage5", None, "action_switch"),
    ("stand", (0.0, 0.0, 0.0), 1.5, 0, "nominal", "stage5", None, "action_switch"),
    ("switch_to_walk", (0.10, 0.0, 0.0), 2.5, 1, "nominal", "stage5", None, "action_switch"),
    ("switch_to_stop", (0.0, 0.0, 0.0), 1.5, 0, "nominal", "stage5", None, "action_switch"),
    ("switch_to_squat", (0.0, 0.0, 0.0), 2.0, 2, "squat", "stage5", None, "action_switch"),
    ("switch_to_stand", (0.0, 0.0, 0.0), 1.5, 0, "nominal", "stage5", None, "action_switch"),
    ("fall", (0.0, 0.0, 0.0), 0.6, 3, "nominal", "stage6", "set_prone", "action_switch"),
    ("fall_recovery", (0.0, 0.0, 0.0), 4.0, 3, "nominal", "stage6", None, "action_switch"),
    ("recovery_stand", (0.0, 0.0, 0.0), 1.5, 0, "nominal", "stage5", "flat_reset", "action_switch"),
    ("forward_left", (0.10, 0.08, 0.0), 4.0, 1, "nominal", "stage7", None, "diagonal_motion"),
    ("forward_right", (0.10, -0.08, 0.0), 4.0, 1, "nominal", "stage7", None, "diagonal_motion"),
    ("backward_left", (-0.08, 0.08, 0.0), 4.0, 1, "nominal", "stage7", None, "diagonal_motion"),
    ("backward_right", (-0.08, -0.08, 0.0), 4.0, 1, "nominal", "stage7", None, "diagonal_motion"),
    ("prepare_obstacle", (0.0, 0.0, 0.0), 1.0, 0, "nominal", "stage5", "obstacle_reset", "action_switch"),
    ("obstacle_crossing", (0.12, 0.0, 0.0), 7.0, 4, "nominal", "stage8", None, "obstacle_crossing"),
    ("ball_approach", (0.12, 0.0, 0.0), 10.0, 1, "nominal", "stage7", "ball_scene_reset", "action_switch"),
    ("ball_kick", (0.10, 0.0, 0.0), 6.0, 5, "nominal", "stage10", "kick_ready", "ball_kick"),
    ("finish_stand", (0.0, 0.0, 0.0), 2.0, 0, "nominal", "stage5", "flat_reset", "action_switch"),
)


def _demo_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--demo",
        choices=(
            "emergency_stop",
            "locomotion",
            "squat",
            "action_switch",
            "fall_recovery",
            "diagonal_motion",
            "obstacle_crossing",
            "ball_kick",
            "full_sequence",
        ),
        default="emergency_stop",
    )
    parser.add_argument("--symmetric_inference", action="store_true")
    parser.add_argument("--symmetry_blend", type=float, default=0.0)
    parser.add_argument("--fixed_camera", action="store_true")
    parser.add_argument("--locomotion_checkpoint_path", default=None)
    parser.add_argument("--locomotion_blend", type=float, default=0.0)
    parser.add_argument("--heading_hold_kp", type=float, default=None)
    parser.add_argument("--cross_track_heading_kp", type=float, default=None)
    parser.add_argument("--checkpoint_path_override", default=None)
    parser.add_argument(
        "--stage5_checkpoint_path",
        default="checkpoints/miniduck_stage5_action_switch_model_14780.pt",
    )
    parser.add_argument(
        "--stage6_checkpoint_path",
        default="checkpoints/miniduck_stage6_recovery_model_15420.pt",
    )
    parser.add_argument(
        "--stage7_checkpoint_path",
        default="checkpoints/miniduck_stage7_diagonal_fast_model_12320.pt",
    )
    parser.add_argument(
        "--stage8_checkpoint_path",
        default="checkpoints/miniduck_stage8_obstacle_model_selected.pt",
    )
    parser.add_argument(
        "--stage10_checkpoint_path",
        default="checkpoints/miniduck_stage10_ball_kick_model_selected.pt",
    )
    parser.add_argument("--policy_transition_s", type=float, default=0.35)
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


def _configure_obstacle_env(env_cfg):
    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.obstacle_course = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.skill_curriculum.forced_stage = 5
    env_cfg.scene.obstacle_enabled = True


def _configure_ball_env(env_cfg):
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.obstacle_course = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.skill_curriculum.forced_stage = 6
    env_cfg.scene.ball_enabled = True


def _load_named_policy(env, args, train_cfg, checkpoint_path):
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    iteration = _load_policy_checkpoint(
        runner, os.path.abspath(checkpoint_path), env.device
    )
    return runner, runner.get_inference_policy(device=env.device), iteration


def _set_demo_pose(env, event):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env._reset_dofs(env_ids)
    if event == "set_prone":
        current_xy = env.root_states[:, :2].clone()
        env.root_states[:] = env.base_init_state
        env.root_states[:, :3] += env.env_origins
        env.root_states[:, :2] = current_xy
        env.root_states[:, 2] = (
            env.env_origins[:, 2]
            + env.cfg.skill_curriculum.recovery_start_height_m
        )
        roll = torch.zeros(env.num_envs, device=env.device)
        pitch = torch.full((env.num_envs,), 0.62, device=env.device)
        yaw = torch.zeros(env.num_envs, device=env.device)
        env.root_states[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        env.demo_recovery_active[:] = True
        env.obstacle_task_active[:] = False
    elif event == "kick_ready":
        ball_xy = env.ball_root_states[:, :2].clone()
        env.root_states[:] = env.base_init_state
        env.root_states[:, :3] += env.env_origins
        env.root_states[:, 0] = (
            ball_xy[:, 0]
            - env.cfg.skill_curriculum.ball_kick_spawn_distance_m
        )
        env.root_states[:, 1] = (
            ball_xy[:, 1]
            - env.cfg.skill_curriculum.ball_spawn_lateral_center_m
        )
        env.obstacle_task_active[:] = False
        env.demo_recovery_active[:] = False
    else:
        env.root_states[:] = env.base_init_state
        env.root_states[:, :3] += env.env_origins
        if env.cfg.terrain.mesh_type in ("heightfield", "trimesh"):
            env.root_states[:, 2] += (
                env.cfg.skill_curriculum.obstacle_spawn_height_offset_m
            )
        if event == "obstacle_reset":
            env.root_states[:, 0] = (
                env.obstacle_world_x
                - env.cfg.skill_curriculum.obstacle_approach_distance_m
            )
            env.obstacle_task_active[:] = False
            env.obstacle_elapsed_steps[:] = 0
        else:
            env.obstacle_task_active[:] = False
        env.demo_recovery_active[:] = False
    env.root_states[:, 7:13] = 0.0
    env.commanded_actions[:] = 0.0
    env.commanded_action_history_1[:] = 0.0
    env.commanded_action_history_2[:] = 0.0
    env.actions[:] = 0.0
    env.previous_motor_targets[:] = env.default_dof_pos
    env.gait_phase_steps[:] = 0
    env.gait_contacts[:] = False
    env.gait_last_contacts[:] = False
    env.gait_first_contacts[:] = 0.0
    actor_ids = env._robot_actor_ids(env_ids)
    if env.ball_root_states is not None:
        if event == "kick_ready":
            env.ball_elapsed_steps[:] = 0
            env.ball_success_latched[:] = False
            env.ball_progress_latched[:] = 0.0
        else:
            env.ball_root_states[:] = 0.0
            if event == "ball_scene_reset":
                ball_distance = env.cfg.skill_curriculum.ball_demo_spawn_distance_m
                ball_height = env.cfg.scene.ball_radius_m
            else:
                ball_distance = 0.0
                ball_height = -1.0
            env.ball_root_states[:, 0] = env.env_origins[:, 0] + ball_distance
            env.ball_root_states[:, 1] = (
                env.env_origins[:, 1]
                + env.cfg.skill_curriculum.ball_spawn_lateral_center_m
            )
            env.ball_root_states[:, 2] = ball_height
            env.ball_root_states[:, 6] = 1.0
            env.ball_start_x[:] = env.ball_root_states[:, 0]
            env.ball_elapsed_steps[:] = 0
            env.ball_success_latched[:] = False
            actor_ids = torch.cat((
                actor_ids,
                actor_ids + env.scene_actor_slots["ball"],
            ))
    env_ids_int32 = actor_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._all_root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    _, _, heading = euler_from_quat(env.root_states[:, 3:7])
    env.command_heading[:] = heading
    env.command_start_xy[:] = env.root_states[:, :2]
    env.line_reference_active[:] = True
    if event == "obstacle_reset":
        obstacle_x = float(env.obstacle_world_x[0].item())
        obstacle_y = float(env.env_origins[0, 1].item())
        env.set_camera(
            [obstacle_x + 0.75, obstacle_y - 1.15, 0.62],
            [obstacle_x, obstacle_y, 0.10],
        )
    elif event == "flat_reset":
        origin_x = float(env.env_origins[0, 0].item())
        origin_y = float(env.env_origins[0, 1].item())
        env.set_camera(
            [origin_x + 1.20, origin_y - 1.35, 0.70],
            [origin_x + 0.35, origin_y, 0.12],
        )
    elif event == "ball_scene_reset":
        ball_x = float(env.ball_root_states[0, 0].item())
        ball_y = float(env.ball_root_states[0, 1].item())
        env.set_camera(
            [ball_x + 0.65, ball_y - 1.00, 0.52],
            [ball_x - 0.10, ball_y, 0.08],
        )


def _draw_goal(env, visible):
    if env.viewer is None:
        return
    env.gym.clear_lines(env.viewer)
    if not visible or env.ball_root_states is None:
        return
    scene = env.cfg.scene
    kick = env.cfg.skill_curriculum
    front_x = float(env.ball_start_x[0].item()) + kick.ball_goal_distance_m
    back_x = front_x + scene.goal_depth_m
    center_y = float(env.env_origins[0, 1].item()) + kick.ball_spawn_lateral_center_m
    left_y = center_y + scene.goal_width_m / 2.0
    right_y = center_y - scene.goal_width_m / 2.0
    top_z = scene.goal_height_m
    lines = []
    for x_pos in (front_x, back_x):
        lines.extend((
            ((x_pos, left_y, 0.0), (x_pos, left_y, top_z)),
            ((x_pos, right_y, 0.0), (x_pos, right_y, top_z)),
            ((x_pos, left_y, top_z), (x_pos, right_y, top_z)),
        ))
    lines.extend((
        ((front_x, left_y, 0.0), (back_x, left_y, 0.0)),
        ((front_x, right_y, 0.0), (back_x, right_y, 0.0)),
        ((front_x, left_y, top_z), (back_x, left_y, top_z)),
        ((front_x, right_y, top_z), (back_x, right_y, top_z)),
    ))
    vertices = np.asarray(lines, dtype=np.float32).reshape(-1, 3)
    colors = np.tile(np.asarray(scene.goal_color, dtype=np.float32), (len(lines), 1))
    env.gym.add_lines(
        env.viewer, env.envs[0], len(lines), vertices, colors
    )


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
    if not 0.0 <= demo.locomotion_blend <= 1.0:
        raise ValueError("locomotion_blend must be between 0 and 1")
    if demo.policy_transition_s < 0.0:
        raise ValueError("policy_transition_s must be non-negative")
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
    if demo.heading_hold_kp is not None:
        env_cfg.skill_curriculum.heading_hold_kp = demo.heading_hold_kp
    if demo.cross_track_heading_kp is not None:
        env_cfg.skill_curriculum.cross_track_heading_kp = (
            demo.cross_track_heading_kp
        )
    if demo.demo == "fall_recovery":
        env_cfg.skill_curriculum.forced_stage = 3
    elif demo.demo == "diagonal_motion":
        env_cfg.skill_curriculum.forced_stage = 4
    elif demo.demo == "obstacle_crossing":
        _configure_obstacle_env(env_cfg)
    elif demo.demo == "ball_kick":
        _configure_ball_env(env_cfg)
    elif demo.demo == "full_sequence":
        _configure_obstacle_env(env_cfg)
        # The remote GPU viewer can deadlock when heightfield rendering and two
        # extra actors are combined. The full-chain demo uses the visible box
        # as its physical obstacle; quantitative stage-8 evaluation remains on
        # the trained heightfield.
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.obstacle_course = False
        env_cfg.scene.ball_enabled = True
    if demo.demo == "full_sequence":
        env_cfg.env.episode_length_s = 200.0
    env_cfg.viewer.pos = [1.20, -1.20, 0.65] if demo.fixed_camera else [0.45, -0.45, 0.35]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.13]

    train_cfg.runner.resume = False
    train_cfg.runner.load_run = (
        -1 if args.load_run is None or str(args.load_run) == "-1" else args.load_run
    )
    train_cfg.runner.checkpoint = -1

    if demo.checkpoint_path_override:
        initial_checkpoint = os.path.abspath(demo.checkpoint_path_override)
    elif demo.demo == "full_sequence":
        initial_checkpoint = os.path.abspath(demo.stage5_checkpoint_path)
    elif demo.demo == "obstacle_crossing":
        initial_checkpoint = os.path.abspath(demo.stage8_checkpoint_path)
    else:
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
    policy_runners = [runner]
    policies = {"primary": policy}
    if demo.demo == "full_sequence":
        policies["stage5"] = policy
        for key, path in (
            ("stage6", demo.stage6_checkpoint_path),
            ("stage7", demo.stage7_checkpoint_path),
            ("stage8", demo.stage8_checkpoint_path),
            ("stage10", demo.stage10_checkpoint_path),
        ):
            named_runner, named_policy, named_iteration = _load_named_policy(
                env, args, train_cfg, path
            )
            policy_runners.append(named_runner)
            policies[key] = named_policy
            print(
                f"Loaded {key} expert {os.path.abspath(path)} "
                f"at iteration {named_iteration}"
            )
    locomotion_policy = None
    if demo.locomotion_checkpoint_path:
        locomotion_runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            log_root=None,
        )
        _load_policy_checkpoint(
            locomotion_runner,
            demo.locomotion_checkpoint_path,
            env.device,
        )
        locomotion_policy = locomotion_runner.get_inference_policy(device=env.device)
        print(
            "Locomotion expert: "
            f"{demo.locomotion_checkpoint_path}; blend={demo.locomotion_blend:.2f}"
        )
    loaded_checkpoint = initial_checkpoint
    print(f"Visualizing one MiniDuck from {loaded_checkpoint}")

    obs = env.get_observations()
    if demo.demo == "emergency_stop":
        base_sequence = tuple((name, command, duration, 1 if command[0] else 0, "nominal") for name, command, duration in EMERGENCY_STOP_SEQUENCE)
    elif demo.demo == "locomotion":
        base_sequence = tuple((name, command, 5.0, 1, "nominal") for name, command in LOCOMOTION_COMMANDS)
    elif demo.demo == "squat":
        base_sequence = SQUAT_SEQUENCE
    elif demo.demo == "action_switch":
        base_sequence = ACTION_SWITCH_SEQUENCE
    elif demo.demo == "diagonal_motion":
        base_sequence = DIAGONAL_SEQUENCE
    elif demo.demo == "obstacle_crossing":
        base_sequence = OBSTACLE_SEQUENCE
    elif demo.demo == "ball_kick":
        base_sequence = BALL_SEQUENCE
    elif demo.demo == "full_sequence":
        sequence = FULL_SEQUENCE
    else:
        base_sequence = (("fall_recovery", (0.0, 0.0, 0.0), 1000.0, 3, "nominal"),)
    if demo.demo != "full_sequence":
        if demo.demo == "diagonal_motion":
            stage_key = "diagonal_motion"
        elif demo.demo == "fall_recovery":
            stage_key = "fall_recovery"
        elif demo.demo == "obstacle_crossing":
            stage_key = "obstacle_crossing"
        elif demo.demo == "ball_kick":
            stage_key = "ball_kick"
        else:
            stage_key = "action_switch"
        sequence = tuple(
            (*item, "primary", None, stage_key) for item in base_sequence
        )
    sequence_steps = [max(1, round(item[2] / env.dt)) for item in sequence]
    cycle_steps = sum(sequence_steps)
    reload_steps = max(1, int(2.0 / env.dt))
    step = 0
    previous_command_name = None
    current_policy_key = None
    transition_steps = max(1, round(demo.policy_transition_s / env.dt))
    transition_step = transition_steps
    transition_start_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )
    last_actions = transition_start_actions.clone()

    while True:
        cycle_step = step % cycle_steps
        command_index = 0
        while cycle_step >= sequence_steps[command_index]:
            cycle_step -= sequence_steps[command_index]
            command_index += 1
        (
            command_name,
            command,
            _,
            skill_mode,
            height_key,
            policy_key,
            event,
            stage_key,
        ) = sequence[command_index]
        target_height = (
            env.cfg.skill_curriculum.squat_body_height_m
            if height_key == "squat"
            else env.cfg.skill_curriculum.nominal_body_height_m
        )
        if command_name != previous_command_name:
            if (
                previous_command_name == "ball_approach"
                and command_name == "ball_kick"
            ):
                ball_relative = env._ball_relative_body()[0]
                approach_progress = (
                    env.ball_root_states[0, 0] - env.ball_start_x[0]
                )
                print(
                    "Ball approach handoff; "
                    f"relative_x_m={float(ball_relative[0].item()):.3f}; "
                    f"relative_y_m={float(ball_relative[1].item()):.3f}; "
                    f"pre_kick_ball_progress_m={float(approach_progress.item()):.3f}"
                )
            print(
                f"Demo phase: {command_name}; command={command}; "
                f"expert={policy_key}"
            )
            env.demo_action_stage_key = stage_key
            env.demo_recovery_active[:] = command_name in (
                "fall",
                "fall_recovery",
            )
            env.obstacle_task_active[:] = skill_mode == env.SKILL_OBSTACLE
            env.ball_task_active[:] = skill_mode == env.SKILL_KICK
            if event is not None:
                _set_demo_pose(env, event)
                env.compute_observations()
                obs = env.get_observations()
                last_actions.zero_()
            env.skill_mode[:] = skill_mode
            env._schedule_skill_height(
                torch.arange(env.num_envs, device=env.device), target_height
            )
            if not torch.any(env.line_reference_active):
                _, _, yaw = euler_from_quat(env.base_quat)
                env.command_heading[:] = yaw
                env.command_start_xy[:] = env.root_states[:, :2]
                env.line_reference_active[:] = True
            if policy_key != current_policy_key:
                transition_start_actions = last_actions.clone()
                transition_step = 0
                current_policy_key = policy_key
            previous_command_name = command_name
        _draw_goal(
            env,
            command_name in ("ball_approach", "ball_kick"),
        )
        env.commands[0, :3] = torch.tensor(command, device=env.device)
        if command_name == "ball_approach":
            ball_relative = env._ball_relative_body()[0]
            lateral_error = (
                float(ball_relative[1].item())
                - env.cfg.skill_curriculum.ball_demo_handoff_lateral_m
            )
            env.commands[0, 1] = max(
                -0.04,
                min(
                    0.04,
                    env.cfg.skill_curriculum.ball_demo_lateral_kp
                    * lateral_error,
                ),
            )
            if (
                float(ball_relative[0].item()) <= 0.10
                and abs(lateral_error)
                > env.cfg.skill_curriculum.ball_demo_handoff_lateral_tolerance_m
            ):
                env.commands[0, 0] = 0.0
        env.skill_mode[:] = skill_mode
        env._apply_straight_heading_hold()
        env._apply_diagonal_heading_hold()
        obs[:, 9:12] = env._command_observation()
        obs[:, 62:64] = env._skill_observation()

        with torch.no_grad():
            active_policy = policies[policy_key]
            direct_actions = active_policy(obs.detach())
            symmetry_blend = (
                1.0 if demo.symmetric_inference else demo.symmetry_blend
            )
            if symmetry_blend > 0.0:
                symmetric_actions = _symmetric_policy_action(
                    active_policy, obs.detach(), direct_action=direct_actions
                )
                actions = torch.lerp(
                    direct_actions, symmetric_actions, symmetry_blend
                )
            else:
                actions = direct_actions
            if locomotion_policy is not None and skill_mode == env.SKILL_LOCOMOTION:
                expert_actions = locomotion_policy(obs.detach())
                actions = torch.lerp(
                    actions,
                    expert_actions,
                    demo.locomotion_blend,
                )
            if transition_step < transition_steps:
                alpha = float(transition_step + 1) / transition_steps
                actions = torch.lerp(
                    transition_start_actions,
                    actions,
                    alpha,
                )
                transition_step += 1
            last_actions = actions.detach().clone()
        obs, _, _, dones, _ = env.step(actions.detach())

        if demo.demo == "full_sequence" and command_name == "ball_approach":
            ball_relative = env._ball_relative_body()[0]
            lateral_error = (
                float(ball_relative[1].item())
                - env.cfg.skill_curriculum.ball_demo_handoff_lateral_m
            )
            approach_complete = (
                float(ball_relative[0].item())
                <= env.cfg.skill_curriculum.ball_demo_handoff_distance_m
                and abs(lateral_error)
                <= env.cfg.skill_curriculum.ball_demo_handoff_lateral_tolerance_m
            )
            if approach_complete:
                print(
                    "Ball approach completed; "
                    f"relative_x_m={float(ball_relative[0].item()):.3f}; "
                    f"relative_y_m={float(ball_relative[1].item()):.3f}"
                )
                step += sequence_steps[command_index] - cycle_step - 1

        if dones[0]:
            if (
                demo.demo == "full_sequence"
                and command_name == "obstacle_crossing"
            ):
                print(
                    "Obstacle trial completed; "
                    f"success={bool(env.obstacle_success_latched[0].item())}"
                )
                step += sequence_steps[command_index] - cycle_step - 1
            elif demo.demo == "full_sequence" and command_name == "ball_kick":
                print(
                    "Ball trial completed; "
                    f"success={bool(env.ball_success_latched[0].item())}; "
                    f"progress_m={float(env.ball_progress_latched[0].item()):.3f}"
                )
                step += sequence_steps[command_index] - cycle_step - 1
            elif demo.demo == "full_sequence":
                step += cycle_steps - (step % cycle_steps) - 1
            env.commands[0, :3] = torch.tensor(command, device=env.device)
            env.line_reference_active[:] = False
            previous_command_name = None

        if not demo.fixed_camera:
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
            latest_checkpoint = (
                initial_checkpoint
                if demo.checkpoint_path_override
                else _checkpoint_path(train_cfg, args)
            )
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
