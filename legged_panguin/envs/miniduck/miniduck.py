# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import pickle

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    euler_from_quat,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    torch_rand_float,
    torch_wrap_to_pi_minuspi,
)
from miniduck_api.action_curriculum import ACTION_STAGES, active_stage_for_step
from legged_panguin.envs import LeggedRobot


class MiniDuck(LeggedRobot):
    SKILL_STAND = 0
    SKILL_LOCOMOTION = 1
    SKILL_SQUAT = 2
    SKILL_RECOVERY = 3
    SKILL_OBSTACLE = 4
    SKILL_KICK = 5

    def _create_additional_assets(self):
        assets = []
        self.scene_actor_slots = {}
        self.scene_body_slots = {}
        self.additional_actor_handles = {}
        scene = self.cfg.scene
        if scene.obstacle_enabled:
            height = float(min(self.cfg.terrain.obstacle_height_range))
            options = gymapi.AssetOptions()
            options.fix_base_link = True
            options.disable_gravity = True
            asset = self.gym.create_box(
                self.sim,
                self.cfg.terrain.obstacle_depth_m,
                self.cfg.terrain.obstacle_width_m,
                height,
                options,
            )
            assets.append(("obstacle", asset, height))
        if scene.ball_enabled:
            options = gymapi.AssetOptions()
            volume = 4.0 / 3.0 * np.pi * scene.ball_radius_m ** 3
            options.density = scene.ball_mass_kg / volume
            asset = self.gym.create_sphere(
                self.sim, scene.ball_radius_m, options
            )
            shape_props = self.gym.get_asset_rigid_shape_properties(asset)
            for prop in shape_props:
                prop.friction = scene.ball_friction
                prop.restitution = scene.ball_restitution
            self.gym.set_asset_rigid_shape_properties(asset, shape_props)
            assets.append(("ball", asset, scene.ball_radius_m))
        for index, (name, _, _) in enumerate(assets):
            self.scene_actor_slots[name] = index + 1
            self.scene_body_slots[name] = self.num_bodies + index
            self.additional_actor_handles[name] = []
        return assets

    def _create_additional_actors(self, env_handle, env_id, assets):
        if not assets:
            return
        for name, asset, vertical_center in assets:
            pose = gymapi.Transform()
            origin = self.env_origins[env_id]
            if name == "obstacle":
                pose.p = gymapi.Vec3(
                    float(origin[0]) + self.cfg.skill_curriculum.obstacle_offset_m,
                    float(origin[1]),
                    vertical_center / 2.0,
                )
                color = self.cfg.scene.obstacle_color
            else:
                pose.p = gymapi.Vec3(
                    float(origin[0]) + self.cfg.skill_curriculum.ball_spawn_distance_m,
                    float(origin[1]) + self.cfg.skill_curriculum.ball_spawn_lateral_center_m,
                    vertical_center,
                )
                color = self.cfg.scene.ball_color
            collision_filter = 1 if name == "obstacle" else 0
            handle = self.gym.create_actor(
                env_handle, asset, pose, name, env_id, collision_filter, 0
            )
            self.gym.set_rigid_body_color(
                env_handle,
                handle,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(*color),
            )
            self.additional_actor_handles[name].append(handle)

    def _init_buffers(self):
        super()._init_buffers()
        self.base_command_ranges = {
            name: list(values) for name, values in self.command_ranges.items()
        }
        init_quat = self.base_init_state[3:7].unsqueeze(0)
        self.target_projected_gravity = quat_rotate_inverse(init_quat, self.gravity_vec[:1])
        self.commanded_actions = torch.zeros_like(self.actions)
        self.commanded_action_history_1 = torch.zeros_like(self.actions)
        self.commanded_action_history_2 = torch.zeros_like(self.actions)
        _, _, yaw = euler_from_quat(self.base_quat)
        self.command_heading = yaw.clone()
        self.command_start_xy = self.root_states[:, :2].clone()
        self.emergency_probe_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.line_reference_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.skill_mode = torch.full(
            (self.num_envs,), self.SKILL_STAND, dtype=torch.long, device=self.device
        )
        nominal_height = self.cfg.skill_curriculum.nominal_body_height_m
        self.target_base_height = torch.full(
            (self.num_envs,), nominal_height, device=self.device
        )
        self.skill_start_height = self.target_base_height.clone()
        self.skill_goal_height = self.target_base_height.clone()
        self.skill_transition_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.squat_target_low = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.recovery_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.recovery_upright_steps = torch.zeros_like(self.recovery_elapsed_steps)
        obstacle_enabled = self._stage_is("obstacle_crossing")
        self.obstacle_task_active = torch.full(
            (self.num_envs,), obstacle_enabled, dtype=torch.bool, device=self.device
        )
        self.demo_action_stage_key = None
        self.demo_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obstacle_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.obstacle_success_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obstacle_progress_latched = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        obstacle_cfg = self.cfg.skill_curriculum
        self.obstacle_world_x = (
            self.env_origins[:, 0] + obstacle_cfg.obstacle_offset_m
        )
        if hasattr(self, "terrain") and hasattr(self.terrain, "obstacle_heights"):
            height_grid = torch.as_tensor(
                self.terrain.obstacle_heights,
                dtype=torch.float32,
                device=self.device,
            )
            self.obstacle_heights = height_grid[
                self.terrain_levels, self.terrain_types
            ]
        else:
            self.obstacle_heights = torch.full(
                (self.num_envs,),
                float(min(self.cfg.terrain.obstacle_height_range)),
                device=self.device,
            )
        body_mask = torch.ones(
            self.contact_forces.shape[1], dtype=torch.bool, device=self.device
        )
        body_mask[self.feet_indices] = False
        self.obstacle_body_indices = torch.arange(
            self.contact_forces.shape[1], device=self.device
        )[body_mask]
        ball_enabled = "ball" in self.scene_actor_slots
        self.ball_task_active = torch.full(
            (self.num_envs,),
            ball_enabled and self._stage_is("ball_kick"),
            dtype=torch.bool,
            device=self.device,
        )
        self.ball_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.ball_success_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.ball_progress_latched = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if ball_enabled:
            self.ball_root_states = self.actor_root_states[
                :, self.scene_actor_slots["ball"], :
            ]
            self.ball_start_x = self.ball_root_states[:, 0].clone()
        else:
            self.ball_root_states = None
            self.ball_start_x = torch.zeros(self.num_envs, device=self.device)

        max_action_delay = self.cfg.domain_rand.max_action_delay
        self.action_delay_buffer = torch.zeros(
            max_action_delay + 1,
            self.num_envs,
            self.num_actions,
            device=self.device,
        )
        self.action_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        max_imu_delay = self.cfg.domain_rand.max_imu_delay
        self.imu_history = torch.zeros(
            max_imu_delay + 1,
            self.num_envs,
            9,
            device=self.device,
        )
        self.imu_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.gyro_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.gravity_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.accel_bias = torch.zeros(self.num_envs, 3, device=self.device)

        self.motor_strengths = torch.ones(
            self.num_envs, self.num_actions, device=self.device
        )
        self.kp_factors = torch.ones_like(self.motor_strengths)
        self.kd_factors = torch.ones_like(self.motor_strengths)
        self.joint_target_offsets = torch.zeros_like(self.motor_strengths)
        self.encoder_offsets = torch.zeros_like(self.motor_strengths)
        self.max_motor_velocities = torch.full_like(
            self.motor_strengths, self.cfg.domain_rand.nominal_motor_velocity
        )
        self.previous_motor_targets = self.default_dof_pos.repeat(self.num_envs, 1)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._all_rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies_per_env, 13
        )
        self.rigid_body_state = self._all_rigid_body_state[:, :self.num_bodies, :]
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._load_teacher_reference()
        self.gait_phase_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.gait_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        self.gait_last_contacts = torch.zeros_like(self.gait_contacts)
        self.gait_first_contacts = torch.zeros_like(self.gait_contacts, dtype=torch.float)
        self.gait_contact_switch = torch.zeros(self.num_envs, device=self.device)
        self.current_teacher_reference = torch.zeros(
            self.num_envs, 60, device=self.device
        )
        self.home_feet_z = torch.mean(
            self.rigid_body_state[:, self.feet_indices, 2], dim=0
        )
        self._update_command_range_schedule()
        self._randomize_dynamic_properties(
            torch.arange(self.num_envs, device=self.device)
        )

    def _load_teacher_reference(self):
        path = getattr(self.cfg.rewards, "teacher_reference_path", "")
        self.use_teacher_reference = bool(path) and os.path.exists(path)
        self.teacher_reference_period_steps = 1
        self.teacher_reference_commands = torch.zeros(1, 3, device=self.device)
        self.teacher_reference_samples = torch.zeros(1, 1, 60, device=self.device)
        self.teacher_command_scale = torch.ones(3, device=self.device)
        self.teacher_home_target = self.default_dof_pos.clone()

        if not self.use_teacher_reference:
            if path:
                print(f"MiniDuck teacher reference not found: {path}")
            return

        with open(path, "rb") as file:
            data = pickle.load(file)

        commands = []
        samples = []
        home_index = 0
        sorted_items = sorted(
            data.items(),
            key=lambda item: tuple(float(value) for value in item[0].split("_")),
        )
        for index, (name, entry) in enumerate(sorted_items):
            command = [float(value) for value in name.split("_")]
            if command == [0.0, 0.0, 0.0]:
                home_index = index
            coefficients = np.asarray(
                [entry["coefficients"][f"dim_{dim}"] for dim in range(60)],
                dtype=np.float32,
            )
            period_steps = int(entry["nb_steps_in_period"])
            phase_samples = []
            flipped = coefficients[:, ::-1]
            for phase_step in range(period_steps):
                t = np.clip(phase_step / period_steps, 0.0, 1.0)
                phase_samples.append(
                    [np.polyval(poly_coeffs, t) for poly_coeffs in flipped]
                )
            commands.append(command)
            samples.append(phase_samples)

        self.teacher_reference_period_steps = len(samples[0])
        self.teacher_reference_commands = torch.tensor(
            commands, dtype=torch.float32, device=self.device
        )
        self.teacher_reference_samples = torch.tensor(
            samples, dtype=torch.float32, device=self.device
        )
        self.teacher_command_scale = torch.tensor(
            self.cfg.rewards.teacher_command_scale,
            dtype=torch.float32,
            device=self.device,
        )
        self.teacher_home_target = self.teacher_reference_samples[
            home_index, 0, 50:60
        ].clone()
        print(
            "MiniDuck loaded BEST_WALK teacher reference: "
            f"{len(commands)} commands, {self.teacher_reference_period_steps} phases"
        )

    def _sample_teacher_reference(self):
        if not self.use_teacher_reference:
            return self.current_teacher_reference.zero_()

        normalized_error = (
            self.commands[:, None, :3] - self.teacher_reference_commands[None, :, :]
        ) / self.teacher_command_scale
        reference_ids = torch.argmin(
            torch.sum(torch.square(normalized_error), dim=-1), dim=1
        )
        phase_ids = self.gait_phase_steps % self.teacher_reference_period_steps
        self.current_teacher_reference = self.teacher_reference_samples[
            reference_ids, phase_ids
        ]
        return self.current_teacher_reference

    def _gait_phase_observation(self):
        phase = (
            self.gait_phase_steps.float()
            / max(self.teacher_reference_period_steps, 1)
            * (2.0 * np.pi)
        )
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)

    def _skill_observation(self):
        features = self._gait_phase_observation()
        stand = self.skill_mode == self.SKILL_STAND
        squat = self.skill_mode == self.SKILL_SQUAT
        recovery = self.skill_mode == self.SKILL_RECOVERY
        features[stand, 0] = 1.0
        features[stand, 1] = 0.0
        if torch.any(squat):
            cfg = self.cfg.skill_curriculum
            span = max(cfg.nominal_body_height_m - cfg.squat_body_height_m, 1.0e-6)
            depth = torch.clamp(
                (cfg.nominal_body_height_m - self.target_base_height[squat]) / span,
                min=0.0,
                max=1.0,
            )
            features[squat, 0] = 1.0 - 2.0 * depth
            features[squat, 1] = 1.0
        features[recovery, 0] = -1.0
        features[recovery, 1] = -1.0
        return features

    def _command_observation(self):
        features = self.commands[:, :3] * self.commands_scale
        obstacle = (
            (self.skill_mode == self.SKILL_OBSTACLE)
            & self.obstacle_task_active
        )
        if torch.any(obstacle):
            distance = self.obstacle_world_x[obstacle] - self.root_states[obstacle, 0]
            features[obstacle, 1] = torch.clamp(distance / 0.80, -1.0, 1.0)
        kick = (
            (self.skill_mode == self.SKILL_KICK)
            & self.ball_task_active
            & (self.ball_root_states is not None)
        )
        if torch.any(kick):
            relative_world = torch.zeros(
                self.num_envs, 3, dtype=self.root_states.dtype, device=self.device
            )
            relative_world[:, :2] = (
                self.ball_root_states[:, :2] - self.root_states[:, :2]
            )
            relative_body = quat_rotate_inverse(self.base_quat, relative_world)
            features[kick, 1] = torch.clamp(
                relative_body[kick, 0] / 0.50, -1.0, 1.0
            )
            features[kick, 2] = torch.clamp(relative_body[kick, 1] / 0.20, -1.0, 1.0)
        return features

    def _schedule_skill_height(self, env_ids, goal_height):
        if len(env_ids) == 0:
            return
        self.skill_start_height[env_ids] = self.target_base_height[env_ids]
        if torch.is_tensor(goal_height):
            self.skill_goal_height[env_ids] = goal_height
        else:
            self.skill_goal_height[env_ids] = float(goal_height)
        self.skill_transition_step[env_ids] = 0

    def _current_squat_goal(self):
        cfg = self.cfg.skill_curriculum
        stage_one_start = cfg.start_step + cfg.stage_steps[0]
        progress = self._ramp_progress(
            self.common_step_counter,
            stage_one_start,
            cfg.stage_steps[1],
        )
        return cfg.squat_start_body_height_m + (
            cfg.squat_body_height_m - cfg.squat_start_body_height_m
        ) * progress

    def _update_skill_targets(self):
        transition_steps = max(
            1,
            round(self.cfg.skill_curriculum.height_transition_s / self.dt),
        )
        progress = torch.clamp(
            self.skill_transition_step.float() / transition_steps,
            min=0.0,
            max=1.0,
        )
        # Smoothstep keeps the requested squat/stand transition free of velocity jumps.
        blend = progress * progress * (3.0 - 2.0 * progress)
        self.target_base_height = self.skill_start_height + blend * (
            self.skill_goal_height - self.skill_start_height
        )
        self.skill_transition_step = torch.clamp(
            self.skill_transition_step + 1,
            max=transition_steps,
        )

    def _apply_straight_heading_hold(self):
        if not self._stage_is("emergency_stop_stand", "squat", "action_switch"):
            return
        active = self.line_reference_active
        if not torch.any(active):
            return
        self.commands[active, 1:3] = 0.0
        moving = active & (
            torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x")
        )
        if not torch.any(moving):
            return
        _, _, yaw = euler_from_quat(self.base_quat)
        heading_error = torch_wrap_to_pi_minuspi(yaw - self.command_heading)
        cfg = self.cfg.skill_curriculum
        delta_xy = self.root_states[:, :2] - self.command_start_xy
        lateral = (
            -delta_xy[:, 0] * torch.sin(self.command_heading)
            + delta_xy[:, 1] * torch.cos(self.command_heading)
        )
        world_velocity = self.root_states[:, 7:9]
        lateral_velocity = (
            -world_velocity[:, 0] * torch.sin(self.command_heading)
            + world_velocity[:, 1] * torch.cos(self.command_heading)
        )
        lateral_correction = -cfg.line_hold_kp * lateral - (
            cfg.line_hold_kd * lateral_velocity
        )
        self.commands[moving, 1] = torch.clamp(
            lateral_correction[moving],
            -cfg.line_hold_max_lateral_mps,
            cfg.line_hold_max_lateral_mps,
        )
        cross_track = (
            cfg.cross_track_heading_kp
            * lateral
            * torch.sign(self.commands[:, 0])
        )
        correction = -cfg.heading_hold_kp * heading_error - cross_track - (
            cfg.heading_hold_kd * self.base_ang_vel[:, 2]
        )
        self.commands[moving, 2] = torch.clamp(
            correction[moving],
            -cfg.heading_hold_max_yaw_rate,
            cfg.heading_hold_max_yaw_rate,
        )

    def _apply_diagonal_heading_hold(self):
        if not self._stage_is("diagonal_motion"):
            return
        active = (
            (torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x"))
            & (torch.abs(self.commands[:, 1]) >= self._current_min_abs("lin_vel_y"))
        )
        if not torch.any(active):
            return
        _, _, yaw = euler_from_quat(self.base_quat)
        heading_error = torch_wrap_to_pi_minuspi(yaw - self.command_heading)
        cfg = self.cfg.skill_curriculum
        correction = (
            -cfg.heading_hold_kp * heading_error
            - cfg.heading_hold_kd * self.base_ang_vel[:, 2]
        )
        self.commands[active, 2] = torch.clamp(
            correction[active],
            -cfg.heading_hold_max_yaw_rate,
            cfg.heading_hold_max_yaw_rate,
        )

    def set_external_skill(self, mode, target_height=None, env_ids=None):
        """Set a deterministic skill target for evaluation and visualization."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.skill_mode[env_ids] = int(mode)
        if target_height is None:
            target_height = self.cfg.skill_curriculum.nominal_body_height_m
        self.target_base_height[env_ids] = float(target_height)
        self.skill_start_height[env_ids] = float(target_height)
        self.skill_goal_height[env_ids] = float(target_height)
        self.skill_transition_step[env_ids] = max(
            1, round(self.cfg.skill_curriculum.height_transition_s / self.dt)
        )

    def _current_foot_contacts(self):
        return self.contact_forces[:, self.feet_indices, 2] > 1.0

    def _update_gait_reference_state(self):
        next_phase = (
            self.gait_phase_steps + 1
        ) % self.teacher_reference_period_steps
        action_stage = self._current_action_stage()
        if action_stage is not None and action_stage.key in (
            "emergency_stop_stand",
            "squat",
            "action_switch",
        ):
            stationary_command = (
                (torch.abs(self.commands[:, 0]) < 0.02)
                & (torch.abs(self.commands[:, 1]) < 0.02)
                & (torch.abs(self.commands[:, 2]) < 0.1)
            )
            locomotion_skill = (
                (self.skill_mode == self.SKILL_LOCOMOTION)
                | (self.skill_mode == self.SKILL_OBSTACLE)
            )
            freeze_phase = stationary_command | ~locomotion_skill
            self.gait_phase_steps = torch.where(
                freeze_phase,
                torch.zeros_like(next_phase),
                next_phase,
            )
        else:
            self.gait_phase_steps = next_phase
        contacts = self._current_foot_contacts()
        self.gait_contacts = contacts
        previous = self.gait_last_contacts
        self.gait_first_contacts = ((~previous) & contacts).float()
        self.gait_contact_switch = torch.clamp(
            torch.sum(torch.abs(contacts.float() - previous.float()), dim=1),
            max=1.0,
        )
        self.gait_last_contacts = contacts
        self._sample_teacher_reference()

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        raw_actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.commanded_action_history_2[:] = self.commanded_action_history_1
        self.commanded_action_history_1[:] = self.commanded_actions
        self.commanded_actions[:] = raw_actions

        self.action_delay_buffer = torch.roll(
            self.action_delay_buffer, shifts=1, dims=0
        )
        self.action_delay_buffer[0] = raw_actions
        delay_strength = self._domain_rand_strength()
        active_max_delay = int(
            round(self.cfg.domain_rand.max_action_delay * delay_strength)
        )
        if active_max_delay > 0:
            self.action_delay_steps = torch.randint(
                0,
                active_max_delay + 1,
                (self.num_envs,),
                device=self.device,
            )
        else:
            self.action_delay_steps.zero_()
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed_actions = self.action_delay_buffer[
            self.action_delay_steps, env_ids
        ]
        return super().step(delayed_actions)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "commanded_actions"):
            return
        self._update_command_range_schedule()
        self.extras["episode"]["max_command_x"] = float(self.command_ranges["lin_vel_x"][1])
        self.extras["episode"]["max_command_y"] = float(self.command_ranges["lin_vel_y"][1])
        self.extras["episode"]["max_command_yaw"] = float(self.command_ranges["ang_vel_yaw"][1])
        self.extras["episode"]["push_max_vel_xy"] = float(self._current_push_max_vel())
        action_stage = self._current_action_stage()
        self.extras["episode"]["action_stage_id"] = float(
            action_stage.index if action_stage is not None else -1
        )

        self.commanded_actions[env_ids] = 0.0
        self.commanded_action_history_1[env_ids] = 0.0
        self.commanded_action_history_2[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.action_delay_buffer[:, env_ids] = 0.0
        self.previous_motor_targets[env_ids] = self.default_dof_pos
        self.gait_phase_steps[env_ids] = 0
        self.gait_contacts[env_ids] = False
        self.gait_last_contacts[env_ids] = False
        self.gait_first_contacts[env_ids] = 0.0
        self.gait_contact_switch[env_ids] = 0.0
        self.current_teacher_reference[env_ids] = 0.0
        self.emergency_probe_active[env_ids] = False
        self.line_reference_active[env_ids] = False
        self.skill_mode[env_ids] = self.SKILL_STAND
        nominal_height = self.cfg.skill_curriculum.nominal_body_height_m
        self.target_base_height[env_ids] = nominal_height
        self.skill_start_height[env_ids] = nominal_height
        self.skill_goal_height[env_ids] = nominal_height
        self.skill_transition_step[env_ids] = 0
        self.squat_target_low[env_ids] = False
        self.recovery_elapsed_steps[env_ids] = 0
        self.recovery_upright_steps[env_ids] = 0
        self.obstacle_elapsed_steps[env_ids] = 0
        self.obstacle_task_active[env_ids] = self._stage_is("obstacle_crossing")
        self.ball_elapsed_steps[env_ids] = 0
        self.ball_task_active[env_ids] = (
            self.ball_root_states is not None and self._stage_is("ball_kick")
        )
        self._randomize_dynamic_properties(env_ids)

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13]
        )
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )
        _, _, yaw = euler_from_quat(self.base_quat[env_ids])
        self.command_heading[env_ids] = yaw
        self.command_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        initial_imu = torch.cat(
            (
                self.base_ang_vel[env_ids],
                self.projected_gravity[env_ids],
                -self.projected_gravity[env_ids] * 9.81,
            ),
            dim=-1,
        )
        self.imu_history[:, env_ids] = initial_imu.unsqueeze(0)
        # LeggedRobot samples commands before MiniDuck-specific buffers are reset.
        # Resample once more so the command and high-level skill stay consistent.
        self._resample_commands(env_ids)

    def _domain_rand_strength(self):
        warmup = self.cfg.domain_rand.curriculum_warmup_steps
        ramp = max(self.cfg.domain_rand.curriculum_ramp_steps, 1)
        return min(max((self.common_step_counter - warmup) / ramp, 0.0), 1.0)

    def _scaled_uniform(self, low, high, shape, strength):
        low = 1.0 + (low - 1.0) * strength
        high = 1.0 + (high - 1.0) * strength
        return torch_rand_float(low, high, shape, device=self.device)

    def _randomize_dynamic_properties(self, env_ids):
        if len(env_ids) == 0:
            return
        strength = self._domain_rand_strength()
        count = len(env_ids)
        cfg = self.cfg.domain_rand

        self.motor_strengths[env_ids] = self._scaled_uniform(
            *cfg.motor_strength_range,
            (count, self.num_actions),
            strength,
        )
        self.kp_factors[env_ids] = self._scaled_uniform(
            *cfg.kp_factor_range,
            (count, self.num_actions),
            strength,
        )
        self.kd_factors[env_ids] = self._scaled_uniform(
            *cfg.kd_factor_range,
            (count, self.num_actions),
            strength,
        )
        motor_velocity_low = (
            cfg.nominal_motor_velocity
            + (cfg.motor_velocity_range[0] - cfg.nominal_motor_velocity) * strength
        )
        motor_velocity_high = (
            cfg.nominal_motor_velocity
            + (cfg.motor_velocity_range[1] - cfg.nominal_motor_velocity) * strength
        )
        self.max_motor_velocities[env_ids] = torch_rand_float(
            motor_velocity_low,
            motor_velocity_high,
            (count, self.num_actions),
            device=self.device,
        )

        target_offset = cfg.joint_target_offset * strength
        encoder_offset = cfg.encoder_offset * strength
        self.joint_target_offsets[env_ids] = torch_rand_float(
            -target_offset,
            target_offset,
            (count, self.num_actions),
            device=self.device,
        )
        self.encoder_offsets[env_ids] = torch_rand_float(
            -encoder_offset,
            encoder_offset,
            (count, self.num_actions),
            device=self.device,
        )

        gyro_bias = cfg.gyro_bias * strength
        gravity_bias = cfg.gravity_bias * strength
        accel_bias = cfg.accel_bias * strength
        self.gyro_bias[env_ids] = torch_rand_float(
            -gyro_bias, gyro_bias, (count, 3), device=self.device
        )
        self.gravity_bias[env_ids] = torch_rand_float(
            -gravity_bias, gravity_bias, (count, 3), device=self.device
        )
        self.accel_bias[env_ids] = torch_rand_float(
            -accel_bias, accel_bias, (count, 3), device=self.device
        )

        active_max_imu_delay = int(round(cfg.max_imu_delay * strength))
        if active_max_imu_delay > 0:
            self.imu_delay_steps[env_ids] = torch.randint(
                0,
                active_max_imu_delay + 1,
                (count,),
                device=self.device,
            )
        else:
            self.imu_delay_steps[env_ids] = 0

    def _process_dof_props(self, props, env_id):
        props = props.copy()
        cfg = self.cfg.domain_rand
        if cfg.use_sts3215_xml_dof_defaults:
            names = props.dtype.names
            if "damping" in names:
                props["damping"][:] = cfg.sts3215_dof_damping
            if "friction" in names:
                props["friction"][:] = cfg.sts3215_dof_friction
            if "armature" in names:
                props["armature"][:] = cfg.sts3215_dof_armature
            if "effort" in names:
                props["effort"][:] = cfg.sts3215_effort
            if "velocity" in names:
                props["velocity"][:] = cfg.sts3215_velocity

        props = super()._process_dof_props(props, env_id)
        if cfg.randomize_dof_properties:
            props["friction"] *= np.random.uniform(*cfg.dof_friction_scale_range)
            props["damping"] *= np.random.uniform(*cfg.dof_damping_scale_range)
            props["armature"] *= np.random.uniform(*cfg.dof_armature_scale_range)
        return props

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        cfg = self.cfg.domain_rand
        if not cfg.randomize_link_mass:
            return props

        for prop in props:
            if prop.mass > 0.01:
                prop.mass *= np.random.uniform(*cfg.link_mass_scale_range)

        trunk_id = 1 if len(props) > 1 else 0
        props[trunk_id].mass = max(
            props[trunk_id].mass + np.random.uniform(*cfg.trunk_added_mass_range),
            0.05,
        )
        com_range = cfg.trunk_com_range
        props[trunk_id].com.x += np.random.uniform(-com_range[0], com_range[0])
        props[trunk_id].com.y += np.random.uniform(-com_range[1], com_range[1])
        props[trunk_id].com.z += np.random.uniform(-com_range[2], com_range[2])
        return props

    def _current_action_stage(self):
        cfg = self.cfg.skill_curriculum
        if not cfg.enabled:
            return None
        if cfg.forced_stage is not None:
            forced_stage = int(cfg.forced_stage)
            if not 0 <= forced_stage <= cfg.max_implemented_stage:
                raise ValueError("forced_stage exceeds the implemented curriculum")
            return ACTION_STAGES[forced_stage]
        return active_stage_for_step(
            self.common_step_counter,
            cfg.start_step,
            cfg.stage_steps,
            cfg.max_implemented_stage,
        )

    def _command_curriculum_stage(self):
        cfg = self.cfg.commands
        if self.common_step_counter < cfg.sagittal_phase_steps:
            return "sagittal"
        if self.common_step_counter < (
            cfg.sagittal_phase_steps + cfg.lateral_phase_steps
        ):
            return "lateral"
        if self.common_step_counter < (
            cfg.sagittal_phase_steps
            + cfg.lateral_phase_steps
            + cfg.yaw_phase_steps
        ):
            return "yaw"
        return "mixed"

    @staticmethod
    def _ramp_progress(step, start_step, ramp_steps):
        return min(max((step - start_step) / max(ramp_steps, 1), 0.0), 1.0)

    def _current_command_stage(self):
        cfg = self.cfg.commands
        if self.common_step_counter < cfg.advanced_curriculum_start_step:
            return self._command_curriculum_stage()

        relative_step = self.common_step_counter - cfg.advanced_curriculum_start_step
        if relative_step < cfg.advanced_lateral_rehearsal_steps:
            return "advanced_lateral_rehearsal"
        if relative_step < (
            cfg.advanced_lateral_rehearsal_steps
            + cfg.advanced_yaw_rehearsal_steps
        ):
            return "advanced_yaw_rehearsal"
        if relative_step < (
            cfg.advanced_lateral_rehearsal_steps
            + cfg.advanced_yaw_rehearsal_steps
            + cfg.advanced_balanced_steps
        ):
            return "advanced_balanced"
        return "advanced_final"

    @staticmethod
    def _interpolate_range(base_range, target_range, progress):
        return [
            base_range[0] + (target_range[0] - base_range[0]) * progress,
            base_range[1] + (target_range[1] - base_range[1]) * progress,
        ]

    def _update_command_range_schedule(self):
        cfg = self.cfg.commands
        x_progress = self._ramp_progress(
            self.common_step_counter,
            cfg.advanced_curriculum_start_step,
            cfg.advanced_x_ramp_steps,
        )
        self.command_ranges["lin_vel_x"] = self._interpolate_range(
            self.base_command_ranges["lin_vel_x"],
            cfg.ranges.advanced_lin_vel_x,
            x_progress,
        )

        lateral_progress = self._ramp_progress(
            self.common_step_counter,
            cfg.advanced_curriculum_start_step,
            cfg.advanced_lateral_ramp_steps,
        )
        self.command_ranges["lin_vel_y"] = self._interpolate_range(
            self.base_command_ranges["lin_vel_y"],
            cfg.ranges.advanced_lin_vel_y,
            lateral_progress,
        )

        yaw_progress = self._ramp_progress(
            self.common_step_counter,
            cfg.advanced_curriculum_start_step + cfg.advanced_lateral_rehearsal_steps,
            cfg.advanced_yaw_ramp_steps,
        )
        self.command_ranges["ang_vel_yaw"] = self._interpolate_range(
            self.base_command_ranges["ang_vel_yaw"],
            cfg.ranges.advanced_ang_vel_yaw,
            yaw_progress,
        )

    def _current_min_abs(self, range_name):
        cfg = self.cfg.commands
        if self.common_step_counter < cfg.advanced_curriculum_start_step:
            if range_name == "lin_vel_x":
                return cfg.min_abs_x
            if range_name == "lin_vel_y":
                return cfg.min_abs_y
            return cfg.min_abs_yaw

        if range_name == "lin_vel_x":
            progress = self._ramp_progress(
                self.common_step_counter,
                cfg.advanced_curriculum_start_step,
                cfg.advanced_x_ramp_steps,
            )
            return cfg.min_abs_x + (cfg.advanced_min_abs_x - cfg.min_abs_x) * progress
        if range_name == "lin_vel_y":
            progress = self._ramp_progress(
                self.common_step_counter,
                cfg.advanced_curriculum_start_step,
                cfg.advanced_lateral_ramp_steps,
            )
            return cfg.min_abs_y + (cfg.advanced_min_abs_y - cfg.min_abs_y) * progress

        progress = self._ramp_progress(
            self.common_step_counter,
            cfg.advanced_curriculum_start_step + cfg.advanced_lateral_rehearsal_steps,
            cfg.advanced_yaw_ramp_steps,
        )
        return cfg.min_abs_yaw + (cfg.advanced_min_abs_yaw - cfg.min_abs_yaw) * progress

    def _current_mode_probabilities(self, stage):
        cfg = self.cfg.commands
        if stage == "advanced_lateral_rehearsal":
            return (
                cfg.advanced_lateral_zero_prob,
                cfg.advanced_lateral_sagittal_prob,
                cfg.advanced_lateral_lateral_prob,
                cfg.advanced_lateral_yaw_prob,
            )
        if stage == "advanced_yaw_rehearsal":
            return (
                cfg.advanced_yaw_zero_prob,
                cfg.advanced_yaw_sagittal_prob,
                cfg.advanced_yaw_lateral_prob,
                cfg.advanced_yaw_yaw_prob,
            )
        if stage == "advanced_balanced":
            return (
                cfg.advanced_balanced_zero_prob,
                cfg.advanced_balanced_sagittal_prob,
                cfg.advanced_balanced_lateral_prob,
                cfg.advanced_balanced_yaw_prob,
            )
        if stage == "advanced_final":
            return (
                cfg.advanced_final_zero_prob,
                cfg.advanced_final_sagittal_prob,
                cfg.advanced_final_lateral_prob,
                cfg.advanced_final_yaw_prob,
            )
        return (
            cfg.zero_prob,
            cfg.sagittal_prob,
            cfg.lateral_prob,
            cfg.yaw_prob,
        )

    def _current_push_max_vel(self):
        cfg = self.cfg.domain_rand
        action_stage = self._current_action_stage()
        if action_stage is not None and action_stage.key == "squat":
            return 0.0
        if action_stage is not None and action_stage.key == "action_switch":
            return min(0.02, cfg.advanced_max_push_vel_xy)
        if action_stage is not None and action_stage.key == "obstacle_crossing":
            return 0.0
        if action_stage is not None and action_stage.key == "ball_kick":
            return 0.0
        if not cfg.curriculum_push_robots:
            return cfg.max_push_vel_xy if cfg.push_robots else 0.0
        progress = self._ramp_progress(
            self.common_step_counter,
            cfg.advanced_push_start_steps,
            cfg.advanced_push_ramp_steps,
        )
        return cfg.advanced_max_push_vel_xy * progress

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_command_range_schedule()
        action_stage = self._current_action_stage()
        if action_stage is not None and action_stage.key == "emergency_stop_stand":
            # A probe is always followed by a zero-command segment.  This makes
            # every sampled probe an actual emergency-stop transition instead
            # of relying on two independent resamples to happen in sequence.
            force_stop = self.emergency_probe_active[env_ids]
            self.commands[env_ids, :3] = 0.0
            moving_mask = (~force_stop) & (torch.rand(len(env_ids), device=self.device) < (
                self.cfg.skill_curriculum.emergency_motion_probe_prob
            ))
            new_reference = ~force_stop
            if torch.any(new_reference):
                reference_ids = env_ids[new_reference]
                _, _, yaw = euler_from_quat(self.base_quat[reference_ids])
                self.command_heading[reference_ids] = yaw
                self.command_start_xy[reference_ids] = self.root_states[reference_ids, :2]
            self.line_reference_active[env_ids] = force_stop | moving_mask
            self.skill_mode[env_ids] = self.SKILL_STAND
            if torch.any(moving_mask):
                self.skill_mode[env_ids[moving_mask]] = self.SKILL_LOCOMOTION
                self._sample_sagittal_command(
                    env_ids[moving_mask], allow_turn=False
                )
            self._schedule_skill_height(
                env_ids, self.cfg.skill_curriculum.nominal_body_height_m
            )
            self.emergency_probe_active[env_ids] = moving_mask
            return

        if action_stage is not None and action_stage.key == "squat":
            self.commands[env_ids, :3] = 0.0
            new_reference = ~self.line_reference_active[env_ids]
            if torch.any(new_reference):
                reference_ids = env_ids[new_reference]
                _, _, yaw = euler_from_quat(self.base_quat[reference_ids])
                self.command_heading[reference_ids] = yaw
                self.command_start_xy[reference_ids] = self.root_states[reference_ids, :2]
            self.line_reference_active[env_ids] = True
            self.squat_target_low[env_ids] = ~self.squat_target_low[env_ids]
            low = self.squat_target_low[env_ids]
            self.skill_mode[env_ids] = self.SKILL_STAND
            self.skill_mode[env_ids[low]] = self.SKILL_SQUAT
            goals = torch.full(
                (len(env_ids),),
                self.cfg.skill_curriculum.nominal_body_height_m,
                device=self.device,
            )
            goals[low] = self._current_squat_goal()
            self._schedule_skill_height(env_ids, goals)
            return

        if action_stage is not None and action_stage.key == "action_switch":
            previous_mode = self.skill_mode[env_ids].clone()
            sample = torch.rand(len(env_ids), device=self.device)
            next_mode = torch.full_like(previous_mode, self.SKILL_SQUAT)
            next_mode[sample < 0.75] = self.SKILL_STAND
            next_mode[sample < 0.50] = self.SKILL_LOCOMOTION
            repeated = next_mode == previous_mode
            repeated_locomotion = repeated & (previous_mode == self.SKILL_LOCOMOTION)
            stationary_fallback = torch.where(
                torch.rand(len(env_ids), device=self.device) < 0.5,
                torch.full_like(previous_mode, self.SKILL_STAND),
                torch.full_like(previous_mode, self.SKILL_SQUAT),
            )
            next_mode = torch.where(
                repeated_locomotion, stationary_fallback, next_mode
            )
            next_mode = torch.where(
                repeated & ~repeated_locomotion,
                torch.full_like(previous_mode, self.SKILL_LOCOMOTION),
                next_mode,
            )
            moving_mask = next_mode == self.SKILL_LOCOMOTION
            # Keep one episode-level line reference across walk, stop, stand,
            # and squat transitions. Resetting it at every stationary switch
            # would permit a slowly rotating "straight" trajectory.
            new_reference = ~self.line_reference_active[env_ids]
            if torch.any(new_reference):
                reference_ids = env_ids[new_reference]
                _, _, yaw = euler_from_quat(self.base_quat[reference_ids])
                self.command_heading[reference_ids] = yaw
                self.command_start_xy[reference_ids] = self.root_states[reference_ids, :2]
            self.commands[env_ids, :3] = 0.0
            self.skill_mode[env_ids] = next_mode
            self.line_reference_active[env_ids] = True
            if torch.any(moving_mask):
                moving_ids = env_ids[moving_mask]
                backward = torch.rand(len(moving_ids), device=self.device) < (
                    self.cfg.skill_curriculum.action_switch_backward_prob
                )
                speed_cfg = self.cfg.skill_curriculum
                forward_speed = torch_rand_float(
                    speed_cfg.action_switch_forward_speed_range[0],
                    speed_cfg.action_switch_forward_speed_range[1],
                    (len(moving_ids), 1),
                    device=self.device,
                ).squeeze(1)
                backward_speed = torch_rand_float(
                    speed_cfg.action_switch_backward_speed_range[0],
                    speed_cfg.action_switch_backward_speed_range[1],
                    (len(moving_ids), 1),
                    device=self.device,
                ).squeeze(1)
                self.commands[moving_ids, 0] = torch.where(
                    backward,
                    backward_speed,
                    forward_speed,
                )
            goals = torch.full(
                (len(env_ids),),
                self.cfg.skill_curriculum.nominal_body_height_m,
                device=self.device,
            )
            goals[next_mode == self.SKILL_SQUAT] = (
                self.cfg.skill_curriculum.squat_body_height_m
            )
            self._schedule_skill_height(env_ids, goals)
            self.emergency_probe_active[env_ids] = False
            return

        if action_stage is not None and action_stage.key == "fall_recovery":
            self.commands[env_ids, :3] = 0.0
            self.skill_mode[env_ids] = self.SKILL_RECOVERY
            self.line_reference_active[env_ids] = False
            self._schedule_skill_height(
                env_ids, self.cfg.skill_curriculum.nominal_body_height_m
            )
            return

        if action_stage is not None and action_stage.key == "diagonal_motion":
            cfg = self.cfg.skill_curriculum
            count = len(env_ids)
            x_magnitude = torch_rand_float(
                cfg.diagonal_forward_speed_range[0],
                cfg.diagonal_forward_speed_range[1],
                (count, 1),
                device=self.device,
            ).squeeze(1)
            y_magnitude = torch_rand_float(
                cfg.diagonal_lateral_speed_range[0],
                cfg.diagonal_lateral_speed_range[1],
                (count, 1),
                device=self.device,
            ).squeeze(1)
            x_sign = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                -torch.ones(count, device=self.device),
                torch.ones(count, device=self.device),
            )
            y_sign = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                -torch.ones(count, device=self.device),
                torch.ones(count, device=self.device),
            )
            self.commands[env_ids, 0] = x_magnitude * x_sign
            self.commands[env_ids, 1] = y_magnitude * y_sign
            self.commands[env_ids, 2] = 0.0
            self.skill_mode[env_ids] = self.SKILL_LOCOMOTION
            self.line_reference_active[env_ids] = True
            _, _, yaw = euler_from_quat(self.base_quat[env_ids])
            self.command_heading[env_ids] = yaw
            self.command_start_xy[env_ids] = self.root_states[env_ids, :2]
            self._schedule_skill_height(
                env_ids, self.cfg.skill_curriculum.nominal_body_height_m
            )
            return

        if action_stage is not None and action_stage.key == "obstacle_crossing":
            cfg = self.cfg.skill_curriculum
            count = len(env_ids)
            self.commands[env_ids, :3] = 0.0
            self.commands[env_ids, 0] = torch_rand_float(
                cfg.obstacle_speed_range[0],
                cfg.obstacle_speed_range[1],
                (count, 1),
                device=self.device,
            ).squeeze(1)
            self.skill_mode[env_ids] = self.SKILL_OBSTACLE
            self.obstacle_task_active[env_ids] = True
            self.obstacle_elapsed_steps[env_ids] = 0
            self.line_reference_active[env_ids] = True
            _, _, yaw = euler_from_quat(self.base_quat[env_ids])
            self.command_heading[env_ids] = yaw
            self.command_start_xy[env_ids] = self.root_states[env_ids, :2]
            self._schedule_skill_height(
                env_ids, self.cfg.skill_curriculum.nominal_body_height_m
            )
            return

        if action_stage is not None and action_stage.key == "ball_kick":
            cfg = self.cfg.skill_curriculum
            self.commands[env_ids, :3] = 0.0
            self.commands[env_ids, 0] = cfg.ball_approach_speed_mps
            self.skill_mode[env_ids] = self.SKILL_KICK
            self.ball_task_active[env_ids] = True
            self.ball_elapsed_steps[env_ids] = 0
            self.line_reference_active[env_ids] = True
            _, _, yaw = euler_from_quat(self.base_quat[env_ids])
            self.command_heading[env_ids] = yaw
            self.command_start_xy[env_ids] = self.root_states[env_ids, :2]
            self._schedule_skill_height(env_ids, cfg.nominal_body_height_m)
            return

        self.commands[env_ids, :3] = 0.0
        _, _, yaw = euler_from_quat(self.base_quat[env_ids])
        self.command_heading[env_ids] = yaw
        self.command_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.line_reference_active[env_ids] = False
        self.skill_mode[env_ids] = self.SKILL_LOCOMOTION

        stage = self._current_command_stage()
        zero_prob = self.cfg.commands.stage_zero_prob

        if stage == "sagittal":
            active_mask = torch.rand(len(env_ids), device=self.device) >= zero_prob
            if torch.any(active_mask):
                sagittal_ids = env_ids[active_mask]
                self._sample_sagittal_command(sagittal_ids, allow_turn=False)
            return

        if stage == "lateral":
            active_mask = torch.rand(len(env_ids), device=self.device) >= zero_prob
            if torch.any(active_mask):
                lateral_ids = env_ids[active_mask]
                self.commands[lateral_ids, 1] = self._sample_axis_command(
                    "lin_vel_y", self._current_min_abs("lin_vel_y"), len(lateral_ids)
                )
            return

        if stage == "yaw":
            active_mask = torch.rand(len(env_ids), device=self.device) >= zero_prob
            if torch.any(active_mask):
                turn_ids = env_ids[active_mask]
                self.commands[turn_ids, 2] = self._sample_axis_command(
                    "ang_vel_yaw", self._current_min_abs("ang_vel_yaw"), len(turn_ids)
                )
            return

        zero_prob, sagittal_prob, lateral_prob, yaw_prob = (
            self._current_mode_probabilities(stage)
        )
        total_prob = zero_prob + sagittal_prob + lateral_prob + yaw_prob
        zero_cut = zero_prob / total_prob
        sagittal_cut = (zero_prob + sagittal_prob) / total_prob
        lateral_cut = (zero_prob + sagittal_prob + lateral_prob) / total_prob

        mode_draw = torch.rand(len(env_ids), device=self.device)
        sagittal_mask = (mode_draw >= zero_cut) & (mode_draw < sagittal_cut)
        lateral_mask = (mode_draw >= sagittal_cut) & (mode_draw < lateral_cut)
        turn_mask = mode_draw >= lateral_cut

        if torch.any(sagittal_mask):
            sagittal_ids = env_ids[sagittal_mask]
            self._sample_sagittal_command(
                sagittal_ids,
                allow_turn=stage.startswith("advanced"),
            )

        if torch.any(lateral_mask):
            lateral_ids = env_ids[lateral_mask]
            self.commands[lateral_ids, 1] = self._sample_axis_command(
                "lin_vel_y", self._current_min_abs("lin_vel_y"), len(lateral_ids)
            )

        if torch.any(turn_mask):
            turn_ids = env_ids[turn_mask]
            self.commands[turn_ids, 2] = self._sample_axis_command(
                "ang_vel_yaw", self._current_min_abs("ang_vel_yaw"), len(turn_ids)
            )

    def _sample_sagittal_command(self, env_ids, allow_turn):
        count = len(env_ids)
        self.commands[env_ids, 0] = self._sample_axis_command(
            "lin_vel_x", self._current_min_abs("lin_vel_x"), count
        )
        if not allow_turn:
            return
        turn_prob = (
            self.cfg.commands.advanced_sagittal_turn_prob
            if self.common_step_counter >= self.cfg.commands.advanced_curriculum_start_step
            else self.cfg.commands.sagittal_turn_prob
        )
        turn_mask = torch.rand(count, device=self.device) < turn_prob
        if torch.any(turn_mask):
            turn_ids = env_ids[turn_mask]
            self.commands[turn_ids, 2] = self._sample_axis_command(
                "ang_vel_yaw",
                self._current_min_abs("ang_vel_yaw"),
                len(turn_ids),
                self.cfg.commands.sagittal_turn_yaw_scale,
            )

    def _sample_axis_command(self, range_name, min_abs, count, max_abs_scale=1.0):
        command_min, command_max = self.command_ranges[range_name]
        signs = torch.ones(count, device=self.device)
        signs[:count // 2] = -1.0
        signs = signs[torch.randperm(count, device=self.device)]
        values = torch.zeros(count, device=self.device)

        positive_mask = signs > 0.0
        if torch.any(positive_mask):
            positive_max = max(command_max * max_abs_scale, min_abs)
            values[positive_mask] = torch_rand_float(
                min_abs,
                positive_max,
                (int(torch.sum(positive_mask).item()), 1),
                device=self.device,
            ).squeeze(1)

        negative_mask = ~positive_mask
        if torch.any(negative_mask):
            negative_max = max(abs(command_min) * max_abs_scale, min_abs)
            values[negative_mask] = -torch_rand_float(
                min_abs,
                negative_max,
                (int(torch.sum(negative_mask).item()), 1),
                device=self.device,
            ).squeeze(1)

        return torch.clip(values, command_min, command_max)

    def _reset_dofs(self, env_ids):
        scale_min, scale_max = self.cfg.init_state.reset_joint_scale_range
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            scale_min, scale_max, (len(env_ids), self.num_dof), device=self.device
        )
        if self.cfg.init_state.reset_joint_offset > 0.0:
            self.dof_pos[env_ids] += torch_rand_float(
                -self.cfg.init_state.reset_joint_offset,
                self.cfg.init_state.reset_joint_offset,
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = self._robot_actor_ids(env_ids).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        xy_range = self.cfg.init_state.reset_xy_range
        if xy_range > 0.0:
            self.root_states[env_ids, :2] += torch_rand_float(
                -xy_range, xy_range, (len(env_ids), 2), device=self.device
            )

        roll = torch_rand_float(
            -self.cfg.init_state.reset_roll_range,
            self.cfg.init_state.reset_roll_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        pitch = torch_rand_float(
            self.cfg.init_state.base_pitch - self.cfg.init_state.reset_pitch_range,
            self.cfg.init_state.base_pitch + self.cfg.init_state.reset_pitch_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        yaw = torch_rand_float(
            -self.cfg.init_state.reset_yaw_range,
            self.cfg.init_state.reset_yaw_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        lin_vel_range = self.cfg.init_state.reset_lin_vel_range
        ang_vel_range = self.cfg.init_state.reset_ang_vel_range
        self.root_states[env_ids, 7:10] = torch_rand_float(
            -lin_vel_range, lin_vel_range, (len(env_ids), 3), device=self.device
        )
        self.root_states[env_ids, 10:13] = torch_rand_float(
            -ang_vel_range, ang_vel_range, (len(env_ids), 3), device=self.device
        )

        if self._stage_is("fall_recovery"):
            cfg = self.cfg.skill_curriculum
            count = len(env_ids)
            progress = self._ramp_progress(
                self.common_step_counter,
                cfg.recovery_curriculum_start_step,
                cfg.recovery_curriculum_ramp_steps,
            )
            pitch_low = cfg.recovery_pitch_start_range_rad[0] + progress * (
                cfg.recovery_pitch_range_rad[0]
                - cfg.recovery_pitch_start_range_rad[0]
            )
            pitch_high = cfg.recovery_pitch_start_range_rad[1] + progress * (
                cfg.recovery_pitch_range_rad[1]
                - cfg.recovery_pitch_start_range_rad[1]
            )
            magnitude = torch_rand_float(
                pitch_low,
                pitch_high,
                (count, 1),
                device=self.device,
            ).squeeze(1)
            direction = torch.where(
                torch.rand(count, device=self.device) < cfg.recovery_supine_prob,
                torch.ones(count, device=self.device),
                -torch.ones(count, device=self.device),
            )
            pitch = magnitude * direction
            roll = torch_rand_float(
                -0.08, 0.08, (count, 1), device=self.device
            ).squeeze(1)
            yaw = torch_rand_float(
                -0.20, 0.20, (count, 1), device=self.device
            ).squeeze(1)
            self.root_states[env_ids, 2] = (
                self.env_origins[env_ids, 2] + cfg.recovery_start_height_m
            )
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
            self.root_states[env_ids, 7:13] = 0.0

        if self._stage_is("obstacle_crossing"):
            cfg = self.cfg.skill_curriculum
            self.root_states[env_ids, 2] += cfg.obstacle_spawn_height_offset_m
            start_x = (
                self.env_origins[env_ids, 0]
                + cfg.obstacle_offset_m
                - cfg.obstacle_approach_distance_m
            )
            self.root_states[env_ids, 0] = start_x + torch_rand_float(
                -0.02, 0.02, (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.root_states[env_ids, 1] = self.env_origins[env_ids, 1] + torch_rand_float(
                -0.015, 0.015, (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.root_states[env_ids, 7:13] = 0.0

        actor_ids = self._robot_actor_ids(env_ids)
        if self.ball_root_states is not None:
            cfg = self.cfg.skill_curriculum
            ball_slot = self.scene_actor_slots["ball"]
            self.ball_root_states[env_ids] = 0.0
            if self._stage_is("ball_kick"):
                spawn_distance = (
                    cfg.ball_kick_spawn_distance_m
                    if cfg.ball_phase == "kick"
                    else cfg.ball_spawn_distance_m
                )
                ball_height = self.cfg.scene.ball_radius_m
            else:
                # Keep the physical ball loaded but completely outside the
                # scene until the kick module begins.
                spawn_distance = 0.0
                ball_height = -1.0
            self.ball_root_states[env_ids, 0] = self.env_origins[env_ids, 0] + spawn_distance
            self.ball_root_states[env_ids, 1] = (
                self.env_origins[env_ids, 1] + cfg.ball_spawn_lateral_center_m
            )
            self.ball_root_states[env_ids, 1] += torch_rand_float(
                -cfg.ball_spawn_lateral_range_m,
                cfg.ball_spawn_lateral_range_m,
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.ball_root_states[env_ids, 2] = ball_height
            self.ball_root_states[env_ids, 6] = 1.0
            self.ball_start_x[env_ids] = self.ball_root_states[env_ids, 0]
            actor_ids = torch.cat((actor_ids, actor_ids + ball_slot))
        env_ids_int32 = actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _compute_torques(self, actions):
        if not self.cfg.control.use_action_scale_curriculum:
            targets = (
                actions * self.cfg.control.action_scale
                + self.default_dof_pos
                + self.joint_target_offsets
            )
            max_target_delta = self.max_motor_velocities * self.sim_params.dt
            targets = torch.clip(
                targets,
                self.previous_motor_targets - max_target_delta,
                self.previous_motor_targets + max_target_delta,
            )
            self.previous_motor_targets[:] = targets
            torques = (
                self.p_gains * self.kp_factors * (targets - self.dof_pos)
                - self.d_gains * self.kd_factors * self.dof_vel
            )
            torques *= self.motor_strengths
            return torch.clip(torques, -self.torque_limits, self.torque_limits)

        warmup_steps = self.cfg.control.action_scale_warmup_steps
        ramp_steps = max(self.cfg.control.action_scale_ramp_steps, 1)
        progress = min(
            max((self.common_step_counter - warmup_steps) / ramp_steps, 0.0),
            1.0,
        )
        action_scale = (
            self.cfg.control.warmup_action_scale * (1.0 - progress)
            + self.cfg.control.action_scale * progress
        )
        targets = (
            actions * action_scale
            + self.default_dof_pos
            + self.joint_target_offsets
        )
        max_target_delta = self.max_motor_velocities * self.sim_params.dt
        targets = torch.clip(
            targets,
            self.previous_motor_targets - max_target_delta,
            self.previous_motor_targets + max_target_delta,
        )
        self.previous_motor_targets[:] = targets
        torques = (
            self.p_gains * self.kp_factors * (targets - self.dof_pos)
            - self.d_gains * self.kd_factors * self.dof_vel
        )
        torques *= self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        world_accel = (
            self.root_states[:, 7:10] - self.last_root_vel[:, :3]
        ) / self.dt
        specific_force = quat_rotate_inverse(
            self.base_quat,
            world_accel - self.gravity_vec * 9.81,
        )
        specific_force = torch.clip(specific_force, -30.0, 30.0)
        current_imu = torch.cat(
            (self.base_ang_vel, self.projected_gravity, specific_force), dim=-1
        )
        self.imu_history = torch.roll(self.imu_history, shifts=1, dims=0)
        self.imu_history[0] = current_imu
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed_imu = self.imu_history[self.imu_delay_steps, env_ids]
        gyro = delayed_imu[:, :3] + self.gyro_bias
        gravity = delayed_imu[:, 3:6] + self.gravity_bias
        accelerometer = delayed_imu[:, 6:9] + self.accel_bias
        measured_dof_pos = self.dof_pos + self.encoder_offsets

        self.obs_buf = torch.cat(
            (
                gyro * self.obs_scales.ang_vel,
                gravity,
                accelerometer * self.obs_scales.accel,
                self._command_observation(),
                (measured_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.commanded_actions,
                self.commanded_action_history_1,
                self.commanded_action_history_2,
                self._skill_observation(),
            ),
            dim=-1,
        )
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0)
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def check_termination(self):
        if self._stage_is("fall_recovery"):
            cfg = self.cfg.skill_curriculum
            timeout_steps = max(1, round(cfg.recovery_timeout_s / self.dt))
            success_steps = max(1, round(cfg.recovery_success_hold_s / self.dt))
            recovered = self.recovery_upright_steps >= success_steps
            timed_out = self.recovery_elapsed_steps >= timeout_steps
            self.reset_buf = recovered | timed_out
            # Successful recoveries are completed tasks, not failures, so they
            # must not receive the standard termination penalty.
            self.time_out_buf = recovered.clone()
            return
        if self._stage_is("obstacle_crossing"):
            target_gravity = self.target_projected_gravity.expand_as(
                self.projected_gravity
            )
            alignment = torch.sum(
                self.projected_gravity * target_gravity, dim=1
            )
            fallen = alignment < torch.cos(
                torch.tensor(
                    self.cfg.rewards.termination_body_angle,
                    device=self.device,
                )
            )
            too_low = self.root_states[:, 2] < self.cfg.rewards.termination_height
            timed_out = self.obstacle_elapsed_steps >= max(
                1,
                round(self.cfg.skill_curriculum.obstacle_timeout_s / self.dt),
            )
            completed = self._obstacle_success_mask()
            self.obstacle_success_latched = completed
            start_x = (
                self.obstacle_world_x
                - self.cfg.skill_curriculum.obstacle_approach_distance_m
            )
            self.obstacle_progress_latched = self.root_states[:, 0] - start_x
            regular_timeout = self.episode_length_buf > self.max_episode_length
            active = self.obstacle_task_active
            self.reset_buf = fallen | too_low
            self.reset_buf |= active & (completed | timed_out)
            self.reset_buf |= (~active) & regular_timeout
            self.time_out_buf = active & (completed | timed_out)
            self.time_out_buf |= (~active) & regular_timeout
            return
        if self._stage_is("ball_kick"):
            target = self.target_projected_gravity.expand_as(self.projected_gravity)
            alignment = torch.sum(self.projected_gravity * target, dim=1)
            fallen = alignment < torch.cos(torch.tensor(
                self.cfg.rewards.termination_body_angle, device=self.device
            ))
            too_low = self.root_states[:, 2] < self.cfg.rewards.termination_height
            timed_out = self.ball_elapsed_steps >= max(
                1, round(self.cfg.skill_curriculum.ball_timeout_s / self.dt)
            )
            completed = self._ball_success_mask()
            self.ball_success_latched = completed
            self.ball_progress_latched = self.ball_root_states[:, 0] - self.ball_start_x
            active = self.ball_task_active
            self.reset_buf = fallen | too_low | (active & (completed | timed_out))
            self.time_out_buf = active & (completed | timed_out)
            return
        target_gravity = self.target_projected_gravity.expand_as(self.projected_gravity)
        gravity_alignment = torch.sum(
            self.projected_gravity * target_gravity,
            dim=1,
        )
        fallen = gravity_alignment < torch.cos(
            torch.tensor(
                self.cfg.rewards.termination_body_angle,
                device=self.device,
            )
        )
        too_low = self.root_states[:, 2] < self.cfg.rewards.termination_height
        recovery_demo = getattr(
            self,
            "demo_recovery_active",
            torch.zeros_like(fallen),
        )
        self.reset_buf = (fallen | too_low) & ~recovery_demo
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        idx = 0
        noise_vec[idx:idx + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3
        noise_vec[idx:idx + 3] = noise_scales.gravity * noise_level
        idx += 3
        noise_vec[idx:idx + 3] = noise_scales.accel * noise_level * self.obs_scales.accel
        idx += 3
        noise_vec[idx:idx + 3] = 0.0
        idx += 3
        noise_vec[idx:idx + self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += self.num_dof
        noise_vec[idx:idx + self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += self.num_dof
        for _ in range(3):
            noise_vec[idx:idx + self.num_actions] = 0.0
            idx += self.num_actions

        if self.cfg.terrain.measure_heights:
            noise_vec[idx:] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    def _reward_support_contact(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        return torch.any(contacts, dim=1).float()

    def _moving_command_mask(self):
        x_active = torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x")
        y_active = torch.abs(self.commands[:, 1]) >= self._current_min_abs("lin_vel_y")
        yaw_active = torch.abs(self.commands[:, 2]) >= self._current_min_abs("ang_vel_yaw")
        return (x_active | y_active | yaw_active).float()

    def _stage_is(self, *keys):
        demo_stage = getattr(self, "demo_action_stage_key", None)
        if demo_stage is not None:
            return demo_stage in keys
        stage = self._current_action_stage()
        return stage is not None and stage.key in keys

    def _stationary_skill_mask(self):
        return (self.skill_mode != self.SKILL_LOCOMOTION).float()

    def _squat_skill_mask(self):
        return (self.skill_mode == self.SKILL_SQUAT).float()

    def _line_tracking_mask(self):
        if self._stage_is("emergency_stop_stand", "squat", "action_switch"):
            return self.line_reference_active.float()
        return self._pure_sagittal_command_mask()

    def _line_penalty_ramp(self):
        if not self._stage_is("emergency_stop_stand"):
            return 1.0
        cfg = self.cfg.skill_curriculum
        progress = (
            self.common_step_counter - cfg.start_step
        ) / max(cfg.stage_steps[0], 1)
        return min(max(progress, 0.25), 1.0)

    def _line_direction_weight(self):
        backward = self.commands[:, 0] <= -self._current_min_abs("lin_vel_x")
        return torch.where(
            backward,
            torch.full_like(self.commands[:, 0], 2.0),
            torch.ones_like(self.commands[:, 0]),
        )

    def _pure_yaw_command_mask(self):
        yaw_active = torch.abs(self.commands[:, 2]) >= self._current_min_abs("ang_vel_yaw")
        low_translation = torch.norm(self.commands[:, :2], dim=1) < 0.05
        return (yaw_active & low_translation).float()

    def _pure_lateral_command_mask(self):
        lateral_active = torch.abs(self.commands[:, 1]) >= self._current_min_abs("lin_vel_y")
        low_sagittal = torch.abs(self.commands[:, 0]) < 0.04
        low_yaw = torch.abs(self.commands[:, 2]) < 0.08
        return (lateral_active & low_sagittal & low_yaw).float()

    def _reference_contacts(self):
        return torch.clip(self.current_teacher_reference[:, 32:34], 0.0, 1.0)

    def _reference_leg_pos(self):
        return torch.cat(
            (
                self.current_teacher_reference[:, :5],
                self.current_teacher_reference[:, 11:16],
            ),
            dim=1,
        )

    def _foot_clearance(self):
        foot_z = self.rigid_body_state[:, self.feet_indices, 2]
        return torch.clamp(foot_z - self.home_feet_z.unsqueeze(0), min=0.0, max=0.05)

    def _normalized_foot_lift(self):
        clearance = self._foot_clearance()
        return torch.clamp(
            (clearance - self.cfg.rewards.foot_lift_deadband)
            / self.cfg.rewards.foot_lift_target,
            min=0.0,
            max=1.0,
        )

    def _single_support(self):
        contacts = self.gait_contacts.float()
        return (torch.sum(contacts, dim=1) == 1).float()

    def _stepping_signal(self):
        return torch.clamp(
            self._single_support()
            + torch.sum(self.gait_first_contacts, dim=1)
            + 0.5 * self.gait_contact_switch,
            min=0.0,
            max=1.0,
        )

    def _clock_swing_gates(self):
        phase = (
            self.gait_phase_steps.float()
            / max(self.teacher_reference_period_steps, 1)
            * (2.0 * np.pi)
        )
        phase_sin = torch.sin(phase)
        gate = self.cfg.rewards.clock_phase_gate
        left_swing_gate = (phase_sin > gate).float()
        right_swing_gate = (phase_sin < -gate).float()
        gate_sum = torch.clamp(left_swing_gate + right_swing_gate, min=1.0)
        return left_swing_gate, right_swing_gate, gate_sum

    def _reward_teacher_action(self):
        if not self.use_teacher_reference:
            return torch.zeros(self.num_envs, device=self.device)
        reference_action = self.current_teacher_reference[:, 40:50]
        action_error = torch.mean(
            torch.square(self.commanded_actions - reference_action), dim=1
        )
        return torch.exp(-3.0 * action_error) * self._moving_command_mask()

    def _reward_teacher_action_error(self):
        if not self.use_teacher_reference:
            return torch.zeros(self.num_envs, device=self.device)
        reference_action = self.current_teacher_reference[:, 40:50]
        action_error = torch.mean(
            torch.square(self.commanded_actions - reference_action), dim=1
        )
        return torch.clamp(action_error, max=4.0) * self._moving_command_mask()

    def _reward_teacher_target(self):
        if not self.use_teacher_reference:
            return torch.zeros(self.num_envs, device=self.device)
        reference_target = self.current_teacher_reference[:, 50:60]
        target_error = torch.mean(
            torch.square(self.previous_motor_targets - reference_target), dim=1
        )
        return torch.exp(-8.0 * target_error) * self._moving_command_mask()

    def _reward_teacher_target_delta_error(self):
        if not self.use_teacher_reference:
            return torch.zeros(self.num_envs, device=self.device)
        reference_delta = self.current_teacher_reference[:, 50:60] - self.teacher_home_target
        target_delta = self.previous_motor_targets - self.default_dof_pos
        delta_error = torch.mean(torch.square(target_delta - reference_delta), dim=1)
        return torch.clamp(delta_error, max=1.0) * self._moving_command_mask()

    def _reward_reference_gait_delta(self):
        if not self.use_teacher_reference:
            return torch.zeros(self.num_envs, device=self.device)
        reference_delta = self.current_teacher_reference[:, 50:60] - self.teacher_home_target
        target_delta = self.previous_motor_targets - self.default_dof_pos
        delta_error = torch.mean(torch.square(target_delta - reference_delta), dim=1)
        return torch.exp(-4.0 * delta_error) * self._moving_command_mask()

    def _reward_phase_support_match(self):
        reference_contact = self._reference_contacts()
        contact = self.gait_contacts.float()
        match = 1.0 - torch.mean(torch.abs(contact - reference_contact), dim=1)
        return match * self._moving_command_mask()

    def _reward_phase_swing_lift(self):
        reference_swing = 1.0 - self._reference_contacts()
        lift = self._normalized_foot_lift()
        swing_count = torch.clamp(torch.sum(reference_swing, dim=1), min=1.0)
        return (
            torch.sum(lift * reference_swing, dim=1)
            / swing_count
            * self._moving_command_mask()
        )

    def _reward_swing_contact(self):
        reference_swing = 1.0 - self._reference_contacts()
        contact = self.gait_contacts.float()
        swing_count = torch.clamp(torch.sum(reference_swing, dim=1), min=1.0)
        return (
            torch.sum(contact * reference_swing, dim=1)
            / swing_count
            * self._moving_command_mask()
        )

    def _reward_foot_lift(self):
        lift = self._normalized_foot_lift()
        return torch.max(lift, dim=1).values * self._moving_command_mask()

    def _reward_swing_clearance(self):
        clearance = self._foot_clearance()
        contact = self.gait_contacts.float()
        return torch.sum(clearance * (1.0 - contact), dim=1) * self._moving_command_mask()

    def _reward_clocked_single_support(self):
        contacts = self.gait_contacts.float()
        left_gate, right_gate, gate_sum = self._clock_swing_gates()
        left_support_ok = (1.0 - contacts[:, 0]) * contacts[:, 1]
        right_support_ok = (1.0 - contacts[:, 1]) * contacts[:, 0]
        return (
            (left_gate * left_support_ok + right_gate * right_support_ok)
            / gate_sum
            * self._moving_command_mask()
        )

    def _reward_clocked_swing_lift(self):
        contacts = self.gait_contacts.float()
        lift = self._normalized_foot_lift()
        left_gate, right_gate, gate_sum = self._clock_swing_gates()
        return (
            (
                left_gate * lift[:, 0] * contacts[:, 1]
                + right_gate * lift[:, 1] * contacts[:, 0]
            )
            / gate_sum
            * self._moving_command_mask()
        )

    def _reward_clocked_swing_contact(self):
        contacts = self.gait_contacts.float()
        left_gate, right_gate, gate_sum = self._clock_swing_gates()
        return (
            (left_gate * contacts[:, 0] + right_gate * contacts[:, 1])
            / gate_sum
            * self._moving_command_mask()
        )

    def _reward_moving_contact_switch(self):
        return self.gait_contact_switch * self._moving_command_mask()

    def _reward_moving_without_step(self):
        x_progress, _ = self._axis_progress_and_lag(
            self.commands[:, 0],
            self.base_lin_vel[:, 0],
            self._current_min_abs("lin_vel_x"),
        )
        y_progress, _ = self._axis_progress_and_lag(
            self.commands[:, 1],
            self.base_lin_vel[:, 1],
            self._current_min_abs("lin_vel_y"),
        )
        yaw_progress, _ = self._axis_progress_and_lag(
            self.commands[:, 2],
            self.base_ang_vel[:, 2],
            self._current_min_abs("ang_vel_yaw"),
        )
        progress = torch.max(torch.stack((x_progress, y_progress, yaw_progress), dim=1), dim=1).values
        progress_without_step = torch.square(
            torch.clamp(
                progress - self.cfg.rewards.moving_without_step_progress,
                min=0.0,
            )
        )
        return (
            progress_without_step
            * (1.0 - self._stepping_signal())
            * self._moving_command_mask()
        )

    def _reward_single_support(self):
        return self._single_support() * self._moving_command_mask()

    def _reward_double_support(self):
        contacts = self.gait_contacts.float()
        return (torch.sum(contacts, dim=1) == 2).float() * self._moving_command_mask()

    def _reward_no_contact(self):
        contacts = self.gait_contacts.float()
        return (torch.sum(contacts, dim=1) == 0).float() * self._moving_command_mask()

    def _reward_yaw_alternating_contact(self):
        return self._single_support() * self._pure_yaw_command_mask()

    def _reward_yaw_contact_switch(self):
        return self.gait_contact_switch * self._pure_yaw_command_mask()

    def _reward_yaw_twist_without_step(self):
        yaw_speed = torch.sign(self.commands[:, 2]) * self.base_ang_vel[:, 2]
        stepping_signal = torch.clamp(
            self._single_support()
            + torch.sum(self.gait_first_contacts, dim=1)
            + 0.5 * self.gait_contact_switch,
            min=0.0,
            max=1.0,
        )
        twist = torch.square(torch.clamp(yaw_speed - 0.10, min=0.0))
        return twist * (1.0 - stepping_signal) * self._pure_yaw_command_mask()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_command_range_schedule()
        self._update_skill_targets()
        self._apply_straight_heading_hold()
        self._apply_diagonal_heading_hold()
        self._update_recovery_state()
        self.obstacle_elapsed_steps += self.obstacle_task_active.long()
        self.ball_elapsed_steps += self.ball_task_active.long()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_gait_reference_state()
        push_max_vel = self._current_push_max_vel()
        if push_max_vel > 0.0 and self.common_step_counter % int(self.cfg.domain_rand.push_interval) == 0:
            self._push_robots()

    def _recovery_upright_mask(self):
        cfg = self.cfg.skill_curriculum
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target, dim=1)
        return (
            (alignment > np.cos(np.deg2rad(cfg.recovery_upright_tilt_deg)))
            & (self.root_states[:, 2] > cfg.recovery_upright_height_m)
            & (torch.norm(self.base_ang_vel, dim=1) < 1.0)
        )

    def _update_recovery_state(self):
        if not self._stage_is("fall_recovery"):
            return
        self.recovery_elapsed_steps += 1
        upright = self._recovery_upright_mask()
        self.recovery_upright_steps = torch.where(
            upright,
            self.recovery_upright_steps + 1,
            torch.zeros_like(self.recovery_upright_steps),
        )

    def _push_robots(self):
        max_vel = self._current_push_max_vel()
        if max_vel <= 0.0:
            return
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self._all_root_states)
        )

    def _reward_emergency_stop_stability(self):
        action_stage = self._current_action_stage()
        if action_stage is None or action_stage.key != "emergency_stop_stand":
            return torch.zeros(self.num_envs, device=self.device)
        stationary_command = (
            (torch.abs(self.commands[:, 0]) < 0.02)
            & (torch.abs(self.commands[:, 1]) < 0.02)
            & (torch.abs(self.commands[:, 2]) < 0.1)
        )
        motion_error = torch.sum(torch.square(self.base_lin_vel), dim=1)
        motion_error += 0.25 * torch.sum(
            torch.square(self.base_ang_vel), dim=1
        )
        return motion_error * stationary_command.float()

    def _reward_emergency_stop_success(self):
        action_stage = self._current_action_stage()
        if action_stage is None or action_stage.key != "emergency_stop_stand":
            return torch.zeros(self.num_envs, device=self.device)
        stationary_command = (
            (torch.abs(self.commands[:, 0]) < 0.02)
            & (torch.abs(self.commands[:, 1]) < 0.02)
            & (torch.abs(self.commands[:, 2]) < 0.1)
        )
        linear_stable = torch.norm(self.base_lin_vel, dim=1) < (
            self.cfg.skill_curriculum.emergency_settle_linear_mps
        )
        angular_stable = torch.norm(self.base_ang_vel, dim=1) < (
            self.cfg.skill_curriculum.emergency_settle_angular_rps
        )
        target_gravity = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target_gravity, dim=1)
        tilt_stable = alignment > np.cos(
            np.deg2rad(self.cfg.skill_curriculum.emergency_settle_tilt_deg)
        )
        _, _, yaw = euler_from_quat(self.base_quat)
        heading_error = torch.abs(torch_wrap_to_pi_minuspi(yaw - self.command_heading))
        delta_xy = self.root_states[:, :2] - self.command_start_xy
        lateral = torch.abs(
            -delta_xy[:, 0] * torch.sin(self.command_heading)
            + delta_xy[:, 1] * torch.cos(self.command_heading)
        )
        line_stable = (~self.line_reference_active) | (
            (heading_error < self.cfg.skill_curriculum.straight_heading_tolerance_rad)
            & (lateral < self.cfg.skill_curriculum.straight_lateral_tolerance_m)
        )
        return (
            stationary_command & linear_stable & angular_stable & tilt_stable & line_stable
        ).float()

    def _reward_skill_height_tracking(self):
        if not self._stage_is("squat", "action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        error = torch.square(self.root_states[:, 2] - self.target_base_height)
        return torch.exp(-error / self.cfg.rewards.skill_height_sigma)

    def _reward_skill_height_error(self):
        if not self._stage_is("squat", "action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        normalized = torch.abs(
            self.root_states[:, 2] - self.target_base_height
        ) / 0.02
        return torch.clamp(normalized, max=3.0)

    def _desired_squat_pose(self):
        cfg = self.cfg.skill_curriculum
        span = max(cfg.nominal_body_height_m - cfg.squat_body_height_m, 1.0e-6)
        depth = torch.clamp(
            (cfg.nominal_body_height_m - self.target_base_height) / span,
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        full_depth_delta = torch.tensor(
            [0.0, 0.0, 0.15, 0.20, 0.12, 0.0, 0.0, -0.15, 0.20, 0.12],
            device=self.device,
        )
        return self.default_dof_pos + depth * full_depth_delta.unsqueeze(0)

    def _reward_squat_pose_tracking(self):
        target = self._desired_squat_pose()
        error = torch.mean(torch.square(self.dof_pos - target), dim=1)
        return torch.exp(-error / 0.005) * self._squat_skill_mask()

    def _reward_squat_pose_error(self):
        target = self._desired_squat_pose()
        error = torch.mean(torch.square(self.dof_pos - target), dim=1) / 0.01
        return torch.clamp(error, max=3.0) * self._squat_skill_mask()

    def _reward_skill_transition_success(self):
        if not self._stage_is("squat", "action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        height_ok = torch.abs(
            self.root_states[:, 2] - self.target_base_height
        ) < self.cfg.skill_curriculum.height_tolerance_m
        target_gravity = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target_gravity, dim=1)
        tilt_ok = alignment > np.cos(
            np.deg2rad(self.cfg.skill_curriculum.emergency_settle_tilt_deg)
        )
        contacts = torch.sum(self.gait_contacts.float(), dim=1) == 2
        stationary_ok = torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.035
        return (
            height_ok
            & tilt_ok
            & contacts
            & (stationary_ok | (self.skill_mode == self.SKILL_LOCOMOTION))
        ).float()

    def _reward_skill_stability(self):
        if not self._stage_is("squat", "action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        motion = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        motion += 0.25 * torch.sum(torch.square(self.base_ang_vel), dim=1)
        return motion * self._stationary_skill_mask()

    def _reward_skill_double_support(self):
        if not self._stage_is("squat", "action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        contacts = torch.sum(self.gait_contacts.float(), dim=1) == 2
        return contacts.float() * self._stationary_skill_mask()

    def _reward_recovery_alignment(self):
        if not self._stage_is("fall_recovery"):
            return torch.zeros(self.num_envs, device=self.device)
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target, dim=1)
        return torch.clamp((alignment + 1.0) * 0.5, min=0.0, max=1.0)

    def _reward_recovery_height(self):
        if not self._stage_is("fall_recovery"):
            return torch.zeros(self.num_envs, device=self.device)
        target = self.cfg.skill_curriculum.nominal_body_height_m
        error = torch.square((self.root_states[:, 2] - target) / 0.04)
        return torch.exp(-error)

    def _reward_recovery_success(self):
        if not self._stage_is("fall_recovery"):
            return torch.zeros(self.num_envs, device=self.device)
        return self._recovery_upright_mask().float()

    def _reward_recovery_stability(self):
        if not self._stage_is("fall_recovery"):
            return torch.zeros(self.num_envs, device=self.device)
        motion = torch.sum(torch.square(self.base_lin_vel), dim=1)
        motion += 0.20 * torch.sum(torch.square(self.base_ang_vel), dim=1)
        return motion * self._recovery_upright_mask().float()

    def _diagonal_command_mask(self):
        return (
            (torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x"))
            & (torch.abs(self.commands[:, 1]) >= self._current_min_abs("lin_vel_y"))
        ).float()

    def _reward_diagonal_velocity_tracking(self):
        if not self._stage_is("diagonal_motion"):
            return torch.zeros(self.num_envs, device=self.device)
        error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-error / 0.0025) * self._diagonal_command_mask()

    def _reward_diagonal_progress(self):
        if not self._stage_is("diagonal_motion"):
            return torch.zeros(self.num_envs, device=self.device)
        command = self.commands[:, :2]
        command_norm = torch.clamp(torch.norm(command, dim=1), min=1.0e-4)
        directed_speed = torch.sum(command * self.base_lin_vel[:, :2], dim=1)
        directed_speed /= command_norm
        ratio = directed_speed / command_norm
        return torch.clamp(ratio, min=0.0, max=1.0) * self._diagonal_command_mask()

    def _reward_diagonal_heading_error(self):
        if not self._stage_is("diagonal_motion"):
            return torch.zeros(self.num_envs, device=self.device)
        _, _, yaw = euler_from_quat(self.base_quat)
        error = torch_wrap_to_pi_minuspi(yaw - self.command_heading)
        return (
            torch.clamp(torch.square(error / 0.20), max=4.0)
            * self._diagonal_command_mask()
        )

    def _reward_diagonal_path_error(self):
        if not self._stage_is("diagonal_motion"):
            return torch.zeros(self.num_envs, device=self.device)
        command = self.commands[:, :2]
        command_norm = torch.clamp(torch.norm(command, dim=1), min=1.0e-4)
        cos_heading = torch.cos(self.command_heading)
        sin_heading = torch.sin(self.command_heading)
        world_direction = torch.stack(
            (
                cos_heading * command[:, 0] - sin_heading * command[:, 1],
                sin_heading * command[:, 0] + cos_heading * command[:, 1],
            ),
            dim=1,
        ) / command_norm.unsqueeze(1)
        perpendicular = torch.stack(
            (-world_direction[:, 1], world_direction[:, 0]), dim=1
        )
        delta = self.root_states[:, :2] - self.command_start_xy
        cross_track = torch.sum(delta * perpendicular, dim=1)
        return (
            torch.clamp(torch.square(cross_track / 0.10), max=9.0)
            * self._diagonal_command_mask()
        )

    def _obstacle_success_mask(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        cfg = self.cfg.skill_curriculum
        passed = self.root_states[:, 0] >= (
            self.obstacle_world_x + cfg.obstacle_success_margin_m
        )
        lateral = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        _, _, yaw = euler_from_quat(self.base_quat)
        heading = torch.abs(torch_wrap_to_pi_minuspi(yaw - self.command_heading))
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target, dim=1)
        upright = alignment > np.cos(
            np.deg2rad(cfg.obstacle_heading_tolerance_deg)
        )
        return (
            self.obstacle_task_active
            & passed
            & upright
            & (lateral < cfg.obstacle_lateral_tolerance_m)
            & (heading < np.deg2rad(cfg.obstacle_heading_tolerance_deg))
        )

    def _obstacle_near_mask(self):
        distance = torch.abs(self.obstacle_world_x - self.root_states[:, 0])
        return (
            self.obstacle_task_active
            & (distance < self.cfg.skill_curriculum.obstacle_near_distance_m)
        ).float()

    def _reward_obstacle_progress(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(self.num_envs, device=self.device)
        commanded = torch.clamp(self.commands[:, 0], min=0.04)
        progress = torch.clamp(self.base_lin_vel[:, 0] / commanded, 0.0, 1.5)
        return progress * self.obstacle_task_active.float()

    def _reward_obstacle_clearance(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(self.num_envs, device=self.device)
        foot_height = self.rigid_body_state[:, self.feet_indices, 2]
        target = (
            self.env_origins[:, 2]
            + self.obstacle_heights
            + self.cfg.skill_curriculum.obstacle_clearance_margin_m
        )
        clearance = torch.clamp(
            (torch.max(foot_height, dim=1).values - target) / 0.025,
            min=0.0,
            max=1.0,
        )
        return clearance * self._obstacle_near_mask()

    def _reward_obstacle_success(self):
        return self._obstacle_success_mask().float()

    def _reward_obstacle_heading_error(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(self.num_envs, device=self.device)
        _, _, yaw = euler_from_quat(self.base_quat)
        error = torch_wrap_to_pi_minuspi(yaw - self.command_heading)
        return (
            torch.clamp(torch.square(error / 0.20), max=4.0)
            * self.obstacle_task_active.float()
        )

    def _reward_obstacle_lateral_error(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(self.num_envs, device=self.device)
        lateral = self.root_states[:, 1] - self.env_origins[:, 1]
        return (
            torch.clamp(torch.square(lateral / 0.10), max=4.0)
            * self.obstacle_task_active.float()
        )

    def _reward_obstacle_body_collision(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(self.num_envs, device=self.device)
        body_force = torch.norm(
            self.contact_forces[:, self.obstacle_body_indices], dim=-1
        )
        collision = torch.any(body_force > 5.0, dim=1).float()
        return collision * self._obstacle_near_mask()

    def _reward_obstacle_stability(self):
        if not self._stage_is("obstacle_crossing"):
            return torch.zeros(self.num_envs, device=self.device)
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target, dim=1)
        return (
            torch.clamp((alignment + 1.0) * 0.5, 0.0, 1.0)
            * self.obstacle_task_active.float()
        )

    def _ball_relative_body(self):
        relative_world = torch.zeros(
            self.num_envs, 3, dtype=self.root_states.dtype, device=self.device
        )
        if self.ball_root_states is not None:
            relative_world[:, :2] = self.ball_root_states[:, :2] - self.root_states[:, :2]
        return quat_rotate_inverse(self.base_quat, relative_world)

    def _ball_success_mask(self):
        if not self._stage_is("ball_kick") or self.ball_root_states is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cfg = self.cfg.skill_curriculum
        progress = self.ball_root_states[:, 0] - self.ball_start_x
        target_lateral = (
            self.env_origins[:, 1] + cfg.ball_spawn_lateral_center_m
        )
        lateral = torch.abs(self.ball_root_states[:, 1] - target_lateral)
        relative = self._ball_relative_body()
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        upright = torch.sum(self.projected_gravity * target, dim=1) > 0.75
        if cfg.ball_phase == "approach":
            task_reached = (
                (relative[:, 0] <= cfg.ball_contact_distance_m)
                & (relative[:, 0] >= -0.03)
                & (torch.abs(relative[:, 1]) <= 0.10)
            )
        else:
            task_reached = progress >= cfg.ball_goal_distance_m
            lateral_limit = min(
                cfg.ball_target_lateral_tolerance_m,
                cfg.ball_goal_half_width_m - self.cfg.scene.ball_radius_m,
            )
            return self.ball_task_active & task_reached & (
                lateral <= lateral_limit
            ) & upright
        return self.ball_task_active & task_reached & (
            lateral <= cfg.ball_target_lateral_tolerance_m
        ) & upright

    def _reward_ball_approach(self):
        if self.ball_root_states is None:
            return torch.zeros(self.num_envs, device=self.device)
        if self.cfg.skill_curriculum.ball_phase == "approach":
            relative = self._ball_relative_body()
            distance_error = torch.abs(relative[:, 0] - 0.09)
            lateral_error = torch.abs(relative[:, 1])
            score = torch.exp(-8.0 * distance_error) * torch.exp(-12.0 * lateral_error)
        else:
            foot_distance = torch.min(torch.norm(
                self.rigid_body_state[:, self.feet_indices, :2]
                - self.ball_root_states[:, None, :2],
                dim=-1,
            ), dim=1).values
            score = torch.exp(-12.0 * foot_distance)
        return score * self.ball_task_active.float()

    def _reward_ball_forward_velocity(self):
        if self.ball_root_states is None:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.clamp(self.ball_root_states[:, 7] / 0.70, 0.0, 1.5) * self.ball_task_active.float()

    def _reward_ball_progress(self):
        if self.ball_root_states is None:
            return torch.zeros(self.num_envs, device=self.device)
        cfg = self.cfg.skill_curriculum
        target = (
            cfg.ball_stage9_success_distance_m
            if cfg.ball_phase == "approach"
            else cfg.ball_goal_distance_m
        )
        progress = torch.clamp((self.ball_root_states[:, 0] - self.ball_start_x) / target, 0.0, 1.5)
        return progress * self.ball_task_active.float()

    def _reward_ball_contact(self):
        if self.ball_root_states is None:
            return torch.zeros(self.num_envs, device=self.device)
        feet_xy = self.rigid_body_state[:, self.feet_indices, :2]
        distance = torch.norm(feet_xy - self.ball_root_states[:, None, :2], dim=-1)
        nearest, nearest_index = torch.min(distance, dim=1)
        foot_forward_velocity = self.rigid_body_state[
            torch.arange(self.num_envs, device=self.device),
            self.feet_indices[nearest_index],
            7,
        ]
        proximity = torch.exp(-24.0 * nearest)
        swing = 0.5 + torch.clamp(foot_forward_velocity / 0.40, 0.0, 1.5)
        return proximity * swing * self.ball_task_active.float()

    def _reward_ball_goal_alignment(self):
        if self.ball_root_states is None:
            return torch.zeros(self.num_envs, device=self.device)
        cfg = self.cfg.skill_curriculum
        if cfg.ball_phase == "approach":
            return torch.zeros(self.num_envs, device=self.device)
        progress = torch.clamp(
            (self.ball_root_states[:, 0] - self.ball_start_x)
            / cfg.ball_goal_distance_m,
            0.0,
            1.0,
        )
        target_y = self.env_origins[:, 1] + cfg.ball_spawn_lateral_center_m
        lateral_error = self.ball_root_states[:, 1] - target_y
        alignment = torch.exp(-40.0 * torch.square(lateral_error))
        return progress * alignment * self.ball_task_active.float()

    def _reward_ball_success(self):
        return self._ball_success_mask().float()

    def _reward_ball_stability(self):
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.sum(self.projected_gravity * target, dim=1)
        return torch.clamp((alignment + 1.0) * 0.5, 0.0, 1.0) * self.ball_task_active.float()

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_lin_sigma)

    def _reward_tracking_lin_vel_x(self):
        x_active = torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x")
        x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-x_error / self.cfg.rewards.tracking_lin_sigma) * x_active.float()

    def _reward_tracking_lin_vel_y(self):
        y_active = torch.abs(self.commands[:, 1]) >= self._current_min_abs("lin_vel_y")
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-y_error / self.cfg.rewards.tracking_lin_sigma) * y_active.float()

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_ang_sigma)

    def _reward_tracking_ang_vel_yaw(self):
        yaw_active = torch.abs(self.commands[:, 2]) >= self._current_min_abs("ang_vel_yaw")
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_ang_sigma) * yaw_active.float()

    def _axis_progress_and_lag(self, command, velocity, min_command):
        active = torch.abs(command) >= min_command
        signed_velocity = torch.sign(command) * velocity
        normalized_velocity = signed_velocity / torch.clamp(
            torch.abs(command),
            min=min_command,
        )
        progress = torch.clamp(normalized_velocity, min=0.0, max=1.0)
        lag = torch.square(torch.clamp(1.0 - normalized_velocity, min=0.0, max=2.0))
        return progress * active.float(), lag * active.float()

    def _reward_sagittal_progress(self):
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 0],
            self.base_lin_vel[:, 0],
            self._current_min_abs("lin_vel_x"),
        )
        return progress

    def _reward_sagittal_step_progress(self):
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 0],
            self.base_lin_vel[:, 0],
            self._current_min_abs("lin_vel_x"),
        )
        return progress * self._stepping_signal()

    def _reward_lateral_progress(self):
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 1],
            self.base_lin_vel[:, 1],
            self._current_min_abs("lin_vel_y"),
        )
        return progress

    def _reward_lateral_step_progress(self):
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 1],
            self.base_lin_vel[:, 1],
            self._current_min_abs("lin_vel_y"),
        )
        return progress * self._stepping_signal()

    def _reward_yaw_progress(self):
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 2],
            self.base_ang_vel[:, 2],
            self._current_min_abs("ang_vel_yaw"),
        )
        return progress

    def _reward_yaw_step_progress(self):
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 2],
            self.base_ang_vel[:, 2],
            self._current_min_abs("ang_vel_yaw"),
        )
        return progress * self._stepping_signal()

    def _reward_sagittal_lag(self):
        _, lag = self._axis_progress_and_lag(
            self.commands[:, 0],
            self.base_lin_vel[:, 0],
            self._current_min_abs("lin_vel_x"),
        )
        return lag

    def _reward_lateral_lag(self):
        _, lag = self._axis_progress_and_lag(
            self.commands[:, 1],
            self.base_lin_vel[:, 1],
            self._current_min_abs("lin_vel_y"),
        )
        return lag

    def _reward_yaw_lag(self):
        _, lag = self._axis_progress_and_lag(
            self.commands[:, 2],
            self.base_ang_vel[:, 2],
            self._current_min_abs("ang_vel_yaw"),
        )
        return lag

    def _reward_command_stall(self):
        x_cmd = self.commands[:, 0]
        y_cmd = self.commands[:, 1]
        yaw_cmd = self.commands[:, 2]
        x_active = torch.abs(x_cmd) >= self._current_min_abs("lin_vel_x")
        y_active = torch.abs(y_cmd) >= self._current_min_abs("lin_vel_y")
        yaw_active = torch.abs(yaw_cmd) >= self._current_min_abs("ang_vel_yaw")

        signed_speed = (
            torch.sign(x_cmd) * self.base_lin_vel[:, 0] * x_active.float()
            + torch.sign(y_cmd) * self.base_lin_vel[:, 1] * y_active.float()
            + torch.sign(yaw_cmd) * self.base_ang_vel[:, 2] * yaw_active.float()
        )
        command_magnitude = (
            torch.abs(x_cmd) * x_active.float()
            + torch.abs(y_cmd) * y_active.float()
            + torch.abs(yaw_cmd) * yaw_active.float()
        )
        normalized_speed = signed_speed / torch.clamp(command_magnitude, min=1.0e-4)
        minimum_ratio = self.cfg.rewards.minimum_command_ratio
        stall = torch.clamp(
            (minimum_ratio - normalized_speed) / minimum_ratio,
            min=0.0,
            max=2.0,
        )
        return stall * (x_active | y_active | yaw_active).float()

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity - self.target_projected_gravity), dim=1)

    def _reward_sagittal_axis_isolation(self):
        sagittal_cmd = torch.abs(self.commands[:, 0]) > self._current_min_abs("lin_vel_x")
        lateral_scale = max(abs(value) for value in self.command_ranges["lin_vel_y"])
        yaw_scale = max(abs(value) for value in self.command_ranges["ang_vel_yaw"])
        lateral_drift = torch.square(self.base_lin_vel[:, 1] / lateral_scale)
        yaw_drift = torch.square(self.base_ang_vel[:, 2] / yaw_scale)
        drift = torch.clamp(lateral_drift + yaw_drift, max=4.0)
        return drift * sagittal_cmd.float()

    def _pure_sagittal_command_mask(self):
        """Select forward/backward commands without an intentional turn."""
        return (
            (torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x"))
            & (torch.abs(self.commands[:, 1]) < self._current_min_abs("lin_vel_y"))
            & (torch.abs(self.commands[:, 2]) < self._current_min_abs("ang_vel_yaw"))
        ).float()

    def _reward_sagittal_heading_error(self):
        """Penalize accumulated yaw drift during a straight command segment."""
        _, _, yaw = euler_from_quat(self.base_quat)
        heading_error = torch_wrap_to_pi_minuspi(yaw - self.command_heading)
        normalized = torch.square(heading_error / 0.10)
        return (
            torch.clamp(normalized, max=4.0)
            * self._line_tracking_mask()
            * self._line_penalty_ramp()
            * self._line_direction_weight()
        )

    def _reward_sagittal_velocity_tracking(self):
        """Prevent the straightness objectives from being solved by standing still."""
        active = (
            self.line_reference_active.float()
            * (torch.abs(self.commands[:, 0]) >= self._current_min_abs("lin_vel_x")).float()
            if self._stage_is("emergency_stop_stand", "action_switch")
            else self._pure_sagittal_command_mask()
        )
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / 9.0e-4) * active

    def _reward_action_switch_velocity_progress(self):
        if not self._stage_is("action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        progress, _ = self._axis_progress_and_lag(
            self.commands[:, 0],
            self.base_lin_vel[:, 0],
            self._current_min_abs("lin_vel_x"),
        )
        return progress * self.line_reference_active.float()

    def _reward_action_switch_velocity_lag(self):
        if not self._stage_is("action_switch"):
            return torch.zeros(self.num_envs, device=self.device)
        _, lag = self._axis_progress_and_lag(
            self.commands[:, 0],
            self.base_lin_vel[:, 0],
            self._current_min_abs("lin_vel_x"),
        )
        return lag * self.line_reference_active.float()

    def _reward_sagittal_lateral_displacement(self):
        """Penalize world displacement perpendicular to the commanded heading."""
        delta_xy = self.root_states[:, :2] - self.command_start_xy
        lateral = (
            -delta_xy[:, 0] * torch.sin(self.command_heading)
            + delta_xy[:, 1] * torch.cos(self.command_heading)
        )
        # Four centimetres is the acceptance tolerance for a complete move-stop
        # segment. Keep gradient out to 16 cm so severe backward drift is not a
        # flat local optimum, while still bounding failed episodes.
        normalized = torch.square(lateral / 0.04)
        return (
            torch.clamp(normalized, max=16.0)
            * self._line_tracking_mask()
            * self._line_penalty_ramp()
            * self._line_direction_weight()
        )

    def _reward_straight_yaw_rate(self):
        normalized = torch.square(self.base_ang_vel[:, 2] / 0.25)
        return (
            torch.clamp(normalized, max=4.0)
            * self._line_tracking_mask()
            * self._line_penalty_ramp()
            * self._line_direction_weight()
        )

    def _reward_lateral_axis_isolation(self):
        lateral_cmd = torch.abs(self.commands[:, 1]) > self._current_min_abs("lin_vel_y")
        sagittal_scale = max(abs(value) for value in self.command_ranges["lin_vel_x"])
        yaw_scale = max(abs(value) for value in self.command_ranges["ang_vel_yaw"])
        drift = (
            torch.square(self.base_lin_vel[:, 0] / sagittal_scale)
            + torch.square(self.base_ang_vel[:, 2] / yaw_scale)
        )
        drift = torch.clamp(drift, max=4.0)
        return drift * lateral_cmd.float()

    def _reward_lateral_yaw_rate(self):
        yaw_scale = max(abs(value) for value in self.command_ranges["ang_vel_yaw"])
        yaw_rate = torch.square(self.base_ang_vel[:, 2] / yaw_scale)
        return torch.clamp(yaw_rate, max=4.0) * self._pure_lateral_command_mask()

    def _reward_lateral_heading_error(self):
        _, _, yaw = euler_from_quat(self.base_quat)
        heading_error = torch_wrap_to_pi_minuspi(yaw - self.command_heading)
        return torch.clamp(torch.square(heading_error / 0.35), max=4.0) * self._pure_lateral_command_mask()

    def _reward_yaw_axis_isolation(self):
        yaw_cmd = torch.abs(self.commands[:, 2]) > self._current_min_abs("ang_vel_yaw")
        sagittal_scale = max(abs(value) for value in self.command_ranges["lin_vel_x"])
        lateral_scale = max(abs(value) for value in self.command_ranges["lin_vel_y"])
        drift = (
            torch.square(self.base_lin_vel[:, 0] / sagittal_scale)
            + torch.square(self.base_lin_vel[:, 1] / lateral_scale)
        )
        return torch.clamp(drift, max=4.0) * yaw_cmd.float()

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(contacts.float(), dim=1) == 1
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.03) | (torch.abs(self.commands[:, 2]) > 0.1)
        return single_contact.float() * moving.float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - 0.35) * first_contact, dim=1)
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.03) | (torch.abs(self.commands[:, 2]) > 0.1)
        rew_air_time *= moving.float()
        self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_feet_slip(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
        feet_xy_vel = self.rigid_body_state[:, self.feet_indices, 7:9]
        slip = torch.sum(torch.square(feet_xy_vel), dim=-1)
        return torch.sum(slip * contacts.float(), dim=1)

    def _reward_dof_pos_default(self):
        pose_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return pose_error * (1.0 - self._squat_skill_mask())

    def _reward_stand_still(self):
        no_motion_cmd = (
            (torch.abs(self.commands[:, 0]) < 0.02)
            & (torch.abs(self.commands[:, 1]) < 0.02)
            & (torch.abs(self.commands[:, 2]) < 0.1)
        )
        pose_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        return pose_error * no_motion_cmd.float() * (1.0 - self._squat_skill_mask())

    def _reward_base_height(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )
        return torch.square(base_height - self.target_base_height)
