# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from legged_panguin import LEGGED_GYM_ROOT_DIR
from legged_panguin.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


JOINT_MIRROR_PERMUTATION = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
JOINT_MIRROR_SIGNS = [-1.0, -1.0, -1.0, 1.0, 1.0] * 2


def _symmetry_layout():
    permutation = list(range(64))
    signs = [1.0] * 64

    signs[0:3] = [-1.0, 1.0, -1.0]  # angular velocity is an axial vector
    signs[3:6] = [1.0, -1.0, 1.0]
    signs[6:9] = [1.0, -1.0, 1.0]
    signs[9:12] = [1.0, -1.0, -1.0]

    for offset in (12, 22, 32, 42, 52):
        permutation[offset:offset + 10] = [
            offset + index for index in JOINT_MIRROR_PERMUTATION
        ]
        signs[offset:offset + 10] = JOINT_MIRROR_SIGNS
    signs[62:64] = [1.0, -1.0]  # gait phase is stored as cos/sin
    return permutation, signs


SYMMETRY_OBS_PERMUTATION, SYMMETRY_OBS_SIGNS = _symmetry_layout()


class MiniDuckFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # gyro, gravity, accelerometer, command, q, dq, three action frames, phase
        num_observations = 64
        num_actions = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        curriculum = False

    class init_state(LeggedRobotCfg.init_state):
        # PhysX evolutionary search result, cross-validated in MuJoCo:
        # 25.895 degrees forward pitch and 50/50 randomized 5 s trials passed.
        base_pitch = 0.451947301626
        pos = [0.0, 0.0, 0.115629099309]
        rot = [0.0, 0.224055365304, 0.0, 0.974576417362]
        reset_xy_range = 0.005
        reset_yaw_range = 0.02
        reset_roll_range = 0.015
        reset_pitch_range = 0.02
        reset_lin_vel_range = 0.005
        reset_ang_vel_range = 0.010
        reset_joint_scale_range = [0.998, 1.002]
        reset_joint_offset = 0.0
        default_joint_angles = {
            'left_hip_yaw': 0.0,
            'left_hip_roll': 0.0,
            'left_hip_pitch': -0.399379879236,
            'left_knee': 1.5,
            'left_ankle': -0.623843669891,
            'right_hip_yaw': 0.0,
            'right_hip_roll': 0.0,
            'right_hip_pitch': 0.399379879236,
            'right_knee': 1.5,
            'right_ankle': -0.623843669891,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            'hip_yaw': 13.37,
            'hip_roll': 13.37,
            'hip_pitch': 13.37,
            'knee': 13.37,
            'ankle': 13.37,
        }
        damping = {
            'hip_yaw': 0.15,
            'hip_roll': 0.15,
            'hip_pitch': 0.15,
            'knee': 0.15,
            'ankle': 0.15,
        }
        # Race curriculum: first preserve the teacher gait, then allow a
        # moderately longer stride.  Increasing this without retraining makes
        # the real servos saturate and is therefore intentionally conservative.
        action_scale = 0.25
        use_action_scale_curriculum = True
        warmup_action_scale = 0.10
        action_scale_warmup_steps = 24_000
        action_scale_ramp_steps = 72_000
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/miniduck/urdf/MINIDUCK.urdf'
        name = 'miniduck'
        foot_name = 'foot_assembly'
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        collapse_fixed_joints = False
        flip_visual_attachments = False
        self_collisions = 1
        thickness = 0.001

    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma = 0.05
        tracking_lin_sigma = 0.015
        tracking_ang_sigma = 0.06
        soft_dof_pos_limit = 0.98
        base_height_target = 0.115629099309
        max_contact_force = 120.0
        termination_height = 0.08
        termination_body_angle = 0.85
        minimum_command_ratio = 0.30
        only_positive_rewards = False
        teacher_reference_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "data",
            "best_walk_teacher_fullgrid_action_no_head_legs_v2.pkl",
        )
        teacher_command_scale = [0.15, 0.18, 0.85]
        foot_lift_deadband = 0.006
        foot_lift_target = 0.025
        clock_phase_gate = 0.18
        moving_without_step_progress = 0.20
        monitor_terms = [
            "tracking_lin_vel_x",
            "tracking_lin_vel_y",
            "tracking_ang_vel_yaw",
            "teacher_action",
            "teacher_target",
            "teacher_action_error",
            "teacher_target_delta_error",
            "reference_gait_delta",
            "phase_support_match",
            "phase_swing_lift",
            "clocked_single_support",
            "clocked_swing_lift",
            "moving_without_step",
            "lateral_yaw_rate",
            "lateral_heading_error",
            "sagittal_heading_error",
            "sagittal_lateral_displacement",
            "yaw_twist_without_step",
        ]

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -250.0
            alive = 2.2
            tracking_lin_vel = 1.0
            tracking_lin_vel_x = 1.5
            tracking_ang_vel = 0.8
            teacher_action = 2.4
            teacher_target = 0.75
            teacher_action_error = -1.0
            teacher_target_delta_error = -5.5
            reference_gait_delta = 1.8
            phase_support_match = 0.0
            phase_swing_lift = 0.0
            swing_contact = 0.0
            clocked_single_support = 2.4
            clocked_swing_lift = 3.6
            clocked_swing_contact = -5.0
            moving_contact_switch = 0.9
            moving_without_step = -2.2
            foot_lift = 1.0
            swing_clearance = 2.5
            single_support = 0.9
            double_support = -0.18
            no_contact = -8.0
            yaw_alternating_contact = 2.4
            yaw_contact_switch = 1.8
            yaw_twist_without_step = -28.0
            sagittal_progress = 0.8
            lateral_progress = 0.7
            yaw_progress = 0.7
            sagittal_step_progress = 2.3
            lateral_step_progress = 2.2
            yaw_step_progress = 2.4
            sagittal_lag = -1.2
            lateral_lag = -1.5
            yaw_lag = -1.6
            command_stall = -0.8
            support_contact = 1.5
            no_fly = 0.75
            feet_air_time = 0.55
            feet_slip = -0.25
            lin_vel_z = -1.0
            ang_vel_xy = -0.35
            orientation = -4.0
            base_height = -6.0
            sagittal_axis_isolation = -0.40
            # Unlike axis isolation, these terms penalize accumulated drift
            # over a command segment and directly match the 2 m race metric.
            sagittal_heading_error = -3.0
            sagittal_lateral_displacement = -2.0
            lateral_axis_isolation = -0.55
            lateral_yaw_rate = -1.6
            lateral_heading_error = -2.2
            yaw_axis_isolation = -0.45
            torques = -1.5e-4
            dof_acc = -2.0e-7
            action_rate = -0.16
            collision = -2.0
            dof_pos_limits = -0.5
            dof_pos_default = -0.04
            stand_still = -0.15

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.0
        zero_prob = 0.05
        sagittal_prob = 0.40
        lateral_prob = 0.30
        yaw_prob = 0.25
        stage_zero_prob = 0.15
        # 12000 PPO iterations * 24 simulator steps per iteration = 288000 steps.
        # This schedule compresses the previously successful 42000-iteration
        # curriculum so a cold-start run reaches the final mixed command regime.
        sagittal_phase_steps = 24_000
        lateral_phase_steps = 24_000
        yaw_phase_steps = 24_000
        min_abs_x = 0.025
        min_abs_y = 0.015
        min_abs_yaw = 0.06
        advanced_curriculum_start_step = 120_000
        advanced_lateral_rehearsal_steps = 24_000
        advanced_yaw_rehearsal_steps = 24_000
        advanced_balanced_steps = 24_000
        advanced_x_ramp_steps = 48_000
        advanced_lateral_ramp_steps = 48_000
        advanced_yaw_ramp_steps = 72_000
        advanced_min_abs_x = 0.035
        advanced_min_abs_y = 0.050
        advanced_min_abs_yaw = 0.25
        sagittal_turn_prob = 0.15
        advanced_sagittal_turn_prob = 0.15
        sagittal_turn_yaw_scale = 0.25
        advanced_lateral_zero_prob = 0.05
        advanced_lateral_sagittal_prob = 0.25
        advanced_lateral_lateral_prob = 0.60
        advanced_lateral_yaw_prob = 0.10
        advanced_yaw_zero_prob = 0.05
        advanced_yaw_sagittal_prob = 0.25
        advanced_yaw_lateral_prob = 0.15
        advanced_yaw_yaw_prob = 0.55
        advanced_balanced_zero_prob = 0.05
        advanced_balanced_sagittal_prob = 0.25
        advanced_balanced_lateral_prob = 0.45
        advanced_balanced_yaw_prob = 0.25
        advanced_final_zero_prob = 0.05
        # Keep the final stage balanced across translation and turning so
        # straight-line accuracy does not erase lateral/yaw flexibility.
        advanced_final_sagittal_prob = 0.35
        advanced_final_lateral_prob = 0.30
        advanced_final_yaw_prob = 0.30

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.08, 0.10]
            lin_vel_y = [-0.10, 0.10]
            ang_vel_yaw = [-0.45, 0.45]
            # 0.16 m/s gives a 12.5 s ideal 2 m time.  Raise further only after
            # the benchmark reaches >85% completion with low drift.
            advanced_lin_vel_x = [-0.08, 0.16]
            advanced_lin_vel_y = [-0.20, 0.20]
            advanced_ang_vel_yaw = [-1.00, 1.00]

    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.65, 1.10]
        randomize_base_mass = False
        randomize_link_mass = True
        link_mass_scale_range = [0.95, 1.05]
        trunk_added_mass_range = [-0.03, 0.03]
        trunk_com_range = [0.0025, 0.0025, 0.0020]
        randomize_dof_properties = True
        use_sts3215_xml_dof_defaults = True
        sts3215_dof_damping = 0.56
        sts3215_dof_friction = 0.068
        sts3215_dof_armature = 0.027
        sts3215_effort = 3.23
        sts3215_velocity = 5.24
        dof_friction_scale_range = [0.95, 1.05]
        dof_damping_scale_range = [0.95, 1.05]
        dof_armature_scale_range = [1.00, 1.02]
        motor_strength_range = [0.90, 1.10]
        kp_factor_range = [0.95, 1.05]
        kd_factor_range = [0.90, 1.10]
        nominal_motor_velocity = 5.24
        motor_velocity_range = [4.20, 5.60]
        joint_target_offset = 0.015
        encoder_offset = 0.010
        gyro_bias = 0.020
        gravity_bias = 0.015
        accel_bias = 0.10
        max_action_delay = 1
        max_imu_delay = 1
        curriculum_warmup_steps = 12_000
        curriculum_ramp_steps = 84_000
        push_robots = False
        curriculum_push_robots = True
        push_start_steps = 96_000
        push_interval_s = 10
        max_push_vel_xy = 0.08
        advanced_push_start_steps = 96_000
        advanced_push_ramp_steps = 144_000
        advanced_max_push_vel_xy = 0.06

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            accel = 0.10

    class noise(LeggedRobotCfg.noise):
        noise_level = 0.55

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.015
            dof_vel = 1.5
            ang_vel = 0.12
            gravity = 0.035
            accel = 0.20

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            contact_offset = 0.002
            rest_offset = 0.0
            max_depenetration_velocity = 0.25


class MiniDuckFlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.16
        reset_loaded_std = 0.14
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 2.0e-3
        learning_rate = 5.0e-5
        schedule = 'fixed'
        max_grad_norm = 0.20
        min_policy_std = 0.08
        symmetry_loss_coef = 0.12
        symmetry_obs_permutation = SYMMETRY_OBS_PERMUTATION
        symmetry_obs_signs = SYMMETRY_OBS_SIGNS
        symmetry_action_permutation = JOINT_MIRROR_PERMUTATION
        symmetry_action_signs = JOINT_MIRROR_SIGNS

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_miniduck'
        max_iterations = 2000
