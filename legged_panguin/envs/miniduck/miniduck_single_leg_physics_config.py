"""Configuration for the independent physics-seeded single-leg line."""

from legged_panguin.envs.base.legged_robot_config import LeggedRobotCfg

from .miniduck_config import MiniDuckFlatCfg, MiniDuckFlatCfgPPO


class MiniDuckSingleLegPhysicsCfg(MiniDuckFlatCfg):
    class env(MiniDuckFlatCfg.env):
        episode_length_s = 12.0

    class init_state(MiniDuckFlatCfg.init_state):
        # V4 physics-search candidate.  It holds for 1.38 s with fixed PD and
        # gives a fresh feedback policy a useful, non-degenerate initial basin.
        base_roll = -0.257519394159317
        base_pitch = 0.13767313957214355
        pos = [0.0, 0.0, 0.1824520230293274]
        rot = [
            -0.12810010626663665,
            0.06821283434134051,
            0.008831926262425155,
            0.9893732203188502,
        ]
        reset_xy_range = 0.002
        reset_yaw_range = 0.008
        reset_roll_range = 0.006
        reset_pitch_range = 0.006
        reset_lin_vel_range = 0.002
        reset_ang_vel_range = 0.004
        reset_joint_scale_range = [0.999, 1.001]
        reset_joint_offset = 0.002
        default_joint_angles = {
            "left_hip_yaw": 0.1740003228187561,
            "left_hip_roll": 0.4000000059604645,
            "left_hip_pitch": -0.5701542496681213,
            "left_knee": 1.0475670099258423,
            "left_ankle": -0.3223668932914734,
            "right_hip_yaw": 0.14658042788505554,
            "right_hip_roll": -0.4000000059604645,
            "right_hip_pitch": 0.8053550720214844,
            "right_knee": 1.3463585376739502,
            "right_ankle": -0.07076922804117203,
        }

    class control(MiniDuckFlatCfg.control):
        action_scale = 0.12
        use_action_scale_curriculum = False

    class commands(MiniDuckFlatCfg.commands):
        curriculum = False
        resampling_time = 12.0
        fixed_support_foot = 0
        # P0 learns the validated left-support problem only.  A mirrored state
        # becomes a separate curriculum stage after this baseline converges.
        fixed_support_steps = 10**12

        class ranges(MiniDuckFlatCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [1.0, 1.0]
            ang_vel_yaw = [0.0, 0.0]
            advanced_lin_vel_x = [0.0, 0.0]
            advanced_lin_vel_y = [1.0, 1.0]
            advanced_ang_vel_yaw = [0.0, 0.0]

    class rewards(MiniDuckFlatCfg.rewards):
        reward_version = "single_leg_physics_v1"
        only_positive_rewards = False
        termination_height = 0.075
        termination_body_angle = 0.90
        teacher_reference_path = ""

        base_height_target = 0.1824520230293274
        base_height_quality_sigma = 0.000225
        com_x_sigma = 0.001225
        com_y_sigma = 0.000225
        com_x_limit = 0.040
        com_y_limit = 0.018
        reference_pose_sigma = 0.025
        swing_clearance_min = 0.025
        swing_clearance_max = 0.050
        swing_clearance_sigma = 0.00010
        # Compatibility parameters for inherited V2 diagnostics.  Foot
        # flatness is deliberately excluded from the Physics reward and its
        # hold threshold remains disabled below.
        foot_flatness_sigma = 0.035
        base_over_support_sigma = 0.0025
        body_attitude_sigma = 0.060
        stillness_sigma = 0.060
        support_slip_sigma = 0.0016
        max_support_contact_force = 25.0

        hold_grace_s = 0.20
        hold_ramp_s = 2.0
        hold_max_base_speed = 0.20
        hold_max_tilt_rate = 0.80
        hold_min_attitude_quality = 0.45
        hold_min_foot_flatness_quality = -1.0

        pose_weight_start = 0.30
        pose_weight_end = 0.08
        pose_decay_steps = 3000 * 24
        swing_contact_penalty = 0.60
        airborne_penalty = 0.70
        impact_penalty = 0.15
        com_outside_penalty = 0.50
        torque_soft_ratio = 0.70
        torque_saturation_penalty = 0.20

        class quality_weights:
            true_com = 0.25
            body_attitude = 0.12
            base_height = 0.10
            swing_clearance = 0.10
            stillness = 0.05
            support_slip = 0.05
            continuous_hold = 0.03

        monitor_terms = [
            "stance_valid",
            "true_com_over_support_quality",
            "reference_pose_quality",
            "body_attitude_quality",
            "base_height_quality",
            "swing_clearance_quality",
            "stillness_quality",
            "support_slip_quality",
            "continuous_hold_quality",
            "com_outside_support",
            "torque_saturation_excess",
            "airborne",
            "swing_contact",
            "impact_excess",
        ]

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -20.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -0.35
            ang_vel_xy = -0.25
            orientation = 0.0
            torques = -8.0e-5
            dof_vel = -5.0e-4
            dof_acc = -1.0e-7
            base_height = 0.0
            feet_air_time = 0.0
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.08
            stand_still = 0.0
            alive = 0.0
            dof_pos_limits = -0.35
            single_leg_physics = 12.0

    class domain_rand(MiniDuckFlatCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_link_mass = False
        randomize_dof_properties = False
        push_robots = False
        curriculum_push_robots = False
        max_action_delay = 0
        max_imu_delay = 0
        joint_target_offset = 0.0
        encoder_offset = 0.0
        gyro_bias = 0.0
        gravity_bias = 0.0
        accel_bias = 0.0

    class noise(MiniDuckFlatCfg.noise):
        add_noise = False


class MiniDuckSingleLegPhysicsCfgPPO(MiniDuckFlatCfgPPO):
    class policy(MiniDuckFlatCfgPPO.policy):
        init_noise_std = 0.04
        reset_loaded_std = None

    class algorithm(MiniDuckFlatCfgPPO.algorithm):
        learning_rate = 3.0e-5
        entropy_coef = 5.0e-4
        min_policy_std = 0.03
        symmetry_loss_coef = 0.0

    class runner(MiniDuckFlatCfgPPO.runner):
        experiment_name = "single_leg_miniduck_physics"
        run_name = ""
        resume = False
        max_iterations = 12000
