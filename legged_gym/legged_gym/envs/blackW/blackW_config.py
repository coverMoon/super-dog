from legged_gym.envs.black.black_config import BlackCfg, BlackCfgPPO


class BlackWCfg(BlackCfg):
    class init_state(BlackCfg.init_state):
        pos = [0.0, 0.0, 0.60]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_hip_joint': 0.0,   'FL_thigh_joint': 0.8014,   'FL_calf_joint': -1.527, 'FL_wheel_joint': 0.0,
            'FR_hip_joint': -0.0,  'FR_thigh_joint': -0.8014,  'FR_calf_joint': 1.527,  'FR_wheel_joint': 0.0,
            'RL_hip_joint': 0.0,   'RL_thigh_joint': 0.8014,   'RL_calf_joint': -1.527, 'RL_wheel_joint': 0.0,
            'RR_hip_joint': -0.0,  'RR_thigh_joint': -0.8014,  'RR_calf_joint': 1.527,  'RR_wheel_joint': 0.0,
        }

    class control(BlackCfg.control):
        control_type = 'P'
        stiffness = {
            'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0,
            'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0,
            'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0,
            'FL_wheel_joint': 0.0, 'RL_wheel_joint': 0.0, 'FR_wheel_joint': 0.0, 'RR_wheel_joint': 0.0,
        }
        damping = {
            'FL_hip_joint': 1.0, 'RL_hip_joint': 1.0, 'FR_hip_joint': 1.0, 'RR_hip_joint': 1.0,
            'FL_thigh_joint': 1.0, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': 1.0, 'RL_calf_joint': 1.0, 'FR_calf_joint': 1.0, 'RR_calf_joint': 1.0,
            'FL_wheel_joint': 1.0, 'RL_wheel_joint': 1.0, 'FR_wheel_joint': 1.0, 'RR_wheel_joint': 1.0,
        }
        action_scale = 0.25
        decimation = 4
        hip_reduction = 1.0

        wheel_control_mode = "learned"
        vel_scale = 10.0
        wheel_residual_scale = 3.0
        wheel_radius = 0.103
        wheel_base_half_width = 0.20975
        wheel_forward_sign = {
            "FL": 1.0,
            "FR": -1.0,
            "RL": 1.0,
            "RR": -1.0,
        }
        wheel_speed = 1

        motor_friction_coulomb = 0.35
        motor_friction_velocity_scale = 3.0
        motor_friction_viscous = 0.1

    class asset(BlackCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/blackW/blackW_description.urdf"
        name = "blackW"
        foot_name = "foot"
        wheel_name = ["wheel"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["thigh", "calf", "base"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class env(BlackCfg.env):
        num_envs = 4096
        num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 16
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        stuck_timeout_s = 4.0
        stuck_vel_threshold = 0.05
        stuck_command_threshold = 0.2
        stuck_grace_s = 1.0

    class terrain(BlackCfg.terrain):
        mesh_type = 'trimesh'  # options: 'plane', 'heightfield', 'trimesh'
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        height_measurement_base_offset = 0.5
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # 地形类型：[平地，光滑斜坡，崎岖斜坡，楼梯上，楼梯下，乱石，梅花桩，沟壑，木板桥，高墙]
        terrain_proportions = [0.1, 0.1, 0.1, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1]
        slope_treshold = 0.75

    class commands(BlackCfg.commands):
        curriculum = True
        
        # Kept for parent/legacy compatibility; current blackW per-axis curriculum only uses buffer_min.
        # curriculum_threshold = 0.7
        # curriculum_ema_alpha = 0.2
        # curriculum_required_passes = 2
        curriculum_buffer_min = 256

        num_commands = 4
        resampling_time = 10.0
        heading_command = False
        low_speed_x_range = [-1.0, 1.0]
        high_vel_env_fraction = 0.2
        high_speed_lateral_disable_x_threshold = 1.5
        xy_norm_stop_threshold = 0.2
        heading_yaw_gain = 0.5
        heading_yaw_clip = 2.0
        x_curriculum_step = 0.2
        y_curriculum_step = 0.1
        yaw_curriculum_step = 0.1
        x_curriculum_score_scale = 0.8
        y_curriculum_score_scale = 0.75
        yaw_curriculum_score_scale = 0.6
        max_curriculum = 2.0
        max_y_curriculum = 1.0
        max_yaw_curriculum = 3.0

        class terrain_command_sampling:
            enabled = True
            # 地形类型索引：0=平地，1=光滑斜坡，2=崎岖斜坡，3=楼梯上，4=楼梯下，
            # 5=乱石，6=梅花桩，7=沟壑，8=木板桥，9=高墙。
            difficult_terrain_types = [3, 4, 6, 7, 8, 9]
            # 简单地形不单独设置，直接跟随全局 command_ranges。
            difficult_lin_vel_y = [-0.5, 0.5]
            difficult_ang_vel_yaw = [-0.5, 0.5]

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

    class domain_rand(BlackCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.25, 1.25]

        randomize_restitution = True
        restitution_range = [0.0, 0.1]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.9, 1.1]

        randomize_kd = True
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]

        randomize_inertia = True
        inertia_range = [0.9, 1.1]

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        delay = True
        lag_timesteps = 3

        randomize_wheel_delay = True
        wheel_lag_timesteps = 4

        randomize_wheel_motor = True
        wheel_motor_strength_range = [0.8, 1.2]
        wheel_vel_ref_scale_range = [0.9, 1.1]

        randomize_wheel_vel_ref_bias = True
        wheel_vel_ref_bias_range = [-0.3, 0.3]

        randomize_wheel_dof_vel_obs_bias = True
        wheel_dof_vel_obs_bias_range = [-0.5, 0.5]

        randomize_wheel_mass = True
        wheel_mass_scale_range = [0.9, 1.1]
        wheel_inertia_scale_range = [0.8, 1.2]

        randomize_wheel_geometry = True
        wheel_radius_scale_range = [0.9, 1.1]
        wheel_base_half_width_scale_range = [0.95, 1.05]

    class normalization(BlackCfg.normalization):
        class obs_scales(BlackCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(BlackCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(BlackCfg.noise.noise_scales):
            dof_pos = 0.08
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards(BlackCfg.rewards):
        cycle_time = 0.8
        clearance_height_target = 0.08
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.53
        only_positive_rewards = False
        max_contact_force = 100.0
        stand_still_cmd_threshold = 0.1
        run_still_cmd_threshold = 0.1
        termination_contact_force_threshold = 1.0
        collision_force_threshold = 0.1
        feet_stumble_ratio = 3.0

        class terrain_adaptive:
            enabled = False
            terrain_variability_clip = 0.30

            class orientation:
                enabled = True
                mode = "decay"
                sigma = 0.009
                min_scale = 0.15
                max_scale = 1.0

            class smoothness:
                enabled = True
                mode = "decay"
                sigma = 0.2
                min_scale = 0.9
                max_scale = 1.0

            class action_rate:
                enabled = True
                mode = "decay"
                sigma = 0.01
                min_scale = 0.20
                max_scale = 1.0

            class torques:
                enabled = False
                mode = "decay"
                sigma = 0.05
                min_scale = 0.9
                max_scale = 1.0

            class progress:
                enabled = False
                mode = "boost"
                sigma = 0.04
                min_scale = 1.0
                max_scale = 1.5

            class foot_clearance:
                enabled = True
                mode = "margin"
                std_gain = 2.0
                max_extra_clearance = 0.15
                stance_gain = 0.5
                swing_high_penalty_weight = 0.25

        class raibert(BlackCfg.rewards.raibert):
            # blackW foothold rewards use the wheel cylinder center rather than the old foot point.
            nominal_front_x = 0.2055
            nominal_rear_x = -0.2197
            nominal_y = 0.1834
            max_linear_offset_x = 0.16
            max_linear_offset_y = 0.06
            vel_error_gain = 0.3
            yaw_gain = 1.0
            max_yaw_offset = 0.1
            tracking_sigma = 0.06
            late_swing_start_x = 0.35
            late_swing_start_latyaw = 0.1
            touchdown_gain = 0.4
            approach_bonus = 0.25
            max_approach_speed = 0.4

        class wheel_obstacle_lift:
            horizontal_force_threshold = 25.0
            command_threshold = 0.2
            obstacle_height_threshold = 0.035
            clearance_margin = 0.05
            min_lift_height = 0.05
            min_progress_span = 0.03
            active_time = 0.8
            progress_weight = 0.7
            target_sigma = 0.05
            forward_offsets = [0.04, 0.08, 0.12, 0.16]
            lateral_offsets = [-0.03, 0.0, 0.03]

        class scales:
            termination = -0.0
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            tracking_ang_vel = 1.25
            progress = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -5.0
            hip_default = -0.6
            stand_still = -0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.08
            smoothness = -0.015
            torques = -5.0e-4
            dof_vel = -1e-7
            dof_acc = -1e-7
            run_still = -0.05

            wheel_obstacle_lift = 1.0
            tracking_lin_vel = 0.0
            joint_power = -0.0
            foot_clearance = -0.0
            feet_air_time = 0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            trot = 0.0
            hip_pos = -0.0
            all_joint_pos = -0.0
            foot_slip = -0.0
            foot_impact_vel = -0.0
            raibert = 0.0

    class viewer(BlackCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11.0, 5.0, 3.0]

    class sim(BlackCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx(BlackCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class BlackWCfgPPO(BlackCfgPPO):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'

    class policy(BlackCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        action_std_groups = None
        action_std_group_init_noise_std = None
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(BlackCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        learning_rate_min = 1e-5
        learning_rate_max = 1e-2
        max_grad_norm = 1.0
        # Single-frame actor obs:
        # commands(3) + base_ang_vel(3) + gravity(3) + dof_pos_err(16) + dof_vel(16) + actions(16).
        # Runtime DOF groups are FL, FR, RR, RL. Mirror swaps left/right legs, flips lateral/yaw
        # signed channels, and keeps wheel channels out of the action symmetry loss.
        sym_loss = True
        obs_permutation = [
            0.00001, -1, -2,
            -3, 4, -5,
            6, -7, 8,
            -13, -14, -15, 12,
            -9, -10, -11, 16,
            -21, -22, -23, 20,
            -17, -18, -19, 24,
            -29, -30, -31, 28,
            -25, -26, -27, 32,
            -37, -38, -39, 36,
            -33, -34, -35, 40,
            -45, -46, -47, 44,
            -41, -42, -43, 48,
            -53, -54, -55, 52,
            -49, -50, -51, 56,
        ]
        act_permutation = [
            -4, -5, -6, 3,
            -0.0001, -1, -2, 7,
            -12, -13, -14, 11,
            -8, -9, -10, 15,
        ]
        sym_action_mask = [
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
        ]
        frame_stack = 6
        sym_coef = 0.6

    class runner(BlackCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        save_interval = 20
        num_steps_per_env = 64
        max_iterations = 5000
        experiment_name = "rough_blackW_dog"
        run_name = ""
        resume = None
        load_run = -1
        checkpoint = -1
        resume_path = None

