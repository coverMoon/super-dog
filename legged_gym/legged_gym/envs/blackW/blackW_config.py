from legged_gym.envs.black.black_config import BlackCfg, BlackCfgPPO


class BlackWCfg(BlackCfg):
    class init_state(BlackCfg.init_state):
        pos = [0.0, 0.0, 0.60]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_hip_joint': 0.05,   'FL_thigh_joint': 0.8014,   'FL_calf_joint': -1.527, 'FL_wheel_joint': 0.0,
            'FR_hip_joint': -0.05,  'FR_thigh_joint': -0.8014,  'FR_calf_joint': 1.527,  'FR_wheel_joint': 0.0,
            'RL_hip_joint': -0.05,   'RL_thigh_joint': 0.8014,   'RL_calf_joint': -1.527, 'RL_wheel_joint': 0.0,
            'RR_hip_joint': 0.05,  'RR_thigh_joint': -0.8014,  'RR_calf_joint': 1.527,  'RR_wheel_joint': 0.0,
        }

    class control(BlackCfg.control):
        control_type = 'P'
        stiffness = {
            'FL_hip_joint': 50.0, 'RL_hip_joint': 50.0, 'FR_hip_joint': 50.0, 'RR_hip_joint': 50.0,
            'FL_thigh_joint': 50.0, 'RL_thigh_joint': 50.0, 'FR_thigh_joint': 50.0, 'RR_thigh_joint': 50.0,
            'FL_calf_joint': 50.0, 'RL_calf_joint': 50.0, 'FR_calf_joint': 50.0, 'RR_calf_joint': 50.0,
            'FL_wheel_joint': 0.0, 'RL_wheel_joint': 0.0, 'FR_wheel_joint': 0.0, 'RR_wheel_joint': 0.0,
        }
        damping = {
            'FL_hip_joint': 1.2, 'RL_hip_joint': 1.2, 'FR_hip_joint': 1.2, 'RR_hip_joint': 1.2,
            'FL_thigh_joint': 1.2, 'RL_thigh_joint': 1.2, 'FR_thigh_joint': 1.2, 'RR_thigh_joint': 1.2,
            'FL_calf_joint': 1.2, 'RL_calf_joint': 1.2, 'FR_calf_joint': 1.2, 'RR_calf_joint': 1.2,
            'FL_wheel_joint': 1.0, 'RL_wheel_joint': 1.0, 'FR_wheel_joint': 1.0, 'RR_wheel_joint': 1.0,
        }
        # stiffness = {
        #     'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0,
        #     'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0,
        #     'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0,
        #     'FL_wheel_joint': 0.0, 'RL_wheel_joint': 0.0, 'FR_wheel_joint': 0.0, 'RR_wheel_joint': 0.0,
        # }
        # damping = {
        #     'FL_hip_joint': 1.0, 'RL_hip_joint': 1.0, 'FR_hip_joint': 1.0, 'RR_hip_joint': 1.0,
        #     'FL_thigh_joint': 1.0, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
        #     'FL_calf_joint': 1.0, 'RL_calf_joint': 1.0, 'FR_calf_joint': 1.0, 'RR_calf_joint': 1.0,
        #     'FL_wheel_joint': 1.0, 'RL_wheel_joint': 1.0, 'FR_wheel_joint': 1.0, 'RR_wheel_joint': 1.0,
        # }
        action_scale = {
            'FL_hip_joint': 0.25, 'RL_hip_joint': 0.25, 'FR_hip_joint': 0.25, 'RR_hip_joint': 0.25,
            'FL_thigh_joint': 0.25, 'RL_thigh_joint': 0.25, 'FR_thigh_joint': 0.25, 'RR_thigh_joint': 0.25,
            'FL_calf_joint': 0.25, 'RL_calf_joint': 0.25, 'FR_calf_joint': 0.25, 'RR_calf_joint': 0.25,
        }
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
        terrain_proportions = [0.1, 0.05, 0.05, 0.6, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
        high_wall_fill_full_block = True
        high_wall_spawn_clearance = 0.8
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
        heading_command = True
        heading_command_difficult_only = True
        simple_heading_command_prob = 0.3
        low_speed_x_range = [-1.0, 1.0]
        high_vel_env_fraction = 0.2
        high_speed_lateral_disable_x_threshold = 1.5
        xy_norm_stop_threshold = 0.2
        heading_yaw_gain = 0.5
        heading_yaw_clip = 3.0
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

        randomize_hip_motor_strength = True
        hip_motor_strength_range = [0.75, 1.05]

        randomize_hip_damping = True
        hip_damping_scale_range = [0.8, 1.5]

        randomize_calf_backlash = True
        # B2 sim2real approximation for calf linkage/gear play. The play mode
        # keeps the effective target inside a backlash window and weakens Kp in free play.
        calf_backlash_mode = "play"
        calf_backlash_range = [0.01, 0.055]
        calf_backlash_min_kp_scale = 0.08
        calf_backlash_engage_start = 0.6
        calf_backlash_leak = 0.02

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

        randomize_wheel_friction = True
        # Multiplied onto per-env base friction; wheel friction is at most the sampled base friction.
        # 乘在每个 env 的基础摩擦系数上，轮子摩擦最高等于基础摩擦。
        wheel_friction_scale_range = [0.4, 1.0]

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
        dof_pos_limit_margin_ratio = 0.10
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

        base_height_target = 0.52
        only_positive_rewards = False
        max_contact_force = 100.0

        stand_still_cmd_threshold = 0.1
        stand_still_yaw_threshold = 0.1
        run_still_x_threshold = 0.1 # 大于时触发
        # Legacy names kept for config compatibility; run_still now decays with y/yaw command magnitude.
        run_still_y_threshold = 0.1
        run_still_yaw_threshold = 0.15

        termination_contact_force_threshold = 1.0
        collision_force_threshold = 0.1
        feet_stumble_ratio = 3.0
        foot_impact_contact_force = 1.0
        foot_impact_vel_threshold = 0.2

        orientation_terrain_adaptive = True
        orientation_terrain_variability_clip = 0.30
        orientation_terrain_sigma = 0.0010
        orientation_terrain_min_scale = 0.03
        orientation_terrain_max_scale = 1.0

        class hip_default:
            cmd_min_scale = 0.5
            y_ref = 0.5
            yaw_ref = 1.0
            y_scale = 0.35
            yaw_scale = 0.35

        class run_still:
            use_command_decay = False
            cmd_min_scale = 0.5
            y_ref = 0.5
            yaw_ref = 1.0
            y_scale = 0.35
            yaw_scale = 0.35

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
            # Keep high-wall behavior configurable while allowing stairs to use the simpler Jun06-style lift target.
            stairs_use_simple_lift = False
            horizontal_force_threshold = 15.0
            command_threshold = 0.2
            obstacle_height_threshold = 0.025
            clearance_margin = 0.065
            high_obstacle_height_threshold = 0.18
            high_obstacle_clearance_margin = 0.03
            high_obstacle_active_height_span = 0.18
            high_obstacle_extra_active_time = 1.2
            min_lift_height = 0.05
            min_progress_span = 0.03
            active_time = 0.8
            progress_weight = 0.55
            target_sigma = 0.05
            over_lift_margin = 0.035
            over_lift_sigma = 0.04
            over_lift_penalty_weight = 0.12
            rear_lift_target_offset = 0.02
            rear_lift_target_offset_terrain_types = [3]
            unloaded_lift_suppress_height = 0.05
            unloaded_lift_suppress_sigma = 0.05
            multi_wheel_coordination = True
            multi_wheel_coordination_terrain_types = [3]
            multi_wheel_pair_residual = 0.2
            multi_wheel_diagonal_residual = 0.2
            multi_wheel_high_wall_pair_residual = 0.4
            multi_wheel_high_wall_diagonal_residual = 0.4
            forward_offsets = [0.04, 0.08, 0.12, 0.16]
            lateral_offsets = [-0.03, 0.0, 0.03]

        class wheel_obstacle_spin:
            # When enabled, stairs type 3 use the older Jun06 hard-threshold anti-spin logic; other terrains keep continuous slip.
            stairs_use_threshold_spin = False
            terrain_types = [3, 4, 9]
            horizontal_force_threshold = 15.0
            command_threshold = 0.2
            obstacle_height_threshold = 0.02
            slip_speed_sigma = 0.25
            progress_speed_sigma = 0.4
            stairs_continuous_scale = 1.5
            spin_threshold = 8.0
            progress_threshold = 0.25

        class stairs_multi_contact_progress:
            command_threshold = 0.2
            horizontal_force_threshold = 15.0
            obstacle_height_threshold = 0.02
            min_contact_count = 2
            min_command_speed = 0.2

        class stairs_pair_escape:
            command_threshold = 0.2
            horizontal_force_threshold = 15.0
            obstacle_height_threshold = 0.02
            min_progress_span = 0.03
            front_pair_weight = 0.9
            rear_pair_weight = 2.3

        class stairs_rear_target_bonus:
            command_threshold = 0.2
            horizontal_force_threshold = 15.0
            obstacle_height_threshold = 0.02
            min_progress_span = 0.03
            high_progress_threshold = 0.7

        class stairs_rear_stuck_escape:
            command_threshold = 0.2
            horizontal_force_threshold = 15.0
            obstacle_height_threshold = 0.02
            min_command_speed = 0.2
            progress_ratio_threshold = 0.4
            slip_speed_threshold = 0.2
            min_progress_span = 0.03
            high_progress_threshold = 0.5

        class wheel_lateral_clearance:
            command_threshold = 0.18
            full_command_threshold = 0.45
            target_extra_height = 0.02
            tracking_sigma = 0.025
            top_k = 2
            terrain_variability_clip = 0.30
            terrain_variability_sigma = 0.0015
            terrain_min_scale = 0.0
            terrain_max_scale = 1.0

        class scales:
            # Command tracking and forward progress.
            # 指令跟踪与前向推进奖励。
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            tracking_ang_vel = 2.0
            progress = 1.0

            # Base posture and body stability.
            # 机身姿态、高度与整体稳定性约束。
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -3.0
            roll_orientation = 0.0
            base_height = -5.0
            dof_pos_limits = -0.2

            # Joint posture and stand behavior.
            # 关节姿态回中与静止行为约束。
            hip_default = -0.35
            stand_still = -0.6
            run_still = -1.0
            stairs_run_still = -0.8
            stand_wheel_action = -0.2
            stand_wheel_vel = -0.02

            # Contact and obstacle handling.
            # 接触碰撞、绊脚与越障相关项。
            collision = -1.0
            feet_stumble = -0.1
            foot_impact_vel = -0.02
            wheel_obstacle_lift = 2.0
            wheel_obstacle_unloaded_lift = -0.05
            wheel_obstacle_spin = -1.0
            stairs_multi_contact_progress = 1.0
            stairs_pair_escape = 1.3
            stairs_rear_target_bonus = 0.0
            stairs_rear_stuck_escape = 0.0
            wheel_lateral_clearance = 0.45

            # Action and actuator regularization.
            # 动作平滑、力矩与关节速度正则项。
            action_rate = -0.067
            smoothness = -0.01
            torques = -6.2e-4
            dof_vel = -1e-7
            dof_acc = -1e-7

            # Disabled legacy or experimental terms.
            # 当前关闭的历史项或预留实验项。
            termination = -0.0
            tracking_lin_vel = 0.0
            joint_power = -0.0
            foot_clearance = -0.0
            feet_air_time = 0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            trot = 0.0
            hip_pos = -0.0 
            all_joint_pos = -0.0
            foot_slip = -0.0
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
        entropy_coef = 0.0032
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
        sym_coef = 1.0

    class runner(BlackCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        save_interval = 50
        num_steps_per_env = 64
        max_iterations = 1000  
        experiment_name = "rough_blackW_dog"
        run_name = ""
        checkpoint = -1
        resume_path = None
        resume = True
        load_run = "Jul01_03-04-42_"
