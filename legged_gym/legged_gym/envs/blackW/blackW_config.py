from legged_gym.envs.black.black_config import BlackCfg, BlackCfgPPO


class BlackWCfg(BlackCfg):
    class init_state(BlackCfg.init_state):
        # 1. 初始姿态
        pos = [0.0, 0.0, 0.55]
        default_joint_angles = {
            'FL_hip_joint': 0.0,   'FL_thigh_joint': 0.8014,   'FL_calf_joint': -1.527, 'FL_wheel_joint': 0.0,
            'FR_hip_joint': -0.0,  'FR_thigh_joint': -0.8014,  'FR_calf_joint': 1.527,  'FR_wheel_joint': 0.0,
            'RL_hip_joint': 0.0,   'RL_thigh_joint': 0.8014,   'RL_calf_joint': -1.527, 'RL_wheel_joint': 0.0,
            'RR_hip_joint': -0.0,  'RR_thigh_joint': -0.8014,  'RR_calf_joint': 1.527,  'RR_wheel_joint': 0.0,
        }

    class control(BlackCfg.control):
        # 2. PD 参数
        control_type = 'P'

        # 刚度 (P Gain)
        stiffness = {
            'FL_hip_joint': 45.0, 'RL_hip_joint': 45.0, 'FR_hip_joint': 45.0, 'RR_hip_joint': 45.0,
            'FL_thigh_joint': 45.0, 'RL_thigh_joint': 45.0, 'FR_thigh_joint': 45.0, 'RR_thigh_joint': 45.0,
            'FL_calf_joint': 45.0, 'RL_calf_joint': 45.0, 'FR_calf_joint': 45.0, 'RR_calf_joint': 45.0,
            'FL_wheel_joint': 0.0, 'RL_wheel_joint': 0.0, 'FR_wheel_joint': 0.0, 'RR_wheel_joint': 0.0,
        }
        # 阻尼 (D Gain)
        damping = {
            'FL_hip_joint': 1.2, 'RL_hip_joint': 1.2, 'FR_hip_joint': 1.2, 'RR_hip_joint': 1.2,
            'FL_thigh_joint': 1.2, 'RL_thigh_joint': 1.2, 'FR_thigh_joint': 1.2, 'RR_thigh_joint': 1.2,
            'FL_calf_joint': 1.2, 'RL_calf_joint': 1.2, 'FR_calf_joint': 1.2, 'RR_calf_joint': 1.2,
            'FL_wheel_joint': 1.0, 'RL_wheel_joint': 1.0, 'FR_wheel_joint': 1.0, 'RR_wheel_joint': 1.0,
        }

        # stiffness = {
        #     'FL_hip_joint': 60.0, 'RL_hip_joint': 60.0, 'FR_hip_joint': 60.0, 'RR_hip_joint': 60.0,
        #     'FL_thigh_joint': 50.0, 'RL_thigh_joint': 50.0, 'FR_thigh_joint': 50.0, 'RR_thigh_joint': 50.0,
        #     'FL_calf_joint': 50.0, 'RL_calf_joint': 50.0, 'FR_calf_joint': 50.0, 'RR_calf_joint': 50.0,
        #     'FL_wheel_joint': 0.0, 'RL_wheel_joint': 0.0, 'FR_wheel_joint': 0.0, 'RR_wheel_joint': 0.0,
        # }
        # damping = {
        #     'FL_hip_joint': 1.5, 'RL_hip_joint': 1.5, 'FR_hip_joint': 1.5, 'RR_hip_joint': 1.5,
        #     'FL_thigh_joint': 1.2, 'RL_thigh_joint': 1.2, 'FR_thigh_joint': 1.2, 'RR_thigh_joint': 1.2,
        #     'FL_calf_joint': 1.2, 'RL_calf_joint': 1.2, 'FR_calf_joint': 1.2, 'RR_calf_joint': 1.2,
        #     'FL_wheel_joint': 2.0, 'RL_wheel_joint': 2.0, 'FR_wheel_joint': 2.0, 'RR_wheel_joint': 2.0,
        # }

        action_scale = 0.25

        # Wheel control:
        # - "learned": policy directly outputs wheel angular velocity reference
        # - "residual": wheel angular velocity reference = command-based target + policy residual
        wheel_control_mode = "learned"
        vel_scale = 20.0
        wheel_residual_scale = 3.0
        # Wheel radius [m]. Used in _target_wheel_velocities() to convert linear velocity
        # command to wheel angular velocity.
        wheel_radius = 0.103
        # Half of left-right wheel track [m]. Used in _target_wheel_velocities()
        # for yaw differential speed.
        wheel_base_half_width = 0.183
        # Sign used to map positive command x to positive rolling direction.
        # Use leg-name keys so deployment/training stay correct even if Isaac Gym
        # reorders the runtime DOF sequence.
        wheel_forward_sign = {
            "FL": 1.0,
            "FR": -1.0,
            "RL": 1.0,
            "RR": -1.0,
        }
        # Control updates once every `decimation` physics steps.
        decimation = 4

    class asset(BlackCfg.asset):
        # 3. 指定 URDF 路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/blackW/blackW_description.urdf'
        name = "blackW"
        foot_name = "foot"
        wheel_name = ["wheel"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base", "thigh"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1  # 1=disable
        replace_cylinder_with_capsule = False

    class commands(BlackCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        curriculum_threshold = 0.6
        curriculum_ema_alpha = 0.5
        curriculum_required_passes = 2
        curriculum_buffer_min = 128
        max_curriculum_y = 1.0
        max_curriculum_yaw = 3.14
        y_curriculum_threshold = 0.45
        yaw_curriculum_threshold = 0.35
        y_curriculum_step = 0.1
        yaw_curriculum_step = 0.2
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        
        # 指定命令空间和采样概率
        stand_command_prob = 0.1
        x_command_prob = 0.3
        y_command_prob = 0.3
        yaw_command_prob = 0.3
        mixed_command_prob = 0.0
        # 最小非零命令值
        min_nonzero_lin_cmd = 0.2
        min_nonzero_yaw_cmd = 0.2

        class ranges(BlackCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(BlackCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-2.0, 4.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.75, 1.25]

        randomize_friction = True
        friction_range = [0.3, 1.35]

        randomize_restitution = False
        restitution_range = [0., 1.0]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_kp = True
        kp_range = [0.8, 1.2]

        randomize_kd = True
        kd_range = [0.8, 1.2]

        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]

        randomize_inertia = True
        inertia_range = [0.5, 1.5]

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        push_robots = True
        push_interval_s = 30
        max_push_vel_xy = 2.5

        # [修改] 延迟设置
        delay = True
        # 延迟步数范围
        lag_timesteps = 3

    class noise(BlackCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(BlackCfg.noise.noise_scales):
            dof_pos = 0.08
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class terrain(BlackCfg.terrain):
        mesh_type = 'plane'  # options: 'plane', 'heightfield', 'trimesh'
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # 地形类型：[平地，光滑斜坡，崎岖斜坡，楼梯下，楼梯上，乱石，梅花桩，沟壑，木板桥，高墙]
        terrain_proportions = [0.2, 0.1, 0.1, 0.2, 0.25, 0.15, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.75

    class rewards(BlackCfg.rewards):
        cycle_time = 0.8    # only for y/yaw
        clearance_height_target = 0.06  # only for y/yaw
        
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        tracking_ang_vel_sigma = 1.0
        command_activity_deadzone = 0.05
        command_activity_full = 0.2
        relative_tracking_sigma = 0.25
        relative_tracking_min_lin_cmd = 0.2
        relative_tracking_min_yaw_cmd = 0.3
        tracking_reward_weight = 0.6
        progress_reward_weight = 0.4
        inactive_lin_vel_weight = 1.0
        inactive_ang_vel_weight = 0.25
        wheel_tracking_relative_sigma = 0.25
        wheel_tracking_min_ref = 0.5
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.80
        base_height_target = 0.50
        only_positive_rewards = False

        class terrain_adaptive(BlackCfg.rewards.terrain_adaptive):
            enabled = False
            terrain_variability_clip = 0.30

            class orientation(BlackCfg.rewards.terrain_adaptive.orientation):
                enabled = True
                mode = "decay"
                sigma = 0.009
                min_scale = 0.15
                max_scale = 1.0

            class smoothness(BlackCfg.rewards.terrain_adaptive.smoothness):
                enabled = True
                mode = "decay"
                sigma = 0.2
                min_scale = 0.9
                max_scale = 1.0

            class action_rate(BlackCfg.rewards.terrain_adaptive.action_rate):
                enabled = True
                mode = "decay"
                sigma = 0.01
                min_scale = 0.20
                max_scale = 1.0

            class torques(BlackCfg.rewards.terrain_adaptive.torques):
                enabled = False
                mode = "decay"
                sigma = 0.05
                min_scale = 0.9
                max_scale = 1.0

            class progress(BlackCfg.rewards.terrain_adaptive.progress):
                enabled = False
                mode = "boost"
                sigma = 0.04
                min_scale = 1.0
                max_scale = 1.5

            class foot_clearance(BlackCfg.rewards.terrain_adaptive.foot_clearance):
                # 轮腿任务当前先关闭地形自适应抬脚项，避免和轮子接触/滚动目标互相干扰
                enabled = False
                mode = "margin"
                std_gain = 2.0
                max_extra_clearance = 0.15
                stance_gain = 0.5
                swing_high_penalty_weight = 0.25

        class raibert(BlackCfg.rewards.raibert):
            nominal_front_x = 0.21
            nominal_rear_x = -0.21
            nominal_y = 0.155
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

        class scales(BlackCfg.rewards.scales):
            # Active task rewards
            tracking_lin_vel = 5.0
            tracking_lin_vel_y = 5.0
            tracking_ang_vel = 5.0
            wheel_vel_ref_tracking = 2.0

            # Active posture/contact penalties
            termination = -500.0
            orientation = -5.0
            base_height = -5.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            inactive_axis_vel = -0.5
            collision = -0.1
            stand_still = -5.0
            stand_still_wheels = -3.0
            hip_pos = -3.0
            all_joint_pos = -0.5
            foothold = -1.0
            foot_clearance = -3.0
            feet_air_time = 0.8
            foot_impact_vel = -5.0
            trot = 2.0
            raibert = 2.0

            # Active smoothness/limit penalties
            action_rate = -0.05
            smoothness = -0.02
            wheel_action_rate = -0.05
            wheel_smoothness = -0.01
            torques = -1e-7
            dof_vel = -1e-7
            dof_pos_limits = -10.0
            torque_limits = -1e-5

            # Disabled rewards/penalties
            progress = 0.0
            dof_acc = -0.0
            joint_power = -0.0
            feet_stumble = -0.0
            dof_vel_limits = -0.0
            foot_slip = -0.0

    class env(BlackCfg.env):
        num_envs = 8192
        # commands(3) + base_ang_vel(3) + gravity(3) + dof_pos_err(16) + dof_vel(16) + actions(16)
        num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6

        # additional: base_lin_vel, external_forces, scan_dots
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187

        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 16
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True
        episode_length_s = 20
        stuck_timeout_s = 4.0
        stuck_vel_threshold = 0.05
        stuck_yaw_vel_threshold = 0.1
        stuck_command_threshold = 0.2
        stuck_grace_s = 1.0


class BlackWCfgPPO(BlackCfgPPO):
    class policy(BlackCfgPPO.policy):
        # Grouped exploration std for wheel-legged control.
        # Group 0: leg joints (hip/thigh/calf), Group 1: wheel joints.
        init_noise_std = 1.0  # fallback when grouped std is disabled
        leg_init_noise_std = 1.0
        wheel_init_noise_std = 0.5
        action_std_groups = [
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
            [3, 7, 11, 15],
        ]
        action_std_group_init_noise_std = [leg_init_noise_std, wheel_init_noise_std]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(BlackCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.003
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        learning_rate_min = 1e-5
        learning_rate_max = 5e-3
        max_grad_norm = 1.
        # 单帧观测布局:
        # commands(3) + base_ang_vel(3) + gravity(3) + dof_pos_err(16) + dof_vel(16) + actions(16)
        # 左右镜像下: vx 保持, vy/yaw 取反; wx/wz 取反, wy 保持; gravity 的 y 分量取反。
        # 腿关节按 FL<->FR, RL<->RR 交换；轮子相关维度保持 identity，不参与对称损失。
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
        run_name = ''
        num_steps_per_env = 100
        experiment_name = 'rough_blackW_dog'
        max_iterations = 500
