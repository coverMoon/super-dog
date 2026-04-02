from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .arm_trajectory_library import ARM_TRAJECTORY_LIBRARY

class BlackArmCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        # 1. 初始姿态
        pos = [0.0, 0.0, 0.45] 
        default_joint_angles = {
            'FL_hip_joint': 0.1,   'FL_thigh_joint': 0.8014,   'FL_calf_joint': -1.527,
            'FR_hip_joint': -0.1,  'FR_thigh_joint': -0.8014,  'FR_calf_joint': 1.527,
            'RL_hip_joint': -0.1,   'RL_thigh_joint': 0.8014,   'RL_calf_joint': -1.527,
            'RR_hip_joint': 0.1,  'RR_thigh_joint': -0.8014,  'RR_calf_joint': 1.527,
            'arm_yaw_joint': 0.0,
            'arm_pitch_1_joint': 3.14,
            'arm_pitch_2_joint': -2.81,
            'arm_pitch_3_joint': -0.34,
            'arm_roll_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 2. PD 参数
        # 刚度 (P Gain)
        stiffness = {
            'FL_hip_joint': 50.0, 'RL_hip_joint': 50.0, 'FR_hip_joint': 50.0, 'RR_hip_joint': 50.0,
            'FL_thigh_joint': 50.0, 'RL_thigh_joint': 50.0, 'FR_thigh_joint': 50.0, 'RR_thigh_joint': 50.0,
            'FL_calf_joint': 50.0, 'RL_calf_joint': 50.0, 'FR_calf_joint': 50.0, 'RR_calf_joint': 50.0,
            'arm_yaw_joint': 30.0, 'arm_pitch_1_joint': 30.0, 'arm_pitch_2_joint': 30.0,
            'arm_pitch_3_joint': 20.0, 'arm_roll_joint': 10.0,
        }
        # 阻尼 (D Gain)
        damping = {
            'FL_hip_joint': 1.5, 'RL_hip_joint': 1.5, 'FR_hip_joint': 1.5, 'RR_hip_joint': 1.5,
            'FL_thigh_joint': 1.5, 'RL_thigh_joint': 1.5, 'FR_thigh_joint': 1.5, 'RR_thigh_joint': 1.5,
            'FL_calf_joint': 1.5, 'RL_calf_joint': 1.5, 'FR_calf_joint': 1.5, 'RR_calf_joint': 1.5,
            'arm_yaw_joint': 1.0, 'arm_pitch_1_joint': 1.0, 'arm_pitch_2_joint': 1.0,
            'arm_pitch_3_joint': 0.8, 'arm_roll_joint': 0.5,
        }

        # # 刚度 (P Gain)
        # stiffness = {
        #     'FL_hip_joint': 60.0, 'RL_hip_joint': 60.0, 'FR_hip_joint': 60.0, 'RR_hip_joint': 60.0,
        #     'FL_thigh_joint': 50.0, 'RL_thigh_joint': 50.0, 'FR_thigh_joint': 50.0, 'RR_thigh_joint': 50.0,
        #     'FL_calf_joint': 50.0, 'RL_calf_joint': 50.0, 'FR_calf_joint': 50.0, 'RR_calf_joint': 50.0
        # }
        # # 阻尼 (D Gain)
        # damping = {
        #     'FL_hip_joint': 1.5, 'RL_hip_joint': 1.5, 'FR_hip_joint': 1.5, 'RR_hip_joint': 1.5,
        #     'FL_thigh_joint': 1.2, 'RL_thigh_joint': 1.2, 'FR_thigh_joint': 1.2, 'RR_thigh_joint': 1.2,
        #     'FL_calf_joint': 1.2, 'RL_calf_joint': 1.2, 'FR_calf_joint': 1.2, 'RR_calf_joint': 1.2
        # }

        # stiffness = {
        #     'FL_hip_joint': 25.0, 'RL_hip_joint': 25.0, 'FR_hip_joint': 25.0, 'RR_hip_joint': 25.0,
        #     'FL_thigh_joint': 25.0, 'RL_thigh_joint': 25.0, 'FR_thigh_joint': 25.0, 'RR_thigh_joint': 25.0,
        #     'FL_calf_joint': 25.0, 'RL_calf_joint': 25.0, 'FR_calf_joint': 25.0, 'RR_calf_joint': 25.0
        # }
        # damping = {
        #     'FL_hip_joint': 0.8, 'RL_hip_joint': 0.8, 'FR_hip_joint': 0.8, 'RR_hip_joint': 0.8,
        #     'FL_thigh_joint': 0.8, 'RL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RR_thigh_joint': 0.8,
        #     'FL_calf_joint': 0.8, 'RL_calf_joint': 0.8, 'FR_calf_joint': 0.8, 'RR_calf_joint': 0.8
        # }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        # 3. 指定 URDF 路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/black_with_arm/urdf/black_with_arm_train.urdf'
        name = "black_arm"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base", "thigh"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1=disable

    class commands:
        curriculum = True
        max_curriculum = 2.0
        curriculum_threshold = 0.7
        curriculum_ema_alpha = 0.2
        curriculum_required_passes = 2
        curriculum_buffer_min = 256
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [0.1, 0.1]   # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # lin_vel_x = [0.0, 0.0] # min max [m/s]
            # lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            # heading = [0.0, 0.0]

    class domain_rand:
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
        
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.08
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # 地形类型：[平地，光滑斜坡，崎岖斜坡，楼梯下，楼梯上，乱石，梅花桩，沟壑，木板桥，高墙]
        # 当前混合地形分支中剥离断桥，避免与下楼梯在盲策略上产生冲突。
        terrain_proportions = [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
  
    class arm:
        motion_enabled = False
        motion_only_for_low_speed_commands = True
        motion_max_lin_speed = 0.0
        motion_max_yaw_speed = 0.0
        start_time = 1.5
        ramp_time = 2.0
        default_segment_duration = 1.0
        payload_enabled = True
        payload_probability = 0.75
        payload_body_name = 'arm_link_5'
        payload_mass_range = [0.35, 0.70]
        payload_local_offset = [0.0, 0.0, 0.125]
        payload_offset_jitter = [0.05, 0.05, 0.03]
        payload_timing_jitter = 0.12
        trajectory_rand_enabled = True
        trajectory_joint_offset_range = [0.10, 0.18, 0.18, 0.15, 0.45]
        trajectory_time_scale_range = [0.7, 1.3]
        task_mode_probs = [0.2, 0.5, 0.3]  # locomotion_only, manip_stationary, carry_move
        manip_lin_vel_x_range = [0.0, 0.0]
        manip_lin_vel_y_range = [0.0, 0.0]
        manip_ang_vel_yaw_range = [0.0, 0.0]
        manip_heading_offset_range = [0.0, 0.0]
        carry_lin_vel_x_range = [0.10, 0.40]
        carry_lin_vel_y_range = [-0.15, 0.15]
        carry_ang_vel_yaw_range = [-0.40, 0.40]
        carry_heading_offset_range = [-0.35, 0.35]
        carry_joint_angles = {
            'arm_yaw_joint': 0.0,
            'arm_pitch_1_joint': 2.72,
            'arm_pitch_2_joint': -2.28,
            'arm_pitch_3_joint': -0.58,
            'arm_roll_joint': -0.5,
        }
        trajectory_library = ARM_TRAJECTORY_LIBRARY

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6

        # additional: (stance_mask, contact_mask), base_lin_vel, external_forces, scan_dots
        num_one_step_privileged_obs = 45 + 3 + 3 + 187

        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        stuck_timeout_s = 4.0
        stuck_vel_threshold = 0.05
        stuck_command_threshold = 0.2
        stuck_grace_s = 1.0

    class rewards(LeggedRobotCfg.rewards):
        cycle_time = 0.5
        clearance_height_target = 0.05
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.80
        base_height_target = 0.421
        only_positive_rewards = False
        class terrain_adaptive:
            enabled = False
            terrain_variability_clip = 0.10

            class orientation:
                enabled = True
                mode = "decay"
                sigma = 0.03
                min_scale = 0.3
                max_scale = 1.0

            class smoothness:
                enabled = True
                mode = "decay"
                sigma = 0.02
                min_scale = 0.8
                max_scale = 1.0

            class action_rate:
                enabled = True
                mode = "decay"
                sigma = 0.02
                min_scale = 0.6
                max_scale = 1.0

            class torques:
                enabled = False
                mode = "decay"
                sigma = 0.05
                min_scale = 0.6
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
                max_extra_clearance = 0.10
                stance_gain = 0.5
                swing_high_penalty_weight = 0.25

        class scales:
            termination = -200.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -10.0
            dof_acc = -0.0
            joint_power = -1e-6
            base_height = -1.0
            foot_clearance = -3.0
            action_rate = -0.05
            smoothness = -0.01
            feet_air_time = 0.3
            collision = -0.05
            feet_stumble = -0.5
            stand_still = -2.0
            torques = -1e-6
            dof_vel = -1e-7
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -1e-5
            trot = 2.0
            hip_pos = -0.8 
            all_joint_pos = -0.001
            foot_slip = -0.3
            # feet_spacing = -0.1
            foot_impact_vel = -0.1
            progress = 0.5

class BlackArmCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 16 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        sym_loss = True
        obs_permutation = [
            # 删除前两个相位相关项
            0.00001, -1, -2,  # 原[2,-3,-4] -> 目标速度，角速度（2-2=0, 3-2=1, 4-2=2）
    
            -3, 4, -5,  # 原[-5,6,-7] -> 角速度（5-2=3, 6-2=4, 7-2=5）
    
            6, -7, 8,  # 原[8,-9,10] -> 重力投影（8-2=6, 9-2=7, 10-2=8）
    
            -12, -13, -14,  # 原[-14,15,16]（14-2=12, 15-2=13, 16-2=14）
            -9, -10, -11,   # 原[-11,12,13]（11-2=9, 12-2=10, 13-2=11）
            -18, -19, -20,  # 原[-20,21,22]（20-2=18, 21-2=19, 22-2=20）
            -15, -16, -17,  # 原[-17,18,19]（17-2=15, 18-2=16, 19-2=17）
    
            -24, -25,  -26,  # 原[-26,27,28]（26-2=24, 27-2=25, 28-2=26）
            -21, -22, -23,  # 原[-23,24,25]（23-2=21, 24-2=22, 25-2=23）
            -30, -31, -32,  # 原[-32,33,34]（32-2=30, 33-2=31, 34-2=32）
            -27, -28, -29,  # 原[-29,30,31]（29-2=27, 30-2=28, 31-2=29）
    
            -36, -37, -38,  # 原[-38,39,40]（38-2=36, 39-2=37, 40-2=38）
            -33, -34, -35,  # 原[-35,36,37]（35-2=33, 36-2=34, 37-2=35）
            -42, -43, -44,  # 原[-44,45,46]（44-2=42, 45-2=43, 46-2=44）
            -39, -40, -41,   # 原[-41,42,43]（41-2=39, 42-2=40, 43-2=41）
            # -45,-46
        ]
        act_permutation = [ -3, -4, -5, -0.0001, -1, -2, -9, -10, -11,-6, -7, -8,]#关节电机的对陈关系
        frame_stack = 6
        sym_coef = 0.8
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_black_arm'
        max_iterations=5000
         
