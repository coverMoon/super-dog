from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BlackCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        # 1. 初始姿态
        pos = [0.0, 0.0, 0.50] 
        # default_joint_angles = {
        #     'FL_hip_joint': 0.0,   'FL_thigh_joint': 0.82,   'FL_calf_joint': -1.5,
        #     'FR_hip_joint': -0.0,  'FR_thigh_joint': -0.82,  'FR_calf_joint': 1.5,
        #     'RL_hip_joint': 0.0,   'RL_thigh_joint': 0.82,   'RL_calf_joint': -1.5,
        #     'RR_hip_joint': -0.0,  'RR_thigh_joint': -0.82,  'RR_calf_joint': 1.5
        # }
        default_joint_angles = {
            'FL_hip_joint': 0.0,   'FL_thigh_joint': 0.82,   'FL_calf_joint': -1.3,
            'FR_hip_joint': -0.0,  'FR_thigh_joint': -0.82,  'FR_calf_joint': 1.3,
            'RL_hip_joint': 0.0,   'RL_thigh_joint': 0.82,   'RL_calf_joint': -1.3,
            'RR_hip_joint': -0.0,  'RR_thigh_joint': -0.82,  'RR_calf_joint': 1.3
        }

    class control(LeggedRobotCfg.control):
        # 2. PD 参数
        # 刚度 (P Gain)
        stiffness = {
            'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0,
            'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0,
            'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0
        }
        # stiffness = {
        #     'FL_hip_joint': 25.0, 'RL_hip_joint': 25.0, 'FR_hip_joint': 25.0, 'RR_hip_joint': 25.0,
        #     'FL_thigh_joint': 25.0, 'RL_thigh_joint': 25.0, 'FR_thigh_joint': 25.0, 'RR_thigh_joint': 25.0,
        #     'FL_calf_joint': 25.0, 'RL_calf_joint': 25.0, 'FR_calf_joint': 25.0, 'RR_calf_joint': 25.0
        # }
        # 阻尼 (D Gain)
        damping = {
            'FL_hip_joint': 1.0, 'RL_hip_joint': 1.0, 'FR_hip_joint': 1.0, 'RR_hip_joint': 1.0,
            'FL_thigh_joint': 1.0, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': 1.0, 'RL_calf_joint': 1.0, 'FR_calf_joint': 1.0, 'RR_calf_joint': 1.0
        }
        # damping = {
        #     'FL_hip_joint': 0.8, 'RL_hip_joint': 0.8, 'FR_hip_joint': 0.8, 'RR_hip_joint': 0.8,
        #     'FL_thigh_joint': 0.8, 'RL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RR_thigh_joint': 0.8,
        #     'FL_calf_joint': 0.8, 'RL_calf_joint': 0.8, 'FR_calf_joint': 0.8, 'RR_calf_joint': 0.8
        # }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        # 3. 指定 URDF 路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/black/urdf/black_description.urdf'
        name = "black"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base", "thigh"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1=disable

    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-2, 3]

        randomize_com_displacement = True
        com_displacement_range = [-0.08, 0.08]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = True
        friction_range = [0.2, 1.5]
        
        randomize_restitution = False
        restitution_range = [0., 1.0]
        
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        
        randomize_kp = True
        kp_range = [0.7, 1.3]
        
        randomize_kd = True
        kd_range = [0.7, 1.3]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.0

        # [修改] 延迟设置
        delay = True
        # 延迟步数范围：0 到 3 步 (每步 20ms，即 0ms-60ms 延迟)
        # 必须小于等于 action_queue 的长度
        lag_timesteps = 3 
        
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
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
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 地形类型：[光滑斜坡，崎岖斜坡，楼梯上，楼梯下，乱石，梅花桩，沟壑，断桥，陷坑]
        terrain_proportions = [0.1, 0.1, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
  
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 47
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 47 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class rewards(LeggedRobotCfg.rewards):
        cycle_time = 0.6
        clearance_height_target = -0.446
        soft_dof_pos_limit = 0.9
        base_height_target = 0.476
        only_positive_rewards = False
        max_contact_force = 100.
        class scales:
            termination = -200.0
            tracking_lin_vel = 1.7
            tracking_ang_vel = 1.2
            lin_vel_z = -2.5
            ang_vel_xy = -0.1
            orientation = -5.0
            dof_acc = -2.5e-6
            joint_power = -2e-5
            base_height = -5.0
            foot_clearance = -0.01
            action_rate = -0.05
            smoothness = -0.01
            feet_air_time = 0.01
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -1.5
            torques = -0.0
            dof_vel = -0.01
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            trot = 2.0
            hip_pos = -0.3
            all_joint_pos = -0.001
            foot_slip = -0.8
            lateral_vel_penalty = -1.0
            feet_spacing = -0.1
            foot_impact_vel = -1.0

class BlackCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_black_dog'
        # 指定算法
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        max_iterations = 1000
        