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

    class control( BlackCfg.control ):
        # PD Drive parameters:
        control_type = 'P' 
        # 刚度 (P Gain)
        stiffness = {
            'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0,
            'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0,
            'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0,
            'FL_wheel_joint': 0.0, 'RL_wheel_joint': 0.0, 'FR_wheel_joint': 0.0, 'RR_wheel_joint': 0.0,
        }
        # 阻尼 (D Gain)
        damping = {
            'FL_hip_joint': 1.0, 'RL_hip_joint': 1.0, 'FR_hip_joint': 1.0, 'RR_hip_joint': 1.0,
            'FL_thigh_joint': 1.0, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': 1.0, 'RL_calf_joint': 1.0, 'FR_calf_joint': 1.0, 'RR_calf_joint': 1.0,
            'FL_wheel_joint': 1.0, 'RL_wheel_joint': 1.0, 'FR_wheel_joint': 1.0, 'RR_wheel_joint': 1.0,
        }
        action_scale = 0.25
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
        decimation = 4
        wheel_speed = 1
    

    class terrain(BlackCfg.terrain):
        mesh_type = "plane"
        static_friction = 0.8
        dynamic_friction = 0.8
        # Map Go2W's [smooth slope, rough slope, stairs up, stairs down, discrete]
        # onto this repo's [plane, smooth slope, rough slope, stairs down,
        # stairs up, discrete, ...] terrain table.
        terrain_proportions = [0.0, 0.1, 0.1, 0.2, 0.35, 0.25, 0.0, 0.0, 0.0, 0.0]

    class commands(BlackCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class asset(BlackCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/blackW/blackW_description.urdf"
        name = "blackW"
        foot_name = "foot"
        wheel_name = ["wheel"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["thigh", "calf", "base"]
        self_collisions = 1
        replace_cylinder_with_capsule = False

    class domain_rand(BlackCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.25, 1.25]

        randomize_restitution = False
        restitution_range = [0.0, 1.0]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.9, 1.1]

        randomize_kd = True
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]

        randomize_inertia = False
        inertia_range = [1.0, 1.0]

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        delay = True
        lag_timesteps = 3

        randomize_wheel_delay = False
        randomize_wheel_motor = False
        randomize_wheel_mass = False
        randomize_wheel_geometry = False

    class rewards(BlackCfg.rewards):
        class scales:
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -10.0
            hip_default = -0.5
            stand_still = -0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.01
            torques = -5.0e-4
            dof_vel = -1e-7
            dof_acc = -1e-7
            run_still = -0.05

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.53
        max_contact_force = 100.0

    class env(BlackCfg.env):
        num_envs = 4096
        num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 16


class BlackWCfgPPO(BlackCfgPPO):
    class policy(BlackCfgPPO.policy):
        init_noise_std = 1.0
        action_std_groups = None
        action_std_group_init_noise_std = None

    class algorithm(BlackCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 1.0e-3
        learning_rate_min = 1e-5
        learning_rate_max = 1e-2
        sym_loss = False
        obs_permutation = None
        act_permutation = None
        sym_action_mask = None

    class runner(BlackCfgPPO.runner):
        save_interval = 20
        num_steps_per_env = 48
        max_iterations = 500
        experiment_name = "blackW_go2w_reward"
        run_name = ""
        resume = None
        load_run = -1
        checkpoint = -1
        resume_path = None


BlackWGo2WRewardCfg = BlackWCfg
BlackWGo2WRewardCfgPPO = BlackWCfgPPO
