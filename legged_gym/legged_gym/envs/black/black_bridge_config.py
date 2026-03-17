from legged_gym.envs.black.black_config import BlackCfg, BlackCfgPPO


class BlackBridgeCfg(BlackCfg):
    class commands(BlackCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        curriculum_threshold = 0.65
        curriculum_ema_alpha = 0.4
        curriculum_required_passes = 2

        class ranges(BlackCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-2.0, 2.0]
            heading = [-2.0, 2.0]

    class terrain(BlackCfg.terrain):
        # flat, smooth slope, rough slope, stairs down, stairs up, discrete, stones, gap, bridge, wall
        terrain_proportions = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]
        max_init_terrain_level = 2

        # Bridge course: (10 cm gap,long plank) -> (10 cm,short) -> (20 cm,long) -> (20 cm,short)
        bridge_gap_options = [0.1, 0.1, 0.2, 0.2]
        bridge_plank_length_options = [0.6, 0.4, 0.6, 0.4]
        bridge_pit_depth_options = [0.2, 0.5, 0.8, 1.0]
        bridge_plank_width = 8.0
        bridge_height = 0.0
        bridge_platform_len = 2.0

    class env(BlackCfg.env):
        episode_length_s = 15
        stuck_timeout_s = 0.8
        stuck_vel_threshold = 0.08
        stuck_foot_height_margin = 0.1
        stuck_command_threshold = 0.2
        stuck_grace_s = 0.6

    class rewards(BlackCfg.rewards):
        class scales(BlackCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            orientation = -1.5
            base_height = -1.0
            foot_clearance = -2.0
            gap_clearance = 8.0
            gap_recovery_burst = 10.0
            edge_escape = 4.0
            action_rate = -0.03
            smoothness = -0.01
            feet_air_time = 0.8
            feet_stumble = -0.3
            foot_slip = -0.15
            progress = 0.5
            collision = -0.01
            dof_acc = -1e-8
            stand_still = -2.0


class BlackBridgeCfgPPO(BlackCfgPPO):
    class runner(BlackCfgPPO.runner):
        experiment_name = "bridge_black_dog"
        max_iterations = 3000
