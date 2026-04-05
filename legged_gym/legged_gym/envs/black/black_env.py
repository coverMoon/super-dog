from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import torch

class BlackEnv(LeggedRobot):
    """
    自定义环境类 BlackEnv，适配 HIMLoco框架
    """

    def _init_base_height_points(self):
        """Use a footprint-aligned area slightly larger than the standing feet rectangle."""
        # Standing footprint is about 0.425 m x 0.311 m; expand it slightly to leave
        # margin around lifted feet and terrain edges under the support polygon.
        x = torch.tensor(
            [-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24, 0.30],
            device=self.device,
            requires_grad=False,
        )
        y = torch.tensor(
            [-0.18, -0.135, -0.09, -0.045, 0.0, 0.045, 0.09, 0.135, 0.18],
            device=self.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _draw_debug_vis(self):
        """Draw default terrain samples plus the under-body adaptive-scaling samples."""
        if not self.terrain.cfg.measure_heights:
            return

        super()._draw_debug_vis()
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        body_sphere = gymutil.WireframeSphereGeometry(0.015, 4, 4, None, color=(0, 1, 1))
        under_body_heights = self._get_under_body_height_samples()
        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            body_heights = under_body_heights[i].cpu().numpy()
            body_points = quat_apply_yaw(
                self.base_quat[i].repeat(body_heights.shape[0]), self.base_height_points[i]
            ).cpu().numpy()
            for j in range(body_heights.shape[0]):
                x = body_points[j, 0] + base_pos[0]
                y = body_points[j, 1] + base_pos[1]
                z = body_heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(body_sphere, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_buffers(self):
        """ 初始化 Buffer，额外获取所有刚体状态用于自定义奖励 """
        super()._init_buffers()
        self._init_raibert_buffers()

        # 获取所有刚体的状态(可用于计算脚部位置、速度等)
        # 形状：(num_envs, num_bodies, 13)
        # 13维包括：pos(3), quat(4), lin_vel(3), ang_vel(3)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

        # 重新初始化动作队列，长度根据 Config 决定
        hist_len = self.cfg.domain_rand.lag_timesteps + 1 # +1 是为了安全冗余
        self.action_queue = torch.zeros(self.num_envs, hist_len, self.num_actions, device=self.device, requires_grad=False)
        self.lag_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_impact_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.stuck_time = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    def _init_raibert_buffers(self):
        """为 trot/clearance/Raibert 奖励建立稳定的脚名映射与名义落脚点。"""
        cfg = self.cfg.rewards.raibert
        num_feet = len(self.feet_indices)
        self.foot_name_to_index = {}
        self.foot_phase_offsets = torch.zeros(num_feet, device=self.device, requires_grad=False)
        self.nominal_foothold_xy = torch.zeros(num_feet, 2, device=self.device, requires_grad=False)

        for i, foot_name in enumerate(self.feet_names):
            leg_prefix = foot_name.split("_")[0]
            self.foot_name_to_index[leg_prefix] = i

            is_front = leg_prefix.startswith("F")
            is_left = leg_prefix.endswith("L")

            self.foot_phase_offsets[i] = 0.0 if leg_prefix in ("FL", "RR") else 0.5
            self.nominal_foothold_xy[i, 0] = cfg.nominal_front_x if is_front else cfg.nominal_rear_x
            self.nominal_foothold_xy[i, 1] = cfg.nominal_y if is_left else -cfg.nominal_y

    def _get_feet_state_in_body_frame(self):
        """返回脚端相对机身的局部位置和速度。"""
        feet_rel_pos = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        feet_rel_vel = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        flat_base_quat = self.base_quat.unsqueeze(1).repeat(1, len(self.feet_indices), 1).view(-1, 4)

        feet_pos_body = quat_rotate_inverse(flat_base_quat, feet_rel_pos.reshape(-1, 3)).view(
            self.num_envs, len(self.feet_indices), 3
        )
        feet_vel_body = quat_rotate_inverse(flat_base_quat, feet_rel_vel.reshape(-1, 3)).view(
            self.num_envs, len(self.feet_indices), 3
        )
        return feet_pos_body, feet_vel_body

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # ============ 跨步延迟 ===============
        # 1. 更新队列：把当前动作放入队首
        # 队列形状：[env, history_len, action_dim]
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self.actions

        # 2. 决定使用哪个延迟动作
        if self.cfg.domain_rand.delay:
            # 使用 reset 时生成的固定延时，而不是每帧随机
            latency_indices = self.lag_buffer

            # 安全裁剪
            latency_indices = torch.clip(latency_indices, max=self.action_queue.size(1)-1)
            delayed_actions = self.action_queue[torch.arange(self.num_envs, device=self.device), latency_indices]
        else:
            delayed_actions = self.actions
        # ====================================

        # self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        # delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        # if self.cfg.domain_rand.delay:
        #     for i in range(self.cfg.control.decimation):
        #         self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)


        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)

            self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids) # 调用父类 reset

        # 为重置的环境随机生成一个新的固定延迟（0 到 max_lag-1）
        if self.cfg.domain_rand.delay:
            max_lag = self.cfg.domain_rand.lag_timesteps
            self.lag_buffer[env_ids] = torch.randint(0, max_lag, (len(env_ids),), device=self.device)
        self.last_impact_contacts[env_ids] = False
        self.stuck_time[env_ids] = 0.0

    def post_physics_step(self):
        """ 物理步后刷新状态 """
        env_ids, termination_privileged_obs = super().post_physics_step()
        # 手动刷新刚体状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """在默认接触/超时终止基础上，补充通用卡死检测。"""
        super().check_termination()

        move_cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        move_cmd = move_cmd_norm > self.cfg.env.stuck_command_threshold
        cmd_dir = self.commands[:, :2] / torch.clamp(move_cmd_norm.unsqueeze(1), min=1e-6)
        progress_speed = torch.sum(self.base_lin_vel[:, :2] * cmd_dir, dim=1)
        stalled = progress_speed < self.cfg.env.stuck_vel_threshold

        grace_done = (self.episode_length_buf.float() * self.dt) > self.cfg.env.stuck_grace_s
        stuck_mask = move_cmd & stalled & grace_done

        self.stuck_time = torch.where(stuck_mask, self.stuck_time + self.dt, torch.zeros_like(self.stuck_time))
        self.reset_buf |= self.stuck_time > self.cfg.env.stuck_timeout_s
    
    # def _post_physics_step_callback(self):
    #     """ Callback called before computing terminations, rewards, and observations
    #         Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
    #     """
    #     # 
    #     env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
    #     self._resample_commands(env_ids)
    #     if self.cfg.commands.heading_command:
    #         forward = quat_apply(self.base_quat, self.forward_vec)
    #         heading = torch.atan2(forward[:, 1], forward[:, 0])

    #         # 1. 计算原始的 Heading 误差
    #         heading_error = wrap_to_pi(self.commands[:, 3] - heading)

    #         # 2. 定义死区阈值
    #         # 线性速度死区：小于 0.1 m/s 视为静止意图
    #         lin_vel_deadzone = 0.1
    #         # 角度误差死区：小于 0.05 rad (约 2.86度) 视为已对准
    #         heading_error_deadzone = 0.05
    #         # 3. 判断是否进入死区
    #         # 如果 (水平指令很小) 且 (角度误差也很小) -> 强制不旋转
    #         is_standing = torch.norm(self.commands[:, :2], dim=1) < lin_vel_deadzone
    #         is_aligned = torch.abs(heading_error) < heading_error_deadzone
    #         should_stop_turning = is_standing & is_aligned

    #         # 4. 计算角速度指令 (P控制器)
    #         ang_vel_target = 0.5 * heading_error
            
    #         # 5. 应用死区：如果满足静止条件，将目标角速度强制置 0
    #         ang_vel_target[should_stop_turning] = 0.0
            
    #         # 6. 写入指令并裁剪
    #         self.commands[:, 2] = torch.clip(ang_vel_target, -2., 2.)

    #     if self.cfg.terrain.measure_heights:
    #         self.measured_heights = self._get_heights()
    #     if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
    #         self._push_robots()
    #     if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
    #         self._disturbance_robots()

    def _get_phase(self):
        """ 
        内部辅助函数，计算相位
        仅用于计算奖励函数，不作为观测输入给网络
        """

        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        return phase
    
    def _get_gait_phase(self):
        """
        根据相位生成理想的触地掩码 (Stance Mask)
        1 表示支撑相 (应触地)，0 表示摆动相 (应抬脚)
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        # 添加双支撑相 (Double Support Phase)
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        
        # 左腿支撑 (Left Stance) -> 对应 sin >= 0
        stance_mask[:, 0] = sin_pos >= 0
        # 右腿支撑 (Right Stance) -> 对应 sin < 0
        stance_mask[:, 1] = sin_pos < 0
        
        # 双支撑相：当 sin 值接近 0 时，两腿都应该着地
        # stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            
            # [新增] 初始化 terrain_levels 为 0，防止 Z 轴惩罚函数报错
            self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        # 随机化惯性张量 (Inertia Tensor)
        if hasattr(self.cfg.domain_rand, "randomize_inertia") and self.cfg.domain_rand.randomize_inertia:
            rng = self.cfg.domain_rand.inertia_range
            for i in range(len(props)):
                # 为三个主轴分别生成独立的随机缩放因子
                # 这模拟了质量分布在不同方向上的不确定性
                scale_x = np.random.uniform(rng[0], rng[1])
                scale_y = np.random.uniform(rng[0], rng[1])
                scale_z = np.random.uniform(rng[0], rng[1])
                
                # 修改惯性张量的对角元素 (Ixx, Iyy, Izz)
                # props[i].inertia 是一个 gymapi.Mat33 矩阵
                props[i].inertia.x.x *= scale_x
                props[i].inertia.y.y *= scale_y
                props[i].inertia.z.z *= scale_z

        return props

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        # =======================================================
        # [Sim2Real 修改] 添加电机经验摩擦模型 (M8010-6)
        # =======================================================
        # 经验参数：针对 Unitree Go1 的 M8010 电机
        # 干摩擦 (Coulomb Friction): ~0.8 Nm
        # 粘滞阻尼 (Viscous Damping): ~0.1 Nm/(rad/s)

        # 摩擦力方向与速度相反
        # friction_torque = 0.8 * torch.sign(self.dof_vel) + 0.1 * self.dof_vel
        # 使用 tanh 代替 sign 实现平滑过渡，避免零速震荡
        # friction_torque = 0.8 * torch.tanh(8.0 * self.dof_vel) + 0.1 * self.dof_vel
        friction_torque = 0.35 * torch.tanh(3.0 * self.dof_vel) + 0.1 * self.dof_vel

        # 从理想 PD 力矩中减去摩擦消耗
        torques = torques - friction_torque

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        """ Computes observations
        """

        # 计算相位信号
        # phase = self._get_phase()
        # sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        # cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        # stance_mask = self._get_gait_phase()    # 理想的触地时序
        # contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.    # 真实的触地状态

        # 构建基础观测向量
        current_obs = torch.cat((   
                                    self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ), dim=-1)

        # 添加噪声
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        # 拼接相位信号
        # 将相位拼接到最前面 (Sin, Cos, Commands, AngVel, ...)
        # 这样网络能最早“看到”周期信号
        # current_obs = torch.cat((sin_pos, cos_pos, base_obs), dim=-1)

        # 添加感知输入 (高度图等)
        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        # 更新历史缓存 (滑动窗口)
        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((
                                            #  stance_mask, # [2] 目标相位掩码
                                            #  contact_mask, # [4] 真实触地掩码
                                             current_obs[:, :self.num_one_step_privileged_obs], 
                                             self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), 

                                             dim=-1)
    
    def compute_termination_observations(self, env_ids):
        """ Computes observations for terminated environments (Critic needs this)
        """
        # 计算相位信号
        phase = self._get_phase()
        # sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        # cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        # stance_mask = self._get_gait_phase()    # 理想的触地时序
        # contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.    # 真实的触地状态

        # 构建基础物理观测
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        # 添加噪声
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        # 拼接相位信号 + 基础观测
        # 顺序匹配 compute_observations: [Sin, Cos, Commands, ...]
        #current_obs = torch.cat((sin_pos, cos_pos, base_obs), dim=-1)

        # 添加感知输入
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        
        # 添加高度图
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        # 返回特权观测
        # 这里只返回 termination_ids 对应的部分
        return torch.cat((
                        #   stance_mask, # [2] 目标相位掩码
                        #   contact_mask, # [4] 真实触地掩码
                          current_obs[:, :self.num_one_step_privileged_obs], 
                          self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), 
                          dim=-1)[env_ids]

    # ----------------------------------------------------------------------
    # 自定义奖励函数区域
    # ----------------------------------------------------------------------

    def _reward_trot(self):
        """
        [Trot 步态引导奖励]
        鼓励对角线脚同时接触地面，且符合目标相位
        """
        # 获取脚底 Z 轴接触力
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        # 使用 sigmoid 将力转换为触地概率 (0~1)
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)

        fl = contact_prob[:, self.foot_name_to_index["FL"]]
        fr = contact_prob[:, self.foot_name_to_index["FR"]]
        rl = contact_prob[:, self.foot_name_to_index["RL"]]
        rr = contact_prob[:, self.foot_name_to_index["RR"]]
        
        # 1. 对角线同步奖励：FL 和 RR 应该状态一致，FR 和 RL 应该状态一致
        diag1_sync = 1.0 - torch.abs(fl - rr)
        diag2_sync = 1.0 - torch.abs(fr - rl)
        diag_sync = 0.5 * (diag1_sync + diag2_sync)
        
        # 2. 计算每组对角线的平均触地情况
        s1 = 0.5 * (fl + rr) # 1号对角线 (FL+RR)
        s2 = 0.5 * (fr + rl) # 2号对角线 (FR+RL)
        
        # 3. 与目标相位匹配
        stance_mask = self._get_gait_phase().float()
        target_s1, target_s2 = stance_mask[:, 0], stance_mask[:, 1]
        match_s1 = 1.0 - torch.abs(s1 - target_s1)
        match_s2 = 1.0 - torch.abs(s2 - target_s2)
        phase_match = 0.5 * (match_s1 + match_s2)

        # 相位匹配略重于纯同步，避免学到“四脚一起跳”式同步
        rew = 0.4 * diag_sync + 0.6 * phase_match
        
        # 只有在有速度或转向指令时才给予奖励 (静止时不需要踏步)
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        rew = rew * move_cmd.float()
        
        # 仅在有移动指令时生效
        return rew
    
    
    def _reward_foot_slip(self):
        """
        [脚底打滑惩罚]
        触地时如果脚有水平速度则惩罚
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm) * contact
        return torch.sum(rew, dim=1)

    def _reward_progress(self):
        """
        轻量的命令方向进展奖励。
        只鼓励沿当前平移指令方向的正向速度，避免台阶前“停住保平衡”。
        """
        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        move_cmd = cmd_norm > 0.1
        cmd_dir = cmd_xy / torch.clamp(cmd_norm.unsqueeze(1), min=1e-6)
        progress_speed = torch.sum(self.base_lin_vel[:, :2] * cmd_dir, dim=1)
        return torch.relu(progress_speed) * move_cmd.float()
    
    def _reward_hip_pos(self):
        """ 
        [髋关节限位惩罚]
        惩罚髋关节 (Hip/Abduction) 偏离默认角度的程度。
        防止机器人两腿张得太开 (劈叉) 或向内收得太多。
        """

        hip_indices = [0, 3, 6, 9]
        # 计算惩罚
        penalty = torch.sum(torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]), dim=1)

        # 获取速度指令
        vy = self.commands[:, 1]
        vw = self.commands[:, 2]

        # 判断是否直行
        is_straight_command = (torch.abs(vy) < 0.1) & (torch.abs(vw) < 0.1)

        # 增加分部缩放因子
        scale = torch.where(is_straight_command, 1.0, 0.2)

        return scale * penalty

    def _reward_all_joint_pos(self):
        """
        [所有关节限位惩罚]
        惩罚所有关节偏离默认角度的程度
        防止动作变形
        """

        return torch.sum(torch.square(self.dof_pos[:,:] - self.default_dof_pos[:,:]), dim=1)
    
    def _reward_feet_spacing(self):
        # 1. 获取脚部世界坐标 (Global Position)
        # shape: (num_envs, 4, 13) -> 只取位置 (num_envs, 4, 3)
        feet_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        
        # 2. 获取基座世界坐标和姿态 (Base Position & Orientation)
        # self.root_states shape: (num_envs, 13)
        base_pos = self.root_states[:, 0:3]
        base_quat = self.root_states[:, 3:7]
        
        # 3. 坐标转换核心逻辑：世界系 -> 身体系
        # 3.1 平移：计算脚相对于基座的向量
        # 需要把 base_pos 维度扩充成 (num_envs, 1, 3) 以便和 feet_states (num_envs, 4, 3) 做减法
        feet_rel_pos_world = feet_states - base_pos.unsqueeze(1)
        
        # 3.2 旋转：将相对向量旋转回身体坐标系
        # 展平以便批量计算: (num_envs * 4, 3)
        flat_feet_rel_pos = feet_rel_pos_world.view(-1, 3)
        # 复制四份 quaternion 以对应四只脚: (num_envs * 4, 4)
        flat_base_quat = base_quat.unsqueeze(1).repeat(1, 4, 1).view(-1, 4)
        
        # 执行逆旋转
        flat_feet_local = quat_rotate_inverse(flat_base_quat, flat_feet_rel_pos)
        
        # 恢复形状: (num_envs, 4, 3)
        feet_local = flat_feet_local.view(self.num_envs, 4, 3)
        
        # 4. 现在取 Local Y
        current_feet_y = feet_local[:, :, 1]
        
        # 5. 定义安全阈值
        # 设置为 0.13，即两脚间距大于 26cm 时不惩罚
        min_safety_width = 0.13
        max_limit_width = 0.20 # (可选)防止劈叉

        # 6. 计算惩罚

        # A. 惩罚过窄
        too_narrow = torch.relu(min_safety_width - torch.abs(current_feet_y))
        
        # B. 惩罚过宽
        too_wide = torch.relu(torch.abs(current_feet_y) - max_limit_width)

        # 组合惩罚
        total_error = torch.sum(too_narrow + too_wide, dim=1)

        return total_error

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        # 获取 Z 轴速度惩罚的原始值
        penalty = torch.square(self.base_lin_vel[:, 2])
    
        # 获取每个环境当前的地形等级 (Level)
        # 注意：self.terrain_levels 在 update_terrain_curriculum 中维护

        # 创建一个系数向量
        level_scale = torch.where(self.terrain_levels > 0, 0.1, 1.0)
    
        # 3. 返回动态惩罚
        return penalty * level_scale
    
    def _reward_foot_impact_vel(self):
        """
        只在首次触地时惩罚过大的向下落地速度。
        小的正常落地速度通过安全阈值过滤，避免把接触后的微小振动也算作冲击。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = contact & (~self.last_impact_contacts)
        self.last_impact_contacts = contact

        # 只关心向下速度；低于安全阈值的正常落地不惩罚
        impact_vel = torch.clamp(-self.feet_vel[:, :, 2] - 0.2, min=0.0)
        return torch.sum(torch.square(impact_vel) * first_contact.float(), dim=1)

    def _get_under_body_height_samples(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            num_envs = len(env_ids) if env_ids is not None else self.num_envs
            return torch.zeros(num_envs, self.num_base_height_points, device=self.device)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_base_height_points),
                self.base_height_points[env_ids],
            ) + self.root_states[env_ids, :3].unsqueeze(1)
            num_envs = len(env_ids)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_base_height_points),
                self.base_height_points,
            ) + self.root_states[:, :3].unsqueeze(1)
            num_envs = self.num_envs

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_variability(self):
        cfg = self.cfg.rewards.terrain_adaptive
        if not cfg.enabled:
            return torch.zeros(self.num_envs, device=self.device)

        under_body_heights = self._get_under_body_height_samples()
        terrain_variability = torch.std(under_body_heights, dim=1)
        return torch.clamp(terrain_variability, min=0.0, max=cfg.terrain_variability_clip)

    def _get_adaptive_decay_scale(self, cfg_node, terrain_variability):
        if not (self.cfg.rewards.terrain_adaptive.enabled and cfg_node.enabled):
            return torch.ones_like(terrain_variability)

        scale = torch.exp(-torch.square(terrain_variability) / cfg_node.sigma)
        return torch.clamp(scale, min=cfg_node.min_scale, max=cfg_node.max_scale)

    def _get_clearance_margin(self, terrain_variability):
        cfg = self.cfg.rewards.terrain_adaptive.foot_clearance
        if not (self.cfg.rewards.terrain_adaptive.enabled and cfg.enabled):
            return torch.zeros(self.num_envs, 1, device=self.device)

        extra_clearance = cfg.std_gain * terrain_variability.unsqueeze(1)
        return torch.clamp(extra_clearance, min=0.0, max=cfg.max_extra_clearance)

    # def _reward_foot_clearance(self):
    #     """
    #     机身坐标系下的最小安全抬脚高度惩罚。
    #     只在机器人有运动意图且脚处于离地/摆动状态时，惩罚抬脚不足；
    #     对高于目标的抬脚不做惩罚，给越障和恢复动作留出自由度。
    #     """
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)

    #     footpos_in_body_frame = torch.zeros(
    #         self.num_envs, len(self.feet_indices), 3, device=self.device
    #     )
    #     footvel_in_body_frame = torch.zeros(
    #         self.num_envs, len(self.feet_indices), 3, device=self.device
    #     )

    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
    #             self.base_quat, cur_footpos_translated[:, i, :]
    #         )
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(
    #             self.base_quat, cur_footvel_translated[:, i, :]
    #         )

    #     foot_lateral_vel = torch.sqrt(
    #         torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    #     )
    #     foot_height_body = footpos_in_body_frame[:, :, 2]

    #     # 只惩罚低于“最低安全高度”的摆动腿，允许更高抬脚来越障
    #     min_clearance_error = torch.relu(
    #         self.cfg.rewards.clearance_height_target - foot_height_body
    #     )

    #     # 用软接触概率区分支撑/摆动，减轻 PhysX 接触抖动带来的误判
    #     contact_force_z = self.contact_forces[:, self.feet_indices, 2]
    #     contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
    #     swing_weight = 1.0 - contact_prob

    #     # 静止时不强迫抬腿；转向时也保留 clearance 约束
    #     move_cmd = (
    #         (torch.norm(self.commands[:, :2], dim=1) > 0.1)
    #         | (torch.abs(self.commands[:, 2]) > 0.1)
    #     ).float().unsqueeze(1)

    #     penalty = torch.abs(min_clearance_error) * foot_lateral_vel * swing_weight * move_cmd
    #     return torch.sum(penalty, dim=1)

    def _reward_foot_clearance(self):
        """
        [地形自适应的相位抬腿惩罚]
        平地时约束脚高贴近期望摆动轨迹；地形起伏增大时，放宽高抬脚惩罚，
        允许机器人为了跨台阶/障碍而抬得更高。
        """
        # 1. 获取当前脚相对脚底局部地形的高度
        feet_height = self._get_feet_heights()

        # 2. 计算每个脚的相位
        phase = self._get_phase().unsqueeze(1)
        feet_phases = (phase + self.foot_phase_offsets.unsqueeze(0)) % 1.0

        # 3. 计算步态相位与移动掩码
        sin_val = torch.sin(2 * torch.pi * feet_phases)
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)

        # 4. 利用地形采样估计当前环境的起伏复杂度。
        # 台阶/障碍越明显，允许的额外抬脚空间越大。
        terrain_variability = self._get_terrain_variability()
        extra_clearance = self._get_clearance_margin(terrain_variability)
        clearance_cfg = self.cfg.rewards.terrain_adaptive.foot_clearance

        # 5. 分阶段计算惩罚
        stance_tolerance = 0.02 + clearance_cfg.stance_gain * extra_clearance
        stance_penalty = torch.relu(feet_height - stance_tolerance)

        swing_target = -sin_val * self.cfg.rewards.clearance_height_target
        swing_low_penalty = torch.relu(swing_target - feet_height)
        swing_high_penalty = torch.relu(feet_height - (swing_target + extra_clearance))
        swing_penalty = swing_low_penalty + clearance_cfg.swing_high_penalty_weight * swing_high_penalty

        # 6. 支撑相贴地，摆动相低抬脚严格罚，高抬脚在复杂地形时放宽
        error = torch.where(sin_val > 0, stance_penalty, swing_penalty)

        return torch.sum(error, dim=1) * move_cmd.float()

    def _reward_raibert(self):
        """
        [Raibert 落脚点奖励]
        将当前命令按可用指令范围归一化，再映射到有界的目标落脚点偏移。
        这样即使速度课程继续放大，奖励给出的目标点也不会无限前冲。
        """
        move_cmd = (
            (torch.norm(self.commands[:, :2], dim=1) > 0.1)
            | (torch.abs(self.commands[:, 2]) > 0.1)
        )
        if not torch.any(move_cmd).item():
            return torch.zeros(self.num_envs, device=self.device)

        cfg = self.cfg.rewards.raibert
        foot_pos_body, foot_vel_body = self._get_feet_state_in_body_frame()

        cmd_limits = self.commands.new_tensor([
            max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]), 1e-6),
            max(abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]), 1e-6),
        ]).view(1, 1, 2)
        cmd_xy_norm = torch.clamp(self.commands[:, :2].unsqueeze(1) / cmd_limits, min=-1.0, max=1.0)
        vel_error_norm = torch.clamp(
            (self.commands[:, :2] - self.base_lin_vel[:, :2]).unsqueeze(1) / cmd_limits,
            min=-1.0,
            max=1.0,
        )
        linear_drive = torch.clamp(cmd_xy_norm + cfg.vel_error_gain * vel_error_norm, min=-1.0, max=1.0)

        max_linear_offset = self.commands.new_tensor(
            [cfg.max_linear_offset_x, cfg.max_linear_offset_y]
        ).view(1, 1, 2)
        target_xy = self.nominal_foothold_xy.unsqueeze(0) + linear_drive * max_linear_offset

        yaw_limit = 2.0 if self.cfg.commands.heading_command else max(
            abs(self.command_ranges["ang_vel_yaw"][0]),
            abs(self.command_ranges["ang_vel_yaw"][1]),
            1e-6,
        )
        yaw_norm = torch.clamp(self.commands[:, 2].view(self.num_envs, 1, 1) / yaw_limit, min=-1.0, max=1.0)
        yaw_basis = torch.stack(
            (-self.nominal_foothold_xy[:, 1], self.nominal_foothold_xy[:, 0]),
            dim=1,
        )
        yaw_basis = yaw_basis / torch.clamp(torch.norm(yaw_basis, dim=1, keepdim=True), min=1e-6)
        target_xy = target_xy + cfg.max_yaw_offset * cfg.yaw_gain * yaw_norm * yaw_basis.unsqueeze(0)

        phase = self._get_phase().unsqueeze(1)
        feet_phases = (phase + self.foot_phase_offsets.unsqueeze(0)) % 1.0
        swing_progress = torch.clamp((feet_phases - 0.5) * 2.0, min=0.0, max=1.0)
        late_swing = torch.clamp(
            (swing_progress - cfg.late_swing_start) / max(1e-6, 1.0 - cfg.late_swing_start),
            min=0.0,
            max=1.0,
        )

        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        planning_weight = late_swing * (1.0 + cfg.touchdown_gain * contact_prob)

        xy_error = foot_pos_body[:, :, :2] - target_xy
        tracking_reward = torch.exp(
            -torch.sum(torch.square(xy_error), dim=2) / max(cfg.tracking_sigma ** 2, 1e-6)
        )

        target_dir = target_xy - foot_pos_body[:, :, :2]
        target_dir = target_dir / torch.clamp(torch.norm(target_dir, dim=2, keepdim=True), min=1e-6)
        approach_speed = torch.sum(foot_vel_body[:, :, :2] * target_dir, dim=2)
        approach_bonus = torch.clamp(approach_speed, min=0.0, max=cfg.max_approach_speed) / max(
            cfg.max_approach_speed, 1e-6
        )

        reward = planning_weight * (
            tracking_reward + cfg.approach_bonus * approach_bonus
        )
        return torch.sum(reward, dim=1) * move_cmd.float()

    def _reward_feet_air_time(self):
        """
        以目标步态周期为参考的腾空时间奖励。
        只在首次落地时结算，鼓励摆动腿完成完整的一步，但不鼓励无限制延长滞空时间。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self.feet_air_time += self.dt

        target_air_time = self.cfg.rewards.cycle_time * 0.5
        min_air_time = target_air_time * 0.5
        first_contact = (self.feet_air_time > min_air_time) * contact_filt

        air_time_error = self.feet_air_time - target_air_time
        rew_air_time = torch.exp(-torch.square(air_time_error) / 0.01) * first_contact

        move_cmd = (
            (torch.norm(self.commands[:, :2], dim=1) > 0.1)
            | (torch.abs(self.commands[:, 2]) > 0.1)
        )
        rew_air_time = torch.sum(rew_air_time, dim=1) * move_cmd.float()

        self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_orientation(self):
        """
        [修改版] 姿态惩罚
        利用特权观测的高度信息，在地形起伏大（如爬楼梯）时，
        降低对 Pitch (前后俯仰) 的惩罚，但保持 Roll (左右侧倾) 的严格惩罚。
        """
        # 1. 分离投影重力的分量
        # projected_gravity[:, 0] -> x轴分量，对应 Pitch (前后俯仰)
        # projected_gravity[:, 1] -> y轴分量，对应 Roll (左右侧倾)
        pitch_proj = self.projected_gravity[:, 0]
        roll_proj = self.projected_gravity[:, 1]

        # 2. 利用特权观测计算地形复杂度
        # self.measured_heights: [num_envs, num_height_points]
        # 计算采样点高度的标准差 (Standard Deviation)
        # 平地 std 接近 0；楼梯/斜坡 std 会显著增大
        terrain_variability = self._get_terrain_variability()
        pitch_scale = self._get_adaptive_decay_scale(
            self.cfg.rewards.terrain_adaptive.orientation,
            terrain_variability,
        )

        # 4. 组合奖励
        # Roll 惩罚 (roll_proj) 保持原样 (甚至可以加权)，因为爬楼梯也不应该侧倾
        # Pitch 惩罚 (pitch_proj) 乘以动态系数
        
        # 注意：这里返回的是正数的惩罚项（在 config 中 scale 是负数）
        penalty = torch.square(roll_proj) + torch.square(pitch_proj) * pitch_scale

        return penalty

    def _reward_action_rate(self):
        """地形复杂时适度放松一阶动作变化惩罚，给越障爆发留出空间。"""
        penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        terrain_variability = self._get_terrain_variability()
        scale = self._get_adaptive_decay_scale(
            self.cfg.rewards.terrain_adaptive.action_rate,
            terrain_variability,
        )
        return penalty * scale

    def _reward_smoothness(self):
        """地形复杂时保留结构性平滑约束，但允许二阶动作变化更灵活。"""
        penalty = torch.sum(
            torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions),
            dim=1,
        )
        terrain_variability = self._get_terrain_variability()
        scale = self._get_adaptive_decay_scale(
            self.cfg.rewards.terrain_adaptive.smoothness,
            terrain_variability,
        )
        return penalty * scale

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # 判定静止条件
        is_still = (torch.norm(self.commands[:, :2], dim=1) < 0.1)

        # 计算位置误差
        pos_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

        # 计算速度误差
        vel_error = torch.sum(torch.abs(self.dof_vel), dim=1)

        # 组合误差
        error = pos_error + 0.05 * vel_error
    
        return error * is_still
