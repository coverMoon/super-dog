from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import torch

class BlackEnv(LeggedRobot):
    """
    自定义环境类 BlackEnv，适配 HIMLoco框架
    """

    def _init_buffers(self):
        """ 初始化 Buffer，额外获取所有刚体状态用于自定义奖励 """
        super()._init_buffers()

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

    def post_physics_step(self):
        """ 物理步后刷新状态 """
        env_ids, termination_privileged_obs = super().post_physics_step()
        # 手动刷新刚体状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return env_ids, termination_privileged_obs
    
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
        
        fl, fr, rl, rr = contact_prob[:, 0], contact_prob[:, 1], contact_prob[:, 2], contact_prob[:, 3]
        
        # 1. 对角线同步奖励：FL 和 RR 应该状态一致，FR 和 RL 应该状态一致
        diag1_sync = 1.0 - torch.abs(fl - rr)
        diag2_sync = 1.0 - torch.abs(fr - rl)
        
        # 2. 计算每组对角线的平均触地情况
        s1 = 0.5 * (fl + rr) # 1号对角线 (FL+RR)
        s2 = 0.5 * (fr + rl) # 2号对角线 (FR+RL)
        
        # 3. 互斥奖励：确保两组对角线不同时触地，也不同时抬起 (s1 + s2 应该接近 1)
        phase_score = 1.0 - torch.abs((s1 + s2) - 1.0)
        
        # 4. 与目标相位匹配
        stance_mask = self._get_gait_phase().float()
        target_s1, target_s2 = stance_mask[:, 0], stance_mask[:, 1]
        match_s1 = 1.0 - torch.abs(s1 - target_s1)
        match_s2 = 1.0 - torch.abs(s2 - target_s2)
        
        # 组合各项分数
        alpha, beta = 0.24, 0.5
        diag1_score = alpha * diag1_sync + (1 - alpha) * match_s1
        diag2_score = alpha * diag2_sync + (1 - alpha) * match_s2
        
        base_rew = 0.5 * (diag1_score + diag2_score)
        rew = base_rew * (beta * phase_score)
        
        # 只有在有速度指令时才给予奖励 (静止时不需要踏步)
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        rew = rew * move_cmd.float()
        
        # 仅在有移动指令时生效
        return rew
    
    def _reward_walk(self):
        """
        [Walk 慢走/四拍步态奖励]
        强制实现四只脚依次抬起的时序。
        时序：RL(0.0) -> FL(0.25) -> RR(0.5) -> FR(0.75)
        """
        # 1. 获取当前接触状态 (0~1)
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5) # [num_envs, 4]
        
        # 2. 获取基础相位 (0~1)
        phase = self._get_phase() # [num_envs]
        
        # 3. 定义四只脚的相位偏移 (Phase Offsets)
        # 索引: 0->FL, 1->FR, 2->RL, 3->RR
        # 对应时序: RL=0, FL=0.25, RR=0.5, FR=0.75
        # 注意：这里的偏移量决定了谁先抬腿
        offsets = torch.tensor([0.25, 0.75, 0.0, 0.5], device=self.device).unsqueeze(0)
        
        # 4. 计算每只脚的目标相位
        # 扩展 phase 维度以匹配 offsets: [num_envs, 4]
        feet_phases = (phase.unsqueeze(1) + offsets) % 1.0
        
        # 5. 生成目标触地信号 (Target Stance)
        # 使用 Sin 波生成目标。
        # Walk 步态的一个难点是 Duty Factor (触地占比)。
        # 标准 Sin 波 > 0 只有 50% 触地，这对 Walk 来说太快了（那是 Trot）。
        # Walk 通常需要 75% 的时间触地。
        # 这里我们修改阈值：Sin > -0.5 视为应当触地 (约 66%~75% 触地时间)
        
        sin_val = torch.sin(2 * torch.pi * feet_phases)
        target_stance = (sin_val > -0.5).float() 
        
        # 6. 计算匹配误差
        # 比较实际触地 (contact_prob) 和 目标触地 (target_stance)
        match_error = torch.square(contact_prob - target_stance)
        
        # 对四只脚的误差求平均
        rew = 1.0 - torch.mean(match_error, dim=1)
        
        # 7. 仅在移动时激活
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)
        
        return rew * move_cmd.float()
    
    def _reward_foot_slip(self):
        """
        [脚底打滑惩罚]
        触地时如果脚有水平速度则惩罚
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm) * contact
        return torch.sum(rew, dim=1)
    
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
        scale = torch.where(is_straight_command, 1.0, 1.0)

        return scale * penalty

    def _reward_all_joint_pos(self):
        """
        [所有关节限位惩罚]
        惩罚所有关节偏离默认角度的程度
        防止动作变形
        """

        return torch.sum(torch.abs(self.dof_pos[:,:] - self.default_dof_pos[:,:]), dim=1)
    
    def _reward_lateral_vel_penalty(self):
        """
        [横向速度惩罚]
        指令为前进时惩罚横向速度
        """
        v_y = self.base_lin_vel[:, 1]

        # 获取横向移动指令
        cmd_y = self.commands[:, 1]

        # 判断是否应该直行
        is_straight_command = torch.abs(cmd_y) < 0.1

        # 计算惩罚
        penalty = torch.square(v_y) * is_straight_command.float()

        return penalty
    
    def _reward_feet_stumble(self):
        # 1. 获取脚部的接触力 (contact_forces)
        contact_forces = self.contact_forces[:, self.feet_indices, :2]
        
        # 2. 计算水平力的模长 (Magnitude)
        # norm(dim=2) 计算 sqrt(x^2 + y^2)
        contact_norm = torch.norm(contact_forces, dim=-1)
        
        # 3. 判定逻辑
        stumble = contact_norm > 8.0
        
        # 4. 返回惩罚项
        # torch.any(dim=1) ：只要四只脚里有任意一只脚触发了阈值，这一帧就算 stumble
        # .float() 将布尔值转换为 0.0 或 1.0
        return torch.any(stumble, dim=1).float()
    
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
        # 判断哪些脚接触地面 (力 > 1N)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        
        # 获取脚的 Z 轴速度 (绝对值 或 平方)
        foot_vel_z = self.feet_vel[:, :, 2]
        
        # 惩罚：只有在接触地面时，脚的 Z 速度才会被惩罚
        # 使用平方形式惩罚大的冲击
        return torch.sum(contact * torch.square(foot_vel_z), dim=1)
    
    def _reward_foot_clearance_by_phase(self):
        """
        [基于相位的动态抬腿奖励]
        目标：支撑相时目标高度为0 (允许2cm误差)，摆动相时目标高度跟随半正弦波。
        解决定值高度导致的支撑相冲突问题。
        """
        # 1. 获取当前脚的高度 (相对于地面)
        feet_height = self._get_feet_heights() 

        # 2. 计算每个脚的相位
        # 基础相位 (num_envs, 1)
        phase = self._get_phase().unsqueeze(1)
        
        # 针对 Trot 步态设置相位偏移
        # 0(FL)和3(RR)是一组，1(FR)和2(RL)是一组，相差 0.5 周期
        offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0)
        
        # 得到每只脚的独立相位 [0, 1)
        feet_phases = (phase + offsets) % 1.0

        # 3. 计算正弦波值
        # sin > 0 为支撑相 (Stance)，sin < 0 为摆动相 (Swing)
        sin_val = torch.sin(2 * torch.pi * feet_phases)

        # 4. 移动指令掩码
        # 只有在有水平移动指令时才生效 (静止时不强迫抬腿)
        move_cmd = torch.norm(self.commands[:, :2], dim=1) > 0.1

        # 5. 分阶段计算惩罚
        
        # [A] 支撑相 (Stance): 
        # 目标是贴地。只要高度 < 0.02m (2cm) 就不惩罚。
        # 允许脚轻微陷入地面 (负值) 或离地很近，避免与物理引擎的接触解算打架。
        # 逻辑：只有当 feet_height > 0.02 时，(feet_height - 0.02) 才是正数，才会有平方惩罚。
        stance_penalty = torch.square(torch.clip(feet_height - 0.02, min=0.)) 
        
        # [B] 摆动相 (Swing): 
        # 严格追踪半正弦波轨迹。
        # sin_val 在此处为负数 (-1 ~ 0)，所以 -sin_val 为正数 (0 ~ 1)
        swing_target = -sin_val * self.cfg.rewards.clearance_height_target
        swing_penalty = torch.square(feet_height - swing_target)

        # 只惩罚低于半正弦波轨迹。允许为了避障而抬得更高。
        # swing_penalty = torch.square(torch.clip(feet_height - swing_target, max=0.0))
        
        # 6. 组合惩罚
        # 根据相位选择使用哪一种惩罚
        error = torch.where(sin_val > 0, stance_penalty, swing_penalty)
        
        # 7. 求和并应用移动掩码
        return torch.sum(error, dim=1) * move_cmd.float()

    # def _reward_foot_clearance_by_phase(self):
    #     """
    #     [基于相位的动态抬腿奖励 - 适配 Walk 步态]
    #     目标：
    #     1. 支撑相 (Stance): 目标高度为0 (允许微小误差)
    #     2. 摆动相 (Swing): 目标高度跟随轨迹
    #     注意：Walk 步态下，摆动相只占周期的 ~25%
    #     """
    #     # 1. 获取当前脚的高度 (相对于地面)
    #     feet_height = self._get_feet_heights() 

    #     # 2. 计算每个脚的相位
    #     phase = self._get_phase().unsqueeze(1)
        
    #     # [关键修改 1] 使用与 Walk 步态一致的 Offsets
    #     # 顺序: FL, FR, RL, RR -> 对应 RL, FL, RR, FR 抬起顺序
    #     offsets = torch.tensor([0.25, 0.75, 0.0, 0.5], device=self.device).unsqueeze(0)
        
    #     feet_phases = (phase + offsets) % 1.0

    #     # 3. 计算正弦波值
    #     sin_val = torch.sin(2 * torch.pi * feet_phases)

    #     # 4. 移动指令掩码 (静止时不强制)
    #     move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)

    #     # 5. 分阶段计算惩罚
        
    #     # [A] 支撑相 (Stance)
    #     # Walk 的支撑相很长，当 sin_val > -0.5 时都视为支撑相
    #     # 目标：贴地 (feet_height < 0.02)
    #     is_stance = sin_val > -0.5
    #     stance_penalty = torch.square(torch.clip(feet_height - 0.02, min=0.)) 
        
    #     # [B] 摆动相 (Swing)
    #     # 只有当 sin_val <= -0.5 时才视为摆动相 (处于正弦波底部)
    #     # 我们需要把 [-1.0, -0.5] 这个区间映射到高度曲线 [0, 1]
        
    #     # 归一化摆动进度：将 [-1, -0.5] 映射为 [0, 1] 的某种曲线用于计算目标高度
    #     # 简单做法：利用 sin_val 的绝对值。
    #     # 当 sin = -1 (波谷) 时，目标高度最大；当 sin = -0.5 时，目标高度接近 0
        
    #     # 计算目标高度曲线
    #     # (sin_val + 0.5) 在摆动期范围是 [-0.5, 0]
    #     # 乘以 -2 变为 [1.0, 0]
    #     swing_scale = (sin_val + 0.5) * -2.0 
    #     swing_scale = torch.clip(swing_scale, 0., 1.) # 截断保护
        
    #     target_height = swing_scale * self.cfg.rewards.clearance_height_target
        
    #     # 计算摆动惩罚：低于目标高度才惩罚
    #     swing_penalty = torch.square(torch.clip(feet_height - target_height, max=0.))
        
    #     # 6. 组合惩罚
    #     # 根据是否是支撑相选择惩罚项
    #     error = torch.where(is_stance, stance_penalty, swing_penalty)
        
    #     # 7. 求和并应用移动掩码
    #     return torch.sum(error, dim=1) * move_cmd.float()
    
    def _reward_roll_orientation(self):
        """ 惩罚 roll 轴倾角"""
        return torch.sum(torch.square(self.projected_gravity[:, 1:2]), dim=1)

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
        terrain_variability = torch.std(self.measured_heights, dim=1)

        # 3. 计算动态权重系数 (Scale Factor)
        # 地形越复杂 -> 系数越小 -> 惩罚越小
        # 使用高斯衰减函数：scale = exp(- (std^2) / sigma)
        # sigma: 敏感度参数。
        sigma = 0.03 
        pitch_scale = torch.exp(-torch.square(terrain_variability) / sigma)

        # 限制最小权重，防止机器人完全“躺平”或翻倒也不受惩罚
        pitch_scale = torch.clip(pitch_scale, min=0.1, max=1.0)

        # 4. 组合奖励
        # Roll 惩罚 (roll_proj) 保持原样 (甚至可以加权)，因为爬楼梯也不应该侧倾
        # Pitch 惩罚 (pitch_proj) 乘以动态系数
        
        # 注意：这里返回的是正数的惩罚项（在 config 中 scale 是负数）
        penalty = torch.square(roll_proj) + torch.square(pitch_proj) * pitch_scale

        return penalty

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # 判定静止条件
        is_still = torch.norm(self.commands[:, :2], dim=1) < 0.1

        # 计算位置误差
        pos_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

        # 计算速度误差
        vel_error = torch.sum(torch.abs(self.dof_vel), dim=1)

        # 组合误差
        error = pos_error + 0.01 * vel_error
    
        return error * is_still


    def _reward_raibert_heuristic(self):
        """
        [Raibert 式落脚位置奖励]
        鼓励脚部落在理想位置以稳定步态
        """
        
        # 获取支撑状态
        # stance_mask: [num_envs, 2]
        # col 0: 对应FL和RR腿，col 1: 对应FR和RL腿
        # 1 表示支撑相，0 表示摆动相
        stance_mask = self._get_gait_phase()

        # 推导四只脚的 is_swing（摆动为1，支撑为0）
        # 顺序: FL, FR, RL, RR
        swing_fl = 1.0 - stance_mask[:, 0]
        swing_fr = 1.0 - stance_mask[:, 1]
        swing_rl = 1.0 - stance_mask[:, 1]
        swing_rr = 1.0 - stance_mask[:, 0]

        # 组合为 [num_envs, 4] 
        is_swing = torch.stack((swing_fl, swing_fr, swing_rl, swing_rr), dim=1)

        # 定义名义站立位置
        nom_x = 0.2126
        nom_y = 0.1554

        # 构造四只脚相对于身体中心的默认站位
        nominal_stance = torch.tensor([
            [ nom_x,  nom_y],  # FL
            [ nom_x, -nom_y],  # FR
            [-nom_x,  nom_y],  # RL
            [-nom_x, -nom_y],  # RR
        ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)  # shape: (num_envs, 4, 2)

        # 计算 Raibert 目标位置
        # 标准公式：Target = Nominal + v * (T_stance / 2) + k * (v - v_cmd)

        cycle_time = self.cfg.rewards.cycle_time
        stance_time = cycle_time * 0.5  # 假设一半周期为支撑相

        # 获取基础速度信息
        # 线性速度 [num_envs, 2]
        v_lin_body = self.base_lin_vel[:, 0:2]
        v_lin_cmd = self.commands[:, 0:2]

        # 角速度
        omega_z = self.base_ang_vel[:, 2]  # shape: (num_envs,)
        omega_cmd = self.commands[:, 2]  # shape: (num_envs,)

        # 计算旋转引起的切向速度 (v = omega x r)
        # r 为脚相对于机身中心的向量
        r_x = nominal_stance[:, :, 0] # shape: (num_envs, 4)
        r_y = nominal_stance[:, :, 1] # shape: (num_envs, 4)
        
        # 实际角速度产生的切向速度 [num_envs, 4, 2]
        # v_x = -omega_z * r_y
        # v_y =  omega_z * r_x
        v_rot_x = -omega_z.unsqueeze(1) * r_y   # shape: (num_envs, 4)
        v_rot_y =  omega_z.unsqueeze(1) * r_x   # shape: (num_envs, 4)
        v_rot_body = torch.stack((v_rot_x, v_rot_y), dim=2)  # shape: (num_envs, 4, 2)

        # 指令角速度产生的切向速度 [num_envs, 4, 2]
        v_cmd_rot_x = -omega_cmd.unsqueeze(1) * r_y
        v_cmd_rot_y =  omega_cmd.unsqueeze(1) * r_x
        v_cmd_rot = torch.stack((v_cmd_rot_x, v_cmd_rot_y), dim=2)


        # 定义增益 
        k_lin = self.cfg.rewards.k_raibert_lin        # 原有的增益，用于平动
        k_rot = self.cfg.rewards.k_raibert_rot

        # 1. 线性部分的 Offset
        # v_lin_err = v_lin_body - v_lin_cmd
        # linear_offset = v_lin_body * (T/2) + k_lin * (err)
        v_lin_body_exp = v_lin_body.unsqueeze(1) # [num_envs, 1, 2]
        v_lin_cmd_exp  = v_lin_cmd.unsqueeze(1)
        
        offset_linear = v_lin_cmd_exp * (stance_time / 2.0) + k_lin * (v_lin_body_exp - v_lin_cmd_exp)

        # 2. 旋转部分的 Offset
        offset_rotation = v_cmd_rot * (stance_time / 2.0) + k_rot * (v_rot_body - v_cmd_rot)

        raibert_offset = offset_linear + 1.5 * offset_rotation  # shape: (num_envs, 4, 2)
        
        p_desired = nominal_stance + raibert_offset  # shape: (num_envs, 4, 2)
        
        # 获取实际脚部位置
        feet_pos_world = self.feet_pos  # [num_envs, 4, 3]
        base_pos = self.root_states[:, 0:3].unsqueeze(1)
        base_quat = self.root_states[:, 3:7]

        # 转换到 Base 坐标系
        feet_pos_rel = feet_pos_world - base_pos
        
        flat_base_quat = base_quat.unsqueeze(1).repeat(1, 4, 1).view(-1, 4)
        flat_feet_rel = feet_pos_rel.view(-1, 3)
        flat_feet_local = quat_rotate_inverse(flat_base_quat, flat_feet_rel)

        p_actual = flat_feet_local.view(self.num_envs, 4, 3)[:, :, 0:2]

        # 计算奖励
        # 计算欧氏距离平方
        error_vec = p_actual - p_desired
        error_sq = torch.sum(torch.square(error_vec), dim=2) # [num_envs, 4]

        # 转换为高斯奖励
        sigma = 0.2
        rew = torch.exp(-error_sq / sigma)

        # 掩码：只关心摆动腿 (Swing Legs)
        rew = rew * is_swing.float()

        # ===== 只在摆动相后半部分计算奖励 =====
        phase = self._get_phase().unsqueeze(1)  # shape: (num_envs, 1)
        offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0)
        feet_phases = (phase + offsets) % 1.0  # shape: (num_envs, 4)
        swing_phase_mask = (feet_phases > 0.9).float()
        rew = rew * swing_phase_mask

        # 平均化：除以摆动腿数量 (加小量防除零)
        num_swing_legs = torch.sum(is_swing, dim=1)
        final_rew = torch.sum(rew, dim=1) / (num_swing_legs + 1e-5)

        # 判断指令是否移动
        is_move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)

        # 仅在有移动指令时生效
        return final_rew * is_move_cmd.float()