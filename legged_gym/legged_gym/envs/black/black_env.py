from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
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
            # 不再硬编码 3，而是从 config 读取最大滞后步数
            # lag_timesteps 在 config 中定义 (例如 3)
            max_lag = self.cfg.domain_rand.lag_timesteps 

            # 生成随机索引
            latency_indices = torch.randint(0, max_lag, (self.num_envs, ), device=self.device)
            
            # 防止索引越界 (因为 buffer 长度是在 _init_buffers 定义的)
            # 确保 latency_indices 不超过 action_queue 的实际长度
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


    def post_physics_step(self):
        """ 物理步后刷新状态 """
        env_ids, termination_privileged_obs = super().post_physics_step()
        # 手动刷新刚体状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return env_ids, termination_privileged_obs

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
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

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

    def compute_observations(self):
        """ Computes observations
        """

        # 计算相位信号
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()    # 理想的触地时序
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.    # 真实的触地状态

        # 构建基础观测向量
        base_obs = torch.cat((   
                                    self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ), dim=-1)

        # 添加噪声
        if self.add_noise:
            base_obs += (2 * torch.rand_like(base_obs) - 1) * self.noise_scale_vec[0:base_obs.shape[1]]

        # 拼接相位信号
        # 将相位拼接到最前面 (Sin, Cos, Commands, AngVel, ...)
        # 这样网络能最早“看到”周期信号
        current_obs = torch.cat((sin_pos, cos_pos, base_obs), dim=-1)

        # 添加感知输入 (高度图等)
        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        # 更新历史缓存 (滑动窗口)
        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], 
                                             stance_mask, # [2] 目标相位掩码
                                             contact_mask, # [4] 真实触地掩码
                                             self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), 

                                             dim=-1)
    
    def compute_termination_observations(self, env_ids):
        """ Computes observations for terminated environments (Critic needs this)
        """
        # 计算相位信号
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()    # 理想的触地时序
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.    # 真实的触地状态

        # 构建基础物理观测
        base_obs = torch.cat((      self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        # 添加噪声
        if self.add_noise:
            base_obs += (2 * torch.rand_like(base_obs) - 1) * self.noise_scale_vec[0:base_obs.shape[1]]

        # 拼接相位信号 + 基础观测
        # 顺序匹配 compute_observations: [Sin, Cos, Commands, ...]
        current_obs = torch.cat((sin_pos, cos_pos, base_obs), dim=-1)

        # 添加感知输入
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        
        # 添加高度图
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        # 返回特权观测
        # 这里只返回 termination_ids 对应的部分
        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], 
                          stance_mask, # [2] 目标相位掩码
                          contact_mask, # [4] 真实触地掩码
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
        rew = rew * (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        
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
    
    def _reward_hip_pos(self):
        """ 
        [髋关节限位惩罚]
        惩罚髋关节 (Hip/Abduction) 偏离默认角度的程度。
        防止机器人两腿张得太开 (劈叉) 或向内收得太多。
        """

        hip_indices = [0, 3, 6, 9]

        return torch.sum(torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]), dim=1)

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
        
        # 6. 组合惩罚
        # 根据相位选择使用哪一种惩罚
        error = torch.where(sin_val > 0, stance_penalty, swing_penalty)
        
        # 7. 求和并应用移动掩码
        return torch.sum(error, dim=1) * move_cmd.float()

