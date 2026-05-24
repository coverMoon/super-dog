import torch

from legged_gym.envs.black.black_env import BlackEnv


class BlackWEnv(BlackEnv):
    """Black wheel-leg environment.

    Policy/Isaac DOF order follows Isaac Gym's runtime `self.dof_names`.
    For the current blackW asset it is:
    FL hip/thigh/calf/wheel, FR hip/thigh/calf/wheel,
    RL hip/thigh/calf/wheel, RR hip/thigh/calf/wheel.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self._init_blackW_dof_indices()

    def _init_blackW_dof_indices(self):
        wheel_names = []
        for key in self.cfg.asset.wheel_name:
            wheel_names.extend([name for name in self.dof_names if key in name])

        hip_names = [name for name in self.dof_names if "hip_joint" in name]
        self.wheel_indices = torch.tensor(
            [self.dof_names.index(name) for name in wheel_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.hip_indices = torch.tensor(
            [self.dof_names.index(name) for name in hip_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        wheel_index_set = set(self.wheel_indices.detach().cpu().tolist())
        self.leg_dof_indices = torch.tensor(
            [i for i in range(self.num_dofs) if i not in wheel_index_set],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.wheel_forward_sign = torch.tensor(
            self._resolve_wheel_forward_sign(wheel_names),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.wheel_side_sign = torch.tensor(
            [1.0 if name.split("_")[0].endswith("L") else -1.0 for name in wheel_names],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        if len(self.wheel_indices) != 4:
            raise RuntimeError(f"Expected 4 wheel joints, got {len(self.wheel_indices)} from {self.dof_names}")
        if len(self.hip_indices) != 4:
            raise RuntimeError(f"Expected 4 hip joints, got {len(self.hip_indices)} from {self.dof_names}")
        if len(self.wheel_forward_sign) != 4:
            raise RuntimeError("cfg.control.wheel_forward_sign must contain 4 values")

        print("### blackW dof names:", self.dof_names)
        print("### blackW wheel indices:", self.wheel_indices.detach().cpu().tolist())
        print("### blackW hip indices:", self.hip_indices.detach().cpu().tolist())
        print("### blackW wheel forward sign:", self.wheel_forward_sign.detach().cpu().tolist())
        print("### blackW wheel control mode:", self.cfg.control.wheel_control_mode)

    def _sample_signed_command(self, env_ids, command_name, min_abs):
        low, high = self.command_ranges[command_name]
        max_abs = max(abs(low), abs(high))
        if max_abs <= min_abs:
            return torch.zeros(len(env_ids), device=self.device)

        magnitude = min_abs + (max_abs - min_abs) * torch.rand(len(env_ids), device=self.device)
        sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )
        return torch.clamp(sign * magnitude, min=low, max=high)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, :] = 0.0

        probs = self.commands.new_tensor([
            getattr(self.cfg.commands, "stand_command_prob", 0.15),
            getattr(self.cfg.commands, "x_command_prob", 0.25),
            getattr(self.cfg.commands, "y_command_prob", 0.25),
            getattr(self.cfg.commands, "yaw_command_prob", 0.25),
            getattr(self.cfg.commands, "mixed_command_prob", 0.10),
        ])
        probs = probs / torch.clamp(torch.sum(probs), min=1e-6)
        bins = torch.cumsum(probs, dim=0)
        sample = torch.rand(len(env_ids), device=self.device)

        stand_mask = sample < bins[0]
        x_mask = (sample >= bins[0]) & (sample < bins[1])
        y_mask = (sample >= bins[1]) & (sample < bins[2])
        yaw_mask = (sample >= bins[2]) & (sample < bins[3])
        mixed_mask = sample >= bins[3]

        min_lin = float(getattr(self.cfg.commands, "min_nonzero_lin_cmd", 0.2))
        min_yaw = float(getattr(self.cfg.commands, "min_nonzero_yaw_cmd", 0.2))

        x_ids = env_ids[x_mask]
        y_ids = env_ids[y_mask]
        yaw_ids = env_ids[yaw_mask]
        mixed_ids = env_ids[mixed_mask]

        if len(x_ids) > 0:
            self.commands[x_ids, 0] = self._sample_signed_command(x_ids, "lin_vel_x", min_lin)
        if len(y_ids) > 0:
            self.commands[y_ids, 1] = self._sample_signed_command(y_ids, "lin_vel_y", min_lin)
        if len(yaw_ids) > 0:
            self.commands[yaw_ids, 2] = self._sample_signed_command(yaw_ids, "ang_vel_yaw", min_yaw)
        if len(mixed_ids) > 0:
            self.commands[mixed_ids, 0] = self._sample_signed_command(mixed_ids, "lin_vel_x", min_lin)
            self.commands[mixed_ids, 1] = self._sample_signed_command(mixed_ids, "lin_vel_y", min_lin)
            self.commands[mixed_ids, 2] = self._sample_signed_command(mixed_ids, "ang_vel_yaw", min_yaw)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.rand(len(env_ids), device=self.device) * (
                self.command_ranges["heading"][1] - self.command_ranges["heading"][0]
            ) + self.command_ranges["heading"][0]
            self.commands[env_ids[stand_mask], 3] = 0.0

    def _resolve_wheel_forward_sign(self, wheel_names):
        cfg_sign = self.cfg.control.wheel_forward_sign
        if isinstance(cfg_sign, dict):
            signs = []
            missing = []
            for name in wheel_names:
                leg_prefix = name.split("_")[0]
                if leg_prefix not in cfg_sign:
                    missing.append(leg_prefix)
                    continue
                signs.append(float(cfg_sign[leg_prefix]))
            if missing:
                raise RuntimeError(
                    "cfg.control.wheel_forward_sign is missing leg prefixes: "
                    f"{sorted(set(missing))}"
                )
            return signs

        if len(cfg_sign) != len(wheel_names):
            raise RuntimeError(
                "cfg.control.wheel_forward_sign must contain one sign per runtime wheel DOF order"
            )
        return [float(sign) for sign in cfg_sign]
    
    def _target_wheel_velocities(self):
        lin_x = self.commands[:, 0].unsqueeze(1)
        yaw = self.commands[:, 2].unsqueeze(1)
        radius = max(self.cfg.control.wheel_radius, 1e-6)
        half_width = self.cfg.control.wheel_base_half_width
        target = (lin_x - yaw * self.wheel_side_sign.unsqueeze(0) * half_width) / radius
        return target * self.wheel_forward_sign.unsqueeze(0)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_indices] *= self.cfg.control.hip_reduction

        wheel_vel_ref = torch.zeros_like(actions)
        wheel_vel_ref[:, self.wheel_indices] = self._compute_wheel_vel_ref(actions)

        pos_err = self.default_dof_pos + actions_scaled - self.dof_pos
        pos_err[:, self.wheel_indices] = 0.0

        if self.cfg.control.control_type == "P":
            torques = self.p_gains * self.Kp_factors * pos_err - self.d_gains * self.Kd_factors * self.dof_vel
            torques[:, self.wheel_indices] += (
                self.d_gains[self.wheel_indices]
                * self.Kd_factors
                * wheel_vel_ref[:, self.wheel_indices]
            )
        elif self.cfg.control.control_type == "V":
            torques = self.p_gains * (wheel_vel_ref + actions_scaled - self.dof_vel)
            torques -= self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif self.cfg.control.control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {self.cfg.control.control_type}")

        friction_torque = 0.35 * torch.tanh(3.0 * self.dof_vel) + 0.1 * self.dof_vel
        torques = torques - friction_torque
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_wheel_vel_ref(self, actions):
        mode = getattr(self.cfg.control, "wheel_control_mode", "residual")
        wheel_actions = actions[:, self.wheel_indices]
        signed_wheel_actions = wheel_actions * self.wheel_forward_sign.unsqueeze(0)

        if mode == "learned":
            return signed_wheel_actions * self.cfg.control.vel_scale
        if mode == "residual":
            return self._target_wheel_velocities() + signed_wheel_actions * self.cfg.control.wheel_residual_scale
        raise NameError(f"Unknown wheel control mode: {mode}")

    def check_termination(self):
        super().check_termination()

        yaw_cmd = torch.abs(self.commands[:, 2]) > self.cfg.env.stuck_command_threshold
        yaw_progress = torch.sign(self.commands[:, 2]) * self.base_ang_vel[:, 2]
        yaw_stalled = yaw_progress < self.cfg.env.stuck_yaw_vel_threshold
        grace_done = (self.episode_length_buf.float() * self.dt) > self.cfg.env.stuck_grace_s
        stuck_mask = yaw_cmd & yaw_stalled & grace_done

        self.stuck_time = torch.where(stuck_mask, self.stuck_time + self.dt, self.stuck_time)
        self.reset_buf |= self.stuck_time > self.cfg.env.stuck_timeout_s

    def _gait_cmd_mask(self):
        return (
            (torch.abs(self.commands[:, 1]) > 0.1)
            | (torch.abs(self.commands[:, 2]) > 0.1)
        )

    def _command_activity(self, command):
        deadzone = getattr(self.cfg.rewards, "command_activity_deadzone", 0.05)
        full = getattr(self.cfg.rewards, "command_activity_full", 0.2)
        span = max(full - deadzone, 1e-6)
        return torch.clamp((torch.abs(command) - deadzone) / span, min=0.0, max=1.0)

    def _axis_tracking_progress_reward(self, command, actual, min_ref):
        activity = self._command_activity(command)
        ref = torch.clamp(torch.abs(command), min=min_ref)
        rel_error = (command - actual) / ref
        tracking_sigma = max(getattr(self.cfg.rewards, "relative_tracking_sigma", 0.25), 1e-6)
        tracking = torch.exp(-torch.square(rel_error) / tracking_sigma)

        signed_progress = torch.sign(command) * actual
        progress = torch.clamp(signed_progress / ref, min=0.0, max=1.0)

        tracking_weight = getattr(self.cfg.rewards, "tracking_reward_weight", 0.6)
        progress_weight = getattr(self.cfg.rewards, "progress_reward_weight", 0.4)
        return activity * (tracking_weight * tracking + progress_weight * progress)

    def _get_wheel_bottom_heights(self):
        # blackW's "foot" links are wheel bodies, so feet_pos is the wheel-center
        # height. Subtract the wheel radius to approximate the bottom/contact height.
        return self._get_feet_heights() - self.cfg.control.wheel_radius

    def _dof_pos_error_obs(self):
        dof_pos_error = self.dof_pos - self.default_dof_pos
        dof_pos_error = dof_pos_error.clone()
        dof_pos_error[:, self.wheel_indices] = 0.0
        return dof_pos_error

    def compute_observations(self):
        dof_pos_error = self._dof_pos_error_obs()
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_pos_error * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        current_obs = torch.cat((
            current_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.disturbance[:, 0, :],
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.,
            ) * self.obs_scales.height_measurements
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[
                (9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)
            ]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((
            current_obs[:, :self.num_one_step_obs],
            self.obs_buf[:, :-self.num_one_step_obs],
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            current_obs[:, :self.num_one_step_privileged_obs],
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs],
        ), dim=-1)

    def compute_termination_observations(self, env_ids):
        dof_pos_error = self._dof_pos_error_obs()
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_pos_error * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        current_obs = torch.cat((
            current_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.disturbance[:, 0, :],
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.,
            ) * self.obs_scales.height_measurements
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[
                (9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)
            ]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((
            current_obs[:, :self.num_one_step_privileged_obs],
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs],
        ), dim=-1)[env_ids]

# ==========================================================================
# Reward function components
# ========================================================================== 
    def _reward_hip_pos(self):
        penalty = torch.sum(
            torch.abs(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1,
        )
        is_straight_command = (torch.abs(self.commands[:, 1]) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        scale = torch.where(is_straight_command, 1.0, 0.2)
        return scale * penalty

    def _reward_all_joint_pos(self):
        err = self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]
        penalty = torch.sum(torch.square(err), dim=1)
        scale = torch.where(self._gait_cmd_mask(), 0.01, 1.0)
        return penalty * scale

    def _reward_wheel_vel_ref_tracking(self):
        wheel_vel = self.dof_vel[:, self.wheel_indices]
        target = self._target_wheel_velocities()
        err = torch.sqrt(torch.mean(torch.square(wheel_vel - target), dim=1))
        ref = torch.clamp(
            torch.mean(torch.abs(target), dim=1),
            min=getattr(self.cfg.rewards, "wheel_tracking_min_ref", 0.5),
        )
        sigma = max(getattr(self.cfg.rewards, "wheel_tracking_relative_sigma", 0.25), 1e-6)
        tracking = torch.exp(-torch.square(err / ref) / sigma)
        move_cmd = torch.maximum(self._command_activity(self.commands[:, 0]), self._command_activity(self.commands[:, 2]))
        return tracking * move_cmd

    def _reward_tracking_lin_vel(self):
        min_ref = getattr(self.cfg.rewards, "relative_tracking_min_lin_cmd", 0.2)
        return self._axis_tracking_progress_reward(self.commands[:, 0], self.base_lin_vel[:, 0], min_ref)

    def _reward_tracking_lin_vel_y(self):
        min_ref = getattr(self.cfg.rewards, "relative_tracking_min_lin_cmd", 0.2)
        return self._axis_tracking_progress_reward(self.commands[:, 1], self.base_lin_vel[:, 1], min_ref)

    def _reward_tracking_ang_vel(self):
        min_ref = getattr(self.cfg.rewards, "relative_tracking_min_yaw_cmd", 0.3)
        return self._axis_tracking_progress_reward(self.commands[:, 2], self.base_ang_vel[:, 2], min_ref)

    def _reward_inactive_axis_vel(self):
        x_activity = self._command_activity(self.commands[:, 0])
        y_activity = self._command_activity(self.commands[:, 1])
        yaw_activity = self._command_activity(self.commands[:, 2])
        lin_weight = getattr(self.cfg.rewards, "inactive_lin_vel_weight", 1.0)
        yaw_weight = getattr(self.cfg.rewards, "inactive_ang_vel_weight", 0.25)
        return (
            lin_weight * (1.0 - x_activity) * torch.square(self.base_lin_vel[:, 0])
            + lin_weight * (1.0 - y_activity) * torch.square(self.base_lin_vel[:, 1])
            + yaw_weight * (1.0 - yaw_activity) * torch.square(self.base_ang_vel[:, 2])
        )

    def _reward_trot(self):
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)

        fl = contact_prob[:, self.foot_name_to_index["FL"]]
        fr = contact_prob[:, self.foot_name_to_index["FR"]]
        rl = contact_prob[:, self.foot_name_to_index["RL"]]
        rr = contact_prob[:, self.foot_name_to_index["RR"]]

        diag1_sync = 1.0 - torch.abs(fl - rr)
        diag2_sync = 1.0 - torch.abs(fr - rl)

        s1 = 0.5 * (fl + rr)
        s2 = 0.5 * (fr + rl)

        stance_mask = self._get_gait_phase().float()
        target_s1, target_s2 = stance_mask[:, 0], stance_mask[:, 1]

        stance_score = target_s1 * s1 + target_s2 * s2
        swing_score = target_s1 * (1.0 - s2) + target_s2 * (1.0 - s1)
        sync_score = target_s1 * diag1_sync + target_s2 * diag2_sync

        rew = stance_score * swing_score * sync_score
        return rew * self._gait_cmd_mask().float()

    def _reward_foot_clearance(self):
        feet_height = self._get_wheel_bottom_heights()

        phase = self._get_phase().unsqueeze(1)
        feet_phases = (phase + self.foot_phase_offsets.unsqueeze(0)) % 1.0
        sin_val = torch.sin(2 * torch.pi * feet_phases)

        terrain_variability = self._get_terrain_variability()
        extra_clearance = self._get_clearance_margin(terrain_variability)
        clearance_cfg = self.cfg.rewards.terrain_adaptive.foot_clearance

        stance_tolerance = 0.02 + clearance_cfg.stance_gain * extra_clearance
        stance_penalty = torch.relu(feet_height - stance_tolerance)

        swing_target = -sin_val * self.cfg.rewards.clearance_height_target
        swing_low_penalty = torch.relu(swing_target - feet_height)
        swing_high_penalty = torch.relu(feet_height - (swing_target + extra_clearance))
        swing_penalty = swing_low_penalty + clearance_cfg.swing_high_penalty_weight * swing_high_penalty

        error = torch.where(sin_val > 0, stance_penalty, swing_penalty)
        return torch.sum(error, dim=1) * self._gait_cmd_mask().float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self.feet_air_time += self.dt

        target_air_time = self.cfg.rewards.cycle_time * 0.5
        min_air_time = target_air_time * 0.5
        first_contact = (self.feet_air_time > min_air_time) * contact_filt

        air_time_error = self.feet_air_time - target_air_time
        rew_air_time = torch.exp(-torch.square(air_time_error) / 0.01) * first_contact
        rew_air_time = torch.sum(rew_air_time, dim=1) * self._gait_cmd_mask().float()

        self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_foot_impact_vel(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = contact & (~self.last_impact_contacts)
        self.last_impact_contacts = contact

        impact_vel = torch.clamp(-self.feet_vel[:, :, 2] - 0.2, min=0.0)
        penalty = torch.sum(torch.square(impact_vel) * first_contact.float(), dim=1)
        return penalty * self._gait_cmd_mask().float()

    def _reward_foothold(self):
        foot_pos_body, _ = self._get_feet_state_in_body_frame()
        nominal_xy = self.nominal_foothold_xy.unsqueeze(0)
        xy_error = foot_pos_body[:, :, :2] - nominal_xy
        penalty = torch.sum(torch.abs(xy_error), dim=(1, 2))
        scale = torch.where(self._gait_cmd_mask(), 0.1, 1.0)
        return penalty * scale

    def _reward_raibert(self):
        move_cmd = self._gait_cmd_mask()
        if not torch.any(move_cmd).item():
            return torch.zeros(self.num_envs, device=self.device)

        cfg = self.cfg.rewards.raibert
        foot_pos_body, foot_vel_body = self._get_feet_state_in_body_frame()
        nominal_xy = self.nominal_foothold_xy.unsqueeze(0)

        y_limit = max(
            abs(self.command_ranges["lin_vel_y"][0]),
            abs(self.command_ranges["lin_vel_y"][1]),
            1e-6,
        )
        yaw_limit = max(
            abs(self.command_ranges["ang_vel_yaw"][0]),
            abs(self.command_ranges["ang_vel_yaw"][1]),
            1e-6,
        )

        cmd_y_norm = torch.clamp(self.commands[:, 1].view(self.num_envs, 1, 1) / y_limit, min=-1.0, max=1.0)
        vel_y_error_norm = torch.clamp(
            (self.commands[:, 1] - self.base_lin_vel[:, 1]).view(self.num_envs, 1, 1) / y_limit,
            min=-1.0,
            max=1.0,
        )
        lateral_drive = torch.clamp(cmd_y_norm + cfg.vel_error_gain * vel_y_error_norm, min=-1.0, max=1.0)
        lateral_offset = torch.cat(
            (
                torch.zeros_like(lateral_drive),
                lateral_drive * cfg.max_linear_offset_y,
            ),
            dim=2,
        )

        yaw_norm = torch.clamp(self.commands[:, 2].view(self.num_envs, 1, 1) / yaw_limit, min=-1.0, max=1.0)
        yaw_basis = torch.stack(
            (-self.nominal_foothold_xy[:, 1], self.nominal_foothold_xy[:, 0]),
            dim=1,
        )
        yaw_basis = yaw_basis / torch.clamp(torch.norm(yaw_basis, dim=1, keepdim=True), min=1e-6)
        yaw_offset = cfg.max_yaw_offset * cfg.yaw_gain * yaw_norm * yaw_basis.unsqueeze(0)

        target_xy = nominal_xy + lateral_offset + yaw_offset
        xy_error = foot_pos_body[:, :, :2] - target_xy
        tracking_reward = torch.exp(
            -torch.sum(torch.square(xy_error), dim=2) / max(cfg.tracking_sigma ** 2, 1e-6)
        )

        phase = self._get_phase().unsqueeze(1)
        feet_phases = (phase + self.foot_phase_offsets.unsqueeze(0)) % 1.0
        swing_progress = torch.clamp((feet_phases - 0.5) * 2.0, min=0.0, max=1.0)
        late_swing_start = getattr(cfg, "late_swing_start_latyaw", getattr(cfg, "late_swing_start", 0.15))
        late_swing = torch.clamp(
            (swing_progress - late_swing_start) / max(1e-6, 1.0 - late_swing_start),
            min=0.0,
            max=1.0,
        )

        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        swing_air = 1.0 - contact_prob
        planning_weight = late_swing * swing_air

        target_dir = target_xy - foot_pos_body[:, :, :2]
        target_dir = target_dir / torch.clamp(torch.norm(target_dir, dim=2, keepdim=True), min=1e-6)
        approach_speed = torch.sum(foot_vel_body[:, :, :2] * target_dir, dim=2)
        approach_bonus = torch.clamp(approach_speed, min=0.0, max=cfg.max_approach_speed) / max(
            cfg.max_approach_speed,
            1e-6,
        )

        reward = planning_weight * (tracking_reward + cfg.approach_bonus * approach_bonus)
        return torch.sum(reward, dim=1) * move_cmd.float()

    def _reward_stand_still(self):
        is_still = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        pos_error = torch.sum(
            torch.abs(self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]),
            dim=1,
        )
        leg_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.leg_dof_indices]), dim=1)
        # wheel_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.wheel_indices]), dim=1)
        return (pos_error + 0.1 * leg_vel_error) * is_still

    def _reward_stand_still_wheels(self):
        is_still = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        wheel_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.wheel_indices]), dim=1)
        return wheel_vel_error * is_still

    def _reward_dof_pos_limits(self):
        dof_pos = self.dof_pos[:, self.leg_dof_indices]
        limits = self.dof_pos_limits[self.leg_dof_indices]
        out_of_limits = -(dof_pos - limits[:, 0]).clip(max=0.)
        out_of_limits += (dof_pos - limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)

    def _reward_dof_acc(self):
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return torch.sum(torch.square(dof_acc[:, self.leg_dof_indices]), dim=1)

    def _reward_joint_power(self):
        leg_vel = torch.abs(self.dof_vel[:, self.leg_dof_indices])
        leg_torque = torch.abs(self.torques[:, self.leg_dof_indices])
        return torch.sum(leg_vel * leg_torque, dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques[:, self.leg_dof_indices]), dim=1)

    def _reward_dof_vel_limits(self):
        leg_vel = torch.abs(self.dof_vel[:, self.leg_dof_indices])
        leg_limits = self.dof_vel_limits[self.leg_dof_indices]
        penalty = leg_vel - leg_limits * self.cfg.rewards.soft_dof_vel_limit
        return torch.sum(penalty.clip(min=0.0, max=1.0), dim=1)

    def _reward_torque_limits(self):
        leg_torque = torch.abs(self.torques[:, self.leg_dof_indices])
        leg_limits = self.torque_limits[self.leg_dof_indices]
        penalty = leg_torque - leg_limits * self.cfg.rewards.soft_torque_limit
        return torch.sum(penalty.clip(min=0.0), dim=1)

    def _reward_action_rate(self):
        diff = self.last_actions[:, self.leg_dof_indices] - self.actions[:, self.leg_dof_indices]
        penalty = torch.sum(torch.square(diff), dim=1)
        terrain_variability = self._get_terrain_variability()
        scale = self._get_adaptive_decay_scale(
            self.cfg.rewards.terrain_adaptive.action_rate,
            terrain_variability,
        )
        return penalty * scale

    def _reward_smoothness(self):
        second_diff = (
            self.actions[:, self.leg_dof_indices]
            - 2.0 * self.last_actions[:, self.leg_dof_indices]
            + self.last_last_actions[:, self.leg_dof_indices]
        )
        penalty = torch.sum(torch.square(second_diff), dim=1)
        terrain_variability = self._get_terrain_variability()
        scale = self._get_adaptive_decay_scale(
            self.cfg.rewards.terrain_adaptive.smoothness,
            terrain_variability,
        )
        return penalty * scale

    def _reward_wheel_action_rate(self):
        diff = self.last_actions[:, self.wheel_indices] - self.actions[:, self.wheel_indices]
        return torch.sum(torch.square(diff), dim=1)

    def _reward_wheel_smoothness(self):
        second_diff = (
            self.actions[:, self.wheel_indices]
            - 2.0 * self.last_actions[:, self.wheel_indices]
            + self.last_last_actions[:, self.wheel_indices]
        )
        return torch.sum(torch.square(second_diff), dim=1)
