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
        target = (lin_x - yaw * half_width * self.wheel_side_sign.unsqueeze(0)) / radius
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

    def _raibert_target_xy(self):
        cfg = self.cfg.rewards.raibert
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

        yaw_limit = max(
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
        return target_xy + cfg.max_yaw_offset * cfg.yaw_gain * yaw_norm * yaw_basis.unsqueeze(0)

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

    def _reward_hip_pos(self):
        penalty = torch.sum(
            torch.abs(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1,
        )
        is_straight_command = (torch.abs(self.commands[:, 1]) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        scale = torch.where(is_straight_command, 1.0, 1.0)
        return scale * penalty

    def _reward_all_joint_pos(self):
        err = self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]
        return torch.sum(torch.square(err), dim=1) 

    def _reward_wheel_vel_ref_tracking(self):
        cfg = self.cfg.rewards.wheel_guidance
        wheel_vel = self.dof_vel[:, self.wheel_indices]
        target = self._target_wheel_velocities()
        err = wheel_vel - target

        vx_gate = torch.clamp(torch.abs(self.commands[:, 0]) / max(cfg.vx_gate_ref, 1e-6), 0.0, 1.0)
        yaw_gate = torch.clamp(torch.abs(self.commands[:, 2]) / max(cfg.yaw_gate_ref, 1e-6), 0.0, 1.0)
        active_gate = torch.maximum(vx_gate, yaw_gate)
        gate = cfg.min_gate + (1.0 - cfg.min_gate) * active_gate
        return torch.exp(-torch.mean(torch.square(err), dim=1) / max(cfg.sigma, 1e-6)) * gate

    def _reward_raibert_foothold(self):
        foot_pos_body, _ = self._get_feet_state_in_body_frame()
        target_xy = self._raibert_target_xy()
        xy_error_sq = torch.sum(torch.square(foot_pos_body[:, :, :2] - target_xy), dim=2)
        move_cmd = (
            (torch.abs(self.commands[:, 1]) > 0.1)
            | (torch.abs(self.commands[:, 2]) > 0.1)
        ).float()
        return torch.sum(xy_error_sq, dim=1) * move_cmd

    def _reward_stand_still(self):
        is_still = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)
        pos_error = torch.sum(
            torch.abs(self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]),
            dim=1,
        )
        leg_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.leg_dof_indices]), dim=1)
        wheel_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.wheel_indices]), dim=1)
        return (pos_error + 0.05 * leg_vel_error + 0.02 * wheel_vel_error) * is_still

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
