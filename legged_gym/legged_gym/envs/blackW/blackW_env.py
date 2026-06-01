import numpy as np
import torch

from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.black.black_env import BlackEnv
from legged_gym.utils.math import wrap_to_pi


class BlackWEnv(BlackEnv):
    """blackW asset with Go2W-style command curriculum and reward terms."""

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
            raise RuntimeError("cfg.control.wheel_forward_sign must contain one sign per runtime wheel DOF order")
        return [float(sign) for sign in cfg_sign]

    def _target_wheel_velocities(self):
        lin_x = self.commands[:, 0].unsqueeze(1)
        yaw = self.commands[:, 2].unsqueeze(1)
        radius = max(self.cfg.control.wheel_radius, 1e-6)
        half_width = self.cfg.control.wheel_base_half_width
        target = (lin_x - yaw * self.wheel_side_sign.unsqueeze(0) * half_width) / radius
        return target * self.wheel_forward_sign.unsqueeze(0)

    def _compute_wheel_vel_ref(self, actions):
        mode = getattr(self.cfg.control, "wheel_control_mode", "learned")
        wheel_actions = actions[:, self.wheel_indices]
        signed_wheel_actions = wheel_actions * self.wheel_forward_sign.unsqueeze(0)
        if mode == "learned":
            return signed_wheel_actions * self.cfg.control.vel_scale
        if mode == "residual":
            residual = signed_wheel_actions * self.cfg.control.wheel_residual_scale
            return self._target_wheel_velocities() + residual
        raise NameError(f"Unknown wheel control mode: {mode}")

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

        torques = torques * self.motor_strength_factors
        friction_torque = 0.35 * torch.tanh(3.0 * self.dof_vel) + 0.1 * self.dof_vel
        torques = torques - friction_torque
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

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

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        high_vel_env_ids = env_ids[env_ids < (self.num_envs * 0.2)]
        self.commands[high_vel_env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(high_vel_env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[high_vel_env_ids, 1:2] *= (
            torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0
        ).unsqueeze(1)
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def _post_physics_step_callback(self):
        resample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                -2.0,
                2.0,
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (
            self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0
        ):
            self._disturbance_robots()

    def update_command_curriculum(self, env_ids):
        low_vel_env_ids = env_ids[env_ids > (self.num_envs * 0.2)]
        high_vel_env_ids = env_ids[env_ids < (self.num_envs * 0.2)]
        if len(low_vel_env_ids) == 0 or len(high_vel_env_ids) == 0:
            return

        low_score = torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length
        high_score = torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length
        threshold = 0.8 * self.reward_scales["tracking_lin_vel"]
        if low_score > threshold and high_score > threshold:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.2,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.2,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_hip_default(self):
        return torch.sum(
            torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1,
        )

    def _reward_stand_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 3.0 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_vel(self):
        dof_vel = self.dof_vel.clone()
        dof_vel[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_run_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)


BlackWGo2WRewardEnv = BlackWEnv
