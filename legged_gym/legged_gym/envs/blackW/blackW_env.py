import numpy as np
import torch

from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.black.black_env import BlackEnv
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi


class BlackWEnv(BlackEnv):
    """blackW asset with Go2W-style command curriculum and reward terms."""

    def _init_buffers(self):
        super()._init_buffers()
        self._init_blackW_dof_indices()
        self._init_blackW_command_curriculum_buffers()
        self._init_blackW_obstacle_lift_buffers()

    def _init_blackW_wheel_randomization_buffers(self):
        self.wheel_vel_ref_scales = torch.ones(
            self.num_envs,
            len(self.wheel_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.wheel_vel_ref_biases = torch.zeros_like(self.wheel_vel_ref_scales)
        self.wheel_dof_vel_obs_biases = torch.zeros_like(self.wheel_vel_ref_scales)

    def _resample_wheel_randomization(self, env_ids):
        if len(env_ids) == 0 or not hasattr(self, "wheel_vel_ref_scales"):
            return
        num_wheels = len(self.wheel_indices)
        if getattr(self.cfg.domain_rand, "randomize_wheel_motor", False):
            scale_range = getattr(self.cfg.domain_rand, "wheel_vel_ref_scale_range", [1.0, 1.0])
            self.wheel_vel_ref_scales[env_ids] = torch_rand_float(
                scale_range[0], scale_range[1], (len(env_ids), num_wheels), device=self.device
            )
        else:
            self.wheel_vel_ref_scales[env_ids] = 1.0

        if getattr(self.cfg.domain_rand, "randomize_wheel_vel_ref_bias", False):
            bias_range = getattr(self.cfg.domain_rand, "wheel_vel_ref_bias_range", [0.0, 0.0])
            self.wheel_vel_ref_biases[env_ids] = torch_rand_float(
                bias_range[0], bias_range[1], (len(env_ids), num_wheels), device=self.device
            )
        else:
            self.wheel_vel_ref_biases[env_ids] = 0.0

        if getattr(self.cfg.domain_rand, "randomize_wheel_dof_vel_obs_bias", False):
            obs_bias_range = getattr(self.cfg.domain_rand, "wheel_dof_vel_obs_bias_range", [0.0, 0.0])
            self.wheel_dof_vel_obs_biases[env_ids] = torch_rand_float(
                obs_bias_range[0], obs_bias_range[1], (len(env_ids), num_wheels), device=self.device
            )
        else:
            self.wheel_dof_vel_obs_biases[env_ids] = 0.0

    def _init_blackW_obstacle_lift_buffers(self):
        self.wheel_obstacle_lift_timer = torch.zeros(
            self.num_envs,
            len(self.wheel_body_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.wheel_obstacle_lift_target_z = torch.zeros_like(self.wheel_obstacle_lift_timer)
        self.wheel_obstacle_lift_start_z = torch.zeros_like(self.wheel_obstacle_lift_timer)

        cfg = self.cfg.rewards.wheel_obstacle_lift
        forward_offsets = torch.tensor(cfg.forward_offsets, dtype=torch.float, device=self.device, requires_grad=False)
        lateral_offsets = torch.tensor(cfg.lateral_offsets, dtype=torch.float, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(forward_offsets, lateral_offsets)
        local_offsets = torch.zeros(
            1,
            len(self.wheel_body_indices),
            grid_x.numel(),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        local_offsets[:, :, :, 0] = grid_x.flatten()
        local_offsets[:, :, :, 1] = grid_y.flatten()
        self.wheel_obstacle_lift_local_offsets = local_offsets

    def _init_blackW_command_curriculum_buffers(self):
        self.cmd_curr_axis_buffers = {
            "x_low": [],
            "x_high": [],
            "y": [],
            "yaw": [],
        }
        self.last_cmd_curr_score = float("nan")
        self.last_cmd_curr_y_score = float("nan")
        if hasattr(self, "last_cmd_curr_yaw_score"):
            delattr(self, "last_cmd_curr_yaw_score")

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
        wheel_body_names = []
        for key in getattr(self.cfg.asset, "wheel_name", []):
            wheel_body_names.extend([name for name in self.body_names if key in name])
        if len(wheel_body_names) == 0:
            wheel_body_names = list(getattr(self, "feet_names", []))
        self.wheel_body_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name) for name in wheel_body_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        if len(self.wheel_indices) != 4:
            raise RuntimeError(f"Expected 4 wheel joints, got {len(self.wheel_indices)} from {self.dof_names}")
        if len(self.hip_indices) != 4:
            raise RuntimeError(f"Expected 4 hip joints, got {len(self.hip_indices)} from {self.dof_names}")
        if len(self.wheel_body_indices) != 4:
            raise RuntimeError(f"Expected 4 wheel bodies, got {len(self.wheel_body_indices)} from {self.body_names}")

        print("### blackW dof names:", self.dof_names)
        print("### blackW wheel indices:", self.wheel_indices.detach().cpu().tolist())
        print("### blackW wheel body indices:", self.wheel_body_indices.detach().cpu().tolist())
        print("### blackW hip indices:", self.hip_indices.detach().cpu().tolist())
        print("### blackW wheel forward sign:", self.wheel_forward_sign.detach().cpu().tolist())
        print("### blackW wheel control mode:", self.cfg.control.wheel_control_mode)

        self._init_blackW_wheel_randomization_buffers()
        self._resample_wheel_randomization(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, "wheel_obstacle_lift_timer"):
            self.wheel_obstacle_lift_timer[env_ids] = 0.0
            self.wheel_obstacle_lift_target_z[env_ids] = 0.0
            self.wheel_obstacle_lift_start_z[env_ids] = 0.0
        self._resample_wheel_randomization(env_ids)

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
            return (
                signed_wheel_actions
                * self.cfg.control.vel_scale
                * self.wheel_vel_ref_scales
                + self.wheel_vel_ref_biases
            )
        if mode == "residual":
            residual = signed_wheel_actions * self.cfg.control.wheel_residual_scale * self.wheel_vel_ref_scales
            return self._target_wheel_velocities() + residual + self.wheel_vel_ref_biases
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
        friction_torque = (
            self.cfg.control.motor_friction_coulomb
            * torch.tanh(self.cfg.control.motor_friction_velocity_scale * self.dof_vel)
            + self.cfg.control.motor_friction_viscous * self.dof_vel
        )
        torques = torques - friction_torque
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _dof_pos_error_obs(self):
        dof_pos_error = self.dof_pos - self.default_dof_pos
        dof_pos_error = dof_pos_error.clone()
        dof_pos_error[:, self.wheel_indices] = 0.0
        return dof_pos_error

    def _dof_vel_obs(self):
        dof_vel = self.dof_vel.clone()
        dof_vel[:, self.wheel_indices] += self.wheel_dof_vel_obs_biases
        return dof_vel

    def compute_observations(self):
        dof_pos_error = self._dof_pos_error_obs()
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_pos_error * self.obs_scales.dof_pos,
            self._dof_vel_obs() * self.obs_scales.dof_vel,
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
                self.root_states[:, 2].unsqueeze(1)
                - self.cfg.terrain.height_measurement_base_offset
                - self.measured_heights,
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
            self._dof_vel_obs() * self.obs_scales.dof_vel,
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
                self.root_states[:, 2].unsqueeze(1)
                - self.cfg.terrain.height_measurement_base_offset
                - self.measured_heights,
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

    def _get_terrain_type_ids(self, env_ids=None):
        num_envs = len(env_ids) if env_ids is not None else self.num_envs
        if not (self.cfg.terrain.curriculum and hasattr(self, "terrain_types")):
            return torch.full((num_envs,), -1, dtype=torch.long, device=self.device)

        proportions = np.cumsum(self.cfg.terrain.terrain_proportions)
        if len(proportions) == 0 or self.cfg.terrain.num_cols <= 0:
            return torch.full((num_envs,), -1, dtype=torch.long, device=self.device)

        terrain_types = self.terrain_types if env_ids is None else self.terrain_types[env_ids]
        terrain_choice = terrain_types.float() / self.cfg.terrain.num_cols + 0.001
        proportions_tensor = torch.tensor(proportions, dtype=torch.float, device=self.device)
        return torch.bucketize(terrain_choice, proportions_tensor)

    def _get_high_wall_env_mask(self, env_ids=None):
        return self._get_terrain_type_ids(env_ids) == 9

    def _get_difficult_command_env_mask(self, env_ids=None):
        num_envs = len(env_ids) if env_ids is not None else self.num_envs
        sampling_cfg = getattr(self.cfg.commands, "terrain_command_sampling", None)
        if sampling_cfg is None or not getattr(sampling_cfg, "enabled", False):
            return torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        terrain_type_ids = self._get_terrain_type_ids(env_ids)
        difficult_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        for terrain_type in getattr(sampling_cfg, "difficult_terrain_types", []):
            difficult_mask |= terrain_type_ids == int(terrain_type)
        return difficult_mask

    def _get_difficult_command_range(self, range_name):
        sampling_cfg = getattr(self.cfg.commands, "terrain_command_sampling", None)
        if sampling_cfg is None or not getattr(sampling_cfg, "enabled", False):
            return None
        value = getattr(sampling_cfg, range_name, None)
        if value is None or len(value) != 2:
            return None
        return float(value[0]), float(value[1])

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.cfg.commands.low_speed_x_range[0],
            self.cfg.commands.low_speed_x_range[1],
            (len(env_ids), 1),
            device=self.device,
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

        high_vel_env_ids = env_ids[env_ids < (self.num_envs * self.cfg.commands.high_vel_env_fraction)]
        self.commands[high_vel_env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(high_vel_env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[high_vel_env_ids, 1:2] *= (
            torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1)
            < self.cfg.commands.high_speed_lateral_disable_x_threshold
        ).unsqueeze(1)

        difficult_env_ids = env_ids[self._get_difficult_command_env_mask(env_ids)]
        if len(difficult_env_ids) > 0:
            y_range = self._get_difficult_command_range("difficult_lin_vel_y")
            if y_range is not None:
                self.commands[difficult_env_ids, 1] = torch_rand_float(
                    y_range[0], y_range[1], (len(difficult_env_ids), 1), device=self.device
                ).squeeze(1)
            if not self.cfg.commands.heading_command:
                yaw_range = self._get_difficult_command_range("difficult_ang_vel_yaw")
                if yaw_range is not None:
                    self.commands[difficult_env_ids, 2] = torch_rand_float(
                        yaw_range[0], yaw_range[1], (len(difficult_env_ids), 1), device=self.device
                    ).squeeze(1)

        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.xy_norm_stop_threshold
        ).unsqueeze(1)

    def _post_physics_step_callback(self):
        resample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                self.cfg.commands.heading_yaw_gain * wrap_to_pi(self.commands[:, 3] - heading),
                -self.cfg.commands.heading_yaw_clip,
                self.cfg.commands.heading_yaw_clip,
            )
            yaw_range = self._get_difficult_command_range("difficult_ang_vel_yaw")
            if yaw_range is not None:
                difficult_envs = self._get_difficult_command_env_mask()
                self.commands[difficult_envs, 2] = torch.clamp(
                    self.commands[difficult_envs, 2], yaw_range[0], yaw_range[1]
                )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (
            self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0
        ):
            self._disturbance_robots()

    def _append_command_curriculum_samples(self, buffer_name, reward_name, env_ids, command_axis):
        if len(env_ids) == 0 or reward_name not in self.episode_sums or reward_name not in self.reward_scales:
            return
        reward_scale = abs(self.reward_scales.get(reward_name, 0.0))
        if reward_scale <= 0.0:
            return

        score_env_ids = env_ids
        cmd_min = 0.05
        axis_mask = torch.abs(self.commands[score_env_ids, command_axis]) > cmd_min
        score_env_ids = score_env_ids[axis_mask]
        if len(score_env_ids) == 0:
            return

        episode_len = torch.clamp(self.episode_length_buf[score_env_ids].float(), min=1.0)
        ratios = self.episode_sums[reward_name][score_env_ids] / episode_len / reward_scale
        ratios = ratios.detach().cpu()
        finite_mask = torch.isfinite(ratios)
        if torch.any(finite_mask):
            self.cmd_curr_axis_buffers[buffer_name].append(ratios[finite_mask])

    def _command_curriculum_buffer_count(self, buffer_name):
        return sum(chunk.numel() for chunk in self.cmd_curr_axis_buffers[buffer_name])

    def _command_curriculum_buffer_mean(self, buffer_name):
        if self._command_curriculum_buffer_count(buffer_name) == 0:
            return None
        return torch.cat(self.cmd_curr_axis_buffers[buffer_name]).mean().item()

    def _clear_command_curriculum_buffer(self, buffer_name):
        self.cmd_curr_axis_buffers[buffer_name] = []

    def _expand_command_range(self, range_name, step, max_abs):
        self.command_ranges[range_name][0] = np.clip(
            self.command_ranges[range_name][0] - step,
            -max_abs,
            0.0,
        )
        self.command_ranges[range_name][1] = np.clip(
            self.command_ranges[range_name][1] + step,
            0.0,
            max_abs,
        )

    def update_command_curriculum(self, env_ids):
        if not hasattr(self, "cmd_curr_axis_buffers"):
            self._init_blackW_command_curriculum_buffers()
        if len(env_ids) == 0:
            return

        high_fraction = self.cfg.commands.high_vel_env_fraction
        high_cutoff = self.num_envs * high_fraction
        low_vel_env_ids = env_ids[env_ids >= high_cutoff]
        high_vel_env_ids = env_ids[env_ids < high_cutoff]
        simple_command_env_ids = env_ids[~self._get_difficult_command_env_mask(env_ids)]

        self._append_command_curriculum_samples("x_low", "tracking_lin_vel_x", low_vel_env_ids, 0)
        self._append_command_curriculum_samples("x_high", "tracking_lin_vel_x", high_vel_env_ids, 0)
        self._append_command_curriculum_samples("y", "tracking_lin_vel_y", simple_command_env_ids, 1)

        buffer_min = max(1, int(getattr(self.cfg.commands, "curriculum_buffer_min", 256)))
        x_high_min = max(4, int(round(buffer_min * high_fraction)))
        x_low_min = max(8, buffer_min - x_high_min)
        if (
            self._command_curriculum_buffer_count("x_low") >= x_low_min
            and self._command_curriculum_buffer_count("x_high") >= x_high_min
        ):
            x_ratio = min(
                self._command_curriculum_buffer_mean("x_low"),
                self._command_curriculum_buffer_mean("x_high"),
            )
            x_threshold = self.cfg.commands.x_curriculum_score_scale
            self.last_cmd_curr_score = x_ratio / max(x_threshold, 1e-6)
            if x_ratio > x_threshold:
                self._expand_command_range(
                    "lin_vel_x",
                    self.cfg.commands.x_curriculum_step,
                    self.cfg.commands.max_curriculum,
                )
            self._clear_command_curriculum_buffer("x_low")
            self._clear_command_curriculum_buffer("x_high")

        if self._command_curriculum_buffer_count("y") >= buffer_min:
            y_ratio = self._command_curriculum_buffer_mean("y")
            y_threshold = self.cfg.commands.y_curriculum_score_scale
            self.last_cmd_curr_y_score = y_ratio / max(y_threshold, 1e-6)
            if y_ratio > y_threshold:
                self._expand_command_range(
                    "lin_vel_y",
                    self.cfg.commands.y_curriculum_step,
                    self.cfg.commands.max_y_curriculum,
                )
            self._clear_command_curriculum_buffer("y")

        if not self.cfg.commands.heading_command:
            self._append_command_curriculum_samples("yaw", "tracking_ang_vel", simple_command_env_ids, 2)
            if self._command_curriculum_buffer_count("yaw") >= buffer_min:
                yaw_ratio = self._command_curriculum_buffer_mean("yaw")
                yaw_threshold = self.cfg.commands.yaw_curriculum_score_scale
                self.last_cmd_curr_yaw_score = yaw_ratio / max(yaw_threshold, 1e-6)
                if yaw_ratio > yaw_threshold:
                    self._expand_command_range(
                        "ang_vel_yaw",
                        self.cfg.commands.yaw_curriculum_step,
                        self.cfg.commands.max_yaw_curriculum,
                    )
                self._clear_command_curriculum_buffer("yaw")
        elif hasattr(self, "last_cmd_curr_yaw_score"):
            delattr(self, "last_cmd_curr_yaw_score")

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
            > self.cfg.rewards.termination_contact_force_threshold,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
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
        terrain_height = torch.mean(self._get_under_body_height_samples(), dim=1)
        high_wall_envs = self._get_high_wall_env_mask()
        terrain_height = torch.where(high_wall_envs, torch.zeros_like(terrain_height), terrain_height)
        base_height = self.root_states[:, 2] - terrain_height
        height_error = torch.abs(base_height - self.cfg.rewards.base_height_target)
        high_wall_base_height_scale = 0.25
        height_error = torch.where(high_wall_envs, height_error * high_wall_base_height_scale, height_error)
        return height_error

    def _reward_hip_default(self):
        return torch.sum(
            torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1,
        )

    def _reward_stand_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(dof_err), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < self.cfg.rewards.stand_still_cmd_threshold
        )

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0
            * (
                torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
                > self.cfg.rewards.collision_force_threshold
            ),
            dim=1,
        )

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > self.cfg.rewards.feet_stumble_ratio
            * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_wheel_obstacle_lift(self):
        cfg = self.cfg.rewards.wheel_obstacle_lift
        wheel_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wheel_body_indices, :]
        wheel_pos = wheel_states[:, :, :3]
        wheel_z = wheel_states[:, :, 2]
        wheel_contact_forces = self.contact_forces[:, self.wheel_body_indices, :]
        horizontal_force = torch.norm(wheel_contact_forces[:, :, :2], dim=2)
        contact_gate = horizontal_force > cfg.horizontal_force_threshold
        command_gate = torch.norm(self.commands[:, :2], dim=1, keepdim=True) > cfg.command_threshold
        obstacle_height = self._sample_wheel_front_obstacle_heights(wheel_pos)
        ground_height = self._sample_terrain_heights_at_points(wheel_pos[:, :, :2], reduce="min")
        obstacle_rel_height = obstacle_height - ground_height
        obstacle_gate = obstacle_rel_height > cfg.obstacle_height_threshold
        trigger = contact_gate & command_gate & obstacle_gate
        active = self.wheel_obstacle_lift_timer > 0.0
        new_trigger = trigger & ~active

        new_start = wheel_z.detach()
        new_target = torch.maximum(
            obstacle_height.detach() + cfg.clearance_margin,
            new_start + cfg.min_lift_height,
        )
        self.wheel_obstacle_lift_start_z = torch.where(new_trigger, new_start, self.wheel_obstacle_lift_start_z)
        self.wheel_obstacle_lift_target_z = torch.where(new_trigger, new_target, self.wheel_obstacle_lift_target_z)
        active_time = max(cfg.active_time, self.dt)
        self.wheel_obstacle_lift_timer = torch.where(
            new_trigger,
            torch.full_like(self.wheel_obstacle_lift_timer, active_time),
            torch.clamp(self.wheel_obstacle_lift_timer - self.dt, min=0.0),
        )

        active = self.wheel_obstacle_lift_timer > 0.0
        lift_span = torch.clamp(
            self.wheel_obstacle_lift_target_z - self.wheel_obstacle_lift_start_z,
            min=cfg.min_progress_span,
        )
        lift_progress = torch.clamp(
            (wheel_z - self.wheel_obstacle_lift_start_z) / lift_span,
            min=0.0,
            max=1.0,
        )
        height_error = torch.clamp(self.wheel_obstacle_lift_target_z - wheel_z, min=0.0)
        sigma = max(cfg.target_sigma, 1e-6)
        target_reward = torch.exp(-torch.square(height_error / sigma))
        lift_reward = cfg.progress_weight * lift_progress + (1.0 - cfg.progress_weight) * target_reward
        return torch.sum(lift_reward * active.float() * command_gate.float(), dim=1)

    def _sample_wheel_front_obstacle_heights(self, wheel_pos):
        local_offsets = self.wheel_obstacle_lift_local_offsets.expand(self.num_envs, -1, -1, -1)
        flat_offsets = local_offsets.reshape(self.num_envs, -1, 3)
        world_offsets = quat_apply_yaw(self.base_quat.repeat(1, flat_offsets.shape[1]), flat_offsets)
        sample_points = (
            wheel_pos[:, :, None, :2]
            + world_offsets.view(self.num_envs, len(self.wheel_body_indices), -1, 3)[:, :, :, :2]
        )
        heights = self._sample_terrain_heights_at_points(sample_points.view(self.num_envs, -1, 2), reduce="max")
        return torch.max(heights.view(self.num_envs, len(self.wheel_body_indices), -1), dim=2).values

    def _sample_terrain_heights_at_points(self, points_xy, reduce="min"):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(points_xy.shape[:2], dtype=torch.float, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = points_xy + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].reshape(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].reshape(-1), 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        if reduce == "max":
            heights = torch.max(torch.max(heights1, heights2), heights3)
        else:
            heights = torch.min(torch.min(heights1, heights2), heights3)
        return heights.view(points_xy.shape[:2]) * self.terrain.cfg.vertical_scale

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
        return torch.sum(torch.abs(dof_err), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) > self.cfg.rewards.run_still_cmd_threshold
        )

