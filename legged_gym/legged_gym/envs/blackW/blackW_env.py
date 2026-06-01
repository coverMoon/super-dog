import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

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
        self._init_blackW_wheel_domain_rand_buffers()
        self._init_blackW_command_curriculum()

    def _init_blackW_command_curriculum(self):
        self.cmd_curr_segment_len = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.cmd_curr_segment_sums = {
            "tracking_lin_vel": torch.zeros(self.num_envs, device=self.device, requires_grad=False),
            "tracking_lin_vel_y": torch.zeros(self.num_envs, device=self.device, requires_grad=False),
            "tracking_ang_vel": torch.zeros(self.num_envs, device=self.device, requires_grad=False),
        }

        self.cmd_curr_ema_low = 0.0
        self.cmd_curr_ema_high = 0.0
        self.cmd_curr_pass_streak = 0
        self.cmd_curr_buffer_cmd_x = []
        self.cmd_curr_buffer_ratio = []
        self.last_cmd_curr_low_count = 0
        self.last_cmd_curr_high_count = 0
        self.last_cmd_curr_eval_count = 0
        self.last_cmd_curr_progressed = 0.0
        self.last_cmd_curr_score = 0.0
        self.last_cmd_curr_threshold_ratio = float(getattr(self.cfg.commands, "curriculum_threshold", 0.8))

        for axis in ("y", "yaw"):
            setattr(self, f"cmd_curr_{axis}_ema_low", 0.0)
            setattr(self, f"cmd_curr_{axis}_ema_high", 0.0)
            setattr(self, f"cmd_curr_{axis}_pass_streak", 0)
            setattr(self, f"cmd_curr_{axis}_buffer_cmd", [])
            setattr(self, f"cmd_curr_{axis}_buffer_ratio", [])
            setattr(self, f"last_cmd_curr_{axis}_low_count", 0)
            setattr(self, f"last_cmd_curr_{axis}_high_count", 0)
            setattr(self, f"last_cmd_curr_{axis}_sample_count", 0)
            setattr(self, f"last_cmd_curr_{axis}_progressed", 0.0)
            setattr(self, f"last_cmd_curr_{axis}_score", 0.0)
            setattr(self, f"last_cmd_curr_{axis}_threshold_ratio", float("nan"))

    def _init_blackW_wheel_domain_rand_buffers(self):
        leg_lag = int(getattr(self.cfg.domain_rand, "lag_timesteps", 0))
        wheel_lag = int(getattr(self.cfg.domain_rand, "wheel_lag_timesteps", leg_lag))
        hist_len = max(leg_lag, wheel_lag) + 1
        if hasattr(self, "action_queue") and self.action_queue.size(1) < hist_len:
            self.action_queue = torch.zeros(
                self.num_envs, hist_len, self.num_actions, device=self.device, requires_grad=False
            )

        self.wheel_lag_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wheel_motor_strength_factors = torch.ones(
            self.num_envs, len(self.wheel_indices), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.wheel_vel_ref_scales = torch.ones(
            self.num_envs, len(self.wheel_indices), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.wheel_radius_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_base_half_width_scale = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_blackW_wheel_domain(torch.arange(self.num_envs, device=self.device))

    def _randomize_blackW_wheel_domain(self, env_ids):
        if len(env_ids) == 0:
            return

        if getattr(self.cfg.domain_rand, "randomize_wheel_delay", False):
            max_lag = int(getattr(self.cfg.domain_rand, "wheel_lag_timesteps", 0))
            self.wheel_lag_buffer[env_ids] = torch.randint(0, max_lag + 1, (len(env_ids),), device=self.device)
        else:
            self.wheel_lag_buffer[env_ids] = self.lag_buffer[env_ids]

        if getattr(self.cfg.domain_rand, "randomize_wheel_motor", False):
            strength_range = getattr(self.cfg.domain_rand, "wheel_motor_strength_range", [1.0, 1.0])
            vel_scale_range = getattr(self.cfg.domain_rand, "wheel_vel_ref_scale_range", [1.0, 1.0])
            self.wheel_motor_strength_factors[env_ids] = torch_rand_float(
                strength_range[0], strength_range[1], (len(env_ids), len(self.wheel_indices)), device=self.device
            )
            self.wheel_vel_ref_scales[env_ids] = torch_rand_float(
                vel_scale_range[0], vel_scale_range[1], (len(env_ids), len(self.wheel_indices)), device=self.device
            )
        else:
            self.wheel_motor_strength_factors[env_ids] = 1.0
            self.wheel_vel_ref_scales[env_ids] = 1.0

        if getattr(self.cfg.domain_rand, "randomize_wheel_geometry", False):
            radius_range = getattr(self.cfg.domain_rand, "wheel_radius_scale_range", [1.0, 1.0])
            width_range = getattr(self.cfg.domain_rand, "wheel_base_half_width_scale_range", [1.0, 1.0])
            self.wheel_radius_scale[env_ids] = torch_rand_float(
                radius_range[0], radius_range[1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.wheel_base_half_width_scale[env_ids] = torch_rand_float(
                width_range[0], width_range[1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        else:
            self.wheel_radius_scale[env_ids] = 1.0
            self.wheel_base_half_width_scale[env_ids] = 1.0

    def _wheel_body_prop_indices(self):
        body_names = getattr(self, "body_names", [])
        wheel_keys = getattr(self.cfg.asset, "wheel_name", [])
        return [i for i, name in enumerate(body_names) if any(key in name for key in wheel_keys)]

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if not getattr(self.cfg.domain_rand, "randomize_wheel_mass", False):
            return props

        mass_range = getattr(self.cfg.domain_rand, "wheel_mass_scale_range", [1.0, 1.0])
        inertia_range = getattr(self.cfg.domain_rand, "wheel_inertia_scale_range", [1.0, 1.0])
        for i in self._wheel_body_prop_indices():
            mass_scale = np.random.uniform(mass_range[0], mass_range[1])
            inertia_scale = np.random.uniform(inertia_range[0], inertia_range[1])
            props[i].mass *= mass_scale
            props[i].inertia.x.x *= inertia_scale
            props[i].inertia.y.y *= inertia_scale
            props[i].inertia.z.z *= inertia_scale
        return props

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self.actions

        if self.cfg.domain_rand.delay:
            latency_indices = torch.clip(self.lag_buffer, max=self.action_queue.size(1) - 1)
            delayed_actions = self.action_queue[torch.arange(self.num_envs, device=self.device), latency_indices]
        else:
            delayed_actions = self.actions.clone()

        if getattr(self.cfg.domain_rand, "randomize_wheel_delay", False):
            wheel_latency_indices = torch.clip(self.wheel_lag_buffer, max=self.action_queue.size(1) - 1)
            wheel_delayed_actions = self.action_queue[
                torch.arange(self.num_envs, device=self.device), wheel_latency_indices
            ]
            delayed_actions = delayed_actions.clone()
            delayed_actions[:, self.wheel_indices] = wheel_delayed_actions[:, self.wheel_indices]

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

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
        hip_index_by_leg = {name.split("_")[0]: self.dof_names.index(name) for name in hip_names}
        self.hip_indices_by_foot = torch.tensor(
            [hip_index_by_leg[name.split("_")[0]] for name in self.feet_names],
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

    def _post_physics_step_callback(self):
        resample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
        self._finalize_command_curriculum_segments(env_ids)
        super()._post_physics_step_callback()

    def compute_reward(self):
        tracked_names = [name for name in self.cmd_curr_segment_sums if name in self.episode_sums]
        before = {name: self.episode_sums[name].clone() for name in tracked_names}
        super().compute_reward()
        for name in tracked_names:
            self.cmd_curr_segment_sums[name] += self.episode_sums[name] - before[name]
        self.cmd_curr_segment_len += 1.0

    def reset_idx(self, env_ids):
        self._finalize_command_curriculum_segments(env_ids)
        preserve_failed_commands = getattr(self.cfg.commands, "preserve_failed_reset_commands", True)
        previous_commands = self.commands[env_ids].clone() if preserve_failed_commands and len(env_ids) > 0 else None
        failed_reset_mask = ~self.time_out_buf[env_ids] if preserve_failed_commands and len(env_ids) > 0 else None

        super().reset_idx(env_ids)

        if preserve_failed_commands and len(env_ids) > 0 and torch.any(failed_reset_mask):
            failed_env_ids = env_ids[failed_reset_mask]
            self.commands[failed_env_ids] = previous_commands[failed_reset_mask]

        self._randomize_blackW_wheel_domain(env_ids)

    def update_command_curriculum(self, env_ids):
        self._update_axis_command_curriculum(
            axis_name="x",
            command_name="lin_vel_x",
            threshold=getattr(self.cfg.commands, "curriculum_threshold", 0.7),
            step=0.1,
            target_abs=getattr(self.cfg.commands, "max_curriculum", 2.0),
            min_abs=getattr(self.cfg.commands, "min_nonzero_lin_cmd", 0.2),
            split_fraction=0.6,
        )
        self._update_axis_command_curriculum(
            axis_name="y",
            command_name="lin_vel_y",
            threshold=getattr(self.cfg.commands, "y_curriculum_threshold", 0.45),
            step=getattr(self.cfg.commands, "y_curriculum_step", 0.1),
            target_abs=getattr(self.cfg.commands, "max_curriculum_y", 1.0),
            min_abs=getattr(self.cfg.commands, "min_nonzero_lin_cmd", 0.2),
            split_fraction=0.5,
        )
        self._update_axis_command_curriculum(
            axis_name="yaw",
            command_name="ang_vel_yaw",
            threshold=getattr(self.cfg.commands, "yaw_curriculum_threshold", 0.4),
            step=getattr(self.cfg.commands, "yaw_curriculum_step", 0.2),
            target_abs=getattr(self.cfg.commands, "max_curriculum_yaw", 3.14),
            min_abs=getattr(self.cfg.commands, "min_nonzero_yaw_cmd", 0.2),
            split_fraction=0.5,
        )

    def _finalize_command_curriculum_segments(self, env_ids):
        if len(env_ids) == 0 or not hasattr(self, "cmd_curr_segment_len"):
            return

        valid = self.cmd_curr_segment_len[env_ids] > 0
        if not torch.any(valid):
            return

        ids = env_ids[valid]
        segment_len = self.cmd_curr_segment_len[ids]
        self._append_axis_command_curriculum_segments(
            axis_name="x",
            command_idx=0,
            reward_name="tracking_lin_vel",
            env_ids=ids,
            segment_len=segment_len,
            min_abs=getattr(self.cfg.commands, "min_nonzero_lin_cmd", 0.2),
        )
        self._append_axis_command_curriculum_segments(
            axis_name="y",
            command_idx=1,
            reward_name="tracking_lin_vel_y",
            env_ids=ids,
            segment_len=segment_len,
            min_abs=getattr(self.cfg.commands, "min_nonzero_lin_cmd", 0.2),
        )
        self._append_axis_command_curriculum_segments(
            axis_name="yaw",
            command_idx=2,
            reward_name="tracking_ang_vel",
            env_ids=ids,
            segment_len=segment_len,
            min_abs=getattr(self.cfg.commands, "min_nonzero_yaw_cmd", 0.2),
        )

        self.cmd_curr_segment_len[ids] = 0.0
        for segment_sum in self.cmd_curr_segment_sums.values():
            segment_sum[ids] = 0.0

    def _append_axis_command_curriculum_segments(self, axis_name, command_idx, reward_name, env_ids, segment_len, min_abs):
        reward_scale = abs(self.reward_scales.get(reward_name, 0.0))
        if reward_scale <= 0.0:
            return

        sample_cmd = torch.abs(self.commands[env_ids, command_idx]).detach().cpu()
        sample_ratio = (self.cmd_curr_segment_sums[reward_name][env_ids] / segment_len / reward_scale).detach().cpu()
        finite_mask = torch.isfinite(sample_ratio) & torch.isfinite(sample_cmd) & (sample_cmd >= min_abs)
        if not torch.any(finite_mask):
            return

        cmd_buffer, ratio_buffer = self._axis_curriculum_buffers(axis_name)
        cmd_buffer.append(sample_cmd[finite_mask])
        ratio_buffer.append(sample_ratio[finite_mask])

    def _update_axis_command_curriculum(self, axis_name, command_name, threshold, step, target_abs, min_abs, split_fraction):
        self._set_axis_curriculum_attr(axis_name, "progressed", 0.0)
        self._set_axis_curriculum_attr(axis_name, "threshold_ratio", float(threshold))
        self._set_axis_curriculum_attr(axis_name, "score", self._axis_curriculum_score(axis_name, threshold))

        cmd_buffer, ratio_buffer = self._axis_curriculum_buffers(axis_name)
        buffer_min = max(1, int(getattr(self.cfg.commands, "curriculum_buffer_min", 256)))
        buffer_count = sum(chunk.numel() for chunk in cmd_buffer)
        self._set_axis_curriculum_attr(axis_name, "sample_count", int(buffer_count))
        self._set_axis_curriculum_attr(axis_name, "low_count", 0)
        self._set_axis_curriculum_attr(axis_name, "high_count", 0)
        if buffer_count < buffer_min:
            return

        env_cmd = torch.cat(cmd_buffer)
        per_segment_ratio = torch.cat(ratio_buffer)
        current_max = max(abs(self.command_ranges[command_name][0]), abs(self.command_ranges[command_name][1]))
        command_floor = min(min_abs, current_max)
        split = command_floor + split_fraction * max(current_max - command_floor, 1e-6)
        low_mask = (env_cmd >= command_floor) & (env_cmd <= split)
        high_mask = env_cmd > split
        low_count = int(torch.sum(low_mask).item())
        high_count = int(torch.sum(high_mask).item())
        self._set_axis_curriculum_attr(axis_name, "low_count", low_count)
        self._set_axis_curriculum_attr(axis_name, "high_count", high_count)

        min_low_count = 8
        min_high_count = 4
        if low_count < min_low_count or high_count < min_high_count:
            self._set_axis_curriculum_attr(axis_name, "pass_streak", 0)
            return

        low_ratio = torch.mean(per_segment_ratio[low_mask])
        high_ratio = torch.mean(per_segment_ratio[high_mask])
        if not torch.isfinite(low_ratio) or not torch.isfinite(high_ratio):
            self._set_axis_curriculum_attr(axis_name, "pass_streak", 0)
            return

        ema_alpha = getattr(self.cfg.commands, "curriculum_ema_alpha", 0.1)
        ema_low = (1.0 - ema_alpha) * self._get_axis_curriculum_attr(axis_name, "ema_low") + ema_alpha * low_ratio.item()
        ema_high = (1.0 - ema_alpha) * self._get_axis_curriculum_attr(axis_name, "ema_high") + ema_alpha * high_ratio.item()
        self._set_axis_curriculum_attr(axis_name, "ema_low", ema_low)
        self._set_axis_curriculum_attr(axis_name, "ema_high", ema_high)
        self._set_axis_curriculum_attr(axis_name, "score", self._axis_curriculum_score(axis_name, threshold))

        high_threshold = max(0.0, threshold - 0.1)
        if ema_low > threshold and ema_high > high_threshold:
            self._set_axis_curriculum_attr(axis_name, "pass_streak", self._get_axis_curriculum_attr(axis_name, "pass_streak") + 1)
        else:
            self._set_axis_curriculum_attr(axis_name, "pass_streak", 0)

        required_passes = max(1, int(getattr(self.cfg.commands, "curriculum_required_passes", 1)))
        if self._get_axis_curriculum_attr(axis_name, "pass_streak") >= required_passes and current_max < target_abs:
            new_max = min(current_max + step, target_abs)
            self.command_ranges[command_name][0] = -new_max
            self.command_ranges[command_name][1] = new_max
            self._set_axis_curriculum_attr(axis_name, "progressed", 1.0)
            self._set_axis_curriculum_attr(axis_name, "pass_streak", 0)

        cmd_buffer.clear()
        ratio_buffer.clear()

    def _axis_curriculum_score(self, axis_name, threshold):
        high_threshold = max(0.0, threshold - 0.1)
        if threshold <= 0.0 or high_threshold <= 0.0:
            return float("nan")
        ema_low = self._get_axis_curriculum_attr(axis_name, "ema_low")
        ema_high = self._get_axis_curriculum_attr(axis_name, "ema_high")
        return min(ema_low / threshold, ema_high / high_threshold)

    def get_command_curriculum_state(self):
        return {
            "version": 1,
            "command_ranges": {
                "lin_vel_x": list(self.command_ranges["lin_vel_x"]),
                "lin_vel_y": list(self.command_ranges["lin_vel_y"]),
                "ang_vel_yaw": list(self.command_ranges["ang_vel_yaw"]),
            },
            "ema": {
                "x_low": self.cmd_curr_ema_low,
                "x_high": self.cmd_curr_ema_high,
                "y_low": self.cmd_curr_y_ema_low,
                "y_high": self.cmd_curr_y_ema_high,
                "yaw_low": self.cmd_curr_yaw_ema_low,
                "yaw_high": self.cmd_curr_yaw_ema_high,
            },
            "pass_streak": {
                "x": self.cmd_curr_pass_streak,
                "y": self.cmd_curr_y_pass_streak,
                "yaw": self.cmd_curr_yaw_pass_streak,
            },
        }

    def load_command_curriculum_state(self, state, mode="range"):
        if mode not in ("range", "full"):
            raise ValueError(f"Unknown command curriculum resume mode: {mode}")

        ranges = state.get("command_ranges", {})
        for command_name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
            if command_name in ranges:
                low, high = ranges[command_name]
                self.command_ranges[command_name][0] = float(low)
                self.command_ranges[command_name][1] = float(high)

        self._reset_command_curriculum_statistics()
        if mode == "full":
            ema = state.get("ema", {})
            self.cmd_curr_ema_low = float(ema.get("x_low", self.cmd_curr_ema_low))
            self.cmd_curr_ema_high = float(ema.get("x_high", self.cmd_curr_ema_high))
            self.cmd_curr_y_ema_low = float(ema.get("y_low", self.cmd_curr_y_ema_low))
            self.cmd_curr_y_ema_high = float(ema.get("y_high", self.cmd_curr_y_ema_high))
            self.cmd_curr_yaw_ema_low = float(ema.get("yaw_low", self.cmd_curr_yaw_ema_low))
            self.cmd_curr_yaw_ema_high = float(ema.get("yaw_high", self.cmd_curr_yaw_ema_high))

            pass_streak = state.get("pass_streak", {})
            self.cmd_curr_pass_streak = int(pass_streak.get("x", self.cmd_curr_pass_streak))
            self.cmd_curr_y_pass_streak = int(pass_streak.get("y", self.cmd_curr_y_pass_streak))
            self.cmd_curr_yaw_pass_streak = int(pass_streak.get("yaw", self.cmd_curr_yaw_pass_streak))
            self._refresh_command_curriculum_scores()

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _refresh_command_curriculum_scores(self):
        self.last_cmd_curr_score = self._axis_curriculum_score(
            "x", getattr(self.cfg.commands, "curriculum_threshold", 0.7)
        )
        self.last_cmd_curr_y_score = self._axis_curriculum_score(
            "y", getattr(self.cfg.commands, "y_curriculum_threshold", 0.45)
        )
        self.last_cmd_curr_yaw_score = self._axis_curriculum_score(
            "yaw", getattr(self.cfg.commands, "yaw_curriculum_threshold", 0.4)
        )

    def _reset_command_curriculum_statistics(self):
        self.cmd_curr_ema_low = 0.0
        self.cmd_curr_ema_high = 0.0
        self.cmd_curr_pass_streak = 0
        self.cmd_curr_buffer_cmd_x.clear()
        self.cmd_curr_buffer_ratio.clear()
        if hasattr(self, "cmd_curr_segment_len"):
            self.cmd_curr_segment_len.zero_()
            for segment_sum in self.cmd_curr_segment_sums.values():
                segment_sum.zero_()
        self.last_cmd_curr_low_count = 0
        self.last_cmd_curr_high_count = 0
        self.last_cmd_curr_eval_count = 0
        self.last_cmd_curr_progressed = 0.0
        self.last_cmd_curr_score = 0.0

        for axis in ("y", "yaw"):
            setattr(self, f"cmd_curr_{axis}_ema_low", 0.0)
            setattr(self, f"cmd_curr_{axis}_ema_high", 0.0)
            setattr(self, f"cmd_curr_{axis}_pass_streak", 0)
            getattr(self, f"cmd_curr_{axis}_buffer_cmd").clear()
            getattr(self, f"cmd_curr_{axis}_buffer_ratio").clear()
            setattr(self, f"last_cmd_curr_{axis}_low_count", 0)
            setattr(self, f"last_cmd_curr_{axis}_high_count", 0)
            setattr(self, f"last_cmd_curr_{axis}_sample_count", 0)
            setattr(self, f"last_cmd_curr_{axis}_progressed", 0.0)
            setattr(self, f"last_cmd_curr_{axis}_score", 0.0)

    def _axis_curriculum_buffers(self, axis_name):
        if axis_name == "x":
            return self.cmd_curr_buffer_cmd_x, self.cmd_curr_buffer_ratio
        return getattr(self, f"cmd_curr_{axis_name}_buffer_cmd"), getattr(self, f"cmd_curr_{axis_name}_buffer_ratio")

    def _get_axis_curriculum_attr(self, axis_name, name):
        if axis_name == "x":
            mapping = {
                "ema_low": "cmd_curr_ema_low",
                "ema_high": "cmd_curr_ema_high",
                "pass_streak": "cmd_curr_pass_streak",
            }
            return getattr(self, mapping[name])
        return getattr(self, f"cmd_curr_{axis_name}_{name}")

    def _set_axis_curriculum_attr(self, axis_name, name, value):
        if axis_name == "x":
            mapping = {
                "ema_low": "cmd_curr_ema_low",
                "ema_high": "cmd_curr_ema_high",
                "pass_streak": "cmd_curr_pass_streak",
                "low_count": "last_cmd_curr_low_count",
                "high_count": "last_cmd_curr_high_count",
                "sample_count": "last_cmd_curr_eval_count",
                "progressed": "last_cmd_curr_progressed",
                "score": "last_cmd_curr_score",
                "threshold_ratio": "last_cmd_curr_threshold_ratio",
            }
            setattr(self, mapping[name], value)
            return
        setattr(self, f"last_cmd_curr_{axis_name}_{name}" if name in ("low_count", "high_count", "sample_count", "progressed", "score", "threshold_ratio") else f"cmd_curr_{axis_name}_{name}", value)

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
    
    def _target_wheel_velocities(self, include_yaw=True):
        lin_x = self.commands[:, 0].unsqueeze(1)
        yaw = self.commands[:, 2].unsqueeze(1) if include_yaw else torch.zeros_like(lin_x)
        radius_scale = getattr(self, "wheel_radius_scale", self.commands.new_ones(self.num_envs)).unsqueeze(1)
        half_width_scale = getattr(self, "wheel_base_half_width_scale", self.commands.new_ones(self.num_envs)).unsqueeze(1)
        radius = torch.clamp(radius_scale * self.cfg.control.wheel_radius, min=1e-6)
        half_width = half_width_scale * self.cfg.control.wheel_base_half_width
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

        torques = torques * self.motor_strength_factors
        if hasattr(self, "wheel_motor_strength_factors"):
            torques[:, self.wheel_indices] *= self.wheel_motor_strength_factors

        friction_torque = 0.35 * torch.tanh(3.0 * self.dof_vel) + 0.1 * self.dof_vel
        torques = torques - friction_torque
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_wheel_vel_ref(self, actions):
        mode = getattr(self.cfg.control, "wheel_control_mode", "residual")
        wheel_actions = actions[:, self.wheel_indices]
        signed_wheel_actions = wheel_actions * self.wheel_forward_sign.unsqueeze(0)

        vel_ref_scale = getattr(self, "wheel_vel_ref_scales", torch.ones_like(signed_wheel_actions))
        if mode == "learned":
            return signed_wheel_actions * self.cfg.control.vel_scale * vel_ref_scale
        if mode == "residual":
            residual = signed_wheel_actions * self.cfg.control.wheel_residual_scale * vel_ref_scale
            return self._target_wheel_velocities() + residual
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

    def _yaw_activity(self):
        yaw_activity = self._command_activity(self.commands[:, 2])
        lin_cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        lin_scale = max(getattr(self.cfg.rewards, "yaw_dominance_lin_vel_scale", 1.0), 1e-6)
        yaw_abs = torch.abs(self.commands[:, 2])
        yaw_dominance = yaw_abs / (yaw_abs + lin_scale * lin_cmd_norm + 1e-6)
        return yaw_activity * yaw_dominance

    def _command_activity(self, command):
        deadzone = getattr(self.cfg.rewards, "command_activity_deadzone", 0.05)
        full = getattr(self.cfg.rewards, "command_activity_full", 0.2)
        span = max(full - deadzone, 1e-6)
        return torch.clamp((torch.abs(command) - deadzone) / span, min=0.0, max=1.0)

    def _axis_tracking_progress_reward(self, command, actual, min_ref):
        activity = self._command_activity(command)
        active = activity > 0.0
        if not torch.any(active).item():
            return torch.zeros_like(command)

        reward = torch.zeros_like(command)
        ref = torch.clamp(torch.abs(command[active]), min=max(min_ref, 1e-6))
        rel_error = (command[active] - actual[active]) / ref
        tracking_sigma = max(getattr(self.cfg.rewards, "relative_tracking_sigma", 0.25), 1e-6)
        tracking = torch.exp(-torch.square(rel_error) / tracking_sigma)

        signed_progress = torch.sign(command[active]) * actual[active]
        progress = torch.clamp(signed_progress / ref, min=0.0, max=1.0)

        tracking_weight = getattr(self.cfg.rewards, "tracking_reward_weight", 0.6)
        progress_weight = getattr(self.cfg.rewards, "progress_reward_weight", 0.4)
        reward[active] = activity[active] * (tracking_weight * tracking + progress_weight * progress)
        return reward

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

    def _reward_yaw_contact_hip_deviation(self):
        yaw = self._yaw_activity()
        if not torch.any(yaw > 0.0).item():
            return torch.zeros(self.num_envs, device=self.device)

        hip_error = torch.abs(
            self.dof_pos[:, self.hip_indices_by_foot]
            - self.default_dof_pos[:, self.hip_indices_by_foot]
        )
        margin = getattr(self.cfg.rewards, "yaw_hip_deviation_margin", 0.18)
        excess = torch.relu(hip_error - margin)

        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        return yaw * torch.sum(torch.abs(excess) * contact_prob, dim=1)

    def _reward_all_joint_pos(self):
        err = self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]
        penalty = torch.sum(torch.square(err), dim=1)
        scale = torch.where(self._gait_cmd_mask(), 0.01, 1.0)
        return penalty * scale

    def _reward_wheel_vel_ref_tracking(self):
        wheel_vel = self.dof_vel[:, self.wheel_indices]
        target = self._target_wheel_velocities(include_yaw=False)
        err = torch.sqrt(torch.mean(torch.square(wheel_vel - target), dim=1))
        ref = torch.clamp(
            torch.mean(torch.abs(target), dim=1),
            min=getattr(self.cfg.rewards, "wheel_tracking_min_ref", 0.5),
        )
        sigma = max(getattr(self.cfg.rewards, "wheel_tracking_relative_sigma", 0.25), 1e-6)
        tracking = torch.exp(-torch.square(err / ref) / sigma)
        move_cmd = self._command_activity(self.commands[:, 0])
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
            lin_weight * (1.0 - x_activity) * torch.abs(self.base_lin_vel[:, 0])
            + lin_weight * (1.0 - y_activity) * torch.abs(self.base_lin_vel[:, 1])
            + yaw_weight * (1.0 - yaw_activity) * torch.abs(self.base_ang_vel[:, 2])
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

    def _stand_command_mask(self):
        return (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (torch.abs(self.commands[:, 2]) < 0.1)

    def _stand_command_elapsed(self):
        resample_interval = max(int(self.cfg.commands.resampling_time / self.dt), 1)
        return (self.episode_length_buf % resample_interval).float() * self.dt

    def _reward_stand_alive(self):
        return self._stand_command_mask().float() * (~self.reset_buf).float()

    def _reward_stand_still(self):
        is_still = self._stand_command_mask()
        pos_error = torch.sum(
            torch.abs(self.dof_pos[:, self.leg_dof_indices] - self.default_dof_pos[:, self.leg_dof_indices]),
            dim=1,
        )
        leg_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.leg_dof_indices]), dim=1)
        return (pos_error + 0.1 * leg_vel_error) * is_still

    def _reward_stand_still_wheels(self):
        is_still = self._stand_command_mask()
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
