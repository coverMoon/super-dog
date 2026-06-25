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
        self._init_blackW_action_scales()
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
        self.hip_motor_strength_factors = torch.ones(
            self.num_envs,
            len(self.hip_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.wheel_friction_scales = torch.ones(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.calf_backlash_widths = torch.zeros(
            self.num_envs,
            len(self.calf_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.calf_backlash_effective_targets = torch.zeros_like(self.calf_backlash_widths)
        self.calf_backlash_last_targets = torch.zeros_like(self.calf_backlash_widths)
        self.calf_backlash_remaining = torch.zeros_like(self.calf_backlash_widths)
        self.calf_backlash_dirs = torch.zeros_like(self.calf_backlash_widths)

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

        if getattr(self.cfg.domain_rand, "randomize_wheel_friction", False):
            friction_scale_range = getattr(self.cfg.domain_rand, "wheel_friction_scale_range", [1.0, 1.0])
            self.wheel_friction_scales[env_ids] = torch_rand_float(
                friction_scale_range[0], friction_scale_range[1], (len(env_ids), 1), device=self.device
            )
        else:
            self.wheel_friction_scales[env_ids] = 1.0

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

        if getattr(self.cfg.domain_rand, "randomize_hip_motor_strength", False):
            strength_range = getattr(self.cfg.domain_rand, "hip_motor_strength_range", [1.0, 1.0])
            self.hip_motor_strength_factors[env_ids] = torch_rand_float(
                strength_range[0], strength_range[1], (len(env_ids), len(self.hip_indices)), device=self.device
            )
        else:
            self.hip_motor_strength_factors[env_ids] = 1.0

        if getattr(self.cfg.domain_rand, "randomize_calf_backlash", False):
            backlash_range = getattr(self.cfg.domain_rand, "calf_backlash_range", [0.0, 0.0])
            self.calf_backlash_widths[env_ids] = torch_rand_float(
                backlash_range[0], backlash_range[1], (len(env_ids), len(self.calf_indices)), device=self.device
            )
        else:
            self.calf_backlash_widths[env_ids] = 0.0
        default_calf_targets = self.default_dof_pos[:, self.calf_indices].expand(len(env_ids), -1)
        self.calf_backlash_effective_targets[env_ids] = default_calf_targets
        self.calf_backlash_last_targets[env_ids] = default_calf_targets
        self.calf_backlash_remaining[env_ids] = 0.0
        self.calf_backlash_dirs[env_ids] = 0.0

    def _init_blackW_wheel_shape_indices(self):
        shape_ranges = self.gym.get_actor_rigid_body_shape_indices(self.envs[0], self.actor_handles[0])
        wheel_shape_indices = []
        for body_id in self.wheel_body_indices.detach().cpu().tolist():
            shape_range = shape_ranges[body_id]
            wheel_shape_indices.extend(range(shape_range.start, shape_range.start + shape_range.count))
        self.wheel_shape_indices = wheel_shape_indices
        print("### blackW wheel shape indices:", self.wheel_shape_indices)

    def _apply_wheel_friction_randomization(self, env_ids):
        if not getattr(self.cfg.domain_rand, "randomize_wheel_friction", False):
            return
        if not hasattr(self, "wheel_shape_indices") or not hasattr(self, "wheel_friction_scales"):
            return
        if len(env_ids) == 0:
            return
        env_id_list = env_ids.detach().cpu().tolist() if torch.is_tensor(env_ids) else list(env_ids)
        for env_id in env_id_list:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id])
            if hasattr(self, "friction_coeffs"):
                base_friction = float(self.friction_coeffs[env_id, 0].item())
            else:
                base_friction = float(self.cfg.terrain.static_friction)
            wheel_friction = base_friction * float(self.wheel_friction_scales[env_id, 0].item())
            for shape_id in self.wheel_shape_indices:
                rigid_shape_props[shape_id].friction = wheel_friction
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id], rigid_shape_props)

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
        self.wheel_obstacle_lift_elapsed = torch.zeros_like(self.wheel_obstacle_lift_timer)

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
        self.heading_command_env_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        if hasattr(self, "last_cmd_curr_yaw_score"):
            delattr(self, "last_cmd_curr_yaw_score")

    def _init_blackW_dof_indices(self):
        wheel_names = []
        for key in self.cfg.asset.wheel_name:
            wheel_names.extend([name for name in self.dof_names if key in name])

        hip_names = [name for name in self.dof_names if "hip_joint" in name]
        calf_names = [name for name in self.dof_names if "calf_joint" in name]
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
        self.calf_indices = torch.tensor(
            [self.dof_names.index(name) for name in calf_names],
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
        if len(self.calf_indices) != 4:
            raise RuntimeError(f"Expected 4 calf joints, got {len(self.calf_indices)} from {self.dof_names}")
        if len(self.wheel_body_indices) != 4:
            raise RuntimeError(f"Expected 4 wheel bodies, got {len(self.wheel_body_indices)} from {self.body_names}")

        wheel_prefix_to_lift_col = {name.split("_")[0]: i for i, name in enumerate(wheel_body_names)}
        try:
            self.front_wheel_lift_indices = torch.tensor(
                [wheel_prefix_to_lift_col["FL"], wheel_prefix_to_lift_col["FR"]],
                dtype=torch.long,
                device=self.device,
                requires_grad=False,
            )
            self.diag_rear_wheel_lift_indices = torch.tensor(
                [wheel_prefix_to_lift_col["RR"], wheel_prefix_to_lift_col["RL"]],
                dtype=torch.long,
                device=self.device,
                requires_grad=False,
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing expected wheel body prefix {exc} from {wheel_body_names}")

        print("### blackW dof names:", self.dof_names)
        print("### blackW wheel indices:", self.wheel_indices.detach().cpu().tolist())
        print("### blackW wheel body indices:", self.wheel_body_indices.detach().cpu().tolist())
        print("### blackW hip indices:", self.hip_indices.detach().cpu().tolist())
        print("### blackW calf indices:", self.calf_indices.detach().cpu().tolist())
        print("### blackW wheel forward sign:", self.wheel_forward_sign.detach().cpu().tolist())
        print("### blackW wheel control mode:", self.cfg.control.wheel_control_mode)

        self._init_blackW_wheel_shape_indices()
        self._init_blackW_wheel_randomization_buffers()
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_wheel_randomization(all_env_ids)
        self._apply_wheel_friction_randomization(all_env_ids)

    def _init_blackW_action_scales(self):
        cfg_scale = self.cfg.control.action_scale
        if isinstance(cfg_scale, dict):
            action_scale = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            wheel_index_set = set(self.wheel_indices.detach().cpu().tolist())
            for i, name in enumerate(self.dof_names):
                if i in wheel_index_set:
                    continue
                matched = False
                for dof_name, scale in cfg_scale.items():
                    if dof_name in name:
                        action_scale[i] = float(scale)
                        matched = True
                        break
                if not matched:
                    raise RuntimeError(f"Action scale of joint {name} was not defined")
        else:
            action_scale = torch.full(
                (self.num_dofs,),
                float(cfg_scale),
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        self.action_scale = action_scale.unsqueeze(0)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, "wheel_obstacle_lift_timer"):
            self.wheel_obstacle_lift_timer[env_ids] = 0.0
            self.wheel_obstacle_lift_target_z[env_ids] = 0.0
            self.wheel_obstacle_lift_start_z[env_ids] = 0.0
            self.wheel_obstacle_lift_elapsed[env_ids] = 0.0
        if len(env_ids) > 0 and hasattr(self, "last_impact_contacts"):
            self.last_impact_contacts[env_ids] = False
            self.prev_feet_vel[env_ids] = 0.0
        self._resample_wheel_randomization(env_ids)
        self._apply_wheel_friction_randomization(env_ids)

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id).copy()
        if getattr(self.cfg.domain_rand, "randomize_hip_damping", False):
            damping_range = getattr(self.cfg.domain_rand, "hip_damping_scale_range", [1.0, 1.0])
            hip_dof_ids = [i for i, name in enumerate(self.dof_names) if "hip_joint" in name]
            if len(hip_dof_ids) > 0:
                damping_scale = np.random.uniform(damping_range[0], damping_range[1], size=len(hip_dof_ids))
                props["damping"][hip_dof_ids] *= damping_scale
        return props

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

    def _apply_calf_backlash(self, calf_targets, calf_pos_err):
        mode = getattr(self.cfg.domain_rand, "calf_backlash_mode", "deadzone")
        if mode == "deadzone":
            return (
                torch.sign(calf_pos_err)
                * torch.clamp(torch.abs(calf_pos_err) - self.calf_backlash_widths, min=0.0)
            )

        if mode == "play":
            widths = torch.clamp(self.calf_backlash_widths, min=1e-6)
            leak = float(getattr(self.cfg.domain_rand, "calf_backlash_leak", 0.0))
            if leak > 0.0:
                self.calf_backlash_effective_targets += leak * (calf_targets - self.calf_backlash_effective_targets)
            lower = calf_targets - widths
            upper = calf_targets + widths
            self.calf_backlash_effective_targets = torch.max(
                torch.min(self.calf_backlash_effective_targets, upper), lower
            )
            gap_ratio = torch.clamp(
                torch.abs(calf_targets - self.calf_backlash_effective_targets) / widths, min=0.0, max=1.0
            )
            engage_start = float(getattr(self.cfg.domain_rand, "calf_backlash_engage_start", 0.6))
            engage_span = max(1.0 - engage_start, 1e-6)
            engage = torch.clamp((gap_ratio - engage_start) / engage_span, min=0.0, max=1.0)
            min_kp_scale = float(getattr(self.cfg.domain_rand, "calf_backlash_min_kp_scale", 0.15))
            kp_scale = min_kp_scale + (1.0 - min_kp_scale) * engage
            target_delta = calf_targets - self.calf_backlash_last_targets
            delta_sign = torch.sign(target_delta)
            self.calf_backlash_last_targets = calf_targets.clone()
            self.calf_backlash_remaining = torch.clamp(
                widths - torch.abs(calf_targets - self.calf_backlash_effective_targets), min=0.0
            )
            self.calf_backlash_dirs = torch.where(delta_sign != 0.0, delta_sign, self.calf_backlash_dirs)
            return (self.calf_backlash_effective_targets - self.dof_pos[:, self.calf_indices]) * kp_scale

        if mode != "hysteresis":
            raise NameError(f"Unknown calf backlash mode: {mode}")

        target_delta = calf_targets - self.calf_backlash_last_targets
        delta_sign = torch.sign(target_delta)
        moving = delta_sign != 0.0
        reversing = moving & (self.calf_backlash_dirs != 0.0) & (delta_sign != self.calf_backlash_dirs)
        remaining = torch.where(reversing, self.calf_backlash_widths, self.calf_backlash_remaining)
        abs_delta = torch.abs(target_delta)
        transmitted_delta = torch.sign(target_delta) * torch.clamp(abs_delta - remaining, min=0.0)

        self.calf_backlash_effective_targets += transmitted_delta
        self.calf_backlash_remaining = torch.clamp(remaining - abs_delta, min=0.0)
        self.calf_backlash_dirs = torch.where(moving, delta_sign, self.calf_backlash_dirs)
        self.calf_backlash_last_targets = calf_targets.clone()
        return self.calf_backlash_effective_targets - self.dof_pos[:, self.calf_indices]

    def _compute_torques(self, actions):
        actions_scaled = actions * self.action_scale
        actions_scaled[:, self.hip_indices] *= self.cfg.control.hip_reduction

        wheel_vel_ref = torch.zeros_like(actions)
        wheel_vel_ref[:, self.wheel_indices] = self._compute_wheel_vel_ref(actions)

        pos_targets = self.default_dof_pos + actions_scaled
        pos_err = pos_targets - self.dof_pos
        pos_err[:, self.wheel_indices] = 0.0
        if getattr(self.cfg.domain_rand, "randomize_calf_backlash", False):
            calf_targets = pos_targets[:, self.calf_indices]
            calf_pos_err = pos_err[:, self.calf_indices]
            pos_err[:, self.calf_indices] = self._apply_calf_backlash(calf_targets, calf_pos_err)

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
        if hasattr(self, "hip_motor_strength_factors"):
            torques[:, self.hip_indices] *= self.hip_motor_strength_factors
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

    def _get_heading_command_env_mask(self, env_ids=None):
        num_envs = len(env_ids) if env_ids is not None else self.num_envs
        if not self.cfg.commands.heading_command:
            return torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if not getattr(self.cfg.commands, "heading_command_difficult_only", False):
            return torch.ones(num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "heading_command_env_mask"):
            self.heading_command_env_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
            )
        return self.heading_command_env_mask if env_ids is None else self.heading_command_env_mask[env_ids]

    def _resample_heading_command_env_mask(self, env_ids):
        if len(env_ids) == 0:
            return
        if not hasattr(self, "heading_command_env_mask"):
            self.heading_command_env_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
            )
        if not self.cfg.commands.heading_command:
            self.heading_command_env_mask[env_ids] = False
            return
        if not getattr(self.cfg.commands, "heading_command_difficult_only", False):
            self.heading_command_env_mask[env_ids] = True
            return

        difficult_mask = self._get_difficult_command_env_mask(env_ids)
        heading_mask = difficult_mask.clone()
        simple_mask = ~difficult_mask
        simple_count = int(torch.sum(simple_mask).item())
        simple_prob = float(np.clip(getattr(self.cfg.commands, "simple_heading_command_prob", 0.0), 0.0, 1.0))
        if simple_count > 0 and simple_prob > 0.0:
            heading_mask[simple_mask] = torch.rand(simple_count, device=self.device) < simple_prob
        self.heading_command_env_mask[env_ids] = heading_mask

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
        self._resample_heading_command_env_mask(env_ids)
        heading_env_mask = self._get_heading_command_env_mask(env_ids)
        heading_env_ids = env_ids[heading_env_mask]
        yaw_env_ids = env_ids[~heading_env_mask]
        if len(heading_env_ids) > 0:
            self.commands[heading_env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(heading_env_ids), 1),
                device=self.device,
            ).squeeze(1)
        if len(yaw_env_ids) > 0:
            self.commands[yaw_env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(yaw_env_ids), 1),
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
            yaw_range = self._get_difficult_command_range("difficult_ang_vel_yaw")
            difficult_yaw_env_ids = difficult_env_ids[~self._get_heading_command_env_mask(difficult_env_ids)]
            if yaw_range is not None and len(difficult_yaw_env_ids) > 0:
                self.commands[difficult_yaw_env_ids, 2] = torch_rand_float(
                    yaw_range[0], yaw_range[1], (len(difficult_yaw_env_ids), 1), device=self.device
                ).squeeze(1)

        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.xy_norm_stop_threshold
        ).unsqueeze(1)

    def _post_physics_step_callback(self):
        resample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        heading_envs = self._get_heading_command_env_mask()
        if torch.any(heading_envs):
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[heading_envs, 2] = torch.clip(
                self.cfg.commands.heading_yaw_gain * wrap_to_pi(self.commands[heading_envs, 3] - heading[heading_envs]),
                -self.cfg.commands.heading_yaw_clip,
                self.cfg.commands.heading_yaw_clip,
            )
            yaw_range = self._get_difficult_command_range("difficult_ang_vel_yaw")
            if yaw_range is not None:
                difficult_heading_envs = heading_envs & self._get_difficult_command_env_mask()
                self.commands[difficult_heading_envs, 2] = torch.clamp(
                    self.commands[difficult_heading_envs, 2], yaw_range[0], yaw_range[1]
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

        yaw_curriculum_env_ids = simple_command_env_ids[~self._get_heading_command_env_mask(simple_command_env_ids)]
        self._append_command_curriculum_samples("yaw", "tracking_ang_vel", yaw_curriculum_env_ids, 2)
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

    def _get_orientation_terrain_variability(self):
        if not getattr(self.cfg.rewards, "orientation_terrain_adaptive", False):
            return torch.zeros(self.num_envs, device=self.device)
        under_body_heights = self._get_under_body_height_samples()
        terrain_variability = torch.std(under_body_heights, dim=1)
        clip = self.cfg.rewards.orientation_terrain_variability_clip
        return torch.clamp(terrain_variability, min=0.0, max=clip)

    def _get_orientation_pitch_scale(self):
        if not getattr(self.cfg.rewards, "orientation_terrain_adaptive", False):
            return torch.ones(self.num_envs, device=self.device)
        terrain_variability = self._get_orientation_terrain_variability()
        sigma = max(self.cfg.rewards.orientation_terrain_sigma, 1e-6)
        scale = torch.exp(-torch.square(terrain_variability) / sigma)
        return torch.clamp(
            scale,
            min=self.cfg.rewards.orientation_terrain_min_scale,
            max=self.cfg.rewards.orientation_terrain_max_scale,
        )

    def _reward_orientation(self):
        pitch_proj = self.projected_gravity[:, 0]
        roll_proj = self.projected_gravity[:, 1]
        pitch_scale = self._get_orientation_pitch_scale()
        return torch.abs(roll_proj) + torch.abs(pitch_proj) * pitch_scale

    def _reward_roll_orientation(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_base_height(self):
        terrain_height = torch.mean(self._get_under_body_height_samples(), dim=1)
        high_wall_envs = self._get_high_wall_env_mask()
        terrain_height = torch.where(high_wall_envs, torch.zeros_like(terrain_height), terrain_height)
        base_height = self.root_states[:, 2] - terrain_height
        height_error = torch.abs(base_height - self.cfg.rewards.base_height_target)
        high_wall_base_height_scale = 0.25
        height_error = torch.where(high_wall_envs, height_error * high_wall_base_height_scale, height_error)
        return height_error

    def _stand_lin_command_mask(self):
        return torch.norm(self.commands[:, :2], dim=1) < self.cfg.rewards.stand_still_cmd_threshold

    def _stand_wheel_command_mask(self):
        lin_stand = self._stand_lin_command_mask()
        yaw_stand = torch.abs(self.commands[:, 2]) < self.cfg.rewards.stand_still_yaw_threshold
        return lin_stand & yaw_stand

    def _reward_hip_default(self):
        hip_error = torch.sum(
            torch.abs(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]),
            dim=1,
        )
        y_ref = max(self.cfg.rewards.hip_default_y_ref, 1e-6)
        yaw_ref = max(self.cfg.rewards.hip_default_yaw_ref, 1e-6)
        y_cmd = torch.clamp(torch.abs(self.commands[:, 1]) / y_ref, max=1.0)
        yaw_cmd = torch.clamp(torch.abs(self.commands[:, 2]) / yaw_ref, max=1.0)
        scale = 1.0 - self.cfg.rewards.hip_default_y_scale * y_cmd
        scale = scale - self.cfg.rewards.hip_default_yaw_scale * yaw_cmd
        scale = torch.clamp(scale, min=self.cfg.rewards.hip_default_cmd_min_scale, max=1.0)
        return hip_error * scale

    def _reward_stand_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(dof_err), dim=1) * self._stand_lin_command_mask()

    def _reward_stand_wheel_action(self):
        wheel_action = self.actions[:, self.wheel_indices]
        return torch.sum(torch.square(wheel_action), dim=1) * self._stand_wheel_command_mask()

    def _reward_stand_wheel_vel(self):
        wheel_vel = self.dof_vel[:, self.wheel_indices]
        return torch.sum(torch.square(wheel_vel), dim=1) * self._stand_wheel_command_mask()

    def _reward_dof_pos_limits(self):
        dof_pos = self.dof_pos[:, self.leg_dof_indices]
        limits = self.dof_pos_limits[self.leg_dof_indices]
        margin_ratio = max(float(getattr(self.cfg.rewards, "dof_pos_limit_margin_ratio", 0.0)), 0.0)
        if margin_ratio <= 0.0:
            out_of_limits = -(dof_pos - limits[:, 0]).clip(max=0.0)
            out_of_limits += (dof_pos - limits[:, 1]).clip(min=0.0)
            return torch.sum(out_of_limits, dim=1)

        dof_range = torch.clamp(limits[:, 1] - limits[:, 0], min=1e-6)
        margin = dof_range * margin_ratio
        lower_dist = dof_pos - limits[:, 0]
        upper_dist = limits[:, 1] - dof_pos
        dist_to_limit = torch.minimum(lower_dist, upper_dist)
        proximity = torch.clamp((margin - dist_to_limit) / margin, min=0.0)
        return torch.sum(torch.square(proximity), dim=1)

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

    def _reward_foot_impact_vel(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.foot_impact_contact_force
        first_contact = contact & (~self.last_impact_contacts)
        self.last_impact_contacts[:] = contact
        impact_vel = torch.clamp(
            -self.prev_feet_vel[:, :, 2] - self.cfg.rewards.foot_impact_vel_threshold,
            min=0.0,
        )
        return torch.sum(torch.square(impact_vel) * first_contact.float(), dim=1)

    def _reward_wheel_lateral_clearance(self):
        cfg = self.cfg.rewards.wheel_lateral_clearance
        wheel_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wheel_body_indices, :]
        wheel_pos = wheel_states[:, :, :3]
        ground_height = self._sample_terrain_heights_at_points(wheel_pos[:, :, :2], reduce="min")
        target_z = ground_height + self.cfg.control.wheel_radius + cfg.target_extra_height
        height_error = torch.abs(target_z - wheel_pos[:, :, 2])
        sigma = max(cfg.tracking_sigma, 1e-6)
        wheel_clearance_reward = torch.exp(-torch.square(height_error / sigma))

        top_k = min(max(int(cfg.top_k), 1), wheel_clearance_reward.shape[1])
        top_reward = torch.topk(wheel_clearance_reward, k=top_k, dim=1).values.mean(dim=1)

        y_command = torch.abs(self.commands[:, 1])
        full_command_threshold = max(
            getattr(cfg, "full_command_threshold", cfg.command_threshold),
            cfg.command_threshold + 1e-6,
        )
        y_command_scale = torch.clamp(
            (y_command - cfg.command_threshold) / (full_command_threshold - cfg.command_threshold),
            min=0.0,
            max=1.0,
        )
        under_body_heights = self._get_under_body_height_samples()
        terrain_variability = torch.std(under_body_heights, dim=1)
        terrain_variability = torch.clamp(terrain_variability, min=0.0, max=cfg.terrain_variability_clip)
        terrain_sigma = max(cfg.terrain_variability_sigma, 1e-6)
        terrain_scale = torch.exp(-torch.square(terrain_variability) / terrain_sigma)
        terrain_scale = torch.clamp(terrain_scale, min=cfg.terrain_min_scale, max=cfg.terrain_max_scale)

        return top_reward * y_command_scale * terrain_scale

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
        terrain_type_ids = self._get_terrain_type_ids()
        if getattr(cfg, "stairs_use_simple_lift", False):
            stairs_simple_lift = torch.unsqueeze(terrain_type_ids == 3, dim=1)
        else:
            stairs_simple_lift = torch.zeros_like(trigger, dtype=torch.bool)
        low_obstacle_target = obstacle_height.detach() + cfg.clearance_margin
        high_obstacle_target = (
            obstacle_height.detach()
            + self.cfg.control.wheel_radius
            + cfg.high_obstacle_clearance_margin
        )
        target_by_height = torch.where(
            obstacle_rel_height > cfg.high_obstacle_height_threshold,
            high_obstacle_target,
            low_obstacle_target,
        )
        target_by_height = torch.where(stairs_simple_lift, low_obstacle_target, target_by_height)
        new_target = torch.maximum(
            target_by_height,
            new_start + cfg.min_lift_height,
        )
        self.wheel_obstacle_lift_start_z = torch.where(new_trigger, new_start, self.wheel_obstacle_lift_start_z)
        self.wheel_obstacle_lift_target_z = torch.where(new_trigger, new_target, self.wheel_obstacle_lift_target_z)
        high_obstacle_active_ratio = torch.clamp(
            (obstacle_rel_height - cfg.high_obstacle_height_threshold)
            / max(cfg.high_obstacle_active_height_span, 1e-6),
            min=0.0,
            max=1.0,
        )
        active_time = torch.clamp(
            torch.full_like(self.wheel_obstacle_lift_timer, cfg.active_time)
            + torch.where(
                stairs_simple_lift,
                torch.zeros_like(high_obstacle_active_ratio),
                high_obstacle_active_ratio * cfg.high_obstacle_extra_active_time,
            ),
            min=self.dt,
        )
        updated_timer = torch.where(
            new_trigger,
            active_time,
            torch.clamp(self.wheel_obstacle_lift_timer - self.dt, min=0.0),
        )
        self.wheel_obstacle_lift_elapsed = torch.where(
            new_trigger,
            torch.zeros_like(self.wheel_obstacle_lift_elapsed),
            torch.where(updated_timer > 0.0, self.wheel_obstacle_lift_elapsed + self.dt, torch.zeros_like(self.wheel_obstacle_lift_elapsed)),
        )
        self.wheel_obstacle_lift_timer = updated_timer

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
        height_error = torch.where(
            stairs_simple_lift,
            torch.clamp(self.wheel_obstacle_lift_target_z - wheel_z, min=0.0),
            torch.abs(self.wheel_obstacle_lift_target_z - wheel_z),
        )
        sigma = max(cfg.target_sigma, 1e-6)
        target_reward = torch.exp(-torch.square(height_error / sigma))
        over_lift = torch.clamp(
            wheel_z - self.wheel_obstacle_lift_target_z - cfg.over_lift_margin,
            min=0.0,
        )
        over_lift_sigma = max(cfg.over_lift_sigma, 1e-6)
        over_lift_penalty = cfg.over_lift_penalty_weight * torch.square(over_lift / over_lift_sigma)
        over_lift_penalty = torch.where(stairs_simple_lift, torch.zeros_like(over_lift_penalty), over_lift_penalty)
        lift_reward = (
            cfg.progress_weight * lift_progress
            + (1.0 - cfg.progress_weight) * target_reward
            - over_lift_penalty
        )
        # Down-stair terrain rewards controlled lowering rather than lifting; avoid rewarding turn-back/lift behavior.
        down_stairs_lift_scale = 0.0
        down_stairs_envs = self._get_terrain_type_ids() == 4
        terrain_scale = torch.where(
            down_stairs_envs,
            torch.full_like(self.wheel_obstacle_lift_timer[:, 0], down_stairs_lift_scale),
            torch.ones_like(self.wheel_obstacle_lift_timer[:, 0]),
        ).unsqueeze(1)
        return torch.sum(lift_reward * active.float() * command_gate.float() * terrain_scale, dim=1)


    def _reward_wheel_obstacle_rear_suppress(self):
        cfg = self.cfg.rewards.wheel_obstacle_lift
        high_wall_envs = self._get_high_wall_env_mask()
        if not torch.any(high_wall_envs):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        wheel_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wheel_body_indices, :]
        wheel_pos = wheel_states[:, :, :3]
        wheel_contact_forces = self.contact_forces[:, self.wheel_body_indices, :]
        horizontal_force = torch.norm(wheel_contact_forces[:, :, :2], dim=2)

        lift_inactive = self.wheel_obstacle_lift_timer <= 0.0
        no_horizontal_load = horizontal_force <= cfg.horizontal_force_threshold
        gate = lift_inactive & no_horizontal_load & high_wall_envs.unsqueeze(1)

        wheel_ground = self._sample_terrain_heights_at_points(wheel_pos[:, :, :2], reduce="min")
        allowed_z = wheel_ground + self.cfg.control.wheel_radius + cfg.diag_rear_lift_suppress_height
        excess = torch.clamp(wheel_pos[:, :, 2] - allowed_z, min=0.0)
        sigma = max(cfg.diag_rear_lift_suppress_sigma, 1e-6)
        return torch.sum(torch.square(excess / sigma) * gate.float(), dim=1)

    def _reward_wheel_obstacle_spin(self):
        cfg = self.cfg.rewards.wheel_obstacle_spin
        wheel_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.wheel_body_indices, :]
        wheel_pos = wheel_states[:, :, :3]
        wheel_contact_forces = self.contact_forces[:, self.wheel_body_indices, :]
        horizontal_force = torch.norm(wheel_contact_forces[:, :, :2], dim=2)
        contact_scale = torch.clamp(horizontal_force / max(cfg.horizontal_force_threshold, 1e-6), min=0.0, max=1.0)

        obstacle_height = self._sample_wheel_front_obstacle_heights(wheel_pos)
        ground_height = self._sample_terrain_heights_at_points(wheel_pos[:, :, :2], reduce="min")
        obstacle_rel_height = torch.clamp(obstacle_height - ground_height, min=0.0)
        obstacle_scale = torch.clamp(obstacle_rel_height / max(cfg.obstacle_height_threshold, 1e-6), min=0.0, max=1.0)

        command_x = torch.clamp(self.commands[:, 0:1], min=0.0)
        command_scale = torch.clamp(command_x / max(cfg.command_threshold, 1e-6), min=0.0, max=1.0)
        forward_speed = torch.clamp(self.base_lin_vel[:, 0:1], min=0.0)
        progress_deficit = torch.clamp(command_x - forward_speed, min=0.0)
        progress_scale = torch.clamp(progress_deficit / max(cfg.progress_speed_sigma, 1e-6), min=0.0, max=1.0)

        terrain_types = getattr(cfg, "terrain_types", [])
        if len(terrain_types) > 0:
            terrain_type_ids = self._get_terrain_type_ids()
            terrain_gate = torch.zeros_like(terrain_type_ids, dtype=torch.bool)
            for terrain_type in terrain_types:
                terrain_gate = terrain_gate | (terrain_type_ids == terrain_type)
        else:
            terrain_gate = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        wheel_surface_speed = torch.abs(self.dof_vel[:, self.wheel_indices]) * self.cfg.control.wheel_radius
        slip_speed = torch.clamp(wheel_surface_speed - forward_speed, min=0.0)
        slip_penalty = torch.square(slip_speed / max(cfg.slip_speed_sigma, 1e-6))
        penalty_scale = contact_scale * obstacle_scale * command_scale * progress_scale * terrain_gate.unsqueeze(1).float()
        continuous_penalty = slip_penalty * penalty_scale

        if getattr(cfg, "stairs_use_threshold_spin", False):
            contact_gate = horizontal_force > cfg.horizontal_force_threshold
            obstacle_gate = obstacle_rel_height > cfg.obstacle_height_threshold
            command_gate = self.commands[:, 0:1] > cfg.command_threshold
            progress_gate = self.base_lin_vel[:, 0:1] < cfg.progress_threshold
            wheel_spin = torch.clamp(torch.abs(self.dof_vel[:, self.wheel_indices]) - cfg.spin_threshold, min=0.0)
            threshold_penalty = torch.square(wheel_spin)
            threshold_gate = contact_gate & obstacle_gate & command_gate & progress_gate & terrain_gate.unsqueeze(1)
            stairs_gate = (terrain_type_ids == 3).unsqueeze(1)
            continuous_penalty = torch.where(
                stairs_gate,
                threshold_penalty * threshold_gate.float(),
                continuous_penalty,
            )

        return torch.sum(continuous_penalty, dim=1)

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
        x_run = torch.abs(self.commands[:, 0]) > self.cfg.rewards.run_still_x_threshold
        y_small = torch.abs(self.commands[:, 1]) < self.cfg.rewards.run_still_y_threshold
        yaw_small = torch.abs(self.commands[:, 2]) < self.cfg.rewards.run_still_yaw_threshold
        return torch.sum(torch.abs(dof_err), dim=1) * (x_run & y_small & yaw_small).float()

    def _reward_stairs_run_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0

        x_run = torch.abs(self.commands[:, 0]) > self.cfg.rewards.run_still_x_threshold
        y_small = torch.abs(self.commands[:, 1]) < self.cfg.rewards.run_still_y_threshold
        yaw_small = torch.abs(self.commands[:, 2]) < self.cfg.rewards.run_still_yaw_threshold

        terrain_type_ids = self._get_terrain_type_ids()
        stairs_gate = torch.zeros_like(terrain_type_ids, dtype=torch.bool)
        for terrain_type in self.cfg.rewards.stairs_run_still_terrain_types:
            stairs_gate |= terrain_type_ids == int(terrain_type)

        gate = x_run & y_small & yaw_small & stairs_gate
        return torch.sum(torch.abs(dof_err), dim=1) * gate.float()
