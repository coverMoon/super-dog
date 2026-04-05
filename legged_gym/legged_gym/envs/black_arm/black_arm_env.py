import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import get_axis_params, quat_apply, quat_rotate_inverse, to_torch, torch_rand_float

from legged_gym.envs.black.black_env import BlackEnv
from legged_gym.utils.math import wrap_to_pi


class BlackArmEnv(BlackEnv):
    """Black quadruped with a scripted arm trajectory and optional suction payload."""

    MODE_LOCOMOTION_ONLY = 0
    MODE_MANIP_STATIONARY = 1
    MODE_CARRY_MOVE = 2

    def _sample_arm_task_modes(self, env_ids):
        if len(env_ids) == 0:
            return
        if not self.arm_motion_enabled:
            self.arm_task_mode[env_ids] = self.MODE_LOCOMOTION_ONLY
            return
        self.arm_task_mode[env_ids] = torch.multinomial(self.arm_task_mode_probs, len(env_ids), replacement=True)

    def _sample_heading_targets(self, env_ids, offset_range):
        if len(env_ids) == 0:
            return
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        offsets = torch_rand_float(offset_range[0], offset_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = wrap_to_pi(heading + offsets)

    def _get_arm_motion_command_mask(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids
        mode = self.arm_task_mode[env_ids]
        mask = mode == self.MODE_MANIP_STATIONARY
        if not self.arm_motion_only_for_low_speed:
            return mask
        lin_speed = torch.norm(self.commands[env_ids, :2], dim=1)
        yaw_speed = torch.abs(self.commands[env_ids, 2])
        mask &= lin_speed <= self.arm_motion_max_lin_speed
        mask &= yaw_speed <= self.arm_motion_max_yaw_speed
        return mask

    def _update_arm_motion_elapsed(self):
        if not self.arm_motion_enabled:
            self.arm_motion_elapsed.zero_()
            return
        active_mask = self._get_arm_motion_command_mask().unsqueeze(1)
        self.arm_motion_elapsed = torch.where(
            active_mask,
            self.arm_motion_elapsed + self.dt,
            torch.zeros_like(self.arm_motion_elapsed),
        )

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self.actions

        if self.cfg.domain_rand.delay:
            latency_indices = torch.clip(self.lag_buffer, max=self.action_queue.size(1) - 1)
            delayed_actions = self.action_queue[torch.arange(self.num_envs, device=self.device), latency_indices]
        else:
            delayed_actions = self.actions

        self._update_arm_motion_elapsed()
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self._apply_arm_payload_force()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

        termination_ids, termination_privileged_obs = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_privileged_obs

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.policy_action_dim = self.num_actions
        leg_idx = []
        arm_idx = []
        leg_names = []
        for i, name in enumerate(self.dof_names):
            if name.startswith('arm_'):
                arm_idx.append(i)
            else:
                leg_idx.append(i)
                leg_names.append(name)
        if len(leg_idx) != self.policy_action_dim:
            raise RuntimeError(
                f'Expected {self.policy_action_dim} leg DOFs for the policy, found {len(leg_idx)}: {leg_names}'
            )

        self.leg_dof_indices = torch.tensor(leg_idx, dtype=torch.long, device=self.device, requires_grad=False)
        self.arm_dof_indices = torch.tensor(arm_idx, dtype=torch.long, device=self.device, requires_grad=False)
        self.hip_action_indices = torch.tensor(
            [i for i, name in enumerate(leg_names) if 'hip_joint' in name],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.policy_action_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.full_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()
        self._init_raibert_buffers()

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i, name in enumerate(self.dof_names):
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ['P', 'V']:
                    print(f'PD gain of joint {name} were not defined, setting them to zero')
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_leg_dof_pos = self.default_dof_pos[:, self.leg_dof_indices]
        self.arm_default_dof_pos = self.default_dof_pos[:, self.arm_dof_indices]
        self.arm_task_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.arm_carry_dof_pos = self._arm_pose_from_mapping(getattr(self.cfg.arm, 'carry_joint_angles', {})).unsqueeze(0)

        self.arm_motion_enabled = bool(getattr(self.cfg.arm, 'motion_enabled', False))
        self.arm_motion_only_for_low_speed = bool(getattr(self.cfg.arm, 'motion_only_for_low_speed_commands', False))
        self.arm_motion_max_lin_speed = float(getattr(self.cfg.arm, 'motion_max_lin_speed', 0.0))
        self.arm_motion_max_yaw_speed = float(getattr(self.cfg.arm, 'motion_max_yaw_speed', 0.0))
        self.arm_motion_start_time = float(getattr(self.cfg.arm, 'start_time', 0.0))
        self.arm_motion_ramp_time = float(getattr(self.cfg.arm, 'ramp_time', 0.0))
        self.arm_default_segment_duration = max(float(getattr(self.cfg.arm, 'default_segment_duration', 1.0)), 1e-3)
        task_mode_probs = to_torch(getattr(self.cfg.arm, 'task_mode_probs', [1.0, 0.0, 0.0]), device=self.device).flatten()
        if task_mode_probs.numel() != 3:
            raise RuntimeError(f'arm.task_mode_probs must have 3 values, got {task_mode_probs.numel()}')
        self.arm_task_mode_probs = task_mode_probs / torch.clamp(task_mode_probs.sum(), min=1e-6)
        self.arm_manip_lin_vel_x_range = getattr(self.cfg.arm, 'manip_lin_vel_x_range', [-0.05, 0.05])
        self.arm_manip_lin_vel_y_range = getattr(self.cfg.arm, 'manip_lin_vel_y_range', [-0.05, 0.05])
        self.arm_manip_ang_vel_yaw_range = getattr(self.cfg.arm, 'manip_ang_vel_yaw_range', [-0.2, 0.2])
        self.arm_manip_heading_offset_range = getattr(self.cfg.arm, 'manip_heading_offset_range', [-0.15, 0.15])
        self.arm_carry_lin_vel_x_range = getattr(self.cfg.arm, 'carry_lin_vel_x_range', [0.1, 0.4])
        self.arm_carry_lin_vel_y_range = getattr(self.cfg.arm, 'carry_lin_vel_y_range', [-0.15, 0.15])
        self.arm_carry_ang_vel_yaw_range = getattr(self.cfg.arm, 'carry_ang_vel_yaw_range', [-0.4, 0.4])
        self.arm_carry_heading_offset_range = getattr(self.cfg.arm, 'carry_heading_offset_range', [-0.35, 0.35])
        self.arm_traj_rand_enabled = bool(getattr(self.cfg.arm, 'trajectory_rand_enabled', False))
        self.arm_traj_joint_offset_range = to_torch(
            getattr(self.cfg.arm, 'trajectory_joint_offset_range', [0.0] * max(len(self.arm_dof_indices), 1)),
            device=self.device,
        ).view(1, -1)
        self.arm_traj_time_scale_range = getattr(self.cfg.arm, 'trajectory_time_scale_range', [1.0, 1.0])
        self.box_payload_enabled = bool(getattr(self.cfg.arm, 'payload_enabled', False))
        self.box_payload_probability = float(getattr(self.cfg.arm, 'payload_probability', 1.0))
        self.box_payload_body_name = getattr(self.cfg.arm, 'payload_body_name', 'arm_link_5')
        self.box_payload_base_offset = to_torch(getattr(self.cfg.arm, 'payload_local_offset', [0.0, 0.0, 0.125]), device=self.device).unsqueeze(0)
        self.box_payload_offset_jitter = to_torch(getattr(self.cfg.arm, 'payload_offset_jitter', [0.0, 0.0, 0.0]), device=self.device).unsqueeze(0)
        self.box_payload_timing_jitter = float(getattr(self.cfg.arm, 'payload_timing_jitter', 0.0))
        payload_mass_range = getattr(self.cfg.arm, 'payload_mass_range', [0.5, 0.5])
        self.box_payload_mass_range = (float(payload_mass_range[0]), float(payload_mass_range[1]))

        self.box_payload_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.box_payload_body_name)
        if self.box_payload_body_index < 0:
            raise RuntimeError(f'Failed to find payload body {self.box_payload_body_name}')

        self._build_arm_trajectory_library()

        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_payload_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_payload_force_positions = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_payload_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_payload_enabled_mask = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_payload_scale = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.box_payload_local_offset = self.box_payload_base_offset.repeat(self.num_envs, 1)
        self.box_payload_timing_offset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_traj_joint_offsets = torch.zeros(self.num_envs, self.arm_dof_indices.numel(), dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_traj_time_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_motion_elapsed = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.cmd_curr_ema_low = 0.0
        self.cmd_curr_ema_high = 0.0
        self.cmd_curr_pass_streak = 0
        self.cmd_curr_buffer_cmd_x = []
        self.cmd_curr_buffer_ratio = []

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)

        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

        hist_len = self.cfg.domain_rand.lag_timesteps + 1
        self.action_queue = torch.zeros(self.num_envs, hist_len, self.policy_action_dim, device=self.device, requires_grad=False)
        self.lag_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_impact_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.stuck_time = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        self._resample_arm_payload(self.all_env_ids)

    def _arm_pose_from_mapping(self, joint_angles):
        pose = self.arm_default_dof_pos.squeeze(0).clone()
        for local_idx, dof_idx in enumerate(self.arm_dof_indices.tolist()):
            dof_name = self.dof_names[dof_idx]
            if dof_name in joint_angles:
                try:
                    pose[local_idx] = float(joint_angles[dof_name])
                except (TypeError, ValueError):
                    continue
        lower = self.dof_pos_limits[self.arm_dof_indices, 0]
        upper = self.dof_pos_limits[self.arm_dof_indices, 1]
        return torch.clamp(pose, lower, upper)

    def _build_arm_trajectory_library(self):
        default_pose = self.arm_default_dof_pos.squeeze(0).clone()
        raw_library = list(getattr(self.cfg.arm, 'trajectory_library', []))
        if not raw_library:
            raw_library = [{'name': 'default_hold', 'segment_duration': self.arm_default_segment_duration, 'waypoints': []}]

        trajectory_refs = []
        payload_states = []
        segment_counts = []
        segment_durations = []
        for traj_cfg in raw_library:
            refs = [default_pose.clone()]
            payload_nodes = [0.0]
            for waypoint_cfg in traj_cfg.get('waypoints', []):
                refs.append(self._arm_pose_from_mapping(waypoint_cfg))
                payload_nodes.append(float(bool(waypoint_cfg.get('payload_on', payload_nodes[-1] > 0.5))))
            if len(refs) > 1:
                refs.append(default_pose.clone())
                payload_nodes.append(0.0)
            traj_tensor = torch.stack(refs, dim=0)
            payload_tensor = torch.tensor(payload_nodes, dtype=torch.float, device=self.device)
            trajectory_refs.append(traj_tensor)
            payload_states.append(payload_tensor)
            segment_counts.append(max(traj_tensor.shape[0] - 1, 0))
            segment_durations.append(max(float(traj_cfg.get('segment_duration', self.arm_default_segment_duration)), 1e-3))

        max_nodes = max(traj.shape[0] for traj in trajectory_refs)
        padded_traj = []
        padded_payload = []
        for traj_tensor, payload_tensor in zip(trajectory_refs, payload_states):
            if traj_tensor.shape[0] < max_nodes:
                pad_count = max_nodes - traj_tensor.shape[0]
                traj_tensor = torch.cat((traj_tensor, traj_tensor[-1:].repeat(pad_count, 1)), dim=0)
                payload_tensor = torch.cat((payload_tensor, payload_tensor[-1:].repeat(pad_count)), dim=0)
            padded_traj.append(traj_tensor)
            padded_payload.append(payload_tensor)

        self.arm_trajectory_library = torch.stack(padded_traj, dim=0)
        self.arm_payload_state_library = torch.stack(padded_payload, dim=0)
        self.arm_num_segments_per_traj = torch.tensor(segment_counts, dtype=torch.long, device=self.device)
        self.arm_segment_duration_per_traj = torch.tensor(segment_durations, dtype=torch.float, device=self.device)
        self.arm_cycle_duration_per_traj = self.arm_segment_duration_per_traj * torch.clamp(self.arm_num_segments_per_traj.float(), min=1.0)
        self.arm_selected_traj_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._resample_arm_trajectories(self.all_env_ids)

    def _resample_arm_trajectories(self, env_ids):
        if self.arm_trajectory_library.shape[0] == 0 or len(env_ids) == 0:
            return
        if self.arm_trajectory_library.shape[0] == 1:
            self.arm_selected_traj_ids[env_ids] = 0
        else:
            self.arm_selected_traj_ids[env_ids] = torch.randint(0, self.arm_trajectory_library.shape[0], (len(env_ids),), device=self.device)

    def _resample_arm_trajectory_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.arm_traj_joint_offsets.shape[1] == 0:
            self.arm_traj_time_scale[env_ids] = 1.0
            return
        if not self.arm_traj_rand_enabled:
            self.arm_traj_joint_offsets[env_ids] = 0.0
            self.arm_traj_time_scale[env_ids] = 1.0
            return

        joint_range = self.arm_traj_joint_offset_range
        if joint_range.shape[1] != self.arm_dof_indices.numel():
            if joint_range.shape[1] == 1:
                joint_range = joint_range.repeat(1, self.arm_dof_indices.numel())
            else:
                raise RuntimeError(
                    f'trajectory_joint_offset_range expects {self.arm_dof_indices.numel()} values, got {joint_range.shape[1]}'
                )
        offset_noise = (2.0 * torch.rand(len(env_ids), self.arm_dof_indices.numel(), device=self.device) - 1.0) * joint_range
        self.arm_traj_joint_offsets[env_ids] = offset_noise
        lo, hi = float(self.arm_traj_time_scale_range[0]), float(self.arm_traj_time_scale_range[1])
        self.arm_traj_time_scale[env_ids] = torch_rand_float(lo, hi, (len(env_ids), 1), device=self.device)

    def _resample_arm_payload(self, env_ids):
        if len(env_ids) == 0:
            return
        self.box_payload_mass[env_ids] = 0.0
        self.box_payload_enabled_mask[env_ids] = 0.0
        self.box_payload_local_offset[env_ids] = self.box_payload_base_offset.repeat(len(env_ids), 1)
        self.box_payload_timing_offset[env_ids] = 0.0

        if not self.box_payload_enabled:
            return

        offset_noise = (2.0 * torch.rand(len(env_ids), 3, device=self.device) - 1.0) * self.box_payload_offset_jitter
        self.box_payload_local_offset[env_ids] = self.box_payload_base_offset.repeat(len(env_ids), 1) + offset_noise
        self.box_payload_mass[env_ids] = torch_rand_float(self.box_payload_mass_range[0], self.box_payload_mass_range[1], (len(env_ids), 1), device=self.device)

        modes = self.arm_task_mode[env_ids]
        manip_mask = modes == self.MODE_MANIP_STATIONARY
        carry_mask = modes == self.MODE_CARRY_MOVE

        if torch.any(manip_mask):
            sampled = (torch.rand(int(manip_mask.sum().item()), 1, device=self.device) < self.box_payload_probability).float()
            self.box_payload_enabled_mask[env_ids[manip_mask]] = sampled
            if self.box_payload_timing_jitter > 0.0:
                self.box_payload_timing_offset[env_ids[manip_mask]] = (2.0 * torch.rand(int(manip_mask.sum().item()), 1, device=self.device) - 1.0) * self.box_payload_timing_jitter

        if torch.any(carry_mask):
            self.box_payload_enabled_mask[env_ids[carry_mask]] = 1.0

    def _get_arm_segment_interp(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids
        num = len(env_ids)
        default_ref = self.arm_default_dof_pos.repeat(num, 1)
        if self.arm_dof_indices.numel() == 0 or not self.arm_motion_enabled:
            return env_ids, default_ref, None, None, None, None, None

        traj_ids = self.arm_selected_traj_ids[env_ids]
        num_segments = self.arm_num_segments_per_traj[traj_ids]
        command_mask = self._get_arm_motion_command_mask(env_ids)
        active_mask = (num_segments > 0) & command_mask
        if not torch.any(active_mask):
            elapsed = torch.clamp(self.arm_motion_elapsed[env_ids].squeeze(1) - self.arm_motion_start_time, min=0.0)
            return env_ids, default_ref, traj_ids, None, None, active_mask, elapsed

        segment_duration = self.arm_segment_duration_per_traj[traj_ids] * self.arm_traj_time_scale[env_ids].squeeze(1)
        cycle_duration = self.arm_cycle_duration_per_traj[traj_ids] * self.arm_traj_time_scale[env_ids].squeeze(1)
        elapsed = self.arm_motion_elapsed[env_ids].squeeze(1) - self.arm_motion_start_time
        elapsed = torch.clamp(elapsed, min=0.0)
        phase_time = torch.remainder(elapsed, cycle_duration)
        raw_segment_idx = torch.floor(phase_time / segment_duration).long()
        segment_idx = torch.minimum(raw_segment_idx, torch.clamp(num_segments - 1, min=0))
        segment_start_time = segment_idx.float() * segment_duration
        alpha = ((phase_time - segment_start_time) / segment_duration).unsqueeze(1)
        return env_ids, default_ref, traj_ids, segment_idx, alpha, active_mask, elapsed

    def _get_current_arm_reference(self, env_ids=None):
        env_ids, default_ref, traj_ids, segment_idx, alpha, active_mask, elapsed = self._get_arm_segment_interp(env_ids)
        mode = self.arm_task_mode[env_ids]
        carry_mask = mode == self.MODE_CARRY_MOVE
        if traj_ids is None or segment_idx is None:
            arm_ref = default_ref
            if torch.any(carry_mask):
                arm_ref = arm_ref.clone()
                arm_ref[carry_mask] = self.arm_carry_dof_pos.repeat(len(env_ids), 1)[carry_mask]
            if self.arm_traj_joint_offsets.shape[1] > 0:
                arm_ref = arm_ref + self.arm_traj_joint_offsets[env_ids]
            lower = self.dof_pos_limits[self.arm_dof_indices, 0]
            upper = self.dof_pos_limits[self.arm_dof_indices, 1]
            return torch.clamp(arm_ref, lower, upper)

        num = len(env_ids)
        traj_refs = self.arm_trajectory_library[traj_ids]
        batch_idx = torch.arange(num, device=self.device)
        start_ref = traj_refs[batch_idx, segment_idx]
        end_ref = traj_refs[batch_idx, torch.clamp(segment_idx + 1, max=traj_refs.shape[1] - 1)]
        arm_ref = start_ref + alpha * (end_ref - start_ref)
        arm_ref = torch.where(active_mask.unsqueeze(1), arm_ref, default_ref)

        if self.arm_motion_ramp_time > 0.0:
            ramp = torch.clamp(elapsed / self.arm_motion_ramp_time, 0.0, 1.0).unsqueeze(1)
            arm_ref = default_ref + ramp * (arm_ref - default_ref)
        if torch.any(carry_mask):
            arm_ref = arm_ref.clone()
            arm_ref[carry_mask] = self.arm_carry_dof_pos.repeat(len(env_ids), 1)[carry_mask]
        if self.arm_traj_joint_offsets.shape[1] > 0:
            arm_ref = arm_ref + self.arm_traj_joint_offsets[env_ids]
        lower = self.dof_pos_limits[self.arm_dof_indices, 0]
        upper = self.dof_pos_limits[self.arm_dof_indices, 1]
        arm_ref = torch.clamp(arm_ref, lower, upper)
        return arm_ref

    def _get_current_payload_scale(self, env_ids=None):
        env_ids, _, traj_ids, segment_idx, alpha, active_mask, elapsed = self._get_arm_segment_interp(env_ids)
        num = len(env_ids)
        scale = torch.zeros(num, 1, device=self.device)
        mode = self.arm_task_mode[env_ids]
        carry_mask = mode == self.MODE_CARRY_MOVE
        if traj_ids is None or segment_idx is None:
            if torch.any(carry_mask):
                scale[carry_mask] = self.box_payload_enabled_mask[env_ids[carry_mask]]
            return scale

        payload_elapsed = elapsed
        if self.box_payload_timing_jitter > 0.0:
            payload_elapsed = torch.clamp(elapsed + self.box_payload_timing_offset[env_ids].squeeze(1), min=0.0)
            segment_duration = self.arm_segment_duration_per_traj[traj_ids] * self.arm_traj_time_scale[env_ids].squeeze(1)
            cycle_duration = self.arm_cycle_duration_per_traj[traj_ids] * self.arm_traj_time_scale[env_ids].squeeze(1)
            payload_phase_time = torch.remainder(payload_elapsed, cycle_duration)
            payload_segment_idx = torch.floor(payload_phase_time / segment_duration).long()
            payload_segment_idx = torch.minimum(payload_segment_idx, torch.clamp(self.arm_num_segments_per_traj[traj_ids] - 1, min=0))
            payload_segment_start_time = payload_segment_idx.float() * segment_duration
            payload_alpha = ((payload_phase_time - payload_segment_start_time) / segment_duration).unsqueeze(1)
        else:
            payload_segment_idx = segment_idx
            payload_alpha = alpha

        batch_idx = torch.arange(num, device=self.device)
        payload_nodes = self.arm_payload_state_library[traj_ids]
        start_state = payload_nodes[batch_idx, payload_segment_idx].unsqueeze(1)
        end_state = payload_nodes[batch_idx, torch.clamp(payload_segment_idx + 1, max=payload_nodes.shape[1] - 1)].unsqueeze(1)
        scale = start_state + payload_alpha * (end_state - start_state)
        scale = torch.where(active_mask.unsqueeze(1), scale, torch.zeros_like(scale))

        if self.arm_motion_ramp_time > 0.0:
            ramp = torch.clamp(elapsed / self.arm_motion_ramp_time, 0.0, 1.0).unsqueeze(1)
            scale = scale * ramp
        scale = scale * self.box_payload_enabled_mask[env_ids]
        if torch.any(carry_mask):
            scale = scale.clone()
            scale[carry_mask] = self.box_payload_enabled_mask[env_ids[carry_mask]]
        return scale

    def _apply_arm_payload_force(self):
        self.box_payload_forces.zero_()
        self.box_payload_force_positions.zero_()

        if self.arm_dof_indices.numel() == 0 or self.box_payload_body_index < 0:
            return

        payload_scale = self._get_current_payload_scale()
        self.box_payload_scale[:] = payload_scale
        effective_mass = self.box_payload_mass * payload_scale
        if not torch.any(effective_mass > 1e-6):
            return

        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        body_pos = body_states[:, self.box_payload_body_index, 0:3]
        body_quat = body_states[:, self.box_payload_body_index, 3:7]
        world_offset = quat_apply(body_quat, self.box_payload_local_offset)
        force_positions = body_pos + world_offset

        self.box_payload_forces[:, self.box_payload_body_index, 2] = -9.81 * effective_mass.squeeze(1)
        self.box_payload_force_positions[:, self.box_payload_body_index, :] = force_positions
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.box_payload_forces),
            gymtorch.unwrap_tensor(self.box_payload_force_positions),
            gymapi.ENV_SPACE,
        )

    def _reset_dofs(self, env_ids):
        self._sample_arm_task_modes(env_ids)
        self._resample_arm_trajectories(env_ids)
        self._resample_arm_trajectory_randomization(env_ids)
        self._resample_arm_payload(env_ids)
        self.arm_motion_elapsed[env_ids] = 0.0

        leg_defaults = self.default_leg_dof_pos.repeat(len(env_ids), 1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            leg_scales = torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_range[0],
                self.cfg.domain_rand.initial_joint_pos_range[1],
                (len(env_ids), len(self.leg_dof_indices)),
                device=self.device,
            )
            self.dof_pos[env_ids[:, None], self.leg_dof_indices] = leg_defaults * leg_scales
        else:
            self.dof_pos[env_ids[:, None], self.leg_dof_indices] = leg_defaults

        arm_defaults = self.arm_default_dof_pos.repeat(len(env_ids), 1)
        self.dof_pos[env_ids[:, None], self.arm_dof_indices] = arm_defaults
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids] = 0.0
        modes = self.arm_task_mode[env_ids]

        loco_ids = env_ids[modes == self.MODE_LOCOMOTION_ONLY]
        manip_ids = env_ids[modes == self.MODE_MANIP_STATIONARY]
        carry_ids = env_ids[modes == self.MODE_CARRY_MOVE]

        if len(loco_ids) > 0:
            self.commands[loco_ids, 0] = torch_rand_float(
                self.command_ranges['lin_vel_x'][0], self.command_ranges['lin_vel_x'][1], (len(loco_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[loco_ids, 1] = torch_rand_float(
                self.command_ranges['lin_vel_y'][0], self.command_ranges['lin_vel_y'][1], (len(loco_ids), 1), device=self.device
            ).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[loco_ids, 3] = torch_rand_float(
                    self.command_ranges['heading'][0], self.command_ranges['heading'][1], (len(loco_ids), 1), device=self.device
                ).squeeze(1)
            else:
                self.commands[loco_ids, 2] = torch_rand_float(
                    self.command_ranges['ang_vel_yaw'][0], self.command_ranges['ang_vel_yaw'][1], (len(loco_ids), 1), device=self.device
                ).squeeze(1)
            self.commands[loco_ids, :2] *= (torch.norm(self.commands[loco_ids, :2], dim=1) > 0.2).unsqueeze(1)

        if len(manip_ids) > 0:
            self.commands[manip_ids, 0] = torch_rand_float(
                self.arm_manip_lin_vel_x_range[0], self.arm_manip_lin_vel_x_range[1], (len(manip_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[manip_ids, 1] = torch_rand_float(
                self.arm_manip_lin_vel_y_range[0], self.arm_manip_lin_vel_y_range[1], (len(manip_ids), 1), device=self.device
            ).squeeze(1)
            if self.cfg.commands.heading_command:
                self._sample_heading_targets(manip_ids, self.arm_manip_heading_offset_range)
            else:
                self.commands[manip_ids, 2] = torch_rand_float(
                    self.arm_manip_ang_vel_yaw_range[0], self.arm_manip_ang_vel_yaw_range[1], (len(manip_ids), 1), device=self.device
                ).squeeze(1)

        if len(carry_ids) > 0:
            self.commands[carry_ids, 0] = torch_rand_float(
                self.arm_carry_lin_vel_x_range[0], self.arm_carry_lin_vel_x_range[1], (len(carry_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[carry_ids, 1] = torch_rand_float(
                self.arm_carry_lin_vel_y_range[0], self.arm_carry_lin_vel_y_range[1], (len(carry_ids), 1), device=self.device
            ).squeeze(1)
            if self.cfg.commands.heading_command:
                self._sample_heading_targets(carry_ids, self.arm_carry_heading_offset_range)
            else:
                self.commands[carry_ids, 2] = torch_rand_float(
                    self.arm_carry_ang_vel_yaw_range[0], self.arm_carry_ang_vel_yaw_range[1], (len(carry_ids), 1), device=self.device
                ).squeeze(1)

    def _expand_policy_actions(self, policy_actions):
        full = self.default_dof_pos.repeat(policy_actions.shape[0], 1)
        if self.arm_dof_indices.numel() > 0:
            full[:, self.arm_dof_indices] = self._get_current_arm_reference()
        leg_actions = policy_actions * self.cfg.control.action_scale
        if self.hip_action_indices.numel() > 0:
            leg_actions[:, self.hip_action_indices] *= self.cfg.control.hip_reduction
        full[:, self.leg_dof_indices] += leg_actions
        return full

    def _compute_arm_pd_torques(self):
        if self.arm_dof_indices.numel() == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        arm_ref = self._get_current_arm_reference()
        arm_pos = self.dof_pos[:, self.arm_dof_indices]
        arm_vel = self.dof_vel[:, self.arm_dof_indices]
        arm_p = self.p_gains[self.arm_dof_indices].unsqueeze(0) * self.Kp_factors
        arm_d = self.d_gains[self.arm_dof_indices].unsqueeze(0) * self.Kd_factors
        return arm_p * (arm_ref - arm_pos) - arm_d * arm_vel

    def _compute_torques(self, actions):
        control_type = self.cfg.control.control_type
        if control_type == 'P':
            self.full_actions = self._expand_policy_actions(actions)
            self.joint_pos_target = self.full_actions
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type == 'V':
            leg_actions = actions * self.cfg.control.action_scale
            if self.hip_action_indices.numel() > 0:
                leg_actions[:, self.hip_action_indices] *= self.cfg.control.hip_reduction
            full_velocity_targets = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            full_velocity_targets[:, self.leg_dof_indices] = leg_actions
            torques = self.p_gains * (full_velocity_targets - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            if self.arm_dof_indices.numel() > 0:
                torques[:, self.arm_dof_indices] = self._compute_arm_pd_torques()
        elif control_type == 'T':
            full_torque_cmd = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            full_torque_cmd[:, self.leg_dof_indices] = actions * self.cfg.control.action_scale
            torques = full_torque_cmd
            if self.arm_dof_indices.numel() > 0:
                torques[:, self.arm_dof_indices] = self._compute_arm_pd_torques()
        else:
            raise NameError(f'Unknown controller type: {control_type}')

        friction_torque = 0.35 * torch.tanh(3.0 * self.dof_vel) + 0.1 * self.dof_vel
        torques = torques - friction_torque
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        leg_dof_pos = self.dof_pos[:, self.leg_dof_indices]
        leg_dof_vel = self.dof_vel[:, self.leg_dof_indices]
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (leg_dof_pos - self.default_leg_dof_pos) * self.obs_scales.dof_pos,
            leg_dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.policy_action_dim):(9 + 3 * self.policy_action_dim + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def compute_termination_observations(self, env_ids):
        leg_dof_pos = self.dof_pos[:, self.leg_dof_indices]
        leg_dof_vel = self.dof_vel[:, self.leg_dof_indices]
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (leg_dof_pos - self.default_leg_dof_pos) * self.obs_scales.dof_pos,
            leg_dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.policy_action_dim):(9 + 3 * self.policy_action_dim + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]

    def _reward_dof_acc(self):
        leg_vel = self.dof_vel[:, self.leg_dof_indices]
        last_leg_vel = self.last_dof_vel[:, self.leg_dof_indices]
        return torch.sum(torch.square((last_leg_vel - leg_vel) / self.dt), dim=1)

    def _reward_joint_power(self):
        leg_vel = torch.abs(self.dof_vel[:, self.leg_dof_indices])
        leg_torques = torch.abs(self.torques[:, self.leg_dof_indices])
        return torch.sum(leg_vel * leg_torques, dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques[:, self.leg_dof_indices]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)

    def _reward_dof_pos_limits(self):
        leg_pos = self.dof_pos[:, self.leg_dof_indices]
        leg_limits = self.dof_pos_limits[self.leg_dof_indices]
        out_of_limits = -(leg_pos - leg_limits[:, 0]).clip(max=0.0)
        out_of_limits += (leg_pos - leg_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        leg_vel = torch.abs(self.dof_vel[:, self.leg_dof_indices])
        leg_limits = self.dof_vel_limits[self.leg_dof_indices]
        return torch.sum((leg_vel - leg_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.0, max=1.0), dim=1)

    def _reward_torque_limits(self):
        leg_torque = torch.abs(self.torques[:, self.leg_dof_indices])
        leg_limits = self.torque_limits[self.leg_dof_indices]
        return torch.sum((leg_torque - leg_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    def _reward_stand_still(self):
        leg_pos = self.dof_pos[:, self.leg_dof_indices]
        leg_vel = self.dof_vel[:, self.leg_dof_indices]
        is_still = (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        pos_error = torch.sum(torch.abs(leg_pos - self.default_leg_dof_pos), dim=1)
        vel_error = torch.sum(torch.abs(leg_vel), dim=1)
        return (pos_error + 0.08 * vel_error) * is_still

    def _reward_all_joint_pos(self):
        leg_pos = self.dof_pos[:, self.leg_dof_indices]
        return torch.sum(torch.square(leg_pos - self.default_leg_dof_pos), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.abs(self.projected_gravity[:, :2]), dim=1)
