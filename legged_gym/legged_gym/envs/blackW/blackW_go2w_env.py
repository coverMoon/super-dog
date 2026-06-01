import numpy as np
import torch

from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.blackW.blackW_env import BlackWEnv
from legged_gym.utils.math import wrap_to_pi


class BlackWGo2WRewardEnv(BlackWEnv):
    """blackW asset with Go2W-style command curriculum and reward terms."""

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
