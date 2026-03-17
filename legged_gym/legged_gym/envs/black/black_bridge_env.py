import torch

from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.black.black_env import BlackEnv
from legged_gym.utils.math import quat_apply_yaw


class BlackBridgeEnv(BlackEnv):
    """Bridge-focused variant of the Black environment."""

    def _init_buffers(self):
        super()._init_buffers()
        self.prev_base_lin_vel = self.base_lin_vel.clone()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.prev_base_lin_vel[env_ids] = self.base_lin_vel[env_ids]

    def post_physics_step(self):
        env_ids, termination_privileged_obs = super().post_physics_step()
        self.prev_base_lin_vel[:] = self.base_lin_vel[:]
        return env_ids, termination_privileged_obs

    def _sample_terrain_height_at_points(self, points):
        """Sample terrain height at arbitrary world-space points."""
        points = points.clone()
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].reshape(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].reshape(-1), 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = (heights1 + heights2 + heights3) / 3
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _reward_gap_clearance(self):
        """
        Reward higher swing-leg lift only when the terrain ahead drops sharply,
        which makes it specific to bridge gaps instead of generic locomotion.
        """
        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float().unsqueeze(1)
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        swing_weight = 1.0 - contact_prob

        forward_world = quat_apply_yaw(self.base_quat, self.forward_vec)
        forward_world = forward_world / torch.clamp(torch.norm(forward_world, dim=1, keepdim=True), min=1e-6)
        look_ahead = 0.18
        ahead_points = self.feet_pos + forward_world.unsqueeze(1) * look_ahead

        current_terrain_height = self.feet_pos[:, :, 2] - self._get_feet_heights()
        ahead_terrain_height = self._sample_terrain_height_at_points(ahead_points)
        ahead_drop = torch.relu(current_terrain_height - ahead_terrain_height - 0.08)

        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, device=self.device
        )
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
                self.base_quat, cur_footpos_translated[:, i, :]
            )
        foot_height_body = footpos_in_body_frame[:, :, 2]

        extra_clearance = torch.relu(foot_height_body - (self.cfg.rewards.clearance_height_target + 0.06))
        reward = ahead_drop * extra_clearance * swing_weight * move_cmd
        return torch.sum(reward, dim=1)

    def _reward_gap_recovery_burst(self):
        """
        Reward command-direction acceleration only when a foot is genuinely falling into
        a bridge gap, so the policy learns to recover with a quick push instead of only
        keeping motions smooth.
        """
        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        move_cmd = cmd_norm > 0.1
        cmd_dir = cmd_xy / torch.clamp(cmd_norm.unsqueeze(1), min=1e-6)

        terrain_height = self.feet_pos[:, :, 2] - self._get_feet_heights()
        support_height_ref = torch.max(terrain_height, dim=1, keepdim=True).values
        gap_depth = torch.relu(support_height_ref - terrain_height - 0.12)

        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, device=self.device
        )
        footvel_in_body_frame = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, device=self.device
        )
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
                self.base_quat, cur_footpos_translated[:, i, :]
            )
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(
                self.base_quat, cur_footvel_translated[:, i, :]
            )

        foot_height_body = footpos_in_body_frame[:, :, 2]
        foot_fall_speed = torch.relu(-footvel_in_body_frame[:, :, 2] - 0.08)
        low_foot = torch.relu((self.cfg.rewards.clearance_height_target - 0.04) - foot_height_body)

        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        no_contact = 1.0 - torch.sigmoid((contact_force_z - 5.0) * 0.5)

        misstep_weight = gap_depth * foot_fall_speed * low_foot * no_contact
        misstep_event = torch.max(misstep_weight, dim=1).values

        base_acc_xy = (self.base_lin_vel[:, :2] - self.prev_base_lin_vel[:, :2]) / self.dt
        forward_acc = torch.relu(torch.sum(base_acc_xy * cmd_dir, dim=1))
        return forward_acc * misstep_event * move_cmd.float()

    def _reward_feet_stumble(self):
        """
        Bridge-specific stumble penalty.
        Penalize low feet that build large horizontal contact forces while the robot has
        a motion command but fails to make matching forward progress. This targets the
        common failure mode where a hind leg catches the next plank edge and keeps pushing.
        """
        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        move_cmd = cmd_norm > 0.1
        cmd_dir = cmd_xy / torch.clamp(cmd_norm.unsqueeze(1), min=1e-6)
        progress_speed = torch.sum(self.base_lin_vel[:, :2] * cmd_dir, dim=1)
        stalled_weight = torch.relu(cmd_norm - progress_speed - 0.05).unsqueeze(1)

        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, device=self.device
        )
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
                self.base_quat, cur_footpos_translated[:, i, :]
            )

        foot_height_body = footpos_in_body_frame[:, :, 2]
        edge_zone = torch.relu((self.cfg.rewards.clearance_height_target + 0.03) - foot_height_body)

        horiz_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        vert_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        snag_force = torch.relu(horiz_force - 1.5 * vert_force - 5.0)

        penalty = snag_force * edge_zone * stalled_weight * move_cmd.float().unsqueeze(1)
        return torch.sum(penalty, dim=1)

    def _reward_edge_escape(self):
        """
        Reward lifting a snagged foot upward when it catches the plank edge.
        Rear feet get more weight because the dominant bridge failure is the hind leg
        hanging on the next plank and pushing until the robot collapses.
        """
        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        move_cmd = cmd_norm > 0.1
        cmd_dir = cmd_xy / torch.clamp(cmd_norm.unsqueeze(1), min=1e-6)
        progress_speed = torch.sum(self.base_lin_vel[:, :2] * cmd_dir, dim=1)
        stalled_weight = torch.relu(cmd_norm - progress_speed - 0.05).unsqueeze(1)

        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, device=self.device
        )
        footvel_in_body_frame = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, device=self.device
        )
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
                self.base_quat, cur_footpos_translated[:, i, :]
            )
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(
                self.base_quat, cur_footvel_translated[:, i, :]
            )

        foot_height_body = footpos_in_body_frame[:, :, 2]
        edge_zone = torch.relu((self.cfg.rewards.clearance_height_target + 0.03) - foot_height_body)

        horiz_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        vert_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        snag_force = torch.relu(horiz_force - 1.5 * vert_force - 5.0)

        upward_vel = torch.relu(footvel_in_body_frame[:, :, 2] - 0.05)
        rear_weight = torch.tensor([0.5, 0.5, 1.0, 1.0], device=self.device).unsqueeze(0)

        reward = upward_vel * snag_force * edge_zone * stalled_weight * rear_weight
        reward = reward * move_cmd.float().unsqueeze(1)
        return torch.sum(reward, dim=1)
