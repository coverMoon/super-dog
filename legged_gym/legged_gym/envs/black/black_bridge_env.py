import torch

from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.black.black_env import BlackEnv
from legged_gym.utils.math import quat_apply_yaw


class BlackBridgeEnv(BlackEnv):
    """Bridge-focused variant of the Black environment."""

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
