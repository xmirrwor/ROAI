"""Independent physics-seeded MiniDuck single-leg learning task."""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_apply,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    torch_rand_float,
)

from .miniduck_single_leg_v2 import MiniDuckSingleLegV2


class MiniDuckSingleLegPhysics(MiniDuckSingleLegV2):
    """Learn active balance as a residual around a physics-derived pose."""

    def _init_buffers(self):
        super()._init_buffers()

        props = self.gym.get_actor_rigid_body_properties(
            self.envs[0], self.actor_handles[0]
        )
        self.physics_body_masses = torch.tensor(
            [prop.mass for prop in props], device=self.device, dtype=torch.float
        )
        com_vectors = [
            prop.com.p if hasattr(prop.com, "p") else prop.com for prop in props
        ]
        self.physics_local_com = torch.tensor(
            [[com.x, com.y, com.z] for com in com_vectors],
            device=self.device,
            dtype=torch.float,
        )

        roll = torch.full(
            (1,), self.cfg.init_state.base_roll, device=self.device
        )
        pitch = torch.full(
            (1,), self.cfg.init_state.base_pitch, device=self.device
        )
        yaw = torch.zeros(1, device=self.device)
        target_quat = quat_from_euler_xyz(roll, pitch, yaw)
        self.target_projected_gravity[:] = quat_rotate_inverse(
            target_quat, self.gravity_vec[:1]
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return

        roll = torch_rand_float(
            self.cfg.init_state.base_roll - self.cfg.init_state.reset_roll_range,
            self.cfg.init_state.base_roll + self.cfg.init_state.reset_roll_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        pitch = torch_rand_float(
            self.cfg.init_state.base_pitch - self.cfg.init_state.reset_pitch_range,
            self.cfg.init_state.base_pitch + self.cfg.init_state.reset_pitch_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        yaw = torch_rand_float(
            -self.cfg.init_state.reset_yaw_range,
            self.cfg.init_state.reset_yaw_range,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _whole_body_com(self):
        body_pos = self.rigid_body_state[:, :, :3]
        body_quat = self.rigid_body_state[:, :, 3:7]
        local_com = self.physics_local_com.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        offsets = quat_apply(
            body_quat.reshape(-1, 4), local_com.reshape(-1, 3)
        ).view(self.num_envs, self.num_bodies, 3)
        body_com = body_pos + offsets
        masses = self.physics_body_masses.view(1, -1, 1)
        return torch.sum(body_com * masses, dim=1) / torch.sum(masses)

    def _true_com_over_support_quality(self):
        support_xy, _ = self._support_and_swing(
            self.rigid_body_state[:, self.feet_indices, :2]
        )
        error = self._whole_body_com()[:, :2] - support_xy
        return torch.exp(
            -torch.square(error[:, 0]) / self.cfg.rewards.com_x_sigma
            -torch.square(error[:, 1]) / self.cfg.rewards.com_y_sigma
        )

    def _reference_pose_quality(self):
        error = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-error / self.cfg.rewards.reference_pose_sigma)

    def _com_outside_support(self):
        support_xy, _ = self._support_and_swing(
            self.rigid_body_state[:, self.feet_indices, :2]
        )
        error = torch.abs(self._whole_body_com()[:, :2] - support_xy)
        x_excess = torch.clamp(
            (error[:, 0] - self.cfg.rewards.com_x_limit)
            / self.cfg.rewards.com_x_limit,
            min=0.0,
        )
        y_excess = torch.clamp(
            (error[:, 1] - self.cfg.rewards.com_y_limit)
            / self.cfg.rewards.com_y_limit,
            min=0.0,
        )
        return torch.clamp(x_excess + y_excess, max=1.0)

    def _torque_saturation_excess(self):
        ratio = torch.abs(self.torques) / torch.clamp(
            self.torque_limits.unsqueeze(0), min=1.0e-6
        )
        excess = torch.clamp(
            (ratio - self.cfg.rewards.torque_soft_ratio)
            / (1.0 - self.cfg.rewards.torque_soft_ratio),
            min=0.0,
        )
        return torch.mean(torch.square(excess), dim=1)

    def _pose_weight(self):
        progress = min(
            self.common_step_counter / max(self.cfg.rewards.pose_decay_steps, 1),
            1.0,
        )
        return (
            self.cfg.rewards.pose_weight_start * (1.0 - progress)
            + self.cfg.rewards.pose_weight_end * progress
        )

    def _reward_single_leg_physics(self):
        support_contact, swing_contact = self._contact_state()
        valid_stance = (support_contact & ~swing_contact).float()
        airborne = (~support_contact & ~swing_contact).float()

        pose_weight = self._pose_weight()
        weights = self.cfg.rewards.quality_weights
        quality = (
            weights.true_com * self._true_com_over_support_quality()
            + pose_weight * self._reference_pose_quality()
            + weights.body_attitude * self._body_attitude_quality()
            + weights.base_height * self._base_height_quality()
            + weights.swing_clearance * self._swing_clearance_quality()
            + weights.stillness * self._stillness_quality()
            + weights.support_slip * self._support_slip_quality()
            + weights.continuous_hold * self._hold_quality()
        )
        normalizer = (
            weights.true_com
            + pose_weight
            + weights.body_attitude
            + weights.base_height
            + weights.swing_clearance
            + weights.stillness
            + weights.support_slip
            + weights.continuous_hold
        )
        quality = quality / max(normalizer, 1.0e-6)

        penalties = (
            self.cfg.rewards.swing_contact_penalty * swing_contact.float()
            + self.cfg.rewards.airborne_penalty * airborne
            + self.cfg.rewards.impact_penalty * self._impact_excess()
            + self.cfg.rewards.com_outside_penalty * self._com_outside_support()
            + self.cfg.rewards.torque_saturation_penalty
            * self._torque_saturation_excess()
        )
        return (0.10 + 0.90 * valid_stance) * quality - penalties

    def _reward_true_com_over_support_quality(self):
        return self._true_com_over_support_quality()

    def _reward_reference_pose_quality(self):
        return self._reference_pose_quality()

    def _reward_com_outside_support(self):
        return self._com_outside_support()

    def _reward_torque_saturation_excess(self):
        return self._torque_saturation_excess()
