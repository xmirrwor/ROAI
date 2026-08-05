"""Second-generation MiniDuck single-leg stability task.

This task deliberately uses one bounded composite reward instead of several
large independent positive rewards.  Correct single-foot contact gates the
positive reward, so hopping or briefly touching the requested foot cannot
outweigh body, foot, and motion quality.
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .miniduck import MiniDuck


class MiniDuckSingleLegV2(MiniDuck):
    def _init_buffers(self):
        super()._init_buffers()
        # The locomotion teacher starts with about 26 degrees of forward pitch.
        # Single-leg standing needs its own morphology-appropriate upright target.
        self.target_projected_gravity.zero_()
        self.target_projected_gravity[:, 2] = -1.0
        self.support_foot = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stable_hold_steps = torch.zeros_like(self.support_foot)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "stable_hold_steps"):
            return
        self.stable_hold_steps[env_ids] = 0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        stable = self._stable_hold_mask()
        self.stable_hold_steps = torch.where(
            stable,
            self.stable_hold_steps + 1,
            torch.zeros_like(self.stable_hold_steps),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.common_step_counter < self.cfg.commands.fixed_support_steps:
            self.support_foot[env_ids] = self.cfg.commands.fixed_support_foot
        else:
            self.support_foot[env_ids] = torch.randint(
                0, 2, (len(env_ids),), device=self.device
            )
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 1] = torch.where(
            self.support_foot[env_ids] == 0, 1.0, -1.0
        )

    def _support_and_swing(self, values):
        env_ids = torch.arange(self.num_envs, device=self.device)
        support = values[env_ids, self.support_foot]
        swing = values[env_ids, 1 - self.support_foot]
        return support, swing

    def _contact_state(self):
        contacts = self._current_foot_contacts()
        return self._support_and_swing(contacts)

    def _support_foot_flatness_quality(self):
        foot_quat = self.rigid_body_state[:, self.feet_indices, 3:7]
        support_quat, _ = self._support_and_swing(foot_quat)
        world_up = torch.zeros(self.num_envs, 3, device=self.device)
        world_up[:, 2] = 1.0
        up_in_foot = quat_rotate_inverse(support_quat, world_up)
        alignment = torch.clamp(up_in_foot[:, 2], min=-1.0, max=1.0)
        return torch.exp(
            -(1.0 - alignment) / self.cfg.rewards.foot_flatness_sigma
        )

    def _body_attitude_quality(self):
        target = self.target_projected_gravity.expand_as(self.projected_gravity)
        alignment = torch.clamp(
            torch.sum(self.projected_gravity * target, dim=1),
            min=-1.0,
            max=1.0,
        )
        return torch.exp(
            -(1.0 - alignment) / self.cfg.rewards.body_attitude_sigma
        )

    def _base_height_quality(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        error = torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.exp(-error / self.cfg.rewards.base_height_quality_sigma)

    def _base_over_support_quality(self):
        foot_xy = self.rigid_body_state[:, self.feet_indices, :2]
        support_xy, _ = self._support_and_swing(foot_xy)
        error = torch.sum(torch.square(self.root_states[:, :2] - support_xy), dim=1)
        return torch.exp(-error / self.cfg.rewards.base_over_support_sigma)

    def _swing_clearance_quality(self):
        clearance = self._foot_clearance()
        _, swing = self._support_and_swing(clearance)
        low_error = torch.square(
            torch.clamp(self.cfg.rewards.swing_clearance_min - swing, min=0.0)
        )
        high_error = torch.square(
            torch.clamp(swing - self.cfg.rewards.swing_clearance_max, min=0.0)
        )
        return torch.exp(
            -(low_error + high_error) / self.cfg.rewards.swing_clearance_sigma
        )

    def _stillness_quality(self):
        linear = torch.sum(torch.square(self.base_lin_vel), dim=1)
        tilt_rate = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        yaw_rate = torch.square(self.base_ang_vel[:, 2])
        motion = linear + 0.25 * tilt_rate + 0.05 * yaw_rate
        return torch.exp(-motion / self.cfg.rewards.stillness_sigma)

    def _support_slip_quality(self):
        foot_velocity = self.rigid_body_state[:, self.feet_indices, 7:10]
        horizontal_speed_sq = torch.sum(torch.square(foot_velocity[:, :, :2]), dim=-1)
        support_speed_sq, _ = self._support_and_swing(horizontal_speed_sq)
        return torch.exp(-support_speed_sq / self.cfg.rewards.support_slip_sigma)

    def _impact_excess(self):
        vertical_force = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2], min=0.0
        )
        support_force, _ = self._support_and_swing(vertical_force)
        threshold = self.cfg.rewards.max_support_contact_force
        ratio = torch.clamp(
            (support_force - threshold) / max(threshold, 1.0e-6), min=0.0
        )
        return torch.clamp(torch.square(ratio), max=1.0)

    def _hold_quality(self):
        hold_time = self.stable_hold_steps.float() * self.dt
        grace = self.cfg.rewards.hold_grace_s
        ramp = max(self.cfg.rewards.hold_ramp_s - grace, self.dt)
        return torch.clamp((hold_time - grace) / ramp, min=0.0, max=1.0)

    def _stable_hold_mask(self):
        support_contact, swing_contact = self._contact_state()
        linear_speed = torch.linalg.vector_norm(self.base_lin_vel, dim=1)
        tilt_rate = torch.linalg.vector_norm(self.base_ang_vel[:, :2], dim=1)
        return (
            support_contact
            & ~swing_contact
            & (linear_speed < self.cfg.rewards.hold_max_base_speed)
            & (tilt_rate < self.cfg.rewards.hold_max_tilt_rate)
            & (
                self._body_attitude_quality()
                > self.cfg.rewards.hold_min_attitude_quality
            )
            & (
                self._support_foot_flatness_quality()
                > self.cfg.rewards.hold_min_foot_flatness_quality
            )
        )

    def _reward_single_leg_stability_v2(self):
        support_contact, swing_contact = self._contact_state()
        valid_stance = (support_contact & ~swing_contact).float()
        airborne = (~support_contact & ~swing_contact).float()

        weights = self.cfg.rewards.quality_weights
        quality = (
            weights.base_over_support * self._base_over_support_quality()
            + weights.support_foot_flatness * self._support_foot_flatness_quality()
            + weights.body_attitude * self._body_attitude_quality()
            + weights.base_height * self._base_height_quality()
            + weights.swing_clearance * self._swing_clearance_quality()
            + weights.stillness * self._stillness_quality()
            + weights.support_slip * self._support_slip_quality()
            + weights.continuous_hold * self._hold_quality()
        )

        penalties = (
            self.cfg.rewards.swing_contact_penalty * swing_contact.float()
            + self.cfg.rewards.airborne_penalty * airborne
            + self.cfg.rewards.impact_penalty * self._impact_excess()
        )
        return (0.05 + 0.95 * valid_stance) * quality - penalties

    # Diagnostic-only reward functions.  They are listed in monitor_terms but
    # have no independent reward scales, keeping the optimisation objective
    # equal to the single composite function above.
    def _reward_stance_valid(self):
        support, swing = self._contact_state()
        return (support & ~swing).float()

    def _reward_continuous_hold_quality(self):
        return self._hold_quality()

    def _reward_base_over_support_quality(self):
        return self._base_over_support_quality()

    def _reward_support_foot_flatness_quality(self):
        return self._support_foot_flatness_quality()

    def _reward_body_attitude_quality(self):
        return self._body_attitude_quality()

    def _reward_base_height_quality(self):
        return self._base_height_quality()

    def _reward_swing_clearance_quality(self):
        return self._swing_clearance_quality()

    def _reward_stillness_quality(self):
        return self._stillness_quality()

    def _reward_support_slip_quality(self):
        return self._support_slip_quality()

    def _reward_airborne(self):
        support, swing = self._contact_state()
        return (~support & ~swing).float()

    def _reward_swing_contact(self):
        _, swing = self._contact_state()
        return swing.float()

    def _reward_impact_excess(self):
        return self._impact_excess()
