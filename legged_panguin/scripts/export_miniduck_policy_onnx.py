import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_panguin.envs.miniduck.miniduck_config import (
    JOINT_MIRROR_PERMUTATION,
    JOINT_MIRROR_SIGNS,
    MiniDuckFlatCfg,
    MiniDuckFlatCfgPPO,
    SYMMETRY_OBS_PERMUTATION,
    SYMMETRY_OBS_SIGNS,
)
from rsl_rl.modules.actor_critic import ActorCritic


JOINT_ORDER = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


class SymmetricActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_permutation", torch.tensor(SYMMETRY_OBS_PERMUTATION))
        self.register_buffer("obs_signs", torch.tensor(SYMMETRY_OBS_SIGNS))
        self.register_buffer("action_permutation", torch.tensor(JOINT_MIRROR_PERMUTATION))
        self.register_buffer("action_signs", torch.tensor(JOINT_MIRROR_SIGNS))

    def forward(self, obs):
        action = self.actor(obs)
        mirrored_action = self.actor(obs[:, self.obs_permutation] * self.obs_signs)
        unmirrored_action = mirrored_action[:, self.action_permutation] * self.action_signs
        return 0.5 * (action + unmirrored_action)


def build_actor_critic():
    policy_cfg = MiniDuckFlatCfgPPO.policy
    return ActorCritic(
        num_actor_obs=MiniDuckFlatCfg.env.num_observations,
        num_critic_obs=MiniDuckFlatCfg.env.num_observations,
        num_actions=MiniDuckFlatCfg.env.num_actions,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std,
    )


def build_metadata(checkpoint_path, output_onnx):
    init_state = MiniDuckFlatCfg.init_state
    control = MiniDuckFlatCfg.control
    commands = MiniDuckFlatCfg.commands
    obs_scales = MiniDuckFlatCfg.normalization.obs_scales
    domain_rand = MiniDuckFlatCfg.domain_rand
    skill_curriculum = MiniDuckFlatCfg.skill_curriculum

    default_joint_angles = init_state.default_joint_angles
    default_actuator = [default_joint_angles[name] for name in JOINT_ORDER]

    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "onnx_path": str(Path(output_onnx).expanduser().resolve()),
        "obs_size": MiniDuckFlatCfg.env.num_observations,
        "action_size": MiniDuckFlatCfg.env.num_actions,
        "joint_names": JOINT_ORDER,
        "default_actuator": default_actuator,
        "action_scale": control.action_scale,
        "policy_dt": control.decimation * MiniDuckFlatCfg.sim.dt,
        "sim_dt": MiniDuckFlatCfg.sim.dt,
        "decimation": control.decimation,
        "gait_phase_period_steps": 27,
        "nominal_motor_velocity": domain_rand.nominal_motor_velocity,
        "nominal_body_height_m": skill_curriculum.nominal_body_height_m,
        "squat_body_height_m": skill_curriculum.squat_body_height_m,
        "heading_hold_kp": skill_curriculum.heading_hold_kp,
        "heading_hold_kd": skill_curriculum.heading_hold_kd,
        "heading_hold_max_yaw_rate": skill_curriculum.heading_hold_max_yaw_rate,
        "line_hold_kp": skill_curriculum.line_hold_kp,
        "line_hold_kd": skill_curriculum.line_hold_kd,
        "line_hold_max_lateral_mps": skill_curriculum.line_hold_max_lateral_mps,
        "heading_hold_stop_s": commands.resampling_time,
        "command_ranges": {
            "lin_vel_x": list(commands.ranges.lin_vel_x),
            "lin_vel_y": list(commands.ranges.lin_vel_y),
            "ang_vel_yaw": list(commands.ranges.ang_vel_yaw),
        },
        "command_min_abs": {
            "lin_vel_x": commands.min_abs_x,
            "lin_vel_y": commands.min_abs_y,
            "ang_vel_yaw": commands.min_abs_yaw,
        },
        "obs_scales": {
            "ang_vel": obs_scales.ang_vel,
            "gravity": 1.0,
            "accel": MiniDuckFlatCfg.normalization.obs_scales.accel,
            "command": [
                obs_scales.lin_vel,
                obs_scales.lin_vel,
                obs_scales.ang_vel,
            ],
            "dof_pos": obs_scales.dof_pos,
            "dof_vel": obs_scales.dof_vel,
        },
        "base_init_pos": list(init_state.pos),
        "base_init_quat_xyzw": list(init_state.rot),
        "base_init_quat_wxyz": [
            init_state.rot[3],
            init_state.rot[0],
            init_state.rot[1],
            init_state.rot[2],
        ],
        "obs_layout": {
            "gyro": [0, 3],
            "gravity": [3, 6],
            "accelerometer": [6, 9],
            "command": [9, 12],
            "joint_pos_error": [12, 22],
            "joint_vel": [22, 32],
            "action_t": [32, 42],
            "action_t_minus_1": [42, 52],
            "action_t_minus_2": [52, 62],
            "gait_phase_or_skill_target": [62, 64],
        },
        "skill_encoding": {
            "locomotion": "[cos(gait_phase), sin(gait_phase)]",
            "stand": [1.0, 0.0],
            "squat": "[1 - 2 * normalized_depth, 1]",
            "recovery": [-1.0, -1.0],
        },
        "symmetric_inference": False,
    }


def load_compatible_state_dict(actor_critic, checkpoint):
    state = checkpoint["model_state_dict"]
    target_state = actor_critic.state_dict()
    for key, value in list(state.items()):
        target_value = target_state.get(key)
        if target_value is None or value.shape == target_value.shape:
            continue
        can_expand_input = (
            key in ("actor.0.weight", "critic.0.weight")
            and value.ndim == 2
            and target_value.ndim == 2
            and value.shape[0] == target_value.shape[0]
            and value.shape[1] < target_value.shape[1]
        )
        if not can_expand_input:
            raise RuntimeError(
                f"Checkpoint tensor shape mismatch for {key}: "
                f"{tuple(value.shape)} -> {tuple(target_value.shape)}"
            )
        expanded = target_value.clone()
        expanded[:, : value.shape[1]] = value
        expanded[:, value.shape[1] :] = 0.0
        state[key] = expanded
    actor_critic.load_state_dict(state)


def export_onnx(checkpoint_path, output_onnx):
    actor_critic = build_actor_critic()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_compatible_state_dict(actor_critic, checkpoint)
    actor = actor_critic.actor.eval()

    dummy_obs = torch.zeros(
        1,
        MiniDuckFlatCfg.env.num_observations,
        dtype=torch.float32,
    )
    torch.onnx.export(
        actor,
        dummy_obs,
        output_onnx,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_onnx", type=str, required=True)
    parser.add_argument("--output_metadata", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    output_onnx = str(Path(args.output_onnx).expanduser().resolve())
    output_metadata = args.output_metadata
    if output_metadata is None:
        output_metadata = str(Path(output_onnx).with_suffix(".json"))
    output_metadata = str(Path(output_metadata).expanduser().resolve())

    Path(output_onnx).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(checkpoint_path, output_onnx)

    metadata = build_metadata(checkpoint_path, output_onnx)
    Path(output_metadata).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Exported ONNX to: {output_onnx}")
    print(f"Wrote metadata to: {output_metadata}")


if __name__ == "__main__":
    main()
