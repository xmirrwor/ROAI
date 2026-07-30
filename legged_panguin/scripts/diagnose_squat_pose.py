"""Measure which symmetric joint offsets actually lower MiniDuck safely."""

import json

import isaacgym  # Must be imported before torch.
import torch

from legged_panguin.envs import *  # noqa: F401,F403
from legged_panguin.utils import get_args, task_registry


CANDIDATES = (
    ("home", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("knees_plus", [0, 0, 0, 0.20, 0, 0, 0, 0, 0.20, 0]),
    ("knees_minus", [0, 0, 0, -0.20, 0, 0, 0, 0, -0.20, 0]),
    ("hips_in_knees_plus", [0, 0, -0.15, 0.20, 0, 0, 0, 0.15, 0.20, 0]),
    ("hips_out_knees_plus", [0, 0, 0.15, 0.20, 0, 0, 0, -0.15, 0.20, 0]),
    ("knees_plus_ankles_minus", [0, 0, 0, 0.20, -0.12, 0, 0, 0, 0.20, -0.12]),
    ("knees_plus_ankles_plus", [0, 0, 0, 0.20, 0.12, 0, 0, 0, 0.20, 0.12]),
    ("hips_in_full", [0, 0, -0.15, 0.20, -0.12, 0, 0, 0.15, 0.20, -0.12]),
)


def main(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = len(CANDIDATES)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.curriculum_push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_dof_properties = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    action_scale = env.cfg.control.action_scale
    actions = torch.tensor(
        [candidate[1] for candidate in CANDIDATES],
        dtype=torch.float,
        device=env.device,
    ) / action_scale
    actions = torch.clamp(actions, -1.0, 1.0)
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    height_sum = torch.zeros(env.num_envs, device=env.device)
    samples = 0
    total_steps = round(2.0 / env.dt)
    window_start = total_steps - round(0.5 / env.dt)
    for step in range(total_steps):
        _, _, _, dones, _ = env.step(actions)
        fallen |= dones.bool()
        if step >= window_start:
            height_sum += env.root_states[:, 2]
            samples += 1
    heights = height_sum / max(samples, 1)
    result = [
        {
            "candidate": name,
            "mean_height_m": float(heights[index].item()),
            "fell": bool(fallen[index].item()),
            "action": actions[index].cpu().tolist(),
        }
        for index, (name, _) in enumerate(CANDIDATES)
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(get_args())
