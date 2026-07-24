import argparse
import os

import torch

from rsl_rl.modules import ActorCritic


OLD_TO_NEW_OBSERVATION_SLICES = (
    (slice(3, 6), slice(0, 3)),    # gyro
    (slice(6, 9), slice(3, 6)),    # projected gravity
    (slice(9, 12), slice(9, 12)),  # command
    (slice(12, 22), slice(12, 22)),
    (slice(22, 32), slice(22, 32)),
    (slice(32, 42), slice(32, 42)),
)


def convert_checkpoint(source, destination, exploration_std):
    checkpoint = torch.load(source, map_location="cpu")
    old_state = checkpoint["model_state_dict"]

    actor_critic = ActorCritic(
        62,
        62,
        10,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        init_noise_std=exploration_std,
    )
    new_state = actor_critic.state_dict()

    for network in ("actor", "critic"):
        old_weight = old_state[f"{network}.0.weight"]
        new_weight = torch.zeros_like(new_state[f"{network}.0.weight"])
        for old_slice, new_slice in OLD_TO_NEW_OBSERVATION_SLICES:
            new_weight[:, new_slice] = old_weight[:, old_slice]
        new_state[f"{network}.0.weight"] = new_weight

    for key, value in old_state.items():
        if key not in ("std", "actor.0.weight", "critic.0.weight"):
            new_state[key] = value
    new_state["std"] = torch.full_like(new_state["std"], exploration_std)
    actor_critic.load_state_dict(new_state)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3.0e-4)
    converted = {
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 0,
        "infos": {
            "source_checkpoint": os.path.abspath(source),
            "observation_layout": "miniduck_s2r_history_62d",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    torch.save(converted, destination)
    print(f"Converted {source} -> {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--exploration-std", type=float, default=0.08)
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination, args.exploration_std)


if __name__ == "__main__":
    main()
