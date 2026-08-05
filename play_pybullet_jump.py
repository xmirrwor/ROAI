"""Replay a trained MiniDuck jump in the PyBullet GUI."""

import argparse
import json
from pathlib import Path

from pybullet_jump.jump_sim import PARAMETER_NAMES, MiniDuckJumpSim


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("checkpoints/pybullet_jump_best_v6.json"),
    )
    args = parser.parse_args()
    if not args.policy.is_absolute():
        args.policy = PROJECT_ROOT / args.policy
    data = json.loads(args.policy.read_text(encoding="utf-8"))
    fallbacks = {
        "tuck_hip": 0.0,
        "tuck_knee": -0.25,
        "tuck_ankle": 0.05,
        "landing_hip": 0.0,
        "landing_knee": -0.18,
        "landing_ankle": 0.03,
        "tuck_time": 0.09,
        "recovery_time": 0.28,
    }
    parameters = [
        data["parameters"].get(name, fallbacks.get(name))
        for name in PARAMETER_NAMES
    ]
    sim = MiniDuckJumpSim(gui=True)
    try:
        while True:
            sim.evaluate(parameters, realtime=True)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
