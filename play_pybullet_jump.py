"""Replay a trained MiniDuck jump in the PyBullet GUI."""

import argparse
import json
from pathlib import Path

from pybullet_jump.jump_sim import PARAMETER_NAMES, MiniDuckJumpSim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=Path("logs/pybullet_jump/best_jump.json"))
    args = parser.parse_args()
    data = json.loads(args.policy.read_text(encoding="utf-8"))
    fallbacks = {
        "tuck_hip": 0.0,
        "tuck_knee": -0.25,
        "tuck_ankle": 0.05,
        "tuck_time": 0.09,
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
