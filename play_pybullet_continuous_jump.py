"""Replay a continuous jump controller in PyBullet GUI."""

import argparse
import json
from pathlib import Path

from pybullet_jump.continuous_jump import MiniDuckContinuousJumpSim
from pybullet_jump.jump_sim import PARAMETER_NAMES


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("checkpoints/pybullet_continuous_jump_best.json"),
    )
    parser.add_argument("--jumps", type=int, default=3)
    args = parser.parse_args()
    policy = args.policy if args.policy.is_absolute() else ROOT / args.policy
    data = json.loads(policy.read_text(encoding="utf-8"))
    parameters = [data["parameters"][name] for name in PARAMETER_NAMES]
    inter_jump_pause = data.get("continuous_parameters", {}).get(
        "inter_jump_pause", 0.25
    )
    sim = MiniDuckContinuousJumpSim(gui=True)
    try:
        while True:
            metrics, _ = sim.evaluate_continuous(
                parameters,
                jumps=args.jumps,
                inter_jump_pause=inter_jump_pause,
                realtime=True,
            )
            print(
                f"completed={metrics.completed_jumps}/{args.jumps} "
                f"min_peak={metrics.min_flight_peak_com:.4f}m "
                f"settled={metrics.all_settled}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
