"""Train a parameterized MiniDuck jump with cross-entropy search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pybullet_jump.jump_sim import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    MiniDuckJumpSim,
    metrics_to_dict,
    parameters_to_dict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("logs/pybullet_jump/best_jump.json"))
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mean = np.array(
        [-0.007, -0.076, 0.030, 0.232, -0.560, 0.185,
         0.00, -0.25, 0.05, 0.215, 0.125, 0.090]
    )
    std = np.array(
        [0.08, 0.06, 0.10, 0.10, 0.16, 0.12,
         0.08, 0.12, 0.10, 0.025, 0.015, 0.020]
    )
    std_floor = np.array(
        [0.025, 0.020, 0.030, 0.030, 0.050, 0.040,
         0.025, 0.035, 0.030, 0.010, 0.008, 0.010]
    )
    elite_count = max(2, int(args.population * args.elite_fraction))
    best = None
    stage_seed = mean.copy()
    previous_stage = -1
    stagnant_iterations = 0
    sim = MiniDuckJumpSim(gui=False)
    try:
        for iteration in range(args.iterations):
            stage = min(2, int(3 * iteration / max(args.iterations, 1)))
            if stage != previous_stage:
                if best is not None:
                    stage_seed = best[1].copy()
                best = None
                stagnant_iterations = 0
                previous_stage = stage
            samples = np.clip(
                rng.normal(mean, std, size=(args.population, len(mean))),
                LOWER_BOUNDS,
                UPPER_BOUNDS,
            )
            random_count = max(1, args.population // 6)
            samples[-random_count:] = rng.uniform(
                LOWER_BOUNDS,
                UPPER_BOUNDS,
                size=(random_count, len(mean)),
            )
            samples[0] = mean
            if args.population > 1:
                samples[1] = stage_seed
            results = [
                (sim.evaluate(sample, curriculum_stage=stage)[0], sample)
                for sample in samples
            ]
            results.sort(key=lambda item: item[0].score, reverse=True)
            elites = np.stack([item[1] for item in results[:elite_count]])
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            std = np.maximum(0.25 * std + 0.75 * elites.std(axis=0), std_floor)
            if best is None or results[0][0].score > best[0].score:
                best = results[0]
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
            if stagnant_iterations >= 8:
                std = np.maximum(std, 0.12 * (UPPER_BOUNDS - LOWER_BOUNDS))
                stagnant_iterations = 0
            metrics = best[0]
            print(
                f"iteration={iteration + 1:03d} stage={stage + 1} "
                f"score={metrics.score:7.3f} "
                f"rise={metrics.com_rise:.4f}m air={metrics.airborne_time:.3f}s "
                f"takeoff={metrics.takeoff_velocity:.3f}m/s "
                f"ballistic={metrics.ballistic_rise:.4f}m "
                f"drift={metrics.horizontal_drift:.4f}m landed={metrics.landed}"
            )
        metrics, frames = sim.evaluate(best[1], capture=True, curriculum_stage=2)
    finally:
        sim.close()
    payload = {
        "format": "miniduck-pybullet-jump-v2",
        "seed": args.seed,
        "parameters": parameters_to_dict(best[1]),
        "metrics": metrics_to_dict(metrics),
        "frames": frames,
    }
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
