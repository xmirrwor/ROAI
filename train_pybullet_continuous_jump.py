"""Train repeated jumps from the preserved 400-iteration single-jump model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pybullet_jump.continuous_jump import (
    MiniDuckContinuousJumpSim,
    continuous_metrics_to_dict,
)
from pybullet_jump.jump_sim import (
    LOWER_BOUNDS,
    PARAMETER_NAMES,
    UPPER_BOUNDS,
    parameters_to_dict,
)


ROOT = Path(__file__).resolve().parent


def rooted(path: Path):
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("checkpoints/pybullet_jump_best_v6.json"),
    )
    parser.add_argument("--jumps", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/pybullet_continuous_jump_best.json"),
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.base = rooted(args.base)
    args.output = rooted(args.output)
    payload = json.loads(args.base.read_text(encoding="utf-8"))
    defaults = np.array(
        [-0.007, -0.076, 0.030, 0.232, -0.560, 0.185,
         0.00, -0.25, 0.05, -0.007, -0.18, 0.030,
         0.215, 0.125, 0.090, 0.28]
    )
    saved = payload["parameters"]
    parameters = np.array(
        [saved.get(name, defaults[index]) for index, name in enumerate(PARAMETER_NAMES)],
        dtype=float,
    )
    parameters = np.clip(parameters, LOWER_BOUNDS, UPPER_BOUNDS)
    inter_jump_pause = payload.get("continuous_parameters", {}).get(
        "inter_jump_pause", 0.25
    )
    search_parameters = np.append(parameters, inter_jump_pause)
    std_floor = np.array(
        [0.010, 0.008, 0.012, 0.012, 0.018, 0.014,
         0.010, 0.012, 0.012, 0.010, 0.014, 0.012,
         0.004, 0.003, 0.004, 0.008]
    )
    search_std = np.append(1.8 * std_floor, 0.05)
    search_lower = np.append(LOWER_BOUNDS, 0.10)
    search_upper = np.append(UPPER_BOUNDS, 0.60)
    rng = np.random.default_rng(args.seed)
    sim = MiniDuckContinuousJumpSim(gui=False)
    try:
        baseline, _ = sim.evaluate_continuous(
            search_parameters[:16],
            jumps=args.jumps,
            inter_jump_pause=search_parameters[16],
        )
        best = (baseline, search_parameters.copy())
        print(
            f"base={args.base} completed={baseline.completed_jumps}/{args.jumps} "
            f"min_peak={baseline.min_flight_peak_com:.4f}m "
            f"impact={baseline.max_landing_impact:.1f}N"
        )
        blocks = (
            np.array([9, 10, 11, 15]),
            np.array([0, 1, 2, 3, 4, 5, 12, 13]),
            np.array([6, 7, 8, 14]),
            np.array([9, 10, 11, 15, 16]),
            np.arange(len(search_parameters)),
        )
        stagnant = 0
        for iteration in range(args.iterations):
            samples = np.tile(best[1], (args.population, 1))
            for sample_index in range(1, args.population):
                block = blocks[(sample_index - 1) % len(blocks)]
                scale = 0.4 if len(block) == len(search_parameters) else 1.0
                samples[sample_index, block] += rng.normal(
                    0.0, scale * search_std[block]
                )
            samples = np.clip(samples, search_lower, search_upper)
            results = [
                (
                    sim.evaluate_continuous(
                        sample[:16],
                        jumps=args.jumps,
                        inter_jump_pause=sample[16],
                    )[0],
                    sample,
                )
                for sample in samples
            ]
            results.sort(key=lambda item: item[0].score, reverse=True)
            improved = results[0][0].score > best[0].score + 1.0e-9
            if improved:
                best = results[0]
                search_std = np.maximum(
                    search_std * 0.98,
                    np.append(std_floor, 0.015),
                )
                stagnant = 0
            else:
                stagnant += 1
            if stagnant >= 10:
                search_std = np.minimum(
                    search_std * 1.20,
                    0.10 * (search_upper - search_lower),
                )
                stagnant = 0
            metrics = best[0]
            print(
                f"iteration={iteration + 1:03d} "
                f"update={'yes' if improved else 'no'} "
                f"completed={metrics.completed_jumps}/{args.jumps} "
                f"flights={metrics.flight_count} landings={metrics.landing_count} "
                f"score={metrics.score:.3f} "
                f"min_peak={metrics.min_flight_peak_com:.4f}m "
                f"mean_peak={metrics.mean_flight_peak_com:.4f}m "
                f"min_foot={metrics.min_foot_clearance:.4f}m "
                f"drift={metrics.max_drift:.4f}m "
                f"impact={metrics.max_landing_impact:.1f}N "
                f"settled={metrics.all_settled}"
            )
            if (
                not args.no_save
                and args.checkpoint_every > 0
                and (iteration + 1) % args.checkpoint_every == 0
            ):
                checkpoint = args.output.with_name(
                    f"{args.output.stem}.checkpoint{args.output.suffix}"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(
                    json.dumps(
                        {
                            "format": "miniduck-pybullet-continuous-jump-v1",
                            "iteration": iteration + 1,
                            "base_model": str(args.base),
                            "parameters": parameters_to_dict(best[1][:16]),
                            "continuous_parameters": {
                                "inter_jump_pause": float(best[1][16])
                            },
                            "metrics": continuous_metrics_to_dict(best[0]),
                            "frames": [],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"checkpoint={checkpoint}")
        metrics, frames = sim.evaluate_continuous(
            best[1][:16],
            jumps=args.jumps,
            inter_jump_pause=best[1][16],
            capture=True,
        )
    finally:
        sim.close()

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "format": "miniduck-pybullet-continuous-jump-v1",
                    "seed": args.seed,
                    "base_model": str(args.base),
                    "parameters": parameters_to_dict(best[1][:16]),
                    "continuous_parameters": {
                        "inter_jump_pause": float(best[1][16])
                    },
                    "metrics": continuous_metrics_to_dict(metrics),
                    "frames": frames,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
