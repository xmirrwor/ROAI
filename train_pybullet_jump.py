"""Train a parameterized MiniDuck jump with cross-entropy search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from pybullet_jump.jump_sim import (
    LOWER_BOUNDS,
    PARAMETER_NAMES,
    UPPER_BOUNDS,
    MiniDuckJumpSim,
    metrics_to_dict,
    parameters_to_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path: Optional[Path]):
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def jump_height_objective(metrics):
    return metrics.peak_flight_com_height


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("logs/pybullet_jump/best_jump.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--height-step", type=float, default=0.003)
    parser.add_argument("--landing-target-force", type=float, default=220.0)
    parser.add_argument("--landing-height-budget", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--physics-substeps", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--contact-erp", type=float)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.resume = project_path(args.resume)
    args.output = project_path(args.output)
    rng = np.random.default_rng(args.seed)
    mean = np.array(
        [-0.007, -0.076, 0.030, 0.232, -0.560, 0.185,
         0.00, -0.25, 0.05, -0.007, -0.18, 0.030,
         0.215, 0.125, 0.090, 0.28]
    )
    std = np.array(
        [0.08, 0.06, 0.10, 0.10, 0.16, 0.12,
         0.08, 0.12, 0.10, 0.07, 0.10, 0.09,
         0.025, 0.015, 0.020, 0.05]
    )
    std_floor = np.array(
        [0.025, 0.020, 0.030, 0.030, 0.050, 0.040,
         0.025, 0.035, 0.030, 0.025, 0.035, 0.030,
         0.010, 0.008, 0.010, 0.020]
    )
    resume_mode = args.resume is not None
    if resume_mode:
        payload = json.loads(args.resume.read_text(encoding="utf-8"))
        saved = payload["parameters"]
        defaults = dict(zip(PARAMETER_NAMES, mean))
        defaults.update(saved)
        mean = np.array([defaults[name] for name in PARAMETER_NAMES], dtype=float)
        mean = np.clip(mean, LOWER_BOUNDS, UPPER_BOUNDS)
        # Resume locally around a proven controller instead of restarting the
        # broad curriculum search.
        std = np.maximum(0.35 * std, std_floor)
    elite_count = max(2, int(args.population * args.elite_fraction))
    best = None
    stage_seed = mean.copy()
    previous_phase = -1
    stagnant_iterations = 0
    optimization_mode = None
    landing_anchor = None
    sim = MiniDuckJumpSim(
        gui=False,
        physics_substeps=args.physics_substeps,
        solver_iterations=args.solver_iterations,
        contact_erp=args.contact_erp,
    )
    try:
        if resume_mode:
            baseline_metrics = sim.evaluate(
                mean,
                curriculum_stage=2,
                target_ballistic_rise=max(
                    0.010, float(payload.get("metrics", {}).get("ballistic_rise", 0.0))
                ),
            )[0]
            if not baseline_metrics.settled:
                raise RuntimeError(
                    "Resume model failed settled-landing validation; use a JSON "
                    "whose metrics replay with settled=true"
                )
            best = (baseline_metrics, mean.copy())
            stage_seed = mean.copy()
            optimization_mode = (
                "landing"
                if baseline_metrics.landing_impact
                > args.landing_target_force + 20.0
                else "height"
            )
            if optimization_mode == "landing":
                landing_anchor = baseline_metrics
            print(
                "resume_validated "
                f"objective={jump_height_objective(baseline_metrics):.5f} "
                f"ballistic={baseline_metrics.ballistic_rise:.4f}m "
                f"foot_clearance={baseline_metrics.max_foot_clearance:.4f}m "
                f"peak_base={baseline_metrics.peak_base_height:.4f}m "
                f"settled={baseline_metrics.settled}"
            )
        for iteration in range(args.iterations):
            evaluated_mode = optimization_mode
            progress = iteration / max(args.iterations, 1)
            if resume_mode:
                phase, stage = 0, 2
                target_height = max(0.010, best[0].ballistic_rise + args.height_step)
            elif progress < 0.20:
                phase, stage, target_height = 0, 0, 0.020
            elif progress < 0.40:
                phase, stage, target_height = 1, 1, 0.020
            elif progress < 0.60:
                phase, stage, target_height = 2, 1, 0.050
            elif progress < 0.80:
                phase, stage, target_height = 3, 2, 0.050
            else:
                phase, stage, target_height = 4, 2, 0.080
            if not resume_mode and phase != previous_phase:
                if best is not None:
                    stage_seed = best[1].copy()
                # Re-score and retain the previous phase champion. Resetting it
                # made the displayed result regress at curriculum boundaries.
                seed_metrics = sim.evaluate(
                    stage_seed,
                    curriculum_stage=stage,
                    target_ballistic_rise=target_height,
                )[0]
                best = (seed_metrics, stage_seed.copy())
                stagnant_iterations = 0
                previous_phase = phase
            sample_center = best[1] if resume_mode else mean
            if resume_mode:
                samples = np.tile(sample_center, (args.population, 1))
                takeoff_block = np.array([0, 1, 2, 3, 4, 5, 12, 13])
                tuck_block = np.array([6, 7, 8, 14])
                landing_block = np.array([9, 10, 11, 15])
                coupled_block = np.arange(len(mean))
                if optimization_mode == "landing":
                    parameter_blocks = (
                        landing_block,
                        landing_block,
                        landing_block,
                        coupled_block,
                        tuck_block,
                    )
                else:
                    parameter_blocks = (
                        takeoff_block,
                        takeoff_block,
                        tuck_block,
                        landing_block,
                        coupled_block,
                    )
                for sample_index in range(1, args.population):
                    block_index = (sample_index - 1) % len(parameter_blocks)
                    block = parameter_blocks[block_index]
                    scale = 0.45 if block_index == 3 else 1.0
                    samples[sample_index, block] += rng.normal(
                        0.0, scale * std[block]
                    )
                samples = np.clip(samples, LOWER_BOUNDS, UPPER_BOUNDS)
            else:
                samples = np.clip(
                    rng.normal(sample_center, std, size=(args.population, len(mean))),
                    LOWER_BOUNDS,
                    UPPER_BOUNDS,
                )
            random_count = 0 if resume_mode else max(1, args.population // 6)
            if random_count:
                samples[-random_count:] = rng.uniform(
                    LOWER_BOUNDS,
                    UPPER_BOUNDS,
                    size=(random_count, len(mean)),
                )
            samples[0] = sample_center
            if args.population > 1:
                samples[1] = stage_seed
            if stage < 2:
                # Landing pose and recovery are optimized only after a useful
                # takeoff/flight trajectory exists.
                samples[:, 9:12] = mean[9:12]
                samples[:, 15] = mean[15]
            results = [
                (
                    sim.evaluate(
                        sample,
                        curriculum_stage=stage,
                        target_ballistic_rise=target_height,
                    )[0],
                    sample,
                )
                for sample in samples
            ]
            results.sort(key=lambda item: item[0].score, reverse=True)
            if resume_mode:
                max_next_height = (
                    best[0].peak_flight_com_height + args.height_step
                )
                if optimization_mode == "landing":
                    anchor_objective = jump_height_objective(landing_anchor)
                    stable_results = [
                        item for item in results
                        if item[0].settled
                        and jump_height_objective(item[0])
                            >= anchor_objective - args.landing_height_budget
                        and item[0].peak_flight_com_height
                            >= landing_anchor.peak_flight_com_height
                                - args.landing_height_budget
                        and item[0].horizontal_drift
                            < 0.035
                        and item[0].post_landing_speed < 0.22
                    ]
                    candidate = min(
                        stable_results,
                        key=lambda item: (
                            item[0].landing_impact,
                            item[0].post_landing_speed,
                            item[0].horizontal_drift,
                            -jump_height_objective(item[0]),
                        ),
                        default=best,
                    )
                    candidate_key = (
                        -candidate[0].landing_impact,
                        -candidate[0].post_landing_speed,
                        -candidate[0].horizontal_drift,
                        jump_height_objective(candidate[0]),
                    )
                    best_key = (
                        -best[0].landing_impact,
                        -best[0].post_landing_speed,
                        -best[0].horizontal_drift,
                        jump_height_objective(best[0]),
                    )
                else:
                    stable_results = [
                        item for item in results
                        if item[0].settled
                        and item[0].peak_flight_com_height
                            <= max_next_height + 1.0e-9
                        and item[0].peak_flight_com_height
                            >= best[0].peak_flight_com_height
                        and item[0].horizontal_drift < 0.035
                        and item[0].post_landing_speed < 0.22
                    ]
                    candidate = max(
                        stable_results,
                        key=lambda item: (
                            item[0].peak_flight_com_height,
                            item[0].ballistic_rise,
                            jump_height_objective(item[0]),
                            item[0].max_foot_clearance,
                            -item[0].landing_impact,
                        ),
                        default=best,
                    )
                    candidate_key = (
                        candidate[0].peak_flight_com_height,
                        candidate[0].ballistic_rise,
                        jump_height_objective(candidate[0]),
                        candidate[0].max_foot_clearance,
                        -candidate[0].landing_impact,
                    )
                    best_key = (
                        best[0].peak_flight_com_height,
                        best[0].ballistic_rise,
                        jump_height_objective(best[0]),
                        best[0].max_foot_clearance,
                        -best[0].landing_impact,
                    )
                improved = candidate_key > best_key
                if improved:
                    best = candidate
                    mean = best[1].copy()
                    std = np.minimum(std * 1.03, 0.20 * (UPPER_BOUNDS - LOWER_BOUNDS))
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
                    std = np.maximum(std * 0.98, std_floor)
                if (
                    optimization_mode == "landing"
                    and best[0].landing_impact <= args.landing_target_force
                ):
                    optimization_mode = "height"
                    landing_anchor = None
                    print("mode_switch=height")
                elif (
                    optimization_mode == "height"
                    and best[0].landing_impact > args.landing_target_force + 20.0
                ):
                    optimization_mode = "landing"
                    landing_anchor = best[0]
                    print("mode_switch=landing")
            else:
                elites = np.stack([item[1] for item in results[:elite_count]])
                mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
                std = np.maximum(0.25 * std + 0.75 * elites.std(axis=0), std_floor)
                improved = best is None or results[0][0].score > best[0].score
                if improved:
                    best = results[0]
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
            if improved:
                stagnant_iterations = 0
            if stagnant_iterations >= 8:
                if resume_mode:
                    std = np.minimum(
                        np.maximum(1.15 * std, std_floor),
                        0.12 * (UPPER_BOUNDS - LOWER_BOUNDS),
                    )
                else:
                    std = np.maximum(std, 0.12 * (UPPER_BOUNDS - LOWER_BOUNDS))
                stagnant_iterations = 0
            metrics = best[0]
            print(
                f"iteration={iteration + 1:03d} stage={stage + 1} "
                f"mode={evaluated_mode or 'curriculum'} "
                f"update={'yes' if improved else 'no'} "
                f"target={target_height:.3f}m "
                f"score={metrics.score:7.3f} "
                f"objective={jump_height_objective(metrics):.5f} "
                f"rise={metrics.com_rise:.4f}m air={metrics.airborne_time:.3f}s "
                f"takeoff={metrics.takeoff_velocity:.3f}m/s "
                f"ballistic={metrics.ballistic_rise:.4f}m "
                f"peak_base={metrics.peak_base_height:.4f}m "
                f"peak_com={metrics.peak_com_height:.4f}m "
                f"flight_peak_com={metrics.peak_flight_com_height:.4f}m "
                f"foot_clearance={metrics.max_foot_clearance:.4f}m "
                f"drift={metrics.horizontal_drift:.4f}m "
                f"impact={metrics.landing_impact:.1f}N "
                f"post_speed={metrics.post_landing_speed:.3f}m/s "
                f"support={metrics.final_support_force:.1f}N "
                f"body_hit={metrics.body_contact_after_takeoff} "
                f"landed={metrics.landed} settled={metrics.settled}"
            )
            if (
                resume_mode
                and not args.no_save
                and args.checkpoint_every > 0
                and (iteration + 1) % args.checkpoint_every == 0
            ):
                checkpoint = args.output.with_name(
                    f"{args.output.stem}.checkpoint{args.output.suffix}"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_payload = json.dumps(
                    {
                        "format": "miniduck-pybullet-jump-v8-checkpoint",
                        "iteration": iteration + 1,
                        "seed": args.seed,
                        "physics": sim.physics_config,
                        "parameters": parameters_to_dict(best[1]),
                        "metrics": metrics_to_dict(best[0]),
                        "frames": [],
                    },
                    indent=2,
                )
                checkpoint.write_text(checkpoint_payload, encoding="utf-8")
                history_checkpoint = args.output.with_name(
                    f"{args.output.stem}.iter{iteration + 1:04d}{args.output.suffix}"
                )
                history_checkpoint.write_text(
                    checkpoint_payload,
                    encoding="utf-8",
                )
                print(f"checkpoint={checkpoint.resolve()}")
        metrics, frames = sim.evaluate(
            best[1],
            capture=True,
            curriculum_stage=2,
            target_ballistic_rise=0.080,
        )
    finally:
        sim.close()
    payload = {
        "format": "miniduck-pybullet-jump-v8",
        "seed": args.seed,
        "physics": sim.physics_config,
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
