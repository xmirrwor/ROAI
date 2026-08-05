# MiniDuck 2 m Race Optimization

This experiment keeps the existing 64-observation, 10-action policy contract and
fine-tunes the stable `model_12000.pt` checkpoint. The race is complete when the
robot covers a 2 m straight chord. Lower elapsed time is better, but a candidate
is rejected if it trades speed for falls, curvature, or heading instability.

## Motion selection

All three motion types were evaluated with 64 deterministic environments and the
same 2 m criterion.

| Motion | Command | Completed | Falls | Mean time | Mean speed | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Forward | 0.16 m/s | 59/64 | 5 | 12.95 s | 0.155 m/s | No |
| Diagonal | 0.156 m/s | 62/64 | 2 | 10.51 s | 0.190 m/s | No |
| Lateral | 0.26 m/s | 63/64 | 1 | 6.92 s | 0.289 m/s | Yes |

Positive lateral motion is selected. Negative lateral motion was also checked at
0.26 m/s: it took 7.53 s, so it is not used for the race.

## Training design

Set `MINIDUCK_RACE_MOTION=lateral` to enable the race-only behavior:

- sample only lateral commands in the configured speed range;
- keep one heading reference for the full episode;
- reward signed lateral speed directly;
- retain balance, orientation, slip, torque, and action-rate constraints;
- disable forward-line and active-yaw rewards that conflict with lateral motion.

The selected run uses only the faster positive direction, 4096 environments,
learning rate `1e-7`, and 20 additional PPO iterations. Checkpoint evaluation
showed that the first update was best; later updates were slower.

## Selected result

The repository checkpoint is:

```text
checkpoints/miniduck_race_lateral_2m_model_12000.pt
```

Its deterministic 64-trial result at a 0.26 m/s command is:

- completion: 63/64 (98.44%);
- falls: 1/64 (1.56%);
- mean 2 m time: 6.9076 s;
- mean effective speed: 0.2896 m/s;
- mean chord deviation: 0.0333 m;
- fifth-percentile path efficiency: 0.9200;
- all acceptance checks passed.

The original stable checkpoint took 6.9203 s under the same command and test.
The improvement is small but measured under identical deterministic conditions.

A 64-trial randomized robustness check completed 61 trials with 3 falls and a
7.1948 s mean time. Its completion and fall-rate gates passed, but its fifth-
percentile path efficiency and heading-error gates did not. The selected model
is therefore suitable for the current deterministic simulation race; it is not
evidence of zero-fall operation under every dynamics perturbation.

## Reproduce training

Run from the repository root on the configured Isaac Gym server:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate miniduck_cu118
ISAAC_BINDINGS="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"

export MINIDUCK_RACE_MOTION=lateral
export MINIDUCK_RACE_LATERAL_MIN_SPEED=0.26
export MINIDUCK_RACE_LATERAL_MAX_SPEED=0.32
export MINIDUCK_RACE_LATERAL_TARGET_SPEED=0.34
export MINIDUCK_RACE_POSITIVE_DIRECTION_PROB=1.0
export MINIDUCK_LEARNING_RATE=1e-7

python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs 4096 \
  --max_iterations 20 \
  --resume \
  --load_run Jul17_04-41-06_miniduck_stable_compressed_curriculum_resume_to_12000_from_8850 \
  --checkpoint 12000 \
  --run_name race_lateral_2m_speed_v3_positive
```

`--max_iterations` is the number of additional iterations in this runner, not an
absolute target iteration.

## Reproduce evaluation

Unset the training-only race mode before evaluation:

```bash
unset MINIDUCK_RACE_MOTION
python legged_panguin/scripts/evaluate_miniduck_race.py \
  --task miniduck_flat \
  --headless \
  --num_envs 64 \
  --motion lateral \
  --command_speed_mps 0.26 \
  --direction_sign 1 \
  --distance_m 2 \
  --timeout_s 20 \
  --checkpoint_path checkpoints/miniduck_race_lateral_2m_model_12000.pt \
  --run_label selected_race_model
```

The evaluator writes a JSON report under `evaluation/race_2m/`.
