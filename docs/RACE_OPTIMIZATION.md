# MiniDuck 2 m Straight Race Optimization

This experiment keeps the existing 64-observation, 10-action policy contract and
fine-tunes the selected lateral-race checkpoint. The final criterion is not net
distance: the robot must advance 2 m along a fixed race axis while remaining
within 0.10 m of the center line.

## Motion selection

The initial speed comparison used 64 deterministic environments:

| Motion | Command | Mean time | Mean speed |
| --- | ---: | ---: | ---: |
| Forward | 0.16 m/s | 12.95 s | 0.155 m/s |
| Diagonal | 0.156 m/s | 10.51 s | 0.190 m/s |
| Lateral | 0.26 m/s | 6.92 s | 0.289 m/s |

Positive lateral motion remains the fastest candidate.

## Rejected evaluation rule

The first evaluator declared completion when net displacement reached 2 m and
measured deviation from the chord joining the observed start and end points.
That rule can label a curved trajectory as straight. The rejected model had a
small 0.03 m chord deviation but about 0.36 m fixed-axis deviation in the short
test, and the acceptance video visibly curved away from the race center line.

The corrected evaluator now:

- declares completion only when fixed-axis progress reaches 2 m;
- requires the finish point to be within 0.10 m of the center line;
- measures maximum fixed-axis cross-track error over the complete trajectory;
- accepts only when its 95th percentile is at most 0.10 m;
- retains completion, fall-rate, path-efficiency, and heading checks.

Without a line controller, the old policy completed 0/64 under this corrected
rule and reached about 1.10 m maximum cross-track error.

## Straight-line training

Race mode adds a closed-loop sagittal correction while lateral speed remains
the primary command. The correction uses displacement and velocity relative to
the initial body-forward axis:

```text
vx_correction = clamp(-kp * cross_track - kd * cross_track_velocity)
```

The selected values are `kp=1.00`, `kd=0.25`, and a `0.12 m/s` correction limit.
Training also adds a fixed-center-line error penalty, while retaining speed,
orientation, slip, torque, action-rate, and termination terms.

The v4 run uses 4096 environments, learning rate `1e-7`, positive lateral
commands in `[0.24, 0.28] m/s`, and 40 additional PPO iterations. Checkpoints
were evaluated independently; `model_12040.pt` was selected.

## Selected result

```text
checkpoints/miniduck_race_lateral_2m_straight_model_12040.pt
```

Deterministic 64-trial evaluation at a 0.26 m/s lateral command:

- completion: 61/64 (95.31%);
- falls: 3/64 (4.69%);
- mean 2 m time: 7.0082 s;
- mean effective speed: 0.2855 m/s;
- mean maximum fixed-line deviation: 0.0605 m;
- 95th-percentile maximum fixed-line deviation: 0.0839 m;
- fifth-percentile path efficiency: 0.9013;
- 95th-percentile heading error: 8.97 degrees;
- all corrected acceptance checks passed.

A randomized 64-trial test completed 60 trials with 4 falls and 0.1297 m
95th-percentile fixed-line deviation. It does not pass the strict randomized
robustness standard; the selected result is for the deterministic simulation
race used by the current visual acceptance workflow.

## Reproduce training

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate miniduck_cu118
ISAAC_BINDINGS="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"

export MINIDUCK_RACE_MOTION=lateral
export MINIDUCK_RACE_LATERAL_MIN_SPEED=0.24
export MINIDUCK_RACE_LATERAL_MAX_SPEED=0.28
export MINIDUCK_RACE_LATERAL_TARGET_SPEED=0.32
export MINIDUCK_RACE_POSITIVE_DIRECTION_PROB=1.0
export MINIDUCK_RACE_LINE_HOLD_KP=1.00
export MINIDUCK_RACE_LINE_HOLD_KD=0.25
export MINIDUCK_RACE_LINE_HOLD_MAX_SAGITTAL_MPS=0.12
export MINIDUCK_LEARNING_RATE=1e-7

python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs 4096 \
  --max_iterations 40 \
  --resume \
  --load_run Aug05_18-05-22_race_lateral_2m_speed_v3_positive \
  --checkpoint 12000 \
  --run_name race_lateral_2m_straight_v4
```

`--max_iterations` is the number of additional iterations.

## Reproduce evaluation

```bash
unset MINIDUCK_RACE_MOTION
python legged_panguin/scripts/evaluate_miniduck_race.py \
  --task miniduck_flat \
  --headless \
  --num_envs 64 \
  --motion lateral \
  --command_speed_mps 0.26 \
  --distance_m 2 \
  --max_cross_track_m 0.10 \
  --timeout_s 20 \
  --checkpoint_path checkpoints/miniduck_race_lateral_2m_straight_model_12040.pt \
  --run_label selected_straight_race
```
