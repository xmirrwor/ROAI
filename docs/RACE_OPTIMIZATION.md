# MiniDuck 2 m Straight Race Optimization

This experiment keeps the existing 64-observation, 10-action policy contract and
fine-tunes the selected lateral-race checkpoint. The final criterion is not net
distance: the robot must advance 2 m along a fixed race axis while keeping its
average, terminal, and peak cross-track errors close to the center line.

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
- requires the finish point to be within 0.04 m of the center line for 95% of trials;
- measures maximum fixed-axis cross-track error over the complete trajectory;
- requires the 95th-percentile mean and peak errors to stay below 0.03 m and
  0.07 m, respectively;
- retains completion, fall-rate, path-efficiency, and heading checks.

Without a line controller, the old policy completed 0/64 under this corrected
rule and reached about 1.10 m maximum cross-track error.

## Straight-line training

Race mode adds a closed-loop sagittal correction while lateral speed remains
the primary command. The correction uses displacement and velocity relative to
the initial body-forward axis:

```text
vx_correction = clamp(-kp * cross_track - kd * cross_track_velocity
                      -ki * integrated_cross_track)
```

The final controller uses `kp=1.10`, `kd=0.25`, `ki=0.25`, an integral limit of
`0.08 m*s`, and a `0.12 m/s` correction limit. The small integral term removes
the consistent one-sided offset without the extra falls observed at higher
integral gains. Training adds a centered-speed reward and a tighter fixed-line
error penalty while retaining speed, orientation, slip, torque, action-rate,
and termination terms.

The v5 run uses 4096 environments, learning rate `1e-7`, positive lateral
commands in `[0.24, 0.30] m/s`, and 120 additional PPO iterations. Checkpoints
were screened independently; `model_12060.pt` was selected.

## Selected result

```text
checkpoints/miniduck_race_lateral_2m_centerline_model_12060.pt
```

Deterministic 64-trial evaluation at a 0.26 m/s lateral command:

- completion: 61/64 (95.31%);
- falls: 3/64 (4.69%);
- mean 2 m time: 7.0230 s;
- mean effective speed: 0.2849 m/s;
- 95th-percentile mean absolute cross-track error: 0.0234 m;
- 95th-percentile finish cross-track error: 0.0222 m;
- 95th-percentile maximum fixed-line deviation: 0.0680 m;
- fifth-percentile path efficiency: 0.9060;
- 95th-percentile heading error: 9.28 degrees;
- all corrected acceptance checks passed.

The selected result is for the deterministic simulation race used by the
current visual acceptance workflow. Randomized robustness is evaluated
separately and is not used to replace the fastest passing deterministic model.

## Reproduce training

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate miniduck_cu118
ISAAC_BINDINGS="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"

export MINIDUCK_RACE_MOTION=lateral
export MINIDUCK_RACE_LATERAL_MIN_SPEED=0.24
export MINIDUCK_RACE_LATERAL_MAX_SPEED=0.30
export MINIDUCK_RACE_LATERAL_TARGET_SPEED=0.34
export MINIDUCK_RACE_POSITIVE_DIRECTION_PROB=1.0
export MINIDUCK_RACE_LINE_HOLD_KP=1.00
export MINIDUCK_RACE_LINE_HOLD_KD=0.25
export MINIDUCK_RACE_LINE_HOLD_KI=0.30
export MINIDUCK_RACE_LINE_HOLD_INTEGRAL_LIMIT=0.08
export MINIDUCK_RACE_LINE_HOLD_MAX_SAGITTAL_MPS=0.12
export MINIDUCK_RACE_CENTERLINE_WIDTH_M=0.03
export MINIDUCK_RACE_CENTERED_SPEED_SCALE=10.0
export MINIDUCK_RACE_PATH_ERROR_SCALE=-12.0
export MINIDUCK_LEARNING_RATE=1e-7

python legged_panguin/scripts/train.py \
  --task miniduck_flat \
  --headless \
  --num_envs 4096 \
  --max_iterations 120 \
  --resume \
  --load_run Aug05_19-08-02_race_lateral_2m_straight_v4 \
  --checkpoint 12040 \
  --run_name race_lateral_2m_centerline_v5
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
  --max_cross_track_m 0.07 \
  --max_mean_abs_cross_track_m 0.03 \
  --max_final_cross_track_m 0.04 \
  --timeout_s 20 \
  --checkpoint_path checkpoints/miniduck_race_lateral_2m_centerline_model_12060.pt \
  --run_label selected_centerline_race
```
