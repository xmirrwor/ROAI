# Stage 8 obstacle prototype and full sequence

## Scope

User stage 8 maps to internal curriculum stage 5 (`obstacle_crossing`). The
delivered skill is a prototype for fixed, low transverse obstacles. It is not a
claim of general obstacle perception or transfer to unseen obstacle geometry.

The full demonstration is a multi-policy sequence. It loads the selected
stage 5, 6, 7, and 8 experts and linearly blends target actions for 0.35 s when
the active expert changes. Controlled resets are used for the fall pose, the
post-recovery standing boundary, and obstacle placement.

## Environment and observation

- Terrain: one transverse ridge per tile, 0.06 m deep and 0.70 m wide.
- Heights: four deterministic groups from 0.012 m to 0.024 m.
- Approach: the robot starts 0.45 m before the obstacle with a 0.03 m spawn
  height offset required by Isaac Gym heightfield contact handling.
- Observation contract remains 64 floats and action contract remains 10
  targets.
- Observation indices 62-63 retain `[cos(phase), sin(phase)]` during obstacle
  motion. Index 10 carries `clip(distance_m / 0.80, -1, 1)` only while the
  obstacle skill is active.

Reward shaping adds forward progress, foot clearance near the ridge, crossing
success, upright stability, heading error, lateral error, and body collision.
Random external pushes are disabled for this prototype; motor, friction, and
sensor randomization remain available.

## Selected checkpoint

`checkpoints/miniduck_stage8_obstacle_model_selected.pt` is copied from
`Jul31_19-27-28_stage8_obstacle_distancephase_512_continue/model_12370.pt`.
ONNX and JSON metadata exports use the same basename.

The selected checkpoint was evaluated with raw per-step trajectories and raw
per-trial outcomes. No smoothing or failed-trial deletion was used.

| Evaluation | Trials | Success | Falls | Body collision | Mean progress | Mean success time |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic | 32 | 81.25% | 0% | 0% | 0.673 m | 4.011 s |
| Randomized | 32 | 71.875% | 0% | 0% | 0.715 m | 4.026 s |

The acceptance gate is success >= 25%, falls <= 50%, and mean progress >=
0.30 m. Both evaluations pass. A single viewer trial can still fail; the full
sequence records that result and continues to the final controlled stand.

## Training limitation

Long GPU-PhysX heightfield runs produced asynchronous CUDA illegal-memory
errors after short segments. The selected result combines the valid saved
segments through iteration 12370. The crashes are retained in the server logs;
they were not treated as successful training. Further refinement should first
replace or isolate this Isaac Gym heightfield backend issue.

## Full visual sequence

The viewer loops through:

1. forward, 2 s emergency stop, backward, 2 s emergency stop;
2. squat/stand and walk/stop/squat/stand action switching;
3. scripted fall pose and learned recovery;
4. forward-left, forward-right, backward-left, backward-right;
5. obstacle placement, obstacle crossing attempt, and final stand.

Run on the server desktop:

```bash
cd /root/miniducktraining/ROAI
export LD_LIBRARY_PATH=/root/miniducktraining/ROAI/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:/opt/conda/envs/miniduck_cu118/lib:/usr/local/cuda/lib64
export MINIDUCK_FORCED_STAGE=5
export DISPLAY=:20
/opt/conda/envs/miniduck_cu118/bin/python \
  legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat --demo full_sequence
```

The no-viewer smoke test completed every named phase, including all four
diagonal directions, obstacle crossing, and `finish_stand`.
