# MiniDuck Kick Approach and Goal Report

## Result

The complete demonstration keeps the ball below the ground in every non-kick
phase. The kick module starts with a stationary ball 0.45 m in front of the
robot, uses the verified stage-7 straight-walk expert to approach it, and hands
off to the stage-10 kick expert before contact. A red 0.26 m wide goal is drawn
only during the approach and kick phases. Goal success requires the ball center
to cross a line 0.15 m beyond its fixed start while remaining inside the goal.

The policy interface remains 64 observations and 10 actions.

## Training and selection

Training resumed from `model_12620.pt`. A wider-spawn curriculum at 0.05 m and
0.10 m was rejected because it reduced contact reliability. A 120-iteration
goal-alignment fine-tune was also screened at iterations 12630, 12650, 12700,
and 12740. None dominated the selected checkpoint on goal rate, ball progress,
and fall rate, so the original selected checkpoint was retained rather than
overwriting it with a worse candidate.

Final deterministic evaluation used 64 trials with failures retained and no
smoothing:

| Metric | Result |
| --- | ---: |
| Goal success rate | 64.0625% |
| Fall rate | 1.5625% |
| Mean maximum ball progress | 0.226 m |
| Goal line distance | 0.150 m |
| Goal width | 0.260 m |

The raw trial table, per-step trajectory, metrics JSON, and trajectory figure
use the prefix
`evaluation/advanced_stages/goal015_selected_12620_deterministic`.

## Reproduce evaluation

```bash
cd /root/miniducktraining/ROAI
conda activate miniduck_cu118
export LD_LIBRARY_PATH="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$CONDA_PREFIX/lib:/usr/local/cuda/lib64"
export MINIDUCK_FORCED_STAGE=6
export MINIDUCK_BALL_PHASE=kick
export MINIDUCK_BALL_KICK_SPAWN_DISTANCE=0.03
export MINIDUCK_BALL_GOAL_DISTANCE=0.15
python legged_panguin/scripts/evaluate_advanced_stages.py \
  --task miniduck_flat --headless --num_envs 64 \
  --protocol ball_kick --duration_s 6 \
  --load_run Aug01_17-22-14_stage10_close_ball_finetune_100 \
  --checkpoint 12620 --run_label goal015_selected_12620
```

## Reproduce full visualization

```bash
cd /root/miniducktraining/ROAI
conda activate miniduck_cu118
export DISPLAY=:20
export LD_LIBRARY_PATH="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$CONDA_PREFIX/lib:/usr/local/cuda/lib64"
python -u legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat --demo full_sequence --num_envs 1 --fixed_camera
```

The sequence order remains unchanged through obstacle crossing; the final kick
module is now `ball_approach -> ball_kick -> finish_stand`.
