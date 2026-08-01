# Visible obstacle and user stages 9-10

## Scope

The user-facing sequence now ends with two ball phases after user stage 8:

- User stage 9: approach a ball from 0.30 m and enter a stable contact-ready zone.
- User stage 10: place the ball on the right-foot line and produce a real PhysX
  contact that moves the ball forward. This is a prototype kick, not a robust
  football controller or a claim of sim-to-real transfer.

The policy interface remains exactly 64 observations and 10 actions. During the
kick skill, observation indexes 10-11 encode ball position in the robot frame;
indexes 62-63 retain the gait phase.

## Obstacle visibility fix

The physical stage-8 obstacle remains the trained heightfield ridge. A bright
blue fixed box is placed at the same location as a visual marker and filtered
from robot collision, so it cannot create a second obstacle. Restoring the
heightfield spawn lift was required to prevent initial interpenetration.

Final deterministic stage-8 regression (`n=32`):

- success: 84.375%
- falls: 0%
- mean maximum forward progress: 0.659 m

## Training and selection

Stage 9 continued the selected stage-7 locomotion checkpoint for 200 PPO
iterations with 512 parallel environments. Stage 10 continued the selected
stage-9 checkpoint for 100 PPO iterations after a geometry curriculum placed
the ball 0.03 m ahead of the base and on the measured right-foot line.

Final raw deterministic evaluations (`n=64`, no smoothing, failed trials kept):

| Phase | Checkpoint | Success | Falls | Main motion metric |
| --- | --- | ---: | ---: | --- |
| Stage 9 approach | `model_12520.pt` | 57.8125% | 1.5625% | mean closest base-ball distance 0.144 m |
| Stage 10 kick | `model_12620.pt` | 70.3125% | 1.5625% | mean max ball progress 0.258 m |

Stage 9 only clears the prototype threshold by a modest margin. More spawn randomization
and a dedicated foot-target observation would be needed before treating it as
a robust approach controller.

## Selected artifacts

- `checkpoints/miniduck_stage9_ball_approach_model_selected.pt`
- `checkpoints/miniduck_stage10_ball_kick_model_selected.pt`
- `exported_policy/miniduck_stage10_ball_kick.onnx`
- `exported_policy/miniduck_stage10_ball_kick.json`
- `evaluation/advanced_stages/stage8_visible_final_with_spawn_lift_*`
- `evaluation/advanced_stages/stage9_final_12520_*`
- `evaluation/advanced_stages/stage10_finetuned_12620_*`

## Visualization

Run the full chain on the remote desktop:

```bash
cd /root/miniducktraining/ROAI
conda activate miniduck_cu118
export MINIDUCK_BALL_PHASE=kick
export LD_LIBRARY_PATH="$PWD/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64:$CONDA_PREFIX/lib:/usr/local/cuda/lib64"
export DISPLAY=:20
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_flat \
  --demo full_sequence \
  --fixed_camera
```

The loop shows stand, forward, emergency stop, backward, emergency stop,
squat/stand and action switches, fall/recovery, four diagonal directions,
visible obstacle crossing, ball preparation/kick, and final stand.

The full-chain viewer uses a plane plus the blue box to avoid a remote viewer
deadlock observed with heightfield rendering and two extra actors. Stage-8
quantitative acceptance continues to use the original physical heightfield.
