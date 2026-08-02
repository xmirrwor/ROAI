# MiniDuck single-leg physics checkpoint

`model_550.pt` is the selected iteration-550 checkpoint from
`single_leg_miniduck_physics/physics_p2_550_seed`. It was trained for the
`miniduck_single_leg_physics` task with the configuration snapshot stored next
to the model.

## Integrity

From the repository root:

```bash
sha256sum -c saved_models/single_leg_physics_best_550/SHA256SUMS
```

Expected SHA-256:

```text
e336ab282ae284ee344f830bef8b69b292a92188cb46864e729f9a9101c1bf90
```

## Visualize

The runner loads checkpoints from `logs/<experiment>/<run>`, so stage the
tracked model there before launching Isaac Gym:

```bash
mkdir -p logs/single_leg_miniduck_physics/single_leg_physics_best_550
cp saved_models/single_leg_physics_best_550/model_550.pt \
  logs/single_leg_miniduck_physics/single_leg_physics_best_550/
python legged_panguin/scripts/play_miniduck.py \
  --task miniduck_single_leg_physics \
  --load_run single_leg_physics_best_550
```

To continue training, use the same staged path with `train.py --resume` and
`--checkpoint 550`.
