# MiniDuck PyBullet jump training

This is a Windows-compatible baseline that does not require Isaac Gym, CUDA,
WSL, or Stable-Baselines3. It loads the existing MiniDuck URDF and searches a
mirrored, parameterized crouch/push/landing trajectory.

## Train on Windows

From the repository root, run:

```powershell
.\run_pybullet_jump.bat
```

The script creates `.venv-pybullet`, installs the two dependencies listed in
`requirements-pybullet.txt`, and runs the optimizer. The default
result is written to:

```text
logs/pybullet_jump/best_jump.json
```

For a short check:

```powershell
.\run_pybullet_jump.bat --iterations 3 --population 8
```

For a broader search:

```powershell
.\run_pybullet_jump.bat --iterations 100 --population 96
```

## Train on Linux

From the repository root, run:

```bash
PYTHONUNBUFFERED=1 PYTHON_BIN=python bash run_pybullet_jump.sh \
  --iterations 120 --population 64 2>&1 | tee jump_training.log
```

The script uses the active Linux or Conda Python environment. Set
`PYTHON_BIN` when a different interpreter is required.

## Replay

After training, run:

```powershell
.\play_pybullet_jump.bat
```

Press `Ctrl+C` in the terminal to stop the replay.

## Success criteria

The console distinguishes absolute peak base/COM height (`peak_base`,
`peak_com`) from the free-flight COM rise (`ballistic`) and simultaneous
foot-bottom clearance (`foot_clearance`). Total `rise` or `peak_base` alone
can be produced by standing extension and is not proof of a jump.

A scored jump requires at least 3 mm ballistic rise, 1 mm simultaneous foot
clearance, and three consecutive fully unsupported physics steps. The whole
robot, not only its feet, must be clear of the ground. A settled landing also
requires both feet to support the robot at the end, no trunk/leg ground hit,
final COM speed below 0.25 m/s, tilt below 12 degrees, drift below 0.05 m, and
support force near the model weight.

The final reward has strict outcome tiers: settled landings outrank unstable
landings, which outrank jumps without landing. Within the same tier, ballistic
rise and foot clearance are rewarded monotonically without a 0.20 m cap.

## Continue from a settled landing

Use a previously saved result that replays with `settled=true` as the parent:

```bash
PYTHONUNBUFFERED=1 PYTHON_BIN=python bash run_pybullet_jump.sh \
  --resume checkpoints/pybullet_jump_best_v6.json \
  --iterations 300 --population 96 --height-step 0.003 \
  --output logs/pybullet_jump/best_jump_v8.json \
  2>&1 | tee jump_resume.log
```

Resume mode validates the parent before training. Every generation keeps the
parent unchanged, samples only nearby controllers, and permits a replacement
only when it still has a settled two-foot landing. `--height-step 0.003`
limits one generation to 3 mm additional ballistic rise. A resumable
`*.checkpoint.json` is written every five iterations.

Resume search perturbs crouch/push, aerial tuck, and landing absorption in
separate parameter blocks. An accepted update must preserve absolute COM
height, ballistic rise, foot clearance, airborne time, drift, and impact
within tight tolerances. The console prints `update=yes` only when a new
non-regressing controller replaces the parent.

The resume height objective is computed only during fully unsupported flight:

```text
flight_peak_com
```

Settled landing is a hard constraint, including landing impact below 20 times
model weight. Stable candidates are scored directly by this objective, so an
accepted update cannot lower the displayed score. Both a rolling checkpoint
and numbered files such as `best_jump_v6.iter0050.json` are retained.

When landing impact rises above 240 N, resume training automatically enters
`mode=landing`: most candidates modify only landing pose/recovery, while the
flight COM peak may not fall from the mode-entry controller by default.
After impact reaches 220 N it switches back to `mode=height`. This hysteresis
prevents height search from stalling against the 300 N settled-landing limit.
Secondary flight metrics may trade off, but every accepted height update must
raise the unsupported-flight COM peak and retain a safe landing: no body hit,
drift below 0.035 m, post-landing speed below 0.22 m/s, and impact below 20
times model weight.

PyBullet uses `pybullet_jump/miniduck_pybullet.urdf`, which preserves the
original visuals, mass, inertia, and joints while replacing the two STL foot
collision meshes with stable primitive boxes. The Isaac Gym URDF is unchanged.

## Continuous jump model

Continuous jumping is a separate policy and never overwrites the preserved
400-iteration single-jump model. Train two jumps first:

```bash
PYTHONUNBUFFERED=1 PYTHON_BIN=python bash run_pybullet_continuous_jump.sh \
  --base checkpoints/pybullet_jump_best_v6.json \
  --jumps 2 --iterations 300 --population 128 \
  --output checkpoints/pybullet_continuous_jump_2.json
```

After the log reaches `completed=2/2`, extend the saved controller to three
jumps without resetting simulation state:

```bash
PYTHONUNBUFFERED=1 PYTHON_BIN=python bash run_pybullet_continuous_jump.sh \
  --base checkpoints/pybullet_continuous_jump_2.json \
  --jumps 3 --iterations 400 --population 128 \
  --output checkpoints/pybullet_continuous_jump_3.json
```

The continuous policy shares the 16 single-jump motion parameters and adds a
trainable inter-jump settling pause. Completion count dominates the score;
after every requested jump lands safely, the minimum flight COM peak across
all jumps is optimized so a single high first jump cannot mask weaker later
jumps.
