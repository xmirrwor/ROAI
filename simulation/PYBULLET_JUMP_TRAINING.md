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
`simulation/requirements-pybullet.txt`, and runs the optimizer. The default
result is written to:

```text
simulation/logs/pybullet_jump/best_jump.json
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
bash run_pybullet_jump.sh --iterations 90 --population 64
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

The console reports total center-of-mass rise, takeoff velocity, ballistic
rise after takeoff, continuous simultaneous airborne time, horizontal drift,
and whether the robot landed. A real jump requires `air > 0`,
`takeoff > 0`, and `ballistic > 0`; total `rise` alone can be a standing
extension. Flight is accepted only after both feet remain below the contact
force threshold for at least three consecutive physics steps.

PyBullet uses `pybullet_jump/miniduck_pybullet.urdf`, which preserves the
original visuals, mass, inertia, and joints while replacing the two STL foot
collision meshes with stable primitive boxes. The Isaac Gym URDF is unchanged.
