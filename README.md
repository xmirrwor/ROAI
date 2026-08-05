# ROAI2041 Simulation Assignment - MiniDuck

This submission contains trained policies and visualization programs for the
10-DoF MiniDuck biped robot. The core policy demonstrates forward and backward
walking. Additional policies demonstrate stopping, squatting, action switching,
fall recovery, diagonal motion, obstacle crossing, ball approach and kicking,
single-leg standing, and jumping.

## Group Information

Complete this section before creating the final submission ZIP. Add or remove
member rows as needed.

**Group name or number:** [Group name or number]

| Group4 
| XU JINGWEN 
| ZHAO ZHIXUAN 
| SHEN JINHAN
| LING ZIQI

## Core Policy: Forward and Backward Walking

**Behavior:** The policy keeps MiniDuck upright while following positive and
negative forward-velocity commands. The submitted visualization also shows
left and right semicircles and one in-place rotation after the required walking
demonstration.

**Trained policy:**
`simulation/checkpoints/miniduck_stable_policy.pt`

**Visualization code:**
`simulation/visualizations/initial_motion.py`

Run the visualization from the repository root:

```bash
python simulation/visualizations/initial_motion.py --cycles 1 --fixed_camera
```

**Expected behavior:** An Isaac Gym window opens with one MiniDuck robot. The
robot travels approximately 0.5 m forward, then 0.5 m backward, completes left
and right semicircles, and finally rotates once in place. Each phase advances
when its measured distance or accumulated yaw reaches the target.

## Additional Policies

All commands below must be run from the repository root. Add `--fixed_camera`
to an Isaac Gym command when a stationary camera is preferred.

| Action or behavior | Trained policy file | Visualization command | Expected behavior |
| --- | --- | --- | --- |
| General locomotion | `simulation/checkpoints/miniduck_stable_policy.pt` | `python simulation/visualizations/locomotion.py --cycles 1` | The robot remains upright while demonstrating forward, backward, curved, lateral, and in-place turning commands. |
| Emergency stop | `simulation/checkpoints/miniduck_emergency_stop_model_12500.pt` | `python simulation/visualizations/emergency_stop.py --cycles 1` | The robot walks forward and backward, stops after each probe, and returns to a stable stance. |
| Squat | `simulation/checkpoints/miniduck_stage4_squat_model_13900.pt` | `python simulation/visualizations/squat.py --cycles 1` | The robot stands, lowers its body into a squat, and returns to standing. |
| Action switching | `simulation/checkpoints/miniduck_stage5_action_switch_model_14780.pt` | `python simulation/visualizations/action_switch.py --cycles 1` | The robot transitions through stand, forward walk, stop, squat, stand, backward walk, and stop. |
| Fall recovery | `simulation/checkpoints/miniduck_stage6_recovery_model_15420.pt` | `python simulation/visualizations/fall_recovery.py --cycles 1` | The robot starts from a prone pose, recovers, and returns to an upright stance. |
| Diagonal motion | `simulation/checkpoints/miniduck_stage7_diagonal_fast_model_12320.pt` | `python simulation/visualizations/diagonal_motion.py --cycles 1` | The robot moves forward-left, forward-right, backward-left, and backward-right. |
| Obstacle crossing | `simulation/checkpoints/miniduck_stage8_obstacle_model_selected.pt` | `python simulation/visualizations/obstacle_crossing.py --cycles 1` | A visible obstacle is placed ahead of the robot; the robot approaches and attempts to cross it. |
| Ball approach and kick | `simulation/checkpoints/miniduck_stage7_diagonal_fast_model_12320.pt` and `simulation/checkpoints/miniduck_stage10_ball_kick_model_selected.pt` | `python simulation/visualizations/ball_approach_and_kick.py --cycles 1` | The robot approaches the visible ball, aligns with it, switches to the kick policy, and kicks toward the displayed goal. |
| Single-leg stand | `simulation/checkpoints/miniduck_single_leg_physics_model_550.pt` | `python simulation/visualizations/single_leg.py --cycles 1` | The robot loads the physics-seeded model at iteration 550 and attempts to maintain the commanded single-leg support pose. |
| Jump | `simulation/logs/pybullet_jump/best_jump_v6.json` | `python simulation/visualizations/jump.py --repetitions 1` | A PyBullet window opens; the robot crouches, pushes off, becomes airborne, lands on both feet, and settles. |

## Complete Demonstration

To run the multi-policy sequence, single-leg visualization, and jump replay in
order, use:

```bash
python simulation/visualizations/full_sequence.py --repetitions 1 --fixed_camera
```

The first Isaac Gym window demonstrates the available multi-policy actions,
including forward and backward walking, stopping, squatting, recovery, diagonal
motion, obstacle crossing, and ball approach and kicking. After it closes, the
single-leg Isaac Gym demonstration and the PyBullet jump replay start in turn.

## Environment and Installation

The Isaac Gym visualizations require the environment used for training:

- Ubuntu 20.04 or 22.04;
- an NVIDIA GPU and compatible driver;
- Python 3.8;
- PyTorch 1.10.2 with CUDA 11.3;
- Isaac Gym Preview 4.

Install the included simulator and training packages from the repository root:

```bash
cd simulation
pip install -e third_party/isaacgym/python
pip install -e third_party/rsl_rl
pip install -e .
cd ..
```

The PyBullet jump visualization is independent of Isaac Gym and can also run on
Windows. Install its dependencies with:

```bash
pip install -r simulation/requirements-pybullet.txt
```

## Testing

Check that every visualization entry point resolves its required policy file:

```bash
python -m unittest simulation.tests.test_visualization_launchers -v
```

Before submission, run every command listed above in its required simulator
environment. Confirm that each policy loads, each visualization opens, and no
file is missing or referenced through a machine-specific absolute path.

## Submission Packaging

The working repository keeps reusable policy files in `simulation/checkpoints/`
and entry points in `simulation/visualizations/`. For the final course ZIP,
follow the assignment packaging rule: create one folder per submitted behavior
and store the trained policy, its working visualization code, and a short README
in the same folder. Keep all behavior folders inside one clearly named group
folder, include the completed group information above, and compress that group
folder into one clearly named ZIP file, such as
`Group_03_Simulation_Assignment.zip`.

Do not move only a visualization entry-point file: the Isaac Gym entry points
also depend on `simulation/visualizations/_launcher.py`, `simulation/play.py`,
the `simulation/legged_panguin/` package, robot resources, and the included
third-party packages. After reorganizing the final ZIP, repeat the loading and
visualization tests to confirm that all relative paths still work.
