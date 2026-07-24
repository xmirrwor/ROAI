#!/usr/bin/env python3
"""Show the MiniDuck URDF in Isaac Gym at its optimized slanted stance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from isaacgym import gymapi


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT_DIR / "resources" / "robots" / "miniduck"
ASSET_FILE = "urdf/MINIDUCK.urdf"

HOME_POS = (0.0, 0.0, 0.115629099309)
HOME_QUAT_XYZW = (0.0, 0.224055365304, 0.0, 0.974576417362)
HOME_JOINTS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.399379879236,
    "left_knee": 1.5,
    "left_ankle": -0.623843669891,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.399379879236,
    "right_knee": 1.5,
    "right_ankle": -0.623843669891,
}


def main() -> None:
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = False
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create Isaac Gym viewer")

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.collapse_fixed_joints = False
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    print(f"Loading asset {ASSET_FILE} from {ASSET_ROOT}")
    asset = gym.load_asset(sim, str(ASSET_ROOT), ASSET_FILE, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load {ASSET_ROOT / ASSET_FILE}")

    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    print("Rigid bodies:")
    for i, name in enumerate(body_names):
        print(f"  {i}: {name}")
    print("DOFs:")
    for i, name in enumerate(dof_names):
        print(f"  {i}: {name}")

    env = gym.create_env(sim, gymapi.Vec3(-0.5, -0.5, 0.0), gymapi.Vec3(0.5, 0.5, 0.5), 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*HOME_POS)
    pose.r = gymapi.Quat(*HOME_QUAT_XYZW)
    actor = gym.create_actor(env, asset, pose, "miniduck_urdf", 0, 1)

    dof_props = gym.get_asset_dof_properties(asset)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(80.0)
    dof_props["damping"].fill(4.0)
    gym.set_actor_dof_properties(env, actor, dof_props)

    dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    for i, name in enumerate(dof_names):
        dof_states["pos"][i] = HOME_JOINTS[name]
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, dof_states["pos"])

    gym.viewer_camera_look_at(
        viewer,
        env,
        gymapi.Vec3(0.45, -0.75, 0.38),
        gymapi.Vec3(-0.02, 0.0, 0.10),
    )

    print("MiniDuck URDF-only viewer is running. Close the viewer window to exit.")
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
