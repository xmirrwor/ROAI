"""Render a continuous-jump checkpoint to a remote-desktop-safe MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p

from pybullet_jump.continuous_jump import MiniDuckContinuousJumpSim
from pybullet_jump.jump_sim import PARAMETER_NAMES


ROOT = Path(__file__).resolve().parent


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_floor_visual(sim: MiniDuckContinuousJumpSim) -> None:
    floor = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[2.5, 2.5, 0.005],
        rgbaColor=[0.34, 0.37, 0.40, 1.0],
        physicsClientId=sim.client,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=floor,
        basePosition=[0.0, 0.0, -0.007],
        physicsClientId=sim.client,
    )
    center_line = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[2.5, 0.006, 0.001],
        rgbaColor=[0.72, 0.75, 0.78, 1.0],
        physicsClientId=sim.client,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=center_line,
        basePosition=[0.0, 0.0, 0.0005],
        physicsClientId=sim.client,
    )


def render_frame(
    sim: MiniDuckContinuousJumpSim,
    frame: dict,
    width: int,
    height: int,
    cycle: int,
    jumps: int,
    rise_cm: float,
) -> bytes:
    p.resetBasePositionAndOrientation(
        sim.robot,
        frame["base_position"],
        frame["base_quaternion"],
        physicsClientId=sim.client,
    )
    for name, value in frame["joint_positions"].items():
        p.resetJointState(
            sim.robot,
            sim.joints[name],
            value,
            targetVelocity=0.0,
            physicsClientId=sim.client,
        )
    target = frame["base_position"]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[target[0], target[1], 0.12],
        distance=0.62,
        yaw=48.0,
        pitch=-18.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=48.0,
        aspect=width / height,
        nearVal=0.02,
        farVal=6.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[-2.0, -3.0, 5.0],
        physicsClientId=sim.client,
    )
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8)[:, :, :3])
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((18, 18, 380, 78), fill=(15, 18, 22, 205))
    font = ImageFont.load_default(size=18)
    draw.text((34, 29), f"CONTINUOUS JUMP  {cycle}/{jumps}", fill=(255, 255, 255, 255), font=font)
    draw.text((34, 53), f"minimum ballistic rise: {rise_cm:.2f} cm", fill=(147, 220, 255, 255), font=font)
    return image.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("checkpoints/pybullet_continuous_jump_best.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/continuous_jump/continuous_jump_validation.mp4"),
    )
    parser.add_argument("--jumps", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    policy = rooted(args.policy)
    output = rooted(args.output)
    data = json.loads(policy.read_text(encoding="utf-8"))
    parameters = [data["parameters"][name] for name in PARAMETER_NAMES]
    pause = data.get("continuous_parameters", {}).get("inter_jump_pause", 0.25)
    physics = data.get("physics", {})
    sim = MiniDuckContinuousJumpSim(
        gui=False,
        physics_substeps=physics.get("substeps", 2),
        solver_iterations=physics.get("solver_iterations", 120),
        contact_erp=physics.get("contact_erp", 0.20),
    )
    try:
        metrics, frames = sim.evaluate_continuous(
            parameters,
            jumps=args.jumps,
            inter_jump_pause=pause,
            capture=True,
        )
        if not metrics.all_settled:
            raise RuntimeError(
                f"Checkpoint failed validation: {metrics.completed_jumps}/{args.jumps}"
            )
        add_floor_visual(sim)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        capture_fps = 60
        stride = max(1, round(capture_fps / args.fps))
        render_frames = frames[::stride]
        frames_per_cycle = max(1, len(render_frames) // args.jumps)
        rise_cm = 100.0 * metrics.min_ballistic_rise
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            for index, frame in enumerate(render_frames):
                cycle = min(index // frames_per_cycle + 1, args.jumps)
                encoder.stdin.write(
                    render_frame(
                        sim,
                        frame,
                        args.width,
                        args.height,
                        cycle,
                        args.jumps,
                        rise_cm,
                    )
                )
            encoder.stdin.close()
            return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")
        print(
            f"saved={output} frames={len(render_frames)} "
            f"completed={metrics.completed_jumps}/{args.jumps} "
            f"rise={metrics.min_ballistic_rise:.4f}m"
        )
    finally:
        sim.close()


if __name__ == "__main__":
    main()
