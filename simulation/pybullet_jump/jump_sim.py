"""MiniDuck PyBullet environment and parameterized jump controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = Path("resources/robots/miniduck/urdf/MINIDUCK.urdf")
PYBULLET_URDF_PATH = Path("pybullet_jump/miniduck_pybullet.urdf")
DT = 1.0 / 240.0
DEFAULT_POSE = {
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
PARAMETER_NAMES = (
    "crouch_hip",
    "crouch_knee",
    "crouch_ankle",
    "extend_hip",
    "extend_knee",
    "extend_ankle",
    "tuck_hip",
    "tuck_knee",
    "tuck_ankle",
    "crouch_time",
    "push_time",
    "tuck_time",
)
LOWER_BOUNDS = np.array(
    [-0.30, -0.35, -0.40, -0.50, -1.40, -0.60, -0.35, -0.65, -0.45, 0.15, 0.075, 0.05]
)
UPPER_BOUNDS = np.array(
    [0.30, 0.00, 0.40, 0.50, 0.00, 0.60, 0.35, 0.00, 0.45, 0.30, 0.18, 0.16]
)
MIN_FLIGHT_STEPS = 3
CONTACT_FORCE_THRESHOLD = 0.5


@dataclass
class JumpMetrics:
    score: float
    com_rise: float
    max_upward_velocity: float
    takeoff_velocity: float
    ballistic_rise: float
    airborne_time: float
    horizontal_drift: float
    max_tilt: float
    peak_contact_force: float
    landed: bool


class MiniDuckJumpSim:
    def __init__(self, gui: bool = False):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Unable to connect to PyBullet")
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(DT, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=DT,
            numSolverIterations=80,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client,
        )
        plane_shape = p.createCollisionShape(
            p.GEOM_PLANE,
            planeNormal=[0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )
        self.plane = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=plane_shape,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.0,
            restitution=0.0,
            physicsClientId=self.client,
        )
        previous_cwd = Path.cwd()
        try:
            # PyBullet on Windows cannot reliably open non-ASCII absolute paths.
            os.chdir(ROOT)
            self._build_pybullet_urdf()
            self.robot = p.loadURDF(
                PYBULLET_URDF_PATH.as_posix(),
                [0.0, 0.0, 0.115629099309],
                p.getQuaternionFromEuler([0.0, 0.451947301626, 0.0]),
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.client,
            )
        finally:
            os.chdir(previous_cwd)
        self.joints = {}
        self.feet = []
        for index in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, index, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            if info[2] == p.JOINT_REVOLUTE:
                self.joints[name] = index
            if link_name.startswith("foot_assembly"):
                self.feet.append(index)
        if set(self.joints) != set(DEFAULT_POSE) or len(self.feet) != 2:
            raise RuntimeError("Unexpected MiniDuck URDF joint or foot layout")
        for index in self.joints.values():
            p.setJointMotorControl2(
                self.robot,
                index,
                p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.client,
            )
        for name, index in self.joints.items():
            p.resetJointState(
                self.robot,
                index,
                DEFAULT_POSE[name],
                physicsClientId=self.client,
            )
        p.performCollisionDetection(physicsClientId=self.client)
        lowest_foot_point = min(
            p.getAABB(self.robot, foot, self.client)[0][2] for foot in self.feet
        )
        self.initial_base_z = 0.115629099309 - lowest_foot_point + 0.002
        for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client)):
            p.changeDynamics(
                self.robot,
                link,
                lateralFriction=1.0,
                restitution=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=self.client,
            )
        if gui:
            p.resetDebugVisualizerCamera(0.7, 55, -18, [0, 0, 0.12], self.client)

    @staticmethod
    def _build_pybullet_urdf():
        if PYBULLET_URDF_PATH.exists():
            return
        tree = ET.parse(URDF_PATH)
        root = tree.getroot()
        for mesh in root.findall(".//visual/geometry/mesh"):
            filename = Path(mesh.attrib["filename"]).name
            mesh.set("filename", f"../resources/robots/miniduck/meshes/{filename}")
        for link in root.findall("link"):
            if not link.attrib.get("name", "").startswith("foot_assembly"):
                continue
            for origin in link.findall("collision/origin"):
                origin.set("xyz", "0 0 -0.010")
                origin.set("rpy", "0 0 0")
            for geometry in link.findall("collision/geometry"):
                geometry.clear()
                ET.SubElement(geometry, "box", {"size": "0.080 0.045 0.015"})
        tree.write(PYBULLET_URDF_PATH, encoding="utf-8", xml_declaration=True)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.robot,
            [0.0, 0.0, self.initial_base_z],
            p.getQuaternionFromEuler([0.0, 0.451947301626, 0.0]),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0], self.client)
        for name, index in self.joints.items():
            p.resetJointState(
                self.robot,
                index,
                DEFAULT_POSE[name],
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        self._command(DEFAULT_POSE)
        for _ in range(int(0.35 / DT)):
            p.stepSimulation(physicsClientId=self.client)

    def _command(self, targets, velocity_gain=1.0):
        for name, index in self.joints.items():
            p.setJointMotorControl2(
                self.robot,
                index,
                p.POSITION_CONTROL,
                targetPosition=float(targets[name]),
                targetVelocity=0.0,
                force=3.23,
                maxVelocity=5.24,
                positionGain=0.08,
                velocityGain=velocity_gain,
                physicsClientId=self.client,
            )

    @staticmethod
    def _pose(delta):
        hip, knee, ankle = delta
        pose = dict(DEFAULT_POSE)
        pose["left_hip_pitch"] += hip
        pose["right_hip_pitch"] -= hip
        pose["left_knee"] += knee
        pose["right_knee"] += knee
        pose["left_ankle"] += ankle
        pose["right_ankle"] += ankle
        return pose

    @staticmethod
    def _blend(start, end, amount):
        amount = amount * amount * (3.0 - 2.0 * amount)
        return {name: start[name] + amount * (end[name] - start[name]) for name in start}

    def _center_of_mass(self):
        weighted = np.zeros(3)
        total_mass = 0.0
        base_mass = p.getDynamicsInfo(self.robot, -1, self.client)[0]
        base_pos = np.asarray(p.getBasePositionAndOrientation(self.robot, self.client)[0])
        weighted += base_mass * base_pos
        total_mass += base_mass
        for link in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            mass = p.getDynamicsInfo(self.robot, link, self.client)[0]
            position = np.asarray(p.getLinkState(self.robot, link, computeForwardKinematics=True, physicsClientId=self.client)[0])
            weighted += mass * position
            total_mass += mass
        return weighted / total_mass

    def evaluate(
        self,
        parameters,
        realtime: bool = False,
        capture: bool = False,
        curriculum_stage: int = 2,
    ):
        parameters = np.clip(np.asarray(parameters, dtype=float), LOWER_BOUNDS, UPPER_BOUNDS)
        crouch = self._pose(parameters[:3])
        extend = self._pose(parameters[3:6])
        tuck = self._pose(parameters[6:9])
        crouch_time, push_time, tuck_time = parameters[9:12]
        self.reset()
        initial_com = self._center_of_mass()
        max_com_z = initial_com[2]
        previous_com_z = initial_com[2]
        previous_vertical_velocity = 0.0
        max_upward_velocity = 0.0
        takeoff_velocity = 0.0
        takeoff_com_z = None
        max_flight_com_z = 0.0
        max_tilt = 0.0
        peak_force = 0.0
        current_airborne_steps = 0
        longest_airborne_steps = 0
        candidate_takeoff_z = 0.0
        candidate_takeoff_velocity = 0.0
        had_flight = False
        landed = False
        landing_step = None
        frames = []
        duration = 1.35
        for step in range(int(duration / DT)):
            t = step * DT
            if t < crouch_time:
                target = self._blend(DEFAULT_POSE, crouch, t / crouch_time)
            elif t < crouch_time + push_time:
                target = self._blend(crouch, extend, (t - crouch_time) / push_time)
            elif landed:
                elapsed = (step - landing_step) * DT
                target = self._blend(crouch, DEFAULT_POSE, min(elapsed / 0.30, 1.0))
            elif had_flight and max_upward_velocity > 0.0 and previous_vertical_velocity < 0.0:
                target = crouch
            else:
                elapsed = t - crouch_time - push_time
                target = self._blend(extend, tuck, min(elapsed / tuck_time, 1.0))
            push_active = crouch_time <= t < crouch_time + push_time
            self._command(target, velocity_gain=0.98 if push_active else 1.0)
            p.stepSimulation(physicsClientId=self.client)
            contacts = [
                p.getContactPoints(self.robot, self.plane, linkIndexA=foot, physicsClientId=self.client)
                for foot in self.feet
            ]
            foot_forces = [sum(point[9] for point in points) for points in contacts]
            foot_contacts = [force > CONTACT_FORCE_THRESHOLD for force in foot_forces]
            both_airborne = not any(foot_contacts)
            com = self._center_of_mass()
            vertical_velocity = float((com[2] - previous_com_z) / DT)
            if both_airborne:
                if current_airborne_steps == 0:
                    candidate_takeoff_z = float(com[2])
                    candidate_takeoff_velocity = max(vertical_velocity, 0.0)
                current_airborne_steps += 1
                longest_airborne_steps = max(longest_airborne_steps, current_airborne_steps)
                if current_airborne_steps >= MIN_FLIGHT_STEPS and not had_flight:
                    had_flight = True
                    takeoff_com_z = candidate_takeoff_z
                    takeoff_velocity = candidate_takeoff_velocity
                    max_flight_com_z = float(com[2])
                if had_flight:
                    max_flight_com_z = max(max_flight_com_z, float(com[2]))
            else:
                current_airborne_steps = 0
            if had_flight and all(foot_contacts) and not landed:
                landed = True
                landing_step = step
            peak_force = max(peak_force, sum(foot_forces))
            max_upward_velocity = max(
                max_upward_velocity,
                vertical_velocity,
            )
            previous_com_z = com[2]
            previous_vertical_velocity = vertical_velocity
            max_com_z = max(max_com_z, com[2])
            position, orientation = p.getBasePositionAndOrientation(self.robot, self.client)
            roll, pitch, _ = p.getEulerFromQuaternion(orientation)
            max_tilt = max(max_tilt, abs(roll), abs(pitch - 0.451947301626))
            if capture and step % 4 == 0:
                joints = {
                    name: p.getJointState(self.robot, index, physicsClientId=self.client)[0]
                    for name, index in self.joints.items()
                }
                frames.append({
                    "time": t,
                    "base_position": list(position),
                    "base_quaternion": list(orientation),
                    "joint_positions": joints,
                })
            if realtime:
                time.sleep(DT)
        final_com = self._center_of_mass()
        drift = float(np.linalg.norm(final_com[:2] - initial_com[:2]))
        rise = float(max_com_z - initial_com[2])
        airtime = longest_airborne_steps * DT if had_flight else 0.0
        ballistic_rise = (
            max(0.0, max_flight_com_z - takeoff_com_z)
            if takeoff_com_z is not None
            else 0.0
        )
        if rise > 0.5 or drift > 0.5:
            score = -1000.0 - rise - drift
        else:
            posture_penalty = 25.0 * drift + 4.0 * max(
                0.0, max_tilt - math.radians(10.0)
            )
            impact_penalty = 0.02 * max(0.0, peak_force - 80.0)
            if curriculum_stage == 0:
                score = 5.0 * min(max_upward_velocity, 1.2) + 30.0 * min(rise, 0.04)
                score -= posture_penalty + impact_penalty
            elif curriculum_stage == 1:
                score = 8.0 * min(takeoff_velocity, 1.2)
                score += 18.0 * min(airtime, 0.22)
                score += 60.0 * min(ballistic_rise, 0.05)
                score -= posture_penalty + impact_penalty
                if not had_flight:
                    score -= 4.0
            else:
                score = 8.0 * min(takeoff_velocity, 1.2)
                score += 20.0 * min(airtime, 0.22)
                score += 70.0 * min(ballistic_rise, 0.05)
                score -= posture_penalty + impact_penalty
                if not had_flight:
                    score -= 6.0
                elif not landed:
                    score -= 5.0
                else:
                    score += 3.0
        metrics = JumpMetrics(
            score,
            rise,
            max_upward_velocity,
            takeoff_velocity,
            ballistic_rise,
            airtime,
            drift,
            max_tilt,
            peak_force,
            landed,
        )
        return metrics, frames


def parameters_to_dict(parameters):
    return dict(zip(PARAMETER_NAMES, map(float, parameters)))


def metrics_to_dict(metrics):
    return asdict(metrics)
