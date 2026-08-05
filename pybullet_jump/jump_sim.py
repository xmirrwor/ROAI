"""MiniDuck PyBullet environment and parameterized jump controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import os
import time
from typing import Optional
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
    "landing_hip",
    "landing_knee",
    "landing_ankle",
    "crouch_time",
    "push_time",
    "tuck_time",
    "recovery_time",
)
LOWER_BOUNDS = np.array(
    [-0.30, -0.35, -0.40, -0.50, -1.40, -0.60, -0.35, -0.65, -0.45,
     -0.25, -0.55, -0.40, 0.15, 0.075, 0.05, 0.15]
)
UPPER_BOUNDS = np.array(
    [0.30, 0.00, 0.40, 0.50, 0.00, 0.60, 0.35, 0.00, 0.45,
     0.25, 0.00, 0.40, 0.30, 0.18, 0.16, 0.45]
)
MIN_FLIGHT_STEPS = 3
CONTACT_FORCE_THRESHOLD = 0.5
FINAL_TARGET_BALLISTIC_RISE = 0.080
FINAL_TARGET_BASE_HEIGHT = 0.200


@dataclass
class JumpMetrics:
    score: float
    com_rise: float
    peak_base_height: float
    peak_com_height: float
    peak_flight_base_height: float
    peak_flight_com_height: float
    max_foot_clearance: float
    max_upward_velocity: float
    takeoff_velocity: float
    takeoff_horizontal_velocity: float
    ballistic_rise: float
    airborne_time: float
    horizontal_drift: float
    max_tilt: float
    peak_contact_force: float
    landing_impact: float
    landing_velocity: float
    foot_contact_delay: float
    landing_force_asymmetry: float
    post_landing_speed: float
    final_speed: float
    final_tilt: float
    final_support_force: float
    final_both_feet_contact: bool
    body_contact_after_takeoff: bool
    landed: bool
    settled: bool


class MiniDuckJumpSim:
    def __init__(
        self,
        gui: bool = False,
        physics_substeps: int = 1,
        solver_iterations: int = 80,
        contact_erp: Optional[float] = None,
    ):
        if physics_substeps < 1:
            raise ValueError("physics_substeps must be at least 1")
        self.gui = gui
        self.physics_config = {
            "substeps": int(physics_substeps),
            "solver_iterations": int(solver_iterations),
            "contact_erp": contact_erp,
        }
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Unable to connect to PyBullet")
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(DT, physicsClientId=self.client)
        physics_parameters = dict(
            fixedTimeStep=DT,
            numSubSteps=physics_substeps,
            numSolverIterations=solver_iterations,
            deterministicOverlappingPairs=1,
        )
        if contact_erp is not None:
            physics_parameters.update(
                contactERP=contact_erp,
                frictionERP=contact_erp,
            )
        p.setPhysicsEngineParameter(
            **physics_parameters,
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
        self.total_mass = sum(
            p.getDynamicsInfo(self.robot, link, self.client)[0]
            for link in range(
                -1, p.getNumJoints(self.robot, physicsClientId=self.client)
            )
        )
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

    def _normal_force(self, contacts):
        # PyBullet reports contact force per internal substep. Normalize it to
        # the outer 240 Hz control step so thresholds remain configuration-safe.
        return self.physics_config["substeps"] * sum(
            point[9] for point in contacts
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

    def _center_of_mass_state(self):
        weighted = np.zeros(3)
        weighted_velocity = np.zeros(3)
        total_mass = 0.0
        base_mass = p.getDynamicsInfo(self.robot, -1, self.client)[0]
        base_pos = np.asarray(p.getBasePositionAndOrientation(self.robot, self.client)[0])
        base_velocity = np.asarray(p.getBaseVelocity(self.robot, self.client)[0])
        weighted += base_mass * base_pos
        weighted_velocity += base_mass * base_velocity
        total_mass += base_mass
        for link in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            mass = p.getDynamicsInfo(self.robot, link, self.client)[0]
            state = p.getLinkState(
                self.robot,
                link,
                computeLinkVelocity=True,
                computeForwardKinematics=True,
                physicsClientId=self.client,
            )
            position = np.asarray(state[0])
            velocity = np.asarray(state[6])
            weighted += mass * position
            weighted_velocity += mass * velocity
            total_mass += mass
        return weighted / total_mass, weighted_velocity / total_mass

    def evaluate(
        self,
        parameters,
        realtime: bool = False,
        capture: bool = False,
        curriculum_stage: int = 2,
        target_ballistic_rise: float = FINAL_TARGET_BALLISTIC_RISE,
    ):
        parameters = np.clip(np.asarray(parameters, dtype=float), LOWER_BOUNDS, UPPER_BOUNDS)
        crouch = self._pose(parameters[:3])
        extend = self._pose(parameters[3:6])
        tuck = self._pose(parameters[6:9])
        landing_pose = self._pose(parameters[9:12])
        crouch_time, push_time, tuck_time, recovery_time = parameters[12:16]
        self.reset()
        initial_com, _ = self._center_of_mass_state()
        max_com_z = initial_com[2]
        max_base_z = p.getBasePositionAndOrientation(self.robot, self.client)[0][2]
        initial_base_z = max_base_z
        max_foot_clearance = 0.0
        previous_com_z = initial_com[2]
        previous_vertical_velocity = 0.0
        max_upward_velocity = 0.0
        takeoff_velocity = 0.0
        takeoff_horizontal_velocity = 0.0
        takeoff_com_z = None
        max_flight_com_z = 0.0
        max_flight_base_z = 0.0
        max_tilt = 0.0
        peak_force = 0.0
        landing_impact = 0.0
        landing_velocity = 0.0
        first_landing_contact_steps = [None, None]
        landing_force_asymmetry = 0.0
        post_landing_speed = None
        push_force_asymmetry = 0.0
        current_airborne_steps = 0
        longest_airborne_steps = 0
        candidate_takeoff_z = 0.0
        candidate_takeoff_velocity = 0.0
        had_flight = False
        flight_complete = False
        landed = False
        body_contact_after_takeoff = False
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
                target = self._blend(
                    landing_pose,
                    DEFAULT_POSE,
                    min(elapsed / recovery_time, 1.0),
                )
            elif had_flight and max_upward_velocity > 0.0 and previous_vertical_velocity < 0.0:
                target = landing_pose
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
            foot_forces = [self._normal_force(points) for points in contacts]
            foot_contacts = [force > CONTACT_FORCE_THRESHOLD for force in foot_forces]
            all_ground_contacts = p.getContactPoints(
                self.robot,
                self.plane,
                physicsClientId=self.client,
            )
            total_ground_force = self._normal_force(all_ground_contacts)
            # A valid flight requires the complete robot to be unsupported.
            # Checking feet alone misclassifies a fallen trunk or shin as flight.
            both_airborne = total_ground_force <= CONTACT_FORCE_THRESHOLD
            com, com_velocity = self._center_of_mass_state()
            vertical_velocity = float(com_velocity[2])
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
                    takeoff_horizontal_velocity = float(np.linalg.norm(com_velocity[:2]))
                    max_flight_com_z = float(com[2])
                    max_flight_base_z = float(
                        p.getBasePositionAndOrientation(self.robot, self.client)[0][2]
                    )
                if had_flight and not flight_complete:
                    max_flight_com_z = max(max_flight_com_z, float(com[2]))
                    max_flight_base_z = max(
                        max_flight_base_z,
                        float(
                            p.getBasePositionAndOrientation(
                                self.robot, self.client
                            )[0][2]
                        ),
                    )
                    simultaneous_clearance = min(
                        p.getAABB(self.robot, foot, self.client)[0][2]
                        for foot in self.feet
                    )
                    max_foot_clearance = max(
                        max_foot_clearance,
                        float(max(0.0, simultaneous_clearance)),
                    )
            else:
                if had_flight:
                    flight_complete = True
                current_airborne_steps = 0
            if had_flight and all(foot_contacts) and not landed:
                landed = True
                landing_step = step
            if had_flight:
                if any(point[3] not in self.feet for point in all_ground_contacts):
                    body_contact_after_takeoff = True
                if any(foot_contacts) and landing_velocity == 0.0:
                    landing_velocity = abs(min(vertical_velocity, 0.0))
                for foot_index, contact in enumerate(foot_contacts):
                    if contact and first_landing_contact_steps[foot_index] is None:
                        first_landing_contact_steps[foot_index] = step
                landing_impact = max(landing_impact, sum(foot_forces))
                landing_force_sum = sum(foot_forces)
                if landing_force_sum > 1.0:
                    landing_force_asymmetry = max(
                        landing_force_asymmetry,
                        abs(foot_forces[0] - foot_forces[1]) / landing_force_sum,
                    )
            if (
                landed
                and post_landing_speed is None
                and (step - landing_step) * DT >= 0.15
            ):
                post_landing_speed = float(np.linalg.norm(com_velocity))
            if push_active:
                force_sum = sum(foot_forces)
                if force_sum > 1.0:
                    push_force_asymmetry = max(
                        push_force_asymmetry,
                        abs(foot_forces[0] - foot_forces[1]) / force_sum,
                    )
            peak_force = max(peak_force, sum(foot_forces))
            if t <= crouch_time + push_time + 0.05:
                max_upward_velocity = max(max_upward_velocity, vertical_velocity)
            previous_com_z = com[2]
            previous_vertical_velocity = vertical_velocity
            max_com_z = max(max_com_z, com[2])
            position, orientation = p.getBasePositionAndOrientation(self.robot, self.client)
            max_base_z = max(max_base_z, float(position[2]))
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
        final_com, final_com_velocity = self._center_of_mass_state()
        drift = float(np.linalg.norm(final_com[:2] - initial_com[:2]))
        rise = float(max_com_z - initial_com[2])
        airtime = longest_airborne_steps * DT if had_flight else 0.0
        ballistic_rise = (
            max(0.0, max_flight_com_z - takeoff_com_z)
            if takeoff_com_z is not None
            else 0.0
        )
        if all(step_value is not None for step_value in first_landing_contact_steps):
            foot_contact_delay = (
                abs(first_landing_contact_steps[0] - first_landing_contact_steps[1]) * DT
            )
        else:
            foot_contact_delay = duration
        final_speed = float(np.linalg.norm(final_com_velocity))
        if post_landing_speed is None:
            post_landing_speed = final_speed
        _, final_orientation = p.getBasePositionAndOrientation(self.robot, self.client)
        final_roll, final_pitch, _ = p.getEulerFromQuaternion(final_orientation)
        final_tilt = max(
            abs(final_roll),
            abs(final_pitch - 0.451947301626),
        )
        final_foot_forces = [
            self._normal_force(
                p.getContactPoints(
                    self.robot,
                    self.plane,
                    linkIndexA=foot,
                    physicsClientId=self.client,
                )
            )
            for foot in self.feet
        ]
        final_both_feet_contact = all(
            force > CONTACT_FORCE_THRESHOLD for force in final_foot_forces
        )
        final_support_force = float(sum(final_foot_forces))
        support_ratio = final_support_force / max(self.total_mass * 9.81, 1.0)
        settled = (
            landed
            and final_both_feet_contact
            and not body_contact_after_takeoff
            and final_speed < 0.25
            and final_tilt < math.radians(12.0)
            and drift < 0.035
            and post_landing_speed < 0.22
            and 0.60 < support_ratio < 1.40
            and landing_impact < 20.0 * self.total_mass * 9.81
        )
        target_takeoff_velocity = math.sqrt(2.0 * 9.81 * target_ballistic_rise)
        target_airtime = 2.0 * target_takeoff_velocity / 9.81
        numerically_invalid = (
            rise > 0.5
            or drift > 0.5
            or max_upward_velocity > 4.0
            or takeoff_velocity > 4.0
            or peak_force > 500.0
        )
        if numerically_invalid:
            score = -1000.0 - rise - drift
        else:
            posture_penalty = 25.0 * drift + 4.0 * max(
                0.0, max_tilt - math.radians(10.0)
            )
            impact_penalty = 0.02 * max(0.0, peak_force - 80.0)
            takeoff_balance_penalty = (
                4.0 * takeoff_horizontal_velocity + 1.5 * push_force_asymmetry
            )
            if curriculum_stage == 0:
                speed_progress = min(max_upward_velocity / target_takeoff_velocity, 1.0)
                score = 5.0 * speed_progress + 30.0 * min(rise, 0.04)
                score -= takeoff_balance_penalty + impact_penalty
            elif curriculum_stage == 1:
                speed_progress = min(takeoff_velocity / target_takeoff_velocity, 1.0)
                air_progress = min(airtime / target_airtime, 1.0)
                height_progress = max(
                    ballistic_rise / max(target_ballistic_rise, 1.0e-6), 0.0
                )
                score = 5.0 * speed_progress + 4.0 * air_progress
                score += 12.0 * math.log1p(height_progress)
                score -= (
                    8.0 * drift
                    + impact_penalty
                    + takeoff_balance_penalty
                )
                if not had_flight:
                    score -= 4.0
            else:
                speed_progress = min(takeoff_velocity / target_takeoff_velocity, 1.0)
                air_progress = min(airtime / target_airtime, 1.0)
                # Expected landing load follows the achieved jump, so a high but
                # well-absorbed landing is not punished merely for being high.
                expected_touchdown_velocity = max(
                    landing_velocity,
                    math.sqrt(2.0 * 9.81 * ballistic_rise),
                )
                expected_peak_force = 3.0 * (
                    self.total_mass * 9.81
                    + self.total_mass * expected_touchdown_velocity / 0.12
                )
                impact_excess = max(
                    landing_impact / max(expected_peak_force, 1.0) - 1.0,
                    0.0,
                )
                landing_penalty = (
                    3.0 * min(impact_excess, 2.0)
                    + 2.0 * min(foot_contact_delay / 0.05, 2.0)
                    + 2.0 * landing_force_asymmetry
                    + 3.0 * min(post_landing_speed / 0.50, 2.0)
                    + 2.0 * min(final_tilt / math.radians(12.0), 2.0)
                    + 2.0 * min(abs(support_ratio - 1.0), 1.0)
                    + (8.0 if body_contact_after_takeoff else 0.0)
                )
                # This is the same physical height objective used by resume
                # selection. Only the unsupported flight segment contributes.
                height_score = 1000.0 * max_flight_com_z
                flight_score = 3.0 * speed_progress + 3.0 * air_progress
                meaningful_jump = (
                    takeoff_velocity >= 0.05
                    and ballistic_rise >= 0.003
                    and max_foot_clearance >= 0.001
                    and airtime >= MIN_FLIGHT_STEPS * DT
                )
                if settled:
                    # Once every landing constraint is satisfied, score is
                    # exactly the physical flight-height objective.
                    score = 500.0 + height_score
                elif meaningful_jump and landed:
                    score = 200.0 + height_score + flight_score
                    score -= landing_penalty + takeoff_balance_penalty
                elif meaningful_jump:
                    score = height_score + flight_score
                    score -= takeoff_balance_penalty
                else:
                    score = -100.0 + flight_score
        metrics = JumpMetrics(
            score,
            rise,
            max_base_z,
            max_com_z,
            max_flight_base_z,
            max_flight_com_z,
            max_foot_clearance,
            max_upward_velocity,
            takeoff_velocity,
            takeoff_horizontal_velocity,
            ballistic_rise,
            airtime,
            drift,
            max_tilt,
            peak_force,
            landing_impact,
            landing_velocity,
            foot_contact_delay,
            landing_force_asymmetry,
            post_landing_speed,
            final_speed,
            final_tilt,
            final_support_force,
            final_both_feet_contact,
            body_contact_after_takeoff,
            landed,
            settled,
        )
        return metrics, frames


def parameters_to_dict(parameters):
    return dict(zip(PARAMETER_NAMES, map(float, parameters)))


def metrics_to_dict(metrics):
    return asdict(metrics)
