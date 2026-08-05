"""Continuous jump evaluation without resetting between jumps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import pybullet as p

from .jump_sim import (
    CONTACT_FORCE_THRESHOLD,
    DEFAULT_POSE,
    DT,
    LOWER_BOUNDS,
    MIN_FLIGHT_STEPS,
    UPPER_BOUNDS,
    MiniDuckJumpSim,
)


@dataclass
class ContinuousJumpMetrics:
    score: float
    requested_jumps: int
    completed_jumps: int
    flight_count: int
    landing_count: int
    min_flight_peak_com: float
    mean_flight_peak_com: float
    min_foot_clearance: float
    max_drift: float
    max_landing_impact: float
    max_post_landing_speed: float
    body_hit: bool
    all_settled: bool
    cycles: list


class MiniDuckContinuousJumpSim(MiniDuckJumpSim):
    def evaluate_continuous(
        self,
        parameters,
        jumps: int = 3,
        cycle_duration: float = 1.35,
        inter_jump_pause: float = 0.25,
        realtime: bool = False,
        capture: bool = False,
    ):
        parameters = np.clip(np.asarray(parameters, dtype=float), LOWER_BOUNDS, UPPER_BOUNDS)
        crouch = self._pose(parameters[:3])
        extend = self._pose(parameters[3:6])
        tuck = self._pose(parameters[6:9])
        landing_pose = self._pose(parameters[9:12])
        crouch_time, push_time, tuck_time, recovery_time = parameters[12:16]

        self.reset()
        global_initial_com, _ = self._center_of_mass_state()
        frames = []
        cycles = []

        for cycle_index in range(jumps):
            cycle_initial_com, _ = self._center_of_mass_state()
            had_flight = False
            landed = False
            flight_complete = False
            landing_step = None
            airborne_steps = 0
            longest_airborne_steps = 0
            candidate_takeoff_z = 0.0
            takeoff_z = None
            takeoff_velocity = 0.0
            max_flight_com = 0.0
            max_foot_clearance = 0.0
            landing_impact = 0.0
            body_hit = False
            post_landing_speed = None
            previous_vertical_velocity = 0.0

            for local_step in range(int(cycle_duration / DT)):
                t = local_step * DT
                if t < crouch_time:
                    target = self._blend(DEFAULT_POSE, crouch, t / crouch_time)
                elif t < crouch_time + push_time:
                    target = self._blend(
                        crouch, extend, (t - crouch_time) / push_time
                    )
                elif landed:
                    elapsed = (local_step - landing_step) * DT
                    target = self._blend(
                        landing_pose,
                        DEFAULT_POSE,
                        min(elapsed / recovery_time, 1.0),
                    )
                elif had_flight and previous_vertical_velocity < 0.0:
                    target = landing_pose
                else:
                    elapsed = t - crouch_time - push_time
                    target = self._blend(
                        extend, tuck, min(elapsed / tuck_time, 1.0)
                    )

                push_active = crouch_time <= t < crouch_time + push_time
                self._command(target, velocity_gain=0.98 if push_active else 1.0)
                p.stepSimulation(physicsClientId=self.client)

                foot_points = [
                    p.getContactPoints(
                        self.robot,
                        self.plane,
                        linkIndexA=foot,
                        physicsClientId=self.client,
                    )
                    for foot in self.feet
                ]
                foot_forces = [sum(point[9] for point in points) for points in foot_points]
                foot_contacts = [force > CONTACT_FORCE_THRESHOLD for force in foot_forces]
                all_contacts = p.getContactPoints(
                    self.robot, self.plane, physicsClientId=self.client
                )
                total_force = sum(point[9] for point in all_contacts)
                unsupported = total_force <= CONTACT_FORCE_THRESHOLD
                com, com_velocity = self._center_of_mass_state()
                vertical_velocity = float(com_velocity[2])

                if unsupported:
                    if airborne_steps == 0:
                        candidate_takeoff_z = float(com[2])
                        takeoff_velocity = max(vertical_velocity, 0.0)
                    airborne_steps += 1
                    longest_airborne_steps = max(longest_airborne_steps, airborne_steps)
                    if airborne_steps >= MIN_FLIGHT_STEPS and not had_flight:
                        had_flight = True
                        takeoff_z = candidate_takeoff_z
                        max_flight_com = float(com[2])
                    if had_flight and not flight_complete:
                        max_flight_com = max(max_flight_com, float(com[2]))
                        clearance = min(
                            p.getAABB(self.robot, foot, self.client)[0][2]
                            for foot in self.feet
                        )
                        max_foot_clearance = max(max_foot_clearance, max(0.0, clearance))
                else:
                    if had_flight:
                        flight_complete = True
                    airborne_steps = 0

                if had_flight:
                    landing_impact = max(landing_impact, total_force)
                    if any(point[3] not in self.feet for point in all_contacts):
                        body_hit = True
                    if all(foot_contacts) and not landed:
                        landed = True
                        landing_step = local_step
                if (
                    landed
                    and post_landing_speed is None
                    and (local_step - landing_step) * DT >= 0.15
                ):
                    post_landing_speed = float(np.linalg.norm(com_velocity))

                previous_vertical_velocity = vertical_velocity
                if capture and local_step % 4 == 0:
                    position, orientation = p.getBasePositionAndOrientation(
                        self.robot, self.client
                    )
                    frames.append(
                        {
                            "time": (cycle_index * cycle_duration) + t,
                            "base_position": list(position),
                            "base_quaternion": list(orientation),
                            "joint_positions": {
                                name: p.getJointState(
                                    self.robot, index, physicsClientId=self.client
                                )[0]
                                for name, index in self.joints.items()
                            },
                        }
                    )
                if realtime:
                    time.sleep(DT)

            final_com, final_velocity = self._center_of_mass_state()
            drift = float(np.linalg.norm(final_com[:2] - cycle_initial_com[:2]))
            _, orientation = p.getBasePositionAndOrientation(self.robot, self.client)
            roll, pitch, _ = p.getEulerFromQuaternion(orientation)
            final_tilt = max(abs(roll), abs(pitch - 0.451947301626))
            final_forces = [
                sum(
                    point[9]
                    for point in p.getContactPoints(
                        self.robot,
                        self.plane,
                        linkIndexA=foot,
                        physicsClientId=self.client,
                    )
                )
                for foot in self.feet
            ]
            final_speed = float(np.linalg.norm(final_velocity))
            if post_landing_speed is None:
                post_landing_speed = final_speed
            support_ratio = sum(final_forces) / max(self.total_mass * 9.81, 1.0)
            ballistic_rise = (
                max(0.0, max_flight_com - takeoff_z) if takeoff_z is not None else 0.0
            )
            settled = (
                landed
                and all(force > CONTACT_FORCE_THRESHOLD for force in final_forces)
                and not body_hit
                and final_speed < 0.25
                and post_landing_speed < 0.22
                and final_tilt < math.radians(12.0)
                and drift < 0.035
                and 0.60 < support_ratio < 1.40
                and landing_impact < 20.0 * self.total_mass * 9.81
            )
            cycles.append(
                {
                    "index": cycle_index + 1,
                    "flight": had_flight,
                    "landed": landed,
                    "settled": settled,
                    "takeoff_velocity": takeoff_velocity,
                    "ballistic_rise": ballistic_rise,
                    "flight_peak_com": max_flight_com,
                    "foot_clearance": max_foot_clearance,
                    "airborne_time": longest_airborne_steps * DT,
                    "drift": drift,
                    "landing_impact": landing_impact,
                    "post_landing_speed": post_landing_speed,
                    "body_hit": body_hit,
                }
            )
            if not settled:
                break
            if cycle_index + 1 < jumps:
                for _ in range(int(inter_jump_pause / DT)):
                    self._command(DEFAULT_POSE)
                    p.stepSimulation(physicsClientId=self.client)
                    if realtime:
                        time.sleep(DT)

        completed = sum(cycle["settled"] for cycle in cycles)
        flights = sum(cycle["flight"] for cycle in cycles)
        landings = sum(cycle["landed"] for cycle in cycles)
        peaks = [cycle["flight_peak_com"] for cycle in cycles if cycle["flight"]]
        clearances = [cycle["foot_clearance"] for cycle in cycles if cycle["flight"]]
        min_peak = min(peaks, default=0.0)
        mean_peak = float(np.mean(peaks)) if peaks else 0.0
        min_clearance = min(clearances, default=0.0)
        max_drift = max((cycle["drift"] for cycle in cycles), default=0.0)
        max_impact = max(
            (cycle["landing_impact"] for cycle in cycles), default=0.0
        )
        max_post_speed = max(
            (cycle["post_landing_speed"] for cycle in cycles), default=0.0
        )
        score = (
            10000.0 * completed
            + 1000.0 * landings
            + 100.0 * flights
            + 1000.0 * min_peak
            + 100.0 * mean_peak
            - 200.0 * min(max_drift, 10.0)
            - 2.0 * max(max_impact - 300.0, 0.0)
            - 100.0 * min(max_post_speed, 5.0)
        )
        metrics = ContinuousJumpMetrics(
            score=score,
            requested_jumps=jumps,
            completed_jumps=completed,
            flight_count=flights,
            landing_count=landings,
            min_flight_peak_com=min_peak,
            mean_flight_peak_com=mean_peak,
            min_foot_clearance=min_clearance,
            max_drift=max_drift,
            max_landing_impact=max_impact,
            max_post_landing_speed=max_post_speed,
            body_hit=any(cycle["body_hit"] for cycle in cycles),
            all_settled=completed == jumps,
            cycles=cycles,
        )
        return metrics, frames


def continuous_metrics_to_dict(metrics):
    return asdict(metrics)
