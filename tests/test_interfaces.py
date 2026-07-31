import math
import unittest

import numpy as np

from miniduck_api.benchmark import StraightLineRace
from miniduck_api.config import PolicyConfig
from miniduck_api.contracts import Command, ImuSample, JointState, RobotState, SkillMode
from miniduck_api.observation import ObservationBuilder
from miniduck_api.agent import MiniDuckAgent


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PolicyConfig.load("miniduck_stable_policy.json")

    def test_observation_contract(self):
        state = RobotState(0.0, ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]), JointState(self.cfg.default_actuator, [0] * 10))
        builder = ObservationBuilder(self.cfg)
        obs = builder.build(state, Command(vx=0.1))
        self.assertEqual(obs.shape, (64,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertAlmostEqual(float(obs[9]), 0.2)

    def test_zero_command_resets_gait_phase_to_stand(self):
        state = RobotState(
            0.0,
            ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]),
            JointState(self.cfg.default_actuator, [0] * 10),
        )
        builder = ObservationBuilder(self.cfg)
        for _ in range(5):
            builder.push_action(np.zeros(10, dtype=np.float32))
        moving = builder.build(state, Command(vx=0.08))
        self.assertNotAlmostEqual(float(moving[-1]), 0.0)
        stopped = builder.build(state, Command())
        self.assertAlmostEqual(float(stopped[-2]), 1.0)
        self.assertAlmostEqual(float(stopped[-1]), 0.0)

    def test_squat_command_encodes_target_without_changing_contract(self):
        state = RobotState(
            0.0,
            ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]),
            JointState(self.cfg.default_actuator, [0] * 10),
        )
        builder = ObservationBuilder(self.cfg)
        obs = builder.build(
            state,
            Command(
                skill=SkillMode.SQUAT,
                body_height_m=self.cfg.squat_body_height_m,
            ),
        )
        self.assertEqual(obs.shape, (64,))
        self.assertAlmostEqual(float(obs[-2]), -1.0)
        self.assertAlmostEqual(float(obs[-1]), 1.0)

    def test_recovery_command_uses_reserved_skill_code(self):
        state = RobotState(
            0.0,
            ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]),
            JointState(self.cfg.default_actuator, [0] * 10),
        )
        builder = ObservationBuilder(self.cfg)
        obs = builder.build(state, Command(skill=SkillMode.RECOVERY))
        self.assertEqual(obs.shape, (64,))
        self.assertAlmostEqual(float(obs[-2]), -1.0)
        self.assertAlmostEqual(float(obs[-1]), -1.0)

    def test_obstacle_command_preserves_phase_and_encodes_distance(self):
        state = RobotState(
            0.0,
            ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]),
            JointState(self.cfg.default_actuator, [0] * 10),
        )
        builder = ObservationBuilder(self.cfg)
        obs = builder.build(
            state,
            Command(
                vx=0.12,
                skill=SkillMode.OBSTACLE,
                obstacle_distance_m=0.10,
                obstacle_height_m=0.015,
            ),
        )
        self.assertEqual(obs.shape, (64,))
        self.assertAlmostEqual(float(obs[10]), 0.125)
        self.assertAlmostEqual(float(obs[-2]), 1.0)
        self.assertAlmostEqual(float(obs[-1]), 0.0)

    def test_straight_command_uses_gyro_heading_hold(self):
        state = RobotState(
            0.0,
            ImuSample([0, 0, 0.2], [0, 0, -1], [0, 0, 0]),
            JointState(self.cfg.default_actuator, [0] * 10),
        )
        builder = ObservationBuilder(self.cfg)
        obs = builder.build(state, Command(vx=0.08))
        self.assertLess(float(obs[11]), 0.0)

    def test_race_coordinates(self):
        race = StraightLineRace(2.0)
        race.start(1.0, 0.0, 0.0, math.pi / 2)
        result = race.sample(5.0, -0.1, 2.0, math.pi / 2 + 0.05)
        self.assertTrue(result.finished)
        self.assertAlmostEqual(result.forward_m, 2.0)
        self.assertAlmostEqual(result.lateral_m, 0.1)

    def test_agent_step(self):
        class ZeroPolicy:
            def infer(self, observation):
                return np.zeros(10, dtype=np.float32)

            def joint_targets(self, action):
                return np.asarray(self.default, dtype=np.float32)

        policy = ZeroPolicy()
        policy.default = self.cfg.default_actuator
        agent = MiniDuckAgent(policy, self.cfg)
        state = RobotState(0.0, ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]), JointState(self.cfg.default_actuator, [0] * 10))
        output = agent.act(state, Command(vx=0.1))
        self.assertEqual(output.observation.shape, (64,))
        self.assertEqual(output.action.shape, (10,))
        np.testing.assert_allclose(output.joint_targets, self.cfg.default_actuator)

    def test_agent_limits_command_and_joint_target_rate(self):
        class UnsafePolicy:
            def infer(self, observation):
                self.observation = observation.copy()
                return np.ones(10, dtype=np.float32)

            def joint_targets(self, action):
                return np.asarray(self.default, dtype=np.float32) + 10.0

        policy = UnsafePolicy()
        policy.default = self.cfg.default_actuator
        agent = MiniDuckAgent(policy, self.cfg)
        state = RobotState(
            0.0,
            ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0]),
            JointState(self.cfg.default_actuator, [0] * 10),
        )
        output = agent.act(state, Command(vx=99.0, vy=-99.0, yaw_rate=99.0))
        self.assertAlmostEqual(
            float(policy.observation[9]),
            self.cfg.command_ranges["lin_vel_x"][1] * self.cfg.obs_scales["command"][0],
        )
        max_delta = self.cfg.nominal_motor_velocity * self.cfg.policy_dt
        np.testing.assert_allclose(
            output.joint_targets,
            np.asarray(self.cfg.default_actuator) + max_delta,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
