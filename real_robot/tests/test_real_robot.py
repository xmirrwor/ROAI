import unittest

import numpy as np

from miniduck_api.contracts import ImuSample
from miniduck_real.adapter import MiniDuckHardwareAdapter
from miniduck_real.config import HardwareConfig


class FakeBus:
    def __init__(self):
        self.positions = np.zeros(10)
        self.velocities = np.zeros(10)
        self.written = None
        self.disabled = False

    def read_positions_rad(self, servo_ids):
        return self.positions

    def read_velocities_rad_s(self, servo_ids):
        return self.velocities

    def write_positions_rad(self, servo_ids, targets_rad):
        self.written = np.asarray(targets_rad)

    def disable_torque(self, servo_ids):
        self.disabled = True


class FakeImu:
    def read_imu(self):
        return ImuSample([0, 0, 0], [0, 0, -1], [0, 0, 0])


class RealRobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = HardwareConfig.load("config/robot.json")

    def test_joint_ids_follow_policy_order(self):
        self.assertEqual(
            self.cfg.servo_ids,
            (20, 21, 22, 23, 24, 10, 11, 12, 13, 14),
        )

    def test_uncalibrated_config_is_blocked_by_default(self):
        with self.assertRaises(RuntimeError):
            MiniDuckHardwareAdapter(FakeBus(), FakeImu(), self.cfg)

    def test_targets_are_limited_and_rate_limited(self):
        bus = FakeBus()
        adapter = MiniDuckHardwareAdapter(
            bus, FakeImu(), self.cfg, require_calibrated=False
        )
        adapter.write_joint_targets(np.zeros(10))
        adapter.write_joint_targets(np.ones(10))
        np.testing.assert_allclose(bus.written, np.full(10, 0.03))

    def test_disable_torque_reaches_bus(self):
        bus = FakeBus()
        adapter = MiniDuckHardwareAdapter(
            bus, FakeImu(), self.cfg, require_calibrated=False
        )
        adapter.disable_torque()
        self.assertTrue(bus.disabled)


if __name__ == "__main__":
    unittest.main()
