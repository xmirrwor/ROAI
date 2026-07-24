"""Template owned by the hardware teammate; replace TODOs with the robot SDK."""

from miniduck_api.contracts import RobotState


class MiniDuckHardwareAdapter:
    def read_state(self) -> RobotState:
        raise NotImplementedError("Read timestamp, IMU and 10 joints from the robot SDK")

    def write_joint_targets(self, targets_rad) -> None:
        raise NotImplementedError("Send 10 joint position targets in metadata order")

    def disable_torque(self) -> None:
        raise NotImplementedError("Immediately disable all motor torque")
