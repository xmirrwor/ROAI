from pathlib import Path

from miniduck_api.config import PolicyConfig
from miniduck_real.config import HardwareConfig


ROOT = Path(__file__).resolve().parents[1]
POLICY_CONFIG = ROOT / "miniduck_stable_policy.json"
HARDWARE_CONFIG = ROOT / "config" / "robot.json"


def main() -> None:
    policy = PolicyConfig.load(POLICY_CONFIG)
    hardware = HardwareConfig.load(HARDWARE_CONFIG)
    if policy.joint_names != hardware.joint_names:
        raise RuntimeError(
            "policy and hardware joint orders differ:\n"
            f"policy:   {policy.joint_names}\n"
            f"hardware: {hardware.joint_names}"
        )
    pending = [joint.name for joint in hardware.joints if not joint.calibrated]
    print("Configuration structure: OK")
    print("Servo IDs in policy order:", hardware.servo_ids)
    if pending:
        print("Calibration pending:", ", ".join(pending))
    else:
        print("All joints are calibrated.")


if __name__ == "__main__":
    main()
