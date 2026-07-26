from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from miniduck_api.config import PolicyConfig
from miniduck_api.policy import PolicyRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the ONNX/team interface contract")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    cfg = PolicyConfig.load(args.metadata)
    runner = PolicyRunner(args.model, cfg)
    action = runner.infer(np.zeros(cfg.obs_size, dtype=np.float32))
    targets = runner.joint_targets(action)
    print(f"OK obs={cfg.obs_size} action={action.shape[0]} dt={cfg.policy_dt:.3f}s")
    print("target range(rad):", float(targets.min()), float(targets.max()))


if __name__ == "__main__":
    main()
