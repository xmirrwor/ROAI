from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PolicyConfig


class PolicyRunner:
    """ONNX inference wrapper shared by simulation tools and the real robot."""

    def __init__(self, model_path: str | Path, config: PolicyConfig):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime to run the exported policy") from exc
        self.config = config
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, self.config.obs_size)
        action = self.session.run(None, {self.input_name: obs})[0][0]
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def joint_targets(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.config.action_size,):
            raise ValueError("invalid action shape")
        return np.asarray(self.config.default_actuator, dtype=np.float32) + self.config.action_scale * action

