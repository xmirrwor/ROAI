"""Stable top-level entry point for MiniDuck Isaac Gym training."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_module("legged_panguin.scripts.train", run_name="__main__")
