"""Stable top-level entry point for MiniDuck Isaac Gym playback."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_module("legged_panguin.scripts.play_miniduck", run_name="__main__")
