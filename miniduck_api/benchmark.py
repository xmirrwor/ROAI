from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RaceResult:
    elapsed_s: float
    forward_m: float
    lateral_m: float
    heading_error_deg: float
    finished: bool

    @property
    def average_speed_mps(self) -> float:
        return self.forward_m / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def score(self) -> float:
        """Higher is better; speed is penalized by drift and heading error."""
        penalty = 1.0 + 2.0 * abs(self.lateral_m) + abs(self.heading_error_deg) / 45.0
        return self.average_speed_mps / penalty


class StraightLineRace:
    def __init__(self, distance_m: float = 2.0):
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        self.distance_m = distance_m
        self._start = None

    def start(self, timestamp_s: float, x_m: float, y_m: float, yaw_rad: float) -> None:
        self._start = (timestamp_s, x_m, y_m, yaw_rad)

    def sample(self, timestamp_s: float, x_m: float, y_m: float, yaw_rad: float) -> RaceResult:
        if self._start is None:
            raise RuntimeError("call start() before sample()")
        t0, x0, y0, yaw0 = self._start
        dx, dy = x_m - x0, y_m - y0
        forward = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        lateral = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        heading = math.atan2(math.sin(yaw_rad - yaw0), math.cos(yaw_rad - yaw0))
        return RaceResult(
            elapsed_s=max(0.0, timestamp_s - t0),
            forward_m=forward,
            lateral_m=lateral,
            heading_error_deg=math.degrees(heading),
            finished=forward >= self.distance_m,
        )
