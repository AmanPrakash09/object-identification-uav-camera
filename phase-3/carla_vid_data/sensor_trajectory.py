"""Overhead / oblique RGB sensor path: altitude profiles and optional world-space translation."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import carla


class OverheadSensorTrajectory:
    """
    Each ``step(dt)`` returns the transform for the current tick (position before integrating),
    then advances XY by ``velocity * dt`` and bumps the frame index for altitude scheduling.

    XY motion is optional (zero velocity = hover). Optional square bounds around ``anchor``
    reflect velocity at the edge so the sensor patrols back and forth.
    """

    def __init__(
        self,
        anchor: carla.Location,
        *,
        velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        bound_half_extent_xy: Optional[float] = None,
        reflect_at_bounds: bool = True,
        pitch_deg: float = -90.0,
        yaw_deg: float = 0.0,
        yaw_follow_velocity: bool = False,
        height_mode: str = "fixed",
        height_fixed_m: float = 55.0,
        height_cycle_m: Sequence[float] = (40.0, 70.0, 100.0),
        height_cycle_frames: int = 120,
        height_oscillate_min_m: float = 35.0,
        height_oscillate_max_m: float = 95.0,
        height_oscillate_period_s: float = 28.0,
    ) -> None:
        self._ax = anchor.x
        self._ay = anchor.y
        self._ground_z = anchor.z
        self._x = anchor.x
        self._y = anchor.y
        self._vx, self._vy, self._vz = velocity_mps
        self._bound = bound_half_extent_xy
        self._reflect = reflect_at_bounds
        self._pitch = pitch_deg
        self._yaw_fixed = yaw_deg
        self._yaw_follow_vel = yaw_follow_velocity
        self._h_mode = height_mode
        self._h_fixed = height_fixed_m
        self._h_cycle = tuple(height_cycle_m)
        self._h_cycle_n = max(int(height_cycle_frames), 1)
        self._h_lo = height_oscillate_min_m
        self._h_hi = height_oscillate_max_m
        self._h_T = max(float(height_oscillate_period_s), 1e-3)
        self._frame = 0

    def _height_m_at_frame(self, frame: int, dt: float) -> float:
        if self._h_mode == "fixed":
            return self._h_fixed
        if self._h_mode == "cycle":
            if not self._h_cycle:
                return self._h_fixed
            idx = (frame // self._h_cycle_n) % len(self._h_cycle)
            return float(self._h_cycle[idx])
        t = frame * dt
        mid = (self._h_lo + self._h_hi) / 2.0
        amp = (self._h_hi - self._h_lo) / 2.0
        return mid + amp * math.sin(2.0 * math.pi * t / self._h_T)

    def _yaw(self) -> float:
        if self._yaw_follow_vel:
            spd = math.hypot(self._vx, self._vy)
            if spd > 0.25:
                return math.degrees(math.atan2(self._vy, self._vx))
        return self._yaw_fixed

    def initial_transform(self, dt: float) -> carla.Transform:
        """Sensor pose before the first ``world.tick`` (does not advance state)."""
        z_off = self._height_m_at_frame(0, dt)
        r = carla.Rotation(pitch=self._pitch, yaw=self._yaw(), roll=0.0)
        return carla.Transform(
            carla.Location(self._x, self._y, self._ground_z + z_off),
            r,
        )

    def step(self, dt: float) -> carla.Transform:
        z_off = self._height_m_at_frame(self._frame, dt)
        yaw = self._yaw()
        loc = carla.Location(self._x, self._y, self._ground_z + z_off)
        tf = carla.Transform(loc, carla.Rotation(pitch=self._pitch, yaw=yaw, roll=0.0))

        self._x += self._vx * dt
        self._y += self._vy * dt
        if self._bound is not None:
            dx = self._x - self._ax
            dy = self._y - self._ay
            if abs(dx) > self._bound:
                if self._reflect:
                    self._vx *= -1.0
                self._x = self._ax + (self._bound if dx > 0 else -self._bound)
            dy = self._y - self._ay
            if abs(dy) > self._bound:
                if self._reflect:
                    self._vy *= -1.0
                self._y = self._ay + (self._bound if dy > 0 else -self._bound)

        self._frame += 1
        return tf


# Backwards compatibility for any external references
OverheadCameraRig = OverheadSensorTrajectory
