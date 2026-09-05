"""M3 GPS-denied sensor models: IMU, optical-flow-like velocity, ToF height.

These are ENGINEERING TEST MODELS, not hardware datasheets. The rates,
noise, bias, scale, latency and dropout values are chosen to expose
sensor error -> estimation error -> mission error in a controlled way.
They are NOT validated MicoAir MTF-02P performance figures.

Boundary rule: sensors may read physical truth internally (that is what
a simulator is for), but only the small sample dataclasses (IMUSample,
FlowSample, ToFSample) may leave this module. None of them contains
absolute position, absolute velocity in world frame, or true attitude.

Coordinate conventions (match fly.py / simulator.py):
- world: MuJoCo world, +Z up, gravity (0, 0, -9.81)
- body: quadrotor body frame, +X forward, +Z up
- specific force: f_body = R_world_to_body @ (a_world - g_world);
  a stationary level vehicle reads (0, 0, +9.81) -- the accelerometer
  measures the support force, i.e. "up", at rest.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])


@dataclass(frozen=True)
class TruthKinematics:
    """Everything the sensor simulation is allowed to know.

    Constructed once per physics step by the simulator. The estimator
    never sees this object.
    """

    time: float
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    rotation_body_to_world: np.ndarray
    velocity_world: np.ndarray
    angular_velocity_body: np.ndarray
    acceleration_world: np.ndarray
    ground_height: float


@dataclass(frozen=True)
class IMUSample:
    time: float
    gyro_body: np.ndarray
    specific_force_body: np.ndarray
    valid: bool = True


@dataclass(frozen=True)
class FlowSample:
    time: float  # measurement timestamp (sim time)
    release_time: float  # when the estimator received it (after latency)
    velocity_body_xy: np.ndarray
    valid: bool


@dataclass(frozen=True)
class ToFSample:
    time: float
    release_time: float
    height_m: float
    valid: bool


@dataclass(frozen=True)
class IMUConfig:
    rate_hz: float = 250.0
    gyro_noise_sigma: float = 0.002  # rad/s per axis
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s
    accel_noise_sigma: float = 0.05  # m/s^2 per axis
    accel_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s^2


@dataclass(frozen=True)
class FlowConfig:
    rate_hz: float = 50.0
    noise_sigma: float = 0.03  # m/s per axis
    scale: float = 1.0
    bias: tuple[float, float] = (0.0, 0.0)  # m/s, body frame
    latency: float = 0.0  # s
    dropouts: tuple[tuple[float, float], ...] = ()  # (start_s, duration_s)


@dataclass(frozen=True)
class ToFConfig:
    rate_hz: float = 50.0
    noise_sigma: float = 0.01  # m
    bias: float = 0.0  # m
    latency: float = 0.0  # s
    dropouts: tuple[tuple[float, float], ...] = ()
    min_range: float = 0.05  # m
    max_range: float = 8.0  # m


def _in_dropout(time: float, dropouts: tuple[tuple[float, float], ...]) -> bool:
    return any(start <= time < start + duration for start, duration in dropouts)


class IMUSensor:
    """Gyro + specific force at a fixed rate divider of the physics loop."""

    def __init__(self, config: IMUConfig, physics_dt: float, seed: int) -> None:
        self.config = config
        self.physics_dt = physics_dt
        self.divider = max(1, round(1.0 / (config.rate_hz * physics_dt)))
        self.rng = np.random.default_rng(seed)
        self._steps = 0

    def update(self, kin: TruthKinematics) -> IMUSample | None:
        sample = None
        if self._steps % self.divider == 0:
            config = self.config
            gyro = (
                kin.angular_velocity_body
                + np.asarray(config.gyro_bias)
                + self.rng.normal(0.0, config.gyro_noise_sigma, 3)
            )
            specific_force_world = kin.acceleration_world - GRAVITY_WORLD
            specific_force_body = (
                kin.rotation_body_to_world.T @ specific_force_world
            )
            specific_force_body = (
                specific_force_body
                + np.asarray(config.accel_bias)
                + self.rng.normal(0.0, config.accel_noise_sigma, 3)
            )
            sample = IMUSample(
                time=kin.time,
                gyro_body=gyro,
                specific_force_body=specific_force_body,
            )
        self._steps += 1
        return sample


class FlowSensor:
    """Optical-flow-LIKE horizontal body-frame velocity.

    This is NOT image-level optical flow: no camera rendering, no feature
    tracking. It abstracts a downward flow module as a direct body-frame
    velocity measurement with scale error, bias, noise, latency and
    dropouts.
    """

    def __init__(self, config: FlowConfig, physics_dt: float, seed: int) -> None:
        self.config = config
        self.divider = max(1, round(1.0 / (config.rate_hz * physics_dt)))
        self.rng = np.random.default_rng(seed)
        self._steps = 0
        self._queue: deque[tuple[float, tuple[np.ndarray, bool]]] = deque()

    def update(self, kin: TruthKinematics) -> FlowSample | None:
        config = self.config
        released: tuple[np.ndarray, bool, float] | None = None
        while self._queue and self._queue[0][0] <= kin.time + 1e-12:
            release_time, (value, valid) = self._queue.popleft()
            released = (value, valid, release_time)
        if self._steps % self.divider == 0:
            velocity_body = kin.rotation_body_to_world.T @ kin.velocity_world
            value = (
                config.scale * velocity_body[:2]
                + np.asarray(config.bias)
                + self.rng.normal(0.0, config.noise_sigma, 2)
            )
            valid = not _in_dropout(kin.time, config.dropouts)
            if config.latency <= 0.0:
                released = (value, valid, kin.time)
            else:
                self._queue.append((kin.time + config.latency, (value, valid)))
        self._steps += 1
        if released is None:
            return None
        value, valid, release_time = released
        return FlowSample(
            time=release_time - config.latency,
            release_time=release_time,
            velocity_body_xy=value,
            valid=valid,
        )


class ToFSensor:
    """Downward rangefinder: height of the body origin above the ground plane."""

    def __init__(self, config: ToFConfig, physics_dt: float, seed: int) -> None:
        self.config = config
        self.divider = max(1, round(1.0 / (config.rate_hz * physics_dt)))
        self.rng = np.random.default_rng(seed)
        self._steps = 0
        self._queue: deque[tuple[float, tuple[float, bool]]] = deque()

    def update(self, kin: TruthKinematics) -> ToFSample | None:
        config = self.config
        released: tuple[float, bool, float] | None = None
        while self._queue and self._queue[0][0] <= kin.time + 1e-12:
            release_time, (value, valid) = self._queue.popleft()
            released = (value, valid, release_time)
        if self._steps % self.divider == 0:
            height = float(kin.position[2]) - kin.ground_height
            in_range = config.min_range <= height <= config.max_range
            valid = in_range and not _in_dropout(kin.time, config.dropouts)
            value = (
                height + config.bias + float(self.rng.normal(0.0, config.noise_sigma))
            )
            if config.latency <= 0.0:
                released = (value, valid, kin.time)
            else:
                self._queue.append((kin.time + config.latency, (value, valid)))
        self._steps += 1
        if released is None:
            return None
        value, valid, release_time = released
        return ToFSample(
            time=release_time - config.latency,
            release_time=release_time,
            height_m=value,
            valid=valid,
        )
