"""M3 minimal GPS-denied localization estimator.

This is deliberately NOT a Kalman filter. It is an understandable
predict/correct chain:

    IMU gyro          -> attitude propagation (quaternion integration)
    IMU specific force -> inertial velocity/position prediction
    optical flow       -> horizontal velocity correction (complementary)
    ToF rangefinder    -> altitude (+ vertical velocity) correction

Frames: the estimator works in its own LOCAL navigation frame. At
startup x = y = 0 and yaw = 0 by definition; local +X is the vehicle's
initial forward direction. There is no magnetometer, so gyro z bias
produces real yaw drift (intentional M3 learning case). There is no
horizontal position sensor, so x/y exist only by integrating the
estimated velocity -- flow scale/bias errors accumulate into real
position error. If x/y ever exactly copy MuJoCo truth, something is
leaking and M3 has failed.

Sensor values here are engineering test values, NOT MTF-02P datasheet
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from fly import QuadrotorState
from m3_sensors import (
    FlowConfig,
    FlowSample,
    FlowSensor,
    IMUConfig,
    IMUSample,
    IMUSensor,
    ToFConfig,
    ToFSample,
    ToFSensor,
    TruthKinematics,
)


GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])


@dataclass(frozen=True)
class EstimatorConfig:
    alpha_flow: float = 0.35  # horizontal velocity complementary gain
    alpha_tof: float = 0.5  # altitude complementary gain
    imu_stale_s: float = 0.05
    flow_stale_s: float = 0.50
    tof_stale_s: float = 0.50


@dataclass(frozen=True)
class EstimatedState:
    """Localization output in the estimator's local navigation frame."""

    time: float
    position: np.ndarray  # local nav frame, z = height above ground
    velocity: np.ndarray  # local nav frame
    quaternion_wxyz: np.ndarray  # body -> local nav frame
    rotation_body_to_world: np.ndarray  # named to match QuadrotorState
    angular_velocity_body: np.ndarray  # gyro measurement passthrough
    healthy: bool
    imu_age: float
    flow_age: float
    tof_age: float

    def to_quadrotor_state(self) -> QuadrotorState:
        """Adapter into the existing controller/mission state type."""
        return QuadrotorState(
            self.position.copy(),
            self.velocity.copy(),
            self.quaternion_wxyz.copy(),
            self.rotation_body_to_world.copy(),
            self.angular_velocity_body.copy(),
        )


class LocalizationEstimator:
    """Predict (IMU) + correct (flow, ToF) estimator in a local frame."""

    def __init__(self, config: EstimatorConfig) -> None:
        self.config = config
        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity_body = np.zeros(3, dtype=float)
        self._last_imu_time: float | None = None
        self._last_flow_time: float | None = None
        self._last_tof_time: float | None = None
        self._last_tof_height: float | None = None

    # ---------------- prediction (IMU) ----------------

    def predict(self, sample: IMUSample) -> None:
        if not sample.valid:
            return
        if self._last_imu_time is None:
            # First sample anchors time; no integration yet.
            self._last_imu_time = sample.time
            self.angular_velocity_body = sample.gyro_body.copy()
            return
        dt = sample.time - self._last_imu_time
        self._last_imu_time = sample.time
        if dt <= 0.0:
            return

        # Attitude propagation: q <- q ⊗ exp(0.5 * omega * dt), using the
        # exact small-rotation quaternion (no first-order approximation).
        omega = sample.gyro_body
        angle = float(np.linalg.norm(omega)) * dt
        if angle > 0.0:
            axis = omega / float(np.linalg.norm(omega))
            half = 0.5 * angle
            delta = np.concatenate(
                ([math.cos(half)], math.sin(half) * axis)
            )
            mujoco.mju_mulQuat(self.quaternion_wxyz, self.quaternion_wxyz, delta)
            self.quaternion_wxyz /= np.linalg.norm(self.quaternion_wxyz)
        self.angular_velocity_body = omega.copy()

        # Inertial prediction: a_nav = R(q_est) f_body + g_world.
        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation_flat, self.quaternion_wxyz)
        rotation = rotation_flat.reshape(3, 3)
        acceleration_nav = rotation @ sample.specific_force_body + GRAVITY_WORLD
        self.position += self.velocity * dt + 0.5 * acceleration_nav * dt * dt
        self.velocity += acceleration_nav * dt

    # ---------------- corrections ----------------

    def correct_flow(self, sample: FlowSample) -> None:
        if not sample.valid:
            return  # staleness is what the health monitor sees
        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation_flat, self.quaternion_wxyz)
        rotation = rotation_flat.reshape(3, 3)
        flow_body = np.array(
            [sample.velocity_body_xy[0], sample.velocity_body_xy[1], 0.0]
        )
        velocity_flow_nav = rotation @ flow_body
        alpha = self.config.alpha_flow
        self.velocity[:2] = (
            (1.0 - alpha) * self.velocity[:2] + alpha * velocity_flow_nav[:2]
        )
        self._last_flow_time = sample.time

    def correct_tof(self, sample: ToFSample) -> None:
        if not sample.valid:
            return
        alpha = self.config.alpha_tof
        self.position[2] = (
            (1.0 - alpha) * self.position[2] + alpha * sample.height_m
        )
        # vz stays purely inertial. A ToF finite-difference vz correction
        # was tried and REJECTED: differencing sigma=0.01 m heights at
        # 50 Hz produces sigma ~ 0.7 m/s velocity noise, which alone
        # violates the mission's |vz| < 0.12 m/s settle gate.
        self._last_tof_time = sample.time
        self._last_tof_height = sample.height_m

    # ---------------- output ----------------

    def state(self, time: float) -> EstimatedState:
        def age(last: float | None) -> float:
            return math.inf if last is None else max(0.0, time - last)

        imu_age = age(self._last_imu_time)
        flow_age = age(self._last_flow_time)
        tof_age = age(self._last_tof_time)
        finite = (
            np.all(np.isfinite(self.position))
            and np.all(np.isfinite(self.velocity))
            and np.all(np.isfinite(self.quaternion_wxyz))
        )
        healthy = bool(
            finite
            and math.isfinite(imu_age)
            and imu_age <= self.config.imu_stale_s
            and math.isfinite(flow_age)
            and flow_age <= self.config.flow_stale_s
            and math.isfinite(tof_age)
            and tof_age <= self.config.tof_stale_s
        )
        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation_flat, self.quaternion_wxyz)
        return EstimatedState(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion_wxyz=self.quaternion_wxyz.copy(),
            rotation_body_to_world=rotation_flat.reshape(3, 3),
            angular_velocity_body=self.angular_velocity_body.copy(),
            healthy=healthy,
            imu_age=imu_age,
            flow_age=flow_age,
            tof_age=tof_age,
        )


class LocalizationPipeline:
    """Sensors + estimator wired together; the M3 navigation boundary.

    update() consumes TruthKinematics (the ONLY place truth enters the
    navigation chain -- sensors are simulators) and returns an
    EstimatedState built purely from sensor samples.
    """

    def __init__(
        self,
        imu: IMUSensor,
        flow: FlowSensor,
        tof: ToFSensor,
        estimator: LocalizationEstimator,
    ) -> None:
        self.imu = imu
        self.flow = flow
        self.tof = tof
        self.estimator = estimator
        self.imu_samples = 0
        self.flow_samples = 0
        self.flow_invalid = 0
        self.tof_samples = 0
        self.tof_invalid = 0
        self.unhealthy_transitions = 0
        self._was_healthy: bool | None = None

    @classmethod
    def from_configs(
        cls,
        imu_config: IMUConfig,
        flow_config: FlowConfig,
        tof_config: ToFConfig,
        estimator_config: EstimatorConfig,
        physics_dt: float,
        seed: int,
    ) -> "LocalizationPipeline":
        # Independent but deterministic noise streams per sensor.
        seeds = np.random.SeedSequence(seed).generate_state(3)
        return cls(
            IMUSensor(imu_config, physics_dt, int(seeds[0])),
            FlowSensor(flow_config, physics_dt, int(seeds[1])),
            ToFSensor(tof_config, physics_dt, int(seeds[2])),
            LocalizationEstimator(estimator_config),
        )

    def update(self, kin: TruthKinematics) -> EstimatedState:
        imu_sample = self.imu.update(kin)
        if imu_sample is not None:
            self.imu_samples += 1
            self.estimator.predict(imu_sample)
        flow_sample = self.flow.update(kin)
        if flow_sample is not None:
            self.flow_samples += 1
            self.flow_invalid += 0 if flow_sample.valid else 1
            self.estimator.correct_flow(flow_sample)
        tof_sample = self.tof.update(kin)
        if tof_sample is not None:
            self.tof_samples += 1
            self.tof_invalid += 0 if tof_sample.valid else 1
            self.estimator.correct_tof(tof_sample)
        state = self.estimator.state(kin.time)
        if self._was_healthy is True and not state.healthy:
            self.unhealthy_transitions += 1
        self._was_healthy = state.healthy
        return state
