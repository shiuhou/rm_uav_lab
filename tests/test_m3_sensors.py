"""M3 sensor model unit tests.

The sensors may read MuJoCo truth INTERNALLY (a simulator must generate
measurements from reality), but only the small sample dataclasses may
cross the sensor boundary. These tests pin down:

- IMU gyro / specific-force conventions (including the +g at rest sign)
- body-frame transforms of the optical-flow-like velocity sensor
- ToF height definition, valid range
- deterministic seeding, deterministic sample rates
- latency via timestamped queues (no wall-clock sleeps)
- dropout -> valid=False
"""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from m3_sensors import (
    FlowConfig,
    FlowSensor,
    IMUConfig,
    IMUSensor,
    ToFConfig,
    ToFSensor,
    TruthKinematics,
)

PHYSICS_DT = 0.002
GRAVITY = 9.81


def quat_to_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(flat, np.asarray(quat_wxyz, dtype=float))
    return flat.reshape(3, 3)


def make_truth(
    time: float = 0.0,
    position=(0.0, 0.0, 1.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    velocity_world=(0.0, 0.0, 0.0),
    angular_velocity_body=(0.0, 0.0, 0.0),
    acceleration_world=(0.0, 0.0, 0.0),
    ground_height: float = 0.0,
) -> TruthKinematics:
    quat = np.asarray(quat_wxyz, dtype=float)
    return TruthKinematics(
        time=float(time),
        position=np.asarray(position, dtype=float),
        quaternion_wxyz=quat,
        rotation_body_to_world=quat_to_rotation(quat),
        velocity_world=np.asarray(velocity_world, dtype=float),
        angular_velocity_body=np.asarray(angular_velocity_body, dtype=float),
        acceleration_world=np.asarray(acceleration_world, dtype=float),
        ground_height=float(ground_height),
    )


def yaw_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))


def roll_quat(roll_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(roll_rad / 2), math.sin(roll_rad / 2), 0.0, 0.0)


class IMUSensorTest(unittest.TestCase):
    def test_stationary_level_gyro_zero_specific_force_plus_g_body_z(self) -> None:
        # Level at rest: accelerometer measures the specific force
        # f = R^T (a - g) = -g_world = +9.81 along body +Z. This sign
        # convention (accelerometer reads "up" when stationary) is pinned
        # by this test; do not flip it casually.
        imu = IMUSensor(
            IMUConfig(
                gyro_noise_sigma=0.0,
                accel_noise_sigma=0.0,
            ),
            physics_dt=PHYSICS_DT,
            seed=0,
        )
        sample = imu.update(make_truth())
        self.assertIsNotNone(sample)
        np.testing.assert_allclose(sample.gyro_body, (0.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(
            sample.specific_force_body, (0.0, 0.0, GRAVITY), atol=1e-9
        )
        self.assertTrue(sample.valid)

    def test_specific_force_includes_linear_acceleration(self) -> None:
        imu = IMUSensor(
            IMUConfig(gyro_noise_sigma=0.0, accel_noise_sigma=0.0),
            physics_dt=PHYSICS_DT,
            seed=0,
        )
        sample = imu.update(make_truth(acceleration_world=(1.0, 2.0, 3.0)))
        np.testing.assert_allclose(
            sample.specific_force_body, (1.0, 2.0, 3.0 + GRAVITY), atol=1e-9
        )

    def test_specific_force_rotates_into_body_frame(self) -> None:
        # Rolled 90 deg: world +Z specific force appears along body +Y.
        imu = IMUSensor(
            IMUConfig(gyro_noise_sigma=0.0, accel_noise_sigma=0.0),
            physics_dt=PHYSICS_DT,
            seed=0,
        )
        sample = imu.update(make_truth(quat_wxyz=roll_quat(math.pi / 2)))
        np.testing.assert_allclose(
            sample.specific_force_body, (0.0, GRAVITY, 0.0), atol=1e-9
        )

    def test_gyro_follows_true_body_rate(self) -> None:
        imu = IMUSensor(
            IMUConfig(gyro_noise_sigma=0.0, accel_noise_sigma=0.0),
            physics_dt=PHYSICS_DT,
            seed=0,
        )
        sample = imu.update(
            make_truth(angular_velocity_body=(0.1, -0.2, 0.3))
        )
        np.testing.assert_allclose(sample.gyro_body, (0.1, -0.2, 0.3), atol=1e-12)

    def test_free_fall_specific_force_is_zero(self) -> None:
        imu = IMUSensor(
            IMUConfig(gyro_noise_sigma=0.0, accel_noise_sigma=0.0),
            physics_dt=PHYSICS_DT,
            seed=0,
        )
        sample = imu.update(make_truth(acceleration_world=(0.0, 0.0, -GRAVITY)))
        np.testing.assert_allclose(
            sample.specific_force_body, (0.0, 0.0, 0.0), atol=1e-9
        )

    def test_same_seed_bit_identical_stream(self) -> None:
        config = IMUConfig()
        first = IMUSensor(config, physics_dt=PHYSICS_DT, seed=7)
        second = IMUSensor(config, physics_dt=PHYSICS_DT, seed=7)
        for step in range(40):
            kin = make_truth(time=step * PHYSICS_DT)
            a = first.update(kin)
            b = second.update(kin)
            self.assertEqual(a is None, b is None)
            if a is not None:
                np.testing.assert_array_equal(a.gyro_body, b.gyro_body)
                np.testing.assert_array_equal(
                    a.specific_force_body, b.specific_force_body
                )

    def test_different_seed_different_stream(self) -> None:
        first = IMUSensor(IMUConfig(), physics_dt=PHYSICS_DT, seed=1)
        second = IMUSensor(IMUConfig(), physics_dt=PHYSICS_DT, seed=2)
        a = first.update(make_truth())
        b = second.update(make_truth())
        self.assertFalse(np.allclose(a.gyro_body, b.gyro_body))

    def test_constant_bias_appears_in_expected_channel(self) -> None:
        imu = IMUSensor(
            IMUConfig(
                gyro_noise_sigma=0.0,
                accel_noise_sigma=0.0,
                gyro_bias=(0.01, -0.02, 0.03),
                accel_bias=(0.1, 0.2, 0.3),
            ),
            physics_dt=PHYSICS_DT,
            seed=0,
        )
        sample = imu.update(make_truth())
        np.testing.assert_allclose(
            sample.gyro_body, (0.01, -0.02, 0.03), atol=1e-12
        )
        np.testing.assert_allclose(
            sample.specific_force_body, (0.1, 0.2, GRAVITY + 0.3), atol=1e-9
        )

    def test_sample_rate_is_deterministic_every_two_physics_steps(self) -> None:
        imu = IMUSensor(IMUConfig(), physics_dt=PHYSICS_DT, seed=0)
        samples = 0
        for step in range(100):
            if imu.update(make_truth(time=step * PHYSICS_DT)) is not None:
                samples += 1
        self.assertEqual(samples, 50)  # 250 Hz at 500 Hz physics


class FlowSensorTest(unittest.TestCase):
    def make_sensor(self, **kwargs) -> FlowSensor:
        defaults = dict(noise_sigma=0.0)
        defaults.update(kwargs)
        return FlowSensor(FlowConfig(**defaults), physics_dt=PHYSICS_DT, seed=0)

    def test_yaw_zero_world_forward_is_body_forward(self) -> None:
        flow = self.make_sensor()
        sample = flow.update(make_truth(velocity_world=(1.0, 0.0, 0.0)))
        np.testing.assert_allclose(sample.velocity_body_xy, (1.0, 0.0), atol=1e-12)

    def test_yaw90_world_y_is_body_forward(self) -> None:
        # Facing world +Y, physical forward motion (0, 1, 0) world must
        # read as body +X. This is the optical-flow frame contract.
        flow = self.make_sensor()
        sample = flow.update(
            make_truth(
                quat_wxyz=yaw_quat(math.pi / 2), velocity_world=(0.0, 1.0, 0.0)
            )
        )
        np.testing.assert_allclose(sample.velocity_body_xy, (1.0, 0.0), atol=1e-12)

    def test_scale_error(self) -> None:
        flow = self.make_sensor(scale=1.02)
        sample = flow.update(make_truth(velocity_world=(1.0, -0.5, 0.0)))
        np.testing.assert_allclose(
            sample.velocity_body_xy, (1.02, -0.51), atol=1e-12
        )

    def test_constant_bias(self) -> None:
        flow = self.make_sensor(bias=(0.01, -0.02))
        sample = flow.update(make_truth(velocity_world=(0.5, 0.25, 0.0)))
        np.testing.assert_allclose(
            sample.velocity_body_xy, (0.51, 0.23), atol=1e-12
        )

    def test_dropout_marks_invalid(self) -> None:
        flow = self.make_sensor(dropouts=((1.0, 0.25),))
        saw_invalid = False
        for step in range(800):
            t = step * PHYSICS_DT
            sample = flow.update(make_truth(time=t))
            if sample is None:
                continue
            if 1.0 <= t < 1.25:
                self.assertFalse(sample.valid)
                saw_invalid = True
            else:
                self.assertTrue(sample.valid)
        self.assertTrue(saw_invalid)

    def test_latency_delivers_older_sample(self) -> None:
        latency = 0.02  # 10 physics steps
        flow = self.make_sensor(latency=latency)
        # Speed ramps 1 m/s per second so each delayed sample is
        # measurably older than the current truth.
        released = []
        for step in range(200):
            t = step * PHYSICS_DT
            kin = make_truth(time=t, velocity_world=(t, 0.0, 0.0))
            sample = flow.update(kin)
            if sample is not None:
                released.append(sample)
        self.assertTrue(released)
        for sample in released[1:]:
            self.assertAlmostEqual(
                sample.time + latency, sample.release_time, places=12
            )
            np.testing.assert_allclose(
                sample.velocity_body_xy[0], sample.time, atol=1e-12
            )

    def test_sample_contains_no_absolute_position(self) -> None:
        flow = self.make_sensor()
        sample = flow.update(make_truth(position=(3.0, 4.0, 5.0)))
        fields = set(type(sample).__dataclass_fields__)
        self.assertNotIn("position", fields)
        self.assertNotIn("x", fields)
        self.assertNotIn("y", fields)

    def test_sample_rate_is_deterministic_50hz(self) -> None:
        flow = self.make_sensor()
        samples = 0
        for step in range(100):
            if flow.update(make_truth(time=step * PHYSICS_DT)) is not None:
                samples += 1
        self.assertEqual(samples, 10)  # 50 Hz at 500 Hz physics

    def test_same_seed_bit_identical(self) -> None:
        config = FlowConfig(noise_sigma=0.03)
        first = FlowSensor(config, physics_dt=PHYSICS_DT, seed=3)
        second = FlowSensor(config, physics_dt=PHYSICS_DT, seed=3)
        for step in range(50):
            a = first.update(make_truth(time=step * PHYSICS_DT))
            b = second.update(make_truth(time=step * PHYSICS_DT))
            self.assertEqual(a is None, b is None)
            if a is not None:
                np.testing.assert_array_equal(a.velocity_body_xy, b.velocity_body_xy)


class ToFSensorTest(unittest.TestCase):
    def make_sensor(self, **kwargs) -> ToFSensor:
        defaults = dict(noise_sigma=0.0)
        defaults.update(kwargs)
        return ToFSensor(ToFConfig(**defaults), physics_dt=PHYSICS_DT, seed=0)

    def test_known_height_above_flat_ground(self) -> None:
        tof = self.make_sensor()
        sample = tof.update(make_truth(position=(0.0, 0.0, 1.5)))
        self.assertTrue(sample.valid)
        self.assertAlmostEqual(sample.height_m, 1.5, places=12)

    def test_height_is_relative_to_ground_plane(self) -> None:
        tof = self.make_sensor()
        sample = tof.update(
            make_truth(position=(0.0, 0.0, 1.5), ground_height=0.1)
        )
        self.assertAlmostEqual(sample.height_m, 1.4, places=12)

    def test_out_of_range_invalid(self) -> None:
        tof = self.make_sensor()
        low = tof.update(make_truth(position=(0.0, 0.0, 0.03)))
        self.assertFalse(low.valid)
        tof2 = self.make_sensor()
        high = tof2.update(make_truth(position=(0.0, 0.0, 9.0)))
        self.assertFalse(high.valid)

    def test_bias(self) -> None:
        tof = self.make_sensor(bias=0.02)
        sample = tof.update(make_truth(position=(0.0, 0.0, 1.0)))
        self.assertAlmostEqual(sample.height_m, 1.02, places=12)

    def test_dropout_marks_invalid(self) -> None:
        tof = self.make_sensor(dropouts=((0.5, 0.25),))
        saw_invalid = False
        for step in range(600):
            t = step * PHYSICS_DT
            sample = tof.update(make_truth(time=t, position=(0.0, 0.0, 1.5)))
            if sample is None:
                continue
            if 0.5 <= t < 0.75:
                self.assertFalse(sample.valid)
                saw_invalid = True
        self.assertTrue(saw_invalid)

    def test_latency_delivers_older_height(self) -> None:
        tof = self.make_sensor(latency=0.02)
        released = []
        for step in range(200):
            t = step * PHYSICS_DT
            sample = tof.update(make_truth(time=t, position=(0.0, 0.0, 1.0 + t)))
            if sample is not None:
                released.append(sample)
        self.assertTrue(released)
        for sample in released[1:]:
            self.assertAlmostEqual(sample.height_m, 1.0 + sample.time, places=12)

    def test_same_seed_bit_identical(self) -> None:
        config = ToFConfig(noise_sigma=0.01)
        first = ToFSensor(config, physics_dt=PHYSICS_DT, seed=5)
        second = ToFSensor(config, physics_dt=PHYSICS_DT, seed=5)
        for step in range(50):
            kin = make_truth(time=step * PHYSICS_DT, position=(0.0, 0.0, 1.5))
            a = first.update(kin)
            b = second.update(kin)
            self.assertEqual(a is None, b is None)
            if a is not None:
                self.assertEqual(a.height_m, b.height_m)


if __name__ == "__main__":
    unittest.main()
