"""M1 Task 5: deterministic measurement noise layer.

The controller in M0 reads MuJoCo ground truth directly. M1 inserts a
measurement layer: QuadrotorSimulation keeps ground truth for metrics and
safety checks, while the controller receives a noisy QuadrotorState.

Noise model (all independent Gaussian, explicit NumPy seed):

    position:            N(0, 0.010 m)    per axis, additive, world frame
    linear velocity:     N(0, 0.020 m/s)  per axis, additive, world frame
    attitude:            small rotation vector N(0, 0.2 deg) per axis,
                         applied as a valid rotation (never raw quaternion
                         component noise)
    angular velocity:    N(0, 0.010 rad/s) per axis, additive, body frame

Noise never touches data.qpos/qvel and never accumulates into truth.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from fly import load_simulation
from m1_realism import MeasurementModel, NoiseConfig


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "quadrotor.xml"


def truth_state():
    _, data, controller = load_simulation(MODEL_PATH)
    return controller.read_state(data)


def measure_sequence(config, seed, truth, count=8):
    measurement = MeasurementModel(config, seed)
    return [measurement.measure(truth) for _ in range(count)]


class MeasurementNoiseTest(unittest.TestCase):
    def test_same_seed_reproduces_identical_sequence(self) -> None:
        truth = truth_state()
        config = NoiseConfig()
        run_a = measure_sequence(config, seed=42, truth=truth)
        run_b = measure_sequence(config, seed=42, truth=truth)
        for a, b in zip(run_a, run_b):
            np.testing.assert_array_equal(a.position, b.position)
            np.testing.assert_array_equal(a.velocity_world, b.velocity_world)
            np.testing.assert_array_equal(a.quaternion_wxyz, b.quaternion_wxyz)
            np.testing.assert_array_equal(
                a.angular_velocity_body, b.angular_velocity_body
            )

    def test_different_seed_produces_different_sequence(self) -> None:
        truth = truth_state()
        config = NoiseConfig()
        run_a = measure_sequence(config, seed=1, truth=truth)
        run_b = measure_sequence(config, seed=2, truth=truth)
        differences = [
            float(np.max(np.abs(a.position - b.position)))
            for a, b in zip(run_a, run_b)
        ]
        self.assertGreater(max(differences), 1e-6)

    def test_zero_noise_returns_exact_ground_truth(self) -> None:
        truth = truth_state()
        config = NoiseConfig(
            position_sigma=0.0,
            velocity_sigma=0.0,
            attitude_sigma=0.0,
            angular_velocity_sigma=0.0,
        )
        measured = MeasurementModel(config, seed=0).measure(truth)
        np.testing.assert_array_equal(measured.position, truth.position)
        np.testing.assert_array_equal(measured.velocity_world, truth.velocity_world)
        np.testing.assert_array_equal(
            measured.quaternion_wxyz, truth.quaternion_wxyz
        )
        np.testing.assert_array_equal(
            measured.rotation_body_to_world, truth.rotation_body_to_world
        )
        np.testing.assert_array_equal(
            measured.angular_velocity_body, truth.angular_velocity_body
        )

    def test_attitude_measurement_is_a_valid_rotation(self) -> None:
        truth = truth_state()
        measurement = MeasurementModel(NoiseConfig(), seed=7)
        for _ in range(20):
            measured = measurement.measure(truth)
            # Unit quaternion, and quaternion consistent with the matrix.
            self.assertAlmostEqual(
                float(np.linalg.norm(measured.quaternion_wxyz)), 1.0, places=12
            )
            rotation = measured.rotation_body_to_world
            np.testing.assert_allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-12
            )
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
            # Small-angle perturbation: far below 5 sigma = 1 degree.
            cos_angle = float(
                np.clip(
                    (np.trace(truth.rotation_body_to_world.T @ rotation) - 1.0)
                    / 2.0,
                    -1.0,
                    1.0,
                )
            )
            self.assertLess(math.acos(cos_angle), math.radians(5.0))

    def test_noise_magnitude_matches_configured_sigma(self) -> None:
        truth = truth_state()
        config = NoiseConfig()
        measurement = MeasurementModel(config, seed=11)
        samples = np.array(
            [measurement.measure(truth).position - truth.position for _ in range(4000)]
        )
        for axis in range(3):
            self.assertAlmostEqual(
                float(np.std(samples[:, axis])), config.position_sigma, delta=0.002
            )

    def test_ground_truth_is_not_modified(self) -> None:
        # Noise lives only in the measurement path: repeated measurement must
        # leave the source state bit-identical.
        _, data, controller = load_simulation(MODEL_PATH)
        truth_before = controller.read_state(data)
        qpos_before = data.qpos.copy()
        qvel_before = data.qvel.copy()
        measurement = MeasurementModel(NoiseConfig(), seed=3)
        measurement.measure(truth_before)
        truth_after = controller.read_state(data)
        np.testing.assert_array_equal(data.qpos, qpos_before)
        np.testing.assert_array_equal(data.qvel, qvel_before)
        np.testing.assert_array_equal(truth_after.position, truth_before.position)
        np.testing.assert_array_equal(
            truth_after.quaternion_wxyz, truth_before.quaternion_wxyz
        )


if __name__ == "__main__":
    unittest.main()
