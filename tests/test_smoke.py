"""Headless regression tests using only Python's standard unittest runner."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fly import (  # noqa: E402
    TakeoffProfile,
    load_simulation,
    make_command,
    rotation_to_euler,
    simulate_headless,
    step_controller,
    wrap_angle,
)


MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor.xml"


class QuadrotorSmokeTest(unittest.TestCase):
    def test_hover_mixer_reconstructs_requested_wrench(self) -> None:
        _, _, controller = load_simulation(MODEL_PATH)
        requested_wrench = np.array([controller.mass * 9.81, 0.0, 0.0, 0.0])
        thrusts = controller.mix_controls(requested_wrench[0], requested_wrench[1:])

        np.testing.assert_allclose(thrusts, np.full(4, 9.81 / 4.0), atol=1e-12)
        np.testing.assert_allclose(
            controller.allocation_matrix @ thrusts, requested_wrench, atol=1e-12
        )

    def test_smooth_takeoff_has_zero_endpoint_vertical_velocity(self) -> None:
        _, data, controller = load_simulation(MODEL_PATH)
        initial_position = controller.read_state(data).position
        command = make_command(initial_position, target_height=1.2)
        profile = TakeoffProfile(initial_position)

        start = profile.evaluate(0.0, command)
        finish = profile.evaluate(profile.duration, command)

        self.assertAlmostEqual(start.position[2], initial_position[2])
        self.assertAlmostEqual(start.velocity[2], 0.0)
        self.assertAlmostEqual(finish.position[2], 1.2)
        self.assertAlmostEqual(finish.velocity[2], 0.0)

    def test_six_second_headless_hover(self) -> None:
        result = simulate_headless(MODEL_PATH, duration=6.0, target_height=1.2)

        self.assertTrue(result.passed, msg="; ".join(result.reasons))
        self.assertGreater(result.final_position[2], 0.5)
        self.assertLess(result.height_error, 0.10)
        self.assertLess(result.horizontal_error, 0.20)
        self.assertTrue(np.all(np.isfinite(result.motor_thrusts)))

    def test_horizontal_position_and_yaw_tracking(self) -> None:
        model, data, controller = load_simulation(MODEL_PATH)
        initial_position = controller.read_state(data).position
        command = make_command(initial_position, target_height=1.2)
        profile = TakeoffProfile(initial_position)

        while data.time < 10.0:
            if data.time >= 3.0:
                command.position[:] = [0.5, 0.4, 1.2]
                command.yaw = math.radians(25.0)
            target = profile.evaluate(data.time, command)
            output = step_controller(model, data, controller, target)

        state = controller.read_state(data)
        euler = rotation_to_euler(state.rotation_body_to_world)
        yaw_error = abs(wrap_angle(command.yaw - euler[2]))

        self.assertLess(np.linalg.norm(command.position - state.position), 0.05)
        self.assertLess(yaw_error, math.radians(1.0))
        self.assertLess(abs(euler[0]), math.radians(2.0))
        self.assertLess(abs(euler[1]), math.radians(2.0))
        self.assertTrue(np.all(output.motor_thrusts >= controller.minimum_thrusts))
        self.assertTrue(np.all(output.motor_thrusts <= controller.maximum_thrusts))


if __name__ == "__main__":
    unittest.main()
