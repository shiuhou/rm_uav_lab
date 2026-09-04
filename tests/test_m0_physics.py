"""M0 physics sanity tests for the MuJoCo quadrotor model.

These tests verify the *plant* itself, independent of the flight controller:

* A. free fall: zero thrust means 1 g downward acceleration, nothing else.
* B. hover thrust: total thrust m*g (m*g/4 per motor) gives zero acceleration.
* C. thrust direction: more collective thrust accelerates the vehicle upward.
* D/E/F. mixer: differential thrust produces roll/pitch/yaw torques with the
  signs documented in models/quadrotor.xml and fly.py.

Sign conventions being pinned down (body frame: +X nose, +Y left, +Z up):

* roll torque  tau_x = sum(y_i * F_i)   ->  left motors raise the left side.
* pitch torque tau_y = sum(-x_i * F_i)  ->  rear-up means nose-down.
* yaw torque   tau_z = sum(gear_z_i * F_i); FL/RR have gear_z = +0.018
  (counter-clockwise seen from above), FR/RL have -0.018.

All accelerations are measured by finite-differencing the state across one or
more mj_step() calls; nothing writes qpos/qvel after the initial condition.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from fly import load_simulation, rotation_to_euler


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "quadrotor.xml"
START_ALTITUDE = 2.0
# Motor order everywhere in this project: front-left, front-right,
# rear-right, rear-left (matches the XML actuator block).
FL, FR, RR, RL = 0, 1, 2, 3


def make_airborne_simulation():
    """Return (model, data, controller) with the quad released at 2 m.

    Setting the initial altitude is an initial condition, not state
    manipulation: every subsequent step is pure MuJoCo physics.
    """

    model, data, controller = load_simulation(MODEL_PATH)
    data.qpos[controller.qpos_address + 2] = START_ALTITUDE
    mujoco.mj_forward(model, data)
    return model, data, controller


def step_with_thrusts(model, data, thrusts, duration):
    """Hold constant motor thrusts for `duration` seconds of sim time."""

    data.ctrl[:] = np.asarray(thrusts, dtype=float)
    steps = int(round(duration / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return steps * model.opt.timestep


class FreeFallTest(unittest.TestCase):
    """A. With motors off, the quad is a 1 kg brick in a 9.81 m/s^2 field."""

    def test_free_fall_matches_gravity_only(self) -> None:
        model, data, controller = make_airborne_simulation()
        gravity = float(np.linalg.norm(model.opt.gravity))
        self.assertAlmostEqual(gravity, 9.81)

        data.ctrl[:] = 0.0
        state0 = controller.read_state(data)
        elapsed = step_with_thrusts(model, data, np.zeros(4), 0.25)
        state1 = controller.read_state(data)

        az = (state1.velocity_world[2] - state0.velocity_world[2]) / elapsed
        self.assertAlmostEqual(az, -gravity, places=9)
        self.assertLess(state1.velocity_world[2], 0.0)

        # Gravity acts at the center of mass, so there is no torque at all:
        # no horizontal motion and no attitude change may appear.
        np.testing.assert_allclose(state1.velocity_world[:2], 0.0, atol=1e-12)
        np.testing.assert_allclose(state1.angular_velocity_body, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            state1.quaternion_wxyz, state0.quaternion_wxyz, atol=1e-12
        )

        # Analytic cross-check: z(t) = z0 - g t^2 / 2 (RK4 integrates a
        # constant-acceleration trajectory exactly).
        expected_z = START_ALTITUDE - 0.5 * gravity * elapsed**2
        self.assertAlmostEqual(state1.position[2], expected_z, places=9)


class HoverThrustTest(unittest.TestCase):
    """B. Total thrust m*g (m*g/4 per motor) must cancel weight exactly."""

    def test_hover_thrust_gives_zero_acceleration(self) -> None:
        model, data, controller = make_airborne_simulation()
        mass = controller.mass
        gravity = float(np.linalg.norm(model.opt.gravity))
        total_hover_thrust = mass * gravity
        per_motor = total_hover_thrust / 4.0
        # 1 kg * 9.81 m/s^2 -> 9.81 N total, 2.4525 N per motor.
        self.assertAlmostEqual(per_motor, 2.4525)

        state0 = controller.read_state(data)
        elapsed = step_with_thrusts(model, data, np.full(4, per_motor), 0.5)
        state1 = controller.read_state(data)

        acceleration = (state1.velocity_world - state0.velocity_world) / elapsed
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-9)
        self.assertAlmostEqual(state1.position[2], START_ALTITUDE, places=9)

        # Equal thrusts on all four motors also cancel every torque: the X
        # layout is symmetric, and FL/RR (+) and FR/RL (-) yaw reaction
        # torques sum to zero.
        np.testing.assert_allclose(state1.angular_velocity_body, 0.0, atol=1e-12)


class ThrustDirectionTest(unittest.TestCase):
    """C. Equal thrust above hover must accelerate the vehicle upward."""

    def test_increased_collective_thrust_accelerates_upward(self) -> None:
        model, data, controller = make_airborne_simulation()
        gravity = float(np.linalg.norm(model.opt.gravity))
        per_motor = controller.mass * gravity / 4.0

        state0 = controller.read_state(data)
        elapsed = step_with_thrusts(model, data, np.full(4, 1.5 * per_motor), 0.2)
        state1 = controller.read_state(data)

        az = (state1.velocity_world[2] - state0.velocity_world[2]) / elapsed
        # 1.5 * m * g upward force on mass m -> net +0.5 g.
        self.assertGreater(az, 0.0)
        self.assertAlmostEqual(az, 0.5 * gravity, places=9)
        self.assertGreater(state1.position[2], START_ALTITUDE)


class MixerTest(unittest.TestCase):
    """D/E/F. Differential thrust around hover trim and the resulting
    roll/pitch/yaw torques.

    Each case applies thrusts = hover + delta * pattern, where the pattern
    keeps total thrust at m*g (so the quad stays near its start altitude and
    the measurement is a clean angular acceleration). The expected body
    torque is taken from controller.allocation_matrix -- which is built from
    the XML motor positions and yaw gear -- and compared against the measured
    angular acceleration via tau = I * alpha.
    """

    DELTA = 0.5  # N of differential thrust per motor

    def _measure(self, pattern: np.ndarray):
        model, data, controller = make_airborne_simulation()
        gravity = float(np.linalg.norm(model.opt.gravity))
        hover = np.full(4, controller.mass * gravity / 4.0)
        thrusts = hover + self.DELTA * np.asarray(pattern, dtype=float)

        # Expected wrench from the model-derived allocation matrix.
        wrench = controller.allocation_matrix @ thrusts
        self.assertAlmostEqual(wrench[0], controller.mass * gravity, places=12)

        # One RK4 step from rest: torque is constant in the body frame and
        # the omega x (I omega) gyroscopic term is second order in dt, so the
        # finite difference recovers alpha = I^-1 tau almost exactly.
        state0 = controller.read_state(data)
        data.ctrl[:] = thrusts
        mujoco.mj_step(model, data)
        state1 = controller.read_state(data)
        dt = model.opt.timestep
        alpha = (state1.angular_velocity_body - state0.angular_velocity_body) / dt

        # Then hold the command a little longer to check the attitude
        # actually rotates in the documented direction.
        step_with_thrusts(model, data, thrusts, 0.1)
        state2 = controller.read_state(data)
        euler = rotation_to_euler(state2.rotation_body_to_world)

        inertia = model.body_inertia[controller.body_id]
        expected_alpha = wrench[1:] / inertia
        return alpha, expected_alpha, euler, state2

    def test_roll_mixer_positive_torque_raises_left_side(self) -> None:
        # Left motors (FL, RL at y=+0.163) up, right motors down.
        alpha, expected_alpha, euler, state2 = self._measure([+1.0, -1.0, -1.0, +1.0])
        np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-3, atol=1e-6)
        self.assertGreater(alpha[0], 0.0)
        self.assertLess(abs(alpha[1]), 1e-6)
        self.assertLess(abs(alpha[2]), 1e-6)
        # Positive rotation about body +X lifts the +Y (left) side: roll > 0.
        self.assertGreater(euler[0], 0.0)
        self.assertGreater(state2.angular_velocity_body[0], 0.0)

    def test_pitch_mixer_rear_up_rotates_nose_down(self) -> None:
        # Rear motors (RR, RL at x=-0.163) up, front motors down: the tail
        # rises, so the +X nose drops. With body +Y pointing left, the
        # right-hand rule makes this a positive tau_y / omega_y.
        # Note the reporting convention: in a Z-up world with ZYX Euler,
        # R[2,0] = -sin(pitch), so nose-DOWN reads as positive pitch here
        # (opposite of the aviation convention, which assumes Z-down).
        alpha, expected_alpha, euler, state2 = self._measure([-1.0, -1.0, +1.0, +1.0])
        np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-3, atol=1e-6)
        self.assertGreater(alpha[1], 0.0)
        self.assertLess(abs(alpha[0]), 1e-6)
        self.assertLess(abs(alpha[2]), 1e-6)
        self.assertGreater(euler[1], 0.0)  # nose-down reads positive in Z-up ZYX
        self.assertGreater(state2.angular_velocity_body[1], 0.0)

    def test_yaw_mixer_diagonal_pair_spins_counter_clockwise(self) -> None:
        # FL/RR (gear_z=+0.018) up, FR/RL (-0.018) down: pure positive yaw
        # torque, i.e. counter-clockwise seen from above. The diagonal pair
        # is symmetric about the origin, so roll and pitch cancel.
        alpha, expected_alpha, euler, state2 = self._measure([+1.0, -1.0, +1.0, -1.0])
        np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-3, atol=1e-6)
        self.assertGreater(alpha[2], 0.0)
        self.assertLess(abs(alpha[0]), 1e-6)
        self.assertLess(abs(alpha[1]), 1e-6)
        self.assertGreater(euler[2], 0.0)
        self.assertGreater(state2.angular_velocity_body[2], 0.0)

    def test_controller_mixer_matches_physics_patterns(self) -> None:
        # The controller's inverse allocation must generate the same
        # differential patterns verified against MuJoCo above.
        _, _, controller = load_simulation(MODEL_PATH)
        hover = controller.mass * 9.81

        roll = controller.mix_controls(hover, np.array([0.1, 0.0, 0.0]))
        self.assertGreater(roll[FL], roll[FR])
        self.assertGreater(roll[RL], roll[RR])

        pitch = controller.mix_controls(hover, np.array([0.0, 0.1, 0.0]))
        self.assertGreater(pitch[RR], pitch[FL])
        self.assertGreater(pitch[RL], pitch[FR])

        yaw = controller.mix_controls(hover, np.array([0.0, 0.0, 0.05]))
        self.assertGreater(yaw[FL], yaw[FR])
        self.assertGreater(yaw[RR], yaw[RL])


if __name__ == "__main__":
    unittest.main()
