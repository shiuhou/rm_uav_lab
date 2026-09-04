"""M1 Task 1: first-order motor lag in models/quadrotor_m1.xml.

The M0 <motor> shortcut is direct drive: actual thrust equals commanded
thrust in the same timestep. The M1 model replaces each motor with an
equivalent <general> actuator using dyntype="filterexact", so the actual
thrust F follows the commanded thrust u through

    dF/dt = (u - F) / tau_motor          (tau_motor = 0.040 s nominal)

For a constant step command u the analytic response is

    F(t) = u * (1 - exp(-t / tau))

so F(tau)/u = 1 - e^-1 = 0.6321. The controller still commands Newtons;
only the plant's response is delayed.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M1_MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor_m1.xml"
M0_MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor.xml"
NOMINAL_TAU = 0.040
STEP_COMMAND = 2.0  # N, a constant step below the 8 N limit


def load_m1():
    model = mujoco.MjModel.from_xml_path(str(M1_MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def step_until(model, data, sim_time):
    """Step until data.time reaches sim_time, then refresh derived fields."""

    while data.time < sim_time - 1e-12:
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)  # refresh actuator_force at this instant


def record_step_response(model, data, command, duration):
    """Return (times, mean_act) for a constant step command on all motors."""

    data.ctrl[:] = command
    times = [0.0]
    acts = [0.0]
    while data.time < duration - 1e-12:
        mujoco.mj_step(model, data)
        times.append(float(data.time))
        acts.append(float(np.mean(data.act)))
    return np.array(times), np.array(acts)


def crossing_time(times, acts, level):
    """Linearly interpolated time at which acts first reach `level`."""

    index = int(np.searchsorted(acts, level))
    assert 0 < index < len(acts), "response never reached the target level"
    t0, t1 = times[index - 1], times[index]
    a0, a1 = acts[index - 1], acts[index]
    return t0 + (level - a0) * (t1 - t0) / (a1 - a0)


class MotorLagTest(unittest.TestCase):
    def test_actuators_are_filterexact_general_with_m0_interface(self) -> None:
        model, _ = load_m1()
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        )
        # Same names/order, gear, ctrlrange and ctrl semantics as M0.
        self.assertEqual(names, ("rotor_fl", "rotor_fr", "rotor_rr", "rotor_rl"))
        self.assertTrue(
            np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_FILTEREXACT)
        )
        np.testing.assert_allclose(model.actuator_dynprm[:, 0], NOMINAL_TAU)
        self.assertTrue(
            np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
        )
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1.0)
        np.testing.assert_allclose(model.actuator_ctrlrange, [[0.0, 8.0]] * 4)
        self.assertTrue(np.all(model.actuator_ctrllimited))

    def test_zero_command_stays_zero(self) -> None:
        model, data = load_m1()
        data.ctrl[:] = 0.0
        step_until(model, data, 0.2)
        np.testing.assert_allclose(data.act, 0.0, atol=1e-15)
        np.testing.assert_allclose(data.actuator_force, 0.0, atol=1e-15)

    def test_step_response_matches_first_order_filter(self) -> None:
        model, data = load_m1()
        times, acts = record_step_response(
            model, data, STEP_COMMAND, 4.0 * NOMINAL_TAU
        )

        # Monotonic approach to the commanded thrust.
        self.assertTrue(np.all(np.diff(acts) > 0.0))

        # F(t63)/F_final = 0.6321 defines the measured lag constant.
        t63 = crossing_time(times, acts, STEP_COMMAND * (1.0 - math.exp(-1.0)))
        # RK4 only integrates qpos/qvel; activation states are advanced after
        # the RK4 pass, which shifts the discrete response by about half a
        # timestep (verified numerically). tau must be recovered to within
        # one integration step.
        self.assertAlmostEqual(t63, NOMINAL_TAU, delta=1.5 * model.opt.timestep)

        # First-order shape check, independent of tau calibration:
        # at twice the 63.2% crossing time the response must read 86.47%.
        level_at_2t63 = float(np.interp(2.0 * t63, times, acts)) / STEP_COMMAND
        self.assertAlmostEqual(level_at_2t63, 1.0 - math.exp(-2.0), delta=0.01)

        # actuator_force carries the same lagged value (gain = 1 N/N).
        step_until(model, data, 10.0 * NOMINAL_TAU)
        np.testing.assert_allclose(data.actuator_force, STEP_COMMAND, rtol=1e-3)

    def test_steady_state_reaches_commanded_thrust(self) -> None:
        model, data = load_m1()
        data.ctrl[:] = STEP_COMMAND
        step_until(model, data, 10.0 * NOMINAL_TAU)
        np.testing.assert_allclose(data.actuator_force, STEP_COMMAND, rtol=1e-3)

    def test_all_motors_share_identical_nominal_lag(self) -> None:
        model, data = load_m1()
        data.ctrl[:] = np.array([1.0, 2.0, 3.0, 4.0])
        for _ in range(40):  # 0.08 s > tau
            mujoco.mj_step(model, data)
            ratios = data.act / data.ctrl
            np.testing.assert_allclose(ratios, ratios[0], atol=1e-12)

    def test_tau_range_020_040_080(self) -> None:
        # The M1 robustness matrix probes faster and slower motors than
        # nominal; each must still follow its own first-order response.
        for tau in (0.020, 0.040, 0.080):
            model, data = load_m1()
            model.actuator_dynprm[:, 0] = tau
            times, acts = record_step_response(model, data, STEP_COMMAND, 4.0 * tau)
            t63 = crossing_time(times, acts, STEP_COMMAND * (1.0 - math.exp(-1.0)))
            self.assertAlmostEqual(
                t63, tau, delta=1.5 * model.opt.timestep,
                msg=f"measured lag constant for tau={tau}",
            )

    def test_m0_model_has_no_lag(self) -> None:
        # Control case proving the lag is a property of the M1 model:
        # the M0 direct-drive motors reach full thrust immediately.
        model = mujoco.MjModel.from_xml_path(str(M0_MODEL_PATH))
        data = mujoco.MjData(model)
        data.ctrl[:] = STEP_COMMAND
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, STEP_COMMAND, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
