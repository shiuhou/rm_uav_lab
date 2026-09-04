"""M1 Tasks 2/3/4/7: drag, parameter uncertainty, motor mismatch, and the
asymmetry transient test.

All plant perturbations go through m1_realism.PlantConfig and are fixed for
a whole rollout; the controller is never modified per case -- the point is
that the existing controller must absorb these non-idealities.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from fly import ROTOR_NAMES, load_simulation, rotation_to_euler
from ipc_protocol import CommandPacket, PROTOCOL_VERSION
from m1_realism import (
    M1_MOTOR_EFFECTIVENESS,
    PlantConfig,
    apply_plant_config,
)
from simulator import FlightParameters, FlightState, QuadrotorSimulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M0_MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor.xml"
M1_MODEL_PATH = PROJECT_ROOT / "models" / "quadrotor_m1.xml"
NOMINAL_INERTIA = np.array([0.018, 0.018, 0.030])


class PlantConfigTest(unittest.TestCase):
    """Task 3: mass/inertia uncertainty is applied to the plant only."""

    def test_scale_factors_are_applied_to_model(self) -> None:
        model, data, _ = load_simulation(M0_MODEL_PATH)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        apply_plant_config(
            model, data, PlantConfig(mass_scale=1.10, inertia_scale=0.90)
        )
        self.assertAlmostEqual(float(model.body_mass[body_id]), 1.10)
        np.testing.assert_allclose(
            model.body_inertia[body_id], 0.90 * NOMINAL_INERTIA, rtol=1e-12
        )

    def test_nominal_scale_reproduces_nominal_exactly(self) -> None:
        model, data, _ = load_simulation(M0_MODEL_PATH)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        apply_plant_config(model, data, PlantConfig())
        self.assertAlmostEqual(float(model.body_mass[body_id]), 1.0)
        np.testing.assert_array_equal(model.body_inertia[body_id], NOMINAL_INERTIA)

    def test_controller_keeps_nominal_mass_while_plant_is_perturbed(self) -> None:
        # Uncertainty means the controller still believes m = 1.0 kg while
        # the physics runs at 1.1 kg.
        simulation = QuadrotorSimulation(
            M0_MODEL_PATH, plant_config=PlantConfig(mass_scale=1.10)
        )
        body_id = simulation.controller.body_id
        self.assertAlmostEqual(simulation.controller.mass, 1.0)
        self.assertAlmostEqual(float(simulation.model.body_mass[body_id]), 1.10)


class MotorMismatchTest(unittest.TestCase):
    """Task 4: per-motor effectiveness breaks the perfect symmetry.

    Motor order is the model's actuator order (FL, FR, RR, RL, see
    fly.ROTOR_NAMES). The M1 nominal asymmetric case is
    FL=0.97, FR=1.00, RR=0.99, RL=1.02.
    """

    def test_effectiveness_scales_individual_actuator_force(self) -> None:
        model, data, _ = load_simulation(M0_MODEL_PATH)
        apply_plant_config(
            model, data, PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS)
        )
        data.ctrl[:] = 2.0
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.actuator_force,
            2.0 * np.array(M1_MOTOR_EFFECTIVENESS),
            rtol=1e-12,
        )

    def test_effectiveness_order_matches_actuator_names(self) -> None:
        # Never trust raw indices: apply by actuator name and read back by
        # actuator name.
        model, data, _ = load_simulation(M0_MODEL_PATH)
        apply_plant_config(
            model, data, PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS)
        )
        for rotor_name, effectiveness in zip(ROTOR_NAMES, M1_MOTOR_EFFECTIVENESS):
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, rotor_name
            )
            self.assertAlmostEqual(
                float(model.actuator_gainprm[actuator_id, 0]), effectiveness
            )

    def test_equal_commands_produce_asymmetric_wrench(self) -> None:
        model, data, controller = load_simulation(M0_MODEL_PATH)
        apply_plant_config(
            model, data, PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS)
        )
        hover = controller.mass * 9.81 / 4.0
        actual = hover * np.array(M1_MOTOR_EFFECTIVENESS)
        wrench = controller.allocation_matrix @ actual
        # The same commands no longer produce a pure vertical wrench:
        # residual pitch/yaw torques must be nonzero (and small).
        # Note: with this particular vector the roll terms cancel exactly
        # (0.163*h*(-0.03 + 0.01 + 0.02) = 0), so the disturbance appears
        # as pitch (tau_y = +0.016 N m) and a small yaw torque.
        self.assertAlmostEqual(float(wrench[1]), 0.0, places=12)
        self.assertGreater(abs(wrench[2]), 1e-3)
        self.assertGreater(abs(wrench[3]), 1e-4)
        self.assertLess(float(np.linalg.norm(wrench[1:])), 0.05)


class DragTest(unittest.TestCase):
    """Task 2: MuJoCo built-in fluid drag decelerates a moving body.

    Isolation setup: gravity off, no thrust, initial +x velocity of 1 m/s.
    The only horizontal force left is fluid drag.
    """

    def _run(self, density, viscosity, duration=3.0):
        model, data, controller = load_simulation(M1_MODEL_PATH)
        model.opt.density = density
        model.opt.viscosity = viscosity
        model.opt.gravity[:] = 0.0
        data.qpos[controller.qpos_address + 2] = 2.0  # airborne, no contact
        data.qvel[0] = 1.0  # free joint qvel[0:3] = world linear velocity
        mujoco.mj_forward(model, data)
        state0 = controller.read_state(data)
        mujoco.mj_step(model, data)
        state1 = controller.read_state(data)
        dt = model.opt.timestep
        initial_accel = (state1.velocity_world - state0.velocity_world) / dt
        while data.time < duration:
            mujoco.mj_step(model, data)
        state_final = controller.read_state(data)
        return state0, initial_accel, state_final

    def test_drag_opposes_motion_and_slows_vehicle(self) -> None:
        state0, initial_accel, state_final = self._run(1.2, 1.8e-5)
        self.assertAlmostEqual(float(state0.velocity_world[0]), 1.0, places=12)
        # Acceleration must oppose velocity from the very first step.
        self.assertLess(initial_accel[0], 0.0)
        self.assertLess(
            float(state_final.velocity_world[0]),
            float(state0.velocity_world[0]),
        )
        # No spurious forces sideways or vertically.
        self.assertAlmostEqual(float(state_final.velocity_world[1]), 0.0, places=9)
        self.assertAlmostEqual(float(state_final.velocity_world[2]), 0.0, places=9)

    def test_no_drag_control_case_keeps_velocity(self) -> None:
        # Same geometry, drag disabled: velocity must stay constant, proving
        # the deceleration above comes from the fluid model.
        state0, initial_accel, state_final = self._run(0.0, 0.0)
        self.assertAlmostEqual(float(initial_accel[0]), 0.0, places=12)
        self.assertAlmostEqual(
            float(state_final.velocity_world[0]),
            float(state0.velocity_world[0]),
            places=12,
        )


class AsymmetryTransientTest(unittest.TestCase):
    """Task 7: asymmetric motors must visibly disturb takeoff, and the
    unmodified controller must reject the disturbance and settle.

    Exactly-zero roll/pitch under asymmetric motors would indicate the
    mismatch is not actually reaching the plant.
    """

    def test_mismatch_disturbs_then_controller_settles(self) -> None:
        simulation = QuadrotorSimulation(
            M0_MODEL_PATH,
            FlightParameters(takeoff_height=1.5),
            plant_config=PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS),
        )
        sequence = 0

        def send(action=None):
            nonlocal sequence
            simulation.accept_command(
                CommandPacket(
                    PROTOCOL_VERSION, sequence, simulation.data.time, action,
                    0.0, 0.0, 0.0, 0.0,
                ),
                simulation.data.time,
            )
            sequence += 1

        send("takeoff")
        next_keepalive = 0.0
        max_transient_tilt_deg = 0.0
        settled_motor_sum = np.zeros(4)
        settled_samples = 0
        final_state = None
        while simulation.data.time < 12.0:
            sim_time = simulation.data.time
            if sim_time >= next_keepalive:
                send()
                next_keepalive = sim_time + 0.05
            vehicle = simulation.step(sim_time)
            euler_deg = np.degrees(
                rotation_to_euler(vehicle.rotation_body_to_world)
            )
            if 0.1 < sim_time < 3.0:  # takeoff transient window
                max_transient_tilt_deg = max(
                    max_transient_tilt_deg,
                    abs(float(euler_deg[0])),
                    abs(float(euler_deg[1])),
                )
            if simulation.machine.state == FlightState.HOVERING:
                final_state = vehicle
                if sim_time > 10.0:  # settled hover
                    settled_motor_sum += simulation.last_motor_thrusts
                    settled_samples += 1
                    self.assertLess(abs(float(euler_deg[0])), 7.0)
                    self.assertLess(abs(float(euler_deg[1])), 7.0)

        # 1. The disturbance really appears during takeoff.
        self.assertGreater(max_transient_tilt_deg, 0.05)
        # 2. The vehicle still reaches and holds hover within the M1 gate.
        self.assertIsNotNone(final_state)
        self.assertLess(abs(float(final_state.position[2]) - 1.5), 0.15)
        # 3. The controller compensates: settled motor commands must order
        #    inversely to effectiveness (FL 0.97 needs the most thrust,
        #    RL 1.02 the least): FL > RR > FR > RL.
        self.assertGreater(settled_samples, 0)
        mean_motors = settled_motor_sum / settled_samples
        self.assertGreater(mean_motors[0], mean_motors[2])  # FL > RR
        self.assertGreater(mean_motors[2], mean_motors[1])  # RR > FR
        self.assertGreater(mean_motors[1], mean_motors[3])  # FR > RL


class MassUncertaintySteadyStateTest(unittest.TestCase):
    """Task 3 follow-up: a pure PD altitude loop has steady-state error
    e = delta_m * g / kp_z = 0.196 m under +-10% mass (measured). A small,
    clamped integral term on the vertical channel must erase it so the
    takeoff->HOVERING transition (|z err| < 0.08 m) can trigger at all.
    """

    def _run_takeoff(self, mass_scale, duration=20.0):
        simulation = QuadrotorSimulation(
            M0_MODEL_PATH,
            FlightParameters(takeoff_height=1.5),
            plant_config=PlantConfig(mass_scale=mass_scale),
        )
        sequence = 0
        simulation.accept_command(
            CommandPacket(PROTOCOL_VERSION, 0, 0.0, "takeoff", 0, 0, 0, 0), 0.0
        )
        sequence = 1
        while simulation.data.time < duration:
            sim_time = simulation.data.time
            if sim_time >= (sequence - 1) * 0.05:
                simulation.accept_command(
                    CommandPacket(
                        PROTOCOL_VERSION, sequence, sim_time, None, 0, 0, 0, 0
                    ),
                    sim_time,
                )
                sequence += 1
            simulation.step(sim_time)
        return simulation

    def test_mass_plus10_reaches_hover_and_erases_steady_state_error(self) -> None:
        simulation = self._run_takeoff(1.10)
        self.assertEqual(simulation.machine.state, FlightState.HOVERING)
        final = simulation.controller.read_state(simulation.data)
        self.assertLess(abs(float(final.position[2]) - 1.5), 0.05)

    def test_mass_minus10_reaches_hover_and_erases_steady_state_error(self) -> None:
        simulation = self._run_takeoff(0.90)
        self.assertEqual(simulation.machine.state, FlightState.HOVERING)
        final = simulation.controller.read_state(simulation.data)
        self.assertLess(abs(float(final.position[2]) - 1.5), 0.05)


if __name__ == "__main__":
    unittest.main()
