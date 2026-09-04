"""Flight state machine, safety action, and headless scenario tests."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from ipc_protocol import CommandPacket, PROTOCOL_VERSION
from simulator import (
    FlightParameters,
    FlightState,
    FlightStateMachine,
    ManualAxes,
    QuadrotorSimulation,
    run_headless_scenario,
)


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "quadrotor.xml"


class FlightSequenceTest(unittest.TestCase):
    def test_takeoff_hover_scenario(self) -> None:
        result = run_headless_scenario("takeoff_hover", 8.0, 1.2, MODEL_PATH)
        self.assertTrue(result.passed, "; ".join(result.reasons))
        self.assertIn("TAKING_OFF", result.state_history)
        self.assertIn("HOVERING", result.state_history)
        self.assertGreater(result.max_altitude, 1.0)
        self.assertLess(result.hover_error, 0.10)
        self.assertLess(result.horizontal_error, 0.20)

    def test_takeoff_land_scenario(self) -> None:
        result = run_headless_scenario("takeoff_land", 14.0, 1.2, MODEL_PATH)
        self.assertTrue(result.passed, "; ".join(result.reasons))
        self.assertIn("LANDING", result.state_history)
        self.assertIn("LANDED", result.state_history)
        self.assertEqual(result.final_state, "DISARMED")
        self.assertLess(result.final_position[2], 0.15)
        self.assertLess(abs(result.final_velocity[2]), 0.15)
        self.assertLess(np.max(np.abs(result.final_motor_thrusts)), 0.05)

    def test_command_timeout_enters_hover(self) -> None:
        result = run_headless_scenario("command_loss", 8.0, 1.2, MODEL_PATH)
        self.assertTrue(result.passed, "; ".join(result.reasons))
        self.assertIn("MANUAL", result.state_history)
        self.assertEqual(result.final_state, "HOVERING")

    def test_duplicate_action_sequence_is_not_retriggered(self) -> None:
        simulation = QuadrotorSimulation(MODEL_PATH)
        packet = CommandPacket.neutral(10, 0.0, "takeoff")
        self.assertTrue(simulation.accept_command(packet, now=0.0))
        self.assertFalse(simulation.accept_command(packet, now=0.1))
        self.assertEqual(simulation.action_counts.get("takeoff"), 1)

    def test_forward_command_is_relative_to_current_yaw(self) -> None:
        simulation = QuadrotorSimulation(MODEL_PATH)
        vehicle = simulation.controller.read_state(simulation.data)
        yaw_90_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        yaw_90_vehicle = type(vehicle)(
            vehicle.position,
            vehicle.velocity_world,
            vehicle.quaternion_wxyz,
            yaw_90_rotation,
            vehicle.angular_velocity_body,
        )
        machine = FlightStateMachine(yaw_90_vehicle, FlightParameters())
        machine.state = FlightState.HOVERING
        start = machine.target_position.copy()
        machine.integrate_manual(
            ManualAxes(forward=1.0), yaw_90_vehicle, 1.0, 1.0
        )
        delta = machine.target_position - start
        self.assertAlmostEqual(delta[0], 0.0, places=12)
        self.assertAlmostEqual(delta[1], 1.0, places=12)

    def test_disarm_is_rejected_while_flying_and_emergency_stops_motors(self) -> None:
        simulation = QuadrotorSimulation(MODEL_PATH)
        simulation.accept_command(CommandPacket.neutral(0, 0.0, "takeoff"), 0.0)
        sequence = 1
        while simulation.data.time < 5.0:
            if sequence == 1 or simulation.data.time >= sequence * 0.05:
                simulation.accept_command(
                    CommandPacket.neutral(sequence, simulation.data.time),
                    simulation.data.time,
                )
                sequence += 1
            simulation.step(simulation.data.time)
        self.assertEqual(simulation.machine.state, FlightState.HOVERING)

        simulation.accept_command(
            CommandPacket.neutral(sequence, simulation.data.time, "disarm"),
            simulation.data.time,
        )
        sequence += 1
        self.assertEqual(simulation.machine.state, FlightState.HOVERING)

        simulation.accept_command(
            CommandPacket.neutral(
                sequence, simulation.data.time, "emergency_stop"
            ),
            simulation.data.time,
        )
        simulation.step(simulation.data.time)
        self.assertEqual(simulation.machine.state, FlightState.EMERGENCY_STOP)
        np.testing.assert_array_equal(simulation.last_motor_thrusts, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
