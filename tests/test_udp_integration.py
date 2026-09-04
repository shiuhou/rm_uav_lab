"""Real localhost UDP round-trip test between controller and simulator links."""

from __future__ import annotations

from pathlib import Path
import socket
import time
import unittest

from pygame_controller import ControlAxes, ControllerLink
from simulator import FlightState, LocalUdpTransport, QuadrotorSimulation


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "quadrotor.xml"


def unused_udp_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return int(port)


class UdpIntegrationTest(unittest.TestCase):
    def test_high_level_command_and_telemetry_round_trip(self) -> None:
        command_port = unused_udp_port()
        telemetry_port = unused_udp_port()
        while telemetry_port == command_port:
            telemetry_port = unused_udp_port()

        simulation = QuadrotorSimulation(MODEL_PATH)
        transport = LocalUdpTransport(
            "127.0.0.1", command_port, telemetry_port
        )
        controller = ControllerLink("127.0.0.1", command_port, telemetry_port)
        try:
            controller.send(ControlAxes(), "takeoff")
            time.sleep(0.01)
            accepted = transport.receive_commands(simulation, time.monotonic())
            self.assertEqual(accepted, 1)
            self.assertEqual(simulation.machine.state, FlightState.TAKING_OFF)

            now = time.monotonic()
            transport.send_telemetry(simulation.telemetry(0, now))
            time.sleep(0.01)
            telemetry = controller.poll(time.monotonic())
            self.assertEqual(telemetry.flight_state, "TAKING_OFF")
            self.assertTrue(controller.is_connected(time.monotonic()))
        finally:
            controller.close()
            transport.close()


if __name__ == "__main__":
    unittest.main()
