"""IPC packet validation and round-trip tests."""

from __future__ import annotations

import math
import unittest

from ipc_protocol import (
    CommandPacket,
    PROTOCOL_VERSION,
    ProtocolError,
    TelemetryPacket,
    decode_command,
    decode_telemetry,
    encode_command,
    encode_telemetry,
)


class ProtocolTest(unittest.TestCase):
    def test_command_round_trip(self) -> None:
        packet = CommandPacket(
            PROTOCOL_VERSION, 7, 12.5, "takeoff", 0.5, -0.25, 0.1, -1.0
        )
        self.assertEqual(decode_command(encode_command(packet)), packet)

    def test_telemetry_round_trip(self) -> None:
        packet = TelemetryPacket(
            PROTOCOL_VERSION,
            8,
            13.0,
            "HOVERING",
            True,
            (0.1, -0.2, 1.2),
            (0.0, 0.0, 0.0),
            (0.5, -0.2, 30.0),
            (0.1, -0.2, 1.2),
            30.0,
            (2.45, 2.46, 2.44, 2.45),
            0.02,
        )
        self.assertEqual(decode_telemetry(encode_telemetry(packet)), packet)

    def test_malformed_and_non_finite_packets_are_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            decode_command(b"not-json")
        with self.assertRaises(ProtocolError):
            decode_command(
                b'{"version":1,"sequence":1,"timestamp":NaN,"action":null,'
                b'"forward":0,"right":0,"up":0,"yaw":0}'
            )
        with self.assertRaises(ProtocolError):
            encode_command(
                CommandPacket(1, 1, 0.0, None, math.inf, 0.0, 0.0, 0.0)
            )

    def test_out_of_range_axis_is_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            encode_command(CommandPacket(1, 1, 0.0, None, 1.01, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
