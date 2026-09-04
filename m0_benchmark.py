#!/usr/bin/env python3
"""M0 takeoff/hover benchmark for the MuJoCo quadrotor baseline.

Reproducible headless run through the *real* control path:

    TAKEOFF action -> FlightStateMachine smoothstep profile
    -> position/SO(3) controller -> four motor thrusts -> MuJoCo physics

The benchmark never writes qpos/qvel after reset; altitude is reached purely
through controller-commanded actuator thrust. It waits for the state machine
to declare HOVERING, then measures a fixed hover window and prints a report
(plus a machine-readable JSON copy).

Usage:

    python m0_benchmark.py
    python m0_benchmark.py --target-height 1.5 --hover-duration 5.0 \
        --json m0_hover_report.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from fly import DEFAULT_MODEL_PATH, positive_float, rotation_to_euler
from ipc_protocol import CommandPacket, PROTOCOL_VERSION
from simulator import FlightParameters, FlightState, QuadrotorSimulation


# Initial M0 acceptance criteria (all SI units; angles in degrees).
MAX_FINAL_ALTITUDE_ERROR = 0.10
MAX_HOVER_RMS_ERROR = 0.10
MAX_HORIZONTAL_DRIFT = 0.10
MAX_ROLL_PITCH_DEG = 5.0
MIN_HOVER_ALTITUDE = 0.5  # below this during the hover window counts as crash
KEEPALIVE_PERIOD = 0.05  # s of sim time between neutral commands (~20 Hz)


@dataclass(frozen=True)
class M0HoverReport:
    target_altitude: float
    hover_duration_requested: float
    hover_duration_measured: float
    reached_hovering: bool
    stayed_hovering: bool
    final_altitude: float
    final_altitude_error: float
    max_altitude_error: float
    hover_rms_error: float
    max_horizontal_drift: float
    max_speed: float
    max_roll_deg: float
    max_pitch_deg: float
    crashed: bool
    has_nan_or_inf: bool
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict:
        return asdict(self)


def run_m0_hover_benchmark(
    target_height: float = 1.5,
    hover_duration: float = 5.0,
    max_total_time: float = 30.0,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> M0HoverReport:
    """Take off, wait for HOVERING, then measure one fixed hover window."""

    parameters = FlightParameters(takeoff_height=target_height)
    simulation = QuadrotorSimulation(Path(model_path), parameters)
    sequence = 0

    def send(action: str | None = None) -> None:
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
    hover_start_time: float | None = None
    hover_anchor_xy: np.ndarray | None = None
    stayed_hovering = True
    crashed = False
    has_nan_or_inf = False

    altitude_errors: list[float] = []
    horizontal_drifts: list[float] = []
    speeds: list[float] = []
    rolls_deg: list[float] = []
    pitches_deg: list[float] = []

    while simulation.data.time < max_total_time:
        sim_time = simulation.data.time
        # Neutral keepalives keep the IPC timeout failsafe quiet; they do not
        # steer the vehicle (all axes zero).
        if sim_time >= next_keepalive:
            send()
            next_keepalive = sim_time + KEEPALIVE_PERIOD

        vehicle = simulation.step(sim_time)

        values = np.concatenate(
            (vehicle.position, vehicle.velocity_world, simulation.last_motor_thrusts)
        )
        if not np.all(np.isfinite(values)):
            has_nan_or_inf = True
            break

        if hover_start_time is None:
            if simulation.machine.state == FlightState.HOVERING:
                # The hover window starts when the state machine itself
                # declares the takeoff complete and stable.
                hover_start_time = simulation.data.time
                hover_anchor_xy = vehicle.position[:2].copy()
            continue

        if simulation.machine.state != FlightState.HOVERING:
            stayed_hovering = False
        if vehicle.position[2] < MIN_HOVER_ALTITUDE:
            crashed = True

        altitude_errors.append(float(vehicle.position[2] - target_height))
        horizontal_drifts.append(
            float(np.linalg.norm(vehicle.position[:2] - hover_anchor_xy))
        )
        speeds.append(float(np.linalg.norm(vehicle.velocity_world)))
        euler_deg = np.degrees(rotation_to_euler(vehicle.rotation_body_to_world))
        rolls_deg.append(abs(float(euler_deg[0])))
        pitches_deg.append(abs(float(euler_deg[1])))

        if simulation.data.time - hover_start_time >= hover_duration:
            break

    measured_hover_time = (
        0.0 if hover_start_time is None
        else min(simulation.data.time - hover_start_time, hover_duration)
    )
    final_state = simulation.controller.read_state(simulation.data)
    final_altitude = float(final_state.position[2])
    final_altitude_error = abs(final_altitude - target_height)
    max_altitude_error = max((abs(e) for e in altitude_errors), default=math.inf)
    hover_rms_error = (
        math.sqrt(sum(e * e for e in altitude_errors) / len(altitude_errors))
        if altitude_errors else math.inf
    )
    max_horizontal_drift = max(horizontal_drifts, default=math.inf)
    max_speed = max(speeds, default=math.inf)
    max_roll_deg = max(rolls_deg, default=math.inf)
    max_pitch_deg = max(pitches_deg, default=math.inf)

    reasons: list[str] = []
    if hover_start_time is None:
        reasons.append("takeoff never reached HOVERING (unstable takeoff)")
    if measured_hover_time + 1e-9 < hover_duration:
        reasons.append(
            f"hover lasted {measured_hover_time:.2f} s < {hover_duration:.2f} s"
        )
    if not stayed_hovering:
        reasons.append("state machine left HOVERING during the hover window")
    if crashed:
        reasons.append(f"altitude dropped below {MIN_HOVER_ALTITUDE} m (crash)")
    if has_nan_or_inf:
        reasons.append("state or motor thrust contains NaN/Inf")
    if final_altitude_error > MAX_FINAL_ALTITUDE_ERROR:
        reasons.append(
            f"final altitude error {final_altitude_error:.3f} m > "
            f"{MAX_FINAL_ALTITUDE_ERROR} m"
        )
    if hover_rms_error > MAX_HOVER_RMS_ERROR:
        reasons.append(
            f"hover RMS error {hover_rms_error:.3f} m > {MAX_HOVER_RMS_ERROR} m"
        )
    if max_horizontal_drift > MAX_HORIZONTAL_DRIFT:
        reasons.append(
            f"horizontal drift {max_horizontal_drift:.3f} m > "
            f"{MAX_HORIZONTAL_DRIFT} m"
        )
    if max_roll_deg > MAX_ROLL_PITCH_DEG:
        reasons.append(f"max roll {max_roll_deg:.2f} deg > {MAX_ROLL_PITCH_DEG} deg")
    if max_pitch_deg > MAX_ROLL_PITCH_DEG:
        reasons.append(
            f"max pitch {max_pitch_deg:.2f} deg > {MAX_ROLL_PITCH_DEG} deg"
        )

    return M0HoverReport(
        target_altitude=target_height,
        hover_duration_requested=hover_duration,
        hover_duration_measured=measured_hover_time,
        reached_hovering=hover_start_time is not None,
        stayed_hovering=stayed_hovering,
        final_altitude=final_altitude,
        final_altitude_error=final_altitude_error,
        max_altitude_error=max_altitude_error,
        hover_rms_error=hover_rms_error,
        max_horizontal_drift=max_horizontal_drift,
        max_speed=max_speed,
        max_roll_deg=max_roll_deg,
        max_pitch_deg=max_pitch_deg,
        crashed=crashed,
        has_nan_or_inf=has_nan_or_inf,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def format_report(report: M0HoverReport) -> str:
    lines = [
        "========== M0 HOVER REPORT ==========",
        f"Target altitude       : {report.target_altitude:.2f} m",
        f"Final altitude        : {report.final_altitude:.3f} m",
        f"Final altitude error  : {report.final_altitude_error:.3f} m",
        "",
        f"Max altitude error    : {report.max_altitude_error:.3f} m",
        f"Hover RMS error       : {report.hover_rms_error:.3f} m",
        "",
        f"Max horizontal drift  : {report.max_horizontal_drift:.3f} m",
        f"Max speed             : {report.max_speed:.3f} m/s",
        "",
        f"Max roll              : {report.max_roll_deg:.2f} deg",
        f"Max pitch             : {report.max_pitch_deg:.2f} deg",
        "",
        f"Hover duration        : {report.hover_duration_measured:.2f} s",
        "",
        f"RESULT                : {'PASS' if report.passed else 'FAIL'}",
    ]
    for reason in report.reasons:
        lines.append(f"  - {reason}")
    lines.append("=====================================")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-height", type=positive_float, default=1.5)
    parser.add_argument("--hover-duration", type=positive_float, default=5.0)
    parser.add_argument("--max-total-time", type=positive_float, default=30.0)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("m0_hover_report.json"),
        help="where to write machine-readable metrics (default: %(default)s)",
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH, metavar="XML"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = run_m0_hover_benchmark(
        target_height=args.target_height,
        hover_duration=args.hover_duration,
        max_total_time=args.max_total_time,
        model_path=args.model_path,
    )
    print(format_report(report))
    args.json.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n")
    print(f"metrics written to {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
