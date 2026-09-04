#!/usr/bin/env python3
"""M1 robustness benchmark: takeoff + 5 s hover across a non-ideal matrix.

Every case runs the production path with no simplified benchmark controller:

    action -> FlightStateMachine -> QuadrotorController -> mixer
    -> actuator (with lag where configured) -> MuJoCo physics
    -> measurement (with noise where configured) -> controller

Plant parameters are fixed per rollout; randomness is fully determined by
each case's explicit seed. Metrics are always computed from ground truth,
never from the noisy measured state the controller sees.

Usage:

    python m1_benchmark.py
    python m1_benchmark.py --json m1_robustness_report.json
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
from m1_realism import (
    M1_MOTOR_EFFECTIVENESS,
    NoiseConfig,
    PlantConfig,
)
from simulator import FlightParameters, FlightState, QuadrotorSimulation


PROJECT_ROOT = Path(__file__).resolve().parent
M0_MODEL = PROJECT_ROOT / "models" / "quadrotor.xml"
M1_MODEL = PROJECT_ROOT / "models" / "quadrotor_m1.xml"

# M1 gates (looser than M0 on purpose; this is a robustness milestone).
GATE_FINAL_ALTITUDE_ERROR = 0.15
GATE_HOVER_RMS_ERROR = 0.10
GATE_HORIZONTAL_DRIFT = 0.15
GATE_TILT_DEG = 7.0
GATE_SETTLED_HORIZONTAL_SPEED = 0.30
GATE_HOVER_DURATION = 5.0
GATE_SATURATION_FRACTION = 0.20  # fraction of hover steps at an actuator limit
MIN_ALTITUDE_AFTER_LIFTOFF = 0.02  # below this counts as ground penetration
TAKEOFF_TRANSIENT = 3.0  # s; "after takeoff transient" metrics start here
KEEPALIVE_PERIOD = 0.05


@dataclass(frozen=True)
class M1Case:
    name: str
    description: str
    model_path: Path
    plant: PlantConfig = field(default_factory=PlantConfig)
    noise: NoiseConfig | None = None
    seed: int = 0


M1_CASES: tuple[M1Case, ...] = (
    M1Case("case0_nominal_m0", "M0-like ideal reference", M0_MODEL),
    M1Case(
        "case1_motor_lag",
        "motor first-order lag tau=0.040 s",
        M1_MODEL,
        plant=PlantConfig(drag=False),
    ),
    M1Case(
        "case2_mass_plus10",
        "mass +10%",
        M0_MODEL,
        plant=PlantConfig(mass_scale=1.10),
    ),
    M1Case(
        "case3_mass_minus10",
        "mass -10%",
        M0_MODEL,
        plant=PlantConfig(mass_scale=0.90),
    ),
    M1Case(
        "case4_inertia_plus10",
        "inertia +10%",
        M0_MODEL,
        plant=PlantConfig(inertia_scale=1.10),
    ),
    M1Case(
        "case5_inertia_minus10",
        "inertia -10%",
        M0_MODEL,
        plant=PlantConfig(inertia_scale=0.90),
    ),
    M1Case(
        "case6_air_drag",
        "air drag enabled (density 1.2, viscosity 1.8e-5)",
        M0_MODEL,
        plant=PlantConfig(drag=True),
    ),
    M1Case(
        "case7_measurement_noise",
        "measurement noise (seed 1)",
        M0_MODEL,
        noise=NoiseConfig(),
        seed=1,
    ),
    M1Case(
        "case8_motor_mismatch",
        "motor effectiveness FL=0.97 FR=1.00 RR=0.99 RL=1.02",
        M0_MODEL,
        plant=PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS),
    ),
    M1Case(
        "case9_combined",
        "lag + drag + mass/inertia +10% + mismatch + noise (seed 27)",
        M1_MODEL,
        plant=PlantConfig(
            mass_scale=1.10,
            inertia_scale=1.10,
            motor_effectiveness=M1_MOTOR_EFFECTIVENESS,
        ),
        noise=NoiseConfig(),
        seed=27,
    ),
)


@dataclass(frozen=True)
class M1CaseReport:
    name: str
    description: str
    model_file: str
    mass_kg: float
    inertia: tuple[float, float, float]
    motor_tau: float
    motor_effectiveness: tuple[float, float, float, float]
    noise: dict | None
    seed: int
    target_altitude: float
    reached_hovering: bool
    stayed_hovering: bool
    hover_duration_measured: float
    final_altitude: float
    final_altitude_error: float
    max_altitude_error_post_takeoff: float
    hover_rms_error: float
    max_horizontal_drift: float
    max_horizontal_speed_hover: float
    max_roll_deg: float
    max_pitch_deg: float
    final_roll_deg: float
    final_pitch_deg: float
    max_motor_command: float
    min_motor_command: float
    saturation_fraction: float
    min_altitude: float
    crashed: bool
    has_nan_or_inf: bool
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def run_m1_case(
    case: M1Case,
    target_height: float = 1.5,
    hover_duration: float = GATE_HOVER_DURATION,
    max_total_time: float = 30.0,
) -> M1CaseReport:
    """Take off through the real control path and measure a 5 s hover."""

    parameters = FlightParameters(takeoff_height=target_height)
    simulation = QuadrotorSimulation(
        case.model_path,
        parameters,
        plant_config=case.plant,
        noise_config=case.noise,
        noise_seed=case.seed,
    )
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

    min_altitude = math.inf
    max_motor_command = 0.0
    min_motor_command = math.inf
    post_takeoff_alt_errors: list[float] = []
    post_takeoff_rolls: list[float] = []
    post_takeoff_pitches: list[float] = []
    hover_alt_errors: list[float] = []
    hover_drifts: list[float] = []
    hover_horizontal_speeds: list[float] = []
    hover_steps = 0
    saturated_steps = 0
    has_nan_or_inf = False

    while simulation.data.time < max_total_time:
        sim_time = simulation.data.time
        if sim_time >= next_keepalive:
            send()
            next_keepalive = sim_time + KEEPALIVE_PERIOD

        vehicle = simulation.step(sim_time)
        motors = simulation.last_motor_thrusts

        values = np.concatenate(
            (vehicle.position, vehicle.velocity_world, motors)
        )
        if not np.all(np.isfinite(values)):
            has_nan_or_inf = True
            break

        min_altitude = min(min_altitude, float(vehicle.position[2]))
        max_motor_command = max(max_motor_command, float(np.max(motors)))
        min_motor_command = min(min_motor_command, float(np.min(motors)))

        euler_deg = np.degrees(rotation_to_euler(vehicle.rotation_body_to_world))
        if sim_time >= TAKEOFF_TRANSIENT:
            post_takeoff_alt_errors.append(
                abs(float(vehicle.position[2]) - target_height)
            )
            post_takeoff_rolls.append(abs(float(euler_deg[0])))
            post_takeoff_pitches.append(abs(float(euler_deg[1])))

        if hover_start_time is None:
            if simulation.machine.state == FlightState.HOVERING:
                hover_start_time = simulation.data.time
                hover_anchor_xy = vehicle.position[:2].copy()
            continue

        if simulation.machine.state != FlightState.HOVERING:
            stayed_hovering = False
        hover_alt_errors.append(float(vehicle.position[2] - target_height))
        hover_drifts.append(
            float(np.linalg.norm(vehicle.position[:2] - hover_anchor_xy))
        )
        hover_horizontal_speeds.append(
            float(np.linalg.norm(vehicle.velocity_world[:2]))
        )
        hover_steps += 1
        limits = simulation.controller
        if np.any(motors >= limits.maximum_thrusts - 1e-9) or np.any(
            motors <= limits.minimum_thrusts + 1e-9
        ):
            saturated_steps += 1

        if simulation.data.time - hover_start_time >= hover_duration:
            break

    measured_hover = (
        0.0 if hover_start_time is None
        else min(simulation.data.time - hover_start_time, hover_duration)
    )
    final_state = simulation.controller.read_state(simulation.data)
    final_euler_deg = np.degrees(
        rotation_to_euler(final_state.rotation_body_to_world)
    )
    final_altitude = float(final_state.position[2])
    final_altitude_error = abs(final_altitude - target_height)
    max_altitude_error = max(post_takeoff_alt_errors, default=math.inf)
    hover_rms_error = (
        math.sqrt(sum(e * e for e in hover_alt_errors) / len(hover_alt_errors))
        if hover_alt_errors else math.inf
    )
    max_horizontal_drift = max(hover_drifts, default=math.inf)
    max_horizontal_speed = max(hover_horizontal_speeds, default=math.inf)
    max_roll = max(post_takeoff_rolls, default=math.inf)
    max_pitch = max(post_takeoff_pitches, default=math.inf)
    saturation_fraction = (
        saturated_steps / hover_steps if hover_steps else math.inf
    )
    crashed = min_altitude < MIN_ALTITUDE_AFTER_LIFTOFF

    reasons: list[str] = []
    if hover_start_time is None:
        reasons.append("takeoff never reached HOVERING")
    if measured_hover + 1e-9 < hover_duration:
        reasons.append(f"hover {measured_hover:.2f} s < {hover_duration:.2f} s")
    if not stayed_hovering:
        reasons.append("left HOVERING during the hover window")
    if crashed:
        reasons.append(f"min altitude {min_altitude:.3f} m (ground penetration)")
    if has_nan_or_inf:
        reasons.append("state or motor thrust contains NaN/Inf")
    if final_altitude_error > GATE_FINAL_ALTITUDE_ERROR:
        reasons.append(f"final altitude error {final_altitude_error:.3f} m")
    if hover_rms_error > GATE_HOVER_RMS_ERROR:
        reasons.append(f"hover RMS error {hover_rms_error:.3f} m")
    if max_horizontal_drift > GATE_HORIZONTAL_DRIFT:
        reasons.append(f"horizontal drift {max_horizontal_drift:.3f} m")
    if max_roll > GATE_TILT_DEG or max_pitch > GATE_TILT_DEG:
        reasons.append(f"max tilt roll={max_roll:.2f} pitch={max_pitch:.2f} deg")
    if max_horizontal_speed > GATE_SETTLED_HORIZONTAL_SPEED:
        reasons.append(f"hover horizontal speed {max_horizontal_speed:.3f} m/s")
    if saturation_fraction > GATE_SATURATION_FRACTION:
        reasons.append(f"actuator saturation fraction {saturation_fraction:.2f}")

    body_id = simulation.controller.body_id
    motor_tau = (
        float(simulation.model.actuator_dynprm[0, 0])
        if simulation.model.na > 0
        else 0.0
    )
    noise_echo = asdict(case.noise) if case.noise is not None else None

    return M1CaseReport(
        name=case.name,
        description=case.description,
        model_file=case.model_path.name,
        mass_kg=float(simulation.model.body_mass[body_id]),
        inertia=tuple(float(v) for v in simulation.model.body_inertia[body_id]),
        motor_tau=motor_tau,
        motor_effectiveness=tuple(
            float(v) for v in case.plant.motor_effectiveness
        ),
        noise=noise_echo,
        seed=case.seed,
        target_altitude=target_height,
        reached_hovering=hover_start_time is not None,
        stayed_hovering=stayed_hovering,
        hover_duration_measured=measured_hover,
        final_altitude=final_altitude,
        final_altitude_error=final_altitude_error,
        max_altitude_error_post_takeoff=max_altitude_error,
        hover_rms_error=hover_rms_error,
        max_horizontal_drift=max_horizontal_drift,
        max_horizontal_speed_hover=max_horizontal_speed,
        max_roll_deg=max_roll,
        max_pitch_deg=max_pitch,
        final_roll_deg=abs(float(final_euler_deg[0])),
        final_pitch_deg=abs(float(final_euler_deg[1])),
        max_motor_command=max_motor_command,
        min_motor_command=0.0 if min_motor_command is math.inf else min_motor_command,
        saturation_fraction=saturation_fraction,
        min_altitude=min_altitude,
        crashed=crashed,
        has_nan_or_inf=has_nan_or_inf,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def run_m1_benchmark(
    target_height: float = 1.5,
    hover_duration: float = GATE_HOVER_DURATION,
    cases: Sequence[M1Case] = M1_CASES,
) -> list[M1CaseReport]:
    return [
        run_m1_case(case, target_height, hover_duration) for case in cases
    ]


def format_summary(reports: Sequence[M1CaseReport]) -> str:
    header = (
        f"{'case':<24} {'final_z_err':>11} {'rms':>7} {'drift':>7} "
        f"{'max_tilt':>8} {'sat_frac':>8} {'result':>6}"
    )
    lines = ["=" * len(header), header, "=" * len(header)]
    for report in reports:
        max_tilt = max(report.max_roll_deg, report.max_pitch_deg)
        lines.append(
            f"{report.name:<24} {report.final_altitude_error:>10.3f}m "
            f"{report.hover_rms_error:>6.3f}m {report.max_horizontal_drift:>6.3f}m "
            f"{max_tilt:>7.2f}d {report.saturation_fraction:>8.3f} "
            f"{'PASS' if report.passed else 'FAIL':>6}"
        )
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    lines.append("=" * len(header))
    all_passed = all(report.passed for report in reports)
    lines.append(f"M1 RESULT: {'PASS' if all_passed else 'FAIL'}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-height", type=positive_float, default=1.5)
    parser.add_argument(
        "--hover-duration", type=positive_float, default=GATE_HOVER_DURATION
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("m1_robustness_report.json"),
        help="where to write machine-readable metrics (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    reports = run_m1_benchmark(args.target_height, args.hover_duration)
    print(format_summary(reports))
    payload = {
        "milestone": "M1",
        "target_altitude": args.target_height,
        "hover_duration": args.hover_duration,
        "gates": {
            "final_altitude_error": GATE_FINAL_ALTITUDE_ERROR,
            "hover_rms_error": GATE_HOVER_RMS_ERROR,
            "horizontal_drift": GATE_HORIZONTAL_DRIFT,
            "tilt_deg": GATE_TILT_DEG,
            "settled_horizontal_speed": GATE_SETTLED_HORIZONTAL_SPEED,
            "hover_duration": GATE_HOVER_DURATION,
            "saturation_fraction": GATE_SATURATION_FRACTION,
        },
        "all_passed": all(report.passed for report in reports),
        "cases": [asdict(report) for report in reports],
    }
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"metrics written to {args.json}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
