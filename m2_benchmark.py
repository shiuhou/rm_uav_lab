#!/usr/bin/env python3
"""M2 mission benchmark: autonomous takeoff -> forward 5 m -> brake ->
hover -> land, across the M1 realism matrix.

Every case runs the REAL production path; there is no benchmark-only
controller:

    MissionStateMachine (measured state only)
    -> normalized velocity action (forward/right/up/yaw)
    -> FlightStateMachine target integration
    -> position + SO(3) controller -> mixer -> motors
    -> MuJoCo physics -> MeasurementModel -> mission/controller feedback

Scoring uses ground truth, which the mission never sees. The report keeps
"what the drone thinks" (measured) and "what actually happened" (truth)
as separate numbers on purpose.

Usage:

    python m2_benchmark.py
    python m2_benchmark.py --json m2_mission_report.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from fly import SimulationSafetyError, rotation_to_euler, wrap_angle
from ipc_protocol import CommandPacket, PROTOCOL_VERSION
from m1_realism import (
    M1_MOTOR_EFFECTIVENESS,
    NoiseConfig,
    PlantConfig,
)
from m2_mission import (
    MissionConfig,
    MissionFrame,
    MissionPhase,
    MissionStateMachine,
)
from simulator import FlightParameters, QuadrotorSimulation


PROJECT_ROOT = Path(__file__).resolve().parent
M0_MODEL = PROJECT_ROOT / "models" / "quadrotor.xml"
M1_MODEL = PROJECT_ROOT / "models" / "quadrotor_m1.xml"

# M2 scoring gates.
GATE_TAKEOFF_ALTITUDE_ERROR = 0.15
GATE_FORWARD_MIN = 4.7
GATE_FORWARD_MAX = 5.3
GATE_LATERAL = 0.20
GATE_TRUTH_SPEED_AFTER_BRAKE = 0.30
GATE_TILT_NOMINAL_DEG = 10.0
GATE_TILT_ROBUST_DEG = 15.0
GATE_SATURATION_FRACTION = 0.20
MIN_ALTITUDE_FLOOR = 0.02  # body origin below this = ground penetration
MAX_TOTAL_TIME = 45.0

ZERO_NOISE = NoiseConfig(
    position_sigma=0.0,
    velocity_sigma=0.0,
    attitude_sigma=0.0,
    angular_velocity_sigma=0.0,
)


@dataclass(frozen=True)
class M2Case:
    name: str
    description: str
    model_path: Path
    plant: PlantConfig = field(default_factory=PlantConfig)
    noise: NoiseConfig = ZERO_NOISE
    seed: int = 0
    initial_yaw_deg: float = 0.0
    tilt_gate_deg: float = GATE_TILT_ROBUST_DEG


M2_CASES: tuple[M2Case, ...] = (
    M2Case("case0_nominal", "nominal, zero noise", M0_MODEL,
           tilt_gate_deg=GATE_TILT_NOMINAL_DEG),
    M2Case("case1_motor_lag", "motor lag tau=0.040 s", M1_MODEL,
           plant=PlantConfig(drag=False)),
    M2Case("case2_mass_plus10", "mass +10%", M0_MODEL,
           plant=PlantConfig(mass_scale=1.10)),
    M2Case("case3_mass_minus10", "mass -10%", M0_MODEL,
           plant=PlantConfig(mass_scale=0.90)),
    M2Case("case4_inertia_plus10", "inertia +10%", M0_MODEL,
           plant=PlantConfig(inertia_scale=1.10)),
    M2Case("case5_inertia_minus10", "inertia -10%", M0_MODEL,
           plant=PlantConfig(inertia_scale=0.90)),
    M2Case("case6_air_drag", "air drag", M0_MODEL,
           plant=PlantConfig(drag=True)),
    M2Case("case7_measurement_noise", "measurement noise (seed 1)", M0_MODEL,
           noise=NoiseConfig(), seed=1),
    M2Case("case8_motor_mismatch", "motor mismatch", M0_MODEL,
           plant=PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS)),
    M2Case("case9_combined", "all M1 realism combined (seed 27)", M1_MODEL,
           plant=PlantConfig(mass_scale=1.10, inertia_scale=1.10,
                             motor_effectiveness=M1_MOTOR_EFFECTIVENESS),
           noise=NoiseConfig(), seed=27),
    M2Case("case10_yaw90_nominal", "nominal, initial yaw = 90 deg", M0_MODEL,
           initial_yaw_deg=90.0, tilt_gate_deg=GATE_TILT_NOMINAL_DEG),
    M2Case("case11_yaw90_combined", "combined realism, initial yaw = 90 deg",
           M1_MODEL,
           plant=PlantConfig(mass_scale=1.10, inertia_scale=1.10,
                             motor_effectiveness=M1_MOTOR_EFFECTIVENESS),
           noise=NoiseConfig(), seed=27, initial_yaw_deg=90.0),
)


@dataclass(frozen=True)
class M2CaseReport:
    name: str
    description: str
    model_file: str
    mass_kg: float
    inertia: tuple[float, float, float]
    motor_tau: float
    motor_effectiveness: tuple[float, float, float, float]
    noise: dict
    seed: int
    initial_yaw_deg: float
    mission: dict
    mission_result: str
    failure_reason: str | None
    takeoff_time: float
    settle_time: float
    forward_time: float
    brake_time: float
    hover_time: float
    land_time: float
    total_time: float
    truth_takeoff_altitude: float
    measured_takeoff_altitude: float
    max_altitude_error_horizontal: float
    measured_forward_at_brake_entry: float
    truth_forward_at_brake_entry: float
    measured_forward_at_land_entry: float
    truth_forward_at_land_entry: float
    measured_lateral_at_land_entry: float
    truth_lateral_at_land_entry: float
    measured_braking_distance: float
    truth_braking_distance: float
    max_horizontal_speed: float
    truth_speed_at_hover_entry: float
    max_roll_deg: float
    max_pitch_deg: float
    max_yaw_deviation_deg: float
    max_motor_command: float
    min_motor_command: float
    saturation_fraction: float
    hover_altitude_rms: float
    hover_horizontal_position_rms: float
    hover_speed_rms: float
    landing_success: bool
    has_nan_or_inf: bool
    min_altitude: float
    ground_penetration: bool
    truth_start_position_xy: tuple[float, float]
    truth_final_position_xy: tuple[float, float]
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _rms(values: list[float]) -> float:
    if not values:
        return math.inf
    return math.sqrt(sum(v * v for v in values) / len(values))


def run_m2_case(
    case: M2Case,
    mission_config: MissionConfig | None = None,
    max_total_time: float = MAX_TOTAL_TIME,
) -> M2CaseReport:
    config = mission_config or MissionConfig()
    parameters = FlightParameters(takeoff_height=config.takeoff_altitude)
    if not math.isclose(
        config.velocity_command_limit, parameters.max_horizontal_speed
    ):
        raise ValueError("mission velocity_command_limit must match simulator")

    simulation = QuadrotorSimulation(
        case.model_path,
        parameters,
        plant_config=case.plant,
        noise_config=case.noise,
        noise_seed=case.seed,
        initial_yaw=math.radians(case.initial_yaw_deg),
    )
    mission = MissionStateMachine(config)

    # Scoring frame from TRUTH (independent of the mission's measured frame).
    truth0 = simulation.controller.read_state(simulation.data)
    truth_yaw0 = float(rotation_to_euler(truth0.rotation_body_to_world)[2])
    truth_frame = MissionFrame(truth0.position[:2].copy(), truth_yaw0)
    truth_start_xy = truth0.position[:2].copy()

    sequence = 0

    def send(command) -> None:
        nonlocal sequence
        simulation.accept_command(
            CommandPacket(
                PROTOCOL_VERSION, sequence, simulation.data.time,
                command.action, command.forward, command.right,
                command.up, command.yaw,
            ),
            simulation.data.time,
        )
        sequence += 1

    command = mission.start(simulation.last_measured_state, simulation.data.time)
    send(command)

    # Truth-side accumulators (scoring only).
    min_altitude = math.inf
    max_horizontal_speed = 0.0
    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    max_yaw_deviation_deg = 0.0
    max_motor_command = 0.0
    min_motor_command = math.inf
    total_steps = 0
    saturated_steps = 0
    has_nan_or_inf = False
    max_altitude_error_horizontal = 0.0
    hover_altitude_errors: list[float] = []
    hover_horizontal_offsets: list[float] = []
    hover_speeds: list[float] = []
    hover_anchor_xy: np.ndarray | None = None
    truth_snapshots: dict[str, dict[str, float]] = {}
    runtime_error: str | None = None
    truth_final_xy = truth_start_xy.copy()
    truth_final_z = float(truth0.position[2])

    previous_phase = mission.phase
    while True:
        try:
            truth = simulation.step(simulation.data.time)
        except SimulationSafetyError as error:
            runtime_error = f"SIMULATION_SAFETY: {error}"
            has_nan_or_inf = True
            break
        truth_final_xy = truth.position[:2].copy()
        truth_final_z = float(truth.position[2])
        motors = simulation.last_motor_thrusts

        min_altitude = min(min_altitude, truth_final_z)
        horizontal_speed = float(np.linalg.norm(truth.velocity_world[:2]))
        max_horizontal_speed = max(max_horizontal_speed, horizontal_speed)
        euler_deg = np.degrees(rotation_to_euler(truth.rotation_body_to_world))
        max_roll_deg = max(max_roll_deg, abs(float(euler_deg[0])))
        max_pitch_deg = max(max_pitch_deg, abs(float(euler_deg[1])))
        yaw_deviation = abs(
            wrap_angle(float(euler_deg[2]) - math.degrees(truth_yaw0))
        )
        max_yaw_deviation_deg = max(max_yaw_deviation_deg, yaw_deviation)
        max_motor_command = max(max_motor_command, float(np.max(motors)))
        min_motor_command = min(min_motor_command, float(np.min(motors)))
        total_steps += 1
        limits = simulation.controller
        if np.any(motors >= limits.maximum_thrusts - 1e-9) or np.any(
            motors <= limits.minimum_thrusts + 1e-9
        ):
            saturated_steps += 1

        if mission.phase in (MissionPhase.FORWARD, MissionPhase.BRAKE,
                             MissionPhase.HOVER):
            max_altitude_error_horizontal = max(
                max_altitude_error_horizontal,
                abs(truth_final_z - config.takeoff_altitude),
            )
        if mission.phase is MissionPhase.HOVER:
            if hover_anchor_xy is None:
                hover_anchor_xy = truth.position[:2].copy()
            hover_altitude_errors.append(
                truth_final_z - config.takeoff_altitude
            )
            hover_horizontal_offsets.append(
                float(np.linalg.norm(truth.position[:2] - hover_anchor_xy))
            )
            hover_speeds.append(horizontal_speed)

        command = mission.update(
            simulation.last_measured_state,
            simulation.machine.state.value,
            simulation.data.time,
        )
        if mission.phase is not previous_phase:
            progress, lateral = truth_frame.project(truth.position[:2])
            truth_snapshots[mission.phase.value] = {
                "time": float(simulation.data.time),
                "progress": progress,
                "lateral": lateral,
                "speed": horizontal_speed,
                "altitude": truth_final_z,
            }
            previous_phase = mission.phase
        send(command)

        if mission.phase in (MissionPhase.DONE, MissionPhase.FAILED):
            break
        if simulation.data.time > max_total_time:
            runtime_error = "BENCHMARK_TIMEOUT"
            break

    log = mission.log
    nan = math.nan

    def log_time(key: str) -> float:
        return float(log.get(key, nan))

    def snapshot(phase: MissionPhase, key: str) -> float:
        entry = truth_snapshots.get(phase.value)
        return float(entry[key]) if entry is not None else nan

    mission_done = mission.phase is MissionPhase.DONE
    mission_result = mission.phase.value
    failure_reason = mission.failure_reason or runtime_error

    takeoff_time = log_time("takeoff_complete_time") - log_time("takeoff_entry_time")
    settle_time = log_time("settle_complete_time") - log_time("takeoff_complete_time")
    forward_time = log_time("brake_entry_time") - log_time("forward_entry_time")
    brake_time = log_time("hover_entry_time") - log_time("brake_entry_time")
    hover_time = log_time("land_entry_time") - log_time("hover_entry_time")
    land_time = log_time("done_time") - log_time("land_entry_time")
    total_time = log_time("done_time") - log_time("takeoff_entry_time")

    truth_forward_at_brake = snapshot(MissionPhase.BRAKE, "progress")
    truth_forward_at_hover = snapshot(MissionPhase.HOVER, "progress")
    truth_forward_at_land = snapshot(MissionPhase.LAND, "progress")
    truth_lateral_at_land = snapshot(MissionPhase.LAND, "lateral")
    measured_forward_at_brake = float(
        log.get("brake_entry_measured_progress", nan)
    )
    measured_forward_at_hover = float(log.get("hover_entry_measured_progress", nan))
    measured_forward_at_land = float(log.get("land_entry_measured_progress", nan))
    measured_lateral_at_land = float(log.get("land_entry_measured_lateral", nan))
    truth_braking = truth_forward_at_hover - truth_forward_at_brake
    measured_braking = measured_forward_at_hover - measured_forward_at_brake
    truth_speed_at_hover = snapshot(MissionPhase.HOVER, "speed")
    truth_takeoff_altitude = snapshot(MissionPhase.SETTLE, "altitude")
    measured_takeoff_altitude = float(log.get("takeoff_measured_altitude", nan))

    saturation_fraction = (
        saturated_steps / total_steps if total_steps else math.inf
    )
    ground_penetration = min_altitude < MIN_ALTITUDE_FLOOR
    landing_success = mission_done and truth_final_z < 0.15

    reasons: list[str] = []
    if not mission_done:
        reasons.append(f"mission {mission_result}: {failure_reason}")
    if not math.isnan(truth_takeoff_altitude) and abs(
        truth_takeoff_altitude - config.takeoff_altitude
    ) > GATE_TAKEOFF_ALTITUDE_ERROR:
        reasons.append(f"takeoff altitude error vs {config.takeoff_altitude} m")
    if not (GATE_FORWARD_MIN <= truth_forward_at_land <= GATE_FORWARD_MAX):
        reasons.append(f"truth forward {truth_forward_at_land:.3f} m outside "
                       f"[{GATE_FORWARD_MIN}, {GATE_FORWARD_MAX}]")
    if not math.isnan(truth_lateral_at_land) and abs(truth_lateral_at_land) > GATE_LATERAL:
        reasons.append(f"truth lateral {truth_lateral_at_land:.3f} m")
    if not math.isnan(truth_speed_at_hover) and (
        truth_speed_at_hover > GATE_TRUTH_SPEED_AFTER_BRAKE
    ):
        reasons.append(f"truth speed after brake {truth_speed_at_hover:.3f} m/s")
    if mission_done and hover_time < config.hover_duration - 1e-9:
        reasons.append(f"hover {hover_time:.2f} s < {config.hover_duration} s")
    if not landing_success:
        reasons.append("landing did not complete with disarm near ground")
    max_tilt = max(max_roll_deg, max_pitch_deg)
    if max_tilt > case.tilt_gate_deg:
        reasons.append(f"max tilt {max_tilt:.2f} deg > {case.tilt_gate_deg} deg")
    if saturation_fraction > GATE_SATURATION_FRACTION:
        reasons.append(f"saturation fraction {saturation_fraction:.2f}")
    if has_nan_or_inf:
        reasons.append("NaN/Inf or simulation safety abort")
    if ground_penetration:
        reasons.append(f"ground penetration: min altitude {min_altitude:.3f} m")

    body_id = simulation.controller.body_id
    motor_tau = (
        float(simulation.model.actuator_dynprm[0, 0])
        if simulation.model.na > 0
        else 0.0
    )

    return M2CaseReport(
        name=case.name,
        description=case.description,
        model_file=case.model_path.name,
        mass_kg=float(simulation.model.body_mass[body_id]),
        inertia=tuple(float(v) for v in simulation.model.body_inertia[body_id]),
        motor_tau=motor_tau,
        motor_effectiveness=tuple(float(v) for v in case.plant.motor_effectiveness),
        noise=asdict(case.noise),
        seed=case.seed,
        initial_yaw_deg=case.initial_yaw_deg,
        mission=asdict(config),
        mission_result=mission_result,
        failure_reason=failure_reason,
        takeoff_time=takeoff_time,
        settle_time=settle_time,
        forward_time=forward_time,
        brake_time=brake_time,
        hover_time=hover_time,
        land_time=land_time,
        total_time=total_time,
        truth_takeoff_altitude=truth_takeoff_altitude,
        measured_takeoff_altitude=measured_takeoff_altitude,
        max_altitude_error_horizontal=max_altitude_error_horizontal,
        measured_forward_at_brake_entry=measured_forward_at_brake,
        truth_forward_at_brake_entry=truth_forward_at_brake,
        measured_forward_at_land_entry=measured_forward_at_land,
        truth_forward_at_land_entry=truth_forward_at_land,
        measured_lateral_at_land_entry=measured_lateral_at_land,
        truth_lateral_at_land_entry=truth_lateral_at_land,
        measured_braking_distance=measured_braking,
        truth_braking_distance=truth_braking,
        max_horizontal_speed=max_horizontal_speed,
        truth_speed_at_hover_entry=truth_speed_at_hover,
        max_roll_deg=max_roll_deg,
        max_pitch_deg=max_pitch_deg,
        max_yaw_deviation_deg=max_yaw_deviation_deg,
        max_motor_command=max_motor_command,
        min_motor_command=0.0 if min_motor_command is math.inf else min_motor_command,
        saturation_fraction=saturation_fraction,
        hover_altitude_rms=_rms(hover_altitude_errors),
        hover_horizontal_position_rms=_rms(hover_horizontal_offsets),
        hover_speed_rms=_rms(hover_speeds),
        landing_success=landing_success,
        has_nan_or_inf=has_nan_or_inf,
        min_altitude=min_altitude,
        ground_penetration=ground_penetration,
        truth_start_position_xy=tuple(float(v) for v in truth_start_xy),
        truth_final_position_xy=tuple(float(v) for v in truth_final_xy),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def run_m2_benchmark(
    cases: Sequence[M2Case] = M2_CASES,
) -> list[M2CaseReport]:
    return [run_m2_case(case) for case in cases]


def format_summary(reports: Sequence[M2CaseReport]) -> str:
    header = (
        f"{'case':<24} {'result':>6} {'truth_fwd':>9} {'meas_fwd':>9} "
        f"{'lateral':>8} {'brake_d':>8} {'tilt':>6} {'time':>6} {'gate':>4}"
    )
    lines = ["=" * len(header), header, "=" * len(header)]
    for report in reports:
        max_tilt = max(report.max_roll_deg, report.max_pitch_deg)
        lines.append(
            f"{report.name:<24} {report.mission_result:>6} "
            f"{report.truth_forward_at_land_entry:>8.3f}m "
            f"{report.measured_forward_at_land_entry:>8.3f}m "
            f"{report.truth_lateral_at_land_entry:>7.3f}m "
            f"{report.truth_braking_distance:>7.3f}m "
            f"{max_tilt:>5.2f}d {report.total_time:>5.1f}s "
            f"{'PASS' if report.passed else 'FAIL':>4}"
        )
        for reason in report.reasons:
            lines.append(f"    - {reason}")
    lines.append("=" * len(header))
    all_passed = all(report.passed for report in reports)
    lines.append(f"M2 RESULT: {'PASS' if all_passed else 'FAIL'}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("m2_mission_report.json"),
        help="where to write machine-readable metrics (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    reports = run_m2_benchmark()
    print(format_summary(reports))
    payload = {
        "milestone": "M2",
        "mission": asdict(MissionConfig()),
        "gates": {
            "takeoff_altitude_error": GATE_TAKEOFF_ALTITUDE_ERROR,
            "truth_forward_range": [GATE_FORWARD_MIN, GATE_FORWARD_MAX],
            "truth_lateral": GATE_LATERAL,
            "truth_speed_after_brake": GATE_TRUTH_SPEED_AFTER_BRAKE,
            "tilt_nominal_deg": GATE_TILT_NOMINAL_DEG,
            "tilt_robust_deg": GATE_TILT_ROBUST_DEG,
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
