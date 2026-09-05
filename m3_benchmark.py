#!/usr/bin/env python3
"""M3 localization benchmark: the full M2 mission flown on an ESTIMATED
navigation state instead of truth+noise measurements.

Production path per physics step:

    MuJoCo truth
      -> IMUSensor / FlowSensor / ToFSensor   (the only truth consumers)
      -> LocalizationEstimator (IMU predict + flow/ToF correct)
      -> EstimatedState
      -> FlightStateMachine target integration + mission + controller
      -> mixer -> motors -> MuJoCo

Ground truth continues to drive physics safety checks, ground-contact
mechanics (landing detection) and ALL scoring below. The mission and
controller never see it (structural tests in test_m3_truth_isolation.py).

Sensor values are M3 engineering test values, NOT MTF-02P specs.

Usage:

    python m3_benchmark.py
    python m3_benchmark.py --json m3_localization_report.json
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
from m1_realism import M1_MOTOR_EFFECTIVENESS, PlantConfig
from m2_benchmark import M0_MODEL, M1_MODEL
from m2_mission import (
    MissionConfig,
    MissionFrame,
    MissionPhase,
    MissionStateMachine,
)
from m3_estimator import EstimatorConfig, LocalizationPipeline
from m3_sensors import FlowConfig, IMUConfig, ToFConfig
from simulator import FlightParameters, QuadrotorSimulation


# M3 scoring gates (drift-tolerant by design; see README M3 section).
GATE_FORWARD_MIN = 4.5
GATE_FORWARD_MAX = 5.5
GATE_LATERAL = 0.30
GATE_ALTITUDE_RMS = 0.15  # est-vs-truth altitude error
GATE_TILT_DEG = 15.0
GATE_SATURATION_FRACTION = 0.20
MIN_ALTITUDE_FLOOR = 0.02
MAX_TOTAL_TIME = 45.0

# Phases where the estimator must be healthy. LAND is excluded: below
# the ToF minimum range the rangefinder is legitimately blind, and
# landing completion is a ground-contact mechanic.
HEALTH_GATED_PHASES = frozenset(
    (
        MissionPhase.TAKEOFF,
        MissionPhase.SETTLE,
        MissionPhase.FORWARD,
        MissionPhase.BRAKE,
        MissionPhase.HOVER,
    )
)

# Startup grace for the health gate: with sensor latency the first
# flow/ToF samples arrive late (20 ms latency + up to one 20 ms sample
# period). A sensor that NEVER delivers still fails via TAKEOFF_TIMEOUT
# because the estimated altitude never climbs.
HEALTH_STARTUP_GRACE_S = 0.25

IDEAL_IMU = IMUConfig(gyro_noise_sigma=0.0, accel_noise_sigma=0.0)
IDEAL_FLOW = FlowConfig(noise_sigma=0.0)
IDEAL_TOF = ToFConfig(noise_sigma=0.0)

NOMINAL_IMU = IMUConfig()
NOMINAL_FLOW = FlowConfig()
NOMINAL_TOF = ToFConfig()

# Absolute-time dropout windows (mission FORWARD spans roughly
# t = 5.1 s .. 13.8 s in nominal timing).
FLOW_DROPOUT = ((7.0, 0.25),)
TOF_DROPOUT = ((7.0, 0.25),)

COMBINED_PLANT = PlantConfig(
    mass_scale=1.10,
    inertia_scale=1.10,
    motor_effectiveness=M1_MOTOR_EFFECTIVENESS,
)
COMBINED_IMU = IMUConfig(
    gyro_bias=(0.0, 0.0, math.radians(0.05)),
    accel_bias=(0.005, 0.005, 0.01),
)
COMBINED_FLOW = FlowConfig(
    scale=1.01,
    bias=(0.005, 0.0),
    latency=0.020,
    dropouts=((7.0, 0.20),),
)
COMBINED_TOF = ToFConfig(
    bias=0.005,
    latency=0.020,
    dropouts=((8.0, 0.15),),
)


@dataclass(frozen=True)
class M3Case:
    name: str
    description: str
    model_path: Path
    plant: PlantConfig = field(default_factory=PlantConfig)
    imu: IMUConfig = IDEAL_IMU
    flow: FlowConfig = IDEAL_FLOW
    tof: ToFConfig = IDEAL_TOF
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    seed: int = 0
    initial_yaw_deg: float = 0.0


M3_CASES: tuple[M3Case, ...] = (
    M3Case("case0_ideal_sensors", "ideal localization sensors", M0_MODEL),
    M3Case("case1_nominal_noise", "nominal sensor noise (seed 1)", M0_MODEL,
           imu=NOMINAL_IMU, flow=NOMINAL_FLOW, tof=NOMINAL_TOF, seed=1),
    M3Case("case2_flow_scale", "flow scale 1.02 (seed 2)", M0_MODEL,
           imu=NOMINAL_IMU,
           flow=FlowConfig(scale=1.02), tof=NOMINAL_TOF, seed=2),
    M3Case("case3_flow_bias", "flow bias +0.01 m/s fwd (seed 3)", M0_MODEL,
           imu=NOMINAL_IMU,
           flow=FlowConfig(bias=(0.01, 0.0)), tof=NOMINAL_TOF, seed=3),
    M3Case("case4_flow_dropout", "flow dropout 0.25 s (seed 4)", M0_MODEL,
           imu=NOMINAL_IMU,
           flow=FlowConfig(dropouts=FLOW_DROPOUT), tof=NOMINAL_TOF, seed=4),
    M3Case("case5_tof_dropout", "ToF dropout 0.25 s (seed 5)", M0_MODEL,
           imu=NOMINAL_IMU, flow=NOMINAL_FLOW,
           tof=ToFConfig(dropouts=TOF_DROPOUT), seed=5),
    M3Case("case6_gyro_z_bias", "gyro z bias 0.1 deg/s (seed 6)", M0_MODEL,
           imu=IMUConfig(gyro_bias=(0.0, 0.0, math.radians(0.1))),
           flow=NOMINAL_FLOW, tof=NOMINAL_TOF, seed=6),
    M3Case("case7_sensor_latency", "flow+ToF latency 20 ms (seed 7)", M0_MODEL,
           imu=NOMINAL_IMU,
           flow=FlowConfig(latency=0.020),
           tof=ToFConfig(latency=0.020), seed=7),
    M3Case("case8_combined", "combined localization + M1 plant (seed 27)",
           M1_MODEL, plant=COMBINED_PLANT,
           imu=COMBINED_IMU, flow=COMBINED_FLOW, tof=COMBINED_TOF, seed=27),
    M3Case("case9_yaw90_ideal", "ideal sensors, initial yaw = 90 deg",
           M0_MODEL, initial_yaw_deg=90.0),
    M3Case("case10_yaw90_combined", "combined, initial yaw = 90 deg",
           M1_MODEL, plant=COMBINED_PLANT,
           imu=COMBINED_IMU, flow=COMBINED_FLOW, tof=COMBINED_TOF,
           seed=27, initial_yaw_deg=90.0),
)


@dataclass(frozen=True)
class M3CaseReport:
    name: str
    description: str
    model_file: str
    seed: int
    initial_yaw_deg: float
    plant: dict
    imu: dict
    flow: dict
    tof: dict
    estimator: dict
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
    est_forward_at_brake: float
    truth_forward_at_brake: float
    est_forward_at_land: float
    truth_forward_at_land: float
    est_lateral_at_land: float
    truth_lateral_at_land: float
    truth_braking_distance: float
    est_final_xyz: tuple[float, float, float]
    truth_final_forward_lateral: tuple[float, float]
    truth_start_position_xy: tuple[float, float]
    truth_final_position_xy: tuple[float, float]
    horizontal_position_rms_error: float
    horizontal_position_max_error: float
    horizontal_position_final_error: float
    altitude_rms_error: float
    altitude_max_error: float
    velocity_rms_error: float
    velocity_max_error: float
    max_yaw_error_deg: float
    final_yaw_error_deg: float
    max_roll_deg: float
    max_pitch_deg: float
    max_motor_command: float
    min_motor_command: float
    saturation_fraction: float
    landing_success: bool
    has_nan_or_inf: bool
    min_altitude: float
    imu_samples: int
    flow_samples: int
    flow_invalid: int
    tof_samples: int
    tof_invalid: int
    unhealthy_transitions: int
    estimator_unhealthy_abort: bool
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _rms(values: list[float]) -> float:
    if not values:
        return math.inf
    return math.sqrt(sum(v * v for v in values) / len(values))


def run_m3_case(
    case: M3Case,
    mission_config: MissionConfig | None = None,
    max_total_time: float = MAX_TOTAL_TIME,
) -> M3CaseReport:
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
        initial_yaw=math.radians(case.initial_yaw_deg),
        # Factory (not instance): reset_model would rebuild an identical
        # deterministic pipeline.
        localization=lambda physics_dt: LocalizationPipeline.from_configs(
            case.imu,
            case.flow,
            case.tof,
            case.estimator,
            physics_dt,
            case.seed,
        ),
    )
    pipeline: LocalizationPipeline = simulation.localization
    mission = MissionStateMachine(config)

    # Scoring frame from TRUTH (evaluation only, exactly like M2).
    truth0 = simulation.controller.read_state(simulation.data)
    truth_yaw0 = float(rotation_to_euler(truth0.rotation_body_to_world)[2])
    truth_frame = MissionFrame(truth0.position[:2].copy(), truth_yaw0)
    cos0, sin0 = math.cos(truth_yaw0), math.sin(truth_yaw0)

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

    # Mission input is the ESTIMATED state. last_measured_state in
    # localization mode is exactly the estimator output.
    command = mission.start(
        simulation.last_measured_state, simulation.data.time
    )
    send(command)

    min_altitude = math.inf
    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    max_motor_command = 0.0
    min_motor_command = math.inf
    total_steps = 0
    saturated_steps = 0
    has_nan_or_inf = False
    runtime_error: str | None = None
    estimator_unhealthy_abort = False

    position_errors: list[float] = []
    altitude_errors: list[float] = []
    velocity_errors: list[float] = []
    yaw_errors_deg: list[float] = []

    truth_snapshots: dict[str, dict[str, float]] = {}
    est_final = simulation.last_measured_state.position.copy()
    truth_final_proj = (0.0, 0.0)
    truth_final_z = float(truth0.position[2])
    truth_start_xy = truth0.position[:2].copy()
    truth_final_xy = truth_start_xy.copy()
    previous_phase = mission.phase

    while True:
        try:
            truth = simulation.step(simulation.data.time)
        except SimulationSafetyError as error:
            runtime_error = f"SIMULATION_SAFETY: {error}"
            has_nan_or_inf = True
            break
        estimated = simulation.last_estimated_state
        est_state = simulation.last_measured_state
        t = simulation.data.time

        truth_forward, truth_lateral = truth_frame.project(truth.position[:2])
        truth_final_proj = (truth_forward, truth_lateral)
        truth_final_xy = truth.position[:2].copy()
        truth_final_z = float(truth.position[2])
        est_final = est_state.position.copy()

        # Estimation-error metrics in the mission frame.
        position_errors.append(
            float(
                np.linalg.norm(
                    est_state.position[:2] - np.array([truth_forward, truth_lateral])
                )
            )
        )
        altitude_errors.append(float(est_state.position[2]) - truth_final_z)
        truth_vx_local = cos0 * truth.velocity_world[0] + sin0 * truth.velocity_world[1]
        truth_vy_local = -sin0 * truth.velocity_world[0] + cos0 * truth.velocity_world[1]
        velocity_errors.append(
            float(
                np.linalg.norm(
                    est_state.velocity_world
                    - np.array([truth_vx_local, truth_vy_local, truth.velocity_world[2]])
                )
            )
        )
        est_yaw = float(
            rotation_to_euler(est_state.rotation_body_to_world)[2]
        )
        truth_yaw_rel = wrap_angle(
            float(rotation_to_euler(truth.rotation_body_to_world)[2]) - truth_yaw0
        )
        yaw_errors_deg.append(math.degrees(abs(wrap_angle(est_yaw - truth_yaw_rel))))

        min_altitude = min(min_altitude, truth_final_z)
        euler_deg = np.degrees(rotation_to_euler(truth.rotation_body_to_world))
        max_roll_deg = max(max_roll_deg, abs(float(euler_deg[0])))
        max_pitch_deg = max(max_pitch_deg, abs(float(euler_deg[1])))
        motors = simulation.last_motor_thrusts
        max_motor_command = max(max_motor_command, float(np.max(motors)))
        min_motor_command = min(min_motor_command, float(np.min(motors)))
        total_steps += 1
        limits = simulation.controller
        if np.any(motors >= limits.maximum_thrusts - 1e-9) or np.any(
            motors <= limits.minimum_thrusts + 1e-9
        ):
            saturated_steps += 1

        # Estimator health gate (LAND excluded: ToF is legitimately blind
        # below its minimum range during touchdown).
        if (
            mission.phase in HEALTH_GATED_PHASES
            and t > HEALTH_STARTUP_GRACE_S
            and not estimated.healthy
        ):
            runtime_error = (
                f"ESTIMATOR_UNHEALTHY in {mission.phase.value} "
                f"(imu_age={estimated.imu_age:.3f} flow_age={estimated.flow_age:.3f} "
                f"tof_age={estimated.tof_age:.3f})"
            )
            estimator_unhealthy_abort = True
            break

        command = mission.update(
            simulation.last_measured_state,
            simulation.machine.state.value,
            t,
        )
        if mission.phase is not previous_phase:
            truth_snapshots[mission.phase.value] = {
                "time": float(t),
                "progress": truth_forward,
                "lateral": truth_lateral,
            }
            previous_phase = mission.phase
        send(command)

        if mission.phase in (MissionPhase.DONE, MissionPhase.FAILED):
            break
        if t > max_total_time:
            runtime_error = "BENCHMARK_TIMEOUT"
            break

    log = mission.log
    nan = math.nan

    def log_time(key: str) -> float:
        return float(log.get(key, nan))

    def snapshot(phase: MissionPhase, key: str) -> float:
        entry = truth_snapshots.get(phase.value)
        return float(entry[key]) if entry is not None else nan

    takeoff_time = log_time("takeoff_complete_time") - log_time("takeoff_entry_time")
    settle_time = log_time("settle_complete_time") - log_time("takeoff_complete_time")
    forward_time = log_time("brake_entry_time") - log_time("forward_entry_time")
    brake_time = log_time("hover_entry_time") - log_time("brake_entry_time")
    hover_time = log_time("land_entry_time") - log_time("hover_entry_time")
    land_time = log_time("done_time") - log_time("land_entry_time")
    total_time = log_time("done_time") - log_time("takeoff_entry_time")

    truth_forward_at_brake = snapshot(MissionPhase.BRAKE, "progress")
    truth_forward_at_land = snapshot(MissionPhase.LAND, "progress")
    truth_lateral_at_land = snapshot(MissionPhase.LAND, "lateral")
    truth_forward_at_hover = snapshot(MissionPhase.HOVER, "progress")
    est_forward_at_brake = float(log.get("brake_entry_measured_progress", nan))
    est_forward_at_land = float(log.get("land_entry_measured_progress", nan))
    est_lateral_at_land = float(log.get("land_entry_measured_lateral", nan))
    truth_braking = truth_forward_at_hover - truth_forward_at_brake

    mission_done = mission.phase is MissionPhase.DONE
    mission_result = mission.phase.value
    failure_reason = mission.failure_reason or runtime_error
    saturation_fraction = (
        saturated_steps / total_steps if total_steps else math.inf
    )
    ground_penetration = min_altitude < MIN_ALTITUDE_FLOOR
    landing_success = mission_done and truth_final_z < 0.15
    max_tilt = max(max_roll_deg, max_pitch_deg)

    reasons: list[str] = []
    if not mission_done:
        reasons.append(f"mission {mission_result}: {failure_reason}")
    if not (GATE_FORWARD_MIN <= truth_forward_at_land <= GATE_FORWARD_MAX):
        reasons.append(
            f"truth forward {truth_forward_at_land:.3f} m outside "
            f"[{GATE_FORWARD_MIN}, {GATE_FORWARD_MAX}]"
        )
    if not math.isnan(truth_lateral_at_land) and abs(truth_lateral_at_land) > GATE_LATERAL:
        reasons.append(f"truth lateral {truth_lateral_at_land:.3f} m")
    altitude_rms = _rms(altitude_errors)
    if altitude_rms > GATE_ALTITUDE_RMS:
        reasons.append(f"altitude est-vs-truth RMS {altitude_rms:.3f} m")
    if not landing_success:
        reasons.append("landing did not complete with disarm near ground")
    if max_tilt > GATE_TILT_DEG:
        reasons.append(f"max tilt {max_tilt:.2f} deg")
    if saturation_fraction > GATE_SATURATION_FRACTION:
        reasons.append(f"saturation fraction {saturation_fraction:.2f}")
    if has_nan_or_inf:
        reasons.append("NaN/Inf or simulation safety abort")
    if ground_penetration:
        reasons.append(f"ground penetration: min altitude {min_altitude:.3f} m")
    if estimator_unhealthy_abort:
        reasons.append("estimator became unhealthy during active mission")

    return M3CaseReport(
        name=case.name,
        description=case.description,
        model_file=case.model_path.name,
        seed=case.seed,
        initial_yaw_deg=case.initial_yaw_deg,
        plant=asdict(case.plant),
        imu=asdict(case.imu),
        flow=asdict(case.flow),
        tof=asdict(case.tof),
        estimator=asdict(case.estimator),
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
        est_forward_at_brake=est_forward_at_brake,
        truth_forward_at_brake=truth_forward_at_brake,
        est_forward_at_land=est_forward_at_land,
        truth_forward_at_land=truth_forward_at_land,
        est_lateral_at_land=est_lateral_at_land,
        truth_lateral_at_land=truth_lateral_at_land,
        truth_braking_distance=truth_braking,
        est_final_xyz=tuple(float(v) for v in est_final),
        truth_final_forward_lateral=tuple(float(v) for v in truth_final_proj),
        truth_start_position_xy=tuple(float(v) for v in truth_start_xy),
        truth_final_position_xy=tuple(float(v) for v in truth_final_xy),
        horizontal_position_rms_error=_rms(position_errors),
        horizontal_position_max_error=max(position_errors, default=math.inf),
        horizontal_position_final_error=position_errors[-1]
        if position_errors
        else math.inf,
        altitude_rms_error=altitude_rms,
        altitude_max_error=max((abs(e) for e in altitude_errors), default=math.inf),
        velocity_rms_error=_rms(velocity_errors),
        velocity_max_error=max(velocity_errors, default=math.inf),
        max_yaw_error_deg=max(yaw_errors_deg, default=math.inf),
        final_yaw_error_deg=yaw_errors_deg[-1] if yaw_errors_deg else math.inf,
        max_roll_deg=max_roll_deg,
        max_pitch_deg=max_pitch_deg,
        max_motor_command=max_motor_command,
        min_motor_command=min_motor_command,
        saturation_fraction=saturation_fraction,
        landing_success=landing_success,
        has_nan_or_inf=has_nan_or_inf,
        min_altitude=min_altitude,
        imu_samples=pipeline.imu_samples,
        flow_samples=pipeline.flow_samples,
        flow_invalid=pipeline.flow_invalid,
        tof_samples=pipeline.tof_samples,
        tof_invalid=pipeline.tof_invalid,
        unhealthy_transitions=pipeline.unhealthy_transitions,
        estimator_unhealthy_abort=estimator_unhealthy_abort,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def run_m3_benchmark(cases: Sequence[M3Case] = M3_CASES) -> list[M3CaseReport]:
    return [run_m3_case(case) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="m3_localization_report.json")
    args = parser.parse_args()

    reports = run_m3_benchmark()

    print("=" * 108)
    print(
        "case                     mission truth_fwd  est_fwd   lat    posRMS "
        "yawerr  alterr  tilt  result"
    )
    print("=" * 108)
    for r in reports:
        print(
            f"{r.name:<24} {r.mission_result:<7} "
            f"{r.truth_forward_at_land:7.3f}m {r.est_forward_at_land:7.3f}m "
            f"{r.truth_lateral_at_land:+6.3f}m "
            f"{r.horizontal_position_rms_error:6.3f}m "
            f"{r.max_yaw_error_deg:6.2f}d {r.altitude_rms_error:6.3f}m "
            f"{max(r.max_roll_deg, r.max_pitch_deg):5.1f}d "
            f"{'PASS' if r.passed else 'FAIL'}"
        )
    print("=" * 108)
    overall = all(r.passed for r in reports)
    print(f"M3 RESULT: {'PASS' if overall else 'FAIL'}")
    for r in reports:
        if not r.passed:
            print(f"  {r.name}: {'; '.join(r.reasons)}")

    payload = {
        "overall_pass": overall,
        "gates": {
            "forward_min": GATE_FORWARD_MIN,
            "forward_max": GATE_FORWARD_MAX,
            "lateral": GATE_LATERAL,
            "altitude_rms": GATE_ALTITUDE_RMS,
            "tilt_deg": GATE_TILT_DEG,
            "saturation_fraction": GATE_SATURATION_FRACTION,
        },
        "cases": [asdict(r) for r in reports],
    }
    Path(args.json).write_text(json.dumps(payload, indent=2))
    print(f"metrics written to {args.json}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
