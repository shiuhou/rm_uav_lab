"""M2 autonomous mission: takeoff -> settle -> forward 5 m -> brake -> hover
-> land, driven ONLY by the measured state.

Architecture rule (central to M2):

    measured/estimated state --> MissionStateMachine --> velocity commands
    ground truth             --> benchmark scoring ONLY

This module never sees MuJoCo ground truth: `start()` and `update()` take a
measured QuadrotorState and the discrete flight-mode string. The benchmark
wires measured state in and scores with ground truth independently.

The mission frame is heading-relative: at mission start the measured
position p0_xy and yaw0 define "forward" as h = [cos yaw0, sin yaw0];
"forward 5 m" means 5 m along h, never world +X.

Commands are expressed in the existing normalized action interface
(forward/right/up/yaw in [-1, 1]), so the production path
(integrate_manual -> position controller -> SO(3) -> mixer -> motors)
is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from fly import QuadrotorState, rotation_to_euler


class MissionPhase(str, Enum):
    READY = "READY"
    TAKEOFF = "TAKEOFF"
    SETTLE = "SETTLE"
    FORWARD = "FORWARD"
    BRAKE = "BRAKE"
    HOVER = "HOVER"
    LAND = "LAND"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MissionConfig:
    """All mission numbers in SI units; sim-time based, not wall clock."""

    takeoff_altitude: float = 1.5
    takeoff_altitude_tolerance: float = 0.10
    takeoff_vertical_speed_tolerance: float = 0.12
    settle_duration: float = 1.0
    settle_horizontal_speed_tolerance: float = 0.15
    forward_distance: float = 5.0
    max_forward_speed: float = 0.8
    # 最小接近速度決定「target 領先機身」的穩態 lag：position-mode 下
    # target 凍結時機身會繼續走到 target，lag ≈ v_approach × τ_lag
    # （τ_lag ≈ 3 s 為 PD + 姿態內迴路的實測等效值）。0.15 m/s 時
    # overshoot 約 0.44 m（超出 5.3 m 門檻，實測確認）；0.05 m/s 時
    # 約 0.15 m。這是 mission 參數修正，不是控制器調參。
    minimum_approach_speed: float = 0.05
    # 減速距離必須大於巡航時的 target 領先量（v_max × τ_cascade ≈
    # 0.8 m/s × 1.35 s ≈ 1.1 m，實測）。1.0 m 時 target 在機身抵達前就
    # 已衝過 5 m，凍結後機身收斂到 5.46 m（實測 FAIL）；2.0 m 時
    # target 漸近 5 m，煞車後停在約 5.05–5.10 m。
    slowdown_distance: float = 2.0
    # measured remaining distance below which the forward command stops.
    arrival_threshold: float = 0.02
    brake_speed_threshold: float = 0.20
    brake_hold_duration: float = 0.4
    hover_duration: float = 1.0
    # m/s represented by a full-axis command; must equal
    # FlightParameters.max_horizontal_speed of the simulator in use.
    velocity_command_limit: float = 1.0
    takeoff_timeout: float = 8.0
    settle_timeout: float = 4.0
    # 減速距離 2.0 m 使 FORWARD 需約 10–11 s；timeout 留足裕度。
    forward_timeout: float = 15.0
    brake_timeout: float = 4.0
    hover_timeout: float = 3.0
    land_timeout: float = 10.0


@dataclass(frozen=True)
class MissionCommand:
    """One tick of mission output in the normalized action interface."""

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0
    action: str | None = None


class MissionFrame:
    """Heading-relative frame captured at mission start."""

    def __init__(self, origin_xy: np.ndarray, yaw: float) -> None:
        self.origin_xy = np.asarray(origin_xy, dtype=float)
        self.yaw = float(yaw)
        self.heading = np.array([math.cos(yaw), math.sin(yaw)])
        self.lateral = np.array([-math.sin(yaw), math.cos(yaw)])

    def project(self, position_xy: np.ndarray) -> tuple[float, float]:
        """Return (forward_progress, lateral_error) of a world XY point."""

        delta = np.asarray(position_xy, dtype=float) - self.origin_xy
        return float(delta @ self.heading), float(delta @ self.lateral)


_PHASE_TIMEOUTS = {
    MissionPhase.TAKEOFF: "takeoff_timeout",
    MissionPhase.SETTLE: "settle_timeout",
    MissionPhase.FORWARD: "forward_timeout",
    MissionPhase.BRAKE: "brake_timeout",
    MissionPhase.HOVER: "hover_timeout",
    MissionPhase.LAND: "land_timeout",
}

# Hold-duration comparisons use sim timestamps; a tiny epsilon absorbs the
# float representation of e.g. 4.6 - 3.6 = 0.9999999999999999.
_TIME_EPS = 1e-9


class MissionStateMachine:
    """Measured-state mission logic. See module docstring for the
    ground-truth rule; this class has no access to it by construction."""

    def __init__(self, config: MissionConfig | None = None) -> None:
        self.config = config or MissionConfig()
        self.phase = MissionPhase.READY
        self.failure_reason: str | None = None
        self.frame: MissionFrame | None = None
        self.measured_start_altitude = 0.0
        self.phase_start_time = 0.0
        self._stable_since: float | None = None
        self._slow_since: float | None = None
        # Event log for the benchmark (measured-side quantities only).
        self.log: dict[str, float] = {}

    def start(self, measured: QuadrotorState, sim_time: float) -> MissionCommand:
        if self.phase is not MissionPhase.READY:
            raise RuntimeError("mission can only start from READY")
        yaw0 = float(rotation_to_euler(measured.rotation_body_to_world)[2])
        self.frame = MissionFrame(measured.position[:2].copy(), yaw0)
        self.measured_start_altitude = float(measured.position[2])
        self._enter(MissionPhase.TAKEOFF, sim_time)
        return MissionCommand(action="takeoff")

    def update(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        if self.phase in (MissionPhase.DONE, MissionPhase.FAILED):
            return MissionCommand()
        if self.phase is MissionPhase.READY:
            return MissionCommand()

        if not self._finite(measured):
            return self._fail("MEASUREMENT_NOT_FINITE", sim_time)
        timeout = getattr(self.config, _PHASE_TIMEOUTS[self.phase])
        if sim_time - self.phase_start_time > timeout:
            return self._fail(f"{self.phase.value}_TIMEOUT", sim_time)

        handler = {
            MissionPhase.TAKEOFF: self._update_takeoff,
            MissionPhase.SETTLE: self._update_settle,
            MissionPhase.FORWARD: self._update_forward,
            MissionPhase.BRAKE: self._update_brake,
            MissionPhase.HOVER: self._update_hover,
            MissionPhase.LAND: self._update_land,
        }[self.phase]
        return handler(measured, flight_state, sim_time)

    # ---- phase handlers ------------------------------------------------

    def _update_takeoff(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        config = self.config
        altitude_ok = (
            abs(measured.position[2] - config.takeoff_altitude)
            < config.takeoff_altitude_tolerance
        )
        speed_ok = (
            abs(measured.velocity_world[2])
            < config.takeoff_vertical_speed_tolerance
        )
        if altitude_ok and speed_ok:
            self.log["takeoff_complete_time"] = sim_time
            self.log["takeoff_measured_altitude"] = float(measured.position[2])
            self._enter(MissionPhase.SETTLE, sim_time)
        return MissionCommand()

    def _update_settle(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        config = self.config
        stable = (
            abs(measured.position[2] - config.takeoff_altitude)
            < config.takeoff_altitude_tolerance
            and abs(measured.velocity_world[2])
            < config.takeoff_vertical_speed_tolerance
            and np.linalg.norm(measured.velocity_world[:2])
            < config.settle_horizontal_speed_tolerance
        )
        if not stable:
            self._stable_since = None
            return MissionCommand()
        if self._stable_since is None:
            self._stable_since = sim_time
        if sim_time - self._stable_since >= config.settle_duration - _TIME_EPS:
            self.log["settle_complete_time"] = sim_time
            self._enter(MissionPhase.FORWARD, sim_time)
            return self._forward_command(measured)
        return MissionCommand()

    def _update_forward(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        progress, _ = self.frame.project(measured.position[:2])
        remaining = self.config.forward_distance - progress
        if remaining <= self.config.arrival_threshold:
            self.log["brake_entry_time"] = sim_time
            self.log["brake_entry_measured_progress"] = progress
            self._enter(MissionPhase.BRAKE, sim_time)
            return MissionCommand()
        return self._forward_command(measured)

    def _update_brake(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        speed = float(np.linalg.norm(measured.velocity_world[:2]))
        if speed >= self.config.brake_speed_threshold:
            self._slow_since = None
            return MissionCommand()
        if self._slow_since is None:
            self._slow_since = sim_time
        if (
            sim_time - self._slow_since
            >= self.config.brake_hold_duration - _TIME_EPS
        ):
            progress, _ = self.frame.project(measured.position[:2])
            self.log["hover_entry_time"] = sim_time
            self.log["hover_entry_measured_progress"] = progress
            self.log["hover_entry_measured_speed"] = speed
            self._enter(MissionPhase.HOVER, sim_time)
        return MissionCommand()

    def _update_hover(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        if (
            sim_time - self.phase_start_time
            >= self.config.hover_duration - _TIME_EPS
        ):
            progress, lateral = self.frame.project(measured.position[:2])
            self.log["land_entry_time"] = sim_time
            self.log["land_entry_measured_progress"] = progress
            self.log["land_entry_measured_lateral"] = lateral
            self._enter(MissionPhase.LAND, sim_time)
            return MissionCommand(action="land")
        return MissionCommand()

    def _update_land(
        self, measured: QuadrotorState, flight_state: str, sim_time: float
    ) -> MissionCommand:
        if flight_state == "DISARMED":
            self.log["done_time"] = sim_time
            self._enter(MissionPhase.DONE, sim_time)
        return MissionCommand()

    # ---- helpers -------------------------------------------------------

    def _forward_command(self, measured: QuadrotorState) -> MissionCommand:
        """Distance-aware forward speed: cruise, then linear slowdown."""

        config = self.config
        progress, _ = self.frame.project(measured.position[:2])
        remaining = config.forward_distance - progress
        if remaining > config.slowdown_distance:
            speed = config.max_forward_speed
        else:
            speed = float(
                np.clip(
                    config.max_forward_speed
                    * remaining
                    / config.slowdown_distance,
                    config.minimum_approach_speed,
                    config.max_forward_speed,
                )
            )
        axis = float(np.clip(speed / config.velocity_command_limit, -1.0, 1.0))
        return MissionCommand(forward=axis)

    def _enter(self, phase: MissionPhase, sim_time: float) -> None:
        self.phase = phase
        self.phase_start_time = sim_time
        self._stable_since = None
        self._slow_since = None
        self.log[f"{phase.value.lower()}_entry_time"] = sim_time

    def _fail(self, reason: str, sim_time: float) -> MissionCommand:
        self.phase = MissionPhase.FAILED
        self.failure_reason = reason
        self.log["failed_time"] = sim_time
        return MissionCommand()

    @staticmethod
    def _finite(state: QuadrotorState) -> bool:
        return bool(
            np.all(np.isfinite(state.position))
            and np.all(np.isfinite(state.velocity_world))
            and np.all(np.isfinite(state.quaternion_wxyz))
            and np.all(np.isfinite(state.angular_velocity_body))
        )
