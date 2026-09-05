#!/usr/bin/env python3
"""MuJoCo simulator process with localhost UDP command and telemetry IPC."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import math
from pathlib import Path
import socket
import sys
import time
from typing import Sequence

import mujoco
import numpy as np

from fly import (
    ControlTarget,
    DEFAULT_MODEL_PATH,
    MAX_TARGET_HEIGHT,
    MIN_TARGET_HEIGHT,
    QuadrotorController,
    QuadrotorState,
    SimulationSafetyError,
    load_simulation,
    positive_float,
    rotation_to_euler,
    step_controller,
    wrap_angle,
)
from ipc_protocol import (
    CommandPacket,
    DEFAULT_COMMAND_PORT,
    DEFAULT_HOST,
    DEFAULT_TELEMETRY_PORT,
    MAX_PACKET_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    TelemetryPacket,
    decode_command,
    encode_telemetry,
)
from m1_realism import (
    MeasurementModel,
    NoiseConfig,
    PlantConfig,
    apply_plant_config,
)
from m3_sensors import TruthKinematics


# 狀態機是飛行模式的唯一來源；UI 只能送 action，不能直接指定下一個 state。
class FlightState(str, Enum):
    DISARMED = "DISARMED"
    ARMED_IDLE = "ARMED_IDLE"
    TAKING_OFF = "TAKING_OFF"
    HOVERING = "HOVERING"
    MANUAL = "MANUAL"
    LANDING = "LANDING"
    LANDED = "LANDED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class FlightParameters:
    # profile 參數管理自動起降；speed/rate 參數限制手動 target 的變化速度。
    takeoff_height: float = 1.2
    takeoff_duration: float = 3.0
    landing_max_speed: float = 0.32
    landing_min_duration: float = 3.0
    ground_target_height: float = 0.04
    max_horizontal_speed: float = 1.0
    max_vertical_speed: float = 0.6
    max_yaw_rate: float = math.radians(60.0)
    horizontal_target_limit: float = 10.0
    command_timeout: float = 0.5
    transition_hold_time: float = 0.4
    motor_ramp_time: float = 0.5


@dataclass(frozen=True)
class ManualAxes:
    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0

    def is_neutral(self, tolerance: float = 1e-6) -> bool:
        return all(
            abs(value) <= tolerance
            for value in (self.forward, self.right, self.up, self.yaw)
        )


@dataclass(frozen=True)
class SmoothVerticalProfile:
    start_position: np.ndarray
    end_height: float
    yaw: float
    start_time: float
    duration: float

    def evaluate(self, sim_time: float) -> ControlTarget:
        phase = float(
            np.clip((sim_time - self.start_time) / self.duration, 0.0, 1.0)
        )
        # s=3t²-2t³ 讓高度與速度在起訖點連續，減少接地/離地衝擊。
        smooth = phase * phase * (3.0 - 2.0 * phase)
        smooth_rate = 6.0 * phase * (1.0 - phase) / self.duration
        delta = self.end_height - float(self.start_position[2])
        position = self.start_position.copy()
        velocity = np.zeros(3, dtype=float)
        position[2] += delta * smooth
        velocity[2] = delta * smooth_rate
        return ControlTarget(position, velocity, self.yaw)

    def complete(self, sim_time: float) -> bool:
        return sim_time >= self.start_time + self.duration


class FlightStateMachine:
    """All flight mode transitions and high-level target generation."""

    # 此類別只產生 target 和馬達收油命令，不直接寫 MuJoCo qpos/qvel。

    def __init__(
        self, initial_state: QuadrotorState, parameters: FlightParameters
    ) -> None:
        self.parameters = parameters
        self.state = FlightState.DISARMED
        self.target_position = initial_state.position.copy()
        self.target_yaw = self._yaw(initial_state)
        self.profile: SmoothVerticalProfile | None = None
        self.condition_since: float | None = None
        self.neutral_since: float | None = None
        self.motor_ramp_start: float | None = None
        self.motor_ramp_initial = np.zeros(4, dtype=float)
        self.landed_since: float | None = None
        self.last_state_change = 0.0
        self.history: list[FlightState] = [self.state]

    @staticmethod
    def _yaw(state: QuadrotorState) -> float:
        return float(rotation_to_euler(state.rotation_body_to_world)[2])

    def _set_state(self, state: FlightState, sim_time: float) -> None:
        if state == self.state:
            return
        self.state = state
        self.last_state_change = sim_time
        self.history.append(state)

    def _ground_safe(self, state: QuadrotorState) -> bool:
        # ARM、DISARM、RESET 都共用這個地面判斷，避免翻覆或仍在移動時切換。
        euler = rotation_to_euler(state.rotation_body_to_world)
        return (
            state.position[2] < 0.16
            and np.linalg.norm(state.velocity_world) < 0.20
            and np.linalg.norm(state.angular_velocity_body) < 0.35
            and abs(euler[0]) < math.radians(15.0)
            and abs(euler[1]) < math.radians(15.0)
        )

    def can_reset(self, state: QuadrotorState) -> bool:
        return self.state in (
            FlightState.DISARMED,
            FlightState.LANDED,
            FlightState.EMERGENCY_STOP,
        ) and self._ground_safe(state)

    def handle_action(
        self, action: str, vehicle: QuadrotorState, sim_time: float
    ) -> bool:
        # 每個 action 由對應函式檢查合法來源 state；回傳值表示是否真的生效。
        if action == "arm":
            return self._arm(vehicle, sim_time)
        if action == "disarm":
            return self._disarm(vehicle, sim_time)
        if action == "takeoff":
            return self._takeoff(vehicle, sim_time)
        if action == "hover":
            return self._hover(vehicle, sim_time)
        if action == "land":
            return self._land(vehicle, sim_time)
        if action == "emergency_stop":
            self.profile = None
            self.motor_ramp_start = None
            self._set_state(FlightState.EMERGENCY_STOP, sim_time)
            return True
        return False

    def _arm(self, vehicle: QuadrotorState, sim_time: float) -> bool:
        if self.state not in (FlightState.DISARMED, FlightState.LANDED):
            return False
        if not self._ground_safe(vehicle):
            return False
        self.target_position = vehicle.position.copy()
        self.target_yaw = self._yaw(vehicle)
        self._set_state(FlightState.ARMED_IDLE, sim_time)
        return True

    def _disarm(self, vehicle: QuadrotorState, sim_time: float) -> bool:
        if self.state not in (FlightState.ARMED_IDLE, FlightState.LANDED):
            return False
        if not self._ground_safe(vehicle):
            return False
        self.profile = None
        self.motor_ramp_start = None
        self._set_state(FlightState.DISARMED, sim_time)
        return True

    def _takeoff(self, vehicle: QuadrotorState, sim_time: float) -> bool:
        # TAKEOFF 可從 DISARMED 一鍵觸發：先走同一套安全 ARM，再建立高度軌跡。
        if self.state in (FlightState.DISARMED, FlightState.LANDED):
            if not self._arm(vehicle, sim_time):
                return False
        if self.state != FlightState.ARMED_IDLE or not self._ground_safe(vehicle):
            return False
        self.target_position = vehicle.position.copy()
        self.target_position[2] = self.parameters.takeoff_height
        self.target_yaw = self._yaw(vehicle)
        self.profile = SmoothVerticalProfile(
            vehicle.position.copy(),
            self.parameters.takeoff_height,
            self.target_yaw,
            sim_time,
            self.parameters.takeoff_duration,
        )
        self.condition_since = None
        self.motor_ramp_start = None
        self._set_state(FlightState.TAKING_OFF, sim_time)
        return True

    def _hover(self, vehicle: QuadrotorState, sim_time: float) -> bool:
        allowed = self.state in (FlightState.HOVERING, FlightState.MANUAL)
        allowed = allowed or (
            self.state == FlightState.LANDING and vehicle.position[2] > 0.25
        )
        if not allowed:
            return False
        self.target_position = vehicle.position.copy()
        self.target_position[2] = np.clip(
            self.target_position[2], MIN_TARGET_HEIGHT, MAX_TARGET_HEIGHT
        )
        self.target_yaw = self._yaw(vehicle)
        self.profile = None
        self.motor_ramp_start = None
        self.condition_since = None
        self._set_state(FlightState.HOVERING, sim_time)
        return True

    def _land(self, vehicle: QuadrotorState, sim_time: float) -> bool:
        if self.state not in (
            FlightState.TAKING_OFF,
            FlightState.HOVERING,
            FlightState.MANUAL,
        ):
            return False
        self.target_position = vehicle.position.copy()
        self.target_position[2] = self.parameters.ground_target_height
        self.target_yaw = self._yaw(vehicle)
        height_delta = max(
            float(vehicle.position[2]) - self.parameters.ground_target_height, 0.0
        )
        # smoothstep 的峰值速度為 1.5*高度差/duration，據此反推 duration，
        # 可保證整段下降不超過 landing_max_speed。
        duration = max(
            self.parameters.landing_min_duration,
            1.5 * height_delta / self.parameters.landing_max_speed,
        )
        self.profile = SmoothVerticalProfile(
            vehicle.position.copy(),
            self.parameters.ground_target_height,
            self.target_yaw,
            sim_time,
            duration,
        )
        self.condition_since = None
        self.motor_ramp_start = None
        self._set_state(FlightState.LANDING, sim_time)
        return True

    def integrate_manual(
        self,
        command: ManualAxes,
        vehicle: QuadrotorState,
        timestep: float,
        sim_time: float,
    ) -> None:
        if self.state not in (FlightState.HOVERING, FlightState.MANUAL):
            return
        if command.is_neutral():
            if self.state == FlightState.MANUAL:
                if self.neutral_since is None:
                    self.neutral_since = sim_time
                elif (
                    sim_time - self.neutral_since >= 0.30
                    and np.linalg.norm(vehicle.velocity_world) < 0.30
                ):
                    self._set_state(FlightState.HOVERING, sim_time)
            return

        self.neutral_since = None
        self._set_state(FlightState.MANUAL, sim_time)
        yaw = self._yaw(vehicle)
        # 指令是 heading-relative：body +X 是前、body -Y 是右；只使用當前 yaw，
        # 因此即使機身有小角度 roll/pitch，WASD 仍在水平面上移動。
        forward_world = np.array([math.cos(yaw), math.sin(yaw)])
        right_world = np.array([math.sin(yaw), -math.cos(yaw)])
        horizontal = command.forward * forward_world + command.right * right_world
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm > 1.0:
            horizontal /= horizontal_norm
        # 積分「目標速度」而非直接施力；放開鍵後 target 停在最後位置供控制器保持。
        self.target_position[:2] += (
            horizontal * self.parameters.max_horizontal_speed * timestep
        )
        self.target_position[2] += (
            command.up * self.parameters.max_vertical_speed * timestep
        )
        self.target_yaw = wrap_angle(
            self.target_yaw + command.yaw * self.parameters.max_yaw_rate * timestep
        )
        self.target_position[:2] = np.clip(
            self.target_position[:2],
            -self.parameters.horizontal_target_limit,
            self.parameters.horizontal_target_limit,
        )
        self.target_position[2] = np.clip(
            self.target_position[2], MIN_TARGET_HEIGHT, MAX_TARGET_HEIGHT
        )

    def handle_timeout(self, vehicle: QuadrotorState, sim_time: float) -> None:
        # 斷線 failsafe 捕捉目前 pose 並 HOVER，不延用最後一個可能非零的軸命令。
        if self.state in (
            FlightState.TAKING_OFF,
            FlightState.HOVERING,
            FlightState.MANUAL,
        ):
            self.target_position = vehicle.position.copy()
            self.target_position[2] = np.clip(
                self.target_position[2], MIN_TARGET_HEIGHT, MAX_TARGET_HEIGHT
            )
            self.target_yaw = self._yaw(vehicle)
            self.profile = None
            self.condition_since = None
            self._set_state(FlightState.HOVERING, sim_time)

    def control_target(self, sim_time: float) -> ControlTarget | None:
        if self.state in (
            FlightState.DISARMED,
            FlightState.ARMED_IDLE,
            FlightState.LANDED,
            FlightState.EMERGENCY_STOP,
        ):
            return None
        if self.profile is not None:
            return self.profile.evaluate(sim_time)
        return ControlTarget(
            self.target_position.copy(), np.zeros(3, dtype=float), self.target_yaw
        )

    def update_after_step(
        self,
        vehicle: QuadrotorState,
        sim_time: float,
        motor_thrusts: np.ndarray,
    ) -> None:
        # 完成 profile 不等於完成轉態；誤差/速度條件需連續成立一段時間，
        # 避免剛好穿越門檻的一個 timestep 就被誤判為穩定。
        if self.state == FlightState.TAKING_OFF and self.profile is not None:
            if self.profile.complete(sim_time):
                height_error = abs(
                    float(vehicle.position[2]) - self.parameters.takeoff_height
                )
                stable = height_error < 0.08 and abs(vehicle.velocity_world[2]) < 0.12
                self._hold_condition(stable, sim_time)
                if self._condition_held(sim_time):
                    self.profile = None
                    self.condition_since = None
                    self._set_state(FlightState.HOVERING, sim_time)

        if (
            self.state == FlightState.LANDING
            and self.profile is not None
            and self.motor_ramp_start is None
            and self.profile.complete(sim_time)
        ):
            euler = rotation_to_euler(vehicle.rotation_body_to_world)
            stable = (
                vehicle.position[2] < 0.10
                and abs(vehicle.velocity_world[2]) < 0.12
                and abs(euler[0]) < math.radians(15.0)
                and abs(euler[1]) < math.radians(15.0)
            )
            self._hold_condition(stable, sim_time)
            if self._condition_held(sim_time):
                # 只有確認接地穩定後才開始收油，避免半空中直接將推力歸零。
                self.motor_ramp_start = sim_time
                self.motor_ramp_initial = motor_thrusts.copy()
                self.condition_since = None

        if self.state == FlightState.LANDED and self.landed_since is not None:
            if sim_time - self.landed_since >= 0.25:
                self._set_state(FlightState.DISARMED, sim_time)

    def _hold_condition(self, condition: bool, sim_time: float) -> None:
        if condition:
            if self.condition_since is None:
                self.condition_since = sim_time
        else:
            self.condition_since = None

    def _condition_held(self, sim_time: float) -> bool:
        return (
            self.condition_since is not None
            and sim_time - self.condition_since >= self.parameters.transition_hold_time
        )

    def motor_ramp_command(self, sim_time: float) -> np.ndarray | None:
        if self.state != FlightState.LANDING or self.motor_ramp_start is None:
            return None
        phase = float(
            np.clip(
                (sim_time - self.motor_ramp_start) / self.parameters.motor_ramp_time,
                0.0,
                1.0,
            )
        )
        # 線性 ramp 只負責最後接地後的收油；正常降落仍由位置控制器完成。
        return self.motor_ramp_initial * (1.0 - phase)

    def finish_motor_ramp_if_ready(self, sim_time: float) -> None:
        if self.motor_ramp_start is None:
            return
        if sim_time - self.motor_ramp_start >= self.parameters.motor_ramp_time:
            self.profile = None
            self.motor_ramp_start = None
            self.landed_since = sim_time
            self._set_state(FlightState.LANDED, sim_time)


class QuadrotorSimulation:
    """Own MuJoCo state, flight state machine, and command timeout behavior."""

    # 這是唯一持有並修改 MjData 的類別；Pygame process 永遠只收發高層封包。

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        parameters: FlightParameters | None = None,
        plant_config: PlantConfig | None = None,
        noise_config: NoiseConfig | None = None,
        noise_seed: int = 0,
        initial_yaw: float = 0.0,
        localization=None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model, self.data, self.controller = load_simulation(self.model_path)
        self.parameters = parameters or FlightParameters()
        # M3：localization（sensor->estimator pipeline）與 M1 的
        # truth+noise measurement 是兩條互斥的導航路徑。M3 模式下
        # controller/guidance/mission 只能看到 EstimatedState。
        if localization is not None and noise_config is not None:
            raise ValueError(
                "M3 localization and M1 noise measurement are mutually "
                "exclusive navigation paths"
            )
        # 接受 instance 或 factory(physics_dt)；factory 讓 reset_model 能重建
        # 一份新的 deterministic pipeline，並讓 sensor rate divider 依據
        # 實際 model timestep 計算。
        self._localization_factory = (
            localization
            if callable(localization)
            else (lambda physics_dt: localization)
        )
        self.localization = (
            self._localization_factory(float(self.model.opt.timestep))
            if localization is not None
            else None
        )
        # ToF 的高度基準：ground plane geom 的 z（目前模型是 0）。
        ground_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self.ground_height = (
            float(self.model.geom_pos[ground_id][2]) if ground_id >= 0 else 0.0
        )
        # M1：plant_config 在 controller 建立之後才套用，因此 controller 永遠
        # 使用 nominal 參數，而物理模型使用 perturb 後的參數（這才是不確定性）。
        self.plant_config = plant_config or PlantConfig()
        apply_plant_config(self.model, self.data, self.plant_config)
        # M2：初始航向（rad）用於 heading-invariance 測試；這是初始條件，
        # 不是飛行中的狀態篡改。注意：必須在 apply_plant_config 之後設定，
        # 因為其中的 mj_setConst 會把 qpos 重置回 qpos0。
        self.initial_yaw = float(initial_yaw)
        self._apply_initial_yaw()
        self.noise_config = noise_config
        self.noise_seed = noise_seed
        # M1：measurement 為 None 時控制器直接讀 ground truth（M0 行為不變）。
        self.measurement = (
            MeasurementModel(noise_config, noise_seed)
            if noise_config is not None
            else None
        )
        self.machine = FlightStateMachine(
            self.controller.read_state(self.data), self.parameters
        )
        initial_truth = self.controller.read_state(self.data)
        # 每一步的 measured state；無 noise model 時即 ground truth（M0 行為）。
        # M3：localization 模式時這裡是 estimator 輸出（EstimatedState）。
        self.last_estimated_state = None
        if self.localization is not None:
            self.last_estimated_state = self.localization.update(
                self._truth_kinematics(initial_truth)
            )
            self.last_measured_state = (
                self.last_estimated_state.to_quadrotor_state()
            )
        else:
            self.last_measured_state = (
                self.measurement.measure(initial_truth)
                if self.measurement is not None
                else initial_truth
            )
        self.manual_axes = ManualAxes()
        self.last_motor_thrusts = np.zeros(4, dtype=float)
        self.last_command_sequence = -1
        self.last_action_sequence = -1
        self.last_command_time: float | None = None
        self.timeout_active = False
        self.action_counts: dict[str, int] = {}
        self.max_altitude = float(self.controller.read_state(self.data).position[2])
        self.min_up_axis_z = 1.0

    def _apply_initial_yaw(self) -> None:
        if self.initial_yaw == 0.0:
            return
        half = 0.5 * self.initial_yaw
        address = self.controller.qpos_address
        self.data.qpos[address + 3 : address + 7] = (
            math.cos(half), 0.0, 0.0, math.sin(half),
        )
        mujoco.mj_forward(self.model, self.data)

    def _truth_kinematics(self, vehicle: QuadrotorState) -> TruthKinematics:
        """M3 sensor-boundary input. Only the sensor pipeline may consume
        this; the estimator, controller, guidance and mission never see it.
        acceleration_world is qacc[0:3] of the free joint, which IS the
        world-frame linear acceleration of the body origin (contacts,
        drag, thrust all included). NOTE: mj_objectAcceleration is NOT
        usable here -- data.cacc is only populated when something
        (e.g. a force sensor) forces mj_rnePostConstraint to run, and
        this model has none, so it silently returns zeros."""
        qvel_address = int(self.model.jnt_dofadr[self.controller.root_joint_id])
        return TruthKinematics(
            time=float(self.data.time),
            position=vehicle.position.copy(),
            quaternion_wxyz=vehicle.quaternion_wxyz.copy(),
            rotation_body_to_world=vehicle.rotation_body_to_world.copy(),
            velocity_world=vehicle.velocity_world.copy(),
            angular_velocity_body=vehicle.angular_velocity_body.copy(),
            acceleration_world=self.data.qacc[qvel_address : qvel_address + 3].copy(),
            ground_height=self.ground_height,
        )

    def reset_model(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._apply_initial_yaw()
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset_vertical_integrator()
        # plant_config 作用在 MjModel 上，reset 後仍然有效；noise 重建 RNG
        # 讓每次 rollout 的雜訊序列可重現。
        if self.noise_config is not None:
            self.measurement = MeasurementModel(self.noise_config, self.noise_seed)
        # M3：重建整條 sensor->estimator pipeline，重置 estimator 的
        # local frame 與 deterministic RNG。
        if self.localization is not None:
            self.localization = self._localization_factory(
                float(self.model.opt.timestep)
            )
        self.machine = FlightStateMachine(
            self.controller.read_state(self.data), self.parameters
        )
        # 與 __init__ 一致：measured state 必須經過 measurement model，
        # 否則 noise case 下 mission 起點會洩漏 ground truth。
        reset_truth = self.controller.read_state(self.data)
        if self.localization is not None:
            self.last_estimated_state = self.localization.update(
                self._truth_kinematics(reset_truth)
            )
            self.last_measured_state = (
                self.last_estimated_state.to_quadrotor_state()
            )
        else:
            self.last_measured_state = (
                self.measurement.measure(reset_truth)
                if self.measurement is not None
                else reset_truth
            )
        self.manual_axes = ManualAxes()
        self.last_motor_thrusts = np.zeros(4, dtype=float)
        self.timeout_active = False

    def accept_command(self, packet: CommandPacket, now: float) -> bool:
        # UDP 可能重送或亂序；sequence 確保舊 axes/action 不會覆蓋新命令。
        if packet.sequence <= self.last_command_sequence:
            return False
        self.last_command_sequence = packet.sequence
        # timeout 使用 simulator 本地 monotonic time，不信任遠端封包的 timestamp。
        self.last_command_time = now
        self.timeout_active = False
        self.manual_axes = ManualAxes(
            packet.forward, packet.right, packet.up, packet.yaw
        )

        if packet.action is not None and packet.sequence > self.last_action_sequence:
            vehicle = self.controller.read_state(self.data)
            # M3：action handlers（ARM/TAKEOFF/LAND 的 target 錨定）在
            # localization 模式下必須使用 estimated state，否則
            # truth-anchored target（例如真實 yaw）會讓閉迴路去追一個
            # estimator frame 裡不存在的姿態。M0/M1/M2 路徑維持原樣。
            decision_state = (
                self.last_measured_state
                if self.localization is not None
                else vehicle
            )
            accepted = False
            if packet.action == "reset":
                if self.machine.can_reset(decision_state):
                    self.reset_model()
                    accepted = True
            else:
                accepted = self.machine.handle_action(
                    packet.action, decision_state, self.data.time
                )
            self.last_action_sequence = packet.sequence
            if accepted:
                self.action_counts[packet.action] = (
                    self.action_counts.get(packet.action, 0) + 1
                )
        return True

    def command_age(self, now: float) -> float:
        if self.last_command_time is None:
            return 999.0
        return float(np.clip(now - self.last_command_time, 0.0, 999.0))

    def _apply_timeout(self, now: float, vehicle: QuadrotorState) -> None:
        if self.command_age(now) <= self.parameters.command_timeout:
            return
        self.manual_axes = ManualAxes()
        if not self.timeout_active:
            self.machine.handle_timeout(vehicle, self.data.time)
            self.timeout_active = True

    def step(self, now: float) -> QuadrotorState:
        vehicle = self.controller.read_state(self.data)
        if self.localization is not None:
            # M3：truth 只進入 sensor pipeline；controller、guidance 與
            # mission（經 last_measured_state）收到的都是 EstimatedState。
            # estimator 失能時也絕不 fallback 到 truth。
            self.last_estimated_state = self.localization.update(
                self._truth_kinematics(vehicle)
            )
            measured = self.last_estimated_state.to_quadrotor_state()
            self.last_measured_state = measured
        else:
            # M1/M2：noise 在每個 tick 只取樣一次，guidance（target 積分的
            # heading 參考）與 controller 使用同一份 measured state；
            # 無 noise model 時兩者都是 ground truth。
            measured = (
                self.measurement.measure(vehicle)
                if self.measurement is not None
                else None
            )
            self.last_measured_state = (
                measured if measured is not None else vehicle
            )
        self._apply_timeout(now, vehicle)
        self.machine.integrate_manual(
            self.manual_axes,
            self.last_measured_state,
            self.model.opt.timestep,
            self.data.time,
        )

        # 三條互斥 actuator 路徑：接地收油、無動力狀態、一般閉迴路控制。
        ramp_command = self.machine.motor_ramp_command(self.data.time)
        target = self.machine.control_target(self.data.time)
        if ramp_command is not None:
            self.data.ctrl[:] = np.clip(
                ramp_command,
                self.controller.minimum_thrusts,
                self.controller.maximum_thrusts,
            )
            self.last_motor_thrusts = self.data.ctrl.copy()
            mujoco.mj_step(self.model, self.data)
            self.machine.finish_motor_ramp_if_ready(self.data.time)
        elif target is None:
            self.data.ctrl[:] = 0.0
            self.last_motor_thrusts = np.zeros(4, dtype=float)
            mujoco.mj_step(self.model, self.data)
        else:
            # 控制器只看到 measured state；安全檢查永遠用 ground truth。
            output = step_controller(
                self.model, self.data, self.controller, target,
                measured_state=measured,
            )
            self.last_motor_thrusts = output.motor_thrusts.copy()

        vehicle = self.controller.read_state(self.data)
        self.machine.update_after_step(
            vehicle, self.data.time, self.last_motor_thrusts
        )
        self.max_altitude = max(self.max_altitude, float(vehicle.position[2]))
        self.min_up_axis_z = min(
            self.min_up_axis_z, float(vehicle.rotation_body_to_world[2, 2])
        )
        return vehicle

    def telemetry(self, sequence: int, now: float) -> TelemetryPacket:
        vehicle = self.controller.read_state(self.data)
        attitude = np.degrees(rotation_to_euler(vehicle.rotation_body_to_world))
        return TelemetryPacket(
            PROTOCOL_VERSION,
            sequence,
            now,
            self.machine.state.value,
            self.command_age(now) <= self.parameters.command_timeout,
            tuple(float(value) for value in vehicle.position),
            tuple(float(value) for value in vehicle.velocity_world),
            tuple(float(value) for value in attitude),
            tuple(float(value) for value in self.machine.target_position),
            math.degrees(self.machine.target_yaw),
            tuple(float(value) for value in self.last_motor_thrusts),
            self.command_age(now),
        )


class LocalUdpTransport:
    """Non-blocking localhost-only command receiver and telemetry sender."""

    def __init__(
        self,
        host: str,
        command_port: int,
        telemetry_port: int,
    ) -> None:
        address = ipaddress.ip_address(host)
        if not address.is_loopback:
            raise ValueError("IPC host must be a loopback address")
        self.host = host
        self.telemetry_address = (host, telemetry_port)
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.command_socket.bind((host, command_port))
        self.command_socket.setblocking(False)
        self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.malformed_packets = 0

    def receive_commands(self, simulation: QuadrotorSimulation, now: float) -> int:
        accepted = 0
        # 每個 physics tick 最多清掉 64 包，避免封包洪水拖住 500 Hz 模擬迴圈。
        for _ in range(64):
            try:
                payload, source = self.command_socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                if not ipaddress.ip_address(source[0]).is_loopback:
                    continue
                if len(payload) > MAX_PACKET_BYTES:
                    raise ProtocolError("command packet exceeds size limit")
                command = decode_command(payload)
                if simulation.accept_command(command, now):
                    accepted += 1
            except (ProtocolError, ValueError):
                # 壞封包只計數並丟棄，不讓網路輸入終止 simulation。
                self.malformed_packets += 1
        return accepted

    def send_telemetry(self, telemetry: TelemetryPacket) -> None:
        self.telemetry_socket.sendto(
            encode_telemetry(telemetry), self.telemetry_address
        )

    def close(self) -> None:
        self.command_socket.close()
        self.telemetry_socket.close()


@dataclass(frozen=True)
class ScenarioResult:
    passed: bool
    scenario: str
    state_history: tuple[str, ...]
    max_altitude: float
    hover_error: float
    horizontal_error: float
    final_position: np.ndarray
    final_velocity: np.ndarray
    final_state: str
    final_motor_thrusts: np.ndarray
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _scenario_command(
    sequence: int,
    timestamp: float,
    action: str | None = None,
    forward: float = 0.0,
) -> CommandPacket:
    return CommandPacket(
        PROTOCOL_VERSION, sequence, timestamp, action, forward, 0.0, 0.0, 0.0
    )


def run_headless_scenario(
    scenario: str,
    duration: float,
    target_height: float = 1.2,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> ScenarioResult:
    # scenario 以 MuJoCo simulation time 驅動，結果不受電腦執行快慢影響。
    parameters = FlightParameters(takeoff_height=target_height)
    simulation = QuadrotorSimulation(model_path, parameters)
    sequence = 0
    next_keepalive = 0.0
    land_sent = False
    hover_error = math.inf
    horizontal_error = math.inf

    simulation.accept_command(
        _scenario_command(sequence, simulation.data.time, "takeoff"),
        simulation.data.time,
    )
    sequence += 1

    while simulation.data.time < duration:
        sim_time = simulation.data.time
        if sim_time >= next_keepalive:
            forward = 1.0 if scenario == "command_loss" and 4.5 <= sim_time < 4.6 else 0.0
            if scenario != "command_loss" or sim_time < 4.6:
                simulation.accept_command(
                    _scenario_command(sequence, sim_time, forward=forward), sim_time
                )
                sequence += 1
            next_keepalive = sim_time + 0.05
        if scenario == "takeoff_land" and sim_time >= 6.0 and not land_sent:
            vehicle = simulation.controller.read_state(simulation.data)
            hover_error = abs(float(vehicle.position[2]) - target_height)
            horizontal_error = float(
                np.linalg.norm(
                    vehicle.position[:2] - simulation.machine.target_position[:2]
                )
            )
            simulation.accept_command(
                _scenario_command(sequence, sim_time, "land"), sim_time
            )
            sequence += 1
            land_sent = True
        simulation.step(sim_time)

    vehicle = simulation.controller.read_state(simulation.data)
    if scenario in ("hover", "takeoff_hover", "command_loss"):
        hover_error = abs(float(vehicle.position[2]) - target_height)
        horizontal_error = float(
            np.linalg.norm(
                vehicle.position[:2] - simulation.machine.target_position[:2]
            )
        )

    reasons: list[str] = []
    history = tuple(state.value for state in simulation.machine.history)
    if not np.all(
        np.isfinite(
            np.concatenate(
                (vehicle.position, vehicle.velocity_world, simulation.last_motor_thrusts)
            )
        )
    ):
        reasons.append("state or motor thrust contains NaN/Inf")
    if simulation.max_altitude <= 1.0:
        reasons.append("maximum altitude did not exceed 1.0 m")
    if "TAKING_OFF" not in history or "HOVERING" not in history:
        reasons.append("takeoff did not reach HOVERING")
    if hover_error >= 0.10:
        reasons.append(f"hover height error {hover_error:.3f} m is not below 0.10 m")
    if horizontal_error >= 0.20:
        reasons.append(
            f"horizontal error {horizontal_error:.3f} m is not below 0.20 m"
        )
    if simulation.min_up_axis_z < math.cos(math.radians(45.0)):
        reasons.append("vehicle tilt exceeded 45 degrees")
    if np.any(simulation.last_motor_thrusts < -1e-9) or np.any(
        simulation.last_motor_thrusts > simulation.controller.maximum_thrusts + 1e-9
    ):
        reasons.append("motor thrust outside legal limits")

    if scenario == "takeoff_land":
        if "LANDING" not in history or not ({"LANDED", "DISARMED"} & set(history)):
            reasons.append("landing did not reach LANDED/DISARMED")
        if vehicle.position[2] >= 0.15:
            reasons.append(f"final altitude {vehicle.position[2]:.3f} m is not below 0.15 m")
        if abs(vehicle.velocity_world[2]) >= 0.15:
            reasons.append("final vertical velocity is not below 0.15 m/s")
        if np.max(np.abs(simulation.last_motor_thrusts)) >= 0.05:
            reasons.append("final motor thrust is not near zero")
    elif scenario == "command_loss":
        if not simulation.timeout_active:
            reasons.append("command timeout did not activate")
        if simulation.machine.state != FlightState.HOVERING:
            reasons.append("command timeout did not enter HOVERING")

    return ScenarioResult(
        not reasons,
        scenario,
        history,
        simulation.max_altitude,
        hover_error,
        horizontal_error,
        vehicle.position.copy(),
        vehicle.velocity_world.copy(),
        simulation.machine.state.value,
        simulation.last_motor_thrusts.copy(),
        tuple(reasons),
    )


def print_scenario_result(result: ScenarioResult) -> None:
    print("Headless scenario result")
    print("------------------------")
    print(f"scenario:          {result.scenario}")
    print(f"state_history:     {' -> '.join(result.state_history)}")
    print(f"max_altitude:      {result.max_altitude:.3f} m")
    print(f"hover_height_error:{result.hover_error: .3f} m")
    print(f"horizontal_error:  {result.horizontal_error:.3f} m")
    print("final_position:    " + np.array2string(result.final_position, precision=3))
    print(f"final_vz:          {result.final_velocity[2]:.4f} m/s")
    print(f"final_state:       {result.final_state}")
    print("motor_thrusts:     " + np.array2string(result.final_motor_thrusts, precision=3))
    if result.passed:
        print("PASS")
    else:
        for reason in result.reasons:
            print(f"FAIL: {reason}")


def run_ipc_simulator(
    model_path: Path,
    parameters: FlightParameters,
    host: str,
    command_port: int,
    telemetry_port: int,
    headless: bool,
    duration: float,
    realtime_factor: float,
) -> int:
    simulation = QuadrotorSimulation(model_path, parameters)
    transport = LocalUdpTransport(host, command_port, telemetry_port)
    telemetry_sequence = 0
    next_telemetry = 0.0
    next_sync = 0.0
    viewer_context = nullcontext(None)
    if not headless:
        import mujoco.viewer

        # 不傳 key_callback：MuJoCo 視窗只供觀察/相機操作，飛行鍵盤權歸 Pygame。
        viewer_context = mujoco.viewer.launch_passive(simulation.model, simulation.data)

    try:
        with viewer_context as viewer:
            while simulation.data.time < duration:
                if viewer is not None and not viewer.is_running():
                    break
                wall_start = time.perf_counter()
                now = time.monotonic()
                # physics 依 XML timestep（500 Hz）；telemetry 與 viewer 各自降頻，
                # 避免畫面更新頻率反過來改變控制器的數值結果。
                transport.receive_commands(simulation, now)
                simulation.step(now)

                if now >= next_telemetry:
                    transport.send_telemetry(
                        simulation.telemetry(telemetry_sequence, now)
                    )
                    telemetry_sequence += 1
                    next_telemetry = now + 1.0 / 25.0
                if viewer is not None and now >= next_sync:
                    viewer.sync()
                    next_sync = now + 1.0 / 60.0

                desired_step = simulation.model.opt.timestep / realtime_factor
                remaining = desired_step - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    except (KeyboardInterrupt, SimulationSafetyError) as error:
        if isinstance(error, SimulationSafetyError):
            print(f"simulation safety failure: {error}", file=sys.stderr)
            return 1
    finally:
        transport.close()
    return 0


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return port


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo quadrotor simulator and localhost IPC server."
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=("hover", "takeoff_hover", "takeoff_land", "command_loss"),
        default=None,
    )
    parser.add_argument("--duration", type=positive_float, default=None)
    parser.add_argument("--target-height", type=float, default=1.2, metavar="METERS")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--command-port", type=port_number, default=DEFAULT_COMMAND_PORT
    )
    parser.add_argument(
        "--telemetry-port", type=port_number, default=DEFAULT_TELEMETRY_PORT
    )
    parser.add_argument("--realtime-factor", type=positive_float, default=1.0)
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH, metavar="XML"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"model file does not exist: {model_path}")
    if not MIN_TARGET_HEIGHT <= args.target_height <= MAX_TARGET_HEIGHT:
        parser.error(
            f"--target-height must be in [{MIN_TARGET_HEIGHT}, {MAX_TARGET_HEIGHT}] m"
        )
    try:
        if not ipaddress.ip_address(args.host).is_loopback:
            parser.error("--host must be a loopback address")
    except ValueError as error:
        parser.error(str(error))

    if args.scenario is not None:
        if not args.headless:
            parser.error("--scenario requires --headless")
        default_duration = 14.0 if args.scenario == "takeoff_land" else 8.0
        duration = args.duration if args.duration is not None else default_duration
        result = run_headless_scenario(
            args.scenario, duration, args.target_height, model_path
        )
        print_scenario_result(result)
        return 0 if result.passed else 1

    parameters = FlightParameters(takeoff_height=args.target_height)
    duration = args.duration if args.duration is not None else math.inf
    return run_ipc_simulator(
        model_path,
        parameters,
        args.host,
        args.command_port,
        args.telemetry_port,
        args.headless,
        duration,
        args.realtime_factor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
