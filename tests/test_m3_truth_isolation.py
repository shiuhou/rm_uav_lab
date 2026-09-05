"""M3 structural truth-isolation tests.

In M3 localization mode the ONLY navigation state available to the
flight controller, the guidance target integration, the flight action
handlers and the mission is the estimator output. Ground truth may
still drive physics-side safety checks, ground-contact mechanics and
benchmark scoring.

These tests inject estimator stubs that deliberately report FALSE
states and verify the whole stack follows the estimate, not the truth.
"""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from ipc_protocol import CommandPacket, PROTOCOL_VERSION
from m2_mission import MissionConfig, MissionPhase, MissionStateMachine
from m3_estimator import EstimatedState
from m3_sensors import TruthKinematics
from simulator import FlightParameters, QuadrotorSimulation
from m2_benchmark import M0_MODEL


def estimated_state(
    time: float,
    position=(0.0, 0.0, 0.12),
    velocity=(0.0, 0.0, 0.0),
    yaw: float = 0.0,
    healthy: bool = True,
) -> EstimatedState:
    quat = np.array(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=float
    )
    flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(flat, quat)
    return EstimatedState(
        time=time,
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        quaternion_wxyz=quat,
        rotation_body_to_world=flat.reshape(3, 3),
        angular_velocity_body=np.zeros(3, dtype=float),
        healthy=healthy,
        imu_age=0.0,
        flow_age=0.0,
        tof_age=0.0,
    )


class FixedStubPipeline:
    """Reports one fixed false state regardless of physics truth."""

    def __init__(self, state: EstimatedState) -> None:
        self._state = state

    def update(self, kin: TruthKinematics) -> EstimatedState:
        return self._state


class ScaledXStubPipeline:
    """Estimates x as scale * truth x (a routing test double).

    Reading truth inside a stub is allowed -- its whole purpose is to
    prove WHERE the estimate goes, by making estimate and truth diverge
    in a controlled way.
    """

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def update(self, kin: TruthKinematics) -> EstimatedState:
        position = kin.position.copy()
        position[0] *= self.scale
        velocity = kin.velocity_world.copy()
        velocity[0] *= self.scale
        # Attitude and angular rate pass through truthfully: this stub
        # isolates POSITION routing. (An always-level attitude estimate
        # removes all attitude feedback and the vehicle flips -- verified.)
        return EstimatedState(
            time=kin.time,
            position=position,
            velocity=velocity,
            quaternion_wxyz=kin.quaternion_wxyz.copy(),
            rotation_body_to_world=kin.rotation_body_to_world.copy(),
            angular_velocity_body=kin.angular_velocity_body.copy(),
            healthy=True,
            imu_age=0.0,
            flow_age=0.0,
            tof_age=0.0,
        )


def make_sim(localization, **kwargs) -> QuadrotorSimulation:
    # 與 benchmark 一致：mission 的 takeoff_altitude=1.5 必須搭配
    # simulator 的 takeoff_height=1.5，否則載具只爬到預設的 1.2 m。
    kwargs.setdefault("parameters", FlightParameters(takeoff_height=1.5))
    return QuadrotorSimulation(
        M0_MODEL, localization=localization, **kwargs
    )


class RoutingTest(unittest.TestCase):
    def test_last_measured_state_is_estimator_output_not_truth(self) -> None:
        stub = FixedStubPipeline(
            estimated_state(0.0, position=(10.0, -3.0, 2.0))
        )
        sim = make_sim(stub)
        sim.step(sim.data.time)
        truth = sim.controller.read_state(sim.data)
        np.testing.assert_allclose(
            sim.last_measured_state.position, (10.0, -3.0, 2.0), atol=1e-12
        )
        self.assertLess(float(truth.position[0]), 1.0)

    def test_controller_follows_estimator_velocity_not_truth(self) -> None:
        # Stub claims the vehicle is climbing at 1 m/s at 1.5 m altitude.
        # Truth: sitting on the ground. The position controller must
        # react to the ESTIMATE and command less than hover thrust,
        # while the physical vehicle stays low.
        stub = FixedStubPipeline(
            estimated_state(
                0.0, position=(0.0, 0.0, 1.5), velocity=(0.0, 0.0, 1.0)
            )
        )
        sim = make_sim(stub)
        sim.accept_command(
            CommandPacket(
                PROTOCOL_VERSION, 0, sim.data.time, "takeoff",
                0.0, 0.0, 0.0, 0.0,
            ),
            sim.data.time,
        )
        truth = None
        for _ in range(1000):  # 2 s
            truth = sim.step(sim.data.time)
        total_thrust = float(np.sum(sim.last_motor_thrusts))
        hover_thrust = 1.0 * 9.81
        self.assertLess(total_thrust, 0.85 * hover_thrust)
        self.assertLess(float(truth.position[2]), 0.5)

    def test_action_handlers_anchor_on_estimated_state(self) -> None:
        # Takeoff target position/yaw must come from the estimate.
        # With stub yaw = 0 while truth yaw = 90 deg, the flight machine
        # target yaw must be 0 -- otherwise the SO(3) controller would
        # spin the physical frame chasing a truth-anchored target.
        stub = FixedStubPipeline(
            estimated_state(0.0, position=(0.0, 0.0, 0.12), yaw=0.0)
        )
        sim = make_sim(stub, initial_yaw=math.radians(90.0))
        sim.accept_command(
            CommandPacket(
                PROTOCOL_VERSION, 0, sim.data.time, "takeoff",
                0.0, 0.0, 0.0, 0.0,
            ),
            sim.data.time,
        )
        self.assertAlmostEqual(sim.machine.target_yaw, 0.0, places=12)
        self.assertAlmostEqual(sim.machine.target_position[0], 0.0, places=12)

    def test_no_truth_fallback_when_estimator_unhealthy(self) -> None:
        stub = FixedStubPipeline(
            estimated_state(0.0, position=(7.0, 7.0, 1.5), healthy=False)
        )
        sim = make_sim(stub)
        sim.step(sim.data.time)
        self.assertFalse(sim.last_estimated_state.healthy)
        # Still the (unhealthy) estimate -- never a silent truth fallback.
        np.testing.assert_allclose(
            sim.last_measured_state.position, (7.0, 7.0, 1.5), atol=1e-12
        )

    def test_localization_and_noise_model_are_mutually_exclusive(self) -> None:
        from m1_realism import NoiseConfig

        stub = FixedStubPipeline(estimated_state(0.0))
        with self.assertRaises(ValueError):
            QuadrotorSimulation(
                M0_MODEL,
                FlightParameters(),
                noise_config=NoiseConfig(),
                localization=stub,
            )


class MissionFollowsEstimatorTest(unittest.TestCase):
    """Full-mission contradictory-state proof through the real routing.

    ScaledXStubPipeline makes the estimate diverge from truth in a
    controlled way; the mission must follow the ESTIMATE while the truth
    scorer records what physically happened.
    """

    @staticmethod
    def _drive(sim, mission, stop_check, max_time: float = 30.0):
        sequence = 0

        def send_command(command) -> None:
            nonlocal sequence
            sim.accept_command(
                CommandPacket(
                    PROTOCOL_VERSION, sequence, sim.data.time,
                    command.action, command.forward, command.right,
                    command.up, command.yaw,
                ),
                sim.data.time,
            )
            sequence += 1

        send_command(mission.start(sim.last_measured_state, sim.data.time))
        result = None
        while sim.data.time < max_time:
            truth = sim.step(sim.data.time)
            command = mission.update(
                sim.last_measured_state,
                sim.machine.state.value,
                sim.data.time,
            )
            result = stop_check(mission, truth)
            if result is not None:
                break
            send_command(command)
            if mission.phase in (MissionPhase.DONE, MissionPhase.FAILED):
                break
        return result

    def test_mission_brakes_when_estimator_says_arrived(self) -> None:
        # Estimate runs 6.4% long: est 5.00 m == truth ~4.70 m.
        sim = make_sim(ScaledXStubPipeline(1.064))
        mission = MissionStateMachine(MissionConfig())

        def stop_check(m, truth):
            if m.phase is MissionPhase.BRAKE:
                return truth
            return None

        truth = self._drive(sim, mission, stop_check)
        self.assertIsNotNone(truth, "mission never entered BRAKE")
        est_progress = float(sim.last_measured_state.position[0])
        truth_progress = float(truth.position[0])
        self.assertGreater(est_progress, 4.9)
        self.assertLess(truth_progress, 4.85)

    def test_mission_stays_forward_when_estimator_short(self) -> None:
        # Estimate runs 6% short: at truth 5.1 m the estimate is only
        # ~4.79 m, so the mission must STILL be in FORWARD.
        sim = make_sim(ScaledXStubPipeline(0.94))
        mission = MissionStateMachine(MissionConfig())

        def stop_check(m, truth):
            if m.phase is MissionPhase.FORWARD and truth.position[0] > 5.10:
                return truth
            return None

        truth = self._drive(sim, mission, stop_check)
        self.assertIsNotNone(
            truth, "mission left FORWARD before estimator reached 5 m"
        )


if __name__ == "__main__":
    unittest.main()
