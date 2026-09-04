"""M2 mission state machine tests (no MuJoCo required).

The mission state machine consumes ONLY a measured QuadrotorState plus the
discrete flight-mode string. Ground truth never enters its API, which is
what makes the measured-vs-truth separation structural rather than a
convention.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from fly import QuadrotorState, ROTOR_NAMES, load_simulation
from m1_realism import (
    M1_MOTOR_EFFECTIVENESS,
    PlantConfig,
    apply_plant_config,
)
from m2_mission import (
    MissionCommand,
    MissionConfig,
    MissionPhase,
    MissionStateMachine,
)


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "quadrotor.xml"


def make_state(position, velocity=(0.0, 0.0, 0.0), yaw=0.0) -> QuadrotorState:
    """Synthetic measured state; mission tests never need MuJoCo."""

    half = 0.5 * yaw
    quaternion = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return QuadrotorState(
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        quaternion,
        rotation,
        np.zeros(3),
    )


class MotorMismatchMappingTest(unittest.TestCase):
    """Audit regression: the M1 mismatch vector by actuator NAME is
    FL=0.97, FR=1.00, RR=0.99, RL=1.02. Any reordering must fail here."""

    def test_effectiveness_maps_to_actuator_names(self) -> None:
        model, data, _ = load_simulation(MODEL_PATH)
        apply_plant_config(
            model, data, PlantConfig(motor_effectiveness=M1_MOTOR_EFFECTIVENESS)
        )
        expected = {
            "rotor_fl": 0.97,
            "rotor_fr": 1.00,
            "rotor_rr": 0.99,
            "rotor_rl": 1.02,
        }
        self.assertEqual(tuple(expected), ROTOR_NAMES)
        for name, value in expected.items():
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            self.assertAlmostEqual(
                float(model.actuator_gainprm[actuator_id, 0]), value, places=12,
                msg=f"{name} effectiveness",
            )


class MissionTransitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MissionConfig()
        self.mission = MissionStateMachine(self.config)
        command = self.mission.start(make_state((0.0, 0.0, 0.12)), 0.0)
        self.assertEqual(command.action, "takeoff")
        self.assertEqual(self.mission.phase, MissionPhase.TAKEOFF)

    def _settle_and_enter_forward(self) -> None:
        # TAKEOFF -> SETTLE
        command = self.mission.update(
            make_state((0.0, 0.0, 1.52)), "HOVERING", 3.5
        )
        self.assertEqual(self.mission.phase, MissionPhase.SETTLE)
        # SETTLE -> FORWARD after 1.0 s continuous stability
        self.mission.update(make_state((0.0, 0.0, 1.50)), "HOVERING", 3.6)
        command = self.mission.update(
            make_state((0.0, 0.0, 1.50)), "HOVERING", 4.6
        )
        self.assertEqual(self.mission.phase, MissionPhase.FORWARD)
        self.assertGreater(command.forward, 0.0)

    def test_happy_path_ready_to_done(self) -> None:
        self._settle_and_enter_forward()
        # FORWARD -> BRAKE once measured remaining <= arrival threshold
        command = self.mission.update(
            make_state((4.99, 0.0, 1.50)), "MANUAL", 10.0
        )
        self.assertEqual(self.mission.phase, MissionPhase.BRAKE)
        self.assertEqual(command.forward, 0.0)
        # BRAKE -> HOVER after 0.4 s below speed threshold
        self.mission.update(
            make_state((5.02, 0.0, 1.5), (0.1, 0.0, 0.0)), "MANUAL", 10.2
        )
        self.assertEqual(self.mission.phase, MissionPhase.BRAKE)
        command = self.mission.update(
            make_state((5.03, 0.0, 1.5), (0.05, 0.0, 0.0)), "HOVERING", 10.7
        )
        self.assertEqual(self.mission.phase, MissionPhase.HOVER)
        # HOVER -> LAND after 1.0 s
        command = self.mission.update(
            make_state((5.03, 0.0, 1.5)), "HOVERING", 11.8
        )
        self.assertEqual(self.mission.phase, MissionPhase.LAND)
        self.assertEqual(command.action, "land")
        # LAND -> DONE only on the existing disarm semantics
        command = self.mission.update(
            make_state((5.03, 0.0, 0.12)), "LANDING", 13.0
        )
        self.assertEqual(self.mission.phase, MissionPhase.LAND)
        command = self.mission.update(
            make_state((5.03, 0.0, 0.12)), "DISARMED", 14.0
        )
        self.assertEqual(self.mission.phase, MissionPhase.DONE)

    def test_takeoff_needs_low_vertical_speed_not_just_altitude(self) -> None:
        command = self.mission.update(
            make_state((0.0, 0.0, 1.52), (0.0, 0.0, 0.5)), "TAKING_OFF", 2.0
        )
        self.assertEqual(self.mission.phase, MissionPhase.TAKEOFF)
        self.assertIsNone(command.action)

    def test_settle_timer_requires_continuous_stability(self) -> None:
        self.mission.update(make_state((0.0, 0.0, 1.52)), "HOVERING", 3.5)
        self.assertEqual(self.mission.phase, MissionPhase.SETTLE)
        # 0.8 s stable, then a disturbance resets the timer.
        self.mission.update(make_state((0.0, 0.0, 1.50)), "HOVERING", 4.3)
        self.mission.update(
            make_state((0.1, 0.0, 1.50), (0.4, 0.0, 0.0)), "HOVERING", 4.4
        )
        command = self.mission.update(
            make_state((0.1, 0.0, 1.50)), "HOVERING", 4.55
        )
        self.assertEqual(self.mission.phase, MissionPhase.SETTLE)
        # The timer restarts at the first stable sample (4.55), so FORWARD
        # requires 1.0 s of continuity after that: 4.55 -> 5.55.
        self.mission.update(make_state((0.1, 0.0, 1.50)), "HOVERING", 5.54)
        self.assertEqual(self.mission.phase, MissionPhase.SETTLE)
        self.mission.update(make_state((0.1, 0.0, 1.50)), "HOVERING", 5.56)
        self.assertEqual(self.mission.phase, MissionPhase.FORWARD)

    def test_forward_velocity_profile_slows_near_target(self) -> None:
        self._settle_and_enter_forward()
        far = self.mission.update(make_state((1.0, 0.0, 1.5)), "MANUAL", 6.0)
        self.assertAlmostEqual(far.forward, 0.8 / self.config.velocity_command_limit)
        near = self.mission.update(make_state((4.5, 0.0, 1.5)), "MANUAL", 8.0)
        self.assertAlmostEqual(
            near.forward, 0.2 / self.config.velocity_command_limit
        )
        closer = self.mission.update(make_state((4.95, 0.0, 1.5)), "MANUAL", 9.0)
        self.assertAlmostEqual(
            closer.forward, 0.05 / self.config.velocity_command_limit
        )

    def test_brake_requires_hold_time(self) -> None:
        self._settle_and_enter_forward()
        self.mission.update(make_state((5.0, 0.0, 1.5)), "MANUAL", 10.0)
        self.assertEqual(self.mission.phase, MissionPhase.BRAKE)
        # Speed dips below threshold briefly but does not hold 0.4 s.
        self.mission.update(
            make_state((5.01, 0.0, 1.5), (0.1, 0.0, 0.0)), "MANUAL", 10.2
        )
        self.mission.update(
            make_state((5.02, 0.0, 1.5), (0.35, 0.0, 0.0)), "MANUAL", 10.3
        )
        self.assertEqual(self.mission.phase, MissionPhase.BRAKE)
        self.mission.update(
            make_state((5.02, 0.0, 1.5), (0.1, 0.0, 0.0)), "MANUAL", 10.4
        )
        command = self.mission.update(
            make_state((5.02, 0.0, 1.5), (0.1, 0.0, 0.0)), "MANUAL", 10.81
        )
        self.assertEqual(self.mission.phase, MissionPhase.HOVER)

    def test_takeoff_timeout_fails_with_specific_reason(self) -> None:
        mission = MissionStateMachine(self.config)
        mission.start(make_state((0.0, 0.0, 0.12)), 0.0)
        mission.update(make_state((0.0, 0.0, 0.12)), "DISARMED", 8.1)
        self.assertEqual(mission.phase, MissionPhase.FAILED)
        self.assertEqual(mission.failure_reason, "TAKEOFF_TIMEOUT")

    def test_forward_timeout(self) -> None:
        self._settle_and_enter_forward()
        # Never make progress; FORWARD must time out at 15 s.
        self.mission.update(make_state((0.0, 0.0, 1.5)), "MANUAL", 4.6 + 15.1)
        self.assertEqual(self.mission.phase, MissionPhase.FAILED)
        self.assertEqual(self.mission.failure_reason, "FORWARD_TIMEOUT")

    def test_brake_timeout(self) -> None:
        self._settle_and_enter_forward()
        self.mission.update(make_state((5.0, 0.0, 1.5)), "MANUAL", 10.0)
        self.assertEqual(self.mission.phase, MissionPhase.BRAKE)
        self.mission.update(
            make_state((5.0, 0.0, 1.5), (0.5, 0.0, 0.0)), "MANUAL", 14.1
        )
        self.assertEqual(self.mission.phase, MissionPhase.FAILED)
        self.assertEqual(self.mission.failure_reason, "BRAKE_TIMEOUT")

    def test_land_timeout(self) -> None:
        self._settle_and_enter_forward()
        self.mission.update(make_state((5.0, 0.0, 1.5)), "MANUAL", 10.0)
        self.mission.update(
            make_state((5.0, 0.0, 1.5), (0.05, 0.0, 0.0)), "HOVERING", 10.5
        )
        # 10.5 + 0.4 hold -> HOVER at 10.9; hover 1.0 s -> LAND at 11.9+.
        self.mission.update(make_state((5.0, 0.0, 1.5)), "HOVERING", 11.0)
        self.assertEqual(self.mission.phase, MissionPhase.HOVER)
        self.mission.update(make_state((5.0, 0.0, 1.5)), "HOVERING", 12.0)
        self.assertEqual(self.mission.phase, MissionPhase.LAND)
        self.mission.update(make_state((5.0, 0.0, 0.5)), "LANDING", 22.1)
        self.assertEqual(self.mission.phase, MissionPhase.FAILED)
        self.assertEqual(self.mission.failure_reason, "LAND_TIMEOUT")

    def test_failed_is_terminal(self) -> None:
        mission = MissionStateMachine(self.config)
        mission.start(make_state((0.0, 0.0, 0.12)), 0.0)
        mission.update(make_state((0.0, 0.0, 0.12)), "DISARMED", 9.0)
        self.assertEqual(mission.phase, MissionPhase.FAILED)
        command = mission.update(
            make_state((0.0, 0.0, 1.5)), "HOVERING", 10.0
        )
        self.assertEqual(mission.phase, MissionPhase.FAILED)
        self.assertIsNone(command.action)
        self.assertEqual(command.forward, 0.0)


class MeasuredVersusTruthTest(unittest.TestCase):
    """Task 2: mission transitions are estimator-driven by construction.

    The MissionStateMachine API only accepts a `measured` QuadrotorState;
    whatever these tests feed as `measured` IS what the mission believes,
    regardless of what a scorer might know independently.
    """

    def _mission_in_forward(self) -> MissionStateMachine:
        mission = MissionStateMachine(MissionConfig())
        mission.start(make_state((0.0, 0.0, 0.12)), 0.0)
        mission.update(make_state((0.0, 0.0, 1.52)), "HOVERING", 3.5)
        mission.update(make_state((0.0, 0.0, 1.50)), "HOVERING", 3.6)
        mission.update(make_state((0.0, 0.0, 1.50)), "HOVERING", 4.6)
        self.assertEqual(mission.phase, MissionPhase.FORWARD)
        return mission

    def test_measured_arrival_triggers_brake(self) -> None:
        # Scenario: measured progress = 5.00 m. (A scorer could know the
        # truth is 4.85 m; the mission cannot see that.)
        mission = self._mission_in_forward()
        command = mission.update(make_state((5.00, 0.0, 1.5)), "MANUAL", 10.0)
        self.assertEqual(mission.phase, MissionPhase.BRAKE)
        self.assertEqual(command.forward, 0.0)

    def test_measured_shortfall_keeps_forward(self) -> None:
        # Opposite scenario: measured progress = 4.80 m (truth could be
        # 5.10 m); the mission must NOT brake yet.
        mission = self._mission_in_forward()
        command = mission.update(make_state((4.80, 0.0, 1.5)), "MANUAL", 10.0)
        self.assertEqual(mission.phase, MissionPhase.FORWARD)
        self.assertGreater(command.forward, 0.0)


if __name__ == "__main__":
    unittest.main()
