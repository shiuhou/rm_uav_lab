"""M2 Task 9 (math part): the mission frame is heading-relative, not world-X.

The mission frame is built from the measured yaw at mission start:

    h = [cos yaw0, sin yaw0]   (initial heading, "forward")
    l = [-sin yaw0, cos yaw0]  (leftward normal)

    forward_progress = dot(p_xy - p0_xy, h)
    lateral_error    = dot(p_xy - p0_xy, l)

"Forward 5 m" must mean 5 m along h for ANY initial yaw.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from m2_mission import MissionFrame


class MissionFrameTest(unittest.TestCase):
    def test_yaw_zero_forward_is_world_x(self) -> None:
        frame = MissionFrame(origin_xy=np.zeros(2), yaw=0.0)
        progress, lateral = frame.project(np.array([5.0, 0.3]))
        self.assertAlmostEqual(progress, 5.0)
        self.assertAlmostEqual(lateral, 0.3)

    def test_yaw_90_forward_is_world_y(self) -> None:
        frame = MissionFrame(origin_xy=np.zeros(2), yaw=math.radians(90.0))
        # Moving along world +Y is "forward" when facing 90 deg.
        progress, lateral = frame.project(np.array([0.2, 5.0]))
        self.assertAlmostEqual(progress, 5.0)
        # l = [-sin 90, cos 90] = (-1, 0): world +X is to the RIGHT of the
        # heading, so lateral error is negative.
        self.assertAlmostEqual(lateral, -0.2, places=12)
        # World +X motion is now purely lateral.
        progress, lateral = frame.project(np.array([1.0, 0.0]))
        self.assertAlmostEqual(progress, 0.0, places=12)
        self.assertAlmostEqual(lateral, -1.0, places=12)

    def test_yaw_minus45(self) -> None:
        yaw = math.radians(-45.0)
        frame = MissionFrame(origin_xy=np.array([1.0, 2.0]), yaw=yaw)
        h = np.array([math.cos(yaw), math.sin(yaw)])
        point = np.array([1.0, 2.0]) + 5.0 * h
        progress, lateral = frame.project(point)
        self.assertAlmostEqual(progress, 5.0)
        self.assertAlmostEqual(lateral, 0.0, places=12)

    def test_origin_offset_is_subtracted(self) -> None:
        frame = MissionFrame(origin_xy=np.array([0.5, -0.25]), yaw=0.0)
        progress, lateral = frame.project(np.array([5.5, -0.25]))
        self.assertAlmostEqual(progress, 5.0)
        self.assertAlmostEqual(lateral, 0.0)


if __name__ == "__main__":
    unittest.main()
