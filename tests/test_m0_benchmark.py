"""M0 milestone test: takeoff to 1.5 m and a stable 5 s hover.

This runs the same code path as `python m0_benchmark.py` and asserts every
initial M0 acceptance criterion, so the milestone gate lives in pytest.
"""

from __future__ import annotations

import unittest

from m0_benchmark import (
    MAX_FINAL_ALTITUDE_ERROR,
    MAX_HORIZONTAL_DRIFT,
    MAX_HOVER_RMS_ERROR,
    MAX_ROLL_PITCH_DEG,
    run_m0_hover_benchmark,
)


class M0HoverBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_m0_hover_benchmark(target_height=1.5, hover_duration=5.0)

    def test_benchmark_passes_all_m0_criteria(self) -> None:
        self.assertTrue(self.report.passed, msg="; ".join(self.report.reasons))

    def test_individual_acceptance_criteria(self) -> None:
        report = self.report
        self.assertTrue(report.reached_hovering)
        self.assertTrue(report.stayed_hovering)
        self.assertFalse(report.crashed)
        self.assertFalse(report.has_nan_or_inf)
        self.assertGreaterEqual(report.hover_duration_measured, 5.0)
        self.assertLessEqual(report.final_altitude_error, MAX_FINAL_ALTITUDE_ERROR)
        self.assertLessEqual(report.hover_rms_error, MAX_HOVER_RMS_ERROR)
        self.assertLessEqual(report.max_horizontal_drift, MAX_HORIZONTAL_DRIFT)
        self.assertLessEqual(report.max_roll_deg, MAX_ROLL_PITCH_DEG)
        self.assertLessEqual(report.max_pitch_deg, MAX_ROLL_PITCH_DEG)


if __name__ == "__main__":
    unittest.main()
