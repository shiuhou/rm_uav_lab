"""M1 Task 6: robustness benchmark matrix as a pytest milestone gate.

Runs the full M1 case matrix (nominal reference plus nine non-ideal cases)
through the production path (action -> state machine -> controller -> mixer
-> actuator -> MuJoCo -> measurement -> controller) and asserts every M1
gate for every case.
"""

from __future__ import annotations

import unittest

from m1_benchmark import M1_CASES, run_m1_benchmark, run_m1_case


class M1BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reports = run_m1_benchmark()

    def test_all_ten_cases_run(self) -> None:
        self.assertEqual(len(self.reports), 10)
        self.assertEqual(len(M1_CASES), 10)

    def test_all_cases_pass_m1_gates(self) -> None:
        for report in self.reports:
            self.assertTrue(
                report.passed,
                msg=f"{report.name}: " + "; ".join(report.reasons),
            )

    def test_every_case_reaches_and_holds_hover(self) -> None:
        for report in self.reports:
            self.assertTrue(report.reached_hovering, msg=report.name)
            self.assertGreaterEqual(report.hover_duration_measured, 5.0)
            self.assertFalse(report.has_nan_or_inf, msg=report.name)
            self.assertFalse(report.crashed, msg=report.name)

    def test_nominal_reference_matches_m0_quality(self) -> None:
        nominal = self.reports[0]
        self.assertEqual(nominal.name, "case0_nominal_m0")
        self.assertLess(nominal.final_altitude_error, 0.02)
        self.assertLess(nominal.hover_rms_error, 0.02)

    def test_combined_case_is_deterministic_with_seed_27(self) -> None:
        combined = [case for case in M1_CASES if case.name == "case9_combined"]
        self.assertEqual(len(combined), 1)
        first = run_m1_case(combined[0])
        second = run_m1_case(combined[0])
        self.assertEqual(first.seed, 27)
        self.assertAlmostEqual(
            first.final_altitude_error, second.final_altitude_error, places=12
        )
        self.assertAlmostEqual(
            first.hover_rms_error, second.hover_rms_error, places=12
        )
        self.assertAlmostEqual(
            first.max_horizontal_drift, second.max_horizontal_drift, places=12
        )


if __name__ == "__main__":
    unittest.main()
