"""M2 mission benchmark matrix as a pytest milestone gate."""

from __future__ import annotations

import unittest

from m2_benchmark import M2_CASES, run_m2_benchmark, run_m2_case


class M2BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reports = {report.name: report for report in run_m2_benchmark()}

    def test_all_cases_run(self) -> None:
        self.assertEqual(len(self.reports), len(M2_CASES))
        self.assertGreaterEqual(len(M2_CASES), 11)

    def test_all_cases_pass_m2_gates(self) -> None:
        for name, report in self.reports.items():
            self.assertTrue(
                report.passed,
                msg=f"{name}: " + "; ".join(report.reasons),
            )

    def test_every_case_completes_with_done(self) -> None:
        for name, report in self.reports.items():
            self.assertEqual(report.mission_result, "DONE", msg=name)
            self.assertIsNone(report.failure_reason, msg=name)
            self.assertFalse(report.has_nan_or_inf, msg=name)
            self.assertTrue(report.landing_success, msg=name)

    def test_truth_forward_distance_gate(self) -> None:
        # Ground-truth scoring gate: 4.7 m <= distance <= 5.3 m.
        for name, report in self.reports.items():
            self.assertGreaterEqual(
                report.truth_forward_at_land_entry, 4.7, msg=name
            )
            self.assertLessEqual(
                report.truth_forward_at_land_entry, 5.3, msg=name
            )
            self.assertLess(abs(report.truth_lateral_at_land_entry), 0.20, msg=name)

    def test_yaw90_case_moves_along_world_y_not_x(self) -> None:
        report = self.reports["case10_yaw90_nominal"]
        delta = tuple(
            f - s
            for f, s in zip(
                report.truth_final_position_xy, report.truth_start_position_xy
            )
        )
        # Facing 90 deg, "forward 5 m" is world +Y; world X stays near zero.
        self.assertGreater(float(delta[1]), 4.5)
        self.assertLess(abs(float(delta[0])), 0.5)

    def test_combined_case_is_deterministic(self) -> None:
        case = [c for c in M2_CASES if c.name == "case9_combined"][0]
        first = run_m2_case(case)
        second = run_m2_case(case)
        self.assertEqual(first.seed, 27)
        self.assertAlmostEqual(
            first.truth_forward_at_land_entry,
            second.truth_forward_at_land_entry,
            places=12,
        )
        self.assertAlmostEqual(
            first.measured_forward_at_brake_entry,
            second.measured_forward_at_brake_entry,
            places=12,
        )

    def test_measured_and_truth_progress_are_reported_separately(self) -> None:
        # The benchmark must not collapse "what the drone thinks" into
        # "what actually happened".
        report = self.reports["case9_combined"]
        self.assertIsNotNone(report.measured_forward_at_brake_entry)
        self.assertIsNotNone(report.truth_forward_at_brake_entry)
        self.assertNotEqual(
            report.measured_forward_at_brake_entry,
            report.truth_forward_at_brake_entry,
        )


if __name__ == "__main__":
    unittest.main()
