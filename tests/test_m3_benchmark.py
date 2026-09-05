"""M3 localization benchmark as a pytest milestone gate.

Beyond the pass gates, these tests pin the SIGNATURES that prove the
estimator is really in the loop: flow scale must push the estimate long,
gyro bias must create yaw/lateral error, and estimate must never be
bit-identical to truth (that would mean a truth leak).
"""

from __future__ import annotations

import unittest

from m3_benchmark import (
    M3_CASES,
    M3Case,
    run_m3_benchmark,
    run_m3_case,
)
from m3_sensors import FlowConfig, IMUConfig, ToFConfig
from m2_benchmark import M0_MODEL


class LongOutageTest(unittest.TestCase):
    def test_long_flow_outage_fails_loudly_without_truth_fallback(self) -> None:
        # Flow lost for 2.0 s (> 0.5 s stale limit) during FORWARD.
        # The estimator must become unhealthy and the benchmark must
        # abort as ESTIMATOR_UNHEALTHY -- never silently read truth.
        case = M3Case(
            "test_long_flow_outage",
            "flow lost for 2.0 s mid-forward",
            M0_MODEL,
            imu=IMUConfig(),
            flow=FlowConfig(dropouts=((7.0, 2.0),)),
            tof=ToFConfig(),
            seed=11,
        )
        report = run_m3_case(case)
        self.assertNotEqual(report.mission_result, "DONE")
        self.assertTrue(report.estimator_unhealthy_abort)
        self.assertIn("ESTIMATOR_UNHEALTHY", report.failure_reason)


class M3BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reports = {report.name: report for report in run_m3_benchmark()}

    def test_all_cases_run(self) -> None:
        self.assertEqual(len(self.reports), len(M3_CASES))
        self.assertGreaterEqual(len(M3_CASES), 11)

    def test_all_cases_pass_m3_gates(self) -> None:
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
            self.assertFalse(report.estimator_unhealthy_abort, msg=name)

    def test_flow_scale_makes_estimate_run_long(self) -> None:
        # +2% flow scale: the estimator believes it is moving 2% faster,
        # so it declares arrival early and the TRUTH distance falls
        # short of the ESTIMATED distance. Sign and rough magnitude are
        # the physics of scale error, not a tuning artifact.
        report = self.reports["case2_flow_scale"]
        self.assertGreater(
            report.est_forward_at_land - report.truth_forward_at_land, 0.04
        )
        self.assertLess(
            report.est_forward_at_land - report.truth_forward_at_land, 0.20
        )

    def test_flow_bias_creates_position_drift(self) -> None:
        report = self.reports["case3_flow_bias"]
        self.assertGreater(report.horizontal_position_rms_error, 0.05)

    def test_gyro_z_bias_creates_yaw_and_lateral_error(self) -> None:
        report = self.reports["case6_gyro_z_bias"]
        self.assertGreater(report.max_yaw_error_deg, 0.5)
        self.assertGreater(abs(report.truth_lateral_at_land), 0.01)

    def test_short_dropouts_are_bridged_and_counted(self) -> None:
        flow_case = self.reports["case4_flow_dropout"]
        tof_case = self.reports["case5_tof_dropout"]
        self.assertGreater(flow_case.flow_invalid, 5)
        self.assertGreater(tof_case.tof_invalid, 5)
        # The 0.25 s dropouts themselves never flip health; the only
        # unhealthy transition allowed is the EXPECTED landing-phase one
        # (ToF is blind below its 0.05 m minimum range after touchdown,
        # which is outside the health-gated phases by design).
        self.assertFalse(flow_case.estimator_unhealthy_abort)
        self.assertFalse(tof_case.estimator_unhealthy_abort)

    def test_estimate_is_not_bit_identical_to_truth(self) -> None:
        # Any exact copy would mean the estimator is bypassed or truth
        # leaks into the navigation state.
        for name, report in self.reports.items():
            if "ideal" in name:
                continue
            self.assertGreater(
                report.horizontal_position_rms_error, 1e-6, msg=name
            )

    def test_yaw90_case_moves_along_world_y_not_x(self) -> None:
        report = self.reports["case9_yaw90_ideal"]
        delta = tuple(
            f - s
            for f, s in zip(
                report.truth_final_position_xy, report.truth_start_position_xy
            )
        )
        self.assertGreater(float(delta[1]), 4.5)
        self.assertLess(abs(float(delta[0])), 0.5)

    def test_combined_case_is_deterministic(self) -> None:
        case = [c for c in M3_CASES if c.name == "case8_combined"][0]
        first = run_m3_case(case)
        second = run_m3_case(case)
        self.assertEqual(first.seed, 27)
        self.assertAlmostEqual(
            first.truth_forward_at_land,
            second.truth_forward_at_land,
            places=12,
        )
        self.assertAlmostEqual(
            first.est_forward_at_land,
            second.est_forward_at_land,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
