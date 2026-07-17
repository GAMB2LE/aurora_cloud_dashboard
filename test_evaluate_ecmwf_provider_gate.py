from __future__ import annotations

import unittest

from evaluate_ecmwf_provider_gate import evaluate


class EarthkitPromotionGateTests(unittest.TestCase):
    def test_short_shadow_window_is_not_eligible(self) -> None:
        result = evaluate(
            [
                {"opened_at_utc": "2026-07-17T00:00:00+00:00", "shadow_status": "compared", "valid_times_match": True, "ssrd_max_abs_difference_j_m2": 0.0},
                {"opened_at_utc": "2026-07-17T03:00:00+00:00", "shadow_status": "compared", "valid_times_match": True, "ssrd_max_abs_difference_j_m2": 0.0},
            ],
            minimum_days=7,
            minimum_samples=2,
            maximum_ssrd_difference=0.001,
        )

        self.assertEqual(result["decision"], "not_eligible")
        self.assertIn("minimum_observation_window_not_reached", result["reasons"])
        self.assertFalse(result["automatic_provider_switch"])

    def test_clean_seven_day_shadow_window_is_ready_for_review(self) -> None:
        result = evaluate(
            [
                {"opened_at_utc": "2026-07-10T00:00:00+00:00", "shadow_status": "compared", "valid_times_match": True, "ssrd_max_abs_difference_j_m2": 0.0},
                {"opened_at_utc": "2026-07-17T00:00:00+00:00", "shadow_status": "compared", "valid_times_match": True, "ssrd_max_abs_difference_j_m2": 0.0},
            ],
            minimum_days=7,
            minimum_samples=2,
            maximum_ssrd_difference=0.001,
        )

        self.assertEqual(result["decision"], "eligible_for_operator_review")
        self.assertEqual(result["reasons"], [])


if __name__ == "__main__":
    unittest.main()
