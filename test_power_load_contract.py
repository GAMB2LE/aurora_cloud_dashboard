from __future__ import annotations

import unittest

import numpy as np

from power_load_contract import (
    CONTROLLED_LOAD_CONTRACT,
    STATE_HOLD_POLICY,
    ControlledLoadEstimate,
    controlled_load_member_levels,
    estimate_controlled_load,
    validate_state_held_load,
)


class ControlledLoadContractTests(unittest.TestCase):
    def test_validation_repairs_bounds_without_moving_the_central_state_load(self) -> None:
        estimate = ControlledLoadEstimate(
            p10_w=700.0,
            p50_w=500.0,
            p90_w=300.0,
            source="fixture",
            sample_count=1,
        ).validated()

        self.assertEqual(estimate.p10_w, 500.0)
        self.assertEqual(estimate.p50_w, 500.0)
        self.assertEqual(estimate.p90_w, 500.0)

    def test_fresh_components_define_current_state_without_cross_state_history(self) -> None:
        estimate = estimate_controlled_load(
            mode="DC-Only + Radar",
            measured_current_w=900.0,
            learned_observations_w=(500.0, 520.0, 540.0),
            learned_level_w=520.0,
            component_estimate_w=515.0,
            dc_only_estimate_w=220.0,
        )

        self.assertEqual(CONTROLLED_LOAD_CONTRACT, "finite_operating_state_v1")
        self.assertIn("explicit_schedule_transition", STATE_HOLD_POLICY)
        self.assertEqual(estimate.source, "fresh_pdu_components_for_current_state")
        self.assertAlmostEqual(estimate.p50_w, 515.0)
        self.assertLessEqual(estimate.p10_w, estimate.p50_w)
        self.assertGreaterEqual(estimate.p90_w, estimate.p50_w)

    def test_member_uncertainty_is_stationary_within_the_state(self) -> None:
        estimate = estimate_controlled_load(
            mode="dc_only",
            measured_current_w=250.0,
            learned_observations_w=(210.0, 220.0, 230.0, 240.0),
            learned_level_w=225.0,
            dc_only_estimate_w=225.0,
        )
        levels = controlled_load_member_levels(estimate, 20)
        loads = np.repeat(levels[:, None], 6, axis=1)

        validate_state_held_load(np.zeros(6, dtype=np.int16), loads)
        self.assertGreater(float(np.ptp(levels)), 0.0)
        np.testing.assert_allclose(np.diff(loads, axis=1), 0.0)

    def test_load_change_without_state_transition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "without an operating-state transition"):
            validate_state_held_load(
                np.asarray([0, 0, 0], dtype=np.int16),
                np.asarray([220.0, 220.0, 400.0]),
            )


if __name__ == "__main__":
    unittest.main()
