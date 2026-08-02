from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from power_load_contract import ControlledLoadEstimate, validate_state_held_load
from power_load_dynamics import (
    PHASE_CODES,
    PHASE_FAN_HIGH,
    PHASE_FAN_LOW,
    PHASE_STARTUP,
    build_controlled_load_profile,
    controlled_load_member_profiles,
    learn_state_load_dynamics,
)


class StateLoadDynamicsTests(unittest.TestCase):
    def _observations(self) -> pd.DataFrame:
        times = pd.date_range("2026-07-20", periods=88, freq="15min")
        modes = np.asarray(["dc_only"] * 8 + ["dc_cl61"] * 40 + ["dc_only"] * 8 + ["dc_cl61"] * 32)
        loads = np.asarray(
            [230.0] * 8
            + [470.0] * 8
            + [275.0] * 16
            + [455.0] * 8
            + [275.0] * 8
            + [230.0] * 8
            + [465.0] * 8
            + [278.0] * 12
            + [452.0] * 8
            + [278.0] * 4,
            dtype=np.float64,
        )
        return pd.DataFrame({"direct_mode": modes, "load_w": loads}, index=times)

    def test_learns_startup_and_fan_phases_inside_exact_state(self) -> None:
        dynamics = learn_state_load_dynamics(self._observations(), "dc_cl61")

        self.assertIsNotNone(dynamics)
        assert dynamics is not None
        self.assertIn(PHASE_STARTUP, dynamics.phase_profiles)
        self.assertIn(PHASE_FAN_LOW, dynamics.phase_profiles)
        self.assertIn(PHASE_FAN_HIGH, dynamics.phase_profiles)
        self.assertAlmostEqual(dynamics.startup_duration_p50_minutes, 120.0, delta=15.0)
        self.assertLess(dynamics.phase_profiles[PHASE_FAN_LOW].p50_w, 300.0)
        self.assertGreater(dynamics.phase_profiles[PHASE_FAN_HIGH].p50_w, 400.0)
        self.assertGreater(dynamics.change_count, 0)

    def test_profile_transitions_from_startup_without_changing_state(self) -> None:
        observations = self._observations().iloc[:58]
        observations.iloc[-2:, observations.columns.get_loc("direct_mode")] = "dc_cl61"
        observations.iloc[-2:, observations.columns.get_loc("load_w")] = 468.0
        dynamics = learn_state_load_dynamics(observations, "dc_cl61")
        assert dynamics is not None
        dynamics = type(dynamics)(
            **{
                **dynamics.__dict__,
                "current_phase": PHASE_STARTUP,
                "state_started_at": observations.index[-2].isoformat(),
                "phase_started_at": observations.index[-2].isoformat(),
            }
        )
        times = pd.date_range(observations.index[-1], periods=8, freq="30min")
        fallback = ControlledLoadEstimate(260.0, 275.0, 470.0, "fixture", 20)

        profile = build_controlled_load_profile(dynamics, times, fallback)

        self.assertEqual(profile.phase_codes[0], PHASE_CODES[PHASE_STARTUP])
        self.assertNotEqual(profile.phase_codes[-1], PHASE_CODES[PHASE_STARTUP])
        validate_state_held_load(
            np.ones(len(times), dtype=np.int16),
            profile.p50_w,
            phase_codes=profile.phase_codes,
        )

    def test_members_only_change_at_learned_phase_boundaries(self) -> None:
        dynamics = learn_state_load_dynamics(self._observations(), "dc_cl61")
        assert dynamics is not None
        times = pd.date_range(self._observations().index[-1], periods=24, freq="1h")
        fallback = ControlledLoadEstimate(260.0, 275.0, 470.0, "fixture", 20)

        loads, phases = controlled_load_member_profiles(dynamics, times, fallback, 20, seed=7)

        validate_state_held_load(
            np.ones(len(times), dtype=np.int16),
            loads,
            phase_codes=phases,
        )
        self.assertGreater(np.count_nonzero(np.diff(phases, axis=1)), 0)


if __name__ == "__main__":
    unittest.main()
