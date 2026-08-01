from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from power_battery_model import BatteryModel, fit_battery_model, soc_delta_percent


class BatteryModelTests(unittest.TestCase):
    def test_charge_and_discharge_efficiencies_are_directional(self) -> None:
        model = BatteryModel(
            usable_capacity_kwh=26.0,
            charge_efficiency=0.90,
            discharge_efficiency=0.80,
            max_charge_w=3_000.0,
            max_discharge_w=3_000.0,
        )

        charging = float(soc_delta_percent(1_000.0, 1.0, model))
        discharging = float(soc_delta_percent(-1_000.0, 1.0, model))

        self.assertAlmostEqual(charging, 100.0 * 0.9 / 26.0)
        self.assertAlmostEqual(discharging, -100.0 / (0.8 * 26.0))
        self.assertGreater(abs(discharging), charging)

    def test_power_limits_bound_energy_transfer(self) -> None:
        model = BatteryModel(
            usable_capacity_kwh=20.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            max_charge_w=500.0,
            max_discharge_w=600.0,
        )

        self.assertAlmostEqual(float(soc_delta_percent(2_000.0, 1.0, model)), 2.5)
        self.assertAlmostEqual(float(soc_delta_percent(-2_000.0, 1.0, model)), -3.0)

    def test_fit_rejects_saturated_and_transition_intervals(self) -> None:
        times = pd.date_range("2026-07-01", periods=12, freq="30min")
        frame = pd.DataFrame(
            {
                "BatterySOC": [100.0] * 6 + list(np.linspace(80.0, 75.0, 6)),
                "BatteryWatts": [-500.0] * 12,
                "ObservedLoadWatts": [200.0] * 9 + [900.0, 200.0, 200.0],
            },
            index=times,
        )

        model = fit_battery_model(frame)

        self.assertLess(model.calibration_sample_count, len(frame))
        self.assertGreater(model.usable_capacity_kwh, 0.0)
        self.assertIn(model.calibration_confidence, {"default", "provisional", "calibrated"})


if __name__ == "__main__":
    unittest.main()
