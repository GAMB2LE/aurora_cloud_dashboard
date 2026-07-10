from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import build_forecast_dataset, build_historical_load_forecast, solar_irradiance_from_ssrd
from grouped_timeseries import build_power_display_summary_dataset


class PowerSocForecastTests(unittest.TestCase):
    def test_ssrd_accumulation_converts_to_irradiance(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="3h")
        ds = xr.Dataset(
            {"ssrd": (("time",), [0.0, 3 * 3600 * 100.0, 3 * 3600 * 250.0, 3 * 3600 * 400.0])},
            coords={"time": times},
        )

        irradiance = solar_irradiance_from_ssrd(ds)

        self.assertEqual(list(irradiance.index), list(times[1:]))
        np.testing.assert_allclose(irradiance.to_numpy(), [100.0, 150.0, 150.0])

    def test_build_forecast_dataset_integrates_soc(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=12, capacity_kwh=26.0)

        self.assertIn("BatterySOCForecast", forecast)
        self.assertIn("ECMWFSolarIrradiance", forecast)
        self.assertEqual(pd.Timestamp(forecast["time"].values[0]), power_times[-1])
        self.assertAlmostEqual(float(forecast["BatterySOCForecast"].values[0]), 70.0)
        self.assertGreaterEqual(float(forecast["BatterySOCForecast"].min()), 0.0)
        self.assertLessEqual(float(forecast["BatterySOCForecast"].max()), 100.0)
        self.assertEqual(forecast.attrs["forecast_horizon_hours"], "12")
        self.assertEqual(forecast.attrs["load_model"], "historical_utc_hour_median")
        self.assertIn("ForecastLoadMAERecent", forecast)

    def test_historical_load_forecast_uses_utc_hour_profile(self) -> None:
        times = pd.date_range("2026-07-07T00:00:00", periods=72, freq="1h")
        daytime = np.where((times.hour >= 9) & (times.hour <= 17), 600.0, 120.0)
        frame = pd.DataFrame(
            {
                "ACOutputWatts": daytime,
                "DCInverterWatts": np.full(len(times), 30.0),
            },
            index=times,
        )
        forecast_times = pd.DatetimeIndex([pd.Timestamp("2026-07-10T10:00:00"), pd.Timestamp("2026-07-10T23:00:00")])

        load = build_historical_load_forecast(frame, forecast_times, end=times[-1], calibration_days=3)

        self.assertGreater(float(load.iloc[0]), float(load.iloc[1]))

    def test_load_bias_skill_corrects_next_forecast(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 200.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 50.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )
        previous = xr.Dataset(
            {"ForecastLoadWatts": (("time",), np.full(len(power_times), 500.0))},
            coords={"time": power_times},
        )

        forecast = build_forecast_dataset(power, solar, previous_forecast=previous, horizon_hours=12, capacity_kwh=26.0)

        self.assertLess(float(forecast.attrs["load_bias_correction_w"]), 0.0)
        self.assertLess(float(forecast["ForecastLoadWatts"].median()), 250.0)
        self.assertIn("ForecastLoadBiasRecent", forecast)

    def test_display_summary_merges_forecast_fields(self) -> None:
        power_times = pd.date_range("2026-07-10T00:00:00", periods=3, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.5, 69.0]),
                "ACOutputWatts": (("time",), [10.0, 10.0, 10.0]),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range("2026-07-10T03:00:00", periods=3, freq="3h")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [68.0, 67.0, 66.0]),
                "ECMWFSolarIrradiance": (("time",), [100.0, 200.0, 100.0]),
                "ForecastLoadMAERecent": (("time",), [10.0, 10.0, 10.0]),
            },
            coords={"time": forecast_times},
        )

        summary = build_power_display_summary_dataset(power, forecast_ds=forecast, freq="1h")

        self.assertIn("BatterySOCForecast", summary)
        self.assertIn("ECMWFSolarIrradiance", summary)
        self.assertIn("ForecastLoadMAERecent", summary)
        self.assertGreater(summary.sizes["time"], power.sizes["time"])


if __name__ == "__main__":
    unittest.main()
