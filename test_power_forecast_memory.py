import unittest
from unittest.mock import patch

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import xarray as xr

import generate_power_soc_forecast as forecast
from power_battery_model import fit_battery_model


def forbidden_read():
    raise AssertionError("Freshness read unrelated or old telemetry")


class ForecastMemoryTests(unittest.TestCase):
    def test_disk_backed_history_retains_every_value_after_file_handle_closes(self):
        import gc
        times = pd.date_range("2026-01-01", periods=150000, freq="s")
        values = np.arange(len(times), dtype=float)
        values[::37] = np.nan
        power = xr.Dataset({"BatterySOC": ("time", values),
                            "BatteryWatts": ("time", -values)}, coords={"time": times}).chunk(time=8192)
        with patch.object(forecast, "POWER_FRAME_MEMMAP_ROWS", 1):
            frame = forecast._power_frame(power)
        gc.collect()
        np.testing.assert_array_equal(frame.BatterySOC, values)
        np.testing.assert_array_equal(frame.BatteryWatts, -values)
        self.assertEqual(len(frame), len(times))

    def test_chunked_row_sum_is_exact_with_nan_and_minimum_coverage(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(size=(150000, 3)), columns=["a", "b", "c"])
        frame.iloc[::3, 0] = np.nan
        frame.iloc[::7, :] = np.nan
        for count in (1, 3):
            pd.testing.assert_series_equal(
                forecast._sum_power_columns(frame, list(frame), min_count=count),
                frame.sum(axis=1, min_count=count), check_exact=True,
            )

    def test_freshness_reads_only_the_last_soc_chunk(self):
        count = 65536
        forbidden = da.from_delayed(delayed(forbidden_read)(), shape=(count,), dtype=float)
        soc = da.concatenate([forbidden, da.full(count, 71.0, chunks=count)])
        times = pd.date_range("2026-01-01", periods=2 * count, freq="s")
        power = xr.Dataset({"BatterySOC": ("time", soc),
                            "BatteryWatts": ("time", da.concatenate([forbidden, forbidden]))},
                           coords={"time": times})
        self.assertEqual(forecast.validate_power_input_freshness(power, max_age_minutes=None), (times[-1], 71.0))

    def test_freshness_preserves_nan_duplicate_and_unsorted_policy(self):
        times = pd.to_datetime(["2026-01-02", None, "2026-01-01", "2026-01-02", "2026-01-03"])
        power = xr.Dataset({"BatterySOC": ("time", [90., 100., 70., np.nan, np.nan])}, coords={"time": times})
        self.assertEqual(forecast._latest_power_soc(power), (pd.Timestamp("2026-01-01"), 70.0))
        with self.assertRaisesRegex(ValueError, "stale"):
            forecast.validate_power_input_freshness(power, max_age_minutes=20, now=pd.Timestamp("2026-01-04"))

    def test_frame_preserves_values_without_copying_normal_columns(self):
        values = np.arange(50, dtype=float)
        power = xr.Dataset({"BatterySOC": ("time", values)},
                           coords={"time": pd.date_range("2026-01-01", periods=50, freq="h")})
        frame = forecast._power_frame(power)
        self.assertTrue(np.shares_memory(frame.BatterySOC.to_numpy(), values))
        np.testing.assert_array_equal(frame.BatterySOC, values)
        nat = power.isel(time=[0]).assign_coords(time=[np.datetime64("NaT", "ns")])
        self.assertTrue(forecast._power_frame(nat).empty)

    def test_battery_window_does_not_change_fit_or_discard_verification_history(self):
        times = pd.date_range("2026-06-01", periods=30 * 24, freq="h")
        power = xr.Dataset({
            "BatterySOC": ("time", 70 + 5 * np.sin(np.arange(len(times)) / 12)),
            "BatteryWatts": ("time", np.full(len(times), -150.)),
            **{f"SolarWatts_{name}": ("time", np.full(len(times), 10.)) for name in ("East", "South", "West")},
            "ACOutputWatts": ("time", np.full(len(times), 100.)),
            "DCInverterWatts": ("time", np.full(len(times), 50.)),
        }, coords={"time": times})
        frame = forecast._power_frame(power)
        expected = fit_battery_model(frame.assign(ObservedLoadWatts=forecast._observed_load_w(frame)), lookback_days=7)
        solar = xr.Dataset({"ssrd": ("time", [0., 0., 0.])},
                           coords={"time": pd.date_range(times[-1], periods=3, freq="3h")})
        with patch.object(forecast, "fit_battery_model", wraps=fit_battery_model) as fit, \
             patch.object(forecast, "evaluate_independent_forecast_archive", return_value={}) as evaluate:
            result = forecast.build_forecast_dataset(power, solar, horizon_hours=6)
        self.assertLess(len(fit.call_args.args[0]), len(times))
        self.assertEqual(len(evaluate.call_args.args[1]), len(times))
        for key, value in expected.attrs().items():
            self.assertEqual(result.attrs[key], value)


if __name__ == "__main__":
    unittest.main()
