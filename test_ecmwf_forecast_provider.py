from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from ecmwf_forecast_provider import open_solar_forecast, validate_provider


class EcmwfForecastProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.input_path = self.root / "solar.nc"
        cycle = pd.Timestamp("2026-07-16T00:00:00")
        step_hours = np.array([0, 3, 6], dtype=np.int32)
        step = pd.to_timedelta(step_hours, unit="h")
        values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
        xr.Dataset(
            {"surface_solar_radiation_downwards": (("step", "latitude", "longitude"), values)},
            coords={
                "time": cycle.to_datetime64(),
                "step": step_hours,
                "valid_time": ("step", (cycle + step).to_numpy()),
                "latitude": [65.0, 64.0],
                "longitude": [336.0, 337.0],
            },
        ).to_netcdf(self.input_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_provider_validation_rejects_unknown_values(self) -> None:
        self.assertEqual(validate_provider(" EarthKit "), "earthkit")
        with self.assertRaises(ValueError):
            validate_provider("automatic")

    def test_legacy_normalizes_site_and_time_coordinates(self) -> None:
        result = open_solar_forecast(
            self.input_path,
            provider="legacy",
            latitude=64.8,
            longitude=-23.2,
            shadow_report_path=None,
        )

        self.assertEqual(result.diagnostics["effective_provider"], "legacy")
        self.assertEqual(result.dataset["ssrd"].dims, ("lead_time",))
        np.testing.assert_allclose(result.dataset["ssrd"].values, [1.0, 5.0, 9.0])
        self.assertEqual(str(result.dataset.attrs["selected_grid_longitude"]), "337.0")
        self.assertGreater(float(result.dataset.attrs["selected_grid_distance_km"]), 0.0)
        self.assertIn("forecast_reference_time", result.dataset.coords)
        self.assertIn("valid_time", result.dataset.coords)

    def test_earthkit_matches_legacy_for_netcdf_fixture(self) -> None:
        legacy = open_solar_forecast(
            self.input_path,
            provider="legacy",
            latitude=64.8,
            longitude=-23.2,
            shadow_report_path=None,
        )
        earthkit = open_solar_forecast(
            self.input_path,
            provider="earthkit",
            latitude=64.8,
            longitude=-23.2,
            shadow_report_path=None,
        )

        self.assertEqual(earthkit.diagnostics["effective_provider"], "earthkit")
        xr.testing.assert_allclose(legacy.dataset["ssrd"], earthkit.dataset["ssrd"])
        np.testing.assert_array_equal(legacy.dataset["valid_time"], earthkit.dataset["valid_time"])

    def test_shadow_publishes_legacy_and_records_comparison(self) -> None:
        report_path = self.root / "shadow.json"
        result = open_solar_forecast(
            self.input_path,
            provider="shadow",
            latitude=64.8,
            longitude=-23.2,
            shadow_report_path=report_path,
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result.diagnostics["effective_provider"], "legacy")
        self.assertEqual(report["shadow_status"], "compared")
        self.assertEqual(report["ssrd_max_abs_difference_j_m2"], 0.0)
        self.assertTrue(report["valid_times_match"])

    def test_earthkit_failure_falls_back_without_losing_forecast(self) -> None:
        with patch("ecmwf_forecast_provider._open_earthkit", side_effect=RuntimeError("decoder unavailable")):
            result = open_solar_forecast(
                self.input_path,
                provider="earthkit",
                latitude=64.8,
                longitude=-23.2,
                shadow_report_path=None,
            )

        self.assertEqual(result.diagnostics["effective_provider"], "legacy")
        self.assertIn("decoder unavailable", result.diagnostics["fallback_reason"])
        self.assertEqual(result.dataset.sizes["lead_time"], 3)


if __name__ == "__main__":
    unittest.main()
