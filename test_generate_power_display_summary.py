from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from generate_power_display_summary import (
    _open_optional_zarr,
    _release_generation_lock,
    _section_subset,
    _try_generation_lock,
    _write_metadata,
    _write_zarr_atomic,
)
from grouped_timeseries import POWER_PANEL_TIME_GROUP_BY_KEY, SUMMARY_LAYOUTS


class PowerDisplaySummaryMetadataTests(unittest.TestCase):
    def test_metadata_records_time_bounds_and_dimensions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "power_display_summary.zarr"
            display = xr.Dataset(
                {"power": ("time", np.asarray([1.0, 2.0]))},
                coords={"time": np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")},
            )
            path = _write_metadata(output, display)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["time_count"], 2)
        self.assertEqual(payload["variable_count"], 1)
        self.assertEqual(str(np.datetime64(payload["time_start_utc"])), "2026-07-17T00:00:00")
        self.assertEqual(str(np.datetime64(payload["time_end_utc"])), "2026-07-17T00:01:00")

    def test_section_products_contain_only_their_panel_variables(self) -> None:
        times = np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")
        fields = {
            trace.var
            for panel in SUMMARY_LAYOUTS["power"]
            for trace in panel.traces
        }
        display = xr.Dataset(
            {name: ("time", np.asarray([1.0, 2.0])) for name in fields},
            coords={"time": times},
        )

        current = _section_subset(display, "current")
        forecast = _section_subset(display, "forecast")
        current_fields = {
            trace.var
            for panel in SUMMARY_LAYOUTS["power"]
            if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key, "observed") == "observed"
            for trace in panel.traces
        }
        forecast_fields = fields - current_fields

        self.assertEqual(set(current.data_vars), current_fields)
        self.assertEqual(set(forecast.data_vars), forecast_fields)

    def test_generation_lock_skips_overlapping_build(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "power_display_summary.zarr"
            first = _try_generation_lock(output)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(_try_generation_lock(output))
            finally:
                _release_generation_lock(first)

            second = _try_generation_lock(output)
            self.assertIsNotNone(second)
            _release_generation_lock(second)

    def test_atomic_store_has_consolidated_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "display.zarr"
            display = xr.Dataset(
                {"power": ("time", np.asarray([1.0, 2.0]))},
                coords={"time": np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")},
            )
            _write_zarr_atomic(display, output, chunk_time=1)
            opened = xr.open_zarr(output, consolidated=True)
            try:
                self.assertEqual(opened.sizes["time"], 2)
            finally:
                opened.close()

    def test_eager_optional_store_is_stable_after_source_replacement(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "forecast.zarr"
            first = xr.Dataset(
                {"soc": (("scenario", "time"), np.asarray([[90.0, 91.0]]))},
                coords={"scenario": ["current"], "time": [0, 1]},
            )
            second = xr.Dataset(
                {"soc": (("scenario", "time"), np.asarray([[80.0, 81.0, 82.0]]))},
                coords={"scenario": ["current"], "time": [0, 1, 2]},
            )
            first.to_zarr(output, mode="w", consolidated=True)
            snapshot = _open_optional_zarr(output, "forecast", eager=True)
            self.assertIsNotNone(snapshot)

            _write_zarr_atomic(second, output, chunk_time=3)

            np.testing.assert_array_equal(snapshot["soc"].values, [[90.0, 91.0]])
            self.assertEqual(snapshot.sizes["time"], 2)


if __name__ == "__main__":
    unittest.main()
