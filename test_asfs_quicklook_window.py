from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import plot_asfs_logger_last24h


class AsfsQuicklookWindowTests(unittest.TestCase):
    def test_latest_plot_recrops_after_augmentation(self) -> None:
        old_times = pd.date_range("2024-01-01", periods=4, freq="h")
        latest_times = pd.date_range("2026-07-05", periods=25, freq="h")
        wide_times = old_times.append(latest_times)
        asfs_ds = xr.Dataset(
            {"batt_volt_Avg": (("time",), np.arange(len(latest_times), dtype=np.float32))},
            coords={"time": latest_times},
        )
        wide_hk = xr.Dataset(
            {"batt_volt_Avg": (("time",), np.arange(len(wide_times), dtype=np.float32))},
            coords={"time": wide_times},
        )
        captured: dict[str, object] = {}

        def fake_plot(ds: xr.Dataset, **kwargs) -> list[str]:
            captured["times"] = pd.DatetimeIndex(ds["time"].values)
            captured["x_limits"] = kwargs["x_limits"]
            return ["batt_volt_Avg"]

        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "asfs_logger.zarr"
            output = Path(tmp) / "latest.png"
            with (
                patch.object(plot_asfs_logger_last24h.xr, "open_zarr", return_value=asfs_ds),
                patch.object(plot_asfs_logger_last24h, "combine_summary_datasets", return_value=asfs_ds),
                patch.object(plot_asfs_logger_last24h, "augment_asfs_from_fast_gas", return_value=wide_hk),
                patch.object(plot_asfs_logger_last24h, "plot_housekeeping_timeseries", side_effect=fake_plot),
            ):
                plot_asfs_logger_last24h.plot_last_24h_group(zarr_path, Path(tmp) / "missing-fast.zarr", output)

        plotted_times = captured["times"]
        start, end = captured["x_limits"]
        self.assertGreaterEqual(plotted_times.min(), start)
        self.assertLessEqual(plotted_times.max(), end)
        self.assertLessEqual((plotted_times.max() - plotted_times.min()) / pd.Timedelta(hours=1), 24)


if __name__ == "__main__":
    unittest.main()
