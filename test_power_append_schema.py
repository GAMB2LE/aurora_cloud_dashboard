"""Regression fixtures for the canonical powerlogger daily CSV contract."""
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from append_new_power_to_zarr import append_new


def test_single_row_bootstrap_preserves_subday_times_and_new_mpp_columns(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = tmp_path / "power.zarr"
    legacy = raw / "power_data_20260908.csv"
    # Duplicate aps_time is the existing producer's index-plus-column format.
    header = "aps_time,BatterySOC,aps_time\n"
    first = "2026-09-08 23:59:58.123456,62,2026-09-08 23:59:58.123456\n"
    legacy.write_text(header + first)
    append_new(raw, store)
    legacy.write_text(header + first + "2026-09-08 23:59:59.654321,63,2026-09-08 23:59:59.654321\n")
    append_new(raw, store)
    current = raw / "power_data_20260909.csv"
    current.write_text(
        "aps_time,BatterySOC,SolarMPPMode_East,SolarMPPMode_South,SolarMPPMode_West,aps_time\n"
        "2026-09-09 00:00:00.000000,64,2,1,,2026-09-09 00:00:00.000000\n"
        "2026-09-09 00:00:01.234567,65,,2,2,2026-09-09 00:00:01.234567\n"
    )
    append_new(raw, store)
    append_new(raw, store)
    with xr.open_zarr(store) as dataset:
        ds = dataset.load()
    expected = pd.to_datetime([
        "2026-09-08 23:59:58.123456", "2026-09-08 23:59:59.654321",
        "2026-09-09 00:00:00.000000", "2026-09-09 00:00:01.234567",
    ])
    np.testing.assert_array_equal(ds.time.values, expected.values)
    np.testing.assert_array_equal(ds.BatterySOC.values, [62, 63, 64, 65])
    np.testing.assert_allclose(ds.SolarMPPMode_East.values, [np.nan, np.nan, 2, np.nan], equal_nan=True)
    np.testing.assert_allclose(ds.SolarMPPMode_South.values, [np.nan, np.nan, 1, 2], equal_nan=True)
    np.testing.assert_allclose(ds.SolarMPPMode_West.values, [np.nan, np.nan, np.nan, 2], equal_nan=True)
    assert ds.indexes["time"].is_unique
    assert ds.indexes["time"].is_monotonic_increasing


def test_append_retains_existing_store_time_encoding(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = tmp_path / "power.zarr"
    times = pd.to_datetime(["2026-09-08 12:00:00", "2026-09-08 12:00:01"])
    original = xr.Dataset({"BatterySOC": ("time", np.array([62, 63], dtype=np.float32))}, coords={"time": times})
    original.to_zarr(store, mode="w", consolidated=True, encoding={"time": {"units": "seconds since 2026-09-08", "dtype": "int64"}})
    before = dict(zarr.open_group(store)["time"].attrs)
    (raw / "power_data_20260908.csv").write_text("aps_time,BatterySOC\n2026-09-08 12:00:02,64\n")
    append_new(raw, store)
    assert dict(zarr.open_group(store)["time"].attrs) == before
    with xr.open_zarr(store) as ds:
        np.testing.assert_array_equal(ds.time.values, pd.date_range("2026-09-08T12:00:00", periods=3, freq="s").values)
        np.testing.assert_array_equal(ds.BatterySOC.values, [62, 63, 64])
