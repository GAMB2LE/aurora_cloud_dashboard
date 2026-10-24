"""Regression coverage for fast-gas append boundaries at nanosecond precision."""

import numpy as np
import pandas as pd
import xarray as xr

from append_new_asfs_fast_gas_to_zarr import append_new


def _write_bursts(path, minute, first_record):
    lines = [
        '"TOA5","asfs-logger"',
        '"TIMESTAMP","RECORD","CO2"',
        '"TS","RN","ppm"',
        '"","",""',
    ]
    for index in range(42):
        second = 0 if index < 21 else 20
        lines.append(
            f'"2026-09-13 00:{minute:02d}:{second:02d}",'
            f'{first_record + index},{400 + first_record + index}'
        )
    path.write_text("\n".join(lines) + "\n")


def _load(store):
    with xr.open_zarr(store) as dataset:
        return dataset.load()


def test_repeat_append_preserves_nanosecond_boundary_and_existing_prefix(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = tmp_path / "fast-gas.zarr"
    _write_bursts(raw / "aurora_asfs_data_fast_gas_202609130001.dat", 0, 0)
    append_new(raw, store)
    original = _load(store)

    # Twenty-one samples per 20-second burst produce a final timestamp with nanosecond
    # precision, which Python datetime would truncate to microseconds.
    assert original.sizes["time"] == 42
    assert pd.Timestamp(original.time.values[-1]).nanosecond != 0
    append_new(raw, store)
    xr.testing.assert_identical(_load(store), original)

    _write_bursts(raw / "aurora_asfs_data_fast_gas_202609130002.dat", 1, 42)
    append_new(raw, store)
    appended = _load(store)
    assert appended.sizes["time"] == 84
    xr.testing.assert_equal(appended.isel(time=slice(0, 42)), original)
    np.testing.assert_array_equal(appended.RECORD.values, np.arange(84))
    assert appended.indexes["time"].is_unique
    assert appended.indexes["time"].is_monotonic_increasing
    assert pd.Timestamp(appended.time.values[-1]).nanosecond != 0

    append_new(raw, store)
    xr.testing.assert_identical(_load(store), appended)
