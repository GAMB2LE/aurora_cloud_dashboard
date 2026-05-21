#!/usr/bin/env python3
"""Render ASFS housekeeping latest-24h PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from grouped_timeseries import (
    augment_asfs_from_fast_gas,
    combine_summary_datasets,
    fast_gas_licor_summary_dataset,
    plot_housekeeping_timeseries,
)

INSTRUMENT = "asfs-logger"
ZARR_DEFAULT = Path("/data/aurora/products/asfs_logger/asfs_logger.zarr")
ASFS_FAST_GAS_ZARR_DEFAULT = Path("/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr")
OUTPUT_DEFAULT = Path("latest_asfs_logger.png")


def plot_timeseries(ds: xr.Dataset, title: str, output: Path) -> None:
    plot_housekeeping_timeseries(ds, instrument=INSTRUMENT, title=title, output=output)


def _window(ds: xr.Dataset, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    time_index = pd.DatetimeIndex(ds["time"].values)
    mask = (time_index >= start) & (time_index <= end)
    return ds.isel(time=mask).sortby("time")


def plot_last_24h_group(zarr_path: Path, fast_gas_zarr_path: Path, output: Path) -> None:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    fast_gas_ds = xr.open_zarr(fast_gas_zarr_path, chunks={}) if fast_gas_zarr_path.exists() else None
    time_index = pd.DatetimeIndex(ds["time"].values)
    fast_gas_time_index = pd.DatetimeIndex(fast_gas_ds["time"].values) if fast_gas_ds is not None and "time" in fast_gas_ds else pd.DatetimeIndex([])
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = max(time_index.max(), fast_gas_time_index.max()) if len(fast_gas_time_index) else time_index.max()
    start_time = end_time - pd.Timedelta(hours=24)
    latest = _window(ds, start_time, end_time)
    inputs = [latest]
    if fast_gas_ds is not None and "time" in fast_gas_ds:
        fast_latest = _window(fast_gas_ds, start_time, end_time)
        fast_summary = fast_gas_licor_summary_dataset(fast_latest)
        if fast_summary.sizes.get("time", 0):
            inputs.append(fast_summary)
    hk = augment_asfs_from_fast_gas(combine_summary_datasets(INSTRUMENT, *inputs))
    plot_housekeeping_timeseries(hk, instrument=INSTRUMENT, title="HK_ASFS - Latest 24 hours", output=output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ASFS housekeeping latest-24h PNGs")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--asfs-fast-gas-zarr", type=Path, default=ASFS_FAST_GAS_ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    plot_last_24h_group(args.zarr_path, args.asfs_fast_gas_zarr, args.output)


if __name__ == "__main__":
    main()
