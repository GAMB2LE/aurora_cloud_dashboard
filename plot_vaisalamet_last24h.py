#!/usr/bin/env python3
"""Render Meteorology housekeeping latest-24h PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import xarray as xr

from grouped_timeseries import plot_housekeeping_timeseries

INSTRUMENT = "vaisalamet"
ZARR_DEFAULT = Path("/data/aurora/products/vaisalamet/vaisalamet.zarr")
ASFS_LOGGER_ZARR_DEFAULT = Path("/data/aurora/products/asfs_logger/asfs_logger.zarr")
OUTPUT_DEFAULT = Path("latest_vaisalamet.png")
MET_HK_EXTRA_VARS = ("kt15_amb_Avg",)


def _meteorology_housekeeping_dataset(vaisala_ds: xr.Dataset, asfs_ds: xr.Dataset) -> xr.Dataset:
    keep = [name for name in MET_HK_EXTRA_VARS if name in asfs_ds.data_vars and asfs_ds[name].dims == ("time",)]
    if not keep:
        return vaisala_ds
    hk_extra = asfs_ds[keep].sortby("time")
    merged = xr.merge([vaisala_ds.sortby("time"), hk_extra], join="outer", compat="override", combine_attrs="drop_conflicts")
    return merged.sortby("time")


def plot_timeseries(ds: xr.Dataset, title: str, output: Path) -> None:
    plot_housekeeping_timeseries(ds, instrument=INSTRUMENT, title=title, output=output)


def plot_last_24h_group(zarr_path: Path, asfs_logger_zarr_path: Path, output: Path) -> None:
    ds = xr.open_zarr(zarr_path, chunks={})
    asfs_ds = xr.open_zarr(asfs_logger_zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    if "time" not in asfs_ds:
        raise KeyError("ASFS logger dataset is missing a time coordinate")
    time_index = ds.indexes["time"]
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_mask = (time_index >= start_time) & (time_index <= end_time)
    asfs_time_index = asfs_ds.indexes["time"]
    asfs_latest_mask = (asfs_time_index >= start_time) & (asfs_time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    hk_day = _meteorology_housekeeping_dataset(
        latest_day,
        asfs_ds.isel(time=asfs_latest_mask).sortby("time"),
    )
    title = "HK_Met - Latest 24 hours"
    plot_housekeeping_timeseries(hk_day, instrument=INSTRUMENT, title=title, output=output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Meteorology housekeeping latest-24h PNGs")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--asfs-logger-zarr", type=Path, default=ASFS_LOGGER_ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    plot_last_24h_group(args.zarr_path, args.asfs_logger_zarr, args.output)


if __name__ == "__main__":
    main()
