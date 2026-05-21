#!/usr/bin/env python3
"""Render Meteorology housekeeping latest-24h PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import xarray as xr

from grouped_timeseries import (
    FAST_SONIC_TO_LOGGER_AVG,
    augment_meteorology_from_fast_sonic,
    combine_summary_datasets,
    fast_sonic_metek_summary_dataset,
    plot_housekeeping_timeseries,
)

INSTRUMENT = "vaisalamet"
ZARR_DEFAULT = Path("/data/aurora/products/vaisalamet/vaisalamet.zarr")
ASFS_LOGGER_ZARR_DEFAULT = Path("/data/aurora/products/asfs_logger/asfs_logger.zarr")
ASFS_FAST_SONIC_ZARR_DEFAULT = Path("/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr")
OUTPUT_DEFAULT = Path("latest_vaisalamet.png")
MET_HK_EXTRA_VARS = ("kt15_amb_Avg", "metek_InclX_out_Avg", "metek_InclY_out_Avg")


def _meteorology_housekeeping_dataset(
    vaisala_ds: xr.Dataset,
    asfs_ds: xr.Dataset,
    fast_sonic_ds: xr.Dataset | None = None,
) -> xr.Dataset:
    keep = [name for name in MET_HK_EXTRA_VARS if name in asfs_ds.data_vars and asfs_ds[name].dims == ("time",)]
    inputs = [vaisala_ds.sortby("time")]
    if keep:
        inputs.append(asfs_ds[keep].sortby("time"))
    fast_summary = fast_sonic_metek_summary_dataset(fast_sonic_ds) if fast_sonic_ds is not None else xr.Dataset()
    if fast_summary.sizes.get("time", 0):
        inputs.append(fast_summary.sortby("time"))
    merged = combine_summary_datasets(INSTRUMENT, *inputs)
    merged = augment_meteorology_from_fast_sonic(merged)
    raw_fast_sonic = [name for name in FAST_SONIC_TO_LOGGER_AVG if name in merged.data_vars]
    if raw_fast_sonic:
        merged = merged.drop_vars(raw_fast_sonic)
    return merged.sortby("time")


def plot_timeseries(ds: xr.Dataset, title: str, output: Path) -> None:
    plot_housekeeping_timeseries(ds, instrument=INSTRUMENT, title=title, output=output)


def plot_last_24h_group(
    zarr_path: Path,
    asfs_logger_zarr_path: Path,
    asfs_fast_sonic_zarr_path: Path,
    output: Path,
) -> None:
    ds = xr.open_zarr(zarr_path, chunks={})
    asfs_ds = xr.open_zarr(asfs_logger_zarr_path, chunks={})
    fast_sonic_ds = xr.open_zarr(asfs_fast_sonic_zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    if "time" not in asfs_ds:
        raise KeyError("ASFS logger dataset is missing a time coordinate")
    if "time" not in fast_sonic_ds:
        raise KeyError("ASFS fast-sonic dataset is missing a time coordinate")
    time_index = ds.indexes["time"]
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_mask = (time_index >= start_time) & (time_index <= end_time)
    asfs_time_index = asfs_ds.indexes["time"]
    asfs_latest_mask = (asfs_time_index >= start_time) & (asfs_time_index <= end_time)
    fast_sonic_time_index = fast_sonic_ds.indexes["time"]
    fast_sonic_latest_mask = (fast_sonic_time_index >= start_time) & (fast_sonic_time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    hk_day = _meteorology_housekeeping_dataset(
        latest_day,
        asfs_ds.isel(time=asfs_latest_mask).sortby("time"),
        fast_sonic_ds.isel(time=fast_sonic_latest_mask).sortby("time"),
    )
    title = "HK_Met - Latest 24 hours"
    plot_housekeeping_timeseries(hk_day, instrument=INSTRUMENT, title=title, output=output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Meteorology housekeeping latest-24h PNGs")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--asfs-logger-zarr", type=Path, default=ASFS_LOGGER_ZARR_DEFAULT)
    parser.add_argument("--asfs-fast-sonic-zarr", type=Path, default=ASFS_FAST_SONIC_ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    plot_last_24h_group(args.zarr_path, args.asfs_logger_zarr, args.asfs_fast_sonic_zarr, args.output)


if __name__ == "__main__":
    main()
