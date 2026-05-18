#!/usr/bin/env python3
"""Generate Meteorology summary and housekeeping quicklook PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import os
import pandas as pd
import xarray as xr

from grouped_timeseries import (
    combine_summary_datasets,
    clear_generated_quicklooks,
    housekeeping_daily_png,
    housekeeping_label,
    housekeeping_latest_png,
    plot_housekeeping_timeseries,
    refresh_legacy_aliases,
    save_summary_png,
    summary_daily_png,
    summary_latest_png,
)

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("VAISALAMET_ZARR_PATH", "/data/aurora/products/vaisalamet/vaisalamet.zarr"))
ASFS_LOGGER_ZARR_PATH = Path(os.environ.get("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("VAISALAMET_QUICKLOOK_DIR", QUICKLOOK_ROOT / "vaisalamet"))
INSTRUMENT = "vaisalamet"
MET_HK_EXTRA_VARS = ("kt15_amb_Avg", "metek_InclX_out_Avg", "metek_InclY_out_Avg")


def _meteorology_housekeeping_dataset(vaisala_ds: xr.Dataset, asfs_ds: xr.Dataset) -> xr.Dataset:
    keep = [name for name in MET_HK_EXTRA_VARS if name in asfs_ds.data_vars and asfs_ds[name].dims == ("time",)]
    if not keep:
        return vaisala_ds
    hk_extra = asfs_ds[keep].sortby("time")
    merged = xr.merge([vaisala_ds.sortby("time"), hk_extra], join="outer", compat="override", combine_attrs="drop_conflicts")
    return merged.sortby("time")


def main(force: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    asfs_ds = xr.open_zarr(ASFS_LOGGER_ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    if "time" not in asfs_ds:
        raise KeyError("ASFS logger dataset is missing a time coordinate")

    time_index = pd.DatetimeIndex(ds["time"].values)
    asfs_time_index = pd.DatetimeIndex(asfs_ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        clear_generated_quicklooks(QUICKLOOK_DIR, INSTRUMENT)
        print("Deleted existing Meteorology quicklook PNGs.")

    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_mask = (time_index >= start_time) & (time_index <= end_time)
    asfs_latest_mask = (asfs_time_index >= start_time) & (asfs_time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    latest_hk = _meteorology_housekeeping_dataset(
        latest_day,
        asfs_ds.isel(time=asfs_latest_mask).sortby("time"),
    )
    latest_summary = combine_summary_datasets(
        INSTRUMENT,
        latest_day,
        asfs_ds.isel(time=asfs_latest_mask).sortby("time"),
    )
    if latest_day.sizes.get("time", 0) >= 2:
        summary_out = summary_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        save_summary_png(latest_summary, INSTRUMENT, "Meteorology - Latest 24 hours", summary_out)
        hk_out = housekeeping_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        if hk_out is not None:
            hk_title = f"{housekeeping_label(INSTRUMENT)} - Latest 24 hours"
            plot_housekeeping_timeseries(latest_hk, INSTRUMENT, hk_title, hk_out)
            refresh_legacy_aliases(QUICKLOOK_DIR, INSTRUMENT, latest_png=hk_out)

    for day in dates:
        start = pd.Timestamp(day)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        asfs_mask = (asfs_time_index >= start) & (asfs_time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        hk_day = _meteorology_housekeeping_dataset(
            ds_day,
            asfs_ds.isel(time=asfs_mask).sortby("time"),
        )
        summary_day = combine_summary_datasets(
            INSTRUMENT,
            ds_day,
            asfs_ds.isel(time=asfs_mask).sortby("time"),
        )
        if ds_day.sizes.get("time", 0) < 2:
            continue
        summary_out = summary_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if force or not summary_out.exists():
            title = pd.Timestamp(day).strftime("Meteorology - %Y-%m-%d")
            save_summary_png(summary_day, INSTRUMENT, title, summary_out)
        hk_out = housekeeping_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if hk_out is not None and (force or not hk_out.exists()):
            hk_title = pd.Timestamp(day).strftime(f"{housekeeping_label(INSTRUMENT)} - %Y-%m-%d")
            plot_housekeeping_timeseries(hk_day, INSTRUMENT, hk_title, hk_out)
            refresh_legacy_aliases(QUICKLOOK_DIR, INSTRUMENT, day_png=hk_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Meteorology summary and housekeeping quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
