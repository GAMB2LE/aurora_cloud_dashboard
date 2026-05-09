#!/usr/bin/env python3
"""Generate grouped daily and latest ASFS fast-sonic quicklook PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import os
import pandas as pd
import xarray as xr

from grouped_timeseries import (
    clear_grouped_quicklooks,
    group_daily_png,
    group_latest_png,
    group_specs,
    plot_grouped_timeseries,
)

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("ASFS_FAST_SONIC_ZARR_PATH", "/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("ASFS_FAST_SONIC_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_fast_sonic"))
INSTRUMENT = "asfs-fast-sonic"


def main(force: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")

    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        clear_grouped_quicklooks(QUICKLOOK_DIR, INSTRUMENT)
        print("Deleted existing grouped ASFS fast-sonic quicklook PNGs.")

    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_mask = (time_index >= start_time) & (time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    if latest_day.sizes.get("time", 0) >= 2:
        for spec in group_specs(INSTRUMENT):
            out = group_latest_png(QUICKLOOK_DIR, INSTRUMENT, spec.label)
            plot_grouped_timeseries(latest_day, INSTRUMENT, spec.label, f"{spec.label} - Latest 24 hours", out)

    for day in dates:
        start = pd.Timestamp(day)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        if ds_day.sizes.get("time", 0) < 2:
            continue
        for spec in group_specs(INSTRUMENT):
            out = group_daily_png(QUICKLOOK_DIR, INSTRUMENT, spec.label, day)
            if out.exists() and not force:
                continue
            title = pd.Timestamp(day).strftime(f"{spec.label} - %Y-%m-%d")
            plot_grouped_timeseries(ds_day, INSTRUMENT, spec.label, title, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate grouped ASFS fast-sonic quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
