#!/usr/bin/env python3
"""Generate one daily stacked time-series quicklook per available Vaisala met day."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

from plot_vaisalamet_last24h import plot_timeseries

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("VAISALAMET_ZARR_PATH", "/data/aurora/products/vaisalamet/vaisalamet.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("VAISALAMET_QUICKLOOK_DIR", QUICKLOOK_ROOT / "vaisalamet"))


def main(force: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")

    time_index = pd.DatetimeIndex(ds["time"].values)
    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        for png in QUICKLOOK_DIR.glob("vaisalamet_*.png"):
            png.unlink()
        print("Deleted existing Vaisala met quicklook PNGs.")

    for day in dates:
        out = QUICKLOOK_DIR / f"vaisalamet_{pd.Timestamp(day).strftime('%Y%m%d')}.png"
        if out.exists():
            continue
        start = pd.Timestamp(day)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        if ds_day.sizes.get("time", 0) < 2:
            continue
        plot_timeseries(ds_day, pd.Timestamp(day).strftime("vaisalamet - %Y-%m-%d"), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Vaisala met daily quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
