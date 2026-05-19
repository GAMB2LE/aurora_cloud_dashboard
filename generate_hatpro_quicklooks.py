#!/usr/bin/env python3
"""Generate daily and latest HATPRO science quicklook PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path

import pandas as pd
import xarray as xr

from plot_hatpro_last24h import _plot_hatpro, plot_window

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("HATPRO_ZARR_PATH", "/data/aurora/products/hatprog5/hatpro.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("HATPRO_QUICKLOOK_DIR", QUICKLOOK_ROOT / "hatpro"))


def _daily_path(day) -> Path:
    return QUICKLOOK_DIR / f"hatpro_{pd.Timestamp(day).strftime('%Y%m%d')}.png"


def main(force: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        for png in QUICKLOOK_DIR.glob("hatpro_*.png"):
            png.unlink()
        latest = QUICKLOOK_DIR / "latest.png"
        if latest.exists():
            latest.unlink()
        print("Deleted existing HATPRO quicklook PNGs.")

    latest_out = QUICKLOOK_DIR / "latest.png"
    plot_window(ZARR_PATH, latest_out)

    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)
    for day in dates:
        output = _daily_path(day)
        if output.exists() and not force:
            continue
        start = pd.Timestamp(day)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        window = ds.isel(time=mask)
        if window.sizes.get("time", 0) < 2:
            continue
        title = pd.Timestamp(day).strftime("Scanning Microwave Radiometer - %Y-%m-%d")
        _plot_hatpro(window, title, output, x_start=start, x_end=end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HATPRO science quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
