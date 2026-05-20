#!/usr/bin/env python3
"""
Generate daily cloud-radar quicklook PNGs with the shared compact layout.
Panels (top→bottom): ZE_dBZ, MeanVel, SpecWidth, SLDR, RHV, SRCX, Skew, Kurt.
"""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path

import pandas as pd
import xarray as xr
from extra_housekeeping import (
    extra_housekeeping_daily_png,
    load_cloud_radar_housekeeping_from_raw,
    plot_cloud_radar_housekeeping,
)
from plot_cloud_radar_last24h import RANGE_MAX, plot_radar_quicklook, required_radar_vars

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("CLOUD_RADAR_ZARR_PATH", "/data/aurora/products/rpgfmcw94/cloud_radar.zarr"))
RAW_ROOT = Path(os.environ.get("CLOUD_RADAR_RAW_ROOT", "/project/aurora/raw/rpgfmcw94"))
QUICKLOOK_DIR = Path(os.environ.get("CLOUD_RADAR_QUICKLOOK_DIR", QUICKLOOK_ROOT / "cloud_radar"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(force: bool = False):
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    needed = required_radar_vars()
    missing = [v for v in needed if v not in ds]
    if missing:
        raise KeyError(f"Dataset missing variables: {', '.join(missing)}")

    time_index = pd.DatetimeIndex(ds["time"].values)
    today = pd.Timestamp.utcnow().replace(tzinfo=None).date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    _ensure_dir(QUICKLOOK_DIR)

    if force:
        for png in QUICKLOOK_DIR.glob("cloud_radar_*.png"):
            png.unlink()
        for png in QUICKLOOK_DIR.glob("cloud_radar__hk_radar__*.png"):
            png.unlink()
        print("Deleted existing quicklook PNGs.")

    current_times = time_index[time_index <= pd.Timestamp.utcnow().replace(tzinfo=None)]
    if len(current_times):
        end_time = current_times.max()
        start_time = end_time - timedelta(hours=24)
        hk_latest_out = QUICKLOOK_DIR / "cloud_radar__hk_radar__latest.png"
        hk_latest = load_cloud_radar_housekeeping_from_raw(RAW_ROOT, start_time, end_time)
        if hk_latest.sizes.get("time", 0) >= 2:
            plot_cloud_radar_housekeeping(hk_latest, "HK_Radar - Latest 24 hours", hk_latest_out)

    for d in dates:
        out = QUICKLOOK_DIR / f"cloud_radar_{pd.Timestamp(d).strftime('%Y%m%d')}.png"
        hk_out = extra_housekeeping_daily_png(QUICKLOOK_DIR, "Cloud Radar", d)
        if out.exists() and hk_out is not None and hk_out.exists():
            continue
        start = pd.Timestamp(d)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        ds_day = ds_day.sel({"range": slice(0, RANGE_MAX)})
        if ds_day.sizes.get("time", 0) < 2:
            continue
        if not out.exists():
            plot_radar_quicklook(
                ds_day,
                pd.Timestamp(d).strftime("Cloud Radar - %Y-%m-%d"),
                out,
                x_start=start,
                x_end=end,
            )
        if hk_out is not None and not hk_out.exists():
            hk_title = pd.Timestamp(d).strftime("HK_Radar - %Y-%m-%d")
            hk_day = load_cloud_radar_housekeeping_from_raw(RAW_ROOT, start, end)
            if hk_day.sizes.get("time", 0) >= 2:
                plot_cloud_radar_housekeeping(hk_day, hk_title, hk_out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cloud radar quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
