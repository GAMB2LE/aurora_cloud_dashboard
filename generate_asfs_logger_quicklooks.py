#!/usr/bin/env python3
"""Generate Radiation summary and housekeeping quicklook PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import os
import pandas as pd
import xarray as xr

from grouped_timeseries import (
    augment_asfs_from_fast_gas,
    build_summary_plotly,
    clear_generated_quicklooks,
    combine_summary_datasets,
    fast_gas_licor_summary_dataset,
    housekeeping_daily_png,
    housekeeping_label,
    housekeeping_latest_png,
    plot_housekeeping_timeseries,
    save_summary_png,
    summary_daily_png,
    summary_latest_png,
    refresh_legacy_aliases,
)

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"))
ASFS_FAST_GAS_ZARR_PATH = Path(
    os.environ.get("ASFS_FAST_GAS_ZARR_PATH", "/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr")
)
QUICKLOOK_DIR = Path(os.environ.get("ASFS_LOGGER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_logger"))
INSTRUMENT = "asfs-logger"
PREWARM_DIR = Path(os.environ.get("AURORA_INTERACTIVE_PREWARM_DIR", "/data/aurora/products/dashboard/prewarm"))
PREWARM_JSON = PREWARM_DIR / "asfs_logger_latest_interactive.json"


def _open_fast_gas_dataset() -> xr.Dataset | None:
    if not ASFS_FAST_GAS_ZARR_PATH.exists():
        return None
    try:
        ds = xr.open_zarr(ASFS_FAST_GAS_ZARR_PATH, chunks={})
    except Exception as exc:
        print(f"Could not open ASFS fast-gas Zarr for LI-COR gap fill: {exc}")
        return None
    return ds if "time" in ds else None


def _window(ds: xr.Dataset, time_index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    mask = (time_index >= start) & (time_index <= end)
    return ds.isel(time=mask).sortby("time")


def _housekeeping_dataset(asfs_ds: xr.Dataset, fast_gas_ds: xr.Dataset | None = None) -> xr.Dataset:
    inputs = [asfs_ds.sortby("time")]
    if fast_gas_ds is not None:
        fast_summary = fast_gas_licor_summary_dataset(fast_gas_ds)
        if fast_summary.sizes.get("time", 0):
            inputs.append(fast_summary.sortby("time"))
    merged = combine_summary_datasets(INSTRUMENT, *inputs)
    return augment_asfs_from_fast_gas(merged).sortby("time")


def main(force: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    fast_gas_ds = _open_fast_gas_dataset()
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")

    time_index = pd.DatetimeIndex(ds["time"].values)
    fast_gas_time_index = pd.DatetimeIndex(fast_gas_ds["time"].values) if fast_gas_ds is not None else pd.DatetimeIndex([])
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        clear_generated_quicklooks(QUICKLOOK_DIR, INSTRUMENT)
        print("Deleted existing Radiation quicklook PNGs.")

    end_time = max(time_index.max(), fast_gas_time_index.max()) if len(fast_gas_time_index) else time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_day = _window(ds, time_index, start_time, end_time)
    latest_fast_gas = _window(fast_gas_ds, fast_gas_time_index, start_time, end_time) if fast_gas_ds is not None else None
    if latest_day.sizes.get("time", 0) >= 2:
        summary_out = summary_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        save_summary_png(
            latest_day,
            INSTRUMENT,
            "Radiation - Latest 24 hours",
            summary_out,
            x_limits=(start_time, end_time),
        )
        PREWARM_DIR.mkdir(parents=True, exist_ok=True)
        fig = build_summary_plotly(latest_day, INSTRUMENT, title="Radiation", max_time_samples=1400)
        fig.write_json(PREWARM_JSON)
        print(f"Wrote {PREWARM_JSON}")
        hk_out = housekeeping_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        if hk_out is not None:
            hk_title = f"{housekeeping_label(INSTRUMENT)} - Latest 24 hours"
            plot_housekeeping_timeseries(
                _housekeeping_dataset(latest_day, latest_fast_gas),
                INSTRUMENT,
                hk_title,
                hk_out,
                x_limits=(start_time, end_time),
            )
            refresh_legacy_aliases(QUICKLOOK_DIR, INSTRUMENT, latest_png=hk_out)

    for day in dates:
        start = pd.Timestamp(day)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        if ds_day.sizes.get("time", 0) < 2:
            continue
        summary_out = summary_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if force or not summary_out.exists():
            title = pd.Timestamp(day).strftime("Radiation - %Y-%m-%d")
            save_summary_png(ds_day, INSTRUMENT, title, summary_out, x_limits=(start, end))
        hk_out = housekeeping_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if hk_out is not None and (force or not hk_out.exists()):
            fast_day = _window(fast_gas_ds, fast_gas_time_index, start, end) if fast_gas_ds is not None else None
            hk_title = pd.Timestamp(day).strftime(f"{housekeeping_label(INSTRUMENT)} - %Y-%m-%d")
            plot_housekeeping_timeseries(
                _housekeeping_dataset(ds_day, fast_day),
                INSTRUMENT,
                hk_title,
                hk_out,
                x_limits=(start, end),
            )
            refresh_legacy_aliases(QUICKLOOK_DIR, INSTRUMENT, day_png=hk_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Radiation summary and housekeeping quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
