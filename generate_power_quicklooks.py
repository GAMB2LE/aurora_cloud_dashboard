#!/usr/bin/env python3
"""Generate Aurora Power Supply summary and housekeeping quicklook PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import os
import pandas as pd
import xarray as xr

from grouped_timeseries import (
    clear_generated_quicklooks,
    combine_summary_datasets,
    housekeeping_daily_png,
    housekeeping_label,
    housekeeping_latest_png,
    plot_housekeeping_timeseries,
    save_summary_png,
    summary_daily_png,
    summary_latest_png,
    refresh_legacy_aliases,
    SUMMARY_DISPLAY_END_ATTR,
    SUMMARY_DISPLAY_START_ATTR,
)

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
ASFS_LOGGER_ZARR_PATH = Path(os.environ.get("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("POWER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "power"))
INSTRUMENT = "power"
ASS_POWER_VAR = "watts_on_48vdc_Avg"


def _optional_ass_power_dataset() -> xr.Dataset | None:
    """Return the ASS 48 V power series used as context on the APS output plot."""
    if not ASFS_LOGGER_ZARR_PATH.exists():
        return None
    try:
        ds = xr.open_zarr(ASFS_LOGGER_ZARR_PATH, chunks={})
    except Exception as exc:
        print(f"Could not open ASFS logger Zarr for ASS power overlay: {exc}")
        return None
    if "time" not in ds or ASS_POWER_VAR not in ds:
        return None
    return ds[[ASS_POWER_VAR]]


def _slice_window(ds: xr.Dataset | None, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset | None:
    if ds is None or "time" not in ds:
        return None
    time_index = pd.DatetimeIndex(ds["time"].values)
    mask = (time_index >= start) & (time_index <= end)
    if not mask.any():
        return None
    return ds.isel(time=mask).sortby("time")


def _with_display_window(ds: xr.Dataset, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    """Mark a context-loaded dataset so summary prep crops to the visible window."""
    ds = ds.copy(deep=False)
    ds.attrs[SUMMARY_DISPLAY_START_ATTR] = pd.Timestamp(start).isoformat()
    ds.attrs[SUMMARY_DISPLAY_END_ATTR] = pd.Timestamp(end).isoformat()
    return ds


def main(force: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    ass_power = _optional_ass_power_dataset()

    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        clear_generated_quicklooks(QUICKLOOK_DIR, INSTRUMENT)
        print("Deleted existing Aurora Power Supply quicklook PNGs.")

    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_context_start = pd.Timestamp(start_time).normalize()
    latest_mask = (time_index >= latest_context_start) & (time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    if latest_day.sizes.get("time", 0) >= 2:
        latest_summary = combine_summary_datasets(
            INSTRUMENT,
            latest_day,
            _slice_window(ass_power, latest_context_start, pd.Timestamp(end_time)),
        )
        latest_summary = _with_display_window(latest_summary, pd.Timestamp(start_time), pd.Timestamp(end_time))
        summary_out = summary_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        save_summary_png(latest_summary, INSTRUMENT, "Aurora Power Supply - Latest 24 hours", summary_out)
        hk_out = housekeeping_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        if hk_out is not None:
            hk_title = f"{housekeeping_label(INSTRUMENT)} - Latest 24 hours"
            plot_housekeeping_timeseries(latest_day, INSTRUMENT, hk_title, hk_out)
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
        summary_day = combine_summary_datasets(
            INSTRUMENT,
            ds_day,
            _slice_window(ass_power, start, end),
        )
        summary_out = summary_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if force or not summary_out.exists():
            title = pd.Timestamp(day).strftime("Aurora Power Supply - %Y-%m-%d")
            save_summary_png(summary_day, INSTRUMENT, title, summary_out)
        hk_out = housekeeping_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if hk_out is not None and (force or not hk_out.exists()):
            hk_title = pd.Timestamp(day).strftime(f"{housekeeping_label(INSTRUMENT)} - %Y-%m-%d")
            plot_housekeeping_timeseries(ds_day, INSTRUMENT, hk_title, hk_out)
            refresh_legacy_aliases(QUICKLOOK_DIR, INSTRUMENT, day_png=hk_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Aurora Power Supply summary and housekeeping quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
