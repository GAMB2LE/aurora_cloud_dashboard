#!/usr/bin/env python3
"""Generate Operations summary and housekeeping quicklook PNGs."""

from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path

import pandas as pd
import xarray as xr

from grouped_timeseries import (
    clear_generated_quicklooks,
    housekeeping_daily_png,
    housekeeping_label,
    housekeeping_latest_png,
    refresh_legacy_aliases,
    save_summary_png,
    summary_daily_png,
    summary_latest_png,
)


APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("OPS_MONITOR_ZARR_PATH", "/data/aurora/products/ops_monitor/ops_monitor.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("OPS_MONITOR_QUICKLOOK_DIR", QUICKLOOK_ROOT / "ops_monitor"))
INSTRUMENT = "ops-monitor"
OPS_HK_LAYOUT_KEY = "ops-monitor-hk"
OPS_STREAMS = (
    "cl61",
    "radar",
    "vaisalamet",
    "asfs_logger",
    "asfs_fast_sonic",
    "power",
    "wxcam",
)


def _ops_housekeeping_dataset(ds: xr.Dataset) -> xr.Dataset:
    hk = ds.copy()
    for stream in OPS_STREAMS:
        local_missing = hk.get(f"{stream}_local_missing_count")
        local_mismatch = hk.get(f"{stream}_local_mismatch_count")
        gws_missing = hk.get(f"{stream}_gws_missing_count")
        gws_mismatch = hk.get(f"{stream}_gws_mismatch_count")
        if local_missing is not None and local_mismatch is not None:
            hk[f"{stream}_local_issue_count"] = local_missing.fillna(0.0) + local_mismatch.fillna(0.0)
        if gws_missing is not None and gws_mismatch is not None:
            hk[f"{stream}_gws_issue_count"] = gws_missing.fillna(0.0) + gws_mismatch.fillna(0.0)

    if "source_sync_enabled_count" in hk and "streams_product_gate_ok_count" in hk:
        hk["streams_product_gate_block_count"] = (
            hk["source_sync_enabled_count"].fillna(0.0) - hk["streams_product_gate_ok_count"].fillna(0.0)
        ).clip(min=0.0)
    if "source_sync_enabled_count" in hk and "streams_prune_ready_count" in hk:
        hk["streams_prune_block_count"] = (
            hk["source_sync_enabled_count"].fillna(0.0) - hk["streams_prune_ready_count"].fillna(0.0)
        ).clip(min=0.0)

    for healthy_name, problem_name in (
        ("gws_available_state", "gws_unavailable_state"),
        ("mirror_verify_service_healthy_state", "mirror_verify_problem_state"),
        ("ops_monitor_append_service_healthy_state", "ops_monitor_append_problem_state"),
        ("ops_monitor_quicklooks_service_healthy_state", "ops_monitor_quicklooks_problem_state"),
        ("dashboard_perf_log_recent_state", "dashboard_perf_log_stale_state"),
    ):
        if healthy_name in hk:
            hk[problem_name] = 1.0 - hk[healthy_name].fillna(1.0)
    return hk


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
        clear_generated_quicklooks(QUICKLOOK_DIR, INSTRUMENT)
        print("Deleted existing Operations quicklook PNGs.")

    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_mask = (time_index >= start_time) & (time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    if latest_day.sizes.get("time", 0) >= 2:
        summary_out = summary_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        save_summary_png(latest_day, INSTRUMENT, "Operations - Latest 24 hours", summary_out)
        hk_out = housekeeping_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        if hk_out is not None:
            hk_title = f"{housekeeping_label(INSTRUMENT)} - Latest 24 hours"
            save_summary_png(_ops_housekeeping_dataset(latest_day), OPS_HK_LAYOUT_KEY, hk_title, hk_out)

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
            title = pd.Timestamp(day).strftime("Operations - %Y-%m-%d")
            save_summary_png(ds_day, INSTRUMENT, title, summary_out)
        hk_out = housekeeping_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if hk_out is not None and (force or not hk_out.exists()):
            hk_title = pd.Timestamp(day).strftime(f"{housekeeping_label(INSTRUMENT)} - %Y-%m-%d")
            save_summary_png(_ops_housekeeping_dataset(ds_day), OPS_HK_LAYOUT_KEY, hk_title, hk_out)
            refresh_legacy_aliases(QUICKLOOK_DIR, INSTRUMENT, day_png=hk_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Operations summary and housekeeping quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
