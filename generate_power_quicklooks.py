#!/usr/bin/env python3
"""Generate Aurora Power Supply summary and housekeeping quicklook PNGs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from datetime import timedelta
from pathlib import Path

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from grouped_timeseries import (
    POWER_SOC_FORECAST_FIELDS,
    clear_generated_quicklooks,
    build_summary_plotly,
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
from generate_power_display_summary import POWER_DISPLAY_SUMMARY_ZARR_PATH, generate as generate_power_display_summary

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
ASFS_LOGGER_ZARR_PATH = Path(os.environ.get("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"))
QUICKLOOK_DIR = Path(os.environ.get("POWER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "power"))
PREWARM_DIR = Path(os.environ.get("AURORA_INTERACTIVE_PREWARM_DIR", "/data/aurora/products/dashboard/prewarm"))
PREWARM_JSON = PREWARM_DIR / "power_latest_interactive.json"
PREWARM_CURRENT_JSON = PREWARM_DIR / "power_current_latest_interactive.json"
PREWARM_FORECAST_JSON = PREWARM_DIR / "power_forecast_latest_interactive.json"
PREWARM_METADATA_JSON = PREWARM_DIR / "power_prewarms_metadata.json"
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
    forecast_names = [name for name in POWER_SOC_FORECAST_FIELDS if name in ds]
    if forecast_names:
        forecast_valid = np.zeros(len(time_index), dtype=bool)
        for name in forecast_names:
            forecast_valid |= np.isfinite(np.asarray(ds[name].values, dtype=np.float64))
        horizon = end + pd.Timedelta(hours=float(os.environ.get("AURORA_POWER_SOC_FORECAST_HOURS", "96")))
        mask |= forecast_valid & (time_index >= start) & (time_index <= horizon)
    if not mask.any():
        return None
    return ds.isel(time=mask).sortby("time")


def _with_display_window(ds: xr.Dataset, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    """Mark a context-loaded dataset so summary prep crops to the visible window."""
    ds = ds.copy(deep=False)
    ds.attrs[SUMMARY_DISPLAY_START_ATTR] = pd.Timestamp(start).isoformat()
    ds.attrs[SUMMARY_DISPLAY_END_ATTR] = pd.Timestamp(end).isoformat()
    return ds


def _optional_display_summary_dataset(*, refresh: bool = True) -> xr.Dataset | None:
    """Return the compact Power display product used by summary panels."""
    if refresh:
        try:
            generate_power_display_summary()
        except Exception as exc:
            print(f"Could not refresh Power display-summary Zarr: {exc}")
    if not POWER_DISPLAY_SUMMARY_ZARR_PATH.exists():
        return None
    try:
        return xr.open_zarr(POWER_DISPLAY_SUMMARY_ZARR_PATH, chunks={})
    except Exception as exc:
        print(f"Could not open Power display-summary Zarr: {exc}")
        return None


def _summary_inputs(
    raw_power_window: xr.Dataset,
    display_summary: xr.Dataset | None,
    ass_power: xr.Dataset | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[xr.Dataset | None, ...]:
    """Prefer the compact display summary and fall back to raw/context inputs."""
    display_window = _slice_window(display_summary, start, end)
    if display_window is not None and display_window.sizes.get("time", 0) >= 2:
        return (display_window,)
    return (
        raw_power_window,
        _slice_window(ass_power, start, end),
    )


def _atomic_write_figure_json(fig, path: Path) -> None:
    """Publish complete Plotly JSON only after its temporary file is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.write_json(temporary)
    temporary.replace(path)


def _write_prewarm_json_worker(section: str, path_text: str, figure_spec: dict) -> tuple[str, str]:
    """Serialize one immutable Plotly payload in an offline worker process."""
    # The worker only receives a plain figure specification and returns an
    # atomically published file path. It never opens a Zarr store or touches a
    # live Panel document.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    path = Path(path_text)
    _atomic_write_figure_json(go.Figure(figure_spec), path)
    return section, str(path)


def _latest_prewarm_window(time_index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the live 24-hour browser window, ending at current UTC time.

    The observed series may end before now. Keeping that empty tail is deliberate:
    it makes a collection gap visible instead of silently shifting the plot back
    to the latest available sample.
    """
    observed_end = pd.Timestamp(time_index.max())
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    end = max(now, observed_end)
    return end - timedelta(hours=24), end


def generate_latest_prewarms(
    ds: xr.Dataset,
    display_summary: xr.Dataset | None,
    ass_power: xr.Dataset | None,
) -> dict[str, Path]:
    """Build the browser's Current and Forecast Power figures in the background."""
    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    start_time, end_time = _latest_prewarm_window(time_index)
    latest_day = _slice_window(ds, start_time, end_time)
    if latest_day is None or latest_day.sizes.get("time", 0) < 2:
        raise ValueError("Dataset has fewer than two samples in the latest Power window")

    latest_summary = combine_summary_datasets(
        INSTRUMENT,
        *_summary_inputs(latest_day, display_summary, ass_power, start_time, end_time),
    )
    latest_summary = _with_display_window(latest_summary, start_time, end_time)
    figures = {
        "all": (PREWARM_JSON, {}),
        "current": (PREWARM_CURRENT_JSON, {"observed"}),
        "forecast": (PREWARM_FORECAST_JSON, {"forecast_24h", "forecast_96h", "verification"}),
    }
    jobs: list[tuple[str, Path, dict]] = []
    for section, (path, panel_groups) in figures.items():
        fig = build_summary_plotly(
            latest_summary,
            INSTRUMENT,
            title="Aurora Power Supply",
            max_time_samples=700,
            panel_groups=panel_groups or None,
        )
        jobs.append((section, path, fig.to_plotly_json()))

    worker_count = max(1, min(int(os.environ.get("AURORA_POWER_PREWARM_WORKERS", "2")), 2))
    written: dict[str, Path] = {}
    if worker_count == 1:
        results = [_write_prewarm_json_worker(section, str(path), spec) for section, path, spec in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    _write_prewarm_json_worker,
                    (section for section, _path, _spec in jobs),
                    (str(path) for _section, path, _spec in jobs),
                    (spec for _section, _path, spec in jobs),
                )
            )
    for section, path_text in results:
        path = Path(path_text)
        written[section] = path
        print(f"Wrote {path}")

    metadata = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "display_start_utc": start_time.tz_localize("UTC").isoformat(),
        "display_end_utc": end_time.tz_localize("UTC").isoformat(),
        "observed_latest_utc": pd.Timestamp(time_index.max()).tz_localize("UTC").isoformat(),
        "sections": {section: str(path) for section, path in written.items()},
    }
    temporary = PREWARM_METADATA_JSON.with_suffix(PREWARM_METADATA_JSON.suffix + ".tmp")
    PREWARM_METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(PREWARM_METADATA_JSON)
    return written


def main(force: bool = False, *, prewarm_only: bool = False, skip_display_refresh: bool = False) -> None:
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    ass_power = _optional_ass_power_dataset()
    display_summary = _optional_display_summary_dataset(refresh=not skip_display_refresh)

    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    if prewarm_only:
        generate_latest_prewarms(ds, display_summary, ass_power)
        return
    today = pd.Timestamp.utcnow().date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        clear_generated_quicklooks(QUICKLOOK_DIR, INSTRUMENT)
        print("Deleted existing Aurora Power Supply quicklook PNGs.")

    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    latest_mask = (time_index >= start_time) & (time_index <= end_time)
    latest_day = ds.isel(time=latest_mask).sortby("time")
    if latest_day.sizes.get("time", 0) >= 2:
        latest_summary = combine_summary_datasets(
            INSTRUMENT,
            *_summary_inputs(
                latest_day,
                display_summary,
                ass_power,
                pd.Timestamp(start_time),
                pd.Timestamp(end_time),
            ),
        )
        latest_summary = _with_display_window(latest_summary, pd.Timestamp(start_time), pd.Timestamp(end_time))
        summary_out = summary_latest_png(QUICKLOOK_DIR, INSTRUMENT)
        save_summary_png(latest_summary, INSTRUMENT, "Aurora Power Supply - Latest 24 hours", summary_out)
        generate_latest_prewarms(ds, display_summary, ass_power)
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
        summary_out = summary_daily_png(QUICKLOOK_DIR, INSTRUMENT, day)
        if force or not summary_out.exists():
            summary_day = combine_summary_datasets(
                INSTRUMENT,
                *_summary_inputs(ds_day, display_summary, ass_power, start, end),
            )
            summary_day = _with_display_window(summary_day, start, end)
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
    parser.add_argument("--prewarm-only", action="store_true", help="Generate only the Current and Forecast browser prewarms")
    parser.add_argument("--skip-display-refresh", action="store_true", help="Reuse an already refreshed compact Power display product")
    args = parser.parse_args()
    main(force=args.force, prewarm_only=args.prewarm_only, skip_display_refresh=args.skip_display_refresh)
