#!/usr/bin/env python3
"""Render Vaisala met variables as stacked time-series PNGs."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ZARR_DEFAULT = Path("/data/aurora/products/vaisalamet/vaisalamet.zarr")
OUTPUT_DEFAULT = Path("latest_vaisalamet.png")
MAX_TIME_SAMPLES = 2200


def numeric_time_vars(ds: xr.Dataset) -> list[str]:
    names: list[str] = []
    for name, da in ds.data_vars.items():
        if da.dims != ("time",):
            continue
        if np.issubdtype(da.dtype, np.number):
            names.append(name)
    return names


def _downsample_time(ds: xr.Dataset, max_time_samples: int = MAX_TIME_SAMPLES) -> xr.Dataset:
    if "time" not in ds:
        return ds
    if ds.sizes.get("time", 0) > max_time_samples:
        step = int(np.ceil(ds.sizes["time"] / max_time_samples))
        ds = ds.isel(time=slice(None, None, step))
    return ds


def plot_timeseries(ds: xr.Dataset, title: str, output: Path) -> None:
    ds = _downsample_time(ds)
    times = pd.to_datetime(ds["time"].values) if "time" in ds else pd.DatetimeIndex([])
    names = numeric_time_vars(ds)
    if len(times) == 0 or not names:
        raise ValueError("No numeric Vaisala met time-series variables to plot")

    height = max(8.0, min(34.0, 1.15 * len(names)))
    fig, axes = plt.subplots(len(names), 1, figsize=(13, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = ["#0b7285", "#c92a2a", "#2b8a3e", "#5f3dc4", "#e67700", "#087f5b", "#364fc7", "#a61e4d"]
    for idx, (ax, name) in enumerate(zip(axes, names)):
        values = np.asarray(ds[name].values, dtype=np.float64)
        ax.plot(times, values, color=colors[idx % len(colors)], linewidth=0.8)
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        ax.grid(True, color="#d9d9d9", linewidth=0.4)
        ax.tick_params(axis="y", labelsize=7)

    span_hours = max((times.max() - times.min()) / np.timedelta64(1, "h"), 1.0)
    interval = 2 if span_hours <= 36 else 6
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(left=0.21, bottom=0.08, top=0.96, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_last_24h(zarr_path: Path, output: Path) -> None:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    plot_timeseries(window, "vaisalamet - Latest 24 hours", output)


def main() -> None:
    zarr_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else ZARR_DEFAULT
    output = Path(sys.argv[2]) if len(sys.argv) >= 3 else OUTPUT_DEFAULT
    plot_last_24h(zarr_path, output)


if __name__ == "__main__":
    main()
