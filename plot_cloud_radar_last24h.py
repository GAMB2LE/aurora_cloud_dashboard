#!/usr/bin/env python3
"""
Render the last 24 hours of cloud radar data to a multi-panel PNG for the dashboard.
Panels (top→bottom): ZE_dBZ, MeanVel, SpecWidth, SLDR, RHV, SRCX, Skew, Kurt.
"""

from __future__ import annotations

import sys
from datetime import timedelta
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from extra_housekeeping import (
    extra_housekeeping_latest_png,
    load_cloud_radar_housekeeping_from_raw,
    plot_cloud_radar_housekeeping,
)
from quicklook_time_axis import apply_quicklook_time_axis

ZARR_DEFAULT = Path("/data/aurora/products/rpgfmcw94/cloud_radar.zarr")
RAW_ROOT_DEFAULT = Path(os.environ.get("CLOUD_RADAR_RAW_ROOT", "/project/aurora/raw/rpgfmcw94"))
OUTPUT_DEFAULT = Path("/data/aurora/products/quicklooks/cloud_radar/latest.png")

# Limits aligned with dashboard defaults
ZE_VMIN, ZE_VMAX = -30.0, 10.0
VEL_VMIN, VEL_VMAX = -5.0, 5.0
SPEC_VMIN, SPEC_VMAX = 0.0, 3.0
SLDR_VMIN, SLDR_VMAX = -100.0, -10.0
RHV_VMIN, RHV_VMAX = 0.8, 1.0
SRCX_VMIN, SRCX_VMAX = 0.8, 1.0
SKEW_VMIN, SKEW_VMAX = -2.0, 2.0
KURT_VMIN, KURT_VMAX = 0.0, 8.0
RANGE_MAX = 9000


def _write_no_data_png(output: Path, start_time: pd.Timestamp, end_time: pd.Timestamp, latest_time: pd.Timestamp | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_text = "No samples are present in the selected 24 hour window."
    if latest_time is not None:
        latest_text = f"Latest radar sample in the Zarr is {latest_time:%Y-%m-%d %H:%M UTC}."

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()
    ax.text(
        0.5,
        0.62,
        "No cloud radar data in the last 24 hours",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#22313f",
    )
    ax.text(
        0.5,
        0.44,
        f"Window: {start_time:%Y-%m-%d %H:%M UTC} to {end_time:%Y-%m-%d %H:%M UTC}",
        ha="center",
        va="center",
        fontsize=11,
        color="#5f6c7b",
    )
    ax.text(
        0.5,
        0.32,
        latest_text,
        ha="center",
        va="center",
        fontsize=11,
        color="#5f6c7b",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote no-data placeholder {output}")


def plot_last_24h(zarr_path: Path, output: Path):
    ds = xr.open_zarr(zarr_path, chunks={})
    needed = ["ZE_dBZ", "MeanVel", "SpecWidth", "SLDR", "RHV", "SRCX", "Skew", "Kurt"]
    missing = [v for v in needed if v not in ds]
    if missing:
        raise KeyError(f"Dataset missing variables: {', '.join(missing)}")

    time_index = pd.DatetimeIndex(ds["time"].values)
    end_time = pd.Timestamp.utcnow().replace(tzinfo=None)
    start_time = end_time - timedelta(hours=24)

    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        latest_time = time_index.max() if len(time_index) else None
        _write_no_data_png(output, start_time, end_time, latest_time)
        hk_output = extra_housekeeping_latest_png(output.parent, "Cloud Radar")
        if hk_output is not None:
            _write_no_data_png(hk_output, start_time, end_time, latest_time)
        return
    window = ds.isel(time=mask).sortby("time")
    window = window.sel({"range": slice(0, RANGE_MAX)})

    fig, axes = plt.subplots(8, 1, figsize=(12, 20), sharex=True, sharey=True)
    vars_titles = [
        ("ZE_dBZ", "ZE (dBZ)", ZE_VMIN, ZE_VMAX, "ZE (dBZ)", "cividis"),
        ("MeanVel", "Mean Velocity", VEL_VMIN, VEL_VMAX, "Velocity (m/s)", "RdBu_r"),
        ("SpecWidth", "Spectrum Width (m/s)", SPEC_VMIN, SPEC_VMAX, "Spec Width (m/s)", "plasma"),
        ("SLDR", "SLDR (dB)", SLDR_VMIN, SLDR_VMAX, "SLDR (dB)", "RdBu_r"),
        ("RHV", "RHV", RHV_VMIN, RHV_VMAX, "RHV", "viridis"),
        ("SRCX", "SRCX", SRCX_VMIN, SRCX_VMAX, "SRCX", "viridis"),
        ("Skew", "Skew", SKEW_VMIN, SKEW_VMAX, "Skew", "RdBu_r"),
        ("Kurt", "Kurtosis", KURT_VMIN, KURT_VMAX, "Kurtosis", "magma"),
    ]

    for ax, (var, title, vmin, vmax, cbar_label, cmap) in zip(axes, vars_titles):
        da = window[var]
        data = da.transpose("time", "range").values
        mesh = ax.pcolormesh(
            da["time"].values,
            da["range"].values,
            data.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_ylabel("range (m)")
        ax.set_title(title)
        ax.set_ylim(0, RANGE_MAX)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label(cbar_label)

    window_times = pd.DatetimeIndex(window["time"].values)
    for ax in axes:
        apply_quicklook_time_axis(ax, window_times, label_rotation=0, label_size=9)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")

    fig.suptitle("Cloud Radar – Last 24 hours")
    fig.tight_layout()
    fig.subplots_adjust(top=0.96, bottom=0.11, hspace=0.18)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")
    hk_output = extra_housekeeping_latest_png(output.parent, "Cloud Radar")
    if hk_output is not None:
        hk_window = load_cloud_radar_housekeeping_from_raw(RAW_ROOT_DEFAULT, start_time, end_time)
        if hk_window.sizes.get("time", 0) >= 2:
            plot_cloud_radar_housekeeping(hk_window, "HK_Radar - Latest 24 hours", hk_output)
        else:
            latest_time = time_index.max() if len(time_index) else None
            _write_no_data_png(hk_output, start_time, end_time, latest_time)


def main():
    zarr_path = ZARR_DEFAULT
    output = OUTPUT_DEFAULT
    if len(sys.argv) >= 2:
        zarr_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output = Path(sys.argv[2])
    plot_last_24h(zarr_path, output)


if __name__ == "__main__":
    main()
