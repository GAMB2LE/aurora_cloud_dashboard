#!/usr/bin/env python3
"""
Render the last 24 hours of cloud radar data to a compact multi-panel PNG.
Panels (top→bottom): ZE_dBZ, MeanVel, SpecWidth, SLDR, RHV, SRCX, Skew, Kurt.
"""

from __future__ import annotations

import sys
from datetime import timedelta
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from extra_housekeeping import (
    extra_housekeeping_latest_png,
    load_cloud_radar_housekeeping_from_raw,
    plot_cloud_radar_housekeeping,
)
from quicklook_time_axis import apply_quicklook_time_axis
from time_gap_breaks import insert_time_gap_breaks

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

RADAR_PANELS = (
    ("ZE_dBZ", "ZE", ZE_VMIN, ZE_VMAX, "ZE (dBZ)", "cividis"),
    ("MeanVel", "Mean Velocity", VEL_VMIN, VEL_VMAX, "Mean Velocity (m/s)", "RdBu_r"),
    ("SpecWidth", "Spectrum Width", SPEC_VMIN, SPEC_VMAX, "Spectrum Width (m/s)", "plasma"),
    ("SLDR", "SLDR", SLDR_VMIN, SLDR_VMAX, "SLDR (dB)", "RdBu_r"),
    ("RHV", "RHV", RHV_VMIN, RHV_VMAX, "RHV", "viridis"),
    ("SRCX", "SRCX", SRCX_VMIN, SRCX_VMAX, "SRCX", "viridis"),
    ("Skew", "Skew", SKEW_VMIN, SKEW_VMAX, "Skew", "RdBu_r"),
    ("Kurt", "Kurtosis", KURT_VMIN, KURT_VMAX, "Kurtosis", "magma"),
)


def required_radar_vars() -> tuple[str, ...]:
    return tuple(panel[0] for panel in RADAR_PANELS)


def _validate_radar_vars(ds: xr.Dataset) -> None:
    missing = [var for var in required_radar_vars() if var not in ds]
    if missing:
        raise KeyError(f"Dataset missing variables: {', '.join(missing)}")


def _axis_times(window_times: pd.DatetimeIndex, x_start: pd.Timestamp, x_end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([x_start, *window_times.to_pydatetime(), x_end])


def plot_radar_quicklook(
    window: xr.Dataset,
    title: str,
    output: Path,
    *,
    x_start: pd.Timestamp | None = None,
    x_end: pd.Timestamp | None = None,
) -> None:
    """Render the shared compact science quicklook layout for radar windows."""
    _validate_radar_vars(window)
    window = window.sortby("time").sel({"range": slice(0, RANGE_MAX)})
    window_times = pd.DatetimeIndex(window["time"].values)
    if len(window_times) < 2:
        raise ValueError("At least two radar time samples are required for a science quicklook.")

    x_start = pd.Timestamp(window_times.min() if x_start is None else x_start)
    x_end = pd.Timestamp(window_times.max() if x_end is None else x_end)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(RADAR_PANELS), 1, figsize=(12, 15.4), sharex=True, sharey=True)
    if len(RADAR_PANELS) == 1:
        axes = [axes]

    colorbars = []
    for ax, (var, panel_title, vmin, vmax, cbar_label, cmap) in zip(axes, RADAR_PANELS):
        da = window[var].transpose("time", "range")
        plot_times, plot_data = insert_time_gap_breaks(da["time"].values, da.values.T, time_axis=1)
        mesh = ax.pcolormesh(
            plot_times,
            da["range"].values,
            plot_data,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_ylabel("Range (m)", fontsize=8)
        ax.set_ylim(0, RANGE_MAX)
        ax.set_xlim(x_start, x_end)
        ax.grid(True, color="#e3e8ee", linewidth=0.45)
        ax.tick_params(axis="y", labelsize=8)
        ax.text(
            0.01,
            0.93,
            panel_title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#22313f",
            bbox={"facecolor": "white", "edgecolor": "#22313f", "boxstyle": "square,pad=0.18", "linewidth": 0.8},
        )
        colorbars.append((mesh, cbar_label, ax))

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")
    apply_quicklook_time_axis(
        axes[-1],
        _axis_times(window_times, x_start, x_end),
        label_rotation=0,
        label_size=8,
    )

    fig.suptitle(title, y=0.992, fontsize=13, color="#22313f")
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.055, top=0.965, hspace=0.08)

    for mesh, cbar_label, ax in colorbars:
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + 0.012, bbox.y0, 0.012, bbox.height])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7, length=2)

    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


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
    _validate_radar_vars(ds)

    time_index = pd.DatetimeIndex(ds["time"].values)
    end_time = pd.Timestamp.utcnow().replace(tzinfo=None)
    start_time = end_time - timedelta(hours=24)

    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any() or int(mask.sum()) < 2:
        latest_time = time_index.max() if len(time_index) else None
        _write_no_data_png(output, start_time, end_time, latest_time)
        hk_output = extra_housekeeping_latest_png(output.parent, "Cloud Radar")
        if hk_output is not None:
            _write_no_data_png(hk_output, start_time, end_time, latest_time)
        return
    window = ds.isel(time=mask).sortby("time")
    window = window.sel({"range": slice(0, RANGE_MAX)})

    plot_radar_quicklook(window, "Cloud Radar - Latest 24 hours", output, x_start=start_time, x_end=end_time)
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
