#!/usr/bin/env python3
"""Render the latest HATPRO radiometer Zarr data to a science quicklook PNG."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from quicklook_time_axis import apply_quicklook_time_axis

ZARR_DEFAULT = Path("/data/aurora/products/hatprog5/hatpro.zarr")
OUTPUT_DEFAULT = Path("/data/aurora/products/quicklooks/hatpro/latest.png")
RANGE_MAX = 10_000.0
MAX_TIME_SAMPLES = 2400
MAX_PROFILE_SAMPLES = 2400


def _thin_time(ds: xr.Dataset) -> xr.Dataset:
    count = int(ds.sizes.get("time", 0))
    if count <= MAX_TIME_SAMPLES:
        return ds
    stride = int(np.ceil(count / MAX_TIME_SAMPLES))
    return ds.isel(time=slice(None, None, stride))


def _valid_profile_window(window: xr.Dataset, name: str = "T_PROF") -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] | None:
    """Return valid profile columns without thinning on unrelated 1D samples."""
    if name not in window or "time" not in window[name].dims:
        return None
    profile = window[name]
    if "range" in profile.dims:
        profile = profile.transpose("time", "range")
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 2:
        return None
    valid_columns = np.isfinite(values).any(axis=1)
    if not np.any(valid_columns):
        return None
    profile = profile.isel(time=valid_columns)
    count = int(profile.sizes.get("time", 0))
    if count > MAX_PROFILE_SAMPLES:
        stride = int(np.ceil(count / MAX_PROFILE_SAMPLES))
        profile = profile.isel(time=slice(None, None, stride))
    times = pd.DatetimeIndex(profile["time"].values)
    heights = np.asarray(profile["range"].values if "range" in profile.coords else np.arange(profile.shape[1]), dtype=np.float64)
    temps = np.asarray(profile.transpose("range", "time"), dtype=np.float64)
    return times, heights, temps


def _write_no_data_png(output: Path, title: str, detail: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=18, fontweight="bold", color="#22313f")
    ax.text(0.5, 0.42, detail, ha="center", va="center", fontsize=11, color="#5f6c7b")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote no-data placeholder {output}")


def _plot_hatpro(
    window: xr.Dataset,
    title: str,
    output: Path,
    *,
    x_start: pd.Timestamp | None = None,
    x_end: pd.Timestamp | None = None,
) -> None:
    full_window = window.sortby("time")
    if "range" in full_window.coords:
        full_window = full_window.sel(range=slice(0, RANGE_MAX))
    full_times = pd.DatetimeIndex(full_window["time"].values)
    if len(full_times) == 0:
        _write_no_data_png(output, "No HATPRO data", "The selected HATPRO window has no time samples.")
        return
    x_start = pd.Timestamp(x_start) if x_start is not None else pd.Timestamp(full_times.min())
    x_end = pd.Timestamp(x_end) if x_end is not None else pd.Timestamp(full_times.max())
    if x_end <= x_start:
        x_end = x_start + pd.Timedelta(minutes=1)

    line_window = _thin_time(full_window)
    times = pd.DatetimeIndex(line_window["time"].values)
    profile_window = _valid_profile_window(full_window)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    if "LWP" in line_window:
        ax.plot(times, np.asarray(line_window["LWP"]), color="#1f77b4", lw=1.4, label="LWP")
    ax.set_ylabel("LWP (g m$^{-2}$)", color="#1f77b4")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax2 = ax.twinx()
    if "IWV" in line_window:
        ax2.plot(times, np.asarray(line_window["IWV"]), color="#2ca02c", lw=1.4, ls=":", label="IWV")
    ax2.set_ylabel("IWV (kg m$^{-2}$)", color="#2ca02c")
    ax2.tick_params(axis="y", colors="#2ca02c")
    ax.set_title("LWP / IWV", loc="left", bbox=dict(facecolor="white", edgecolor="#22313f", boxstyle="square,pad=0.2"))

    ax = axes[1]
    if "IRR_Map" in line_window:
        ax.plot(times, np.asarray(line_window["IRR_Map"]), color="#d62728", lw=1.4, label="IRR")
    if "SURF_T" in line_window:
        ax.plot(times, np.asarray(line_window["SURF_T"]) - 273.15, color="#9467bd", lw=1.2, ls="--", label="SURF_T")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title("Surface Temperature", loc="left", bbox=dict(facecolor="white", edgecolor="#22313f", boxstyle="square,pad=0.2"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax = axes[2]
    profile_mesh = None
    if profile_window is not None:
        profile_times, heights, temps = profile_window
        profile_mesh = ax.pcolormesh(
            profile_times,
            heights,
            temps,
            shading="auto",
            vmin=210.0,
            vmax=310.0,
            cmap="inferno",
        )
        if len(profile_times) >= 2 and np.isfinite(temps).any():
            ax.contour(
                profile_times,
                heights,
                temps,
                levels=np.arange(220.0, 311.0, 10.0),
                colors="white",
                linewidths=0.45,
                alpha=0.75,
            )
    else:
        ax.text(0.5, 0.5, "No temperature profile samples", transform=ax.transAxes, ha="center", va="center", color="#5f6c7b")
    ax.set_ylim(0, RANGE_MAX)
    ax.set_ylabel("Height (m)")
    ax.set_title("Temperature Profile", loc="left", bbox=dict(facecolor="white", edgecolor="#22313f", boxstyle="square,pad=0.2"))

    for ax in axes:
        ax.set_xlim(x_start, x_end)
        ax.grid(True, color="#e5eaef", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_color("#c5d0da")
        ax.spines["left"].set_color("#c5d0da")
        ax.spines["bottom"].set_color("#c5d0da")

    apply_quicklook_time_axis(axes[-1], pd.DatetimeIndex([x_start, x_end]), label_rotation=0, label_size=9)

    fig.suptitle(title, fontsize=14, color="#22313f")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.08, right=0.86)
    if profile_mesh is not None:
        profile_bbox = axes[2].get_position()
        cax = fig.add_axes([profile_bbox.x1 + 0.012, profile_bbox.y0, 0.018, profile_bbox.height])
        cbar = fig.colorbar(profile_mesh, cax=cax)
        cbar.set_label("Temperature (K)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_window(zarr_path: Path, output: Path, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> None:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        _write_no_data_png(output, "No HATPRO data", "The HATPRO Zarr has no time samples.")
        return

    if end is None:
        end = time_index.max()
    if start is None:
        start = end - timedelta(hours=24)
    mask = (time_index >= start) & (time_index <= end)
    if not mask.any():
        _write_no_data_png(
            output,
            "No HATPRO data in selected window",
            f"Window: {start:%Y-%m-%d %H:%M UTC} to {end:%Y-%m-%d %H:%M UTC}",
        )
        return
    window = ds.isel(time=mask)
    title = f"Scanning Microwave Radiometer - {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M UTC}"
    _plot_hatpro(window, title, output, x_start=start, x_end=end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HATPRO latest 24 hour science quicklook")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    plot_window(args.zarr_path, args.output)


if __name__ == "__main__":
    main()
