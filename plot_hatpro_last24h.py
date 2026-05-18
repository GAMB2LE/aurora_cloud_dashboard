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


def _thin_time(ds: xr.Dataset) -> xr.Dataset:
    count = int(ds.sizes.get("time", 0))
    if count <= MAX_TIME_SAMPLES:
        return ds
    stride = int(np.ceil(count / MAX_TIME_SAMPLES))
    return ds.isel(time=slice(None, None, stride))


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


def _plot_hatpro(window: xr.Dataset, title: str, output: Path) -> None:
    window = _thin_time(window.sortby("time"))
    if "range" in window.coords:
        window = window.sel(range=slice(0, RANGE_MAX))
    times = pd.DatetimeIndex(window["time"].values)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    if "LWP" in window:
        ax.plot(times, np.asarray(window["LWP"]), color="#0b7285", lw=1.4, label="LWP")
    ax.set_ylabel("LWP (g m$^{-2}$)", color="#0b7285")
    ax.tick_params(axis="y", colors="#0b7285")
    ax2 = ax.twinx()
    if "IWV" in window:
        ax2.plot(times, np.asarray(window["IWV"]), color="#4f8c63", lw=1.4, label="IWV")
    ax2.set_ylabel("IWV (kg m$^{-2}$)", color="#4f8c63")
    ax2.tick_params(axis="y", colors="#4f8c63")
    ax.set_title("LWP / IWV", loc="left", bbox=dict(facecolor="white", edgecolor="#22313f", boxstyle="square,pad=0.2"))

    ax = axes[1]
    if "IRR_Map" in window:
        ax.plot(times, np.asarray(window["IRR_Map"]), color="#7768b8", lw=1.4, label="Infrared surface")
    if "SURF_T" in window:
        ax.plot(times, np.asarray(window["SURF_T"]) - 273.15, color="#4d6fb3", lw=1.2, ls="--", label="Surface met")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title("Surface Temperature", loc="left", bbox=dict(facecolor="white", edgecolor="#22313f", boxstyle="square,pad=0.2"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax = axes[2]
    if "T_PROF" in window:
        profile = window["T_PROF"].transpose("range", "time")
        mesh = ax.pcolormesh(
            times,
            np.asarray(profile["range"]),
            np.asarray(profile),
            shading="auto",
            vmin=210.0,
            vmax=310.0,
            cmap="inferno",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label("Temperature (K)")
    ax.set_ylim(0, RANGE_MAX)
    ax.set_ylabel("Height (m)")
    ax.set_title("Temperature Profile", loc="left", bbox=dict(facecolor="white", edgecolor="#22313f", boxstyle="square,pad=0.2"))

    for ax in axes:
        ax.grid(True, color="#e5eaef", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_color("#c5d0da")
        ax.spines["left"].set_color("#c5d0da")
        ax.spines["bottom"].set_color("#c5d0da")

    apply_quicklook_time_axis(axes[-1], times, label_rotation=0, label_size=9)

    fig.suptitle(title, fontsize=14, color="#22313f")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.08)
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
    _plot_hatpro(window, title, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HATPRO latest 24 hour science quicklook")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    plot_window(args.zarr_path, args.output)


if __name__ == "__main__":
    main()
