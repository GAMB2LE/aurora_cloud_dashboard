#!/usr/bin/env python3
"""Render grouped power time-series PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from grouped_timeseries import (
    default_calendar_label,
    plot_grouped_timeseries,
    plot_last_24h,
)

INSTRUMENT = "power"
ZARR_DEFAULT = Path("/data/aurora/products/power/power.zarr")
OUTPUT_DEFAULT = Path("latest_power.png")


def plot_timeseries(ds: xr.Dataset, title: str, output: Path, group: str | None = None) -> None:
    plot_grouped_timeseries(ds, instrument=INSTRUMENT, selection=group, title=title, output=output)


def plot_last_24h_group(zarr_path: Path, output: Path, group: str | None = None) -> None:
    plot_last_24h(zarr_path, output, instrument=INSTRUMENT, selection=group)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render grouped power latest-24h PNGs")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--group", default=default_calendar_label(INSTRUMENT), help="Display label or group key to render")
    args = parser.parse_args()
    plot_last_24h_group(args.zarr_path, args.output, group=args.group)


if __name__ == "__main__":
    main()
