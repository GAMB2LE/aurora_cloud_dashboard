#!/usr/bin/env python3
"""Render ASFS housekeeping latest-24h PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from grouped_timeseries import plot_housekeeping_last_24h, plot_housekeeping_timeseries

INSTRUMENT = "asfs-logger"
ZARR_DEFAULT = Path("/data/aurora/products/asfs_logger/asfs_logger.zarr")
OUTPUT_DEFAULT = Path("latest_asfs_logger.png")


def plot_timeseries(ds: xr.Dataset, title: str, output: Path) -> None:
    plot_housekeeping_timeseries(ds, instrument=INSTRUMENT, title=title, output=output)


def plot_last_24h_group(zarr_path: Path, output: Path) -> None:
    plot_housekeeping_last_24h(zarr_path, output, instrument=INSTRUMENT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ASFS housekeeping latest-24h PNGs")
    parser.add_argument("zarr_path", nargs="?", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    plot_last_24h_group(args.zarr_path, args.output)


if __name__ == "__main__":
    main()
