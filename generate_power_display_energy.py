#!/usr/bin/env python3
"""Generate compact Aurora Power Supply display-energy Zarr products."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import xarray as xr

from grouped_timeseries import (
    POWER_DISPLAY_ENERGY_FREQ,
    build_power_display_energy_dataset,
)

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
POWER_DISPLAY_ENERGY_ZARR_PATH = Path(
    os.environ.get("POWER_DISPLAY_ENERGY_ZARR_PATH", "/data/aurora/products/power/power_display_energy.zarr")
)


def generate(power_zarr: Path = POWER_ZARR_PATH, output_zarr: Path = POWER_DISPLAY_ENERGY_ZARR_PATH, freq: str = POWER_DISPLAY_ENERGY_FREQ) -> Path:
    """Build the derived one-minute display-energy store from the raw Power Zarr."""
    ds = xr.open_zarr(power_zarr, chunks={})
    display = build_power_display_energy_dataset(ds, freq=freq)
    if display.sizes.get("time", 0) == 0:
        raise ValueError("No display-energy samples could be generated from the Power Zarr")
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    display = display.chunk({"time": 1440})
    display.to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)
    print(f"Wrote {output_zarr} with {display.sizes.get('time', 0)} samples")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the compact Power display-energy Zarr")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_DISPLAY_ENERGY_ZARR_PATH)
    parser.add_argument("--freq", default=POWER_DISPLAY_ENERGY_FREQ)
    args = parser.parse_args()
    generate(args.power_zarr, args.output_zarr, args.freq)


if __name__ == "__main__":
    main()
