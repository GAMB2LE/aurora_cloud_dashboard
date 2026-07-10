#!/usr/bin/env python3
"""Generate compact Aurora Power Supply display-summary Zarr products."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import xarray as xr

from grouped_timeseries import (
    POWER_DISPLAY_ENERGY_ATTR,
    POWER_DISPLAY_ENERGY_MAP,
    POWER_DISPLAY_SUMMARY_ATTR,
    POWER_DISPLAY_SUMMARY_FREQ,
    build_power_display_summary_dataset,
)

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
ASFS_LOGGER_ZARR_PATH = Path(os.environ.get("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"))
PDU_ZARR_PATH = Path(os.environ.get("PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))
POWER_SOC_FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
POWER_DISPLAY_SUMMARY_ZARR_PATH = Path(
    os.environ.get("POWER_DISPLAY_SUMMARY_ZARR_PATH", "/data/aurora/products/power/power_display_summary.zarr")
)
POWER_DISPLAY_ENERGY_ZARR_PATH = Path(
    os.environ.get("POWER_DISPLAY_ENERGY_ZARR_PATH", "/data/aurora/products/power/power_display_energy.zarr")
)


def _write_zarr_atomic(ds: xr.Dataset, output_zarr: Path, chunk_time: int = 1440) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.chunk({"time": chunk_time}).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _open_optional_zarr(path: Path, label: str) -> xr.Dataset | None:
    if not path.exists():
        return None
    try:
        return xr.open_zarr(path, chunks={})
    except Exception as exc:
        print(f"Could not open {label} Zarr for Power display summary: {exc}")
        return None


def _energy_subset(summary: xr.Dataset, freq: str) -> xr.Dataset:
    names = [name for name in POWER_DISPLAY_ENERGY_MAP.values() if name in summary]
    if not names:
        return xr.Dataset()
    out = summary[names].copy(deep=False)
    out.attrs = {
        POWER_DISPLAY_ENERGY_ATTR: "true",
        "source": "derived from power_display_summary.zarr",
        "frequency": freq,
        "description": "Display-only one-minute cumulative APS energy traces for dashboard plotting.",
    }
    for name in out.data_vars:
        out[name].attrs["units"] = "kWh"
    return out


def generate(
    power_zarr: Path = POWER_ZARR_PATH,
    output_zarr: Path = POWER_DISPLAY_SUMMARY_ZARR_PATH,
    ass_logger_zarr: Path = ASFS_LOGGER_ZARR_PATH,
    pdu_zarr: Path = PDU_ZARR_PATH,
    forecast_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    energy_output_zarr: Path | None = POWER_DISPLAY_ENERGY_ZARR_PATH,
    freq: str = POWER_DISPLAY_SUMMARY_FREQ,
) -> Path:
    """Build the derived one-minute display-summary store from Power inputs."""
    power = xr.open_zarr(power_zarr, chunks={})
    ass_logger = _open_optional_zarr(ass_logger_zarr, "ASFS logger")
    pdu = _open_optional_zarr(pdu_zarr, "ASS PDU")
    forecast = _open_optional_zarr(forecast_zarr, "Power SOC forecast")
    display = build_power_display_summary_dataset(power, ass_logger, pdu, forecast, freq=freq)
    if display.sizes.get("time", 0) == 0:
        raise ValueError("No display-summary samples could be generated from the Power Zarr")

    display.attrs[POWER_DISPLAY_SUMMARY_ATTR] = "true"
    _write_zarr_atomic(display, output_zarr)
    print(f"Wrote {output_zarr} with {display.sizes.get('time', 0)} samples and {len(display.data_vars)} variables")

    if energy_output_zarr is not None:
        energy = _energy_subset(display, freq)
        if energy.sizes.get("time", 0) and len(energy.data_vars):
            _write_zarr_atomic(energy, energy_output_zarr)
            print(f"Wrote {energy_output_zarr} with {energy.sizes.get('time', 0)} samples")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the compact Power display-summary Zarr")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--asfs-logger-zarr", type=Path, default=ASFS_LOGGER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=PDU_ZARR_PATH)
    parser.add_argument("--forecast-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_DISPLAY_SUMMARY_ZARR_PATH)
    parser.add_argument("--energy-output-zarr", type=Path, default=POWER_DISPLAY_ENERGY_ZARR_PATH)
    parser.add_argument("--no-energy-output", action="store_true", help="Do not refresh the legacy cumulative-energy display Zarr")
    parser.add_argument("--freq", default=POWER_DISPLAY_SUMMARY_FREQ)
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        output_zarr=args.output_zarr,
        ass_logger_zarr=args.asfs_logger_zarr,
        pdu_zarr=args.pdu_zarr,
        forecast_zarr=args.forecast_zarr,
        energy_output_zarr=None if args.no_energy_output else args.energy_output_zarr,
        freq=args.freq,
    )


if __name__ == "__main__":
    main()
