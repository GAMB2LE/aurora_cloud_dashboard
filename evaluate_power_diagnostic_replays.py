#!/usr/bin/env python3
"""Write a bounded retrospective attribution product, never a live forecast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from power_archive_io import write_forecast_archive
from power_observation_truth import (
    bounded_observation_view, diagnostic_soc_replays, forecast_interval_starts,
)
from repair_power_forecast_archive import verified_issue_snapshot


def run_diagnostic(issue_directory: Path, power_zarr: Path, output_zarr: Path) -> dict:
    # Validate the exact snapshot before reading the forecast. This command
    # never repairs incomplete identities or writes into the input archive.
    verified_issue_snapshot(issue_directory)
    paths = [Path(p).resolve() for p in (issue_directory, power_zarr, output_zarr)]
    if any(paths[2] == p or p in paths[2].parents or paths[2] in p.parents for p in paths[:2]):
        raise ValueError("Diagnostic output overlaps a protected input")
    if output_zarr.exists():
        raise FileExistsError("Diagnostic output must be a new path")
    with xr.open_zarr(issue_directory / "forecast.zarr", chunks={}) as opened:
        forecast = opened.load()
    times = forecast.time.values
    starts = forecast_interval_starts(times, forecast.attrs["initial_soc_time"])
    with xr.open_zarr(power_zarr, chunks={}) as power:
        subset = bounded_observation_view(power, starts, times).load()
    result = diagnostic_soc_replays(forecast, subset)
    result.attrs.update({
        "baseline_forecast_identity_id": str(forecast.attrs["forecast_identity_id"]),
        "baseline_publication_signature": str(forecast.attrs["publication_signature"]),
    })
    write_forecast_archive(result, output_zarr)
    return {"authority": "retrospective_diagnostic_only", "output": str(output_zarr),
            "finite_samples": {name: int(np.isfinite(var.values).sum())
                               for name, var in result.data_vars.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-directory", type=Path, required=True)
    parser.add_argument("--power-zarr", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.issue_directory, args.power_zarr, args.output_zarr)))


if __name__ == "__main__":
    main()
