#!/usr/bin/env python3
"""
Append new RPG FMCW 94 GHz cloud radar NetCDF files into an existing Zarr store.
- Scans the root directory recursively for *.NC files newer than the latest time in the Zarr.
- Uses the same conversion logic as cloud_radar_to_zarr.py (Chirp 1 Ze -> dBZ, SLDR).
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

ROOT_DEFAULT = Path("/mnt/data/ass/rpgfmcw94")
ZARR_DEFAULT = Path("/mnt/data/ass/rpgfmcw94/cloud_radar.zarr")
TIME_ZERO = np.datetime64("2001-01-01T00:00:00")
NC_REGEX = re.compile(r"_(\d{6})_(\d{6})")  # yymmdd_hhmmss
FUTURE_TIME_TOLERANCE = timedelta(days=2)


def _parse_timestamp(path: Path) -> datetime | None:
    m = NC_REGEX.search(path.name)
    if not m:
        return None
    date_part, time_part = m.groups()
    try:
        return datetime.strptime(date_part + time_part, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _list_files_after(root: Path, after: datetime | None = None) -> List[Path]:
    files: List[tuple[datetime, Path]] = []
    for p in root.rglob("*.NC"):
        if not p.name.upper().endswith("LV1.NC"):
            continue
        ts = _parse_timestamp(p)
        if ts is None:
            continue
        if after is None or ts > after:
            files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def _load_nc(path: Path) -> xr.Dataset:
    raw = xr.open_dataset(path, decode_times=False)
    base = TIME_ZERO
    time = base + raw["Time"].astype("timedelta64[s]") + raw["Timems"].astype("timedelta64[ms]")
    time_vals = np.array(time.values)

    required = ["C1Range", "C2Range"]
    for r in required:
        if r not in raw:
            raise KeyError(f"Missing {r} in {path}")

    r1 = raw["C1Range"].values
    r2 = raw["C2Range"].values
    ranges = np.concatenate([r1, r2])
    t_len = raw["Time"].sizes["Time"]

    var_specs = [
        ("ZE_dBZ", "C1ZE", "C2ZE", "dbz"),
        ("ZE45_dBZ", "C1ZE45", "C2ZE45", "dbz"),
        ("MeanVel", "C1MeanVel", "C2MeanVel", "linear"),
        ("ZDR", "C1ZDR", "C2ZDR", "linear"),
        ("SRCX", "C1SRCX", "C2SRCX", "linear"),
        ("SpecWidth", "C1SpecWidth", "C2SpecWidth", "linear"),
        ("SLDR", "C1SLDR", "C2SLDR", "linear"),
        ("Skew", "C1Skew", "C2Skew", "linear"),
        ("RHV", "C1RHV", "C2RHV", "linear"),
        ("PhiDP", "C1PhiDP", "C2PhiDP", "linear"),
        ("Kurt", "C1Kurt", "C2Kurt", "linear"),
        ("KDP", "C1KDP", "C2KDP", "linear"),
        ("DiffAtt", "C1DiffAtt", "C2DiffAtt", "linear"),
    ]

    data_vars = {}
    for out_name, c1, c2, mode in var_specs:
        if c1 not in raw or c2 not in raw:
            continue
        arr = np.full((t_len, len(ranges)), np.nan, dtype=np.float32)
        arr[:, : len(r1)] = raw[c1].values
        arr[:, len(r1) :] = raw[c2].values
        arr = np.where(arr <= -900, np.nan, arr)
        if mode == "dbz":
            arr = np.where(arr > 0, arr, np.nan)
            with np.errstate(divide="ignore"):
                arr = 10.0 * np.log10(arr)
        data_vars[out_name] = (("time", "range"), arr.astype(np.float32))

    ds = xr.Dataset(data_vars, coords={"time": time_vals, "range": ranges})
    return ds.sortby("time")


def _deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    times = np.asarray(ds["time"].values)
    _, unique_idx = np.unique(times, return_index=True)
    if len(unique_idx) != len(times):
        print(f"Dropping {len(times) - len(unique_idx)} duplicate time samples")
        ds = ds.isel(time=np.sort(unique_idx))
    return ds


def _load_files(files: List[Path], chunks: dict | str | None = None) -> xr.Dataset:
    datasets = []
    for f in files:
        print(f"  {f.name}")
        try:
            ds = _load_nc(f)
        except Exception as exc:
            print(f"Skipping unreadable radar file {f}: {exc}")
            continue
        datasets.append(ds)
    if not datasets:
        return xr.Dataset()
    combined = xr.concat(datasets, dim="time").sortby("time")
    combined = _deduplicate_time(combined)
    if chunks:
        combined = combined.chunk(chunks)
    return combined


def _latest_valid_time(base: xr.Dataset) -> datetime:
    times = np.asarray(base["time"].values)
    if times.size == 0:
        raise ValueError("Zarr store has no time samples")
    valid = ~np.isnat(times)
    cutoff = np.datetime64((datetime.now(timezone.utc) + FUTURE_TIME_TOLERANCE).replace(tzinfo=None))
    future_mask = valid & (times > cutoff)
    if np.any(future_mask):
        print(f"Ignoring {int(np.count_nonzero(future_mask))} bogus future radar timestamps when computing append frontier.")
    valid &= times <= cutoff
    if not np.any(valid):
        valid = ~np.isnat(times)
    latest = pd.Timestamp(times[valid].max()).to_pydatetime(warn=False)
    return latest.replace(tzinfo=timezone.utc)


def append_new(
    root: Path,
    zarr_path: Path,
    chunks: dict | str | None = None,
    max_backfill_days: int | None = 11,
    lookback_hours: int = 6,
):
    if not zarr_path.exists():
        start_cutoff = None
        if max_backfill_days is not None:
            start_cutoff = datetime.now(timezone.utc) - timedelta(days=max_backfill_days)
            print(f"Zarr store not found; bootstrapping from files newer than {start_cutoff}.")
        else:
            print("Zarr store not found; bootstrapping from all matching files.")
        files = _list_files_after(root, start_cutoff)
        if not files:
            print("No radar .LV1.NC files available to bootstrap.")
            return
        print(f"Bootstrapping radar Zarr from {len(files)} files")
        combined = _load_files(files, chunks=chunks)
        if combined.sizes.get("time", 0) == 0:
            print("No readable radar samples available to bootstrap.")
            return
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_zarr(zarr_path, mode="w", consolidated=True)
        print("Bootstrap complete.")
        return

    base = xr.open_zarr(zarr_path, chunks={})
    if "time" not in base:
        raise KeyError("Zarr store missing time coordinate")
    last_time = _latest_valid_time(base)
    print(f"Latest time in Zarr: {last_time}")

    scan_after = last_time - timedelta(hours=max(lookback_hours, 0))
    files = _list_files_after(root, scan_after)
    if not files:
        print("No new .NC files to append.")
        return

    print(f"Scanning {len(files)} candidate files")
    combined = _load_files(files, chunks=chunks)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no readable radar samples.")
        return
    new_time_mask = (combined["time"] > np.datetime64(last_time.replace(tzinfo=None))).values
    combined = combined.isel(time=new_time_mask)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no samples newer than the existing Zarr.")
        return
    # Ensure chunk alignment with existing store; disable strict chunk safety to avoid overlap errors.
    combined.to_zarr(zarr_path, mode="a", append_dim="time", safe_chunks=False)
    print("Append complete.")


def main():
    parser = argparse.ArgumentParser(description="Append new cloud radar NC files into existing Zarr.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--chunk-time", type=int, default=400)
    parser.add_argument("--max-backfill-days", type=int, default=11)
    parser.add_argument("--lookback-hours", type=int, default=6)
    args = parser.parse_args()

    chunks = {"time": args.chunk_time} if args.chunk_time else None
    append_new(
        args.root,
        args.zarr,
        chunks=chunks,
        max_backfill_days=args.max_backfill_days,
        lookback_hours=args.lookback_hours,
    )


if __name__ == "__main__":
    main()
