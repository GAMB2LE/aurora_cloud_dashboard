#!/usr/bin/env python3
"""Append ASFS fast-gas TOA5 .dat files into a time-indexed Zarr store."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr

ROOT_DEFAULT = Path("/project/aurora/raw/asfs")
ZARR_DEFAULT = Path("/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr")
MIN_TIME_DEFAULT = os.environ.get("ASFS_ZARR_MIN_TIME", "2026-05-02T00:00:00")
LOGGERNET_FILE_REGEX = re.compile(r"asfs-logger_fast_gas_(\d{2})_(\d{2})_(\d{4})\.dat$")
CRD_FILE_REGEX = re.compile(r"aurora_asfs_data_fast_gas_(\d{12})\.dat$")
FILE_PATTERNS = ("asfs-logger_fast_gas_*.dat", "aurora_asfs_data_fast_gas_*.dat")


def _parse_file_time(path: Path) -> datetime | None:
    match = LOGGERNET_FILE_REGEX.match(path.name)
    if not match:
        crd_match = CRD_FILE_REGEX.match(path.name)
        if not crd_match:
            return None
        try:
            return datetime.strptime(crd_match.group(1), "%Y%m%d%H%M")
        except ValueError:
            return None
    day, month, year = match.groups()
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def _list_files(root: Path, start_date: date | None = None) -> list[Path]:
    files: list[tuple[datetime, Path]] = []
    seen: set[Path] = set()
    for pattern in FILE_PATTERNS:
        for path in root.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            file_time = _parse_file_time(path)
            if file_time is None:
                continue
            if start_date is None or file_time.date() >= start_date:
                files.append((file_time, path))
    files.sort(key=lambda item: (item[0], str(item[1])))
    return [path for _, path in files]


def _spread_duplicate_timestamps(timestamps: pd.Series) -> pd.Series:
    """Make repeated logger timestamps unique while preserving their block time.

    The CR1000X fast-gas table records a burst of samples under one timestamp
    every ~20 seconds. Zarr coordinates need useful ordering, so samples in each
    repeated-timestamp block are spread evenly before the next block.
    """
    adjusted = timestamps.copy()
    valid = adjusted.notna()
    if not valid.any():
        return adjusted
    valid_times = pd.DatetimeIndex(adjusted.loc[valid])
    unique_times = pd.Index(valid_times.unique()).sort_values()
    next_by_time = {
        time: (unique_times[idx + 1] - time).total_seconds()
        for idx, time in enumerate(unique_times[:-1])
    }
    default_span = float(np.nanmedian([span for span in next_by_time.values() if span > 0])) if next_by_time else 1.0
    if not np.isfinite(default_span) or default_span <= 0:
        default_span = 1.0
    for _, indexes in adjusted.loc[valid].groupby(adjusted.loc[valid]).groups.items():
        count = len(indexes)
        if count <= 1:
            continue
        base = adjusted.iloc[indexes[0]]
        span = next_by_time.get(base, default_span)
        step = min(max(span / count, 0.001), 1.0)
        offsets = pd.to_timedelta(np.arange(count, dtype=np.float64) * step, unit="s")
        adjusted.iloc[list(indexes)] = base + offsets
    return adjusted


def _deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    times = np.asarray(ds["time"].values)
    _, reverse_idx = np.unique(times[::-1], return_index=True)
    unique_idx = len(times) - 1 - reverse_idx
    if len(unique_idx) != len(times):
        print(f"Dropping {len(times) - len(unique_idx)} duplicate time samples")
        ds = ds.isel(time=np.sort(unique_idx))
    return ds


def _read_file(path: Path) -> xr.Dataset:
    frame = pd.read_csv(
        path,
        header=1,
        skiprows=[2, 3],
        na_values=["NAN", "nan", "NaN"],
        keep_default_na=True,
    )
    frame.columns = [str(col).strip() for col in frame.columns]
    if "TIMESTAMP" not in frame.columns:
        raise KeyError(f"{path} does not contain a TIMESTAMP column")

    timestamps = pd.to_datetime(frame.pop("TIMESTAMP"), errors="coerce")
    valid_time = timestamps.notna()
    if not valid_time.any():
        return xr.Dataset()
    frame = frame.loc[valid_time].reset_index(drop=True)
    timestamps = _spread_duplicate_timestamps(timestamps.loc[valid_time].reset_index(drop=True))
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(None)

    data_vars = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        data_vars[column] = (("time",), values.to_numpy(dtype=np.float32))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": timestamps.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "instrument": "asfs-fast-gas",
            "source_file": str(path),
        },
    )
    return _deduplicate_time(ds.sortby("time"))


def _load_files(files: Iterable[Path], chunks: dict[str, int] | None = None) -> xr.Dataset:
    datasets = []
    for path in files:
        print(f"  {path.name}")
        try:
            ds = _read_file(path)
        except Exception as exc:
            print(f"Skipping unreadable ASFS fast-gas file {path}: {exc}")
            continue
        if ds.sizes.get("time", 0) == 0:
            continue
        datasets.append(ds)
    if not datasets:
        return xr.Dataset()
    combined = xr.concat(datasets, dim="time", join="outer").sortby("time")
    combined = _deduplicate_time(combined)
    if MIN_TIME_DEFAULT:
        combined = combined.sel(time=slice(np.datetime64(MIN_TIME_DEFAULT), None))
    combined.attrs.update(
        {
            "instrument": "asfs-fast-gas",
            "title": "ASFS fast-gas data",
            "source": "asfs-logger_fast_gas_DD_MM_YYYY.dat or aurora_asfs_data_fast_gas_YYYYMMDDHHMM.dat",
        }
    )
    if chunks:
        combined = combined.chunk(chunks)
    return combined


def _align_to_existing(combined: xr.Dataset, existing: xr.Dataset) -> xr.Dataset:
    existing_vars = list(existing.data_vars)
    for name in existing_vars:
        if name not in combined:
            combined[name] = (("time",), np.full(combined.sizes["time"], np.nan, dtype=np.float32))
    extras = [name for name in combined.data_vars if name not in existing_vars]
    if extras:
        print(f"Dropping new variables not present in existing Zarr: {', '.join(extras)}")
        combined = combined.drop_vars(extras)
    return combined[existing_vars]


def _consolidate(zarr_path: Path) -> None:
    try:
        zarr.consolidate_metadata(str(zarr_path))
    except Exception as exc:
        print(f"Could not consolidate Zarr metadata for {zarr_path}: {exc}")


def append_new(root: Path, zarr_path: Path, chunks: dict[str, int] | None = None, lookback_days: int = 2) -> None:
    if not root.exists():
        print(f"Raw ASFS directory does not exist: {root}")
        return

    if not zarr_path.exists():
        files = _list_files(root)
        if not files:
            print("No matching ASFS fast-gas .dat files available to bootstrap.")
            return
        print(f"Bootstrapping ASFS fast-gas Zarr from {len(files)} files")
        combined = _load_files(files, chunks=chunks)
        if combined.sizes.get("time", 0) == 0:
            print("No readable ASFS fast-gas samples available to bootstrap.")
            return
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_zarr(zarr_path, mode="w", consolidated=True)
        print("Bootstrap complete.")
        return

    existing = xr.open_zarr(zarr_path, chunks={})
    if "time" not in existing:
        raise KeyError("Zarr store missing time coordinate")
    last_time = pd.to_datetime(existing["time"].max().values).to_pydatetime()
    print(f"Latest time in Zarr: {last_time}")

    scan_date = (last_time - timedelta(days=max(lookback_days, 0))).date()
    files = _list_files(root, scan_date)
    if not files:
        print("No candidate ASFS fast-gas .dat files to append.")
        return

    print(f"Scanning {len(files)} candidate files")
    combined = _load_files(files, chunks=None)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no readable ASFS fast-gas samples.")
        return
    combined = combined.isel(time=(combined["time"] > np.datetime64(last_time)).values)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no samples newer than the existing Zarr.")
        return
    combined = _align_to_existing(combined, existing)
    combined = combined.load()
    combined.to_zarr(zarr_path, mode="a", append_dim="time")
    _consolidate(zarr_path)
    print("Append complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append ASFS fast-gas TOA5 .dat files into a Zarr store.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--chunk-time", type=int, default=24000)
    parser.add_argument("--lookback-days", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true", help="Remove the existing Zarr before rebuilding from all raw files.")
    args = parser.parse_args()

    if args.rebuild and args.zarr.exists():
        shutil.rmtree(args.zarr)
        print(f"Removed existing Zarr store: {args.zarr}")

    chunks = {"time": args.chunk_time} if args.chunk_time else None
    append_new(args.root, args.zarr, chunks=chunks, lookback_days=args.lookback_days)


if __name__ == "__main__":
    main()
