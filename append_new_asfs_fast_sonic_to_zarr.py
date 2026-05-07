#!/usr/bin/env python3
"""Append ASFS fast-sonic LoggerNet TOA5 .dat files into a Zarr store."""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr

ROOT_DEFAULT = Path("/project/aurora/raw/asfs/loggernet")
ZARR_DEFAULT = Path("/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr")
FILE_REGEX = re.compile(r"asfs-logger_fast_sonic_(\d{2})_(\d{2})_(\d{4})\.dat$")


def _parse_file_date(path: Path) -> date | None:
    match = FILE_REGEX.match(path.name)
    if not match:
        return None
    day, month, year = match.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _list_files(root: Path, start_date: date | None = None) -> list[Path]:
    files: list[tuple[date, Path]] = []
    for path in root.glob("asfs-logger_fast_sonic_*.dat"):
        file_date = _parse_file_date(path)
        if file_date is None:
            continue
        if start_date is None or file_date >= start_date:
            files.append((file_date, path))
    files.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in files]


def _deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    times = np.asarray(ds["time"].values)
    _, unique_idx = np.unique(times, return_index=True)
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
    if "metek_msec_out" in frame.columns:
        milliseconds = pd.to_numeric(frame["metek_msec_out"], errors="coerce")
        timestamps = timestamps + pd.to_timedelta(milliseconds.fillna(0.0), unit="ms")
    valid_time = timestamps.notna()
    if not valid_time.any():
        return xr.Dataset()
    frame = frame.loc[valid_time].reset_index(drop=True)
    timestamps = timestamps.loc[valid_time].reset_index(drop=True)
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
            "instrument": "asfs-fast-sonic",
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
            print(f"Skipping unreadable ASFS fast-sonic file {path}: {exc}")
            continue
        if ds.sizes.get("time", 0) == 0:
            continue
        datasets.append(ds)
    if not datasets:
        return xr.Dataset()
    combined = xr.concat(datasets, dim="time", join="outer").sortby("time")
    combined = _deduplicate_time(combined)
    combined.attrs.update(
        {
            "instrument": "asfs-fast-sonic",
            "title": "ASFS LoggerNet fast-sonic data",
            "source": "asfs-logger_fast_sonic_DD_MM_YYYY.dat",
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
        print(f"Raw ASFS LoggerNet directory does not exist: {root}")
        return

    if not zarr_path.exists():
        files = _list_files(root)
        if not files:
            print("No matching ASFS fast-sonic .dat files available to bootstrap.")
            return
        print(f"Bootstrapping ASFS fast-sonic Zarr from {len(files)} files")
        combined = _load_files(files, chunks=chunks)
        if combined.sizes.get("time", 0) == 0:
            print("No readable ASFS fast-sonic samples available to bootstrap.")
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
        print("No candidate ASFS fast-sonic .dat files to append.")
        return

    print(f"Scanning {len(files)} candidate files")
    combined = _load_files(files, chunks=chunks)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no readable ASFS fast-sonic samples.")
        return
    combined = combined.isel(time=(combined["time"] > np.datetime64(last_time)).values)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no samples newer than the existing Zarr.")
        return
    combined = _align_to_existing(combined, existing)
    if chunks:
        combined = combined.chunk(chunks)
    combined.to_zarr(zarr_path, mode="a", append_dim="time", safe_chunks=False)
    _consolidate(zarr_path)
    print("Append complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append ASFS fast-sonic TOA5 .dat files into a Zarr store.")
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
