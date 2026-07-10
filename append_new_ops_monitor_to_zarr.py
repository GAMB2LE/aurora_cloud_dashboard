#!/usr/bin/env python3
"""Append Aurora operations JSONL snapshots into a time-indexed Zarr store."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from rebuild_cutoff import cutoff_date, filter_dataset_from_time, parse_from_time


ROOT_DEFAULT = Path("/project/aurora/raw/ops_monitor")
ZARR_DEFAULT = Path("/data/aurora/products/ops_monitor/ops_monitor.zarr")
FILE_REGEX = re.compile(r"ops_monitor_(\d{4})(\d{2})(\d{2})\.jsonl$")


class SchemaExpansionRequired(RuntimeError):
    """Raised when the raw ops snapshots contain variables missing from the Zarr schema."""


def _parse_file_date(path: Path) -> date | None:
    match = FILE_REGEX.match(path.name)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _list_files(root: Path, start_date: date | None = None) -> list[Path]:
    files: list[tuple[date, Path]] = []
    for path in root.glob("ops_monitor_*.jsonl"):
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
        ds = ds.isel(time=np.sort(unique_idx))
    return ds


def _has_sorted_unique_time(ds: xr.Dataset) -> bool:
    if "time" not in ds.coords:
        return True
    times = np.asarray(ds["time"].values)
    if times.size < 2:
        return True
    diffs = np.diff(times.astype("datetime64[ns]").astype("int64"))
    return bool(np.all(diffs > 0))


def _read_file(path: Path) -> xr.Dataset:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed operations snapshot in {path}: {exc}")
    if not rows:
        return xr.Dataset()

    frame = pd.DataFrame(rows)
    if "time_utc" not in frame.columns:
        raise KeyError(f"{path} does not contain time_utc")
    timestamps = pd.to_datetime(frame.pop("time_utc"), errors="coerce", utc=True)
    valid_time = timestamps.notna()
    if not valid_time.any():
        return xr.Dataset()
    frame = frame.loc[valid_time].reset_index(drop=True)
    timestamps = timestamps.loc[valid_time].dt.tz_convert("UTC").dt.tz_localize(None).reset_index(drop=True)

    data_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        data_vars[column] = (("time",), values.to_numpy(dtype=np.float32))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": timestamps.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "instrument": "ops-monitor",
            "title": "Aurora operations monitoring",
            "source": "ops_monitor_YYYYMMDD.jsonl",
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
            print(f"Skipping unreadable operations snapshot file {path}: {exc}")
            continue
        if ds.sizes.get("time", 0) == 0:
            continue
        datasets.append(ds)
    if not datasets:
        return xr.Dataset()
    combined = xr.concat(datasets, dim="time", join="outer").sortby("time")
    combined = _deduplicate_time(combined)
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
        raise SchemaExpansionRequired(
            f"raw ops snapshots contain new variables missing from Zarr schema: {', '.join(extras)}"
        )
    return combined[existing_vars]


def _consolidate(zarr_path: Path) -> None:
    try:
        zarr.consolidate_metadata(str(zarr_path))
    except Exception as exc:
        print(f"Could not consolidate Zarr metadata for {zarr_path}: {exc}")


def _backup_existing_store(zarr_path: Path, reason: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = zarr_path.with_name(f"{zarr_path.stem}.backup_{reason}_{timestamp}{zarr_path.suffix}")
    shutil.move(str(zarr_path), str(backup_path))
    print(f"Moved existing operations Zarr to {backup_path}")


def append_new(
    root: Path,
    zarr_path: Path,
    chunks: dict[str, int] | None = None,
    lookback_days: int = 2,
    from_time: datetime | None = None,
) -> None:
    if not root.exists():
        print(f"Raw operations snapshot directory does not exist: {root}")
        return

    from_date = cutoff_date(from_time)
    if not zarr_path.exists():
        files = _list_files(root, from_date)
        if not files:
            print("No operations snapshot files available to bootstrap.")
            return
        print(f"Bootstrapping operations Zarr from {len(files)} files")
        combined = _load_files(files, chunks=chunks)
        combined = filter_dataset_from_time(combined, from_time)
        if combined.sizes.get("time", 0) == 0:
            print("No readable operations snapshots available to bootstrap.")
            return
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_zarr(zarr_path, mode="w", consolidated=True)
        print("Bootstrap complete.")
        return

    try:
        existing = xr.open_zarr(zarr_path, chunks={})
    except Exception as exc:
        print(f"Existing operations Zarr is unreadable ({exc}); rebuilding from raw snapshots.")
        _backup_existing_store(zarr_path, "corrupt")
        return append_new(root, zarr_path, chunks=chunks, lookback_days=lookback_days, from_time=from_time)
    if "time" not in existing:
        raise KeyError("Zarr store missing time coordinate")
    if not _has_sorted_unique_time(existing):
        print("Existing operations Zarr has unsorted or duplicate time samples; rebuilding from raw snapshots.")
        _backup_existing_store(zarr_path, "time_order")
        return append_new(root, zarr_path, chunks=chunks, lookback_days=lookback_days, from_time=from_time)
    last_time = pd.to_datetime(existing["time"].max().values).to_pydatetime()
    print(f"Latest time in Zarr: {last_time}")

    scan_date = (last_time - timedelta(days=max(lookback_days, 0))).date()
    if from_date is not None:
        scan_date = max(scan_date, from_date)
    files = _list_files(root, scan_date)
    if not files:
        print("No candidate operations snapshot files to append.")
        return

    print(f"Scanning {len(files)} candidate files")
    combined = _load_files(files, chunks=None)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no readable operations snapshots.")
        return
    combined = filter_dataset_from_time(combined, from_time)
    combined = combined.isel(time=(combined["time"] > np.datetime64(last_time)).values)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no snapshots newer than the existing Zarr.")
        return
    try:
        combined = _align_to_existing(combined, existing)
    except SchemaExpansionRequired as exc:
        print(f"{exc}; rebuilding from raw snapshots.")
        _backup_existing_store(zarr_path, "schema")
        return append_new(root, zarr_path, chunks=chunks, lookback_days=lookback_days, from_time=from_time)
    combined = combined.load()
    combined.to_zarr(zarr_path, mode="a", append_dim="time")
    _consolidate(zarr_path)
    print("Append complete.")
    return


def main() -> None:
    parser = argparse.ArgumentParser(description="Append Aurora operations snapshots into a Zarr store.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--chunk-time", type=int, default=720)
    parser.add_argument("--lookback-days", type=int, default=3)
    parser.add_argument("--from-time", help="Only write samples at or after this UTC ISO timestamp.")
    parser.add_argument("--rebuild", action="store_true", help="Remove the existing Zarr before rebuilding from all raw snapshots.")
    args = parser.parse_args()

    if args.rebuild and args.zarr.exists():
        shutil.rmtree(args.zarr)
        print(f"Removed existing Zarr store: {args.zarr}")

    chunks = {"time": args.chunk_time} if args.chunk_time else None
    append_new(
        args.root,
        args.zarr,
        chunks=chunks,
        lookback_days=args.lookback_days,
        from_time=parse_from_time(args.from_time),
    )


if __name__ == "__main__":
    main()
