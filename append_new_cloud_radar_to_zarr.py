#!/usr/bin/env python3
"""
Append new RPG FMCW 94 GHz cloud radar NetCDF files into an existing Zarr store.
- Scans the root directory recursively for *.NC files newer than the latest time in the Zarr.
- Combines chirp ranges and normalizes radar variables into the dashboard cloud-radar Zarr.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from rebuild_cutoff import filter_dataset_from_time, naive_utc, parse_from_time

ROOT_DEFAULT = Path("/project/aurora/raw/rpgfmcw94")
ZARR_DEFAULT = Path("/data/aurora/products/rpgfmcw94/cloud_radar.zarr")
TIME_ZERO = np.datetime64("2001-01-01T00:00:00")
NC_REGEX = re.compile(r"_(\d{6})_(\d{6})")  # yymmdd_hhmmss
FUTURE_TIME_TOLERANCE = timedelta(days=2)


class GeometryRecord(NamedTuple):
    timestamp: datetime
    path: Path
    key: str
    range_count: int


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


def _range_values_from_raw(raw: xr.Dataset, path: Path) -> np.ndarray:
    required = ["C1Range", "C2Range"]
    for r in required:
        if r not in raw:
            raise KeyError(f"Missing {r} in {path}")
    r1 = raw["C1Range"].values
    r2 = raw["C2Range"].values
    return np.concatenate([r1, r2])


def _range_key(ranges: np.ndarray) -> str:
    arr = np.asarray(ranges, dtype=np.float32)
    digest = hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return f"{arr.size}:{digest}"


def _inspect_file_geometry(path: Path) -> tuple[str, int]:
    raw = xr.open_dataset(path, decode_times=False)
    ranges = _range_values_from_raw(raw, path)
    return _range_key(ranges), int(ranges.size)


def _scan_geometries(files: List[Path]) -> List[GeometryRecord]:
    records: List[GeometryRecord] = []
    for path in files:
        ts = _parse_timestamp(path)
        if ts is None:
            continue
        try:
            key, count = _inspect_file_geometry(path)
        except Exception as exc:
            print(f"Skipping unreadable radar geometry {path}: {exc}")
            continue
        records.append(GeometryRecord(ts, path, key, count))
    return records


def _latest_geometry_run(root: Path) -> tuple[list[GeometryRecord], str, int]:
    all_files = _list_files_after(root, None)
    records = _scan_geometries(all_files)
    if not records:
        raise ValueError("No readable radar .LV1.NC files available to inspect geometry.")
    target_key = records[-1].key
    target_count = records[-1].range_count
    start_idx = len(records) - 1
    while start_idx > 0 and records[start_idx - 1].key == target_key:
        start_idx -= 1
    run = records[start_idx:]
    return run, target_key, target_count


def _backup_path(zarr_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = zarr_path.with_name(f"{zarr_path.stem}.backup_{stamp}{zarr_path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = zarr_path.with_name(f"{zarr_path.stem}.backup_{stamp}_{counter}{zarr_path.suffix}")
        counter += 1
    return candidate


def _backup_existing_store(zarr_path: Path) -> Path | None:
    if not zarr_path.exists():
        return None
    backup = _backup_path(zarr_path)
    print(f"Backing up existing radar Zarr to {backup}")
    shutil.move(str(zarr_path), str(backup))
    return backup


def _load_nc(path: Path) -> xr.Dataset:
    raw = xr.open_dataset(path, decode_times=False)
    base = TIME_ZERO
    time = base + raw["Time"].astype("timedelta64[s]") + raw["Timems"].astype("timedelta64[ms]")
    time_vals = np.array(time.values)
    ranges = _range_values_from_raw(raw, path)
    r1 = raw["C1Range"].values
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
    ds.attrs["range_layout_key"] = _range_key(ranges)
    ds.attrs["range_count"] = int(ranges.size)
    ds = ds.sortby("time")
    cutoff = np.datetime64((datetime.now(timezone.utc) + FUTURE_TIME_TOLERANCE).replace(tzinfo=None))
    valid_time = (~np.isnat(ds["time"].values)) & (ds["time"].values <= cutoff)
    dropped = int(ds.sizes.get("time", 0) - np.count_nonzero(valid_time))
    if dropped:
        print(f"Skipping {dropped} invalid or future radar samples in {path.name}")
        ds = ds.isel(time=valid_time)
    return ds


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


def _existing_range_key(base: xr.Dataset) -> tuple[str, int]:
    if "range" not in base.coords:
        raise KeyError("Zarr store missing range coordinate")
    ranges = np.asarray(base["range"].values)
    return _range_key(ranges), int(ranges.size)


def _bootstrap_store(
    files: List[Path],
    zarr_path: Path,
    chunks: dict | str | None = None,
    description: str = "bootstrap",
    from_time: datetime | None = None,
):
    if not files:
        print(f"No radar files available for {description}.")
        return
    print(f"{description.capitalize()} from {len(files)} files")
    combined = _load_files(files, chunks=chunks)
    combined = filter_dataset_from_time(combined, from_time)
    if combined.sizes.get("time", 0) == 0:
        print(f"No readable radar samples available for {description}.")
        return
    combined.attrs["range_layout_key"] = combined.attrs.get("range_layout_key", _range_key(np.asarray(combined["range"].values)))
    combined.attrs["range_count"] = int(combined.sizes.get("range", 0))
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_zarr(zarr_path, mode="w", consolidated=True)
    print(f"{description.capitalize()} complete.")


def rebuild_from_latest_geometry_run(
    root: Path,
    zarr_path: Path,
    chunks: dict | str | None = None,
    backup_existing: bool = True,
    from_time: datetime | None = None,
):
    run, target_key, target_count = _latest_geometry_run(root)
    print(
        "Rebuilding radar Zarr from latest geometry run: "
        f"{target_count} range gates, starting at {run[0].timestamp}, "
        f"ending at {run[-1].timestamp}."
    )
    if backup_existing:
        _backup_existing_store(zarr_path)
    _bootstrap_store(
        [record.path for record in run],
        zarr_path,
        chunks=chunks,
        description=f"latest-geometry rebuild ({target_count} gates, key={target_key})",
        from_time=from_time,
    )


def append_new(
    root: Path,
    zarr_path: Path,
    chunks: dict | str | None = None,
    max_backfill_days: int | None = 11,
    lookback_hours: int = 6,
    rebuild_latest_geometry: bool = False,
    from_time: datetime | None = None,
):
    if rebuild_latest_geometry:
        rebuild_from_latest_geometry_run(root, zarr_path, chunks=chunks, backup_existing=True, from_time=from_time)
        return

    if not zarr_path.exists():
        start_cutoff = None
        if from_time is not None:
            start_cutoff = parse_from_time(from_time) - timedelta(microseconds=1)
            print(f"Zarr store not found; bootstrapping from samples at or after {parse_from_time(from_time)}.")
        elif max_backfill_days is not None:
            start_cutoff = datetime.now(timezone.utc) - timedelta(days=max_backfill_days)
            print(f"Zarr store not found; bootstrapping from files newer than {start_cutoff}.")
        else:
            print("Zarr store not found; bootstrapping from all matching files.")
        files = _list_files_after(root, start_cutoff)
        if not files:
            print("No radar .LV1.NC files available to bootstrap.")
            return
        _bootstrap_store(files, zarr_path, chunks=chunks, description="bootstrap", from_time=from_time)
        return

    base = xr.open_zarr(zarr_path, chunks={})
    if "time" not in base:
        raise KeyError("Zarr store missing time coordinate")
    base_key, base_count = _existing_range_key(base)
    last_time = _latest_valid_time(base)
    print(f"Latest time in Zarr: {last_time}")
    print(f"Existing radar geometry: {base_count} range gates ({base_key})")

    scan_after = last_time - timedelta(hours=max(lookback_hours, 0))
    from_time_naive = naive_utc(from_time)
    if from_time_naive is not None:
        from_time_aware = from_time_naive.replace(tzinfo=timezone.utc) - timedelta(microseconds=1)
        if from_time_aware > scan_after:
            scan_after = from_time_aware
    files = _list_files_after(root, scan_after)
    if not files:
        print("No new .NC files to append.")
        return

    print(f"Scanning {len(files)} candidate files")
    geometry_records = _scan_geometries(files)
    if not geometry_records:
        print("Candidate files contain no readable radar geometry metadata.")
        return
    latest_record = geometry_records[-1]
    if latest_record.key != base_key:
        print(
            "Detected radar geometry change: "
            f"store has {base_count} gates ({base_key}), "
            f"latest file has {latest_record.range_count} gates ({latest_record.key})."
        )
        rebuild_from_latest_geometry_run(root, zarr_path, chunks=chunks, backup_existing=True, from_time=from_time)
        return
    files = [record.path for record in geometry_records if record.key == base_key]
    skipped = len(geometry_records) - len(files)
    if skipped:
        print(f"Skipping {skipped} candidate files with mismatched radar geometry.")
    # Keep normal appends unchunked until after the time filter.  The lookback
    # scan intentionally reloads older files so we can tolerate late file
    # arrival, but chunking that overlapping dataset before boolean indexing can
    # produce partial/misaligned Zarr writes when appending.  Materializing the
    # small "new only" block avoids variable-specific all-NaN chunk stripes.
    combined = _load_files(files, chunks=None)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no readable radar samples.")
        return
    combined = filter_dataset_from_time(combined, from_time)
    new_time_mask = (combined["time"] > np.datetime64(last_time.replace(tzinfo=None))).values
    combined = combined.isel(time=new_time_mask)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate files contain no samples newer than the existing Zarr.")
        return
    combined = _deduplicate_time(combined.sortby("time")).load()
    combined.to_zarr(zarr_path, mode="a", append_dim="time")
    print("Append complete.")


def main():
    parser = argparse.ArgumentParser(description="Append new cloud radar NC files into existing Zarr.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--chunk-time", type=int, default=400)
    parser.add_argument("--max-backfill-days", type=int, default=11)
    parser.add_argument("--lookback-hours", type=int, default=6)
    parser.add_argument("--from-time", help="Only write samples at or after this UTC ISO timestamp.")
    parser.add_argument(
        "--rebuild-latest-geometry",
        action="store_true",
        help="Back up any existing store and rebuild from the newest contiguous radar geometry run.",
    )
    args = parser.parse_args()

    chunks = {"time": args.chunk_time} if args.chunk_time else None
    append_new(
        args.root,
        args.zarr,
        chunks=chunks,
        max_backfill_days=args.max_backfill_days,
        lookback_hours=args.lookback_hours,
        rebuild_latest_geometry=args.rebuild_latest_geometry,
        from_time=parse_from_time(args.from_time),
    )


if __name__ == "__main__":
    main()
