#!/usr/bin/env python3
"""Append ASS PDU outlet CSV files into a time-indexed Zarr store."""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from rebuild_cutoff import cutoff_date, filter_dataset_from_time, parse_from_time

ROOT_DEFAULT = Path("/project/aurora/raw/pdu")
ZARR_DEFAULT = Path("/data/aurora/products/power/pdu.zarr")
FILE_REGEX = re.compile(r"pdu_(\d{2})(\d{2})(\d{4})\.csv$")
OUTLET_COUNT = 8


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
    for path in root.glob("pdu_*.csv"):
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
        print(f"Dropping {len(times) - len(unique_idx)} duplicate PDU time samples")
        ds = ds.isel(time=np.sort(unique_idx))
    return ds


def _outlet_var(outlet: int, metric: str) -> str:
    return f"PDUOutlet{outlet}{metric}"


def _read_file(path: Path) -> xr.Dataset:
    frame = pd.read_csv(
        path,
        usecols=["datetime", "outlet", "name", "state", "amps", "watts"],
        low_memory=False,
    )
    if frame.empty:
        return xr.Dataset()

    frame.columns = [str(col).strip() for col in frame.columns]
    frame["time"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["outlet"] = pd.to_numeric(frame["outlet"], errors="coerce").astype("Int64")
    frame["amps"] = pd.to_numeric(frame["amps"], errors="coerce")
    frame["watts"] = pd.to_numeric(frame["watts"], errors="coerce")
    frame["state_numeric"] = frame["state"].map({"On": 1.0, "Off": 0.0, "on": 1.0, "off": 0.0})
    frame = frame.dropna(subset=["time", "outlet"])
    frame = frame[(frame["outlet"] >= 1) & (frame["outlet"] <= OUTLET_COUNT)]
    if frame.empty:
        return xr.Dataset()

    frame["outlet"] = frame["outlet"].astype(int)
    frame = frame.sort_values(["time", "outlet"])
    time_index = pd.DatetimeIndex(sorted(frame["time"].dropna().unique()))
    data_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    outlet_names: dict[int, str] = {}

    for outlet in range(1, OUTLET_COUNT + 1):
        rows = frame[frame["outlet"] == outlet]
        if rows.empty:
            continue
        names = rows["name"].dropna().astype(str)
        if not names.empty:
            outlet_names[outlet] = names.mode().iloc[0]
        by_time = rows.drop_duplicates("time", keep="last").set_index("time").reindex(time_index)
        for metric, column in (("Amps", "amps"), ("Watts", "watts"), ("State", "state_numeric")):
            values = by_time[column].to_numpy(dtype=np.float32)
            if np.isfinite(values).any():
                data_vars[_outlet_var(outlet, metric)] = (("time",), values)

    if not data_vars:
        return xr.Dataset()

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time_index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "instrument": "pdu",
            "title": "ASS PDU outlet data",
            "source_file": str(path),
            "source": "pdu_DDMMYYYY.csv",
        },
    )
    for outlet, name in outlet_names.items():
        ds.attrs[f"outlet_{outlet}_name"] = name
    for name in ds.data_vars:
        if name.endswith("Amps"):
            ds[name].attrs["units"] = "A"
        elif name.endswith("Watts"):
            ds[name].attrs["units"] = "W"
        elif name.endswith("State"):
            ds[name].attrs["units"] = "state"
            ds[name].attrs["description"] = "PDU outlet state encoded as 1 for On and 0 for Off."
    return _deduplicate_time(ds.sortby("time"))


def _load_files(files: Iterable[Path], chunks: dict[str, int] | None = None) -> xr.Dataset:
    datasets = []
    for path in files:
        print(f"  {path.name}")
        try:
            ds = _read_file(path)
        except Exception as exc:
            print(f"Skipping unreadable PDU file {path}: {exc}")
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
            "instrument": "pdu",
            "title": "ASS PDU outlet data",
            "source": "pdu_DDMMYYYY.csv",
        }
    )
    for ds in datasets:
        for key, value in ds.attrs.items():
            if key.startswith("outlet_") and key.endswith("_name"):
                combined.attrs.setdefault(key, value)
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
        print(f"Dropping new PDU variables not present in existing Zarr: {', '.join(extras)}")
        combined = combined.drop_vars(extras)
    return combined[existing_vars]


def _consolidate(zarr_path: Path) -> None:
    try:
        zarr.consolidate_metadata(str(zarr_path))
    except Exception as exc:
        print(f"Could not consolidate Zarr metadata for {zarr_path}: {exc}")


def append_new(
    root: Path,
    zarr_path: Path,
    chunks: dict[str, int] | None = None,
    lookback_days: int = 2,
    from_time: datetime | None = None,
) -> None:
    if not root.exists():
        print(f"Raw PDU directory does not exist: {root}")
        return

    from_date = cutoff_date(from_time)
    if not zarr_path.exists():
        files = _list_files(root, from_date)
        if not files:
            print("No matching PDU CSV files available to bootstrap.")
            return
        print(f"Bootstrapping PDU Zarr from {len(files)} files")
        combined = _load_files(files, chunks=chunks)
        combined = filter_dataset_from_time(combined, from_time)
        if combined.sizes.get("time", 0) == 0:
            print("No readable PDU samples available to bootstrap.")
            return
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_zarr(zarr_path, mode="w", consolidated=True)
        print("Bootstrap complete.")
        return

    existing = xr.open_zarr(zarr_path, chunks={})
    if "time" not in existing:
        raise KeyError("PDU Zarr store missing time coordinate")
    last_time = existing["time"].max().values
    last_timestamp = pd.Timestamp(last_time)
    print(f"Latest time in PDU Zarr: {last_timestamp}")

    scan_date = (last_timestamp - pd.Timedelta(days=max(lookback_days, 0))).date()
    if from_date is not None:
        scan_date = max(scan_date, from_date)
    files = _list_files(root, scan_date)
    if not files:
        print("No candidate PDU CSV files to append.")
        return

    print(f"Scanning {len(files)} candidate PDU files")
    combined = _load_files(files, chunks=None)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate PDU files contain no readable samples.")
        return
    combined = filter_dataset_from_time(combined, from_time)
    combined = combined.isel(time=(combined["time"] > last_time).values)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate PDU files contain no samples newer than the existing Zarr.")
        return
    combined = _align_to_existing(combined, existing)
    combined = combined.load()
    combined.to_zarr(zarr_path, mode="a", append_dim="time")
    _consolidate(zarr_path)
    print("Append complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append ASS PDU CSV files into a Zarr store.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--chunk-time", type=int, default=1440)
    parser.add_argument("--lookback-days", type=int, default=2)
    parser.add_argument("--from-time", help="Only write samples at or after this UTC ISO timestamp.")
    parser.add_argument("--rebuild", action="store_true", help="Remove the existing Zarr before rebuilding from all raw files.")
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
