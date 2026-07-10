#!/usr/bin/env python3
"""
Append any new NetCDF files into an existing Zarr store.

This script:
1) Opens the target Zarr store to find the most recent time coordinate.
2) Finds NetCDF files matching a pattern whose encoded timestamp is newer.
3) Appends those files along the time dimension.

Filenames are expected to contain a timestamp like YYYYMMDD_HHMMSS, e.g.:
    gamb2le_depolarisation_lidar_ceilometer_aurora_20251203_000056.nc
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from rebuild_cutoff import parse_from_time
from netcdf_to_zarr import (
    choose_engine,
    extract_datetime_from_name,
    make_zarr_from_netcdf,
    _filter_time_floor,
)


def _sort_and_deduplicate(ds, dim):
    if dim not in ds.coords:
        return ds
    ds = ds.sortby(dim)
    values = np.asarray(ds[dim].values)
    _, unique_idx = np.unique(values, return_index=True)
    if len(unique_idx) != len(values):
        ds = ds.isel({dim: np.sort(unique_idx)})
    return ds


def _filter_readable(files, engine):
    """
    Return (readable_files, skipped) after cheap header/size probes.
    """
    readable = []
    skipped = []
    for f in files:
        size = os.path.getsize(f)
        if size == 0:
            skipped.append((f, "empty file"))
            print(f"Skipping unreadable file {os.path.basename(f)}: empty file")
            continue
        try:
            with open(f, "rb") as fh:
                sig = fh.read(8)
            if not (sig.startswith(b"CDF") or sig == b"\x89HDF\r\n\x1a\n"):
                skipped.append((f, "unexpected header"))
                print(f"Skipping unreadable file {os.path.basename(f)}: unexpected header")
                continue
            readable.append(f)
        except Exception as exc:
            skipped.append((f, exc))
            print(f"Skipping unreadable file {os.path.basename(f)}: {exc}")
    return readable, skipped


def _drop_bad_files(files, engine):
    """
    Attempt to open each file with xarray; return (good, skipped).
    """
    good = []
    skipped = []
    for f in files:
        try:
            xr.open_dataset(f, engine=engine).close()
            good.append(f)
        except Exception as exc:
            skipped.append((f, exc))
            print(f"Skipping unreadable file {os.path.basename(f)}: {exc}")
    return good, skipped


def append_new_files(
    input_dir,
    pattern,
    zarr_path,
    chunks="auto",
    engine="h5netcdf",
    append_dim="time",
    max_backfill_days=11,
    batch_size=200,
    from_time=None,
):
    engine = choose_engine(engine)
    parsed_from_time = parse_from_time(from_time)
    from_timestamp = pd.Timestamp(parsed_from_time) if parsed_from_time is not None else None
    if from_timestamp is not None:
        from_timestamp = from_timestamp.tz_convert("UTC").tz_localize(None)

    file_pattern = os.path.join(input_dir, pattern)
    all_files = sorted(glob.glob(file_pattern))
    if not all_files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    if not os.path.isdir(zarr_path):
        start_date = None
        start_time = None
        if from_timestamp is not None:
            start_date = from_timestamp.date()
            start_time = from_timestamp.to_pydatetime()
            print(f"Zarr store not found; bootstrapping from samples at or after {from_timestamp.isoformat()}.")
        elif max_backfill_days is not None:
            start_date = (
                pd.Timestamp.utcnow().replace(tzinfo=None) - pd.Timedelta(days=max_backfill_days)
            ).date()
            print(f"Zarr store not found; bootstrapping from files on/after {start_date}.")
        else:
            print("Zarr store not found; bootstrapping from all matching files.")
        make_zarr_from_netcdf(
            input_dir=input_dir,
            pattern=pattern,
            output_zarr=zarr_path,
            chunks=chunks,
            engine=engine,
            start_date=start_date,
            start_time=start_time,
        )
        return

    ds_existing = xr.open_zarr(zarr_path, chunks={})
    if append_dim not in ds_existing:
        raise ValueError(f"Append dimension '{append_dim}' not found in Zarr store.")

    last_time = pd.Timestamp(ds_existing[append_dim].max().values)
    print(f"Latest {append_dim} in Zarr: {last_time.isoformat()}")

    start_cutoff = last_time
    if from_timestamp is not None and from_timestamp > start_cutoff:
        start_cutoff = from_timestamp
        print(f"Limiting append scan to samples at or after {start_cutoff}")
    if max_backfill_days is not None:
        cutoff = pd.Timestamp.utcnow().replace(tzinfo=None) - pd.Timedelta(
            days=max_backfill_days
        )
        if cutoff > start_cutoff:
            start_cutoff = cutoff
            print(
                f"Limiting backfill to files on/after {start_cutoff} "
                f"(max_backfill_days={max_backfill_days})"
            )

    new_files = []
    for f in all_files:
        ts = extract_datetime_from_name(f)
        if ts is None:
            print(f"Skipping file without parsable timestamp: {f}")
            continue
        if pd.Timestamp(ts) > start_cutoff:
            new_files.append((ts, f))

    if not new_files:
        print("No new files to append.")
        return

    new_files.sort(key=lambda x: x[0])
    ordered_files = [f for _, f in new_files]

    readable_files, skipped = _filter_readable(ordered_files, engine)
    if skipped:
        print(f"Skipped {len(skipped)} unreadable files.")
    if not readable_files:
        print("No readable new files to append.")
        return

    print(f"Appending {len(readable_files)} new files:")
    for f in readable_files:
        print("  ", os.path.basename(f))

    total = len(readable_files)
    batches = [
        readable_files[i : i + batch_size] for i in range(0, total, batch_size)
    ]
    for idx, batch in enumerate(batches, start=1):
        print(f"Writing batch {idx}/{len(batches)} ({len(batch)} files)...")
        batch_files = list(batch)
        while batch_files:
            try:
                ds_new = xr.open_mfdataset(
                    batch_files,
                    combine="nested",
                    concat_dim=append_dim,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    chunks=None,
                    engine=engine,
                    parallel=False,
                )
                break
            except Exception:
                batch_files, newly_skipped = _drop_bad_files(batch_files, engine)
                skipped.extend(newly_skipped)
                if not batch_files:
                    print("Batch became empty after dropping unreadable files; skipping batch.")
                    ds_new = None
                    break
        if ds_new is None:
            continue
        ds_new = _sort_and_deduplicate(ds_new, append_dim)
        ds_new = _filter_time_floor(ds_new, time_floor=start_cutoff, time_dim=append_dim)
        new_time_mask = (ds_new[append_dim] > last_time.to_datetime64()).values
        ds_new = ds_new.isel({append_dim: new_time_mask})
        if ds_new.sizes.get(append_dim, 0) == 0:
            print(
                "Batch contains no samples newer than the existing Zarr; "
                "skipping batch."
            )
            ds_new.close()
            continue
        # Appends deliberately materialize the already-filtered new samples.
        # Chunking the larger overlap scan before the boolean time filter can
        # create partial chunk writes and all-NaN stripes in range-resolved
        # products such as CL61.
        ds_new = ds_new.load()
        ds_new.to_zarr(
            zarr_path,
            mode="a",
            append_dim=append_dim,
        )
        ds_new.close()
    print("Append complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append or bootstrap CL61 NetCDF files into a Zarr store.")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("CEILOMETER_DIR", "/mnt/data/cl61")))
    parser.add_argument("--pattern", default="gamb2le_depolarisation_lidar_ceilometer_aurora_*.nc")
    parser.add_argument(
        "--zarr",
        type=Path,
        default=Path(
            os.environ.get(
                "CEILOMETER_ZARR_PATH",
                "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr",
            )
        ),
    )
    parser.add_argument("--chunk-time", type=int, default=30)
    parser.add_argument("--engine", default="h5netcdf")
    parser.add_argument("--append-dim", default="time")
    parser.add_argument("--max-backfill-days", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--from-time", help="Only write samples at or after this UTC ISO timestamp.")
    args = parser.parse_args()

    append_new_files(
        input_dir=str(args.input_dir),
        pattern=args.pattern,
        zarr_path=str(args.zarr),
        chunks={"time": args.chunk_time} if args.chunk_time else "auto",
        engine=args.engine,
        append_dim=args.append_dim,
        max_backfill_days=args.max_backfill_days,
        batch_size=args.batch_size,
        from_time=args.from_time,
    )
