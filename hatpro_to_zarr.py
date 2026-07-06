#!/usr/bin/env python3
"""
Build a consolidated Zarr store for the RPG HATPRO scanning microwave radiometer.

Aggregates:
- LWP from *.LWP.NC
- IWV from *.IWV.NC
- IRR_Map (squeezed to 1D) from *.IRT.NC
- Temperature profile (T_prof) from non-CMP *.TPC.NC and *.TPB.NC
- Composite temperature profile (T_prof) from *.CMP.TPC.NC

Outputs /data/aurora/products/hatprog5/hatpro.zarr by default, chunked along time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
from typing import Iterable, List

import numpy as np
import pandas as pd
import xarray as xr
import zarr

ROOT_DEFAULT = Path("/project/aurora/raw/hatprog5")
ZARR_DEFAULT = Path("/data/aurora/products/hatprog5/hatpro.zarr")
FILE_REGEX = re.compile(r"_(\d{6})_(\d{6})")


def _parse_file_timestamp(path: Path) -> datetime | None:
    match = FILE_REGEX.search(path.name)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return datetime.strptime(date_part + time_part, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _prefer_nested_paths(root: Path, files: List[Path]) -> List[Path]:
    """Prefer canonical Yyyyy/Mmm/Ddd files over legacy flat mirror duplicates."""
    buckets: dict[str, list[Path]] = {}
    for path in files:
        buckets.setdefault(path.name, []).append(path)

    selected: list[Path] = []
    skipped_flat = 0
    for paths in buckets.values():
        nested = [path for path in paths if len(path.relative_to(root).parts) > 1]
        if nested:
            selected.extend(sorted(nested))
            skipped_flat += len(paths) - len(nested)
        else:
            selected.extend(sorted(paths))
    if skipped_flat:
        print(f"[normalize] Ignoring {skipped_flat} legacy flat duplicate files")
    return selected


def _list_files(root: Path, pattern: str, since: datetime | None = None) -> List[Path]:
    files = [path for path in root.rglob(pattern) if path.is_file()]
    files = _prefer_nested_paths(root, files)
    if since is not None:
        files = [
            path
            for path in files
            if (file_time := _parse_file_timestamp(path)) is not None and file_time >= since
        ]
    files.sort(key=lambda path: (_parse_file_timestamp(path) or datetime.min.replace(tzinfo=timezone.utc), path.as_posix()))
    return files


def _open_sorted_concat(files: List[Path], *, chunk_time: int, chunk_range: int | None = None) -> xr.Dataset:
    """
    Open a list of NetCDFs and concatenate along time after sorting each file.
    Avoids combine_by_coords monotonicity errors when timestamps overlap or wobble.
    """
    if not files:
        raise ValueError("No files provided")
    datasets = []
    for f in sorted(files):
        chunks = {"time": chunk_time}
        if chunk_range:
            chunks["altitude_layer"] = chunk_range
        ds = xr.open_dataset(f, chunks=chunks, decode_timedelta=False)
        if "time" in ds:
            ds = ds.sortby("time")
        datasets.append(ds)
    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
    )
    return combined


def _load_timeseries(root: Path, pattern: str, var: str, chunk_time: int, since: datetime | None = None) -> xr.Dataset | None:
    files = _list_files(root, pattern, since)
    if not files:
        print(f"[skip] {pattern}: no files found")
        return None
    print(f"[load] {pattern}: {len(files)} files")
    ds = _open_sorted_concat(files, chunk_time=chunk_time)
    if var not in ds:
        print(f"[warn] {pattern}: missing variable {var}")
        return None
    da = ds[var].sortby("time")
    # Collapse wavelength dimension if present (IRR_Map has number_wavelength=1)
    da = da.squeeze(drop=True)
    if da.ndim == 2 and da.shape[1] > 1:
        da = da.mean(dim=da.dims[1])
    return da.to_dataset(name=var)


def _load_tprof_files(files: List[Path], label: str, output_name: str, chunk_time: int, chunk_range: int) -> xr.Dataset | None:
    if not files:
        print(f"[skip] {label}: no files found")
        return None
    print(f"[load] {label}: {len(files)} files")
    ds = _open_sorted_concat(files, chunk_time=chunk_time, chunk_range=chunk_range)
    if "T_prof" not in ds:
        print(f"[warn] {label}: T_prof variable missing")
        return None
    tprof = ds["T_prof"].rename({"altitude_layer": "range"}).sortby("time")
    if "altitude" in ds:
        alt = ds["altitude"].rename({"altitude_layer": "range"})
        tprof = tprof.assign_coords(range=alt)
    return tprof.to_dataset(name=output_name)


def _load_tprof(root: Path, patterns: Iterable[str], chunk_time: int, chunk_range: int) -> xr.Dataset | None:
    files: List[Path] = []
    for pat in patterns:
        files.extend(_list_files(root, pat))
    files = [f for f in files if not f.name.endswith(".CMP.TPC.NC")]
    return _load_tprof_files(files, "T_PROF", "T_PROF", chunk_time, chunk_range)


def _load_tprof_since(root: Path, patterns: Iterable[str], chunk_time: int, chunk_range: int, since: datetime | None = None) -> xr.Dataset | None:
    files: List[Path] = []
    for pat in patterns:
        files.extend(_list_files(root, pat, since))
    files = [f for f in files if not f.name.endswith(".CMP.TPC.NC")]
    return _load_tprof_files(files, "T_PROF", "T_PROF", chunk_time, chunk_range)


def _load_cmp_tprof(root: Path, chunk_time: int, chunk_range: int, since: datetime | None = None) -> xr.Dataset | None:
    files = _list_files(root, "*.CMP.TPC.NC", since)
    return _load_tprof_files(files, "T_PROF_CMP", "T_PROF_CMP", chunk_time, chunk_range)


def _load_met(root: Path, chunk_time: int, since: datetime | None = None) -> xr.Dataset | None:
    files = _list_files(root, "*.MET.NC", since)
    if not files:
        print("[skip] MET: no files found")
        return None
    print(f"[load] MET: {len(files)} files")
    ds = _open_sorted_concat(files, chunk_time=chunk_time)
    if "Surf_T" not in ds:
        print("[warn] MET: missing Surf_T")
        return None
    surf = ds["Surf_T"].rename("SURF_T").sortby("time")
    return surf.to_dataset()


def _deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    times = np.asarray(ds["time"].values)
    _, unique_idx = np.unique(times, return_index=True)
    if len(unique_idx) != len(times):
        print(f"[dedup] Dropping {len(times) - len(unique_idx)} duplicate time stamps")
        ds = ds.isel(time=np.sort(unique_idx))
    return ds


def _build_dataset(
    root: Path,
    chunk_time: int = 600,
    chunk_range: int = 48,
    since: datetime | None = None,
    allow_empty: bool = False,
) -> xr.Dataset:
    parts: List[xr.Dataset] = []

    for pattern, var in [
        ("*.LWP.NC", "LWP"),
        ("*.IWV.NC", "IWV"),
        ("*.IRT.NC", "IRR_Map"),
    ]:
        ds = _load_timeseries(root, pattern, var, chunk_time, since)
        if ds is not None:
            parts.append(ds)

    tprof_ds = _load_tprof_since(root, ("*.TPC.NC", "*.TPB.NC"), chunk_time, chunk_range, since)
    if tprof_ds is not None:
        parts.append(tprof_ds)

    cmp_tprof_ds = _load_cmp_tprof(root, chunk_time, chunk_range, since)
    if cmp_tprof_ds is not None:
        parts.append(cmp_tprof_ds)

    met_ds = _load_met(root, chunk_time, since)
    if met_ds is not None:
        parts.append(met_ds)

    if not parts:
        if allow_empty:
            return xr.Dataset()
        raise SystemExit("No datasets loaded; nothing to write")

    # Deduplicate time in each component before alignment to avoid merge errors.
    parts = [_deduplicate_time(p) for p in parts]
    merged = xr.merge(parts, combine_attrs="override")
    merged = merged.sortby("time")
    merged = _deduplicate_time(merged)

    # Apply chunking hints
    chunk_map = {"time": chunk_time}
    if "range" in merged.dims:
        chunk_map["range"] = chunk_range
    merged = merged.chunk(chunk_map)
    merged.attrs.update(
        {
            "title": "AURORA HATPRO G5 scanning microwave radiometer",
            "source": "Mirrored raw HATPRO files from aurora@100.124.55.22:/home/aurora/data/hatprog5",
            "raw_mirror": str(root),
            "profile_policy": "T_PROF uses non-CMP TPC plus TPB files; T_PROF_CMP uses CMP.TPC files.",
            "created_by": "hatpro_to_zarr.py",
        }
    )
    return merged


def _consolidate(zarr_path: Path) -> None:
    try:
        zarr.consolidate_metadata(str(zarr_path))
    except Exception as exc:
        print(f"Could not consolidate Zarr metadata for {zarr_path}: {exc}")


def _nan_fill(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return np.full(shape, np.nan, dtype=dtype)
    return np.full(shape, np.nan, dtype=np.float32)


def _align_to_existing(combined: xr.Dataset, existing: xr.Dataset) -> xr.Dataset:
    for coord_name, coord in existing.coords.items():
        if coord_name == "time":
            continue
        if coord_name in combined.coords:
            if not np.array_equal(np.asarray(combined[coord_name].values), np.asarray(coord.values)):
                combined = combined.reindex({coord_name: coord.values})
        elif coord_name in combined.dims:
            combined = combined.assign_coords({coord_name: coord})

    existing_vars = list(existing.data_vars)
    time_len = combined.sizes.get("time", 0)
    for name in existing_vars:
        if name in combined:
            continue
        template = existing[name]
        shape: list[int] = []
        for dim in template.dims:
            shape.append(time_len if dim == "time" else existing.sizes[dim])
        combined[name] = (template.dims, _nan_fill(tuple(shape), template.dtype))

    extras = [name for name in combined.data_vars if name not in existing_vars]
    if extras:
        print(f"[schema] Dropping new variables not present in existing Zarr: {', '.join(extras)}")
        combined = combined.drop_vars(extras)
    return combined[existing_vars]


def build_zarr(root: Path, zarr_path: Path, chunk_time: int = 600, chunk_range: int = 48):
    merged = _build_dataset(root, chunk_time=chunk_time, chunk_range=chunk_range)

    tmp_path = zarr_path.with_name(f"{zarr_path.name}.tmp")
    old_path = zarr_path.with_name(f"{zarr_path.name}.old")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if old_path.exists():
        shutil.rmtree(old_path)

    print(f"[write] {tmp_path}")
    merged.to_zarr(tmp_path, mode="w", consolidated=True)
    if zarr_path.exists():
        zarr_path.rename(old_path)
    tmp_path.rename(zarr_path)
    if old_path.exists():
        shutil.rmtree(old_path)
    print("[done]")


def append_new(root: Path, zarr_path: Path, chunk_time: int = 600, chunk_range: int = 48, lookback_hours: int = 6) -> None:
    if not root.exists():
        print(f"Raw HATPRO directory does not exist: {root}")
        return
    if not zarr_path.exists():
        print("No existing HATPRO Zarr found; bootstrapping from raw files.")
        build_zarr(root, zarr_path, chunk_time=chunk_time, chunk_range=chunk_range)
        return

    existing = xr.open_zarr(zarr_path, chunks={})
    if "time" not in existing:
        raise KeyError("Zarr store missing time coordinate")
    last_time = pd.Timestamp(existing["time"].max().values).to_pydatetime()
    if last_time.tzinfo is None:
        last_time_utc = last_time.replace(tzinfo=timezone.utc)
    else:
        last_time_utc = last_time.astimezone(timezone.utc)
    print(f"Latest time in Zarr: {last_time_utc.isoformat()}")

    scan_start = last_time_utc - timedelta(hours=max(lookback_hours, 0))
    combined = _build_dataset(root, chunk_time=chunk_time, chunk_range=chunk_range, since=scan_start, allow_empty=True)
    if combined.sizes.get("time", 0) == 0:
        print("No candidate HATPRO files to append.")
        return
    last_np = np.datetime64(last_time_utc.replace(tzinfo=None))
    combined = combined.isel(time=(combined["time"] > last_np).values)
    if combined.sizes.get("time", 0) == 0:
        print("Candidate HATPRO files contain no samples newer than the existing Zarr.")
        return

    combined = _align_to_existing(combined, existing)
    combined = combined.load()
    combined.to_zarr(zarr_path, mode="a", append_dim="time")
    _consolidate(zarr_path)
    print("[append done]")


def main():
    parser = argparse.ArgumentParser(description="Build HATPRO radiometer Zarr store")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="Root directory with HATPRO files")
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT, help="Output Zarr path")
    parser.add_argument("--chunk-time", type=int, default=600, help="Time chunk size")
    parser.add_argument("--chunk-range", type=int, default=48, help="Range/altitude chunk size")
    parser.add_argument("--lookback-hours", type=int, default=6, help="Hours before the existing Zarr frontier to rescan when appending")
    parser.add_argument("--rebuild", action="store_true", help="Rewrite the full HATPRO Zarr from all raw files")
    args = parser.parse_args()

    if args.rebuild:
        build_zarr(args.root, args.zarr, chunk_time=args.chunk_time, chunk_range=args.chunk_range)
    else:
        append_new(
            args.root,
            args.zarr,
            chunk_time=args.chunk_time,
            chunk_range=args.chunk_range,
            lookback_hours=args.lookback_hours,
        )


if __name__ == "__main__":
    main()
