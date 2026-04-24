#!/usr/bin/env python3
"""
Build a consolidated Zarr store for the RPG HATPRO scanning microwave radiometer.

Aggregates:
- LWP from *.LWP.NC
- IWV from *.IWV.NC
- IRR_Map (squeezed to 1D) from *.IRT.NC
- Temperature profile (T_prof) from *.TPC.NC and *.TPB.NC

Outputs /mnt/data/ass/hatprog5/hatpro.zarr by default, chunked along time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import xarray as xr

ROOT_DEFAULT = Path("/mnt/data/ass/hatprog5")
ZARR_DEFAULT = Path("/mnt/data/ass/hatprog5/hatpro.zarr")


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
        ds = xr.open_dataset(f, chunks=chunks)
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


def _load_timeseries(root: Path, pattern: str, var: str, chunk_time: int) -> xr.Dataset | None:
    files = sorted(root.rglob(pattern))
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


def _load_tprof(root: Path, patterns: Iterable[str], chunk_time: int, chunk_range: int) -> xr.Dataset | None:
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    if not files:
        print("[skip] T_prof: no files found")
        return None
    print(f"[load] T_prof: {len(files)} files")
    ds = _open_sorted_concat(files, chunk_time=chunk_time, chunk_range=chunk_range)
    if "T_prof" not in ds:
        print("[warn] T_prof variable missing")
        return None
    tprof = ds["T_prof"].rename({"altitude_layer": "range"}).sortby("time")
    if "altitude" in ds:
        alt = ds["altitude"].rename({"altitude_layer": "range"})
        tprof = tprof.assign_coords(range=alt)
    return tprof.to_dataset(name="T_PROF")


def _load_met(root: Path, chunk_time: int) -> xr.Dataset | None:
    files = sorted(root.rglob("*.MET.NC"))
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


def build_zarr(root: Path, zarr_path: Path, chunk_time: int = 600, chunk_range: int = 48):
    parts: List[xr.Dataset] = []

    for pattern, var in [
        ("*.LWP.NC", "LWP"),
        ("*.IWV.NC", "IWV"),
        ("*.IRT.NC", "IRR_Map"),
    ]:
        ds = _load_timeseries(root, pattern, var, chunk_time)
        if ds is not None:
            parts.append(ds)

    tprof_ds = _load_tprof(root, ("*.TPC.NC", "*.TPB.NC"), chunk_time, chunk_range)
    if tprof_ds is not None:
        parts.append(tprof_ds)

    met_ds = _load_met(root, chunk_time)
    if met_ds is not None:
        parts.append(met_ds)

    if not parts:
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

    print(f"[write] {zarr_path}")
    merged.to_zarr(zarr_path, mode="w", consolidated=True)
    print("[done]")


def main():
    parser = argparse.ArgumentParser(description="Build HATPRO radiometer Zarr store")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="Root directory with HATPRO files")
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT, help="Output Zarr path")
    parser.add_argument("--chunk-time", type=int, default=600, help="Time chunk size")
    parser.add_argument("--chunk-range", type=int, default=48, help="Range/altitude chunk size")
    args = parser.parse_args()

    build_zarr(args.root, args.zarr, chunk_time=args.chunk_time, chunk_range=args.chunk_range)


if __name__ == "__main__":
    main()
