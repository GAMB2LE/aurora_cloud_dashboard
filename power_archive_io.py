"""Lossless serialization of small forecast archives and issue metadata."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr


def prepare_archive_for_storage(dataset: xr.Dataset) -> xr.Dataset:
    """Discard source-store encodings after concatenation or schema changes.

    Xarray retains a reopened variable's encoding even when concatenation has
    widened its in-memory dtype. In particular, legacy ``<U3`` (often inferred
    from missing ``nan`` values) silently truncates new identity strings.
    Old chunk sizes can also conflict with the archive's new Dask chunks.
    """
    prepared = dataset.copy()
    for name in list(prepared.variables):
        variable = prepared[name]
        if variable.dtype.kind in "OUS":
            # Only small metadata variables are materialized here. Preserve
            # legacy missing-value strings; never fabricate a missing identity.
            values = np.asarray(variable.values).astype(str)
            prepared[name] = xr.DataArray(
                values, dims=variable.dims, attrs=dict(variable.attrs)
            )
        prepared[name].encoding = {}
        if prepared[name].dtype.kind == "M":
            # Xarray's datetime encoder must see the same unit in memory as
            # on disk (microsecond arrays plus nanosecond encoding can shift
            # the stored epoch in older xarray versions).
            prepared[name] = prepared[name].astype("datetime64[ns]")
            prepared[name].encoding = {}
            prepared[name].encoding.update(
                units="nanoseconds since 1970-01-01", dtype="int64"
            )
    return prepared


def validate_archive_roundtrip(expected: xr.Dataset, persisted: xr.Dataset) -> None:
    """Reject publication if any identity or coordinate changed on disk."""
    if dict(expected.sizes) != dict(persisted.sizes):
        raise ValueError("Forecast archive dimensions changed during serialization")
    if set(expected.variables) != set(persisted.variables):
        raise ValueError("Forecast archive schema changed during serialization")
    for name, variable in expected.variables.items():
        if variable.dtype.kind not in "OUSM" and name not in expected.coords:
            continue
        left = np.asarray(variable.values)
        right = np.asarray(persisted[name].values)
        if left.dtype.kind == "M":
            equal = np.array_equal(left.astype("datetime64[ns]").astype("int64"),
                                   right.astype("datetime64[ns]").astype("int64"))
        elif left.dtype.kind in "OUS":
            equal = np.array_equal(left.astype(str), right.astype(str))
        else:
            equal = np.array_equal(left, right, equal_nan=True)
        if not equal:
            raise ValueError(f"Forecast archive round-trip mismatch: {name}")


def write_forecast_archive(dataset: xr.Dataset, output_zarr: Path) -> None:
    """Stage and verify an archive before replacing an existing store."""
    output = Path(output_zarr)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Forecast archive output must be a direct directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare_archive_for_storage(dataset)
    chunks = {
        dim: min(max(size, 1), 64)
        for dim, size in prepared.sizes.items()
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    backup: Path | None = None
    try:
        prepared.chunk(chunks).to_zarr(staging, mode="w", consolidated=True)
        with xr.open_zarr(staging, chunks={}, consolidated=True) as reopened:
            validate_archive_roundtrip(prepared, reopened)
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
