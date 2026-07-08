#!/usr/bin/env python3
"""Build the AURORACam metadata Zarr from MX4 JPEG archives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import xarray as xr

from auroracam_catalog import AURORACAM_CAMERAS, AuroracamRecord, iter_image_records

RAW_DEFAULT = Path("/project/aurora/raw/auroracam")
ZARR_DEFAULT = Path("/data/aurora/products/auroracam/auroracam.zarr")


def _fixed_strings(values: list[str], width: int) -> np.ndarray:
    return np.array([value.encode("utf-8")[:width] for value in values], dtype=f"S{width}")


def _records_to_dataset(records: list[AuroracamRecord], root: Path) -> xr.Dataset:
    camera_index = {camera_id: idx for idx, camera_id in enumerate(AURORACAM_CAMERAS)}
    n = len(records)
    ds = xr.Dataset(
        data_vars={
            "time_epoch_ns": ("record", np.array([record.time_epoch_ns for record in records], dtype="int64")),
            "camera_index": ("record", np.array([camera_index[record.camera_id] for record in records], dtype="int16")),
            "camera_id": ("record", _fixed_strings([record.camera_id for record in records], 64)),
            "camera_label": ("record", _fixed_strings([record.label for record in records], 64)),
            "camera_ip": ("record", _fixed_strings([record.ip for record in records], 32)),
            "day_utc": ("record", _fixed_strings([record.day_utc for record in records], 16)),
            "filename": ("record", _fixed_strings([record.filename for record in records], 160)),
            "relative_path": ("record", _fixed_strings([record.relative_path for record in records], 320)),
            "raw_path": ("record", _fixed_strings([record.raw_path for record in records], 512)),
            "size_bytes": ("record", np.array([record.size_bytes for record in records], dtype="int64")),
            "mtime_ns": ("record", np.array([record.mtime_ns for record in records], dtype="int64")),
        },
        coords={
            "record": np.arange(n, dtype="int64"),
            "time": ("record", np.array([np.datetime64(record.time_utc.replace(" ", "T")) for record in records], dtype="datetime64[ns]")),
            "camera": ("camera", _fixed_strings(list(AURORACAM_CAMERAS), 64)),
        },
        attrs={
            "title": "AURORACam MX4 JPEG metadata index",
            "source": str(root),
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "product_version": "1",
            "description": (
                "Metadata index for MOBOTIX M24 one-minute JPEG files. "
                "Full-resolution images remain as JPEG files served by the dashboard media route."
            ),
        },
    )
    for name, spec in AURORACAM_CAMERAS.items():
        ds.attrs[f"camera_{name}_label"] = spec["label"]
        ds.attrs[f"camera_{name}_ip"] = spec["ip"]
    return ds


def build_zarr(root: Path, zarr_path: Path, *, allow_empty: bool = False) -> Path:
    records = sorted(
        iter_image_records(root),
        key=lambda record: (record.time_epoch_ns, record.camera_id, record.filename),
    )
    if not records and not allow_empty:
        raise SystemExit(f"No AURORACam JPEGs found under {root}")

    ds = _records_to_dataset(records, root)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zarr_path.with_name(f"{zarr_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    ds.to_zarr(tmp_path, mode="w", consolidated=True)
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    tmp_path.rename(zarr_path)
    print(f"Wrote {zarr_path} with {len(records)} records")
    return zarr_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--rebuild", action="store_true", help="Accepted for parity with append scripts; this builder always writes atomically.")
    parser.add_argument("--allow-empty", action="store_true", help="Write an empty product instead of failing when no JPEGs are present.")
    args = parser.parse_args()
    build_zarr(args.root, args.zarr, allow_empty=args.allow_empty)


if __name__ == "__main__":
    main()
