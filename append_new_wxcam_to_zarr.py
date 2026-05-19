#!/usr/bin/env python3
"""Append new wxcam images into per-stream Zarr groups from the image catalog frontier forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
from PIL import Image
import xarray as xr
import zarr

from wxcam_catalog import WXCAM_IMAGE_TYPES, catalog_frontier, ns_to_datetime, records_after

CATALOG_DEFAULT = Path("/data/aurora/products/wxcam/wxcam_catalog.sqlite")
STATE_DEFAULT = Path("/var/lib/aurora-cloud/wxcam-zarr-state.json")
ZARR_DEFAULT = Path("/data/aurora/products/wxcam/wxcam.zarr")
DEFAULT_BATCH_SIZE = 4
TRANSIENT_FILE_GRACE_SECONDS = 10 * 60


def _ensure_store(zarr_path: Path) -> None:
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs.update(
        {
            "instrument": "wxcam",
            "title": "Aurora wxcam HDR images",
            "storage_policy": "Contains locally retained FISH HDR and PANO HDR JPG image data with timestamps derived from filenames; MP4 products are stored separately.",
        }
    )


def _load_state(state_path: Path) -> dict[str, int] | None:
    if not state_path.exists():
        return None
    try:
        return {key: int(value) for key, value in json.loads(state_path.read_text()).items()}
    except Exception as exc:
        print(f"Could not read wxcam Zarr state {state_path}: {exc}")
        return None


def _save_state(state_path: Path, state: dict[str, int]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _initialize_state(catalog_path: Path, zarr_path: Path, state_path: Path) -> dict[str, int]:
    _ensure_store(zarr_path)
    frontier = catalog_frontier(catalog_path, media_kind="image")
    state = {image_type: int(frontier.get(image_type, 0)) for image_type in WXCAM_IMAGE_TYPES}
    _save_state(state_path, state)
    print("Initialized wxcam pixel-Zarr state at the current catalog frontier.")
    for image_type, last_ns in state.items():
        if last_ns:
            print(f"  {image_type}: {ns_to_datetime(last_ns)}")
    return state


def _batched(records: list, size: int) -> Iterable[list]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _group_exists(zarr_path: Path, image_type: str) -> bool:
    if not zarr_path.exists():
        return False
    root = zarr.open_group(str(zarr_path), mode="a")
    return image_type in root.group_keys()


def _row_age_seconds(row, path: Path | None, now_ts: float) -> float:
    if path is not None:
        try:
            return max(0.0, now_ts - path.stat().st_mtime)
        except FileNotFoundError:
            pass
    return max(0.0, now_ts - (int(row["mtime_ns"]) / 1_000_000_000))


def _build_dataset(
    image_type: str,
    rows: list,
    transient_grace_seconds: int = TRANSIENT_FILE_GRACE_SECONDS,
) -> tuple[xr.Dataset | None, int | None, bool]:
    # Fresh catalog rows can point at files that rsync has created but not finished
    # writing yet. Defer those rows so the Zarr frontier does not skip ahead.
    images: list[np.ndarray] = []
    times: list[np.datetime64] = []
    size_bytes: list[int] = []
    width_values: list[int] = []
    height_values: list[int] = []
    filenames: list[str] = []
    expected_shape: tuple[int, int, int] | None = None
    last_advanced_ns: int | None = None
    deferred = False
    now_ts = time.time()

    for row in rows:
        path = Path(row["raw_path"])
        if not path.exists():
            age_seconds = _row_age_seconds(row, None, now_ts)
            if age_seconds < transient_grace_seconds:
                print(f"Deferring wxcam append until in-flight image arrives: {path}")
                deferred = True
                break
            print(f"Skipping stale missing wxcam image {path}")
            last_advanced_ns = int(row["time_epoch_ns"])
            continue
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            age_seconds = _row_age_seconds(row, path, now_ts)
            if age_seconds < transient_grace_seconds:
                print(f"Deferring wxcam append for fresh unreadable image {path}: {exc}")
                deferred = True
                break
            print(f"Skipping stale unreadable wxcam image {path}: {exc}")
            last_advanced_ns = int(row["time_epoch_ns"])
            continue
        if expected_shape is None:
            expected_shape = rgb.shape
        if rgb.shape != expected_shape:
            age_seconds = _row_age_seconds(row, path, now_ts)
            if age_seconds < transient_grace_seconds:
                print(
                    "Deferring wxcam append for fresh shape-mismatched image "
                    f"{path} ({rgb.shape} != {expected_shape})"
                )
                deferred = True
                break
            print(f"Skipping stale shape-mismatched wxcam image {path} ({rgb.shape} != {expected_shape})")
            last_advanced_ns = int(row["time_epoch_ns"])
            continue
        images.append(rgb)
        times.append(np.datetime64(row["time_utc"].replace(" ", "T")))
        size_bytes.append(int(row["size_bytes"]))
        width_values.append(int(row["width"]))
        height_values.append(int(row["height"]))
        filenames.append(str(row["filename"]))
        last_advanced_ns = int(row["time_epoch_ns"])

    if not images:
        return None, last_advanced_ns, deferred

    stack = np.stack(images, axis=0)
    y_size, x_size = stack.shape[1], stack.shape[2]
    ds = xr.Dataset(
        data_vars={
            "image": (("time", "y", "x", "channel"), stack),
            "size_bytes": (("time",), np.asarray(size_bytes, dtype=np.int64)),
            "width": (("time",), np.asarray(width_values, dtype=np.int32)),
            "height": (("time",), np.asarray(height_values, dtype=np.int32)),
            "filename": (("time",), np.asarray(filenames, dtype="U256")),
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "y": np.arange(y_size, dtype=np.int32),
            "x": np.arange(x_size, dtype=np.int32),
            "channel": np.asarray(["R", "G", "B"], dtype="U1"),
        },
        attrs={
            "instrument": "wxcam",
            "image_type": image_type,
            "label": WXCAM_IMAGE_TYPES[image_type]["label"],
            "stream": WXCAM_IMAGE_TYPES[image_type]["stream"],
        },
    )
    return ds, last_advanced_ns, deferred


def _consolidate(zarr_path: Path) -> None:
    try:
        zarr.consolidate_metadata(str(zarr_path))
    except Exception as exc:
        print(f"Could not consolidate Zarr metadata for {zarr_path}: {exc}")


def append_new(catalog_path: Path, zarr_path: Path, state_path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    if not catalog_path.exists():
        print(f"wxcam catalog does not exist: {catalog_path}")
        return

    state = _load_state(state_path)
    if state is None:
        state = _initialize_state(catalog_path, zarr_path, state_path)
        return

    _ensure_store(zarr_path)
    changed = False
    for image_type in WXCAM_IMAGE_TYPES:
        last_ns = int(state.get(image_type, 0))
        rows = records_after(catalog_path, image_type, last_ns, media_kind="image")
        if not rows:
            print(f"No new wxcam catalog records for {image_type}.")
            continue

        print(f"Appending {len(rows)} new wxcam images for {image_type}")
        group_exists = _group_exists(zarr_path, image_type)
        last_appended_ns = last_ns
        for batch in _batched(rows, max(batch_size, 1)):
            ds, advanced_ns, deferred = _build_dataset(image_type, batch)
            if ds is not None and ds.sizes.get("time", 0) > 0:
                y_chunk = min(ds.sizes["y"], 1024)
                x_chunk = min(ds.sizes["x"], 1024)
                time_chunk = min(ds.sizes["time"], max(batch_size, 1))
                encoding = {
                    "image": {"chunks": (1, y_chunk, x_chunk, ds.sizes["channel"])},
                    "size_bytes": {"chunks": (time_chunk,)},
                    "width": {"chunks": (time_chunk,)},
                    "height": {"chunks": (time_chunk,)},
                    "filename": {"chunks": (time_chunk,)},
                }
                if group_exists:
                    ds.to_zarr(
                        zarr_path,
                        group=image_type,
                        mode="a",
                        append_dim="time",
                    )
                else:
                    ds.to_zarr(
                        zarr_path,
                        group=image_type,
                        mode="a",
                        consolidated=False,
                        encoding=encoding,
                    )
                    group_exists = True
                changed = True
            if advanced_ns is not None:
                last_appended_ns = max(last_appended_ns, advanced_ns)
            if deferred:
                break
        state[image_type] = last_appended_ns

    _save_state(state_path, state)
    if changed:
        _consolidate(zarr_path)
        print(f"wxcam append complete: {zarr_path}")
    else:
        print("No new wxcam images appended.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append new wxcam images into per-stream Zarr groups.")
    parser.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--state", type=Path, default=STATE_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    append_new(args.catalog, args.zarr, args.state, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
