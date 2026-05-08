#!/usr/bin/env python3
"""Append new wxcam images into per-stream Zarr groups from the image catalog frontier forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def _ensure_store(zarr_path: Path) -> None:
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs.update(
        {
            "instrument": "wxcam",
            "title": "Aurora wxcam HDR images",
            "storage_policy": "Catalog includes historical HDR images; pixel Zarr appends only new images indexed after initialization.",
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
    print("Initialized wxcam pixel-Zarr state at the current catalog frontier; historical images remain catalog-only.")
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


def _build_dataset(image_type: str, rows: list) -> xr.Dataset | None:
    images: list[np.ndarray] = []
    times: list[np.datetime64] = []
    size_bytes: list[int] = []
    width_values: list[int] = []
    height_values: list[int] = []
    filenames: list[str] = []
    expected_shape: tuple[int, int, int] | None = None

    for row in rows:
        path = Path(row["raw_path"])
        if not path.exists():
            print(f"Skipping missing wxcam image {path}")
            continue
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            print(f"Skipping unreadable wxcam image {path}: {exc}")
            continue
        if expected_shape is None:
            expected_shape = rgb.shape
        if rgb.shape != expected_shape:
            print(f"Skipping shape-mismatched {path} ({rgb.shape} != {expected_shape})")
            continue
        images.append(rgb)
        times.append(np.datetime64(row["time_utc"].replace(" ", "T")))
        size_bytes.append(int(row["size_bytes"]))
        width_values.append(int(row["width"]))
        height_values.append(int(row["height"]))
        filenames.append(str(row["filename"]))

    if not images:
        return None

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
    return ds


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
            ds = _build_dataset(image_type, batch)
            if ds is None or ds.sizes.get("time", 0) == 0:
                continue
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
                    safe_chunks=False,
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
            batch_max = max(int(row["time_epoch_ns"]) for row in batch)
            last_appended_ns = max(last_appended_ns, batch_max)
            changed = True
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
