"""Panel-free dataset preparation services for dashboard renderers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, MutableMapping

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class WindowRequest:
    """A bounded dataset request prepared by a tab controller."""

    start: datetime
    end: datetime
    bottom_m: float | None = None
    top_m: float | None = None
    height_load_max: float = 10_000
    render_quality: str = "full"


def coarsen_targets(
    duration: timedelta | None,
    height_span: float | None,
    *,
    base_time_subsample: int = 2,
    base_time_target: int = 300,
    base_height_target: int = 200,
) -> tuple[int, int, int]:
    """Return sampling targets that retain detail for short or shallow windows."""
    time_subsample = base_time_subsample
    time_target = base_time_target
    height_target = base_height_target
    if duration is not None:
        hours = duration.total_seconds() / 3600.0
        if hours <= 2:
            time_subsample = 1
            time_target = 1200
        elif hours <= 6:
            time_subsample = 1
            time_target = 800
        elif hours <= 24:
            time_subsample = 1
            time_target = 400
    if height_span is not None:
        if height_span <= 1000:
            height_target = 400
        elif height_span <= 3000:
            height_target = 300
    return time_subsample, time_target, height_target


def prepare_dataset_window(
    base: xr.Dataset | None,
    request: WindowRequest,
    *,
    valid_time_mask,
    perf: MutableMapping[str, Any] | None = None,
) -> xr.Dataset:
    """Slice and coarsen one dataset without accessing Panel or global state."""
    metrics = perf if perf is not None else {}
    if request.start >= request.end:
        metrics["status"] = "invalid_window"
        return xr.Dataset()
    if base is None:
        metrics["status"] = "no_dataset"
        return xr.Dataset()

    duration = request.end - request.start
    metrics["window_hours"] = round(duration.total_seconds() / 3600.0, 3)
    height_span = None
    if request.bottom_m is not None or request.top_m is not None:
        bottom = max(request.bottom_m or 0.0, 0.0)
        top = request.top_m if request.top_m is not None else request.height_load_max
        height_span = max(top - bottom, 0.0)
    time_subsample, time_target, height_target = coarsen_targets(duration, height_span)
    if request.render_quality == "coarse":
        time_subsample = max(time_subsample, 2)
        time_target = max(96, time_target // 3)
        height_target = max(72, height_target // 2)
    preserve_time_detail = request.render_quality == "summary_full_time"
    metrics.update(
        time_subsample=int(time_subsample),
        time_target=int(time_target),
        height_target=int(height_target),
        preserve_time_detail=bool(preserve_time_detail),
        base_time_count=int(base.sizes.get("time", 0)),
        base_range_count=int(base.sizes.get("range", 0)),
    )

    phase_start = perf_counter()
    try:
        times = base["time"].values
        mask = (
            valid_time_mask(times)
            & (times >= np.datetime64(request.start))
            & (times <= np.datetime64(request.end))
        )
        metrics["matched_time_count"] = int(np.count_nonzero(mask))
        if not np.any(mask):
            metrics["status"] = "no_match"
            metrics["select_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)
            return xr.Dataset()
        dataset = base.isel(time=np.nonzero(mask)[0])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        dataset = base
        metrics["time_select_error"] = str(exc)
    metrics["select_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)

    has_range = "range" in dataset.coords or "range" in dataset.dims
    phase_start = perf_counter()
    if has_range:
        try:
            dataset = dataset.sel({"range": slice(0, request.height_load_max)})
        except (KeyError, TypeError, ValueError):
            dataset = dataset.where(dataset["range"] <= request.height_load_max, drop=True)
    if has_range and (request.bottom_m is not None or request.top_m is not None):
        low = max(request.bottom_m or 0.0, 0.0)
        high = min(
            request.top_m or request.height_load_max,
            request.height_load_max,
        )
        try:
            dataset = dataset.sel({"range": slice(low, high)})
        except (KeyError, TypeError, ValueError):
            dataset = dataset.where(
                (dataset["range"] >= low) & (dataset["range"] <= high),
                drop=True,
            )
    metrics["range_filter_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)

    phase_start = perf_counter()
    if not preserve_time_detail and time_subsample > 1:
        dataset = dataset.isel(time=slice(None, None, time_subsample))
    try:
        if dataset.sizes.get("range", 0) > height_target:
            factor = max(int(np.ceil(dataset.sizes["range"] / height_target)), 1)
            metrics["range_coarsen_factor"] = factor
            dataset = dataset.coarsen({"range": factor}, boundary="trim").mean()
        if not preserve_time_detail and dataset.sizes.get("time", 0) > time_target:
            factor = max(int(np.ceil(dataset.sizes["time"] / time_target)), 1)
            metrics["time_coarsen_factor"] = factor
            dataset = dataset.coarsen({"time": factor}, boundary="trim").mean()
    except (KeyError, TypeError, ValueError) as exc:
        metrics["coarsen_error"] = str(exc)
    metrics["coarsen_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)
    metrics["output_time_count"] = int(dataset.sizes.get("time", 0))
    metrics["output_range_count"] = int(dataset.sizes.get("range", 0))
    metrics["status"] = "ok"
    return dataset
