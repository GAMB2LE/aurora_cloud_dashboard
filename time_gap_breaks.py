"""Helpers for rendering visible breaks across time gaps in curtain plots."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


DEFAULT_GAP_FACTOR = float(os.environ.get("AURORA_GAP_BREAK_FACTOR", "3.5"))
DEFAULT_MIN_GAP_SECONDS = float(os.environ.get("AURORA_GAP_BREAK_MIN_SECONDS", "60"))


def insert_time_gap_breaks(
    times,
    values,
    *,
    time_axis: int = -1,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert all-NaN columns around large time gaps.

    Heatmap renderers infer cell widths from neighboring time samples. Without
    explicit missing columns, a sample before a long source outage can be drawn
    halfway across the outage and look like real data. This helper inserts one
    NaN column just after the sample before a gap and one just before the sample
    after the gap, leaving the gap visibly white.
    """
    time_index = pd.DatetimeIndex(pd.to_datetime(times))
    if len(time_index) < 2:
        return np.asarray(time_index.values), np.asarray(values)

    data = np.asarray(values)
    axis = time_axis if time_axis >= 0 else data.ndim + time_axis
    if axis < 0 or axis >= data.ndim or data.shape[axis] != len(time_index):
        return np.asarray(time_index.values), data

    time_values = np.asarray(time_index.values, dtype="datetime64[ns]")
    deltas = np.diff(time_values).astype("timedelta64[ns]").astype(np.int64)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return time_values, data

    cadence_ns = int(np.nanmedian(positive))
    threshold_ns = max(int(cadence_ns * gap_factor), int(min_gap_seconds * 1_000_000_000))
    gap_indices = set(np.flatnonzero(deltas > threshold_ns).tolist())
    if not gap_indices:
        return time_values, data

    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    else:
        data = data.copy()
    moved = np.moveaxis(data, axis, 0)
    nan_row = np.full_like(moved[:1], np.nan)

    out_times: list[np.datetime64] = []
    out_blocks: list[np.ndarray] = []
    for idx, current_time in enumerate(time_values):
        out_times.append(current_time)
        out_blocks.append(moved[idx : idx + 1])
        if idx not in gap_indices:
            continue

        gap_ns = int(deltas[idx])
        pad_ns = min(cadence_ns, max(1, gap_ns // 3))
        first = current_time + np.timedelta64(pad_ns, "ns")
        second = time_values[idx + 1] - np.timedelta64(pad_ns, "ns")
        if first < second:
            out_times.extend([first, second])
            out_blocks.extend([nan_row.copy(), nan_row.copy()])
        else:
            midpoint = current_time + np.timedelta64(max(1, gap_ns // 2), "ns")
            out_times.append(midpoint)
            out_blocks.append(nan_row.copy())

    expanded = np.concatenate(out_blocks, axis=0)
    expanded = np.moveaxis(expanded, 0, axis)
    return np.asarray(out_times, dtype="datetime64[ns]"), expanded
