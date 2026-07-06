"""Shared UTC time-axis formatting for static quicklook PNGs."""

from __future__ import annotations

from datetime import timezone

import matplotlib.dates as mdates
import pandas as pd


def _clean_time_index(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(times).dropna().sort_values()
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index


def _clean_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _clean_time_limits(x_limits) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if x_limits is None:
        return None
    start, end = (_clean_timestamp(value) for value in x_limits)
    if pd.isna(start) or pd.isna(end) or end <= start:
        return None
    return start, end


def _tick_interval_hours(times: pd.DatetimeIndex, *, max_ticks: int = 16) -> int:
    if len(times) == 0:
        return 1
    span_hours = max((times.max() - times.min()) / pd.Timedelta(hours=1), 1.0)
    max_ticks = max(int(max_ticks), 2)
    if span_hours <= max_ticks:
        return 1
    return max(1, int(span_hours // max_ticks) + 1)


def _label_time_for_window(times: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Choose the tick that should carry the date label.

    Prefer midnight when it is comfortably inside the window. Rolling latest
    quicklooks can start just before midnight, so use noon instead when the
    midnight label would be crowded against the left or right edge.
    """
    if len(times) == 0:
        return None

    start = pd.Timestamp(times.min())
    end = pd.Timestamp(times.max())
    if end <= start:
        return None

    span = end - start
    edge_margin = min(pd.Timedelta(hours=2), span * 0.15)
    midpoint = start + span / 2
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")

    def comfortably_inside(stamp: pd.Timestamp) -> bool:
        return start <= stamp <= end and (stamp - start) >= edge_margin and (end - stamp) >= edge_margin

    for day in days:
        midnight = pd.Timestamp(day)
        if comfortably_inside(midnight):
            return midnight

    noon_candidates = [pd.Timestamp(day) + pd.Timedelta(hours=12) for day in days]
    noon_candidates.sort(key=lambda stamp: abs(stamp - midpoint))
    for noon in noon_candidates:
        if comfortably_inside(noon):
            return noon

    return None


class _DateOnReferenceTickFormatter(mdates.DateFormatter):
    def __init__(self, reference_time: pd.Timestamp | None):
        super().__init__("%H:%M", tz=timezone.utc)
        self.reference_time = reference_time

    def __call__(self, x, pos=None):
        label = super().__call__(x, pos)
        if self.reference_time is None:
            return label

        tick_time = pd.Timestamp(mdates.num2date(x, tz=timezone.utc)).tz_localize(None)
        if abs(tick_time - self.reference_time) <= pd.Timedelta(seconds=30):
            return f"{label}\n{tick_time:%Y-%m-%d}"
        return label


def apply_quicklook_time_axis(
    ax,
    times: pd.DatetimeIndex,
    *,
    interval_hours: int | None = None,
    label_rotation: float = 0,
    label_size: float = 9,
    x_limits=None,
    max_ticks: int = 16,
) -> None:
    """Format a Matplotlib time axis as UTC hours with one inline date label."""
    clean_times = _clean_time_index(times)
    clean_limits = _clean_time_limits(x_limits)
    if clean_limits is not None:
        start, end = clean_limits
        ax.set_xlim(start.to_pydatetime(), end.to_pydatetime())
        clean_times = pd.DatetimeIndex([start, end])
    if interval_hours is None:
        interval_hours = _tick_interval_hours(clean_times, max_ticks=max_ticks)

    if interval_hours <= 24:
        tick_hours = list(range(0, 24, max(interval_hours, 1)))
        locator = mdates.HourLocator(byhour=tick_hours, tz=timezone.utc)
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=max(int(max_ticks), 2), tz=timezone.utc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(_DateOnReferenceTickFormatter(_label_time_for_window(clean_times)))
    ax.tick_params(axis="x", labelrotation=label_rotation, labelsize=label_size)
    for label in ax.get_xticklabels():
        label.set_rotation(label_rotation)
        label.set_ha("right" if label_rotation else "center")
    ax.set_xlabel("Time (UTC)")
