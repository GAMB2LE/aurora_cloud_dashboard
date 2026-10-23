"""Measured charger contribution; no device control or independent services."""
from __future__ import annotations

import numpy as np
import pandas as pd

GRID_FIELDS = ("GridWatts", "MultiPlusDCWatts")
GRID_CONTRACT = "measured_multiplus_dc_v1"


def station_load(frame: pd.DataFrame) -> pd.Series:
    solar = frame.reindex(columns=[f"SolarWatts_{side}" for side in ("East", "South", "West")]).sum(axis=1, min_count=3)
    battery = frame.get("BatteryWatts", pd.Series(np.nan, index=frame.index))
    charger = frame.get("MultiPlusDCWatts", pd.Series(np.nan, index=frame.index))
    # Negative MultiPlus power is consumption, already included in battery
    # discharge. Only its positive supply needs adding back to infer demand.
    load = solar + charger.clip(lower=0) - battery
    return load.where(np.isfinite(load) & (load >= 0)).rename("ObservedLoadWatts")


def continued_grid_rate(frame: pd.DataFrame, anchor: pd.Timestamp, *, now: pd.Timestamp | None = None,
                        max_age_minutes: float = 20.0) -> tuple[float | None, str]:
    """A conditional rate, not a prediction that grid supply will persist."""
    anchor = pd.Timestamp(anchor)
    reference = pd.Timestamp(now) if now is not None else anchor
    if reference - anchor > pd.Timedelta(minutes=max_age_minutes):
        return None, "stale power observations"
    if not all(name in frame for name in GRID_FIELDS):
        return None, "charger telemetry unavailable"
    window = frame.loc[(frame.index > anchor - pd.Timedelta(minutes=15)) & (frame.index <= anchor), list(GRID_FIELDS)]
    # One-second bins measure temporal coverage, not the count of repeated
    # samples; reindex prevents gaps being hidden by dropped rows.
    bins = pd.date_range(anchor.floor("s") - pd.Timedelta(seconds=899), anchor.floor("s"), freq="s")
    paired = window.reindex(bins, method="nearest", tolerance=pd.Timedelta(seconds=1))
    valid = np.isfinite(paired).all(axis=1) & paired.GridWatts.ge(0) & paired.MultiPlusDCWatts.ge(0)
    if valid.mean() < 0.90 or not valid.iloc[-1]:
        return None, "insufficient recent paired grid observations"
    return float(paired.loc[valid, "MultiPlusDCWatts"].median()), "ready"


def no_grid_intervals(frame, starts, ends) -> np.ndarray:
    """Require covered DC observations and no charging anywhere in each interval."""
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    rejected = np.zeros(len(starts), dtype=bool)
    if frame.empty or "MultiPlusDCWatts" not in frame:
        return rejected
    source = frame["MultiPlusDCWatts"].sort_index()
    dc = source.reindex(pd.date_range(source.index[0].ceil("s"), source.index[-1].floor("s"), freq="s"),
                        method="nearest", tolerance=pd.Timedelta(seconds=1))
    if dc.empty:
        return rejected
    valid = np.isfinite(dc.to_numpy())
    counts = np.r_[0, np.cumsum(valid)]
    # Check charging on original samples too: a brief positive reading must
    # not disappear between the coverage bins.
    charging = np.r_[0, np.cumsum(source.gt(0).to_numpy())]
    raw_left = source.index.searchsorted(starts, side="left")
    raw_right = source.index.searchsorted(ends, side="right")
    left = dc.index.searchsorted(starts.floor("s"), side="left")
    right = dc.index.searchsorted(ends.floor("s"), side="right")
    expected = np.maximum(((ends - starts).total_seconds() + 1), 1)
    return ((starts >= dc.index[0]) & (ends <= dc.index[-1]) &
            ((counts[right] - counts[left]) / expected >= 0.90) &
            (charging[raw_right] == charging[raw_left]))
