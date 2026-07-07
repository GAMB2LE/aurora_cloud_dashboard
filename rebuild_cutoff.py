"""Shared UTC cutoff handling for Aurora product Zarr rebuilds."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr


def parse_from_time(value: str | datetime | None) -> datetime | None:
    """Parse an ISO cutoff timestamp and return an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cutoff_date(value: str | datetime | None) -> date | None:
    parsed = parse_from_time(value)
    return parsed.date() if parsed is not None else None


def naive_utc(value: str | datetime | None) -> datetime | None:
    parsed = parse_from_time(value)
    return parsed.replace(tzinfo=None) if parsed is not None else None


def np_utc64(value: str | datetime | None) -> np.datetime64 | None:
    parsed = naive_utc(value)
    if parsed is None:
        return None
    return np.datetime64(pd.Timestamp(parsed).to_datetime64(), "ns")


def filter_dataset_from_time(
    ds: xr.Dataset,
    from_time: str | datetime | None,
    *,
    time_dim: str = "time",
) -> xr.Dataset:
    """Return only samples whose time coordinate is >= ``from_time``."""
    cutoff = np_utc64(from_time)
    if cutoff is None or time_dim not in ds.coords:
        return ds
    keep = np.asarray(ds[time_dim].values).astype("datetime64[ns]") >= cutoff
    if keep.all():
        return ds
    return ds.isel({time_dim: keep})


def latest_cutoff(*values: datetime | None) -> datetime | None:
    present = [parse_from_time(value) for value in values if value is not None]
    return max(present) if present else None
