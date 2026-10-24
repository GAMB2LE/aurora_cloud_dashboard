"""Measured collection evidence, separate from transfer and archive health.

These timestamps describe cloud products, not direct logger/edge acquisition.
Legacy source metrics remain readable, but product evidence never overwrites
them or claims to localise an upstream failure.
"""

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
from typing import Any


UTC = timezone.utc
PRODUCTS = {
    "cl61": ("CEILOMETER_ZARR_PATH", "cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr", 90.0),
    "radar": ("CLOUD_RADAR_ZARR_PATH", "rpgfmcw94/cloud_radar.zarr", 90.0),
    "hatpro": ("HATPRO_ZARR_PATH", "hatprog5/hatpro.zarr", 180.0),
    "vaisalamet": ("VAISALAMET_ZARR_PATH", "vaisalamet/vaisalamet.zarr", 120.0),
    "asfs_logger": ("ASFS_LOGGER_ZARR_PATH", "asfs_logger/asfs_logger.zarr", 120.0),
    "asfs_fast_sonic": ("ASFS_FAST_SONIC_ZARR_PATH", "asfs_fast_sonic/asfs_fast_sonic.zarr", 120.0),
    "asfs_fast_gas": ("ASFS_FAST_GAS_ZARR_PATH", "asfs_fast_gas/asfs_fast_gas.zarr", 120.0),
    "power": ("POWER_ZARR_PATH", "power/power.zarr", 120.0),
    "pdu": ("PDU_ZARR_PATH", "power/pdu.zarr", 30.0),
    "wxcam": ("WXCAM_CATALOG_PATH", "wxcam/wxcam_catalog.sqlite", 120.0),
}
PDU_STREAM_OUTLETS = {"cl61": 5, "radar": 6, "hatpro": 8}


def parse_time(value: Any) -> datetime | None:
    try:
        # Python 3.10 rejects the 9-digit fractions emitted for nanosecond Zarr
        # times. datetime's evidence precision is microseconds on every host.
        text = re.sub(r"(\.\d{6})\d+", r"\1", str(value))
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def product_path(prefix: str) -> Path:
    environment, relative, _threshold = PRODUCTS[prefix]
    return Path(os.environ.get(environment, f"/data/aurora/products/{relative}"))


def latest_product_time(prefix: str) -> datetime | None:
    """Read one Zarr time chunk or the latest indexed camera timestamp."""
    path = product_path(prefix)
    if not path.exists():
        return None
    try:
        if prefix == "wxcam":
            from wxcam_catalog import open_catalog

            connection = open_catalog(path, readonly=True)
            try:
                row = connection.execute("SELECT time_utc FROM images ORDER BY time_epoch_ns DESC LIMIT 1").fetchone()
                return parse_time(row[0]) if row else None
            finally:
                connection.close()
        import pandas as pd
        import xarray as xr
        import zarr

        group = zarr.open_group(str(path), mode="r")
        if "time" not in group or not group["time"].shape or not group["time"].shape[0]:
            return None
        coordinate = group["time"]
        decoded = xr.coding.times.decode_cf_datetime(
            [coordinate[-1]], coordinate.attrs["units"],
            calendar=coordinate.attrs.get("calendar", "standard"), use_cftime=False,
        )[0]
        if pd.isna(decoded):
            return None
        return parse_time(pd.Timestamp(decoded).isoformat())
    except (ImportError, OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def collect_product_freshness(now: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix, (_environment, _relative, threshold) in PRODUCTS.items():
        stamp = latest_product_time(prefix)
        age = (now - stamp).total_seconds() / 60 if stamp else None
        valid = age is not None and math.isfinite(age) and age >= -5
        result.update({
            f"{prefix}_product_sample_time_utc": stamp.isoformat().replace("+00:00", "Z") if valid else None,
            f"{prefix}_product_age_min": max(age, 0) if valid else None,
            f"{prefix}_product_recent_state": int(age <= threshold) if valid else None,
            f"{prefix}_product_sample_available_state": int(valid),
            f"{prefix}_collection_evidence": "cloud_product_sample",
        })
    return result


def collection_state(snapshot: dict[str, Any], prefix: str, now: datetime) -> dict[str, Any]:
    """Recompute ages at read time so a stopped collector cannot stay green."""
    threshold = PRODUCTS.get(prefix, (None, None, 120.0))[2]
    stamp = parse_time(snapshot.get(f"{prefix}_product_sample_time_utc"))
    evidence = "cloud_product_sample"
    age = None
    if stamp:
        age = (now - stamp).total_seconds() / 60
    elif f"{prefix}_product_sample_available_state" not in snapshot:
        # Compatibility with snapshots from before product monitoring. An age
        # is evidence only when finite; a successful service is never a sample.
        evidence = "source_snapshot"
        try:
            age = float(snapshot[f"{prefix}_source_age_min"])
            sampled = parse_time(snapshot.get("snapshot_time_utc") or snapshot.get("time_utc"))
            if sampled:
                age += max((now - sampled).total_seconds() / 60, 0)
        except (KeyError, TypeError, ValueError):
            pass
    if age is None or not math.isfinite(age) or age < -5:
        return {"level": "unknown", "ageMinutes": None, "sampleAt": None, "evidence": evidence}
    age = max(age, 0)
    return {
        "level": "green" if age <= threshold else "red",
        "ageMinutes": age,
        "sampleAt": stamp.isoformat().replace("+00:00", "Z") if stamp else None,
        "evidence": evidence,
    }


def recent_pdu_states(now: datetime) -> dict[int, bool] | None:
    """Only fresh, non-future electrical evidence can suppress collection alarms."""
    stamp = latest_product_time("pdu")
    if stamp is None or not -5 <= (now - stamp).total_seconds() / 60 <= 30:
        return None
    try:
        import zarr

        group = zarr.open_group(str(product_path("pdu")), mode="r")
        states = {}
        for outlet in PDU_STREAM_OUTLETS.values():
            field = f"PDUOutlet{outlet}State"
            if field in group:
                value = float(group[field][-1])
                if math.isfinite(value):
                    states[outlet] = value >= 0.5
        return states or None
    except (ImportError, OSError, ValueError, TypeError, KeyError):
        return None
