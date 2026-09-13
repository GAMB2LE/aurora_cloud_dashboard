"""Read-only interpretation of the edge-published ASFS directory estimate."""

from datetime import datetime
import math
from typing import Any

from collection_freshness import parse_time


PREFIX = "asfs_logger_storage_"
MAX_AGE_MINUTES = 120


def storage_metrics(payload: dict[str, Any], now: datetime) -> dict[str, Any]:
    stamp = parse_time(payload.get("generated_at"))
    age = (now - stamp).total_seconds() / 60 if stamp else None
    valid = payload.get("schema_version") == 1 and age is not None and -5 <= age <= MAX_AGE_MINUTES
    state = str(payload.get("state", "unknown")) if valid else "unknown"
    if state not in {"normal", "warning", "critical", "unknown"}:
        state = "unknown"
    counts = {}
    for field in ("file_count", "estimated_directory_slots", "estimated_remaining_slots", "directory_slot_limit"):
        value = payload.get(field)
        try:
            number = float(value)
            counts[field] = int(number) if math.isfinite(number) and number >= 0 and number.is_integer() else None
        except (ValueError, TypeError, OverflowError):
            counts[field] = None
    slots = counts["estimated_directory_slots"]
    limit = counts["directory_slot_limit"]
    if slots is None or limit != 65536 or payload.get("estimate_only") is not True:
        state = "unknown"
    elif valid and state != "unknown":
        # The monitor cannot be green when counts already cross the documented
        # conservative thresholds, even if the producer's label is inconsistent.
        if slots >= 60000:
            state = "critical"
        elif slots >= 55000 and state == "normal":
            state = "warning"
    level = {"normal": "green", "warning": "amber", "critical": "red", "unknown": "amber"}[state]
    if state == "unknown":
        detail = "ASFS logger directory estimate is unavailable or older than 2 hours"
    else:
        detail = f"Estimated {slots:,} of {limit:,} FAT directory slots used; this is a filename estimate, not authenticated card status"
    result = {
        "state": state, "level": level, "detail": detail,
        "generated_at": stamp.isoformat() if stamp else None,
        "age_min": max(age, 0) if age is not None else None,
        "evidence_available_state": int(state != "unknown"),
        "estimate_only": 1,
        **counts,
    }
    return {PREFIX + key: value for key, value in result.items()}


def storage_status(snapshot: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    if PREFIX + "state" not in snapshot:
        return None
    payload = {
        "schema_version": 1,
        "generated_at": snapshot.get(PREFIX + "generated_at"),
        "state": snapshot.get(PREFIX + "state"),
        "estimate_only": True,
        **{key: snapshot.get(PREFIX + key) for key in (
            "file_count", "estimated_directory_slots", "estimated_remaining_slots", "directory_slot_limit",
        )},
    }
    return {key.removeprefix(PREFIX): value for key, value in storage_metrics(payload, now).items()}
