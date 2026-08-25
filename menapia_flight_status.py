"""Present Menapia flight ingest health from the AURORA health-v2 contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


UTC = timezone.utc


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _integer(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _age_minutes(value: object, now: datetime) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return max((now - stamp.astimezone(UTC)).total_seconds() / 60, 0.0)


def summarize_menapia_flight(
    archive_health: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    """Return a stable dashboard record without exposing credential material."""
    current = now or datetime.now(UTC)
    source_ingest = _mapping(archive_health.get("source_ingest"))
    source = _mapping(source_ingest.get("menapia"))
    metrics = _mapping(archive_health.get("metrics"))
    credential = _mapping(source.get("credential"))
    delivery = _mapping(source.get("archive_delivery"))
    latest = _mapping(source.get("latest_source_flight"))

    enabled = bool(source.get("enabled"))
    commissioned = bool(source.get("commissioned"))
    state = str(source.get("state") or "unavailable")
    authentication_failure = bool(source.get("authentication_failure"))
    failure_count = _integer(source.get("failure_count"))
    last_success_at = source.get("last_success_at")
    age = _age_minutes(last_success_at, current)
    days_remaining = credential.get("days_remaining")
    try:
        days_remaining = int(days_remaining) if days_remaining is not None else None
    except (TypeError, ValueError):
        days_remaining = None

    level = "green"
    title = "Flight data ingest is healthy"
    if not enabled:
        level = "unknown"
        title = "Flight data ingest is not configured"
    elif authentication_failure:
        level = "red"
        title = "Menapia authentication failed"
    elif not commissioned:
        level = "amber"
        title = "Flight data ingest awaits commissioning"
    elif failure_count or state in {"failed", "partial_failure", "missing", "unavailable"}:
        level = "red"
        title = "Flight data ingest needs attention"
    elif age is None:
        level = "red"
        title = "No successful flight-data ingest recorded"
    elif age >= 120:
        level = "red"
        title = "Flight data ingest is stale"
    elif age >= 45:
        level = "amber"
        title = "Flight data ingest is delayed"

    if commissioned and days_remaining is not None:
        if days_remaining <= 7 and level in {"green", "amber"}:
            level = "red"
            title = "Menapia credential rotation is urgent"
        elif days_remaining <= 30 and level == "green":
            level = "amber"
            title = "Menapia credential rotation is due"

    gws_pending = _integer(
        delivery.get("gws_pending_files", metrics.get("menapia_flight_gws_pending_files"))
    )
    object_pending = _integer(
        delivery.get(
            "object_store_pending_files",
            metrics.get("menapia_flight_object_store_pending_files"),
        )
    )
    unclassified = _integer(
        source.get(
            "unclassified_objects",
            metrics.get("menapia_flight_unclassified_object_count"),
        )
    )
    detail_parts = [f"Source state: {state.replace('_', ' ')}."]
    if age is not None:
        detail_parts.append(f"Last success {age:.0f} min ago.")
    if gws_pending or object_pending:
        detail_parts.append(
            f"Pending delivery: GWS {gws_pending:,}, object store {object_pending:,}."
        )
    if unclassified:
        detail_parts.append(
            f"{unclassified:,} source object(s) remain campaign-unclassified."
        )
    if days_remaining is not None:
        detail_parts.append(f"Credential: {days_remaining} day(s) remaining.")

    return {
        "level": level,
        "title": title,
        "detail": " ".join(detail_parts),
        "enabled": enabled,
        "commissioned": commissioned,
        "state": state,
        "lastSuccessAt": last_success_at,
        "lastSuccessAgeMinutes": age,
        "objectsExamined": _integer(source.get("upstream_objects_examined")),
        "newObjects": _integer(source.get("new_objects_ingested")),
        "bytesTransferred": _integer(source.get("bytes_transferred")),
        "unclassifiedObjects": unclassified,
        "latestSourceDate": latest.get("date"),
        "latestSourceFlight": latest.get("flight"),
        "credentialExpiresOn": credential.get("expires_on"),
        "credentialDaysRemaining": days_remaining,
        "gwsPendingFiles": gws_pending,
        "objectStorePendingFiles": object_pending,
        "dualDeliveredFiles": _integer(delivery.get("dual_delivered_files")),
    }
