"""Read-only display health independent of the generated Operations snapshot."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


UTC = timezone.utc
RESERVE_BYTES = 8 * 1024**3
PUBLICATION_MAX_AGE_MINUTES = 30
WORKERS = (
    ("core", "Power and camera continuity"),
    ("cl61", "CL61 continuity"),
    ("radar", "Radar continuity"),
    ("hatpro", "HATPRO continuity"),
    ("wxcam", "WXcam imagery continuity"),
)


def _read(path: Path) -> dict:
    try:
        # Status/manifest documents are small. Never read a product or raw data.
        if path.stat().st_size > 1024 * 1024:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _time(value, now: datetime) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        moment = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
        return moment if moment <= now else None
    except (TypeError, ValueError):
        return None


def collect_infrastructure_status(
    core_status_path: Path, science_root: Path, *, now: datetime | None = None,
) -> dict:
    """Inspect publication records and the serving filesystem on each request."""
    now = now or datetime.now(UTC)
    workers = []
    alerts = []
    disk = {"level": "amber", "reserveBytes": RESERVE_BYTES}
    try:
        usage = shutil.disk_usage(core_status_path.parent)
        disk.update(freeBytes=usage.free, totalBytes=usage.total)
        disk["level"] = "red" if usage.free < RESERVE_BYTES else "amber" if usage.free < 2 * RESERVE_BYTES else "green"
        disk_detail = f"{usage.free / 1024**3:.1f} GiB free; continuity reserve {RESERVE_BYTES / 1024**3:.0f} GiB."
    except OSError:
        disk_detail = "Serving filesystem capacity could not be checked."
    disk["detail"] = disk_detail
    if disk["level"] != "green":
        alerts.append({
            "id": "infrastructure:disk", "level": disk["level"],
            "title": "Storage below continuity reserve" if disk["level"] == "red" else "Continuity storage needs attention",
            "detail": disk_detail,
        })

    for key, title in WORKERS:
        status_path = core_status_path if key == "core" else science_root / key / "status.json"
        status = _read(status_path)
        # The published pointer identifies the successful generation even while
        # status.json describes a newer failed/running attempt.
        generation = _read(status_path.parent / "current" / "generation.json")
        published_at = _time(generation.get("completed_at"), now)
        if not generation and status.get("state") in {"complete", "degraded"}:
            published_at = _time(status.get("completed_at"), now)
        age = (now - published_at).total_seconds() / 60 if published_at else None
        attempt_at = _time(status.get("failed_at") or status.get("completed_at") or status.get("started_at"), now)
        state = str(status.get("state") or "unknown")
        if state == "failed":
            level, reason = "red", "Latest publication attempt failed."
        elif published_at is None:
            level, reason = "amber", "Successful publication time is unavailable."
        elif age > PUBLICATION_MAX_AGE_MINUTES:
            level, reason = "red", f"No successful publication for {age:.0f} minutes."
        elif attempt_at is None or state not in {"complete", "degraded", "running"}:
            level, reason = "amber", "Worker status is unavailable or has an invalid timestamp."
        elif state == "degraded":
            level, reason = "amber", "Latest publication is degraded."
        else:
            level, reason = "green", "Recent publication available."
        published_label = published_at.strftime("%d %b %H:%M UTC") if published_at else "unknown"
        detail = f"{reason} Last successful publication: {published_label}."
        error = str(status.get("error") or "")
        if state == "failed" and error:
            # Keep low-space errors readable without exposing command lines or
            # arbitrary worker exception contents in a new public warning.
            detail += " Insufficient free disk space." if "insufficient free disk" in error.lower() else "See the worker service log for the failure."
        worker = {
            "id": key, "title": title, "state": state, "level": level,
            "lastSuccessfulPublicationAt": published_at.isoformat() if published_at else None,
            "lastAttemptAt": attempt_at.isoformat() if attempt_at else None,
            "publicationAgeMinutes": age, "detail": detail,
        }
        workers.append(worker)
        if level != "green":
            alerts.append({"id": f"infrastructure:continuity:{key}", "title": title, "level": level, "detail": detail})

    level = "red" if any(a["level"] == "red" for a in alerts) else "amber" if alerts else "green"
    affected = [f"{w['title'].removesuffix(' continuity')} ({w['state'] if w['state'] in {'failed', 'degraded'} else 'publication delayed or unknown'})" for w in workers if w["level"] != "green"]
    core = workers[0]
    summary = disk_detail
    if affected:
        summary += f" Publication needs attention: {', '.join(affected)}."
    else:
        summary += " Continuity publications are recent."
    summary += f" Last core publication: {core['lastSuccessfulPublicationAt'] or 'unknown'}."
    checked = now.isoformat()
    return {
        "checkedAt": checked, "level": level, "summary": summary,
        "disk": disk, "workers": workers, "alerts": alerts,
        "card": {
            "id": "infrastructure", "title": "Infrastructure",
            "value": {"green": "Healthy", "amber": "Attention", "red": "Action"}[level],
            "level": level, "updatedAt": checked, "detail": summary,
        },
    }
