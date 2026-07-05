"""Read-only product catalog helpers for the Aurora mobile API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any


APP_DIR = Path(__file__).resolve().parent
DATE_TOKEN_RE = re.compile(r"(20\d{6})")
WXCAM_DAY_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Instrument:
    id: str
    title: str
    system_image: str
    quicklook_subdir: str
    science_prefixes: tuple[str, ...]
    housekeeping_prefixes: tuple[str, ...] = ()
    visible: bool = True
    summary_supported: bool = True


INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument("power", "Aurora Power Supply", "battery.100percent", "power", ("power__summary", "power"), ("power__HK_APS", "power__hk_aps")),
    Instrument("ceilometer", "Ceilometer", "laser.burst", "ceilometer", ("ceilometer",), ("ceilometer__HK_Ceilometer", "ceilometer__hk_ceilometer")),
    Instrument("cloud-radar", "Cloud Radar", "dot.radiowaves.left.and.right", "cloud_radar", ("cloud_radar",), ("cloud_radar__HK_Radar", "cloud_radar__hk_radar")),
    Instrument("hatpro", "Scanning Microwave Radiometer", "antenna.radiowaves.left.and.right", "hatpro", ("hatpro",)),
    Instrument("vaisalamet", "Meteorology", "cloud.sun", "vaisalamet", ("vaisalamet__summary", "vaisalamet"), ("vaisalamet__HK_Met", "vaisalamet__hk_met")),
    Instrument("asfs-logger", "Radiation", "sun.max", "asfs_logger", ("asfs_logger__summary", "asfs_logger"), ("asfs_logger__HK_ASFS", "asfs_logger__hk_asfs")),
    Instrument("ops-monitor", "Operations", "gauge.with.dots.needle.bottom.50percent", "ops_monitor", ("ops_monitor__summary", "ops_monitor"), ("ops_monitor__HK_Operations", "ops_monitor__hk_operations")),
    Instrument("wxcam", "WXcam", "video", "wxcam", ("wxcam",), ("wxcam__HK_WXcam", "wxcam__hk_wxcam"), summary_supported=False),
)

INSTRUMENT_BY_ID = {instrument.id: instrument for instrument in INSTRUMENTS}

WXCAM_STREAMS = {
    "fish_hdr": {"title": "FISH HDR", "systemImage": "camera.aperture"},
    "pano_hdr": {"title": "PANO HDR", "systemImage": "photo"},
}

OPERATIONS_STREAMS = (
    {
        "id": "ceilometer",
        "title": "Ceilometer",
        "source": "cl61_source_sync_service_healthy_state",
        "services": ("ceilometer_append_service_healthy_state", "ceilometer_quicklooks_service_healthy_state"),
    },
    {
        "id": "cloud-radar",
        "title": "Cloud Radar",
        "source": "radar_source_sync_service_healthy_state",
        "services": ("radar_append_service_healthy_state", "radar_quicklooks_service_healthy_state"),
    },
    {
        "id": "hatpro",
        "title": "Scanning Microwave Radiometer",
        "source": "hatpro_source_sync_service_healthy_state",
        "services": ("hatpro_append_service_healthy_state", "hatpro_quicklooks_service_healthy_state"),
    },
    {
        "id": "vaisalamet",
        "title": "Meteorology",
        "source": "vaisalamet_source_sync_service_healthy_state",
        "services": ("vaisalamet_append_service_healthy_state", "vaisalamet_quicklooks_service_healthy_state"),
    },
    {
        "id": "asfs-logger",
        "title": "Radiation",
        "source": "asfs_logger_source_sync_service_healthy_state",
        "services": ("asfs_logger_append_service_healthy_state", "asfs_logger_quicklooks_service_healthy_state"),
    },
    {
        "id": "power",
        "title": "Aurora Power Supply",
        "source": "power_source_sync_service_healthy_state",
        "services": ("power_append_service_healthy_state", "power_quicklooks_service_healthy_state"),
    },
    {
        "id": "wxcam",
        "title": "WXcam",
        "source": "wxcam_source_sync_service_healthy_state",
        "services": (
            "wxcam_append_service_healthy_state",
            "wxcam_catalog_service_healthy_state",
            "wxcam_daily_videos_service_healthy_state",
        ),
    },
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def quicklook_root() -> Path:
    return env_path("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks")


def wxcam_daily_video_root() -> Path:
    return env_path("WXCAM_DAILY_VIDEO_DIR", "/data/aurora/products/wxcam/daily_videos")


def wxcam_hourly_thumbnail_root() -> Path:
    return env_path("WXCAM_HOURLY_THUMB_DIR", "/data/aurora/products/wxcam/hourly_thumbnails")


def wxcam_catalog_path() -> Path:
    return env_path("WXCAM_CATALOG_PATH", "/data/aurora/products/wxcam/wxcam_catalog.sqlite")


def operations_snapshot_path() -> Path:
    return env_path("OPS_MONITOR_LATEST_SNAPSHOT", "/project/aurora/raw/ops_monitor/latest.json")


def operations_health_path() -> Path:
    return env_path("OPS_MONITOR_LATEST_HEALTH", "/data/aurora/products/ops_monitor/health/latest_health.json")


def operations_alert_state_path() -> Path:
    return env_path("OPS_MONITOR_ALERT_STATE", "/data/aurora/products/ops_monitor/alerts/state.json")


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_error": f"Invalid JSON: {exc}"}
    except OSError as exc:
        return {"_error": str(exc)}
    return value if isinstance(value, dict) else {"value": value}


def file_record(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "sizeBytes": stat_result.st_size,
        "modifiedAt": datetime.fromtimestamp(stat_result.st_mtime, UTC).isoformat().replace("+00:00", "Z"),
    }


def normalize_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"green", "ok", "healthy", "good", "1", "true"}:
        return "green"
    if text in {"amber", "yellow", "warning", "warn"}:
        return "amber"
    if text in {"red", "critical", "failed", "error", "0", "false"}:
        return "red"
    return "unknown"


def level_from_booleans(values: list[Any]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return "unknown"
    if any(str(value).strip().lower() in {"0", "false", "red", "failed", "error"} for value in known):
        return "red"
    return "green"


def media_url(*parts: str) -> str:
    return "/media/" + "/".join(part.strip("/") for part in parts)


def manifest() -> dict[str, Any]:
    return {
        "serverTime": utc_now_iso(),
        "minimumRefreshIntervalSeconds": 60,
        "sections": [
            {"id": "operations", "title": "Operations", "systemImage": "gauge.with.dots.needle.bottom.50percent"},
            {"id": "interactive", "title": "Interactive", "systemImage": "chart.xyaxis.line"},
            {"id": "quicklooks", "title": "Quicklooks", "systemImage": "photo.on.rectangle.angled"},
            {"id": "wxcam", "title": "WXcam", "systemImage": "video"},
            {"id": "settings", "title": "Settings", "systemImage": "gearshape"},
        ],
        "instruments": [
            {
                "id": instrument.id,
                "title": instrument.title,
                "systemImage": instrument.system_image,
                "visible": instrument.visible,
                "supportsSummary": instrument.summary_supported,
                "supportsScienceQuicklooks": bool(instrument.science_prefixes),
                "supportsHousekeepingQuicklooks": bool(instrument.housekeeping_prefixes),
            }
            for instrument in INSTRUMENTS
        ],
        "wxcamStreams": [
            {"id": stream_id, "title": spec["title"], "systemImage": spec["systemImage"]}
            for stream_id, spec in WXCAM_STREAMS.items()
        ],
    }


def operations() -> dict[str, Any]:
    health = read_json_file(operations_health_path())
    snapshot = read_json_file(operations_snapshot_path())
    alert_state = read_json_file(operations_alert_state_path())

    health_error = health.get("_error")
    snapshot_error = snapshot.get("_error")
    overall = normalize_level(health.get("overall_level") or snapshot.get("overall_level"))

    stream_states = [_stream_state(snapshot, spec) for spec in OPERATIONS_STREAMS]
    if overall == "unknown":
        if any(stream["level"] == "red" for stream in stream_states):
            overall = "red"
        elif any(stream["level"] == "green" for stream in stream_states):
            overall = "green"

    updated_at = health.get("time_utc") or health.get("snapshot_time_utc") or snapshot.get("time_utc") or snapshot.get("snapshot_time_utc")
    active_alerts = _active_alerts(alert_state)

    return {
        "serverTime": utc_now_iso(),
        "updatedAt": updated_at,
        "overallLevel": overall,
        "summary": _operations_summary(overall, stream_states, health_error, snapshot_error),
        "checkCounts": _check_counts(health, stream_states),
        "streamStates": stream_states,
        "rootCauseGroups": _root_cause_groups(snapshot, stream_states),
        "alerts": active_alerts,
        "trendCards": _trend_cards(snapshot),
        "sources": {
            "health": {**file_record(operations_health_path()), "path": str(operations_health_path())},
            "snapshot": {**file_record(operations_snapshot_path()), "path": str(operations_snapshot_path())},
        },
    }


def _stream_state(snapshot: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    source_value = snapshot.get(str(spec["source"]))
    service_values = [snapshot.get(str(key)) for key in spec["services"]]
    source_level = level_from_booleans([source_value])
    service_level = level_from_booleans(service_values)
    level = "red" if "red" in {source_level, service_level} else "green" if "green" in {source_level, service_level} else "unknown"

    failed_services = [
        key.removesuffix("_service_healthy_state").replace("_", " ")
        for key, value in zip(spec["services"], service_values, strict=False)
        if normalize_level(value) == "red"
    ]
    if normalize_level(source_value) == "red":
        detail = "Source sync is unhealthy"
    elif failed_services:
        detail = "Unhealthy: " + ", ".join(failed_services[:3])
    elif level == "green":
        detail = "Source and processing services healthy"
    else:
        detail = "No current status sample"

    return {
        "id": spec["id"],
        "title": spec["title"],
        "level": level,
        "detail": detail,
        "sourceHealthy": source_value,
        "serviceHealthyCount": sum(1 for value in service_values if normalize_level(value) == "green"),
        "serviceCount": len(service_values),
    }


def _operations_summary(overall: str, streams: list[dict[str, Any]], health_error: Any, snapshot_error: Any) -> str:
    if health_error:
        return f"Health JSON error: {health_error}"
    if snapshot_error:
        return f"Snapshot JSON error: {snapshot_error}"
    red_count = sum(1 for stream in streams if stream["level"] == "red")
    unknown_count = sum(1 for stream in streams if stream["level"] == "unknown")
    if red_count:
        return f"{red_count} stream group{'s' if red_count != 1 else ''} need attention"
    if unknown_count == len(streams):
        return "No operations snapshot available"
    if overall == "green":
        return "All visible stream groups are healthy"
    return "Operations status is partially available"


def _check_counts(health: dict[str, Any], streams: list[dict[str, Any]]) -> dict[str, int]:
    counts = health.get("check_counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items() if isinstance(value, int | float)}
    return {
        "green": sum(1 for stream in streams if stream["level"] == "green"),
        "amber": sum(1 for stream in streams if stream["level"] == "amber"),
        "red": sum(1 for stream in streams if stream["level"] == "red"),
        "unknown": sum(1 for stream in streams if stream["level"] == "unknown"),
    }


def _root_cause_groups(snapshot: dict[str, Any], streams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_issues = [stream["title"] for stream in streams if normalize_level(stream.get("sourceHealthy")) == "red"]
    service_issues = [stream["title"] for stream in streams if stream["level"] == "red" and stream["title"] not in source_issues]
    storage_level = "red" if any(float(snapshot.get(key, 0) or 0) >= 80 for key in ("aurora_data_used_pct", "aurora_root_used_pct", "gws_used_pct")) else "green" if snapshot else "unknown"
    dashboard_level = level_from_booleans([snapshot.get("dashboard_http_healthy_state"), snapshot.get("primary_dashboard_http_healthy_state"), snapshot.get("standby_dashboard_http_healthy_state")])
    return [
        {"id": "source", "title": "Source freshness", "level": "red" if source_issues else "green" if snapshot else "unknown", "detail": ", ".join(source_issues[:4]) if source_issues else "No source freshness issues"},
        {"id": "processing", "title": "Local processing", "level": "red" if service_issues else "green" if snapshot else "unknown", "detail": ", ".join(service_issues[:4]) if service_issues else "Append, catalog, and quicklook services healthy"},
        {"id": "storage", "title": "Storage pressure", "level": storage_level, "detail": "Storage is below alert thresholds" if storage_level == "green" else "Storage needs attention"},
        {"id": "dashboard", "title": "Public dashboard", "level": dashboard_level, "detail": "Dashboard endpoint probes are healthy" if dashboard_level == "green" else "Dashboard endpoint probe needs attention"},
    ]


def _trend_cards(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("storage", "Worst storage use", "%", ("aurora_data_used_pct", "aurora_root_used_pct", "gws_used_pct")),
        ("battery-soc", "APS state of charge", "%", ("BatterySOC", "power_battery_soc")),
        ("battery-voltage", "APS battery voltage", "V", ("DCInverterVolts", "power_battery_voltage")),
        ("source-lag", "Worst source lag", "min", ("worst_source_lag_min", "source_lag_max_min")),
        ("gws-lag", "Worst GWS lag", "min", ("worst_gws_lag_min", "gws_lag_max_min")),
    )
    cards: list[dict[str, Any]] = []
    for card_id, title, unit, keys in specs:
        values = [snapshot.get(key) for key in keys if isinstance(snapshot.get(key), int | float)]
        value = max(values) if values and "Worst" in title else values[0] if values else None
        cards.append({"id": card_id, "title": title, "value": value, "unit": unit, "level": _trend_level(card_id, value)})
    return cards


def _trend_level(card_id: str, value: Any) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    if card_id == "storage":
        return "red" if value >= 90 else "amber" if value >= 80 else "green"
    if card_id == "battery-soc":
        return "red" if value < 25 else "amber" if value < 50 else "green"
    if card_id == "battery-voltage":
        return "red" if value < 50 else "amber" if value < 52 else "green"
    return "red" if value >= 180 else "amber" if value >= 90 else "green"


def _active_alerts(alert_state: dict[str, Any]) -> list[dict[str, Any]]:
    active = alert_state.get("active") or alert_state.get("active_alerts") or alert_state.get("alerts") or []
    if isinstance(active, dict):
        iterator = active.items()
    elif isinstance(active, list):
        iterator = enumerate(active)
    else:
        return []
    alerts = []
    for key, value in iterator:
        if isinstance(value, dict):
            title = str(value.get("title") or value.get("kind") or key)
            level = normalize_level(value.get("level") or value.get("severity") or "red")
            detail = str(value.get("message") or value.get("detail") or "")
        else:
            title = str(value)
            level = "red"
            detail = ""
        alerts.append({"id": str(key), "title": title, "level": level, "detail": detail})
    return alerts


def instrument_summary(instrument_id: str, window: str = "24h") -> dict[str, Any]:
    instrument = _instrument_or_raise(instrument_id)
    latest = resolve_quicklook_path("science", instrument_id, "latest")
    entries = quicklooks("science", instrument_id).get("entries", [])[:8]
    panels = []
    if latest:
        panels.append(
            {
                "id": "latest-quicklook",
                "title": "Latest quicklook",
                "kind": "image",
                "imageURL": media_url("quicklook", "science", instrument_id, "latest"),
                "level": "green",
                "detail": "Latest generated quicklook",
            }
        )
    else:
        panels.append(
            {
                "id": "latest-quicklook",
                "title": "Latest quicklook",
                "kind": "empty",
                "level": "unknown",
                "detail": "No generated quicklook was found",
            }
        )
    return {
        "serverTime": utc_now_iso(),
        "instrument": {
            "id": instrument.id,
            "title": instrument.title,
            "systemImage": instrument.system_image,
            "supportsSummary": instrument.summary_supported,
        },
        "window": window,
        "updatedAt": file_record(latest).get("modifiedAt") if latest else None,
        "panels": panels,
        "recentQuicklooks": entries,
    }


def quicklooks(kind: str, instrument_id: str) -> dict[str, Any]:
    instrument = _instrument_or_raise(instrument_id)
    if kind not in {"science", "housekeeping"}:
        raise KeyError(f"Unknown quicklook kind: {kind}")
    prefixes = instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes
    entries = _quicklook_entries(instrument, kind, prefixes)
    latest = next((entry for entry in entries if entry["token"] == "latest"), entries[0] if entries else None)
    return {
        "serverTime": utc_now_iso(),
        "kind": kind,
        "instrument": {"id": instrument.id, "title": instrument.title, "systemImage": instrument.system_image},
        "latest": latest,
        "entries": entries,
    }


def _quicklook_entries(instrument: Instrument, kind: str, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    directory = quicklook_root() / instrument.quicklook_subdir
    if not prefixes or not directory.exists():
        return []
    paths: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
            continue
        name = path.name
        is_latest = "latest" in name.lower()
        token_match = DATE_TOKEN_RE.search(name)
        matches_prefix = any(name.startswith(prefix) for prefix in prefixes)
        if not matches_prefix and not (instrument.id in {"ceilometer", "cloud-radar", "hatpro", "wxcam"} and kind == "science"):
            continue
        if is_latest:
            paths.setdefault("latest", path)
        elif token_match:
            paths.setdefault(token_match.group(1), path)

    def sort_key(item: tuple[str, Path]) -> tuple[int, str]:
        token, _path = item
        return (1 if token == "latest" else 0, token)

    entries = []
    for token, path in sorted(paths.items(), key=sort_key, reverse=True):
        entries.append(
            {
                "id": f"{instrument.id}-{kind}-{token}",
                "token": token,
                "title": "Latest" if token == "latest" else _format_date_token(token),
                "imageURL": media_url("quicklook", kind, instrument.id, token),
                **file_record(path),
            }
        )
    return entries


def resolve_quicklook_path(kind: str, instrument_id: str, token: str) -> Path | None:
    instrument = _instrument_or_raise(instrument_id)
    entries = _quicklook_entries(
        instrument,
        kind,
        instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes,
    )
    match = next((entry for entry in entries if entry["token"] == token), None)
    if not match:
        return None
    return _find_quicklook_path_by_record(instrument, kind, token)


def _find_quicklook_path_by_record(instrument: Instrument, kind: str, token: str) -> Path | None:
    prefixes = instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes
    directory = quicklook_root() / instrument.quicklook_subdir
    if not directory.exists():
        return None
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if token == "latest" and "latest" not in path.name.lower():
            continue
        if token != "latest" and token not in path.name:
            continue
        if any(path.name.startswith(prefix) for prefix in prefixes) or kind == "science":
            return path
    return None


def wxcam(stream: str = "fish_hdr", day: str = "latest") -> dict[str, Any]:
    if stream not in WXCAM_STREAMS:
        raise KeyError(f"Unknown WXcam stream: {stream}")
    resolved_day = _resolve_wxcam_day(stream, day)
    video_path = resolve_wxcam_video_path(stream, resolved_day)
    thumbs = wxcam_thumbnail_records(stream, resolved_day)
    return {
        "serverTime": utc_now_iso(),
        "stream": {"id": stream, **WXCAM_STREAMS[stream]},
        "selectedDay": resolved_day,
        "availableDays": available_wxcam_days(stream),
        "video": {
            "url": media_url("wxcam", "video", stream, resolved_day),
            **(file_record(video_path) if video_path else {"exists": False}),
        },
        "posterURL": thumbs[0]["imageURL"] if thumbs else None,
        "thumbnails": thumbs,
    }


def resolve_wxcam_video_path(stream: str, day: str) -> Path | None:
    token = _wxcam_day_token(day)
    if not token:
        return None
    path = wxcam_daily_video_root() / stream / f"{token}.mp4"
    return path if path.exists() else None


def available_wxcam_days(stream: str) -> list[str]:
    directory = wxcam_daily_video_root() / stream
    if not directory.exists():
        return []
    days = []
    for path in sorted(directory.glob("*.mp4")):
        if path.stem == "latest":
            continue
        match = DATE_TOKEN_RE.fullmatch(path.stem)
        if match:
            days.append(_format_date_token(match.group(1)))
    return sorted(days, reverse=True)


def wxcam_thumbnail_records(stream: str, day: str) -> list[dict[str, Any]]:
    token = _wxcam_day_token(day)
    if not token:
        return []
    directory = wxcam_hourly_thumbnail_root() / stream / token
    if not directory.exists():
        return []
    records = []
    for index, path in enumerate(sorted(directory.glob("*.jpg"))[:24]):
        records.append(
            {
                "id": path.stem,
                "title": path.stem,
                "hourUTC": index,
                "imageURL": media_url("wxcam", "thumb", stream, token, path.name),
                **file_record(path),
            }
        )
    return records


def resolve_wxcam_thumbnail_path(stream: str, day_token: str, filename: str) -> Path | None:
    if stream not in WXCAM_STREAMS or DATE_TOKEN_RE.fullmatch(day_token) is None or Path(filename).name != filename:
        return None
    path = wxcam_hourly_thumbnail_root() / stream / day_token / filename
    return path if path.exists() else None


def latest_wxcam_catalog_record(stream: str, media_kind: str) -> dict[str, Any] | None:
    if stream not in WXCAM_STREAMS or media_kind not in {"image", "video"}:
        return None
    path = wxcam_catalog_path()
    if not path.exists():
        return None
    uri = f"file:{path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT image_type, media_kind, time_utc, day_utc, filename, relative_path
                FROM images
                WHERE image_type = ? AND media_kind = ?
                ORDER BY time_epoch_ns DESC, raw_path DESC
                LIMIT 1
                """,
                (stream, media_kind),
            ).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row else None


def _resolve_wxcam_day(stream: str, day: str) -> str:
    if day != "latest":
        return day if _wxcam_day_token(day) else "latest"
    record = latest_wxcam_catalog_record(stream, "video")
    if record and record.get("day_utc"):
        return str(record["day_utc"])
    days = available_wxcam_days(stream)
    return days[0] if days else "latest"


def _format_date_token(token: str) -> str:
    return f"{token[:4]}-{token[4:6]}-{token[6:8]}"


def _wxcam_day_token(day: str) -> str | None:
    if day == "latest":
        return "latest"
    if DATE_TOKEN_RE.fullmatch(day):
        return day
    if WXCAM_DAY_RE.fullmatch(day):
        return day.replace("-", "")
    return None


def _instrument_or_raise(instrument_id: str) -> Instrument:
    instrument = INSTRUMENT_BY_ID.get(instrument_id)
    if not instrument:
        raise KeyError(f"Unknown instrument: {instrument_id}")
    return instrument
