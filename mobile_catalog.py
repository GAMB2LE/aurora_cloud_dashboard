"""Read-only product catalog helpers for the Aurora mobile API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from auroracam_catalog import AURORACAM_CAMERAS, available_days as auroracam_available_days, day_records as auroracam_day_records, latest_records as auroracam_latest_records
from display_artifact_manifest import load_manifest
from uas_mqtt import load_uas_mqtt_log
from instrument_registry import (
    INSTRUMENTS,
    INSTRUMENT_BY_ID,
    PDU_INSTRUMENTS as PDU_INSTRUMENT_CONTRACTS,
    SCIENCE_DC_INSTRUMENTS as SCIENCE_DC_INSTRUMENT_CONTRACTS,
)


UTC = timezone.utc


APP_DIR = Path(__file__).resolve().parent
DATE_TOKEN_RE = re.compile(r"(20\d{6})")
WXCAM_DAY_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")
AURORACAM_DAY_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


WXCAM_STREAMS = {
    "fish_hdr": {"title": "FISH HDR", "systemImage": "camera.aperture"},
    "pano_hdr": {"title": "PANO HDR", "systemImage": "photo"},
}

PDU_INSTRUMENTS = tuple(
    (
        instrument.id,
        instrument.pdu_title or instrument.title,
        instrument.system_image,
        instrument.pdu_outlet,
    )
    for instrument in PDU_INSTRUMENT_CONTRACTS
)
PDU_INSTRUMENT_BY_ID = {instrument_id: (title, icon, outlet) for instrument_id, title, icon, outlet in PDU_INSTRUMENTS}
PDU_STATE_FRESHNESS_MINUTES = 30.0

# These Science-tab products have no individual PDU outlet state. Their mobile
# status is therefore collection freshness, never an inferred power state.
SCIENCE_DC_INSTRUMENTS = tuple(
    (
        instrument.id,
        instrument.title,
        instrument.system_image,
        instrument.quicklook_subdir,
    )
    for instrument in SCIENCE_DC_INSTRUMENT_CONTRACTS
)

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


def auroracam_root() -> Path:
    return env_path("AURORACAM_RAW_ROOT", os.environ.get("AURORACAM_ROOT", "/project/aurora/raw/auroracam"))


def auroracam_preview_cache_root() -> Path:
    return env_path("AURORA_MOBILE_PREVIEW_CACHE", "/var/cache/aurora-mobile-api/auroracam")


def power_display_summary_path() -> Path:
    return env_path("POWER_DISPLAY_SUMMARY_ZARR_PATH", "/data/aurora/products/power/power_display_summary.zarr")


def power_operating_scenario_paths() -> tuple[Path, ...]:
    """Locate the authoritative operating-plan product for native clients."""
    configured = env_path(
        "POWER_OPERATING_SCENARIOS_ZARR_PATH",
        "/data/aurora/products/power/power_operating_scenarios.zarr",
    )
    mirrored = Path("/data/aurora/products/power/power_operating_scenarios.zarr")
    return tuple(dict.fromkeys((configured, mirrored)))


def uas_mqtt_log_path() -> Path:
    return env_path("UAS_MQTT_LOG_PATH", "/project/aurora/raw/menapia/menapia_mqtt.log")


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


def display_artifacts() -> dict[str, Any]:
    """Return the latest publishable dashboard-artifact manifest when present."""
    return load_manifest()


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


def dashboard_revision() -> str | None:
    configured = os.environ.get("AURORA_DASHBOARD_REVISION", "").strip()
    if configured:
        return configured

    head_path = APP_DIR / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = APP_DIR / ".git" / head.removeprefix("ref: ")
            head = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return head[:12] if head else None


def deployment_descriptor() -> dict[str, Any]:
    domain = os.environ.get("AURORA_DOMAIN", "").strip() or "data-ocean.gamb2le.co.uk"
    environment = os.environ.get("AURORA_SITE_ENV", "").strip().lower()
    if not environment:
        environment = "development" if "data-ocean" in domain else "production"
    data_role = "live-mirror" if environment == "development" else "authoritative"
    return {
        "environment": environment,
        "domain": domain,
        "dashboardURL": f"https://{domain}/app",
        "dataRole": data_role,
        "revision": dashboard_revision(),
    }


def manifest() -> dict[str, Any]:
    return {
        "serverTime": utc_now_iso(),
        "schemaVersion": 3,
        "minimumRefreshIntervalSeconds": 60,
        "deployment": deployment_descriptor(),
        "sections": [
            {"id": "overview", "title": "Overview", "systemImage": "rectangle.3.group"},
            {"id": "power", "title": "Power", "systemImage": "bolt.batteryblock"},
            {"id": "plots", "title": "Plots", "systemImage": "chart.xyaxis.line"},
            {"id": "camera", "title": "Camera", "systemImage": "camera"},
            {"id": "ops", "title": "Ops", "systemImage": "gauge.with.dots.needle.bottom.50percent"},
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
        # This is the cross-platform contract. Native clients can choose a
        # compact presentation, while the browser remains the full explorer.
        "capabilities": {
            "shared": [
                "power.current_system_ecmwf_p10_p90",
                "power.assigned_pdu_outlets",
                "operations.instrument_state",
                "operations.live_status",
                "quicklooks.science_housekeeping",
                "camera.auroracam_wxcam",
            ],
            "browser": [
                "explore.arbitrary_variables_ranges",
                "plots.plotly_investigation",
                "uas.full_history_events",
                "camera.shareable_state",
            ],
            "native": [
                "overview.cached_snapshot",
                "overview.endpoint_failover",
                "overview.dynamic_type",
                "overview.pull_to_refresh",
            ],
        },
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
        return "red" if value <= 40 else "amber" if value < 50 else "green"
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
            # The alert sender keeps recovered entries in state.json for
            # notification history.  Only entries that are still active
            # belong in the mobile operations payload.
            if value.get("active") is False:
                continue
            title = str(value.get("title") or value.get("kind") or key)
            level = normalize_level(value.get("level") or value.get("severity") or "red")
            detail = str(value.get("message") or value.get("detail") or "")
        else:
            title = str(value)
            level = "red"
            detail = ""
        alerts.append({"id": str(key), "title": title, "level": level, "detail": detail})
    return alerts


def overview() -> dict[str, Any]:
    """Return the lightweight operational cards shown first in the native app."""
    snapshot = read_json_file(operations_snapshot_path())
    status = operations()
    latest_cameras = auroracam("latest")
    camera_times = [record.get("timeUTC") for record in latest_cameras["frames"] if record.get("timeUTC")]
    latest_camera_time = max(camera_times) if camera_times else None
    latest_power_time = snapshot.get("power_latest_time_utc") or _latest_power_time()
    depletion_value, depletion_detail = _battery_depletion_text(snapshot)
    cards = [
        _overview_card("operations", "Operations", _operations_value(status["overallLevel"]), status["overallLevel"], status.get("updatedAt"), status["summary"]),
        _overview_card("battery-soc", "State of Charge", _metric_text(snapshot, ("aps_battery_soc_pct", "BatterySOC"), "%"), _trend_level("battery-soc", _metric_value(snapshot, ("aps_battery_soc_pct", "BatterySOC"))), status.get("updatedAt"), _metric_age_detail(snapshot, "aps_battery_soc_age_min")),
        _overview_card("battery-voltage", "Battery Voltage", _metric_text(snapshot, ("aps_battery_voltage_v", "DCInverterVolts"), "V"), _trend_level("battery-voltage", _metric_value(snapshot, ("aps_battery_voltage_v", "DCInverterVolts"))), status.get("updatedAt"), _metric_age_detail(snapshot, "aps_battery_voltage_age_min")),
        _overview_card("battery-depletion", "Time to Depleted", depletion_value, _battery_depletion_level(snapshot), status.get("updatedAt"), depletion_detail),
        _overview_card("power", "Power Data", _power_time_text(latest_power_time), _age_level(latest_power_time, 30, 120), latest_power_time, _power_age_text(latest_power_time)),
        _overview_card("auroracam", "AURORACam", _age_text(latest_camera_time), _age_level(latest_camera_time, 30, 120), latest_camera_time, "Latest station camera frame"),
    ]
    return {
        "serverTime": utc_now_iso(),
        "cards": cards,
        "instrumentPower": _instrument_power_states(snapshot),
        "activeAlerts": status["alerts"],
    }


def _instrument_power_states(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return PDU power states plus collection states for DC science streams."""
    states, detail = _pdu_power_snapshot()

    pdu_rows = [
        _pdu_instrument_status(instrument_id, states, detail)
        for instrument_id, _title, _icon, _outlet in PDU_INSTRUMENTS
    ]
    science_rows = []
    for instrument_id, title, icon, prefix in SCIENCE_DC_INSTRUMENTS:
        source_age = _metric_value(snapshot, (f"{prefix}_source_age_min",))
        recent = snapshot.get(f"{prefix}_source_recent_state")
        if recent == 1:
            state, level = "Collecting", "green"
        elif recent == 0:
            state, level = "No recent data", "red"
        else:
            state, level = "Unknown", "amber"
        detail = (
            f"Source sample {_duration_text(source_age / 60)} old"
            if source_age is not None
            else "Source freshness unavailable"
        )
        science_rows.append(
            {
                "id": instrument_id,
                "title": title,
                "systemImage": icon,
                "state": state,
                "level": level,
                "detail": detail,
            }
        )
    # Collection freshness is the first operational signal on the mobile overview.
    # Keep it ahead of PDU outlet state regardless of changes to either inventory.
    return [*science_rows, *pdu_rows]


def pdu_instrument_status(instrument_id: str) -> dict[str, Any] | None:
    """Return the current assigned PDU state for a powered instrument, if known."""
    if instrument_id not in PDU_INSTRUMENT_BY_ID:
        return None
    states, detail = _pdu_power_snapshot()
    return _pdu_instrument_status(instrument_id, states, detail)


def pdu_outlet_states() -> dict[int, bool] | None:
    """Return fresh outlet states for dashboard health policy, if available."""
    states, _detail = _pdu_power_snapshot()
    return states or None


def _pdu_power_snapshot() -> tuple[dict[int, bool], str]:
    """Read one fresh PDU sample without inferring a state from stale data."""
    path = Path(os.environ.get("PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))
    try:
        import pandas as pd
        import xarray as xr

        dataset = xr.open_zarr(path, consolidated=False)
        try:
            if "time" not in dataset or dataset.sizes.get("time", 0) == 0:
                raise ValueError("no PDU samples")
            sample_time = pd.Timestamp(dataset["time"].values[-1]).to_pydatetime()
            if sample_time.tzinfo is None:
                sample_time = sample_time.replace(tzinfo=UTC)
            age_minutes = max((datetime.now(UTC) - sample_time.astimezone(UTC)).total_seconds() / 60, 0)
            if age_minutes > PDU_STATE_FRESHNESS_MINUTES:
                raise ValueError("stale PDU sample")
            states = {
                outlet: float(dataset[f"PDUOutlet{outlet}State"].values[-1]) >= 0.5
                for _id, _title, _icon, outlet in PDU_INSTRUMENTS
                if f"PDUOutlet{outlet}State" in dataset
            }
            detail = f"PDU sample {_duration_text(age_minutes / 60)} old"
        finally:
            dataset.close()
    except Exception:
        states = {}
        detail = "PDU status unavailable"
    return states, detail


def _pdu_instrument_status(instrument_id: str, states: dict[int, bool], detail: str) -> dict[str, Any]:
    title, icon, outlet = PDU_INSTRUMENT_BY_ID[instrument_id]
    powered = states.get(outlet)
    return {
        "id": instrument_id,
        "title": title,
        "systemImage": icon,
        "state": "On" if powered is True else "Off" if powered is False else "Unknown",
        "level": "green" if powered is True else "unknown" if powered is False else "amber",
        "detail": detail,
    }


def _overview_card(card_id: str, title: str, value: str, level: str, updated_at: str | None, detail: str = "") -> dict[str, Any]:
    return {"id": card_id, "title": title, "value": value, "level": level, "updatedAt": updated_at, "detail": detail}


def _operations_value(level: str) -> str:
    return {"green": "Healthy", "amber": "Attention", "red": "Action"}.get(level, "Waiting")


def _metric_age_detail(snapshot: dict[str, Any], key: str) -> str:
    age = _metric_value(snapshot, (key,))
    return "Age unknown" if age is None else f"{_duration_text(age / 60)} old"


def _power_time_text(value: Any) -> str:
    moment = _parse_utc(str(value)) if value else None
    return moment.strftime("%H:%M UTC") if moment else "No data"


def _power_age_text(value: Any) -> str:
    moment = _parse_utc(str(value)) if value else None
    if moment is None:
        return "Latest measured power timestamp unavailable"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return f"Updated {_duration_text(age_minutes / 60)} ago"


def _latest_power_time() -> str | None:
    """Read the latest measured timestamp from the existing display summary."""
    path = power_display_summary_path()
    if not path.exists():
        return None
    dataset = None
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        dataset = xr.open_zarr(path, consolidated=True, chunks=None)
        if "time" not in dataset:
            return None
        times = pd.DatetimeIndex(dataset["time"].values)
        now = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
        latest = None
        for name in ("BatterySOC", "BatteryWatts", "DCInverterVolts", "ACOutputWatts"):
            if name not in dataset or dataset[name].dims != ("time",):
                continue
            values = np.asarray(dataset[name].values, dtype=np.float64)
            mask = np.isfinite(values) & (times <= now)
            if mask.any():
                candidate = times[mask].max()
                latest = candidate if latest is None or candidate > latest else latest
        return None if latest is None else pd.Timestamp(latest).isoformat() + "Z"
    except Exception:
        return None
    finally:
        if dataset is not None:
            dataset.close()


def _battery_depletion_text(snapshot: dict[str, Any]) -> tuple[str, str]:
    soc = _metric_value(snapshot, ("aps_battery_soc_pct", "BatterySOC"))
    power_w = _metric_value(snapshot, ("aps_battery_power_w", "BatteryWatts"))
    if soc is None or power_w is None:
        return "No data", "Needs battery state of charge and power"

    capacity_kwh = _metric_value(snapshot, ("aps_battery_capacity_kwh",)) or 26.0
    deadband_w = _metric_value(snapshot, ("aps_battery_depletion_deadband_w",)) or 25.0
    remaining_kwh = _metric_value(snapshot, ("aps_battery_remaining_kwh",))
    if remaining_kwh is None:
        remaining_kwh = max(soc, 0.0) / 100.0 * capacity_kwh
    energy_text = f"{remaining_kwh:.1f} kWh remaining from {capacity_kwh:.0f} kWh"

    if power_w < -deadband_w:
        hours = _metric_value(snapshot, ("aps_battery_depletion_hours",))
        if hours is None:
            hours = remaining_kwh / (abs(power_w) / 1000.0)
        return _duration_text(hours), f"{energy_text}; discharging at {abs(power_w):.0f} W"
    if power_w > deadband_w:
        return "Charging", f"{energy_text}; charging at {power_w:.0f} W"
    return "Flat", f"{energy_text}; battery power {power_w:.0f} W"


def _battery_depletion_level(snapshot: dict[str, Any]) -> str:
    soc = _metric_value(snapshot, ("aps_battery_soc_pct", "BatterySOC"))
    power_w = _metric_value(snapshot, ("aps_battery_power_w", "BatteryWatts"))
    if soc is None or power_w is None:
        return "unknown"
    deadband_w = _metric_value(snapshot, ("aps_battery_depletion_deadband_w",)) or 25.0
    if power_w >= -deadband_w:
        return "green"
    hours = _metric_value(snapshot, ("aps_battery_depletion_hours",))
    if hours is None:
        capacity_kwh = _metric_value(snapshot, ("aps_battery_capacity_kwh",)) or 26.0
        hours = (max(soc, 0.0) / 100.0 * capacity_kwh) / (abs(power_w) / 1000.0)
    return "green" if hours >= 24 else "amber" if hours >= 12 else "red"


def _duration_text(hours: float) -> str:
    total_minutes = max(int(round(hours * 60)), 0)
    days, remainder = divmod(total_minutes, 24 * 60)
    hour_count, minutes = divmod(remainder, 60)
    if days:
        return f"{days}d {hour_count}h"
    if hour_count:
        return f"{hour_count}h {minutes}m"
    return f"{minutes}m"


def _metric_value(snapshot: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _metric_text(snapshot: dict[str, Any], keys: tuple[str, ...], unit: str) -> str:
    value = _metric_value(snapshot, keys)
    if value is not None:
        return f"{value:.1f}{unit}" if unit else f"{value:.1f}"
    for key in keys:
        value = snapshot.get(key)
        if value:
            return str(value)
    return "Unavailable"


def _age_level(value: str | None, green_minutes: float, amber_minutes: float) -> str:
    moment = _parse_utc(value)
    if moment is None:
        return "unknown"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return "green" if age_minutes < green_minutes else "amber" if age_minutes < amber_minutes else "red"


def _age_text(value: str | None) -> str:
    moment = _parse_utc(value)
    if moment is None:
        return "No image"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return f"{age_minutes:.0f} min old"


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def auroracam(day: str = "latest", time_utc: str | None = None) -> dict[str, Any]:
    root = auroracam_root()
    days = sorted(auroracam_available_days(root), reverse=True)
    if day != "latest" and (not AURORACAM_DAY_RE.fullmatch(day) or day not in days):
        raise KeyError(f"Unknown AURORACam day: {day}")

    available_times: list[str] = []
    if day == "latest":
        selected_day = days[0] if days else None
        records = auroracam_latest_records(root)
    else:
        selected_day = day
        records = {}
        for camera_id in AURORACAM_CAMERAS:
            candidates = auroracam_day_records(root, camera_id, day)
            available_times.extend(record.time_utc for record in candidates)
            if time_utc:
                candidates = [record for record in candidates if record.time_utc == time_utc]
            if candidates:
                records[camera_id] = candidates[-1]

    frames = []

    for camera_id in AURORACAM_CAMERAS:
        record = records.get(camera_id)
        if record is None:
            continue
        frames.append(
            {
                "id": record.camera_id,
                "cameraID": record.camera_id,
                "title": record.label,
                "timeUTC": record.time_utc.replace(" ", "T") + "Z",
                "dayUTC": record.day_utc,
                "previewURL": media_url("auroracam", "preview", record.camera_id, record.day_utc, record.filename),
                "originalURL": media_url("auroracam", "original", record.camera_id, record.day_utc, record.filename),
                "sizeBytes": record.size_bytes,
                "modifiedAt": datetime.fromtimestamp(record.mtime_ns / 1_000_000_000, UTC).isoformat().replace("+00:00", "Z"),
            }
        )
    return {
        "serverTime": utc_now_iso(),
        "selectedDay": selected_day,
        "selectedTimeUTC": time_utc,
        "availableDays": days,
        "availableTimesUTC": sorted(set(available_times), reverse=True)[:288],
        "frames": frames,
    }


def resolve_auroracam_image_path(camera_id: str, day: str, filename: str) -> Path | None:
    if camera_id not in AURORACAM_CAMERAS or AURORACAM_DAY_RE.fullmatch(day) is None or Path(filename).name != filename:
        return None
    path = auroracam_root() / camera_id / day / filename
    return path if path.is_file() else None


def create_auroracam_preview(source: Path, max_dimension: int = 960, quality: int = 80) -> Path:
    """Create a bounded, on-demand preview; original camera JPEGs stay untouched."""
    from PIL import Image

    cache = auroracam_preview_cache_root()
    relative = source.resolve().relative_to(auroracam_root().resolve())
    target = cache / relative
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        image.save(target, format="JPEG", quality=quality, optimize=True)
    _prune_preview_cache(cache)
    return target


def _prune_preview_cache(cache: Path, max_bytes: int = 50 * 1024 * 1024) -> None:
    try:
        files = [path for path in cache.rglob("*.jpg") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        for path in sorted(files, key=lambda item: item.stat().st_mtime_ns):
            if total <= max_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
    except OSError:
        return


def power(window: str = "24h", group: str = "all") -> dict[str, Any]:
    """Return compact chart points from the existing display-summary Zarr only."""
    supported_groups = {
        "all",
        "current",
        "forecast",
        "observed",
        "forecast_24h",
        "forecast_96h",
        "verification",
    }
    if window not in {"24h", "96h"} or group not in supported_groups:
        raise KeyError("Unsupported Power window or group")
    path = power_display_summary_path()
    payload: dict[str, Any] = {
        "serverTime": utc_now_iso(),
        "window": window,
        "group": group,
        "source": {**file_record(path), "path": str(path)},
        "minimumOperationalSOCPct": 40,
        "panels": [],
    }
    if not path.exists():
        payload["warning"] = "Power display-summary product is unavailable"
        return payload
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
        from grouped_timeseries import (
            POWER_PANEL_TIME_GROUPS,
            POWER_PANEL_TIME_GROUP_BY_KEY,
            SUMMARY_LAYOUTS,
            build_power_forecast_info,
            build_power_verification_guidance,
            merge_operating_scenarios_into_display_summary,
        )

        dataset = xr.open_zarr(path, chunks={"time": 1440}, consolidated=True)
        # The display summary can lag behind the fast planner.  Always replace
        # baked operating traces with the standalone contract, which rejects a
        # plan whose SOC anchor differs from the current ensemble forecast.
        scenarios = None
        for scenario_path in power_operating_scenario_paths():
            if not scenario_path.exists():
                continue
            try:
                candidate = xr.open_zarr(scenario_path, chunks={}, consolidated=True)
            except Exception:
                continue
            scenarios = candidate
            break
        dataset = merge_operating_scenarios_into_display_summary(dataset, scenarios)
        times = pd.DatetimeIndex(dataset["time"].values)
        now = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
        start = now - pd.Timedelta(hours=24)
        horizon = 24 if window == "24h" else 96
        end = now + pd.Timedelta(hours=horizon)
        if group == "all":
            selected_groups = tuple(POWER_PANEL_TIME_GROUPS)
        elif group == "current":
            selected_groups = ("observed",)
        elif group == "forecast":
            selected_groups = ("forecast_24h", "forecast_96h", "verification")
        else:
            selected_groups = (group,)
        panel_keys = {
            panel_key
            for selected_group in selected_groups
            for panel_key in POWER_PANEL_TIME_GROUPS[selected_group]
        }
        for panel in SUMMARY_LAYOUTS["power"]:
            if panel.key not in panel_keys:
                continue
            forecast_panel = POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key) in {"forecast_24h", "forecast_96h"}
            panel_start = _forecast_panel_start(dataset, times, panel) if forecast_panel else start
            panel_end = end if forecast_panel else now
            traces = []
            for trace in panel.traces:
                if trace.var not in dataset or dataset[trace.var].dims != ("time",):
                    continue
                values = np.asarray(dataset[trace.var].values, dtype=np.float64)
                mask = np.isfinite(values) & (times >= panel_start) & (times <= panel_end)
                if trace.valid_min is not None:
                    mask &= values >= float(trace.valid_min)
                if trace.valid_max is not None:
                    mask &= values <= float(trace.valid_max)
                selected_times = times[mask]
                selected_values = values[mask] * float(trace.scale)
                if len(selected_times) > 260:
                    selected = np.linspace(0, len(selected_times) - 1, 260, dtype=int)
                    selected_times = selected_times[selected]
                    selected_values = selected_values[selected]
                if not len(selected_times):
                    continue
                segment_ids = [0]
                if len(selected_times) > 1:
                    gaps = np.diff(selected_times.asi8) / 1_000_000_000
                    typical_gap = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 60.0
                    gap_threshold = max(typical_gap * 4, 300.0)
                    for gap in gaps:
                        segment_ids.append(segment_ids[-1] + (1 if gap > gap_threshold else 0))
                traces.append(
                    {
                        "id": trace.var,
                        "label": trace.label,
                        "color": trace.color,
                        "axis": trace.axis,
                        "dash": trace.dash,
                        "unit": str(dataset[trace.var].attrs.get("units", "")),
                        "points": [
                            {"time": pd.Timestamp(moment).isoformat() + "Z", "value": round(float(value), 5), "segment": segment}
                            for moment, value, segment in zip(selected_times, selected_values, segment_ids, strict=True)
                        ],
                    }
                )
            if traces:
                forecast_context = _power_forecast_context(dataset, panel.key, traces)
                payload["panels"].append(
                    {
                        "id": panel.key,
                        "title": panel.label,
                        "explanation": panel.description,
                        "info": build_power_forecast_info(panel.key, dataset),
                        "guidance": build_power_verification_guidance(panel.key, dataset),
                        "leftAxisLabel": panel.left_axis_label,
                        "rightAxisLabel": panel.right_axis_label,
                        "traces": traces,
                        **({"forecastContext": forecast_context} if forecast_context else {}),
                    }
                )
        dataset.close()
        if scenarios is not None:
            scenarios.close()
    except Exception as exc:
        payload["warning"] = f"Power display data unavailable: {exc}"
    return payload


def _forecast_panel_start(dataset, times, panel):
    """Return the first valid operational forecast time for a forecast-only panel."""
    import numpy as np
    import pandas as pd

    preferred_fields = {
        "soc_projection": ("BatterySOCForecast",),
        "soc_24h_forecast": ("BatterySOCForecast",),
        "soc_ecmwf_forecast": ("BatterySOCForecastP50", "BatterySOCForecast"),
        "ecmwf_solar_forecast": ("ForecastSolarWatts", "ECMWFSolarIrradiance"),
        "operating_plan_scenarios": ("OperatingCL61OptimizedSOCP50",),
        "operating_plan_schedule": ("OperatingCL61OptimizedCL61On",),
    }
    fields = preferred_fields.get(panel.key, tuple(trace.var for trace in panel.traces))
    for field in fields:
        if field not in dataset or dataset[field].dims != ("time",):
            continue
        values = np.asarray(dataset[field].values, dtype=np.float64)
        valid = np.isfinite(values)
        if valid.any():
            return pd.Timestamp(times[valid][0])
    return pd.Timestamp(datetime.now(UTC)).tz_localize(None)


def _power_forecast_context(dataset, panel_key: str, traces: list[dict[str, Any]]) -> dict[str, str] | None:
    """Return one anchor and one valid time for all values shown on a forecast card."""
    import pandas as pd

    forecast_panels = {
        "soc_24h_forecast",
        "soc_ecmwf_forecast",
        "ecmwf_solar_forecast",
        "operating_plan_scenarios",
        "operating_plan_schedule",
    }
    if panel_key not in forecast_panels:
        return None
    end_times: list[pd.Timestamp] = []
    for trace in traces:
        points = trace.get("points", [])
        if not points:
            continue
        value = pd.to_datetime(points[-1].get("time"), errors="coerce", utc=True)
        if pd.notna(value):
            end_times.append(pd.Timestamp(value).tz_convert("UTC").tz_localize(None))
    if not end_times:
        return None
    if panel_key.startswith("operating_plan"):
        anchor = dataset.attrs.get("operating_planning_forecast_initial_soc_time", "")
        issued = dataset.attrs.get("operating_planning_forecast_generated_at_utc", "")
        kind = "Operating-plan forecast"
    else:
        anchor = dataset.attrs.get("forecast_initial_soc_time", "")
        issued = dataset.attrs.get("forecast_generated_at_utc", "")
        kind = "System forecast"
    return {
        "kind": kind,
        "anchorTime": str(anchor),
        "issuedTime": str(issued),
        # The minimum end time is the last valid point shared by every trace.
        "validTime": min(end_times).isoformat() + "Z",
    }


def uas(window: str = "24h") -> dict[str, Any]:
    windows = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "all": None,
    }
    if window not in windows:
        raise KeyError(f"Unknown UAS window: {window}")

    result = load_uas_mqtt_log(uas_mqtt_log_path())
    latest = result.records[-1] if result.records else None
    age_seconds = (datetime.now(UTC) - latest.timestamp).total_seconds() if latest else None
    level = "red" if result.missing or result.error else "amber" if age_seconds is None or age_seconds > 300 else "green"
    duration = windows[window]
    cutoff = datetime.now(UTC) - duration if duration is not None else None
    records = [record for record in result.records if cutoff is None or record.timestamp >= cutoff]
    # A corrupted or unexpectedly high-rate log must not make the mobile API
    # response unbounded. The newest records preserve the current state.
    records = records[-2_000:]
    return {
        "serverTime": utc_now_iso(),
        "window": window,
        "level": level,
        "latest": None if latest is None else {
            "timeUTC": latest.timestamp.isoformat().replace("+00:00", "Z"),
            "reportedTier": latest.reported_tier,
            "effectiveTier": latest.effective_tier,
            "eventType": latest.event_type,
        },
        "source": {**file_record(result.path), "path": str(result.path)},
        "malformedLineCount": len(result.malformed_lines),
        "records": [
            {
                "timeUTC": record.timestamp.isoformat().replace("+00:00", "Z"),
                "reportedTier": record.reported_tier,
                "effectiveTier": record.effective_tier,
                "eventType": record.event_type,
            }
            for record in records
        ],
    }


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
        "powerStatus": pdu_instrument_status(instrument_id),
    }


def _quicklook_paths(instrument: Instrument, kind: str, prefixes: tuple[str, ...]) -> dict[str, Path]:
    directory = quicklook_root() / instrument.quicklook_subdir
    if not prefixes or not directory.exists():
        return {}
    paths: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
            continue
        name = path.name
        is_latest = "latest" in name.lower()
        token_match = DATE_TOKEN_RE.search(name)
        matches_prefix = any(name.startswith(prefix) for prefix in prefixes)
        # ``latest.png`` is the legacy science alias. It has no instrument
        # prefix but is still a valid science quicklook inside this directory.
        is_science_latest_alias = kind == "science" and name.lower() == "latest.png"
        # Science product names may share a base prefix with their housekeeping
        # counterpart (for example ``cloud_radar`` and ``cloud_radar__hk_radar``).
        # Always reject the latter before accepting a science image.
        is_housekeeping_image = any(name.startswith(prefix) for prefix in instrument.housekeeping_prefixes)
        if not (matches_prefix or is_science_latest_alias) or (kind == "science" and is_housekeeping_image):
            continue
        if is_latest:
            paths.setdefault("latest", path)
        elif token_match:
            paths.setdefault(token_match.group(1), path)

    # The legacy ``latest.png`` aliases are maintained by separate jobs. If a
    # newer dated science quicklook exists, serve that data rather than an old
    # alias that still calls itself "Last 24 hours".
    if kind == "science":
        dated_paths = [path for token, path in paths.items() if token != "latest"]
        newest_dated = max(dated_paths, key=lambda path: path.name) if dated_paths else None
        latest = paths.get("latest")
        if newest_dated is not None and (latest is None or newest_dated.stat().st_mtime_ns > latest.stat().st_mtime_ns):
            paths["latest"] = newest_dated
    return paths


def _quicklook_entries(instrument: Instrument, kind: str, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    paths = _quicklook_paths(instrument, kind, prefixes)

    def sort_key(item: tuple[str, Path]) -> tuple[int, str]:
        token, _path = item
        return (1 if token == "latest" else 0, token)

    entries = []
    for token, path in sorted(paths.items(), key=sort_key, reverse=True):
        dated_latest = token == "latest" and DATE_TOKEN_RE.search(path.name)
        entries.append(
            {
                "id": f"{instrument.id}-{kind}-{token}",
                "token": token,
                "title": (
                    f"Latest available ({_format_date_token(DATE_TOKEN_RE.search(path.name).group(1))})"
                    if dated_latest
                    else "Latest" if token == "latest" else _format_date_token(token)
                ),
                "imageURL": media_url("quicklook", kind, instrument.id, token),
                **file_record(path),
            }
        )
    return entries


def resolve_quicklook_path(kind: str, instrument_id: str, token: str) -> Path | None:
    instrument = _instrument_or_raise(instrument_id)
    paths = _quicklook_paths(
        instrument,
        kind,
        instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes,
    )
    return paths.get(token)


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
        is_housekeeping_image = any(path.name.startswith(prefix) for prefix in instrument.housekeeping_prefixes)
        is_science_latest_alias = kind == "science" and path.name.lower() == "latest.png"
        if (any(path.name.startswith(prefix) for prefix in prefixes) or is_science_latest_alias) and not (kind == "science" and is_housekeeping_image):
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
