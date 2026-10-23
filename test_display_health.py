"""Regression tests for frozen battery cards and an independently failed publisher."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import app
import infrastructure_status as infrastructure
import mobile_catalog


NOW = datetime(2026, 10, 23, 6, 42, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def fixed_clock():
    with patch.object(app, "datetime", wraps=datetime) as browser, patch.object(mobile_catalog, "datetime", wraps=datetime) as catalog:
        browser.now.return_value = catalog.now.return_value = NOW
        yield


def battery_snapshot(**changes):
    value = {
        "time_utc": "2026-10-23T00:42:00Z",
        "aps_battery_soc_pct": 100.0, "aps_battery_voltage_v": 55.36,
        "aps_battery_power_w": 49.0, "aps_battery_depletion_deadband_w": 50.0,
        "aps_battery_soc_age_min": 2.0, "aps_battery_voltage_age_min": 2.0,
        "aps_battery_power_age_min": 2.0,
        "power_latest_time_utc": "2026-10-23T00:40:00Z",
    }
    value.update(changes)
    return value


def browser_cards(snapshot):
    environmental = {"title": "Untouched environment", "value": "original", "detail": "original detail", "level": "amber"}
    with (
        patch.object(app, "_ops_read_snapshot", return_value=snapshot),
        patch.object(app, "_mobile_auroracam_freshness", return_value=("Radar", "2m old", "green")),
        patch.object(app, "_mobile_power_latest_measured_time", return_value=None),
        patch.object(mobile_catalog, "environmental_signal_cards", return_value=[environmental]),
        patch.object(mobile_catalog, "infrastructure_status", return_value=None),
        patch.object(app, "_mobile_status_card_markup", wraps=app._mobile_status_card_markup) as rendered,
    ):
        app._mobile_overview_markup()
    return {args.args[0]: args.args[1:] for args in rendered.call_args_list}


def test_frozen_snapshot_ages_and_colors_keep_advancing():
    snapshot = battery_snapshot()
    cards = browser_cards(snapshot)
    assert cards["State of Charge"][2] == cards["Battery Voltage"][2] == "red"
    assert "6h 2m old" in cards["State of Charge"][1]
    assert "6h 2m old" in cards["Battery Voltage"][1]
    assert cards["Time to Depleted"][0] == "Unavailable"
    assert cards["Time to Depleted"][2] == "red"
    assert cards["Power Data"][2] == "red"
    assert cards["Untouched environment"] == ("original", "original detail", "amber")


@pytest.mark.parametrize("timestamp", [None, "not-a-date", "2026-10-23T07:00:00Z"])
def test_invalid_or_future_metric_times_never_show_fresh_green(timestamp):
    snapshot = battery_snapshot(time_utc=NOW.isoformat(), power_latest_time_utc="2026-10-23T06:40:00Z")
    for metric in ("soc", "voltage", "power"):
        snapshot[f"aps_battery_{metric}_time_utc"] = timestamp
    cards = browser_cards(snapshot)
    assert cards["State of Charge"][2] == cards["Battery Voltage"][2] == "gray"
    assert "Age unknown" in cards["State of Charge"][1]
    assert cards["Time to Depleted"][0] == "Unavailable"


@pytest.mark.parametrize("snapshot_time", [None, "invalid", "2026-10-24T00:00:00Z"])
def test_age_only_samples_need_a_valid_nonfuture_snapshot_time(snapshot_time):
    snapshot = battery_snapshot(time_utc=snapshot_time)
    assert mobile_catalog.battery_display_freshness(snapshot)["soc"]["level"] == "unknown"
    assert app._ops_battery_depletion_text(snapshot)[0] == "Unavailable"


def test_fresh_power_does_not_mask_stale_soc_for_depletion():
    snapshot = battery_snapshot(
        aps_battery_soc_time_utc="2026-10-23T00:40:00Z",
        aps_battery_power_time_utc="2026-10-23T06:41:00Z",
        aps_battery_voltage_time_utc="2026-10-23T06:41:00Z",
    )
    freshness = mobile_catalog.battery_display_freshness(snapshot)
    assert freshness["power"]["level"] == "green"
    assert freshness["depletion"]["level"] == "red"
    assert app._ops_battery_depletion_text(snapshot)[0] == "Unavailable"


def test_fresh_and_delayed_battery_readings_keep_value_thresholds():
    for age, expected in ((2, "green"), (45, "amber")):
        timestamp = (NOW - timedelta(minutes=age)).isoformat()
        snapshot = battery_snapshot(**{f"aps_battery_{m}_time_utc": timestamp for m in ("soc", "voltage", "power")})
        assert app._ops_battery_metric_level(snapshot, "soc", "green") == expected
        assert app._ops_level_from_battery_depletion(snapshot) == expected
        assert app._ops_battery_depletion_text(snapshot)[0] != "Unavailable"
        assert app._ops_battery_metric_level(snapshot, "soc", "red") == "red"


def test_future_power_summary_time_is_unavailable():
    snapshot = battery_snapshot(power_latest_time_utc="2026-10-24T00:00:00Z")
    assert mobile_catalog._snapshot_power_time(snapshot) is None
    assert browser_cards(snapshot)["Power Data"][2] == "gray"


def make_worker(root: Path, state="complete", completed_at=None, **status_fields):
    root.mkdir(parents=True, exist_ok=True)
    generation = root / "current"
    generation.mkdir(exist_ok=True)
    completed_at = completed_at or (NOW - timedelta(minutes=2)).isoformat()
    (generation / "generation.json").write_text(json.dumps({"completed_at": completed_at}))
    payload = {"state": state, "started_at": NOW.isoformat(), "completed_at": completed_at, **status_fields}
    (root / "status.json").write_text(json.dumps(payload))
    return root / "status.json"


def healthy_workers(tmp_path):
    core = make_worker(tmp_path / "core")
    science = tmp_path / "science"
    for key in ("cl61", "radar", "hatpro", "wxcam"):
        make_worker(science / key)
    return core, science


def read_infrastructure(core, science, free_gib=30):
    with patch.object(infrastructure.shutil, "disk_usage", return_value=SimpleNamespace(free=free_gib * 1024**3, total=1000 * 1024**3)):
        return infrastructure.collect_infrastructure_status(core, science, now=NOW)


def test_failed_attempt_and_disk_reserve_are_read_without_operations_snapshot(tmp_path):
    core, science = healthy_workers(tmp_path)
    old = "2026-10-23T00:42:00Z"
    make_worker(core.parent, state="failed", completed_at=old, failed_at=NOW.isoformat(), error="RuntimeError: Insufficient free disk: 6000; need 8000+1000 reserve")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    status = read_infrastructure(core, science, free_gib=5.6)
    assert status["level"] == "red"
    assert status["workers"][0]["lastSuccessfulPublicationAt"] == "2026-10-23T00:42:00+00:00"
    assert status["workers"][0]["lastAttemptAt"] == NOW.isoformat()
    assert "failed" in status["summary"]
    assert "5.6 GiB free" in status["summary"]
    assert any(a["id"] == "infrastructure:disk" for a in status["alerts"])
    assert any("Insufficient free disk space" in a["detail"] for a in status["alerts"])
    assert all(p.read_bytes() == original for p, original in before.items())


@pytest.mark.parametrize("completed_at,expected", [("2026-10-23T00:42:00Z", "red"), ("2026-10-24T00:42:00Z", "amber"), ("invalid", "amber")])
def test_complete_flag_without_recent_valid_publication_is_not_healthy(tmp_path, completed_at, expected):
    core, science = healthy_workers(tmp_path)
    make_worker(core.parent, completed_at=completed_at)
    status = read_infrastructure(core, science)
    assert status["workers"][0]["level"] == expected


def test_missing_or_corrupt_worker_status_is_visible(tmp_path):
    core, science = healthy_workers(tmp_path)
    core.write_text("{invalid")
    (science / "cl61" / "status.json").unlink()
    status = read_infrastructure(core, science)
    assert status["workers"][0]["level"] == "amber"
    assert status["workers"][1]["level"] == "amber"


def test_successful_retry_clears_warnings_and_preserves_publication_time(tmp_path):
    core, science = healthy_workers(tmp_path)
    status = read_infrastructure(core, science)
    assert status["level"] == "green"
    assert status["alerts"] == []
    assert status["checkedAt"] == NOW.isoformat()
    assert status["workers"][0]["lastSuccessfulPublicationAt"] != status["checkedAt"]


def test_api_and_browser_surface_infrastructure_when_snapshot_cannot_refresh(tmp_path):
    core, science = healthy_workers(tmp_path)
    make_worker(core.parent, state="failed", failed_at=NOW.isoformat())
    status = read_infrastructure(core, science)
    with (
        patch.object(mobile_catalog, "infrastructure_status", return_value=status),
        patch.object(mobile_catalog, "read_json_file", return_value={}),
        patch.object(app, "_ops_read_snapshot", return_value={"_missing": True}),
    ):
        api = mobile_catalog.operations()
        browser = app._ops_operations_markup()
    assert api["overallLevel"] == "red"
    assert api["updatedAt"] is None  # A fresh infrastructure check is not a fresh snapshot.
    assert any(a["id"] == "infrastructure:continuity:core" for a in api["alerts"])
    assert "Infrastructure" in browser and "failed" in browser
