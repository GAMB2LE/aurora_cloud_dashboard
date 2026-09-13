from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import re
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import collection_freshness as freshness
import collect_operations_snapshot as collector
import mobile_catalog as mobile
from send_ops_alerts import evaluate_alerts


NOW = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)


def _iso(age_minutes):
    return (NOW - timedelta(minutes=age_minutes)).isoformat()


def _healthy_snapshot():
    snapshot = {"snapshot_time_utc": NOW.isoformat()}
    for spec in mobile.OPERATIONS_STREAMS:
        prefix = mobile._stream_prefix(spec)
        snapshot[spec["source"]] = 1
        snapshot.update({key: 1 for key in spec["services"]})
        snapshot[f"{prefix}_product_sample_time_utc"] = _iso(5)
        snapshot[f"{prefix}_product_sample_available_state"] = 1
    return snapshot


def _operations(snapshot, paused=None):
    archive = {"overall_level": "green", "operator_status": {"level": "green"}, "metrics": {}}

    def read(path):
        if path == mobile.operations_snapshot_path():
            return dict(snapshot)
        if path == mobile.archive_health_path():
            return archive
        if path == mobile.operations_health_path():
            return {"overall_level": "green", "check_counts": {"green": 8}}
        return {}

    with patch.object(mobile, "read_json_file", side_effect=read), patch.object(
        mobile, "_intentionally_paused_streams", return_value=paused or set()
    ), patch.object(mobile, "_power_freshness_alert", return_value=None), patch.object(
        mobile, "_trend_cards", return_value=[]
    ), patch.object(mobile, "datetime", wraps=datetime) as clock:
        clock.now.return_value = NOW
        return mobile.operations()


def test_incident_successful_noop_services_and_green_archive_cannot_mask_stale_samples():
    snapshot = _healthy_snapshot()
    snapshot["asfs_logger_product_sample_time_utc"] = _iso(330)
    response = _operations(snapshot)
    radiation = next(row for row in response["streamStates"] if row["id"] == "asfs-logger")
    assert response["overallLevel"] == "red"
    assert radiation["level"] == "red"
    assert radiation["sourceSyncHealthy"] == 1
    assert radiation["serviceHealthyCount"] == 2
    assert radiation["collectionAgeMinutes"] == 330
    assert radiation["collectionEvidence"] == "cloud_product_sample"
    assert response["checkCounts"]["red"] == 1
    assert "need attention" in response["summary"]
    assert next(row for row in response["rootCauseGroups"] if row["id"] == "source")["level"] == "red"
    assert next(row for row in response["rootCauseGroups"] if row["id"] == "processing")["level"] == "green"


def test_missing_collection_evidence_is_degraded_despite_successful_services():
    snapshot = _healthy_snapshot()
    del snapshot["asfs_logger_product_sample_time_utc"]
    snapshot["asfs_logger_product_sample_available_state"] = 0
    response = _operations(snapshot)
    assert response["overallLevel"] == "amber"
    assert response["checkCounts"]["amber"] == 1
    assert "incomplete" in response["summary"]


def test_stopped_collector_does_not_freeze_sample_age_or_green_state():
    snapshot = {"asfs_logger_product_sample_time_utc": _iso(5), "asfs_logger_product_recent_state": 1}
    assert freshness.collection_state(snapshot, "asfs_logger", NOW)["level"] == "green"
    aged = freshness.collection_state(snapshot, "asfs_logger", NOW + timedelta(hours=3))
    assert aged["level"] == "red"
    assert aged["ageMinutes"] == 185


def test_nanosecond_product_timestamps_work_with_python310_iso_parser():
    def production_parser(text):
        if re.search(r"\.\d{7,}", text):
            raise ValueError("Python 3.10 requires three or six fractional digits")
        return datetime.fromisoformat(text)

    with patch.object(freshness, "datetime", SimpleNamespace(fromisoformat=production_parser)):
        stamp = freshness.parse_time("2026-09-13T06:30:57.195348633Z")
    assert stamp == datetime(2026, 9, 13, 6, 30, 57, 195348, tzinfo=timezone.utc)


def test_overview_and_operations_agree_on_collection_timestamp():
    snapshot = _healthy_snapshot()
    snapshot["asfs_logger_product_sample_time_utc"] = _iso(330)
    with patch.object(mobile, "_pdu_power_snapshot", return_value=({}, "Unavailable")), patch.object(
        mobile, "datetime", wraps=datetime
    ) as clock:
        clock.now.return_value = NOW
        rows = mobile._instrument_power_states(snapshot)
    assert next(row for row in rows if row["id"] == "asfs-logger")["state"] == "No recent data"
    assert next(row for row in rows if row["id"] == "vaisalamet")["state"] == "Collecting"


def test_only_confirmed_off_suppresses_collection_failure():
    snapshot = _healthy_snapshot()
    snapshot["radar_product_sample_time_utc"] = _iso(500)
    assert _operations(snapshot)["overallLevel"] == "red"
    response = _operations(snapshot, {"radar"})
    radar = next(row for row in response["streamStates"] if row["id"] == "cloud-radar")
    assert response["overallLevel"] == "green"
    assert radar["collectionExpected"] is False
    assert "Intentionally off" in radar["detail"]
    assert radar["collectionAgeMinutes"] == 500


def test_product_alert_is_independent_of_archive_verifier_and_source_age_fields():
    snapshot = {"asfs_logger_product_sample_time_utc": _iso(330), "mirror_summary_recent_state": 0}
    rules = evaluate_alerts(snapshot, now=NOW)
    rule = next(rule for rule in rules if rule.id == "stream:asfs_logger:source_stale")
    assert rule.value == 330
    assert "cloud product sample" in rule.message


def test_missing_product_alert_has_persistence_and_respects_confirmed_off():
    snapshot = {"radar_product_sample_available_state": 0, "asfs_logger_product_sample_available_state": 0}
    rules = evaluate_alerts(snapshot, now=NOW, pdu_outlet_states={6: False})
    assert all(rule.id != "stream:radar:collection_unknown" for rule in rules)
    rule = next(rule for rule in rules if rule.id == "stream:asfs_logger:collection_unknown")
    assert rule.hold_minutes == 180


def test_fast_gas_outage_is_monitored_separately():
    rules = evaluate_alerts({"asfs_fast_gas_product_sample_time_utc": _iso(200)}, now=NOW)
    assert any(rule.id == "stream:asfs_fast_gas:source_stale" for rule in rules)


@pytest.mark.parametrize("stamp", ["invalid", _iso(-6), None])
def test_invalid_or_future_product_timestamp_cannot_be_green(stamp):
    state = freshness.collection_state({
        "asfs_logger_product_sample_time_utc": stamp,
        "asfs_logger_product_sample_available_state": 1,
        "asfs_logger_source_age_min": 1,
    }, "asfs_logger", NOW)
    assert state["level"] == "unknown"


def test_collector_reads_real_product_timestamps_without_relabeling_source_evidence(tmp_path, monkeypatch):
    for prefix, (environment, _relative, _threshold) in freshness.PRODUCTS.items():
        monkeypatch.setenv(environment, str(tmp_path / prefix))
    path = tmp_path / "asfs_logger"
    times = np.array(["2026-09-13T00:00:00", "2026-09-13T00:30:00"], dtype="datetime64[ns]")
    xr.Dataset({"sr30_swd_Irr_Avg": ("time", [0.4, 0.5])}, coords={"time": times}).to_zarr(path)
    metrics = collector.collect_product_freshness(NOW)
    assert metrics["asfs_logger_product_age_min"] == 330
    assert metrics["asfs_logger_product_recent_state"] == 0
    assert metrics["asfs_logger_product_sample_time_utc"] == "2026-09-13T00:30:00Z"
    assert metrics["vaisalamet_product_sample_available_state"] == 0
    assert metrics["vaisalamet_product_recent_state"] is None
    assert not any("source_age" in key or "source_recent" in key for key in metrics)


@pytest.mark.parametrize("age", [1, 31, -6])
def test_pdu_suppression_requires_recent_non_future_sample(tmp_path, monkeypatch, age):
    monkeypatch.setenv("PDU_ZARR_PATH", str(tmp_path / "pdu.zarr"))
    stamp = (NOW - timedelta(minutes=age)).replace(tzinfo=None)
    xr.Dataset({"PDUOutlet6State": ("time", [0.0])}, coords={"time": [stamp]}).to_zarr(tmp_path / "pdu.zarr")
    states = freshness.recent_pdu_states(NOW)
    assert states == ({6: False} if age == 1 else None)
