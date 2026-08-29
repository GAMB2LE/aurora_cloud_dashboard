import json
from datetime import timedelta
from unittest.mock import patch

from send_ops_alerts import evaluate_alerts, process_alerts


def _ids(snapshot):
    return {alert.id for alert in evaluate_alerts(snapshot)}


def test_low_internal_temperature_alerts_separately_from_high_temp():
    assert "power:internal_temp_low" in _ids({"aps_internal_temp_c": 4.9})
    assert "power:internal_temp" in _ids({"aps_internal_temp_c": 45.0})


def test_normal_internal_temperature_does_not_alert():
    ids = _ids({"aps_internal_temp_c": 10.0})

    assert "power:internal_temp_low" not in ids
    assert "power:internal_temp" not in ids


def test_battery_soc_alert_uses_40_percent_operational_minimum():
    assert "power:battery_soc" in _ids({"aps_battery_soc_pct": 40.0})
    assert "power:battery_soc" not in _ids({"aps_battery_soc_pct": 40.1})


def test_dewpoint_alert_requires_internal_humidity_available():
    assert "power:internal_dewpoint" not in _ids(
        {
            "aps_internal_humidity_available_state": 0,
            "aps_internal_dewpoint_margin_c": -1.0,
        }
    )


def test_dewpoint_alerts_when_margin_is_at_or_below_zero():
    ids = _ids(
        {
            "aps_internal_humidity_available_state": 1,
            "aps_internal_dewpoint_margin_c": 0.0,
            "aps_internal_humidity_pct": 100.0,
            "aps_internal_dewpoint_c": 5.0,
            "aps_internal_dewpoint_temp_c": 5.0,
        }
    )

    assert "power:internal_dewpoint" in ids


def test_stale_manifest_replaces_false_stream_staleness():
    ids = _ids(
        {
            "mirror_summary_age_min": 20_000,
            "mirror_summary_recent_state": 0,
            "cl61_source_age_min": 20_000,
        }
    )

    assert "transfer:mirror_manifest_stale" in ids
    assert "stream:cl61:source_stale" not in ids


def test_red_archive_contract_generates_one_authoritative_alert():
    alerts = evaluate_alerts(
        {
            "archive_health_level": "red",
            "archive_health_failures": [
                "object_store_raw_missing=3",
                "object_store_stable_parity=false",
            ],
            "mirror_summary_recent_state": 1,
        }
    )

    archive = [alert for alert in alerts if alert.id == "archive:health_red"]
    assert len(archive) == 1
    assert "object_store_raw_missing=3" in archive[0].message


def test_red_archive_alert_uses_operator_wording():
    alerts = evaluate_alerts(
        {
            "archive_health_level": "red",
            "archive_health_title": "Archive delivery needs action",
            "archive_health_detail": "Object storage is missing 3 settled raw files.",
            "archive_health_failures": ["object_store_raw_missing=3"],
            "mirror_summary_recent_state": 1,
        }
    )

    archive = next(alert for alert in alerts if alert.id == "archive:health_red")
    assert archive.title == "Archive delivery needs action"
    assert archive.message == "Object storage is missing 3 settled raw files."


def test_recent_manifest_preserves_real_stream_staleness():
    ids = _ids(
        {
            "mirror_summary_age_min": 5,
            "mirror_summary_recent_state": 1,
            "cl61_source_age_min": 181,
        }
    )

    assert "transfer:mirror_manifest_stale" not in ids
    assert "stream:cl61:source_stale" in ids


def test_pdu_power_off_suppresses_expected_instrument_staleness():
    alerts = evaluate_alerts(
        {
            "mirror_summary_recent_state": 1,
            "cl61_source_age_min": 447,
            "hatpro_source_age_min": 451,
            "radar_source_age_min": 451,
        },
        pdu_outlet_states={5: False, 6: True, 8: False},
    )
    ids = {alert.id for alert in alerts}

    assert "stream:cl61:source_stale" not in ids
    assert "stream:hatpro:source_stale" not in ids
    assert "stream:radar:source_stale" in ids


def test_missing_pdu_evidence_does_not_suppress_staleness():
    alerts = evaluate_alerts(
        {"mirror_summary_recent_state": 1, "cl61_source_age_min": 447},
        pdu_outlet_states=None,
    )

    assert "stream:cl61:source_stale" in {alert.id for alert in alerts}


def test_storage_alerts_deduplicate_shared_remote_filesystem():
    alerts = evaluate_alerts(
        {
            "host_celine_data_used_pct": 86,
            "host_celine_data_free_gb": 420,
            "host_celine_data_resolved_path": "/home/aurora/data",
            "host_celine_data_probe_target": "aurora@100.124.55.22",
            "host_celine_data_filesystem": "/dev/sdb1",
            "host_ass_data_used_pct": 86,
            "host_ass_data_free_gb": 420,
            "host_ass_data_resolved_path": "/home/aurora/data",
            "host_ass_data_probe_target": "aurora@100.124.55.22",
            "host_ass_data_filesystem": "/dev/sdb1",
        }
    )
    storage = [alert for alert in alerts if alert.id.startswith("storage:")]

    assert [alert.id for alert in storage] == ["storage:host_ass_data"]
    assert storage[0].title == "ASS shared data disk storage at 86.0%"


def test_storage_alert_severity_escalates_at_90_percent():
    below_threshold = evaluate_alerts({"aurora_root_used_pct": 79.9})
    attention = evaluate_alerts({"aurora_root_used_pct": 80.0})
    still_attention = evaluate_alerts({"aurora_root_used_pct": 89.9})
    action = evaluate_alerts({"aurora_root_used_pct": 90.0})

    assert not [alert for alert in below_threshold if alert.id == "storage:aurora_root"]
    assert next(alert for alert in attention if alert.id == "storage:aurora_root").level == "amber"
    assert next(alert for alert in still_attention if alert.id == "storage:aurora_root").level == "amber"
    assert next(alert for alert in action if alert.id == "storage:aurora_root").level == "red"


def test_storage_alert_context_and_severity_survive_state_round_trip(tmp_path):
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "alerts.jsonl"
    first_snapshot = {
        "time_utc": "2026-08-29T08:00:00Z",
        "aurora_root_used_pct": 80.0,
        "aurora_root_free_gb": 19.5,
        "aurora_root_resolved_path": "/",
    }
    escalated_snapshot = {
        **first_snapshot,
        "time_utc": "2026-08-29T08:05:00Z",
        "aurora_root_used_pct": 92.0,
        "aurora_root_free_gb": 7.5,
    }

    with patch("send_ops_alerts._recent_pdu_outlet_states", return_value=None), patch(
        "send_ops_alerts._transport_configured", return_value=False
    ):
        process_alerts(first_snapshot, state_path=state_path, log_path=log_path)
        first_state = json.loads(state_path.read_text(encoding="utf-8"))
        process_alerts(
            escalated_snapshot,
            state_path=state_path,
            log_path=log_path,
            repeat_after=timedelta(hours=12),
        )

    first_entry = first_state["alerts"]["storage:aurora_root"]
    assert first_entry["level"] == "amber"
    assert first_entry["message"] == (
        "AURORA Cloud root disk is using 80.0% of capacity, free=19.5 GB. Path: /."
    )
    assert first_entry["threshold"] == ">= 80%"

    persisted_entry = json.loads(state_path.read_text(encoding="utf-8"))["alerts"][
        "storage:aurora_root"
    ]
    assert persisted_entry["active"] is True
    assert persisted_entry["first_seen_utc"] == "2026-08-29T08:00:00Z"
    assert persisted_entry["last_seen_utc"] == "2026-08-29T08:05:00Z"
    assert persisted_entry["last_value"] == 92.0
    assert persisted_entry["level"] == "red"
    assert persisted_entry["message"] == (
        "AURORA Cloud root disk is using 92.0% of capacity, free=7.50 GB. Path: /."
    )
