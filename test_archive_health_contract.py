import json
from datetime import datetime, timezone

from collect_operations_snapshot import _merge_archive_health, build_health_assessment


def test_archive_health_contract_is_consumed_without_manifest_or_gws_inputs(tmp_path):
    path = tmp_path / "health-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "health-v1",
                "generated_at": "2026-07-25T20:00:00Z",
                "overall_level": "red",
                "failures": ["object_store_raw_missing=3"],
                "metrics": {
                    "hatpro_gws_missing_count": 0,
                    "radar_prune_ready_state": 0,
                    "mirror_verify_service_healthy_state": 1,
                    "cl61_source_sync_service_healthy_state": 1,
                    "cl61_source_sync_timer_active_state": 1,
                    "failed_source_sync_unit_count": 0,
                    "source_sync_enabled_count": 13,
                },
            }
        ),
        encoding="utf-8",
    )
    record = {}

    _merge_archive_health(
        record,
        path,
        datetime(2026, 7, 25, 20, 5, tzinfo=timezone.utc),
    )

    assert record["archive_health_contract_available_state"] == 1
    assert record["archive_health_level"] == "red"
    assert record["archive_health_failures"] == ["object_store_raw_missing=3"]
    assert record["hatpro_gws_missing_count"] == 0
    assert record["radar_prune_ready_state"] == 0
    assert record["mirror_summary_age_min"] == 5
    assert record["mirror_summary_recent_state"] == 1
    assert record["gws_probe_ok_state"] == 1
    assert record["cl61_source_sync_service_healthy_state"] == 1
    assert record["cl61_source_sync_timer_active_state"] == 1
    assert record["failed_source_sync_unit_count"] == 0
    assert record["source_sync_enabled_count"] == 13


def test_missing_archive_health_contract_fails_closed(tmp_path):
    record = {}

    _merge_archive_health(
        record,
        tmp_path / "missing.json",
        datetime(2026, 7, 25, 20, 5, tzinfo=timezone.utc),
    )

    assert record["archive_health_contract_available_state"] == 0
    assert record["archive_health_level"] == "red"
    assert record["archive_health_failures"] == ["archive_health_contract_missing"]


def test_health_assessment_does_not_require_legacy_source_sync_units():
    health = build_health_assessment({"time_utc": "2026-07-27T00:00:00Z"})

    systemd_names = {
        check["message"]
        for check in health["checks"]
        if check["component"] == "systemd"
    }
    assert "aurora-radar-source-sync.service" not in systemd_names
    assert "aurora-radar-append.service" in systemd_names
