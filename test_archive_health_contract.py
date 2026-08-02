import json
from datetime import datetime, timezone

from collect_operations_snapshot import _merge_archive_health


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


def test_archive_health_contract_exposes_operator_status(tmp_path):
    path = tmp_path / "health-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "health-v1",
                "generated_at": "2026-08-02T12:00:00Z",
                "overall_level": "amber",
                "failures": ["object_store_evidence_stale_hours=14.5"],
                "operator_status": {
                    "level": "amber",
                    "title": "Archive verification is running",
                    "detail": "Strict audit 4 of 5 families complete.",
                    "pruning_paused": True,
                },
                "metrics": {"archive_delivery_pending_count": 12},
            }
        ),
        encoding="utf-8",
    )
    record = {}

    _merge_archive_health(
        record,
        path,
        datetime(2026, 8, 2, 12, 1, tzinfo=timezone.utc),
    )

    assert record["archive_contract_level"] == "amber"
    assert record["archive_health_level"] == "amber"
    assert record["archive_health_title"] == "Archive verification is running"
    assert record["archive_health_detail"] == "Strict audit 4 of 5 families complete."
    assert record["archive_pruning_paused_state"] == 1
    assert record["archive_delivery_pending_count"] == 12
