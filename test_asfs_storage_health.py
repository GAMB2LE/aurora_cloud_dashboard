import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from asfs_storage_health import PREFIX, storage_metrics, storage_status
from collect_operations_snapshot import _collect_asfs_storage_metrics
from send_ops_alerts import evaluate_alerts
from test_collection_freshness import NOW, _iso, _healthy_snapshot, _operations


def _payload(slots=54000, age=5):
    return {
        "schema_version": 1, "generated_at": _iso(age), "state": "normal",
        "file_count": 14000, "estimated_directory_slots": slots,
        "estimated_remaining_slots": 65536 - slots,
        "directory_slot_limit": 65536, "reserve_slots": 256,
        "estimate_only": True,
    }


@pytest.mark.parametrize("slots,level", [(54000, "green"), (55000, "amber"), (60000, "red"), (65517, "red")])
def test_headroom_thresholds_do_not_require_authenticated_card_status(slots, level):
    metrics = storage_metrics(_payload(slots), NOW)
    assert metrics[PREFIX + "level"] == level
    assert "filename estimate" in metrics[PREFIX + "detail"]
    assert "not authenticated card status" in metrics[PREFIX + "detail"]


def test_expired_or_missing_storage_evidence_degrades_and_alerts_with_persistence():
    for payload in ({}, _payload(age=121)):
        metrics = storage_metrics(payload, NOW)
        assert storage_status(metrics, NOW)["state"] == "unknown"
        rule = next(rule for rule in evaluate_alerts(metrics, now=NOW) if rule.id == "logger:directory-headroom")
        assert rule.level == "amber"
        assert rule.hold_minutes == 180


def test_current_products_cannot_mask_critical_logger_headroom():
    snapshot = _healthy_snapshot()
    snapshot.update(storage_metrics(_payload(65517), NOW))
    result = _operations(snapshot)
    assert result["overallLevel"] == "red"
    assert result["checkCounts"]["red"] == 1
    assert "headroom" in result["summary"]
    assert next(group for group in result["rootCauseGroups"] if group["id"] == "logger-storage")["level"] == "red"
    assert next(alert for alert in result["alerts"] if alert["id"] == "logger:directory-headroom")["level"] == "red"


def test_storage_ssh_read_is_bounded_and_timeout_is_visible(monkeypatch):
    monkeypatch.setenv("ASFS_LOGGER_SOURCE_HOST", "ass.example")
    monkeypatch.setenv("ASFS_LOGGER_SOURCE_USER", "aurora")
    record = {}
    with patch("collect_operations_snapshot._run", return_value=SimpleNamespace(stdout=json.dumps(_payload(65517)))) as run:
        _collect_asfs_storage_metrics(record, NOW)
    assert run.call_args.kwargs["timeout"] == 15
    assert run.call_args.args[0][-1] == "head -c 65536 /home/aurora/data/asfs/logger_storage_health.json"
    assert record[PREFIX + "level"] == "red"
    with patch("collect_operations_snapshot._run", side_effect=subprocess.TimeoutExpired("ssh", 15)):
        _collect_asfs_storage_metrics(record, NOW)
    assert record[PREFIX + "state"] == "unknown"
