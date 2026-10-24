"""HATPRO publisher failures are independent of power and cached samples."""
from datetime import datetime, timezone
import json
from unittest.mock import patch

import pytest
import mobile_catalog


@pytest.fixture
def continuity(tmp_path, monkeypatch):
    science = tmp_path / "science"
    hatpro = science / "hatpro"
    (hatpro / "current").mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    (hatpro / "current" / "generation.json").write_text(json.dumps({"completed_at": now}))
    status = hatpro / "status.json"
    status.write_text(json.dumps({"state": "failed", "failed_at": now, "stream": "hatpro",
                                  "error": "private command should not be displayed"}))
    monkeypatch.setenv("AURORA_DATA_MODE", "direct-field-continuity")
    monkeypatch.setenv("AURORA_SCIENCE_CONTINUITY_ROOT", str(science))
    monkeypatch.setenv("AURORA_CONTINUITY_STATUS_PATH", str(tmp_path / "core" / "status.json"))
    return status


def test_failure_overrides_fresh_collection_on_overview_and_instrument(continuity):
    with patch.object(mobile_catalog, "_pdu_power_snapshot", return_value=({8: True}, "PDU sample 1m old")), \
         patch.object(mobile_catalog, "_powered_instrument_labels", return_value={}), \
         patch.object(mobile_catalog, "_powered_collection_detail", return_value="Source sample 1m old"):
        overview = next(r for r in mobile_catalog._instrument_power_states({}) if r["id"] == "hatpro")
        instrument = mobile_catalog.pdu_instrument_status("hatpro")
    assert instrument == overview
    assert (overview["state"], overview["level"]) == ("Publication failed", "red")
    assert "Last successful publication:" in overview["detail"]
    assert "private command" not in overview["detail"]


def test_failure_does_not_require_fresh_pdu_or_product(continuity):
    row = mobile_catalog._pdu_instrument_status("hatpro", {}, "PDU unavailable")
    assert (row["state"], row["level"]) == ("Publication failed", "red")


def test_confirmed_power_off_remains_off(continuity):
    row = mobile_catalog._pdu_instrument_status("hatpro", {8: False}, "PDU sample 1m old")
    assert row["state"] == "Off"
    assert row["detail"] == "PDU sample 1m old"
    assert mobile_catalog.infrastructure_status()["workers"][3]["state"] == "failed"


@pytest.mark.parametrize("state", ["complete", "degraded", "running", "unknown"])
def test_only_failed_publication_overrides_status(continuity, state):
    continuity.write_text(json.dumps({"state": state, "completed_at": datetime.now(timezone.utc).isoformat()}))
    row = mobile_catalog._pdu_instrument_status("hatpro", {8: True}, "PDU sample 1m old",
                                               collection_detail="Source sample 1m old")
    assert (row["state"], row["level"]) == ("Collecting", "green")


@pytest.mark.parametrize("instrument,outlet", [("ceilometer", 5), ("cloud-radar", 6), ("uas", 4)])
def test_other_instruments_unchanged(continuity, instrument, outlet):
    with patch.object(mobile_catalog, "infrastructure_status", side_effect=AssertionError("unexpected read")):
        row = mobile_catalog._pdu_instrument_status(instrument, {outlet: True}, "PDU sample 1m old",
                                                   collection_detail="Source sample 1m old")
    assert (row["state"], row["level"]) == ("Collecting", "green")


def test_canonical_mode_ignores_continuity_status(continuity, monkeypatch):
    monkeypatch.delenv("AURORA_DATA_MODE")
    row = mobile_catalog._pdu_instrument_status("hatpro", {8: True}, "PDU sample 1m old")
    assert (row["state"], row["level"]) == ("On", "green")


@pytest.mark.parametrize("content", [None, "invalid json", "[]"])
def test_missing_or_invalid_status_does_not_invent_failure(continuity, content):
    if content is None:
        continuity.unlink()
    else:
        continuity.write_text(content)
    row = mobile_catalog._pdu_instrument_status("hatpro", {8: True}, "PDU sample 1m old")
    assert (row["state"], row["level"]) == ("On", "green")
