from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import mobile_catalog
from menapia_flight_status import summarize_menapia_flight


NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def health(**overrides):
    source = {
        "enabled": True,
        "commissioned": True,
        "state": "success",
        "last_success_at": "2026-08-25T11:50:00Z",
        "upstream_objects_examined": 12,
        "new_objects_ingested": 2,
        "bytes_transferred": 1024,
        "unclassified_objects": 12,
        "failure_count": 0,
        "authentication_failure": False,
        "latest_source_flight": {"date": "2026-08-25", "flight": "flight-7"},
        "credential": {
            "expires_on": "2026-09-30",
            "days_remaining": 36,
            "level": "green",
        },
        "archive_delivery": {
            "gws_pending_files": 1,
            "object_store_pending_files": 2,
            "dual_delivered_files": 9,
        },
    }
    source.update(overrides)
    return {"source_ingest": {"menapia": source}, "metrics": {}}


def test_healthy_ingest_exposes_delivery_and_classification_context():
    result = summarize_menapia_flight(health(), now=NOW)

    assert result["level"] == "green"
    assert result["latestSourceFlight"] == "flight-7"
    assert result["gwsPendingFiles"] == 1
    assert result["objectStorePendingFiles"] == 2
    assert result["unclassifiedObjects"] == 12
    assert "campaign-unclassified" in result["detail"]


def test_uncommissioned_source_is_visible_without_claiming_success():
    result = summarize_menapia_flight(
        health(commissioned=False, state="unavailable", last_success_at=None),
        now=NOW,
    )

    assert result["level"] == "amber"
    assert "awaits commissioning" in result["title"]


def test_authentication_failure_is_red():
    result = summarize_menapia_flight(
        health(authentication_failure=True, state="failed", failure_count=1),
        now=NOW,
    )

    assert result["level"] == "red"
    assert result["title"] == "Menapia authentication failed"


def test_authentication_failure_is_visible_before_commissioning():
    result = summarize_menapia_flight(
        health(
            commissioned=False,
            authentication_failure=True,
            state="failed",
            failure_count=1,
            last_success_at=None,
        ),
        now=NOW,
    )

    assert result["level"] == "red"
    assert result["title"] == "Menapia authentication failed"


def test_credential_warning_thresholds_are_applied():
    amber = summarize_menapia_flight(
        health(credential={"expires_on": "2026-09-30", "days_remaining": 20}),
        now=NOW,
    )
    red = summarize_menapia_flight(
        health(credential={"expires_on": "2026-09-30", "days_remaining": 7}),
        now=NOW,
    )

    assert amber["level"] == "amber"
    assert red["level"] == "red"


def test_stale_source_is_red_even_when_last_exit_state_was_success():
    result = summarize_menapia_flight(
        health(last_success_at="2026-08-25T09:00:00Z"),
        now=NOW,
    )

    assert result["level"] == "red"
    assert result["title"] == "Flight data ingest is stale"


def test_mobile_uas_payload_includes_flight_ingest_status():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        uas_log = root / "menapia_mqtt.log"
        archive = root / "health-v1.json"
        uas_log.write_text("2026-08-25 11:59:00: 4 4\n", encoding="utf-8")
        archive.write_text(json.dumps(health()), encoding="utf-8")
        with patch.dict(
            os.environ,
            {
                "UAS_MQTT_LOG_PATH": str(uas_log),
                "ARCHIVE_HEALTH_PATH": str(archive),
            },
        ):
            response = mobile_catalog.uas()

    assert response["flightData"]["latestSourceFlight"] == "flight-7"
    assert response["flightData"]["dualDeliveredFiles"] == 9


def test_desktop_uas_panel_contains_flight_data_cards():
    app_source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")

    assert '"Flight-data ingest"' in app_source
    assert '"GWS pending"' in app_source
    assert '"Object-store pending"' in app_source
