from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

try:
    from fastapi.testclient import TestClient
    import mobile_api
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without optional API deps
    TestClient = None
    mobile_api = None
    FASTAPI_IMPORT_ERROR = exc
else:
    FASTAPI_IMPORT_ERROR = None


@unittest.skipIf(TestClient is None, f"FastAPI test dependencies are not installed: {FASTAPI_IMPORT_ERROR}")
class MobileAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(mobile_api.app)

    def test_health_is_public_and_reports_token_configuration(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["authRequired"])
        self.assertTrue(response.json()["tokenConfigured"])

    def test_manifest_requires_bearer_token(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False):
            unauthorized = self.client.get("/manifest")
            authorized = self.client.get("/manifest", headers={"Authorization": "Bearer secret"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertIn("power", {instrument["id"] for instrument in authorized.json()["instruments"]})
        self.assertIn("deployment", authorized.json())

    def test_read_only_payloads_are_short_term_cacheable(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False):
            response = self.client.get("/manifest", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "private, max-age=30, stale-while-revalidate=60")

    def test_display_artifact_manifest_requires_bearer_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "latest.json"
            manifest.write_text(json.dumps({"schemaVersion": 1, "artifactCount": 2}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_DISPLAY_ARTIFACT_MANIFEST": str(manifest)},
                clear=False,
            ):
                unauthorized = self.client.get("/artifacts/manifest")
                authorized = self.client.get("/artifacts/manifest", headers={"Authorization": "Bearer secret"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertTrue(authorized.json()["available"])
        self.assertEqual(authorized.json()["artifactCount"], 2)

    def test_operations_endpoint_reads_fixture_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            alerts = root / "state.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "time_utc": "2026-07-05T07:30:00Z",
                        "cl61_source_sync_service_healthy_state": 1,
                        "ceilometer_append_service_healthy_state": 0,
                        "ceilometer_quicklooks_service_healthy_state": 1,
                    }
                ),
                encoding="utf-8",
            )
            health.write_text(json.dumps({"overall_level": "red"}), encoding="utf-8")
            alerts.write_text(json.dumps({"active": {"storage": {"title": "Storage high", "level": "red"}}}), encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
                clear=False,
            ):
                response = self.client.get("/operations", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["overallLevel"], "red")
        self.assertEqual(body["alerts"][0]["title"], "Storage high")
        ceilometer = next(stream for stream in body["streamStates"] if stream["id"] == "ceilometer")
        self.assertEqual(ceilometer["level"], "red")

    def test_overview_and_uas_endpoints_require_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "menapia_mqtt.log"
            log.write_text("2026-07-05 07:30:00: 4 3\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "UAS_MQTT_LOG_PATH": str(log)},
                clear=False,
            ):
                self.assertEqual(self.client.get("/overview").status_code, 401)
                overview = self.client.get("/overview", headers={"Authorization": "Bearer secret"})
                uas = self.client.get("/uas", headers={"Authorization": "Bearer secret"})

        self.assertEqual(overview.status_code, 200)
        self.assertEqual(uas.status_code, 200)
        self.assertEqual(uas.json()["latest"]["effectiveTier"], 3)

    def test_power_accepts_all_group_without_generating_a_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.zarr"
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "POWER_DISPLAY_SUMMARY_ZARR_PATH": str(missing)},
                clear=False,
            ):
                response = self.client.get("/power?window=24h&group=all", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["group"], "all")
        self.assertEqual(response.json()["panels"], [])

    def test_power_accepts_current_and_forecast_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.zarr"
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "POWER_DISPLAY_SUMMARY_ZARR_PATH": str(missing)},
                clear=False,
            ):
                current = self.client.get(
                    "/power?window=24h&group=current",
                    headers={"Authorization": "Bearer secret"},
                )
                forecast = self.client.get(
                    "/power?window=96h&group=forecast",
                    headers={"Authorization": "Bearer secret"},
                )

        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["group"], "current")
        self.assertEqual(forecast.status_code, 200)
        self.assertEqual(forecast.json()["group"], "forecast")

    def test_power_current_group_prefers_the_section_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            current_path = Path(tmp) / "power_current_display.zarr"
            now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).floor("h")
            times = pd.date_range(now - pd.Timedelta(hours=2), periods=4, freq="1h")
            xr.Dataset(
                {"BatterySOC": ("time", np.asarray([70.0, 71.0, 72.0, 73.0]))},
                coords={"time": times},
            ).to_zarr(current_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "POWER_CURRENT_DISPLAY_ZARR_PATH": str(current_path),
                    "POWER_DISPLAY_SUMMARY_ZARR_PATH": str(Path(tmp) / "missing.zarr"),
                },
                clear=False,
            ):
                response = self.client.get("/power?window=24h&group=current", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"]["path"], str(current_path))

    def test_power_prewarmed_figure_is_a_cacheable_media_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "power_current_latest_interactive.json"
            current.write_text('{"data":[],"layout":{}}', encoding="utf-8")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_INTERACTIVE_PREWARM_DIR": str(root)},
                clear=False,
            ):
                response = self.client.get("/media/power/figure/current", headers={"Authorization": "Bearer secret"})
                unavailable = self.client.get("/media/power/figure/unknown", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "private, max-age=60")
        self.assertEqual(response.json(), {"data": [], "layout": {}})
        self.assertEqual(unavailable.status_code, 404)

    def test_auroracam_listing_and_original_media_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "radar-cam" / "2026-07-05"
            source.mkdir(parents=True)
            image = source / "radar-cam_2026-07-05_12-30.jpg"
            image.write_bytes(b"jpeg")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "AURORACAM_RAW_ROOT": str(root)},
                clear=False,
            ):
                listing = self.client.get("/auroracam", headers={"Authorization": "Bearer secret"})
                media = self.client.get(
                    "/media/auroracam/original/radar-cam/2026-07-05/radar-cam_2026-07-05_12-30.jpg",
                    headers={"Authorization": "Bearer secret"},
                )

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["frames"][0]["cameraID"], "radar-cam")
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.content, b"jpeg")

    def test_quicklook_listing_and_media_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "power"
            quicklook_dir.mkdir()
            (quicklook_dir / "power__summary__latest.png").write_bytes(b"png")

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "AURORA_QUICKLOOK_ROOT": str(root),
                },
                clear=False,
            ):
                listing = self.client.get(
                    "/quicklooks?kind=science&instrument=power",
                    headers={"Authorization": "Bearer secret"},
                )
                media = self.client.get(
                    "/media/quicklook/science/power/latest",
                    headers={"Authorization": "Bearer secret"},
                )
                not_modified = self.client.get(
                    "/media/quicklook/science/power/latest",
                    headers={"Authorization": "Bearer secret", "If-None-Match": media.headers["ETag"]},
                )

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["latest"]["imageURL"], "/media/quicklook/science/power/latest")
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.content, b"png")
        self.assertIn("ETag", media.headers)

        self.assertEqual(not_modified.status_code, 304)

    def test_wxcam_listing_and_media_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos" / "fish_hdr"
            thumbs = root / "thumbs" / "fish_hdr" / "20260705"
            videos.mkdir(parents=True)
            thumbs.mkdir(parents=True)
            (videos / "20260705.mp4").write_bytes(b"video")
            (thumbs / "sample.jpg").write_bytes(b"thumb")

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "WXCAM_DAILY_VIDEO_DIR": str(root / "videos"),
                    "WXCAM_HOURLY_THUMB_DIR": str(root / "thumbs"),
                    "WXCAM_CATALOG_PATH": str(root / "missing.sqlite"),
                },
                clear=False,
            ):
                listing = self.client.get(
                    "/wxcam?stream=fish_hdr&day=2026-07-05",
                    headers={"Authorization": "Bearer secret"},
                )
                video = self.client.get(
                    "/media/wxcam/video/fish_hdr/2026-07-05",
                    headers={"Authorization": "Bearer secret"},
                )
                thumb = self.client.get(
                    "/media/wxcam/thumb/fish_hdr/20260705/sample.jpg",
                    headers={"Authorization": "Bearer secret"},
                )

        self.assertEqual(listing.status_code, 200)
        self.assertTrue(listing.json()["video"]["exists"])
        self.assertEqual(video.status_code, 200)
        self.assertEqual(video.content, b"video")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.content, b"thumb")


if __name__ == "__main__":
    unittest.main()
