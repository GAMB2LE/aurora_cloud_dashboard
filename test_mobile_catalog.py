from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mobile_catalog


class MobileCatalogTests(unittest.TestCase):
    def test_manifest_contains_native_sections_and_visible_instruments(self) -> None:
        manifest = mobile_catalog.manifest()

        self.assertEqual([section["id"] for section in manifest["sections"]], ["overview", "power", "plots", "camera", "ops"])
        self.assertIn("power", {instrument["id"] for instrument in manifest["instruments"]})
        power = next(instrument for instrument in manifest["instruments"] if instrument["id"] == "power")
        self.assertTrue(power["supportsHousekeepingQuicklooks"])
        self.assertIn("fish_hdr", {stream["id"] for stream in manifest["wxcamStreams"]})

    def test_quicklooks_find_latest_and_dated_summary_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "power"
            quicklook_dir.mkdir()
            (quicklook_dir / "power__summary__latest.png").write_bytes(b"latest")
            (quicklook_dir / "power__summary__20260705.png").write_bytes(b"dated")

            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root)}):
                response = mobile_catalog.quicklooks("science", "power")

        self.assertEqual(response["latest"]["token"], "latest")
        self.assertEqual([entry["token"] for entry in response["entries"]], ["latest", "20260705"])

    def test_operations_derives_stream_levels_from_snapshot(self) -> None:
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
            alerts.write_text(json.dumps({"active": {"a": {"title": "Storage high", "level": "red"}}}), encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        ceilometer = next(stream for stream in response["streamStates"] if stream["id"] == "ceilometer")
        self.assertEqual(response["overallLevel"], "red")
        self.assertEqual(ceilometer["level"], "red")
        self.assertEqual(response["alerts"][0]["title"], "Storage high")

    def test_operations_uses_current_soc_thresholds(self) -> None:
        self.assertEqual(mobile_catalog._trend_level("battery-soc", 40), "red")
        self.assertEqual(mobile_catalog._trend_level("battery-soc", 45), "amber")
        self.assertEqual(mobile_catalog._trend_level("battery-soc", 50), "green")

    def test_auroracam_latest_listing_and_media_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "radar-cam" / "2026-07-05"
            source.mkdir(parents=True)
            image = source / "radar-cam_2026-07-05_12-30.jpg"
            image.write_bytes(b"jpeg")

            with patch.dict(os.environ, {"AURORACAM_RAW_ROOT": str(root)}):
                response = mobile_catalog.auroracam()
                resolved = mobile_catalog.resolve_auroracam_image_path("radar-cam", "2026-07-05", image.name)

        self.assertEqual(response["selectedDay"], "2026-07-05")
        self.assertEqual(response["frames"][0]["previewURL"], "/media/auroracam/preview/radar-cam/2026-07-05/radar-cam_2026-07-05_12-30.jpg")
        self.assertEqual(resolved, image)

    def test_power_returns_a_small_unavailable_payload_without_creating_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.zarr"
            with patch.dict(os.environ, {"POWER_DISPLAY_SUMMARY_ZARR_PATH": str(missing)}):
                response = mobile_catalog.power()

        self.assertEqual(response["panels"], [])
        self.assertEqual(response["group"], "all")
        self.assertIn("warning", response)

    def test_overview_matches_browser_mobile_card_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            alerts = root / "state.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "time_utc": "2026-07-05T07:30:00Z",
                        "aps_battery_soc_pct": 56,
                        "aps_battery_voltage_v": 52.56,
                        "aps_battery_power_w": -57,
                        "aps_battery_capacity_kwh": 26,
                        "aps_battery_depletion_hours": 255,
                        "power_latest_time_utc": "2026-07-05T07:29:00Z",
                    }
                ),
                encoding="utf-8",
            )
            health.write_text(json.dumps({"overall_level": "amber"}), encoding="utf-8")
            alerts.write_text(json.dumps({"active": {}}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                    "AURORACAM_RAW_ROOT": str(root / "camera"),
                },
            ):
                response = mobile_catalog.overview()

        self.assertEqual(
            [card["id"] for card in response["cards"]],
            ["operations", "battery-soc", "battery-voltage", "battery-depletion", "power", "auroracam"],
        )
        depletion = response["cards"][3]
        self.assertEqual(depletion["value"], "10d 15h")
        self.assertIn("14.6 kWh remaining", depletion["detail"])

    def test_wxcam_discovers_videos_and_thumbnails(self) -> None:
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
                    "WXCAM_DAILY_VIDEO_DIR": str(root / "videos"),
                    "WXCAM_HOURLY_THUMB_DIR": str(root / "thumbs"),
                    "WXCAM_CATALOG_PATH": str(root / "missing.sqlite"),
                },
            ):
                response = mobile_catalog.wxcam("fish_hdr", "2026-07-05")

        self.assertTrue(response["video"]["exists"])
        self.assertEqual(response["availableDays"], ["2026-07-05"])
        self.assertEqual(response["thumbnails"][0]["imageURL"], "/media/wxcam/thumb/fish_hdr/20260705/sample.jpg")

    def test_wxcam_media_resolvers_reject_malformed_day_tokens(self) -> None:
        self.assertIsNone(mobile_catalog.resolve_wxcam_video_path("fish_hdr", ".."))
        self.assertIsNone(mobile_catalog.resolve_wxcam_thumbnail_path("fish_hdr", "..", "sample.jpg"))
        self.assertIsNone(mobile_catalog.resolve_wxcam_thumbnail_path("fish_hdr", "20260705", "../sample.jpg"))


if __name__ == "__main__":
    unittest.main()
