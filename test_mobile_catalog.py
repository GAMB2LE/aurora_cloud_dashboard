from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import mobile_catalog


class MobileCatalogTests(unittest.TestCase):
    def test_manifest_contains_native_sections_and_visible_instruments(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AURORA_SITE_ENV": "development",
                "AURORA_DOMAIN": "data-ocean.gamb2le.co.uk",
                "AURORA_DASHBOARD_REVISION": "abc123def456",
            },
        ):
            manifest = mobile_catalog.manifest()

        self.assertEqual([section["id"] for section in manifest["sections"]], ["overview", "power", "plots", "camera", "ops"])
        self.assertIn("power", {instrument["id"] for instrument in manifest["instruments"]})
        power = next(instrument for instrument in manifest["instruments"] if instrument["id"] == "power")
        self.assertTrue(power["supportsHousekeepingQuicklooks"])
        self.assertIn("fish_hdr", {stream["id"] for stream in manifest["wxcamStreams"]})
        self.assertEqual(manifest["schemaVersion"], 3)
        self.assertTrue(
            {
                "power.current_system_ecmwf_p10_p90",
                "power.assigned_pdu_outlets",
                "operations.instrument_state",
            }.issubset(manifest["capabilities"]["shared"])
        )
        self.assertIn("explore.arbitrary_variables_ranges", manifest["capabilities"]["browser"])
        self.assertEqual(
            manifest["deployment"],
            {
                "environment": "development",
                "domain": "data-ocean.gamb2le.co.uk",
                "dashboardURL": "https://data-ocean.gamb2le.co.uk/app",
                "dataRole": "live-mirror",
                "revision": "abc123def456",
            },
        )

    def test_auroracam_lists_day_times_for_native_time_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "radar-cam" / "2026-07-05"
            source.mkdir(parents=True)
            for stamp in ("12-00", "12-30"):
                (source / f"radar-cam_2026-07-05_{stamp}.jpg").write_bytes(b"jpeg")

            with patch.dict(os.environ, {"AURORACAM_RAW_ROOT": str(root)}):
                response = mobile_catalog.auroracam("2026-07-05")

        self.assertEqual(response["availableTimesUTC"], ["2026-07-05 12:30", "2026-07-05 12:00"])

    def test_uas_window_is_filtered_by_the_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "menapia_mqtt.log"
            recent = datetime.now(timezone.utc) - timedelta(minutes=5)
            path.write_text(
                "2026-07-01 12:00:00: Tier change 1 2\n"
                f"{recent:%Y-%m-%d %H:%M:%S}: Tier change 2 3\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"UAS_MQTT_LOG_PATH": str(path)}):
                response = mobile_catalog.uas("24h")

        self.assertEqual(response["window"], "24h")
        self.assertEqual([record["effectiveTier"] for record in response["records"]], [3])

    def test_shared_pdu_contract_has_only_assigned_outlets(self) -> None:
        self.assertEqual(
            [(title, outlet) for _, title, _, outlet in mobile_catalog.PDU_INSTRUMENTS],
            [("UAS", 4), ("CL61", 5), ("Cloud Radar", 6), ("HATPRO", 8)],
        )

    def test_quicklooks_reports_assigned_instrument_power_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdu = root / "pdu.zarr"
            import numpy as np
            import xarray as xr

            xr.Dataset(
                {"PDUOutlet5State": (("time",), np.array([0.0]))},
                coords={"time": [datetime.now(timezone.utc).replace(tzinfo=None)]},
            ).to_zarr(pdu, mode="w")
            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root), "PDU_ZARR_PATH": str(pdu)}):
                response = mobile_catalog.quicklooks("science", "ceilometer")

        self.assertEqual(response["powerStatus"]["state"], "Off")
        self.assertIn("PDU sample", response["powerStatus"]["detail"])

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

    def test_science_radar_quicklooks_exclude_housekeeping_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "cloud_radar"
            quicklook_dir.mkdir()
            science = quicklook_dir / "latest.png"
            housekeeping = quicklook_dir / "cloud_radar__hk_radar__latest.png"
            science.write_bytes(b"science")
            housekeeping.write_bytes(b"housekeeping")

            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root)}):
                response = mobile_catalog.quicklooks("science", "cloud-radar")
                resolved = mobile_catalog.resolve_quicklook_path("science", "cloud-radar", "latest")

        self.assertEqual(response["latest"]["token"], "latest")
        self.assertEqual(resolved, science)

    def test_science_latest_uses_newer_dated_product_when_alias_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "ceilometer"
            quicklook_dir.mkdir()
            stale_latest = quicklook_dir / "latest.png"
            fresh_daily = quicklook_dir / "ceilometer_20260716.png"
            stale_latest.write_bytes(b"stale")
            fresh_daily.write_bytes(b"fresh")
            os.utime(stale_latest, (1, 1))
            os.utime(fresh_daily, (2, 2))

            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root)}):
                response = mobile_catalog.quicklooks("science", "ceilometer")
                resolved = mobile_catalog.resolve_quicklook_path("science", "ceilometer", "latest")

        self.assertEqual(resolved, fresh_daily)
        self.assertEqual(response["latest"]["title"], "Latest available (2026-07-16)")

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

    def test_operations_excludes_recovered_alert_history(self) -> None:
        alerts = mobile_catalog._active_alerts(
            {
                "alerts": {
                    "active": {"active": True, "title": "Current condition"},
                    "recovered": {"active": False, "title": "Old condition"},
                }
            }
        )

        self.assertEqual([alert["title"] for alert in alerts], ["Current condition"])

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

    def test_overview_includes_meteorology_and_radiation_collection_states(self) -> None:
        rows = mobile_catalog._instrument_power_states(
            {
                "vaisalamet_source_age_min": 5,
                "vaisalamet_source_recent_state": 1,
                "asfs_logger_source_age_min": 185,
                "asfs_logger_source_recent_state": 0,
            }
        )

        meteorology = next(row for row in rows if row["id"] == "vaisalamet")
        radiation = next(row for row in rows if row["id"] == "asfs-logger")
        self.assertEqual([row["id"] for row in rows[:2]], ["vaisalamet", "asfs-logger"])
        self.assertEqual((meteorology["state"], meteorology["level"]), ("Collecting", "green"))
        self.assertEqual((radiation["state"], radiation["level"]), ("No recent data", "red"))

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
