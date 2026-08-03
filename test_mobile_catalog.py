from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import mobile_catalog


class MobileCatalogTests(unittest.TestCase):
    def test_operations_prefers_and_merges_archive_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot.write_text('{"overall_level":"green","cl61_gws_missing_count":9}')
            health.write_text('{"overall_level":"green"}')
            archive.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-25T20:00:00Z",
                        "overall_level": "red",
                        "failures": ["object_store_raw_missing=3"],
                        "metrics": {"cl61_gws_missing_count": 0},
                    }
                )
            )
            alerts.write_text("{}")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        self.assertEqual(response["overallLevel"], "red")
        self.assertEqual(response["updatedAt"], "2026-07-25T20:00:00Z")
        archive_alert = next(
            alert
            for alert in response["alerts"]
            if alert["id"] == "archive:verification"
        )
        self.assertIn("object_store_raw_missing=3", archive_alert["detail"])
        archive_group = next(
            group
            for group in response["rootCauseGroups"]
            if group["id"] == "archive"
        )
        self.assertEqual(archive_group["level"], "red")
        self.assertEqual(response["checkCounts"]["red"], 1)
        self.assertEqual(
            response["sources"]["archiveHealth"]["path"],
            str(archive),
        )

    def test_operations_presents_clean_inventory_timeout_as_amber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot_payload: dict[str, int | str] = {
                "time_utc": "2026-08-02T07:09:00Z"
            }
            for spec in mobile_catalog.OPERATIONS_STREAMS:
                snapshot_payload[str(spec["source"])] = 1
                for service in spec["services"]:
                    snapshot_payload[str(service)] = 1
            snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")
            health.write_text('{"overall_level":"red"}', encoding="utf-8")
            archive.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-08-02T07:09:34Z",
                        "overall_level": "red",
                        "failures": [
                            "object_store_evidence_stale_hours=9.81",
                            "archive_service_unhealthy=aurora-object-store-inventory.service",
                        ],
                        "metrics": {
                            "streams_gws_issue_count": 0,
                            "object_store_all_missing_count": 0,
                            "object_store_all_mismatch_count": 0,
                            "gws_all_missing_count": 0,
                            "gws_all_mismatch_count": 0,
                        },
                        "evidence": {
                            "object_store_gate": {
                                "clean": True,
                                "stable_parity": True,
                            },
                            "object_store_inventory_progress": {
                                "state": "failed",
                                "current_job": "products",
                                "error": (
                                    "CalledProcessError: rclone lsjson "
                                    "gamb2le-o/data/output/aurora-cloud/products/"
                                    "asfs_fast_gas"
                                ),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            alerts.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        archive_alert = next(
            alert
            for alert in response["alerts"]
            if alert["id"] == "archive:verification"
        )
        self.assertEqual(response["overallLevel"], "amber")
        self.assertEqual(response["checkCounts"]["amber"], 1)
        self.assertEqual(archive_alert["level"], "amber")
        self.assertEqual(archive_alert["title"], "Archive verification failed")
        self.assertEqual(
            archive_alert["detail"],
            "JASMIN object-store listing timed out for ASFS fast-gas products. "
            "Last complete verification was clean. New pruning is paused until "
            "verification succeeds.",
        )
        archive_group = next(
            group
            for group in response["rootCauseGroups"]
            if group["id"] == "archive"
        )
        self.assertEqual(archive_group["level"], "amber")
        self.assertIn(archive_alert["detail"], archive_group["detail"])
        self.assertIn("Newest-first live delivery queue is clear", archive_group["detail"])
        self.assertIn("last complete check was clean", response["summary"])

    def test_operations_uses_infrastructure_operator_status_while_audit_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot.write_text('{"time_utc":"2026-08-02T12:00:00Z"}')
            health.write_text('{"overall_level":"green"}')
            archive.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-08-02T12:00:00Z",
                        "overall_level": "amber",
                        "failures": ["object_store_evidence_stale_hours=14.5"],
                        "operator_status": {
                            "level": "amber",
                            "title": "Archive verification is running",
                            "detail": (
                                "Strict audit 4 of 5 families complete; checking manifests. "
                                "Last complete verification was clean. New pruning is paused."
                            ),
                            "pruning_paused": True,
                        },
                        "metrics": {
                            "archive_delivery_pending_count": 12,
                            "archive_delivery_gws_pending_count": 0,
                            "archive_delivery_object_store_pending_count": 12,
                        },
                    }
                )
            )
            alerts.write_text(
                json.dumps(
                    {
                        "active": {
                            "archive:health_red": {
                                "active": True,
                                "title": "Archive health is red",
                                "level": "red",
                            }
                        }
                    }
                )
            )
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        self.assertEqual(response["overallLevel"], "amber")
        archive_alerts = [
            alert for alert in response["alerts"] if alert["id"].startswith("archive:")
        ]
        self.assertEqual(len(archive_alerts), 1)
        self.assertEqual(archive_alerts[0]["id"], "archive:verification")
        self.assertEqual(archive_alerts[0]["title"], "Archive verification is running")

    def test_power_trace_sampling_is_bounded_and_preserves_extrema(self) -> None:
        import numpy as np

        values = np.zeros(1_000)
        values[333] = 99.0
        values[777] = -42.0

        indices = mobile_catalog._representative_power_indices(values)

        self.assertLessEqual(len(indices), mobile_catalog.MOBILE_POWER_MAX_POINTS)
        self.assertIn(0, indices)
        self.assertIn(len(values) - 1, indices)
        self.assertIn(333, indices)
        self.assertIn(777, indices)

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

    def test_operations_trend_cards_use_current_aps_metric_names(self) -> None:
        snapshot = {
            "host_aps_data_used_pct": 74.0,
            "aurora_data_used_pct": 77.0,
            "aps_battery_soc_pct": 100.0,
            "aps_battery_voltage_v": 55.6,
        }
        with patch.object(mobile_catalog, "_intentionally_paused_streams", return_value=set()), patch.object(
            mobile_catalog, "_operations_trend_values", return_value={}
        ):
            cards = {card["id"]: card for card in mobile_catalog._trend_cards(snapshot)}

        self.assertEqual(cards["storage"]["value"], 77.0)
        self.assertEqual(cards["battery-soc"]["value"], 100.0)
        self.assertEqual(cards["battery-voltage"]["value"], 55.6)

    def test_operations_trend_history_excludes_intentionally_paused_streams(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ops_monitor.zarr"
            times = pd.date_range(
                datetime.now(timezone.utc) - timedelta(hours=1), periods=3, freq="30min"
            ).tz_localize(None)
            xr.Dataset(
                {
                    "aurora_data_used_pct": (("time",), np.array([75.0, 76.0, 77.0])),
                    "aps_battery_soc_pct": (("time",), np.array([98.0, 99.0, 100.0])),
                    "aps_battery_voltage_v": (("time",), np.array([55.0, 55.3, 55.6])),
                    "cl61_source_age_min": (("time",), np.array([2.0, 3.0, 4.0])),
                    "radar_source_age_min": (("time",), np.array([4.0, 6.0, 8.0])),
                    "hatpro_source_age_min": (("time",), np.array([700.0, 800.0, 900.0])),
                    "cl61_gws_lag_min": (("time",), np.array([1.0, 2.0, 3.0])),
                    "radar_gws_lag_min": (("time",), np.array([2.0, 4.0, 6.0])),
                    "hatpro_gws_lag_min": (("time",), np.array([70.0, 80.0, 90.0])),
                },
                coords={"time": times},
            ).to_zarr(path, mode="w", consolidated=True)
            mobile_catalog._OPERATIONS_TREND_CACHE.clear()
            with patch.dict(os.environ, {"OPS_MONITOR_ZARR_PATH": str(path)}):
                values = mobile_catalog._operations_trend_values({"hatpro"})

        self.assertEqual(values["storage"], 77.0)
        self.assertEqual(values["battery-soc"], 100.0)
        self.assertEqual(values["battery-voltage"], 55.6)
        self.assertEqual(values["source-lag"], 8.0)
        self.assertEqual(values["gws-lag"], 6.0)

    def test_operations_trend_does_not_reuse_an_old_finite_source_lag(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ops_monitor.zarr"
            times = pd.date_range(
                datetime.now(timezone.utc) - timedelta(minutes=10),
                periods=2,
                freq="10min",
            ).tz_localize(None)
            xr.Dataset(
                {
                    "cl61_source_age_min": (
                        ("time",),
                        np.array([12_760.2, np.nan]),
                    ),
                    "cl61_gws_lag_min": (("time",), np.array([4.0, 5.6])),
                },
                coords={"time": times},
            ).to_zarr(path, mode="w", consolidated=True)
            mobile_catalog._OPERATIONS_TREND_CACHE.clear()
            with patch.dict(os.environ, {"OPS_MONITOR_ZARR_PATH": str(path)}):
                values = mobile_catalog._operations_trend_values(set())

        self.assertNotIn("source-lag", values)
        self.assertEqual(values["gws-lag"], 5.6)

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

    def test_forecast_panels_start_at_their_first_forecast_time(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power_display_summary.zarr"
            times = pd.date_range("2026-07-19T07:00:00", periods=31, freq="1h")
            solar = np.full(len(times), np.nan)
            solar[24:] = [150.0, 350.0, 500.0, 300.0, 100.0, 0.0, 0.0]
            xr.Dataset(
                {
                    "ForecastSolarWatts": (("time",), solar),
                    "OperatingCurrentLoadP50Watts": (("time",), np.full(len(times), 250.0)),
                },
                coords={"time": times},
            ).to_zarr(path, mode="w")
            with patch.dict(os.environ, {"POWER_DISPLAY_SUMMARY_ZARR_PATH": str(path)}), patch.object(
                mobile_catalog, "datetime", wraps=datetime
            ) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="24h", group="forecast_96h")

        panel = next(panel for panel in response["panels"] if panel["id"] == "ecmwf_solar_forecast")
        self.assertEqual(panel["info"]["title"], "ECMWF solar and load forecast")
        self.assertTrue(panel["info"]["implementation"])
        for trace in panel["traces"]:
            self.assertTrue(all(point["time"] >= "2026-07-20T07:00:00" for point in trace["points"]))

    def test_forecast_group_merges_latest_uas_tier3_operating_scenario(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display_path = root / "power_forecast_display.zarr"
            scenario_path = root / "power_operating_scenarios.zarr"
            times = pd.date_range("2026-07-20T07:00:00", periods=4, freq="3h")
            xr.Dataset(
                {
                    "BatterySOCForecastP50": (("time",), [80.0, 78.0, 76.0, 74.0]),
                    "OperatingSuggested1SOCP50": (("time",), [99.0, 98.0, 97.0, 96.0]),
                },
                coords={"time": times},
                attrs={"forecast_initial_soc_time": times[0].isoformat()},
            ).to_zarr(display_path, mode="w", consolidated=True)
            scenario_ids = np.asarray(["suggested_all_uas_tier3"], dtype=str)
            xr.Dataset(
                {
                    "ScenarioSOCP50": (("scenario", "time"), [[80.0, 72.0, 64.0, 56.0]]),
                    "scenario_label": (("scenario",), ["All instruments + UAS tier 3 (provisional)"]),
                },
                coords={"scenario": scenario_ids, "time": times},
                attrs={
                    "planning_status": "ready",
                    "initial_soc_time": times[0].isoformat(),
                },
            ).to_zarr(scenario_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "POWER_FORECAST_DISPLAY_ZARR_PATH": str(display_path),
                    "POWER_OPERATING_SCENARIOS_ZARR_PATH": str(scenario_path),
                },
            ), patch.object(mobile_catalog, "datetime", wraps=datetime) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="96h", group="forecast")

        panel = next(panel for panel in response["panels"] if panel["id"] == "operating_plan_scenarios")
        traces = {trace["id"]: trace for trace in panel["traces"]}
        self.assertNotIn("OperatingSuggested1SOCP50", traces)
        self.assertEqual(
            [point["value"] for point in traces["OperatingSuggested8SOCP50"]["points"]],
            [80.0, 72.0, 64.0, 56.0],
        )

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
            ), patch.object(
                mobile_catalog,
                "_environmental_signal_cards",
                return_value=[
                    {"id": "shortwave-down"},
                    {"id": "wind-speed"},
                    {"id": "air-temperature"},
                    {"id": "kt15"},
                ],
            ):
                response = mobile_catalog.overview()

        self.assertEqual(
            [card["id"] for card in response["cards"]],
            [
                "operations", "battery-soc", "battery-voltage", "battery-depletion", "power", "auroracam",
                "shortwave-down", "wind-speed", "air-temperature", "kt15",
            ],
        )
        depletion = response["cards"][3]
        self.assertEqual(depletion["value"], "10d 15h")
        self.assertIn("14.6 kWh remaining", depletion["detail"])

    def test_environmental_signal_cards_derive_wind_and_preserve_source_times(self) -> None:
        with patch.object(
            mobile_catalog,
            "_latest_zarr_sample",
            side_effect=[
                {"time": "2026-07-24T06:06:53Z", "t2_t": 10.48297},
                {
                    "time": "2026-07-24T06:00:00Z",
                    "sr30_swd_Irr_Avg": 21.92821,
                    "kt15_tem_Avg": 10.96667,
                    "metek_x_out_Avg": -3.745334,
                    "metek_y_out_Avg": 0.8551666,
                },
            ],
        ):
            cards = mobile_catalog._environmental_signal_cards()

        by_id = {card["id"]: card for card in cards}
        self.assertEqual(list(by_id), ["shortwave-down", "wind-speed", "air-temperature", "kt15"])
        self.assertEqual(by_id["air-temperature"]["value"], "10.5 C")
        self.assertEqual(by_id["shortwave-down"]["value"], "21.9 W/m2")
        self.assertEqual(by_id["wind-speed"]["value"], "3.8 m/s")
        self.assertEqual(by_id["kt15"]["value"], "11.0 C")
        self.assertEqual(by_id["kt15"]["updatedAt"], "2026-07-24T06:00:00Z")

    def test_overview_prefers_cached_measured_power_time_without_opening_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            alerts = root / "state.json"
            snapshot.write_text(
                json.dumps({"aps_battery_power_time_utc": "2026-07-21T20:40:00Z"}),
                encoding="utf-8",
            )
            health.write_text(json.dumps({}), encoding="utf-8")
            alerts.write_text(json.dumps({}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                    "AURORACAM_RAW_ROOT": str(root / "camera"),
                },
            ), patch.object(
                mobile_catalog, "_latest_power_time", side_effect=AssertionError("display Zarr opened")
            ):
                response = mobile_catalog.overview()

        power = next(card for card in response["cards"] if card["id"] == "power")
        self.assertEqual(power["updatedAt"], "2026-07-21T20:40:00Z")
        self.assertEqual(power["value"], "20:40 UTC")

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

    def test_collection_states_fall_back_to_environmental_sample_times(self) -> None:
        rows = mobile_catalog._instrument_power_states(
            {},
            {
                "vaisalamet": mobile_catalog.utc_now_iso(),
                "asfs-logger": mobile_catalog.utc_now_iso(),
            },
        )

        meteorology = next(row for row in rows if row["id"] == "vaisalamet")
        radiation = next(row for row in rows if row["id"] == "asfs-logger")
        self.assertEqual((meteorology["state"], meteorology["level"]), ("Collecting", "green"))
        self.assertEqual((radiation["state"], radiation["level"]), ("Collecting", "green"))
        self.assertIn("Source sample", meteorology["detail"])

    def test_cl61_overview_names_the_learned_heater_blower_phase(self) -> None:
        fake_group = SimpleNamespace(
            attrs={
                "current_mode": "dc_cl61",
                "last_observation_time_utc": mobile_catalog.utc_now_iso(),
                "mode_load_profiles": json.dumps(
                    {"dc_cl61": {"current_phase": "fan_high"}}
                ),
            }
        )
        fake_zarr = SimpleNamespace(open_group=lambda *_args, **_kwargs: fake_group)
        with patch.dict(sys.modules, {"zarr": fake_zarr}):
            labels = mobile_catalog._automatic_power_labels()

        self.assertEqual(labels, {"ceilometer": "On with Heater/Blower"})
        row = mobile_catalog._pdu_instrument_status(
            "ceilometer",
            {5: True},
            "PDU sample 2m old",
            labels,
        )
        self.assertEqual(row["state"], "On with Heater/Blower")
        self.assertEqual(row["level"], "green")

    def test_cl61_heater_blower_label_never_overrides_powered_off(self) -> None:
        row = mobile_catalog._pdu_instrument_status(
            "ceilometer",
            {5: False},
            "PDU sample 2m old",
            {"ceilometer": "On with Heater/Blower"},
        )

        self.assertEqual(row["state"], "Off")

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
