from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import app


class DashboardShellTests(TestCase):
    def test_browser_performance_probe_is_development_only(self) -> None:
        with (
            patch.object(app, "SITE_ENV", "development"),
            patch.dict("os.environ", {"AURORA_BROWSER_RUM_ENABLED": "1"}, clear=False),
        ):
            probe = app._browser_performance_probe()

        self.assertIsInstance(probe, app.BrowserPerformanceProbe)
        self.assertIn("browser_first_power_plot", probe._esm)
        self.assertIn("browser_power_section_switch", probe._esm)

        with (
            patch.object(app, "SITE_ENV", "production"),
            patch.dict("os.environ", {"AURORA_BROWSER_RUM_ENABLED": "1"}, clear=False),
        ):
            self.assertIsNone(app._browser_performance_probe())

    def test_browser_performance_probe_rejects_unknown_events(self) -> None:
        probe = app.BrowserPerformanceProbe()
        events = []
        with patch.object(app, "_perf_log", side_effect=lambda event, **fields: events.append((event, fields))):
            probe._handle_msg({"event": "untrusted", "duration_ms": 1})
            probe._handle_msg({"event": "browser_document_ready", "duration_ms": 12.5, "path": "/app"})

        self.assertEqual(events, [("browser_document_ready", {"duration_ms": 12.5, "instrument": "power", "path": "/app"})])

    def test_power_section_prewarm_paths_are_distinct(self) -> None:
        original = app.power_view_select.value
        try:
            app.power_view_select.value = "current"
            current = app._prewarmed_interactive_path("power")
            app.power_view_select.value = "forecast"
            forecast = app._prewarmed_interactive_path("power")
        finally:
            app.power_view_select.value = original

        self.assertEqual(current.name, "power_current_latest_interactive.json")
        self.assertEqual(forecast.name, "power_forecast_latest_interactive.json")

    def test_power_live_window_uses_a_fresh_section_prewarm(self) -> None:
        end = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        start = end - app.DEFAULT_WINDOW
        cache_key = (
            "power", "current", app._interactive_final_quality("power"), "power_latest_5min", "power_latest_5min",
            start.isoformat(), end.isoformat(), 0, 1, "", "", 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        )
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"AURORA_INTERACTIVE_PREWARM_DIR": tmpdir}, clear=False
        ):
            path = Path(tmpdir) / "power_current_latest_interactive.json"
            path.write_text('{"data":[],"layout":{}}', encoding="utf-8")
            self.assertTrue(app._cache_key_targets_latest_prewarm(cache_key, "power"))

    def test_power_live_window_does_not_open_raw_data_when_prewarmed(self) -> None:
        end = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        start = end - app.DEFAULT_WINDOW
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"AURORA_INTERACTIVE_PREWARM_DIR": tmpdir}, clear=False
        ), patch.object(app, "_dataset_time_bounds", side_effect=AssertionError("raw data should not be read")):
            (Path(tmpdir) / "power_current_latest_interactive.json").write_text("{}", encoding="utf-8")
            self.assertTrue(app._is_power_latest_window(start, end, "power"))

    def test_power_section_window_reads_only_the_selected_compact_store(self) -> None:
        times = pd.date_range("2026-07-20T00:00:00", periods=5, freq="1h")
        dataset = xr.Dataset(
            {"BatterySOC": (("time",), np.arange(len(times), dtype=float))},
            coords={"time": times},
        )
        with TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir) / "power_current_display.zarr"
            dataset.to_zarr(current_path, mode="w", consolidated=True)
            previous = dict(app._POWER_DISPLAY_SECTION_DS)
            previous_times = dict(app._POWER_DISPLAY_SECTION_REFRESHED_AT)
            app._POWER_DISPLAY_SECTION_DS.clear()
            app._POWER_DISPLAY_SECTION_REFRESHED_AT.clear()
            try:
                with patch.dict("os.environ", {"POWER_CURRENT_DISPLAY_ZARR_PATH": str(current_path)}, clear=False):
                    result = app._open_power_display_summary_window(times[1], times[3], section="current")
            finally:
                app._refresh_power_display_energy_dataset()
                app._POWER_DISPLAY_SECTION_DS.update(previous)
                app._POWER_DISPLAY_SECTION_REFRESHED_AT.update(previous_times)

        self.assertIsNotNone(result)
        self.assertEqual(list(pd.DatetimeIndex(result["time"].values)), list(times[1:4]))

    def test_power_query_selects_power_before_interactive_callbacks(self) -> None:
        original_instrument = app.instrument_select.value
        original_view = app.power_view_select.value
        try:
            with patch.object(app, "_request_query_args", return_value={"tab": "power", "power_view": "forecast"}):
                app._apply_query_state()
            self.assertEqual(app.instrument_select.value, "power")
            self.assertEqual(app.power_view_select.value, "forecast")
        finally:
            app.instrument_select.value = original_instrument
            app.power_view_select.value = original_view

    def test_slow_interactive_render_emits_a_budget_event(self) -> None:
        events = []
        with (
            patch.object(app, "INTERACTIVE_RENDER_BUDGET_MS", 0),
            patch.object(app, "_perf_log", side_effect=lambda event, **fields: events.append((event, fields))),
        ):
            with app._timed_perf("interactive_view_update", instrument="power") as details:
                details["status"] = "ok"

        self.assertEqual(events[0][0], "interactive_view_update")
        self.assertEqual(events[1][0], "interactive_render_budget_exceeded")
        self.assertEqual(events[1][1]["source_event"], "interactive_view_update")

    def test_forecast_info_control_uses_deployed_panel_widget_api(self) -> None:
        panel = next(
            panel
            for panel in app.SUMMARY_LAYOUTS["power"]
            if panel.key == "soc_ecmwf_forecast"
        )

        control = app._forecast_plot_info_control(panel, xr.Dataset())

        self.assertIsNotNone(control)
        self.assertEqual(control.objects[0].objects[-1].name, "Info")

    def test_desktop_shell_has_full_named_tabs(self) -> None:
        labels = [label for label, _slug, _panel in app.DESKTOP_TAB_SPECS]

        self.assertEqual(
            labels,
            [
                "Overview",
                "Interactive Data Browser",
                "Power",
                "Science Quicklooks",
                "House Keeping Quicklooks",
                "AURORACam",
                "UAS",
                "Operations Dashboard",
            ],
        )
        self.assertEqual(len(app.desktop_tabs), len(labels))
        self.assertTrue(app.desktop_tabs.dynamic)
        self.assertIn("Overview", labels)

    def test_desktop_interactive_and_power_tabs_use_distinct_hosts(self) -> None:
        self.assertIsNot(app.TAB_PANEL_BY_SLUG["interactive"], app.TAB_PANEL_BY_SLUG["power"])

        app._sync_browser_tab_instrument("power")
        self.assertEqual(app.interactive_tab_host.objects, [])
        self.assertEqual(app.power_tab_host.objects, [app.interactive_tab])

        app._sync_browser_tab_instrument("interactive")
        self.assertEqual(app.interactive_tab_host.objects, [app.interactive_tab])
        self.assertEqual(app.power_tab_host.objects, [])

    def test_desktop_tab_labels_scroll_without_abbreviating(self) -> None:
        self.assertIn(":host(.desktop-tabs) .bk-header", app.css)
        self.assertIn("overflow-x: auto", app.css)
        self.assertIn(":host(.desktop-tabs) .bk-tab", app.css)
        self.assertIn("white-space: nowrap", app.css)

    def test_desktop_controls_keep_compact_navigation_rows(self) -> None:
        controls_body = app.controls.objects[0]
        first_row_names = [widget.name for widget in controls_body.objects[0].objects]
        second_row_names = [widget.name for widget in controls_body.objects[1].objects]

        self.assertEqual(first_row_names, ["Instrument", "Start (UTC)", "End (UTC)", "Live Off"])
        self.assertEqual(second_row_names, ["Previous Day", "Reset View Defaults", "Next Day/Current Day"])

    def test_phone_shell_keeps_operational_groups(self) -> None:
        self.assertEqual(list(app.MOBILE_TAB_OPTIONS), ["Overview", "Power", "Plots", "Camera", "Ops"])

    def test_browser_overview_uses_one_icon_led_instrument_status_list(self) -> None:
        overview = {
            "instrumentPower": [
                {"id": "vaisalamet", "title": "Meteorology", "systemImage": "cloud.sun", "state": "Collecting", "level": "green", "detail": "Latest sample 1 min old"},
                {"id": "asfs-logger", "title": "Radiation", "systemImage": "sun.max", "state": "Collecting", "level": "green", "detail": "Latest sample 1 min old"},
                {"id": "uas", "title": "UAS", "systemImage": "airplane", "state": "On", "level": "green", "detail": "PDU sample 2 min old"},
                {"id": "ceilometer", "title": "CL61", "systemImage": "laser.burst", "state": "Off", "level": "unknown", "detail": "PDU sample 2 min old"},
                {"id": "cloud-radar", "title": "Cloud Radar", "systemImage": "dot.radiowaves.left.and.right", "state": "On", "level": "green", "detail": "PDU sample 2 min old"},
                {"id": "hatpro", "title": "HATPRO", "systemImage": "antenna.radiowaves.left.and.right", "state": "Off", "level": "unknown", "detail": "PDU sample 2 min old"},
            ]
        }
        with patch.object(app.mobile_catalog, "overview", return_value=overview):
            markup = app._browser_overview_instrument_markup()

        self.assertIn("Instrument status", markup)
        self.assertNotIn("PDU-controlled instruments", markup)
        self.assertNotIn("Collection-only instruments", markup)
        self.assertEqual(markup.count("data-instrument-id="), 6)
        for instrument_id, system_image in (
            ("vaisalamet", "cloud.sun"),
            ("asfs-logger", "sun.max"),
            ("uas", "airplane"),
            ("ceilometer", "laser.burst"),
            ("cloud-radar", "dot.radiowaves.left.and.right"),
            ("hatpro", "antenna.radiowaves.left.and.right"),
        ):
            self.assertIn(f"data-instrument-id='{instrument_id}'", markup)
            self.assertIn(f"data-instrument-icon='{system_image}'", markup)

    def test_overview_refreshes_when_selected(self) -> None:
        with patch.object(app, "_refresh_browser_overview") as refresh:
            app._ensure_active_tab_loaded("overview")

        refresh.assert_called_once_with()

    def test_empty_pdu_instrument_view_explains_intentional_power_off(self) -> None:
        with patch.object(
            app.mobile_catalog,
            "pdu_instrument_status",
            return_value={"state": "Off", "detail": "PDU sample 2 min old"},
        ):
            figure = app._empty_interactive_figure("Ceilometer", "No samples", start=datetime(2026, 7, 19), end=datetime(2026, 7, 20))

        annotation = figure.layout.annotations[0]
        self.assertIn("INTENTIONAL POWER-OFF", annotation.text)
        self.assertIn("Data collection is paused", annotation.text)
        self.assertEqual(annotation.bgcolor, "#edf8f6")

    def test_operations_marks_stale_pdu_off_streams_as_paused(self) -> None:
        snapshot = {
            "time_utc": "2026-07-19T12:00:00Z",
            "cl61_source_recent_state": 0,
            "cl61_source_age_min": 447,
            "radar_source_recent_state": 0,
            "radar_source_age_min": 451,
            "hatpro_source_recent_state": 0,
            "hatpro_source_age_min": 451,
            "vaisalamet_source_recent_state": 1,
            "asfs_logger_source_recent_state": 1,
            "asfs_fast_sonic_source_recent_state": 1,
            "power_source_recent_state": 1,
            "wxcam_source_recent_state": 1,
            "source_host_probe_fail_count": 0,
        }
        with patch.object(app.mobile_catalog, "pdu_outlet_states", return_value={5: False, 6: True, 8: False}):
            paused = app._ops_expected_paused_prefixes()
            recent, stale, paused_count = app._ops_source_health(snapshot, paused)

        self.assertEqual(paused, {"cl61", "hatpro"})
        self.assertEqual((recent, stale, paused_count), (5, 1, 2))
        self.assertIn("Paused - PDU outlet off", app._ops_source_freshness_text(snapshot, "cl61", intentionally_paused=True))

    def test_combined_operations_series_allows_all_missing_columns(self) -> None:
        dataset = xr.Dataset({"source_age": (("time",), np.array([np.nan, np.nan]))})

        combined = app._ops_combined_series(dataset, ("source_age",))

        self.assertTrue(np.isnan(combined).all())

    def test_live_query_uses_current_window_instead_of_stale_url_dates(self) -> None:
        current_start = datetime(2026, 7, 15, 10, 30)
        current_end = datetime(2026, 7, 16, 10, 30)
        with patch.object(app, "_last_24h_utc_window", return_value=(current_start, current_end)):
            state = app._query_interactive_time_state(
                {"start": "2026-07-15T07:04:01", "end": "2026-07-16T07:04:01", "live": "1"},
                "power",
            )

        self.assertEqual(state, (current_start, current_end, True))

    def test_non_live_query_preserves_historical_window(self) -> None:
        state = app._query_interactive_time_state(
            {"start": "2026-07-15T07:04:01", "end": "2026-07-16T07:04:01", "live": "0"},
            "power",
        )

        self.assertEqual(
            state,
            (datetime(2026, 7, 15, 7, 4, 1), datetime(2026, 7, 16, 7, 4, 1), False),
        )

    def test_power_time_bounds_ignore_forecast_only_rows(self) -> None:
        times = pd.date_range("2026-07-16T08:00:00", periods=7, freq="1h")
        measured = np.array([50.0, 51.0, 52.0, np.nan, np.nan, np.nan, np.nan])
        forecast = np.array([np.nan, np.nan, 52.0, 53.0, 54.0, 55.0, 56.0])
        dataset = xr.Dataset(
            {"BatterySOC": (("time",), measured), "BatterySOCForecast": (("time",), forecast)},
            coords={"time": times},
        )

        lower, upper, raw_count, valid_count = app._time_bounds_from_power_display_dataset(dataset)

        self.assertEqual(lower, datetime(2026, 7, 16, 8, 0))
        self.assertEqual(upper, datetime(2026, 7, 16, 10, 0))
        self.assertEqual(raw_count, 7)
        self.assertEqual(valid_count, 3)

    def test_operating_scenario_cache_reopens_an_incomplete_mirror(self) -> None:
        incomplete = xr.Dataset(coords={"time": pd.date_range("2026-07-20", periods=2, freq="1h")})
        complete = xr.Dataset(
            {
                "SolarEnsembleWatts": (("member", "time"), np.ones((1, 2))),
                "ComponentLoadWatts": (("member", "component"), np.ones((1, 1))),
            },
            coords={
                "time": pd.date_range("2026-07-20", periods=2, freq="1h"),
                "member": [0],
                "component": ["DC"],
            },
        )
        with (
            patch.object(app, "_POWER_OPERATING_SCENARIOS_DS", incomplete),
            patch.object(app, "_power_operating_scenario_paths") as paths,
            patch.object(app.xr, "open_zarr", side_effect=(incomplete, complete)) as open_zarr,
        ):
            paths.return_value = (app.Path(__file__), app.Path(app.__file__))
            result = app._get_power_operating_scenarios_dataset()

        self.assertIs(result, complete)
        self.assertEqual(open_zarr.call_count, 2)
