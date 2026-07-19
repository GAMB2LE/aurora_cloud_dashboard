from __future__ import annotations

from datetime import datetime
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import app


class DashboardShellTests(TestCase):
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
