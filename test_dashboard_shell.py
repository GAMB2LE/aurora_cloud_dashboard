from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import app


def test_desktop_shell_has_full_named_tabs() -> None:
    labels = [label for label, _slug, _panel in app.DESKTOP_TAB_SPECS]

    assert labels == [
        "Overview",
        "Interactive Data Browser",
        "Power",
        "Science Quicklooks",
        "House Keeping Quicklooks",
        "AURORACam",
        "UAS",
        "Operations Dashboard",
    ]
    assert len(app.desktop_tabs) == len(labels)
    assert app.desktop_tabs.dynamic
    assert "Overview" in labels


def test_desktop_tab_labels_scroll_without_abbreviating() -> None:
    assert ":host(.desktop-tabs) .bk-header" in app.css
    assert "overflow-x: auto" in app.css
    assert ":host(.desktop-tabs) .bk-tab" in app.css
    assert "white-space: nowrap" in app.css


def test_desktop_controls_keep_compact_navigation_rows() -> None:
    controls_body = app.controls.objects[0]
    first_row_names = [widget.name for widget in controls_body.objects[0].objects]
    second_row_names = [widget.name for widget in controls_body.objects[1].objects]

    assert first_row_names == ["Instrument", "Start (UTC)", "End (UTC)", "Live Off"]
    assert second_row_names == ["Previous Day", "Reset View Defaults", "Next Day/Current Day"]


def test_phone_shell_keeps_operational_groups() -> None:
    assert list(app.MOBILE_TAB_OPTIONS) == ["Overview", "Power", "Plots", "Camera", "Ops"]


def test_browser_overview_uses_shared_instrument_state_groups(monkeypatch) -> None:
    monkeypatch.setattr(
        app.mobile_catalog,
        "overview",
        lambda: {
            "instrumentPower": [
                {"title": "UAS", "state": "On", "level": "green", "detail": "PDU sample 2 min old"},
                {"title": "CL61", "state": "Off", "level": "unknown", "detail": "PDU sample 2 min old"},
                {"title": "Cloud Radar", "state": "On", "level": "green", "detail": "PDU sample 2 min old"},
                {"title": "HATPRO", "state": "Off", "level": "unknown", "detail": "PDU sample 2 min old"},
                {"title": "Meteorology", "state": "Collecting", "level": "green", "detail": "Latest sample 1 min old"},
            ]
        },
    )

    markup = app._browser_overview_instrument_markup()

    assert "PDU-controlled instruments" in markup
    assert "Collection-only instruments" in markup
    assert "Cloud Radar" in markup
    assert "Meteorology" in markup


def test_live_query_uses_current_window_instead_of_stale_url_dates(monkeypatch) -> None:
    current_start = datetime(2026, 7, 15, 10, 30)
    current_end = datetime(2026, 7, 16, 10, 30)
    monkeypatch.setattr(app, "_last_24h_utc_window", lambda: (current_start, current_end))

    state = app._query_interactive_time_state(
        {
            "start": "2026-07-15T07:04:01",
            "end": "2026-07-16T07:04:01",
            "live": "1",
        },
        "power",
    )

    assert state == (current_start, current_end, True)


def test_non_live_query_preserves_historical_window() -> None:
    state = app._query_interactive_time_state(
        {
            "start": "2026-07-15T07:04:01",
            "end": "2026-07-16T07:04:01",
            "live": "0",
        },
        "power",
    )

    assert state == (
        datetime(2026, 7, 15, 7, 4, 1),
        datetime(2026, 7, 16, 7, 4, 1),
        False,
    )


def test_power_time_bounds_ignore_forecast_only_rows() -> None:
    times = pd.date_range("2026-07-16T08:00:00", periods=7, freq="1h")
    measured = np.array([50.0, 51.0, 52.0, np.nan, np.nan, np.nan, np.nan])
    forecast = np.array([np.nan, np.nan, 52.0, 53.0, 54.0, 55.0, 56.0])
    ds = xr.Dataset(
        {
            "BatterySOC": (("time",), measured),
            "BatterySOCForecast": (("time",), forecast),
        },
        coords={"time": times},
    )

    lower, upper, raw_count, valid_count = app._time_bounds_from_power_display_dataset(ds)

    assert lower == datetime(2026, 7, 16, 8, 0)
    assert upper == datetime(2026, 7, 16, 10, 0)
    assert raw_count == 7
    assert valid_count == 3
