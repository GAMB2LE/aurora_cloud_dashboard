"""The web grid card shares cached API truth and stays within Current Power."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import panel as pn
import pytest
import xarray as xr

import app
import mobile_catalog
from dashboard_components import OverviewSnapshotHTML


NOW = datetime(2026, 10, 26, 22, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("watts,minutes", [
    (1930, 7), (0, 7), (-20, 7), (1930, 30), (1930, 31),
    (None, 7), (1930, -1),
])
def test_grid_markup_matches_cached_api_card(watts, minutes):
    snapshot = {
        "aps_grid_power_w": watts,
        "aps_grid_power_time_utc": (NOW - timedelta(minutes=minutes)).isoformat(),
    }
    with (
        patch.object(mobile_catalog, "datetime", wraps=datetime) as clock,
        patch.object(mobile_catalog, "read_json_file", return_value=snapshot) as read,
        patch.object(app, "_mobile_status_card_markup", wraps=app._mobile_status_card_markup) as render,
    ):
        clock.now.return_value = NOW
        markup = app._power_grid_markup()
    expected = mobile_catalog._grid_power_card(snapshot, now=NOW)
    read.assert_called_once_with(mobile_catalog.operations_snapshot_path())
    assert render.call_args.args == (expected["title"], expected["value"], expected["detail"], expected["level"])
    assert render.call_args.kwargs == {"grid_sample": expected["updatedAt"] or ""}
    assert "data-grid-sample=" in markup


def test_grid_markup_escapes_sample_and_text():
    markup = app._mobile_status_card_markup("Grid <Power>", "Unknown", grid_sample="'<unsafe>")
    assert "data-grid-sample='&#x27;&lt;unsafe&gt;'" in markup
    assert "Grid &lt;Power&gt;" in markup


def test_desktop_grid_pane_stays_before_current_charts_and_refreshes_in_place():
    pane = app.browser_power_grid_pane
    original = pane.object
    objects = app.interactive_tab.objects
    assert objects.index(app.power_section_intro_container) < objects.index(pane) < objects.index(app.interactive_content)
    try:
        with (
            patch.object(app, "instrument_select", SimpleNamespace(value="power")),
            patch.object(app, "power_view_select", SimpleNamespace(value="current")),
            patch.object(app, "_power_grid_markup", return_value="<div>Grid fixture</div>") as read,
        ):
            app._refresh_browser_power_grid()
            app._refresh_browser_power_grid()
        assert app.browser_power_grid_pane is pane
        assert pane.object == "<div>Grid fixture</div>"
        assert read.call_count == 2
    finally:
        pane.object = original


@pytest.mark.parametrize("instrument,section,visible", [
    ("power", "current", True), ("power", "forecast", False), ("Ceilometer", "current", False),
])
def test_grid_is_visible_and_read_only_for_current_power(instrument, section, visible):
    pane = OverviewSnapshotHTML(object="")
    with (
        patch.object(app, "instrument_select", SimpleNamespace(value=instrument)),
        patch.object(app, "power_view_select", SimpleNamespace(value=section)),
        patch.object(app, "browser_power_grid_pane", pane),
        patch.object(app, "power_view_select_container", pn.Column()),
        patch.object(app, "power_section_intro_container", pn.Column()),
        patch.object(app, "interactive_content", pn.Column()),
        patch.object(app, "enclosure_power_container", pn.Column()),
        patch.object(app, "power_browser_guidance_container", pn.Column()),
        patch.object(app, "power_plan_editor_container", pn.Column()),
        patch.object(app, "_browser_power_briefing", return_value=pn.Spacer()),
        patch.object(app, "_get_power_plan_editor", return_value=pn.Spacer()),
        patch.object(app, "_power_grid_markup", return_value="<div>Grid fixture</div>") as read,
    ):
        app._sync_power_section_visibility()
        app._refresh_browser_power_grid()
    assert pane.visible is visible
    assert read.call_count == int(visible)


@pytest.mark.parametrize("has_chart", [False, True])
def test_mobile_grid_precedes_current_charts_even_when_charts_unavailable(has_chart):
    dataset = xr.Dataset(coords={"time": [NOW.replace(tzinfo=None)]}) if has_chart else None
    with (
        patch.object(app, "_power_grid_markup", return_value="<div>Grid fixture</div>"),
        patch.object(app, "_mobile_power_window", return_value=dataset),
        patch.object(app, "_mobile_power_card", return_value=pn.pane.HTML("<div>Chart fixture</div>")),
    ):
        current = app._mobile_power_section("current", None)
    assert isinstance(current.objects[0], OverviewSnapshotHTML)
    assert current.objects[0].object == "<div>Grid fixture</div>"
    assert len(current.objects) > 1


def test_mobile_forecast_does_not_read_or_render_grid():
    with (
        patch.object(app, "_power_grid_markup", side_effect=AssertionError("Forecast must not read grid card")),
        patch.object(app, "_mobile_power_window", return_value=None),
    ):
        forecast = app._mobile_power_section("forecast", None)
    assert not any(isinstance(item, OverviewSnapshotHTML) for item in forecast.objects)
