"""Grid history shares the existing observed chart transport on every client."""

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import grouped_timeseries as charts
import mobile_catalog


NOW = datetime(2026, 10, 28, 22, tzinfo=timezone.utc)


@pytest.fixture
def stored_power(tmp_path, monkeypatch):
    times = pd.date_range("2026-10-28T12:00:00", periods=601, freq="min")
    ac = np.full(len(times), 1770.0)
    dc = np.full(len(times), 1590.0)
    ac[100:160] = dc[100:160] = np.nan
    ac[200:210] = dc[200:210] = 0.0
    ac[220] = -50.0
    dc[220] = -20.0
    ac[-1] = dc[-1] = np.nan
    dataset = xr.Dataset({
        "SolarWatts_East": ("time", np.zeros(len(times))),
        "GridWatts": ("time", ac),
        "MultiPlusDCWatts": ("time", dc),
        "BatteryWatts": ("time", np.full(len(times), 1200.0)),
    }, coords={"time": times})
    for variable in dataset.data_vars:
        dataset[variable].attrs["units"] = "W"
    path = tmp_path / "current.zarr"
    dataset.to_zarr(path, mode="w", consolidated=True)
    monkeypatch.setattr(mobile_catalog, "power_display_section_path", lambda section: path)
    monkeypatch.setattr(mobile_catalog, "power_forecast_bundle_status", lambda path: {"enabled": False})
    monkeypatch.setattr(mobile_catalog, "power_forecast_only_publication_active", lambda: False)
    monkeypatch.setattr(mobile_catalog, "power_forecast_serving_selection", lambda: None)
    monkeypatch.setattr(mobile_catalog, "power_operating_scenario_paths", lambda: ())
    monkeypatch.setattr(mobile_catalog, "enclosure_power_panel", lambda *args, **kwargs: {"traces": []})
    monkeypatch.setattr(mobile_catalog, "read_json_file", lambda path: {})
    with patch.object(mobile_catalog, "datetime", wraps=datetime) as clock:
        clock.now.return_value = NOW
        yield dataset, path


@pytest.mark.parametrize("group", ["current", "observed", "all"])
def test_native_grid_history_is_separate_bounded_and_keeps_measurements(stored_power, group):
    with patch.object(xr, "open_zarr", wraps=xr.open_zarr) as opened:
        payload = mobile_catalog.power(group=group)
    assert "warning" not in payload
    assert opened.call_count == 1
    assert [panel["id"] for panel in payload["panels"]][:3] == [
        "renewables", "grid_power", "battery_charging",
    ]
    panel = payload["panels"][1]
    assert panel["title"] == "Grid Power"
    assert panel["leftAxisLabel"] == "Power [W]"
    assert panel["rightAxisLabel"] is None
    assert [trace["id"] for trace in panel["traces"]] == ["GridWatts", "MultiPlusDCWatts"]
    assert panel["traces"][1]["dash"] == "dash"
    for trace, negative in zip(panel["traces"], [-50.0, -20.0], strict=True):
        assert trace["axis"] == "left"
        assert trace["unit"] == "W"
        points = trace["points"]
        assert len(points) <= mobile_catalog.MOBILE_POWER_MAX_POINTS
        assert negative in [point["value"] for point in points]
        assert 0.0 in [point["value"] for point in points]
        assert len({point["segment"] for point in points}) == 2
        assert points[-1]["time"] == "2026-10-28T21:59:00Z"
    battery = payload["panels"][2]
    assert [trace["id"] for trace in battery["traces"]] == ["BatteryWatts"]
    # An old finite history point never supplies the current-status card.
    assert payload["currentSignals"][0]["value"] == "Unknown"


@pytest.mark.parametrize("group", ["forecast", "forecast_24h", "forecast_96h", "verification"])
def test_grid_history_stays_out_of_forecast_groups(stored_power, group):
    payload = mobile_catalog.power(group=group)
    assert "warning" not in payload
    assert all(panel["id"] != "grid_power" for panel in payload["panels"])


@pytest.mark.parametrize("missing", [(), ("GridWatts",), ("MultiPlusDCWatts",), ("GridWatts", "MultiPlusDCWatts")])
def test_grid_history_uses_only_available_optional_fields(stored_power, missing):
    dataset, path = stored_power
    subset = dataset.drop_vars(list(missing))
    subset.to_zarr(path, mode="w", consolidated=True)
    payload = mobile_catalog.power(group="current")
    assert "warning" not in payload
    panels = [panel for panel in payload["panels"] if panel["id"] == "grid_power"]
    available = [name for name in ("GridWatts", "MultiPlusDCWatts") if name not in missing]
    if available:
        assert [trace["id"] for trace in panels[0]["traces"]] == available
    else:
        assert panels == []


def test_desktop_grid_history_shares_one_axis_and_preserves_gaps(stored_power):
    dataset, _ = stored_power
    figure = charts.build_summary_plotly(dataset, "power", panel_groups={"observed"})
    titles = [annotation.text for annotation in figure.layout.annotations]
    assert titles[:3] == ["Renewables", "Grid Power", "Battery Charging"]
    grid = next(trace for trace in figure.data if trace.name == "Grid AC input")
    dc = next(trace for trace in figure.data if trace.name == "MultiPlus DC charging")
    battery = next(trace for trace in figure.data if trace.name == "Power In/Out")
    assert grid.yaxis == dc.yaxis != battery.yaxis
    assert dc.line.dash == "dash"
    assert -50.0 in grid.y
    assert 0.0 in grid.y
    assert np.isnan(np.asarray(grid.y, dtype=float)).any()
    assert not grid.connectgaps
