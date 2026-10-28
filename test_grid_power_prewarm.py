"""Split legacy cached Power figures without reading or changing observations."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

import app
from grouped_timeseries import SUMMARY_LAYOUTS, build_summary_plotly


@pytest.fixture
def legacy_figure():
    """Reproduce the pinned writer's layout, including optional gaps and signs."""
    panels = {panel.key: panel for panel in SUMMARY_LAYOUTS["power"]}
    old_panels = (
        panels["renewables"],
        replace(panels["battery_charging"], traces=panels["battery_charging"].traces + tuple(
            replace(trace, axis="right", dash="solid") for trace in panels["grid_power"].traces
        )),
        panels["output_power"],
        panels["thermal_state"],
    )
    times = pd.date_range("2026-10-28T00:00", periods=7, freq="5min")
    dataset = xr.Dataset({
        "SolarWatts_East": ("time", [0, 10, 0, 15, 0, 0, 0]),
        "SolarVolts_East": ("time", [50] * 7),
        "BatteryAmps": ("time", [-8, -7, -6, -5, -4, -3, -2]),
        "BatteryWatts": ("time", [-400, -350, -300, -250, -200, -150, -100]),
        "GridWatts": ("time", [np.nan, 1930, 0, np.nan, -20, 1840, 1800]),
        "MultiPlusDCWatts": ("time", [np.nan, 1600, 0, np.nan, -30, 1560, 1540]),
        "ACOutputWatts": ("time", [200] * 7),
        "DCInverterWatts": ("time", [220] * 7),
        "ChargerTemperature": ("time", [20] * 7),
    }, coords={"time": times})
    with patch.dict(SUMMARY_LAYOUTS, {"power": old_panels}):
        figure = build_summary_plotly(dataset, "power", panel_groups={"observed"})
    # Exercise the exact JSON representation used by the prewarm reader.
    return go.Figure(json.loads(figure.to_json()))


def _title(figure, text):
    return next(annotation for annotation in figure.layout.annotations if annotation.text == text)


def _axis_key(reference):
    return reference[0] + "axis" + reference[1:]


def test_legacy_split_preserves_samples_gaps_styles_axes_and_windows(legacy_figure):
    original = legacy_figure.to_json()
    result = app._separate_prewarmed_grid_power(legacy_figure)
    assert legacy_figure.to_json() == original
    assert len(result.data) == len(legacy_figure.data)
    grid = [trace for trace in result.data if trace.name in {"Grid AC input", "MultiPlus DC charging"}]
    assert len(grid) == 2
    assert grid[0].yaxis == grid[1].yaxis
    assert result.layout[_axis_key(grid[0].yaxis)].title.text == "Power [W]"
    assert result.layout[_axis_key(grid[0].yaxis)].autorange is True
    assert _title(result, "Renewables").y > _title(result, "Grid Power").y > _title(result, "Battery Charging").y
    for before, after in zip(legacy_figure.data, result.data):
        expected = before.to_plotly_json()
        if before.name in {"Grid AC input", "MultiPlus DC charging"}:
            expected.update(xaxis=after.xaxis, yaxis=after.yaxis, legend=after.legend)
            if before.name == "MultiPlus DC charging":
                expected["line"]["dash"] = "dash"
            old_time_axis = legacy_figure.layout[_axis_key(before.xaxis)]
            new_time_axis = result.layout[_axis_key(after.xaxis)]
            assert new_time_axis.range == old_time_axis.range
            assert new_time_axis.matches == old_time_axis.matches
        assert after.to_plotly_json() == expected
    before_layout = legacy_figure.layout.to_plotly_json()
    after_layout = result.layout.to_plotly_json()
    old_inner = before_layout["height"] - 130
    new_inner = after_layout["height"] - 130
    for key, old_axis in before_layout.items():
        if key.startswith("xaxis"):
            assert after_layout[key] == old_axis
        elif key.startswith("yaxis"):
            new_axis = dict(after_layout[key])
            if "domain" in old_axis:
                domain = new_axis.pop("domain")
                assert (domain[1] - domain[0]) * new_inner == pytest.approx(
                    (old_axis["domain"][1] - old_axis["domain"][0]) * old_inner, abs=1
                )
                old_axis = {k: v for k, v in old_axis.items() if k != "domain"}
            assert new_axis == old_axis


def test_grid_watts_axis_does_not_inherit_battery_current_ticks(legacy_figure):
    legacy_figure.layout.yaxis3.update(tickmode="array", tickvals=[-1, 0, 1],
                                       ticktext=["-1 A", "0 A", "1 A"], dtick=1, tick0=0,
                                       matches="y", scaleanchor="y")
    result = app._separate_prewarmed_grid_power(legacy_figure)
    trace = next(trace for trace in result.data if trace.name == "Grid AC input")
    axis = result.layout[_axis_key(trace.yaxis)]
    for option in ("range", "tickvals", "ticktext", "tickmode", "tick0", "dtick",
                   "matches", "overlaying", "scaleanchor", "scaleratio"):
        assert axis[option] is None
    assert axis.autorange is True
    assert result.layout.yaxis3.ticktext == legacy_figure.layout.yaxis3.ticktext


@pytest.mark.parametrize("missing", ["Grid AC input", "MultiPlus DC charging"])
def test_one_optional_grid_trace_still_gets_its_own_panel(legacy_figure, missing):
    legacy_figure.data = [trace for trace in legacy_figure.data if trace.name != missing]
    result = app._separate_prewarmed_grid_power(legacy_figure)
    assert _title(result, "Grid Power")
    assert len(result.data) == len(legacy_figure.data)


def test_split_is_idempotent_after_json_round_trip(legacy_figure):
    adapted = app._separate_prewarmed_grid_power(legacy_figure)
    reloaded = go.Figure(json.loads(adapted.to_json()))
    assert app._separate_prewarmed_grid_power(reloaded).to_json() == adapted.to_json()


@pytest.mark.parametrize("unsupported", ["absent", "duplicate", "missing-title", "missing-axis-label", "different-axis"])
def test_absent_or_unsupported_figures_remain_unchanged(legacy_figure, unsupported):
    if unsupported == "absent":
        legacy_figure.data = [trace for trace in legacy_figure.data
                              if trace.name not in {"Grid AC input", "MultiPlus DC charging"}]
    elif unsupported == "duplicate":
        legacy_figure.add_trace(next(trace for trace in legacy_figure.data if trace.name == "Grid AC input"))
    elif unsupported == "missing-title":
        legacy_figure.layout.annotations = [item for item in legacy_figure.layout.annotations if item.text != "Renewables"]
    elif unsupported == "missing-axis-label":
        legacy_figure.layout.annotations = [item for item in legacy_figure.layout.annotations if item.text != "Current In/Out [A]"]
    else:
        next(trace for trace in legacy_figure.data if trace.name == "Grid AC input").xaxis = "x"
    original = legacy_figure.to_json()
    assert app._separate_prewarmed_grid_power(legacy_figure).to_json() == original


@pytest.mark.parametrize("instrument,section,split", [
    ("power", "current", True), ("power", "forecast", False), ("vaisalamet", "current", False),
])
def test_sync_reader_adapts_only_current_power_without_raw_reads(tmp_path, legacy_figure, instrument, section, split):
    path = tmp_path / "prewarm.json"
    legacy_figure.write_json(path)
    with (
        patch.object(app, "_cache_key_targets_latest_prewarm", return_value=True),
        patch.object(app, "_prewarmed_interactive_path", return_value=path),
        patch.object(app, "_dataset_time_bounds", side_effect=AssertionError("must not read raw data")),
        patch.object(app, "_open_power_display_summary_window", side_effect=AssertionError("must not read stores")),
    ):
        result = app._load_prewarmed_interactive_figure((instrument, section), instrument)
    assert any(item.text == "Grid Power" for item in result.layout.annotations) is split
    assert go.Figure(json.loads(path.read_text())).to_json() == legacy_figure.to_json()


@pytest.mark.parametrize("active", [True, False])
def test_async_reader_keeps_document_publication_and_stale_request_guard(tmp_path, legacy_figure, active):
    path = tmp_path / "prewarm.json"
    legacy_figure.write_json(path)
    callbacks = []
    doc = SimpleNamespace(add_next_tick_callback=callbacks.append)
    cache_key = ("power", "current")
    cache = OrderedDict()
    in_flight = {cache_key}
    shown = Mock()
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        patch.object(app, "_BACKGROUND_PREPARATION_EXECUTOR", executor),
        patch.object(app, "_INTERACTIVE_RENDER_CACHE", cache),
        patch.object(app, "_IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS", in_flight),
        patch.object(app, "_render_request_active", return_value=active),
        patch.object(app, "_show_plot", shown),
        patch.object(app, "_clear_interactive_loading"),
        patch.object(app, "_schedule_timeout"),
        patch.object(app, "_render_interactive_view", side_effect=AssertionError("must not rerender from source")),
    ):
        asyncio.run(app._load_prewarmed_interactive_figure_async(path, cache_key, "power", 42, doc, ()))
        assert len(callbacks) == 1
        shown.assert_not_called()
        callbacks[0]()
    assert cache_key not in in_flight
    assert (cache_key in cache) is active
    if active:
        shown.assert_called_once()
        assert _title(shown.call_args.args[0], "Grid Power")
    else:
        shown.assert_not_called()
