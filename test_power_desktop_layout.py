from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from grouped_timeseries import (
    PLOTLY_SUMMARY_POWER_MAX_HEIGHT,
    PLOTLY_SUMMARY_POWER_PANEL_GAP,
    PLOTLY_SUMMARY_POWER_PANEL_HEIGHT,
    SUMMARY_DISPLAY_END_ATTR,
    SUMMARY_DISPLAY_START_ATTR,
    SUMMARY_LAYOUTS,
    PanelSpec,
    TraceSpec,
    _plotly_time_tick_options,
    build_summary_plotly,
    operating_mode_intervals,
)


def test_power_time_ticks_include_date_time_and_utc() -> None:
    options = _plotly_time_tick_options(pd.Timestamp("2026-07-19T00:00:00"), pd.Timestamp("2026-07-20T00:00:00"))

    assert options["tickformat"] == "%a %d %b<br>%H:%M UTC"


def _power_layout_dataset() -> xr.Dataset:
    times = pd.date_range("2026-07-15T00:00:00", periods=33, freq="3h")
    values = np.linspace(1.0, 2.0, len(times))
    return xr.Dataset(
        {
            "observed_1": (("time",), values * 100.0),
            "observed_2": (("time",), values),
            "forecast_24": (("time",), 70.0 - values),
            "forecast_input": (("time",), values * 200.0),
            "forecast_96": (("time",), 65.0 - values),
            "forecast_load": (("time",), 60.0 - values),
            "hindcast": (("time",), 68.0 - values),
            "forecast_skill": (("time",), values),
            "ensemble_skill": (("time",), values / 2.0),
            "power_skill": (("time",), values * 10.0),
        },
        coords={"time": times},
    )


def test_soc_hindcast_labels_explain_forecast_issue_time() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "soc_hindcast")

    assert panel.label == "Battery SOC: Measured vs Earlier Forecasts"
    assert panel.description is not None
    assert "before that time" in panel.description
    assert [trace.label for trace in panel.traces] == [
        "Measured battery SOC",
        "Forecast issued 6 h before valid time",
        "Forecast issued 24 h before valid time",
        "Forecast issued 48 h before valid time",
        "Forecast issued 72 h before valid time",
    ]


def _power_layout_panels() -> tuple[PanelSpec, ...]:
    panel_specs = (
        ("renewables", "Renewables", "observed_1"),
        ("battery_charging", "Battery Charging", "observed_2"),
        ("soc_24h_forecast", "SOC Next 24 h Forecast", "forecast_24"),
        ("ecmwf_solar_forecast", "ECMWF Solar & Load Forecast", "forecast_input"),
        ("soc_ecmwf_forecast", "SOC 96 h Forecast", "forecast_96"),
        ("operating_plan_scenarios", "Learned Operating-Mode SOC Plans", "forecast_load"),
        ("soc_hindcast", "SOC Hindcast: Forecasts vs Observed", "hindcast"),
        ("soc_forecast_skill", "SOC Forecast Verification", "forecast_skill"),
        ("soc_ensemble_skill", "SOC Ensemble Verification", "ensemble_skill"),
        ("forecast_power_skill", "Solar and Load Forecast Verification", "power_skill"),
    )
    return tuple(
        PanelSpec(key, label, "Value", None, (TraceSpec(variable, label, "#0b7285"),))
        for key, label, variable in panel_specs
    )


def test_power_desktop_panels_are_tall_and_grouped_by_time_axis() -> None:
    ds = _power_layout_dataset()
    times = pd.DatetimeIndex(ds["time"].values)

    with patch.dict(SUMMARY_LAYOUTS, {"power": _power_layout_panels()}):
        figure = build_summary_plotly(ds, "power", x_limits=(times[0], times[8]))

    expected_titles = [
        "Renewables",
        "Battery Charging",
        "SOC Next 24 h Forecast",
        "ECMWF Solar & Load Forecast",
        "SOC 96 h Forecast",
        "Learned Operating-Mode SOC Plans",
        "SOC Hindcast: Forecasts vs Observed",
        "SOC Forecast Verification",
        "SOC Ensemble Verification",
        "Solar and Load Forecast Verification",
    ]
    assert [annotation.text for annotation in figure.layout.annotations[: len(expected_titles)]] == expected_titles
    expected_height = min(
        PLOTLY_SUMMARY_POWER_MAX_HEIGHT,
        PLOTLY_SUMMARY_POWER_PANEL_HEIGHT * len(expected_titles)
        + PLOTLY_SUMMARY_POWER_PANEL_GAP * (len(expected_titles) - 1)
        + 90,
    )
    assert figure.layout.height == expected_height
    first_axis = figure.layout.yaxis.domain
    second_axis = figure.layout.yaxis2.domain
    assert (first_axis[0] - second_axis[1]) * figure.layout.height >= PLOTLY_SUMMARY_POWER_PANEL_GAP - 1

    xaxes = [
        getattr(figure.layout, "xaxis" if index == 1 else f"xaxis{index}")
        for index in range(1, len(expected_titles) + 1)
    ]
    assert all(axis.showticklabels for axis in xaxes)
    assert [axis.title.text for axis in xaxes] == [
        "Time (UTC)",
        "Time (UTC)",
        "Forecast Time (UTC)",
        "Forecast Time (UTC)",
        "Forecast Time (UTC)",
        "Forecast Time (UTC)",
        "Verification Time (UTC)",
        "Verification Time (UTC)",
        "Verification Time (UTC)",
        "Verification Time (UTC)",
    ]
    assert [axis.matches for axis in xaxes] == [None, "x", None, None, "x4", "x4", None, "x7", "x7", "x7"]


def test_non_power_summary_height_is_unchanged() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=5, freq="1h")
    ds = xr.Dataset({"h1_t": (("time",), np.linspace(0.0, 1.0, len(times)))}, coords={"time": times})

    figure = build_summary_plotly(ds, "vaisalamet")

    assert figure.layout.height < PLOTLY_SUMMARY_POWER_PANEL_HEIGHT * 4


def test_operating_mode_intervals_identify_each_planned_instrument() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=5, freq="1h")

    intervals = operating_mode_intervals(times, np.array([0, 1, 1, 2, 0]))

    assert [(label, start, end) for start, end, label, _color in intervals] == [
        ("CL61", times[1], times[3]),
        ("Radar", times[3], times[4]),
    ]


def test_right_axis_only_panel_retains_its_primary_subplot_anchor() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=5, freq="1h")
    ds = xr.Dataset({"cycles": (("time",), np.arange(len(times), dtype=float))}, coords={"time": times})
    panels = (
        PanelSpec(
            "soc_forecast_skill",
            "SOC Forecast Verification",
            "SOC MAE [percentage points]",
            "Independent ECMWF Cycles [count]",
            (TraceSpec("cycles", "Independent ECMWF Cycles", "#4f7d8d", axis="right"),),
        ),
    )

    with patch.dict(SUMMARY_LAYOUTS, {"power": panels}):
        figure = build_summary_plotly(ds, "power")

    assert len(figure.data) == 2
    right_trace, anchor_trace = figure.data
    assert right_trace.yaxis == "y2"
    assert anchor_trace.yaxis == "y"
    assert anchor_trace.showlegend is False
    assert anchor_trace.opacity == 0.0
    assert figure.layout.yaxis.domain == (0.0, 1.0)
    assert figure.layout.yaxis2.overlaying == "y"


def test_power_prewarm_observed_axes_use_measured_display_window() -> None:
    times = pd.date_range("2026-07-15T10:00:00", periods=41, freq="3h")
    observed_end_index = 8
    observed = np.full(len(times), np.nan)
    observed[: observed_end_index + 1] = np.linspace(50.0, 60.0, observed_end_index + 1)
    forecast = np.full(len(times), np.nan)
    forecast[observed_end_index:] = np.linspace(60.0, 75.0, len(times) - observed_end_index)
    ds = xr.Dataset(
        {
            "BatterySOC": (("time",), observed),
            "BatterySOCForecast": (("time",), forecast),
        },
        coords={"time": times},
        attrs={
            SUMMARY_DISPLAY_START_ATTR: times[0].isoformat(),
            SUMMARY_DISPLAY_END_ATTR: times[observed_end_index].isoformat(),
        },
    )
    panels = (
        PanelSpec("cumulative_power", "Observed SOC", "SOC [%]", None, (TraceSpec("BatterySOC", "Observed", "#468b61"),)),
        PanelSpec(
            "soc_ecmwf_forecast",
            "SOC 96 h Forecast",
            "SOC [%]",
            None,
            (TraceSpec("BatterySOCForecast", "Forecast", "#468b61"),),
        ),
    )

    with patch.dict(SUMMARY_LAYOUTS, {"power": panels}):
        figure = build_summary_plotly(ds, "power")

    assert list(figure.layout.xaxis.range) == [times[0], times[observed_end_index]]
    assert figure.layout.xaxis2.range[1] == times[-1]
