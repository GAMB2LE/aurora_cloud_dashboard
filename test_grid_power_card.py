"""Current grid input uses the existing collector, never a historical plot tip."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import collect_operations_snapshot as collector
import mobile_catalog


NOW = datetime(2026, 10, 26, 22, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("watts,value,level", [
    (1926., "1.93 kW", "green"), (750., "750.0 W", "green"),
    (0., "0.0 W", "unknown"), (-50., "-50.0 W", "unknown"),
    (-1500., "-1.50 kW", "unknown"),
])
def test_grid_card_keeps_signed_ac_measurement(watts, value, level):
    card = mobile_catalog._grid_power_card({
        "aps_grid_power_w": watts,
        "aps_grid_power_time_utc": (NOW - timedelta(minutes=7)).isoformat(),
    }, now=NOW)
    assert card == {
        "id": "grid-power", "title": "Grid Power", "value": value,
        "level": level, "updatedAt": (NOW - timedelta(minutes=7)).isoformat(),
        "detail": "Grid AC input · Sample 7m ago",
    }


@pytest.mark.parametrize("watts", [None, True, "1926", float("nan"), float("inf"), -9999])
def test_invalid_grid_measurement_is_unknown(watts):
    card = mobile_catalog._grid_power_card({
        "aps_grid_power_w": watts, "aps_grid_power_time_utc": NOW.isoformat(),
    }, now=NOW)
    assert card["value"] == "Unknown"
    assert card["level"] == "unknown"
    assert card["detail"] == "Grid telemetry unavailable"


@pytest.mark.parametrize("age,value", [(30., "1.93 kW"), (30.01, "Unknown"), (35, "Unknown"), (-1, "Unknown")])
def test_freshness_uses_grid_measurement_time(age, value):
    card = mobile_catalog._grid_power_card({
        "time_utc": NOW.isoformat(),
        "aps_multiplus_power_time_utc": NOW.isoformat(),
        "aps_grid_power_w": 1926.,
        "aps_grid_power_time_utc": (NOW - timedelta(minutes=age)).isoformat(),
    }, now=NOW)
    assert card["value"] == value
    if age > 30:
        assert card["detail"] == f"Last sample {int(age)}m ago"
    if age < 0:
        assert card["updatedAt"] is None


@pytest.mark.parametrize("stamp", [None, "", "bad-date", 123])
def test_missing_or_invalid_grid_time_does_not_borrow_other_timestamps(stamp):
    card = mobile_catalog._grid_power_card({
        "time_utc": NOW.isoformat(), "aps_multiplus_power_w": 1800.,
        "aps_multiplus_power_time_utc": NOW.isoformat(),
        "aps_grid_power_w": 1926., "aps_grid_power_time_utc": stamp,
    }, now=NOW)
    assert card["value"] == "Unknown"
    assert card["updatedAt"] is None


@pytest.mark.parametrize("last", [0., -50., 1926., np.nan, np.inf, -9999.])
def test_collector_reads_grid_and_charger_together_without_filling_gaps(tmp_path, monkeypatch, last):
    path = tmp_path / "power.zarr"
    xr.Dataset({
        "GridWatts": ("time", [2000., last]),
        "MultiPlusDCWatts": ("time", [1800., 1700.]),
    }, coords={"time": pd.date_range("2026-10-26T21:59:00", periods=2, freq="min")}).to_zarr(path)
    monkeypatch.setenv("POWER_ZARR_PATH", str(path))
    with patch.object(collector.xr, "open_zarr", wraps=xr.open_zarr) as opened:
        record = {}
        collector._collect_multiplus_metrics(record)
    assert opened.call_count == 1
    assert record["aps_grid_power_time_utc"] == NOW.isoformat()
    assert record["aps_grid_power_w"] == (float(last) if np.isfinite(last) and last != -9999 else None)
    assert record["aps_multiplus_power_w"] == 1700.


def test_grid_collection_is_independent_of_missing_charger_column(tmp_path, monkeypatch):
    path = tmp_path / "power.zarr"
    xr.Dataset({"GridWatts": ("time", [1926.])}, coords={"time": [NOW.replace(tzinfo=None)]}).to_zarr(path)
    monkeypatch.setenv("POWER_ZARR_PATH", str(path))
    record = {}
    collector._collect_multiplus_metrics(record)
    assert record["aps_grid_power_w"] == 1926.
    assert record["aps_multiplus_power_w"] is None


@pytest.mark.parametrize("failed,prefix,healthy,watts", [
    ("GridWatts", "aps_grid_power", "aps_multiplus_power", 1700.),
    ("MultiPlusDCWatts", "aps_multiplus_power", "aps_grid_power", 1926.),
])
def test_optional_chunk_read_failure_preserves_healthy_sibling(tmp_path, monkeypatch, failed, prefix, healthy, watts):
    dataset = xr.Dataset({
        "GridWatts": ("time", [1926.]), "MultiPlusDCWatts": ("time", [1700.]),
    }, coords={"time": [NOW.replace(tzinfo=None)]})
    monkeypatch.setenv("POWER_ZARR_PATH", str(tmp_path))
    original_values = xr.DataArray.values.fget

    def read_values(array):
        if array.name == failed:
            raise OSError("partial mirrored chunk")
        return original_values(array)

    with patch.object(collector.xr, "open_zarr", return_value=dataset), \
         patch.object(xr.DataArray, "values", new=property(read_values)):
        record = {}
        collector._collect_multiplus_metrics(record)
    assert record[f"{prefix}_w"] is None
    assert record[f"{healthy}_w"] == watts
    assert record[f"{healthy}_time_utc"] == NOW.isoformat()


@pytest.mark.parametrize("group", ["current", "observed", "all", "forecast", "forecast_24h", "forecast_96h", "verification"])
def test_power_current_signal_is_additive_and_uses_cached_snapshot(monkeypatch, group):
    monkeypatch.setattr(mobile_catalog, "_power_without_enclosure", lambda **kwargs: {"panels": [], "group": group})
    monkeypatch.setattr(mobile_catalog, "enclosure_power_panel", lambda *args, **kwargs: {"traces": []})
    snapshot = {"aps_grid_power_w": 1926., "aps_grid_power_time_utc": datetime.now(timezone.utc).isoformat()}
    with patch.object(mobile_catalog, "read_json_file", return_value=snapshot) as cached, \
         patch.object(collector.xr, "open_zarr", side_effect=AssertionError("No data-product read for current card")), \
         patch.object(collector, "_collect_multiplus_metrics", side_effect=AssertionError("No on-demand collection")):
        result = mobile_catalog.power(group=group)
    if group in {"current", "observed", "all"}:
        assert result["currentSignals"][0]["value"] == "1.93 kW"
        cached.assert_called_once_with(mobile_catalog.operations_snapshot_path())
    else:
        assert "currentSignals" not in result
        cached.assert_not_called()


def test_old_snapshot_has_unknown_grid_card():
    assert mobile_catalog._grid_power_card({}, now=NOW)["value"] == "Unknown"
