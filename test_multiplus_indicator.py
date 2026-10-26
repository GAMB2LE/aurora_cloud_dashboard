"""The charger indication is derived entirely from stored cloud telemetry."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import collect_operations_snapshot as collector
import mobile_catalog


NOW = datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("value,expected", [(1780., "Charging"), (0., "Not charging"), (-20., "Not charging")])
def test_charging_means_measured_output(value, expected):
    snapshot = {"aps_multiplus_power_w": value, "aps_multiplus_power_time_utc": (NOW - timedelta(minutes=8)).isoformat()}
    card = mobile_catalog._multiplus_charger_card(snapshot, now=NOW)
    assert card["value"] == expected
    assert card["level"] == ("green" if value > 0 else "unknown")
    assert card["detail"].endswith(" · Sample 8m ago")
    if value == 1780:
        assert card["detail"] == "1.78 kW · Sample 8m ago"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -9999, True, "1780"])
def test_unavailable_is_never_off(value):
    card = mobile_catalog._multiplus_charger_card({"aps_multiplus_power_w": value, "aps_multiplus_power_time_utc": NOW.isoformat()}, now=NOW)
    assert card["value"] == "Unknown"
    assert card["level"] == "unknown"


@pytest.mark.parametrize("minutes,expected", [(30, "Charging"), (30.01, "Unknown"), (35, "Unknown"), (-1, "Unknown")])
def test_age_uses_measurement_not_snapshot_time(minutes, expected):
    card = mobile_catalog._multiplus_charger_card({
        "time_utc": NOW.isoformat(), "aps_multiplus_power_w": 1780,
        "aps_multiplus_power_time_utc": (NOW - timedelta(minutes=minutes)).isoformat(),
    }, now=NOW)
    assert card["value"] == expected
    if minutes == 35:
        assert card["detail"] == "Last sample 35m ago"


def test_old_snapshot_has_no_charger_inference():
    card = mobile_catalog._multiplus_charger_card({"BatteryWatts": 2000, "GridWatts": 2500, "time_utc": NOW.isoformat()}, now=NOW)
    assert card["value"] == "Unknown"
    assert card["updatedAt"] is None


@pytest.mark.parametrize("last", [0., -20., 1780., np.nan, np.inf, -9999.])
def test_collector_reads_last_row_including_missing(tmp_path, monkeypatch, last):
    path = tmp_path / "power.zarr"
    xr.Dataset({"MultiPlusDCWatts": ("time", [2500., last])}, coords={"time": pd.date_range("2026-10-26T19:59:00", periods=2, freq="min")}).to_zarr(path)
    monkeypatch.setenv("POWER_ZARR_PATH", str(path))
    record = {}
    collector._collect_multiplus_metrics(record)
    assert record["aps_multiplus_power_time_utc"] == NOW.isoformat()
    assert record["aps_multiplus_power_w"] == (float(last) if np.isfinite(last) and last != -9999 else None)


def test_optional_read_failure_preserves_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("POWER_ZARR_PATH", str(tmp_path))
    with patch.object(collector.xr, "open_zarr", side_effect=OSError("partial mirror")):
        record = {"other_metric": 42}
        collector._collect_multiplus_metrics(record)
    assert record["other_metric"] == 42
    assert record["aps_multiplus_power_w"] is None


def test_repeated_card_reads_do_not_collect_or_open_data():
    with patch.object(collector, "_collect_multiplus_metrics", side_effect=AssertionError("collection requested")), patch.object(xr, "open_zarr", side_effect=AssertionError("data opened")):
        for _ in range(5):
            assert mobile_catalog._multiplus_charger_card({}, now=NOW)["value"] == "Unknown"
