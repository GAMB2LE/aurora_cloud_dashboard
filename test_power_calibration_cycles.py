"""Regression cases for sunset calibration and independent ENS evidence."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from generate_power_soc_forecast import (
    LEAD_BUCKETS,
    _deterministic_source_cycle_status,
    apply_forecast_identity,
    _expected_cl61_phase_profile,
    SOLAR_CALIBRATION_METHOD,
    build_forecast_dataset,
    calibrate_solar_factor,
    embedded_source_meteorology,
    solar_calibration_contract_id,
    solar_calibration_intervals,
)
from generate_power_soc_ensemble import _set_ensemble_cycle_eligibility, build_ensemble_dataset
from power_battery_model import BatteryModel
from power_load_dynamics import ControlledLoadProfile


def test_same_source_reanchor_cannot_be_a_second_independent_issue():
    issue = pd.Timestamp("2026-09-11T12:00")
    previous = xr.Dataset(attrs={
        "forecast_model_version": "13", "candidate_lane": "",
        "source_cycle_set_id": "source-a", "initial_soc_time": issue.isoformat(),
        "independent_cycle": "true",
    })
    kwargs = dict(model_version="13", candidate_lane="", archive=None,
                  previous_forecast=previous, state={})
    assert _deterministic_source_cycle_status("source-a", issue_time=issue, **kwargs) == (True, True)
    assert _deterministic_source_cycle_status("source-a", issue_time=issue + pd.Timedelta(hours=1), **kwargs) == (True, False)
    assert _deterministic_source_cycle_status("source-b", issue_time=issue, **kwargs) == (False, False)
    kwargs.update(previous_forecast=None, state={"calibration_source_cycle_key": "13::source-a"})
    assert _deterministic_source_cycle_status("source-a", issue_time=issue, **kwargs) == (True, False)


def test_implementation_digest_preserves_campaign_across_unrelated_commits():
    def identified(revision, digest):
        return apply_forecast_identity(xr.Dataset(attrs={"forecast_model_contract_id": "base"}), {
            "forecast_code_revision": revision, "forecast_implementation_digest": digest,
            "forecast_system_version": "corrected", "feature_set_version": "features-v1",
        })
    first = identified("commit-a", "sha256:" + "a" * 64)
    other_commit = identified("commit-b", "sha256:" + "a" * 64)
    changed_code = identified("commit-b", "sha256:" + "b" * 64)
    assert first.attrs["forecast_model_contract_id"] == other_commit.attrs["forecast_model_contract_id"]
    assert first.attrs["forecast_identity_id"] != other_commit.attrs["forecast_identity_id"]
    assert first.attrs["forecast_model_contract_id"] != changed_code.attrs["forecast_model_contract_id"]


def _power(times, solar_w=1000.0):
    return xr.Dataset(
        {
            "BatterySOC": ("time", np.full(len(times), 70.0)),
            "SolarWatts_East": ("time", np.full(len(times), solar_w / 3)),
            "SolarWatts_South": ("time", np.full(len(times), solar_w / 3)),
            "SolarWatts_West": ("time", np.full(len(times), solar_w / 3)),
            "ACOutputWatts": ("time", np.full(len(times), 100.0)),
            "DCInverterWatts": ("time", np.full(len(times), 20.0)),
        }, coords={"time": times},
    )


def _solar(times, members=False):
    values = np.arange(len(times)) * 3 * 3600 * 200.0
    if members:
        return xr.Dataset({"ssrd": (("number", "time"), np.stack([values, values * 1.1]))}, coords={"number": [1, 2], "time": times})
    return xr.Dataset({"ssrd": ("time", values)}, coords={"time": times})


def test_5819_sunset_records_cannot_calibrate_against_one_future_interval():
    end = pd.Timestamp("2026-09-11T20:44:29")
    times = pd.date_range("2026-09-11T19:00:00", end, periods=5819)
    frame = _power(times, solar_w=3.3).to_dataframe()
    irradiance = pd.Series([57.98, 0.0, 0.0], index=pd.date_range("2026-09-11T21:00", periods=3, freq="3h"))
    assert solar_calibration_intervals(frame, irradiance, end=end).empty
    assert calibrate_solar_factor(frame, irradiance, end=end, fallback_factor=8.0) == 8.0


def test_fit_counts_complete_intervals_and_rejects_mppt_censoring():
    times = pd.date_range("2026-09-01T00:00", periods=97, freq="5min")
    frame = _power(times).to_dataframe()
    irradiance = pd.Series(200.0, index=pd.date_range(times[0], periods=9, freq="1h"))
    paired = solar_calibration_intervals(frame, irradiance, end=times[-1])
    assert len(paired) == 8
    assert calibrate_solar_factor(frame, irradiance, end=times[-1]) == 5.0
    frame["SolarMPPMode_East"] = 2
    frame["SolarMPPMode_South"] = 2
    frame["SolarMPPMode_West"] = 3
    assert solar_calibration_intervals(frame, irradiance, end=times[-1]).empty
    assert calibrate_solar_factor(frame, irradiance, end=times[-1], fallback_factor=7.0) == 7.0


def test_incomplete_interval_is_not_rescued_by_dense_samples():
    times = pd.date_range("2026-09-01T00:00", periods=7200, freq="1s")
    frame = _power(times).to_dataframe()
    irradiance = pd.Series(200.0, index=pd.date_range(times[0], periods=3, freq="1h"))
    assert len(solar_calibration_intervals(frame, irradiance, end=times[-1])) == 1
    assert calibrate_solar_factor(frame, irradiance, end=times[-1], fallback_factor=7.0) == 7.0


def test_cached_reanchor_preserves_solar_battery_and_soc_calibration():
    times = pd.date_range("2026-09-01T00:00", periods=577, freq="5min")
    solar = _solar(pd.date_range(times[0], periods=21, freq="3h"))
    calibration = {
        "method": SOLAR_CALIBRATION_METHOD, "base_factor": 4.0, "raw_factor": 4.0,
        "lead_mos": {bucket: 1.0 for bucket, _, _ in LEAD_BUCKETS},
        "interval_count": 8, "interval_evidence_id": "old-evidence",
        "training_cutoff_utc": "2026-09-01T12:00:00",
    }
    state = {
        "solar_calibration_state": calibration,
        "soc_bias_correction_pct_points_by_bucket": {"0_6h": 2.0},
        "battery_model": BatteryModel(usable_capacity_kwh=24.5).attrs(),
    }
    with patch("generate_power_soc_forecast.fit_battery_model", side_effect=AssertionError("cache must not refit")):
        result = build_forecast_dataset(_power(times), solar, state=state, horizon_hours=12, allow_calibration_update=False)
    assert json.loads(result.attrs["solar_calibration_state"]) == calibration
    assert json.loads(result.attrs["soc_bias_correction_pct_points_by_bucket"]) == {"0_6h": 2.0}
    assert float(result.attrs["battery_capacity_kwh"]) == 24.5
    assert result.attrs["solar_training_cutoff_utc"] == calibration["training_cutoff_utc"]
    assert result.attrs["forecast_model_version"] == "13"


def test_old_sunset_factor_is_not_seeded_into_corrected_model():
    times = pd.date_range("2026-09-01T00:00", periods=13, freq="1h")
    result = build_forecast_dataset(
        _power(times), _solar(pd.date_range(times[-1], periods=5, freq="3h")),
        state={"solar_calibration_factor_w_per_wm2": 0.0001, "soc_bias_correction_pct_points_by_bucket": {"0_6h": 8.0}},
        horizon_hours=12, allow_calibration_update=False,
    )
    assert result.attrs["solar_calibration_status"] == "default_uncalibrated"
    assert "solar_calibration_unavailable" in result.attrs["solar_degradation_codes"]
    assert json.loads(result.attrs["soc_bias_correction_pct_points_by_bucket"]) == {}


def test_new_ensemble_cycle_is_eligible_despite_cached_deterministic_input():
    forecast = xr.Dataset(attrs={
        "ensemble_independent_cycle_id": "ens-source-one", "forecast_model_contract_id": "corrected-contract",
        "forecast_refresh_kind": "cached_reanchor", "independent_cycle": "false", "forecast_verification_eligible": "false",
    })
    _set_ensemble_cycle_eligibility(forecast, None)
    assert forecast.attrs["independent_cycle"] == "true"
    archive = xr.Dataset({
        "EnsembleIndependentCycleID": ("issue_time", ["ens-source-one"]),
        "ForecastModelContractID": ("issue_time", ["corrected-contract"]),
    })
    _set_ensemble_cycle_eligibility(forecast, archive)
    assert forecast.attrs["independent_cycle"] == "false"
    forecast.attrs["ensemble_independent_cycle_id"] = "ens-source-two"
    _set_ensemble_cycle_eligibility(forecast, archive)
    assert forecast.attrs["forecast_verification_eligible"] == "true"


def test_ensemble_uses_exact_deterministic_anchor_and_verified_calibration():
    times = pd.date_range("2026-09-01T00:00", periods=7, freq="1h")
    power = _power(times)
    power["BatterySOC"] = ("time", [70, 69, 68, 67, 66, 65, 64])
    mos = {bucket: 1.0 for bucket, _, _ in LEAD_BUCKETS}
    calibration = {"base_factor": 4.0, "lead_mos": mos}
    deterministic = xr.Dataset(attrs={
        "initial_soc_time": times[3].isoformat(), "initial_soc_pct": "67",
        "forecast_load_w": "120", "solar_calibration_factor_w_per_wm2": "4",
        "solar_calibration_state": json.dumps(calibration),
        "solar_calibration_contract_id": solar_calibration_contract_id(4.0, mos),
    })
    ensemble = build_ensemble_dataset(power, deterministic, _solar(pd.date_range(times[3], periods=5, freq="3h"), members=True), horizon_hours=12)
    assert pd.Timestamp(ensemble.time.values[0]) == times[3]
    assert np.all(ensemble.BatterySOCForecastEnsemble.values[:, 0] == 67)
    assert ensemble.attrs["solar_calibration_state"] == deterministic.attrs["solar_calibration_state"]
    deterministic.attrs["solar_calibration_contract_id"] = "corrupted-calibration"
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_ensemble_dataset(power, deterministic, _solar(pd.date_range(times[3], periods=5, freq="3h"), members=True), horizon_hours=12)


def test_cl61_expected_phase_energy_is_bounded_and_grid_invariant():
    dynamics = SimpleNamespace(
        state="dc_cl61", current_phase="fan_high", change_count=16,
        phase_weights={"fan_low": 0.55, "fan_high": 0.45},
        phase_dwell_minutes={"fan_low": 742.5, "fan_high": 607.5},
        phase_profiles={"fan_low": SimpleNamespace(p50_w=292.0), "fan_high": SimpleNamespace(p50_w=480.0)},
    )
    energy = []
    for frequency, count in [("1h", 97), ("3h", 33)]:
        times = pd.date_range("2026-09-11", periods=count, freq=frequency)
        profile = ControlledLoadProfile(np.full(count, 280.0), np.full(count, 480.0), np.full(count, 510.0), np.full(count, 3), "current_high")
        result = _expected_cl61_phase_profile(dynamics, times, profile)
        assert result.p50_w[0] == 480.0
        assert np.all((result.p50_w >= 292) & (result.p50_w <= 480))
        assert abs(result.p50_w[-1] - (0.55 * 292 + 0.45 * 480)) < 0.001
        assert np.all(result.phase_codes[1:] == 4)
        energy.append(np.sum(result.p50_w[1:] * np.diff(times.asi8) / 3.6e12))
    assert np.isclose(energy[0], energy[1], atol=1e-8)


def test_only_genuine_embedded_provider_fields_are_carried_forward():
    times = pd.date_range("2026-09-11", periods=4, freq="3h")
    baseline = xr.Dataset({
        "ECMWFSourceAirTemperatureC": ("time", [np.nan, 5.0, 6.0, 7.0]),
        "ECMWFDirectHorizontalIrradiance": ("time", [np.nan, 50.0, 60.0, 70.0]),
    }, coords={"time": times})
    fields = embedded_source_meteorology(baseline)
    assert set(fields) == {"air_temperature_c"}
    np.testing.assert_allclose(fields["air_temperature_c"].values[1:], [5, 6, 7])
    assert fields["air_temperature_c"].index.equals(pd.DatetimeIndex(times))


def test_repeated_full_build_does_not_learn_from_the_same_intervals_again():
    times = pd.date_range("2026-09-01T00:00", periods=577, freq="5min")
    solar = _solar(pd.date_range(times[0], periods=21, freq="3h"))
    first = build_forecast_dataset(_power(times), solar, horizon_hours=12)
    state = {
        "solar_calibration_state": json.loads(first.attrs["solar_calibration_state"]),
        "battery_model": BatteryModel.from_attrs(first.attrs).attrs(),
        "soc_bias_correction_pct_points_by_bucket": {"0_6h": 2.0},
    }
    with patch("generate_power_soc_forecast.fit_battery_model", side_effect=AssertionError("unchanged evidence must not refit")):
        repeated = build_forecast_dataset(_power(times), solar, horizon_hours=12, state=state)
    assert repeated.attrs["solar_calibration_state"] == first.attrs["solar_calibration_state"]
    assert json.loads(repeated.attrs["soc_bias_correction_pct_points_by_bucket"]) == {"0_6h": 2.0}


def test_paired_load_trace_is_replayed_before_any_candidate_residual():
    times = pd.date_range("2026-09-01T00:00", periods=25, freq="1h")
    solar = _solar(pd.date_range(times[-1], periods=5, freq="3h"))
    baseline = build_forecast_dataset(_power(times), solar, horizon_hours=12)
    reference = baseline.copy(deep=True)
    reference["ForecastLoadWatts"][:] = 450.0
    reference["ForecastLoadP10Watts"][:] = 430.0
    reference["ForecastLoadP90Watts"][:] = 480.0
    result = build_forecast_dataset(_power(times), solar, horizon_hours=12, fixed_load_reference=reference)
    np.testing.assert_array_equal(result.ForecastLoadWatts.values, reference.ForecastLoadWatts.values)
    np.testing.assert_array_equal(result.ForecastLoadP10Watts.values, reference.ForecastLoadP10Watts.values)
