from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from power_observation_truth import (
    ObservationPolicy, bounded_observation_view, diagnostic_soc_replays,
    evaluate_power_observations, forecast_interval_starts,
)
from power_v12_hybrid import _load_training_rows, evaluation_contract_from_forecast, _matches_evaluation_contract


def _power(times=None, *, solar=300.0, load=200.0):
    times = times if times is not None else pd.date_range("2026-09-01", periods=37, freq="5min")
    solar = np.broadcast_to(np.asarray(solar, dtype=float), len(times))
    load = np.broadcast_to(np.asarray(load, dtype=float), len(times))
    return xr.Dataset({
        "BatterySOC": ("time", np.linspace(50, 53, len(times))),
        "BatteryWatts": ("time", solar - load),
        "ACOutputWatts": ("time", np.full(len(times), 40.0)),
        "DCInverterWatts": ("time", np.full(len(times), 10.0)),
        **{f"SolarWatts_{side}": ("time", solar / 3) for side in ("East", "South", "West")},
        **{f"SolarMPPMode_{side}": ("time", np.full(len(times), 2.0)) for side in ("East", "South", "West")},
    }, coords={"time": times})


def _pdu(times, cl61=None):
    result = xr.Dataset({f"PDUOutlet{outlet}State": ("time", np.zeros(len(times))) for outlet in (4, 5, 6, 8)}, coords={"time": times})
    if cl61 is not None:
        result["PDUOutlet5State"] = ("time", np.asarray(cl61, dtype=float))
    return result


def test_interval_means_use_integrated_truth_not_endpoint_and_preserve_partial_first_interval():
    times = pd.date_range("2026-09-01", periods=37, freq="5min")
    power = _power(times, solar=np.linspace(0, 600, len(times)), load=200)
    endpoints = pd.DatetimeIndex(["2026-09-01T00:30", "2026-09-01T01:00", "2026-09-01T03:00"])
    starts = forecast_interval_starts(endpoints, endpoints[0])
    truth = evaluate_power_observations(power, starts, endpoints)
    assert pd.isna(starts[0])
    assert np.isnan(truth.load_w.iloc[0])
    assert truth.soc.iloc[0] == 50.5
    assert truth.solar_delivered_w.iloc[1] == pytest.approx(150)
    assert truth.solar_delivered_energy_wh.iloc[1] == pytest.approx(75)
    assert truth.solar_delivered_w.iloc[2] == pytest.approx(400)
    assert truth.load_w.iloc[1] == pytest.approx(200)
    assert truth.load_fallback_w.iloc[1] == 50
    assert truth.load_measurement.iloc[1] == "total_solar_minus_battery"


def test_missing_solar_sensor_does_not_become_partial_station_truth():
    power = _power().drop_vars("SolarWatts_East")
    truth = evaluate_power_observations(power, ["2026-09-01"], ["2026-09-01T01:00"])
    assert np.isnan(truth.load_w.iloc[0])
    assert np.isnan(truth.solar_delivered_w.iloc[0])
    assert truth.load_fallback_w.iloc[0] == 50
    assert truth.load_measurement.iloc[0] == "fallback_ac_inverter_not_total"


def test_no_battery_load_fallback_remains_explicit_and_diagnostic():
    truth = evaluate_power_observations(_power().drop_vars("BatteryWatts"), ["2026-09-01"], ["2026-09-01T01:00"])
    assert np.isnan(truth.load_w.iloc[0])
    assert truth.load_fallback_w.iloc[0] == 50
    assert truth.solar_delivered_w.iloc[0] == 300


def test_large_gap_fails_even_if_overall_coverage_threshold_would_pass():
    times = pd.date_range("2026-09-01", periods=24 * 12 + 1, freq="5min")
    power = _power(times).isel(time=np.r_[0:120, 123:len(times)])
    truth = evaluate_power_observations(power, [times[0]], [times[-1]])
    assert truth.load_coverage.iloc[0] > 0.90
    assert np.isnan(truth.load_w.iloc[0])
    assert truth.load_status.iloc[0] == "gap_exceeds_limit"


def test_missing_values_report_coverage_and_never_synthesize_zero():
    power = _power()
    power["BatteryWatts"].values[2:8] = np.nan
    truth = evaluate_power_observations(power, ["2026-09-01"], ["2026-09-01T01:00"])
    assert truth.load_status.iloc[0] == "insufficient_coverage"
    assert np.isnan(truth.load_energy_wh.iloc[0])
    assert 0 < truth.load_coverage.iloc[0] < 0.9


def test_mppt_censoring_preserves_delivered_energy_and_excludes_available_truth():
    power = _power()
    power["SolarMPPMode_South"].values[:] = 1
    truth = evaluate_power_observations(power, ["2026-09-01"], ["2026-09-01T01:00"])
    assert truth.solar_delivered_energy_wh.iloc[0] == 300
    assert np.isnan(truth.solar_available_w.iloc[0])
    assert truth.solar_available_status.iloc[0] == "censored_or_missing_mpp"
    assert truth.load_w.iloc[0] == 200


def test_bounded_view_keeps_interval_interior_modes_and_battery_but_is_lazy():
    times = pd.date_range("2026-08-01", periods=40 * 24 * 12, freq="5min")
    power = _power(times).assign(UnrelatedGrid=("time", np.ones(len(times)))).chunk({"time": 288})
    starts = pd.DatetimeIndex(["2026-09-01T00:00", "2026-09-02T00:00"])
    ends = starts + pd.Timedelta(hours=1)
    bounded = bounded_observation_view(power, starts, ends)
    assert 24 < bounded.sizes["time"] < 60
    assert "BatteryWatts" in bounded and "SolarMPPMode_South" in bounded
    assert "UnrelatedGrid" not in bounded
    assert bounded.BatteryWatts.chunks is not None
    expected = evaluate_power_observations(power, starts, ends)
    actual = evaluate_power_observations(bounded, starts, ends)
    np.testing.assert_allclose(actual.load_w, expected.load_w)
    np.testing.assert_allclose(actual.solar_available_energy_wh, expected.solar_available_energy_wh)


def test_direct_pdu_states_distinguish_unchanged_transition_mismatch_and_unknown():
    power = _power()
    times = pd.DatetimeIndex(power.time.values)
    cl61 = (times >= times[18]).astype(float)
    pdu = _pdu(times, cl61)
    starts = [times[0], times[12], times[24]]
    ends = [times[12], times[24], times[36]]
    truth = evaluate_power_observations(power, starts, ends, expected_modes=["DC-Only"] * 3, operating_state=pdu)
    assert truth.operating_state_status.tolist() == ["unchanged", "transition", "mismatch"]
    assert truth.realized_mode.iloc[2] == "DC-Only + CL61"
    partial = pdu.drop_vars("PDUOutlet4State")
    unknown = evaluate_power_observations(power, starts, ends, expected_modes=["DC-Only"] * 3, operating_state=partial)
    assert unknown.operating_state_status.eq("unknown").all()


def test_operating_phase_change_cannot_be_hidden_by_same_pdu_mode():
    power = _power()
    times = pd.DatetimeIndex(power.time.values)
    state = _pdu(times, np.ones(len(times)))
    state["OperatingLoadState"] = ("time", np.where(np.arange(len(times)) < 6, "CL61_low", "CL61_heater"))
    truth = evaluate_power_observations(power, [times[0]], [times[12]], expected_modes=["Ceilometer-on-AC"], operating_state=state)
    assert truth.operating_state_status.iloc[0] == "transition"


def test_no_backward_pdu_fill_from_later_observation():
    power = _power()
    times = pd.DatetimeIndex(power.time.values)
    truth = evaluate_power_observations(power, [times[0]], [times[12]], expected_modes=["DC-Only"], operating_state=_pdu(times[1:]))
    assert truth.operating_state_status.iloc[0] == "unknown"


def test_residual_training_excludes_observed_transition_and_mismatch():
    power = _power()
    times = pd.DatetimeIndex(power.time.values)
    endpoints = pd.date_range(times[0] + pd.Timedelta(hours=1), periods=3, freq="1h")
    archive = xr.Dataset({
        "ForecastLoadWatts": (("issue_time", "forecast_step"), [[100.0] * 3]),
        "ForecastValidTime": (("issue_time", "forecast_step"), [endpoints.values]),
        "ForecastLeadHours": (("issue_time", "forecast_step"), [[1.0, 2.0, 3.0]]),
        "ECMWFCycleTime": ("issue_time", [times[0]]), "LoadMode": ("issue_time", ["DC-Only"]),
        "ForecastModelContractID": ("issue_time", ["contract"]), "ForecastSystemVersion": ("issue_time", ["system"]),
    }, coords={"issue_time": [times[0]], "forecast_step": range(3)})
    pdu = _pdu(times, ((times >= times[18]) & (times < times[30])).astype(float))
    rows = _load_training_rows(archive, power, cutoff=times[-1], load_mode="DC-Only",
        control_forecast_model_contract_id="contract", control_forecast_system_version="system", operating_state=pdu)
    assert len(rows) == 1
    assert rows.valid_time.iloc[0] == endpoints[0]
    assert rows.observed_load_w.iloc[0] == 200
    assert rows.realized_state_status.iloc[0] == "unchanged"


def test_residual_never_trains_without_current_verified_state():
    from test_power_v12_hybrid import _archive_with_load_history
    times = pd.date_range("2026-06-01", periods=5 * 24 * 12, freq="5min")
    power = _power(times)
    archive = _archive_with_load_history()
    rows = _load_training_rows(archive, power, cutoff=pd.Timestamp("2026-06-04"), load_mode="DC-Only",
        control_forecast_model_contract_id="v10", control_forecast_system_version="v10-control")
    assert rows.empty
    assert rows.attrs["training_exclusions"]["current_exact_state_unavailable"]


def test_residual_spread_uses_unseen_day_errors_not_in_sample_fit_residuals():
    from test_power_v12_hybrid import _archive_with_load_history, _dense_load_truth
    from power_v12_hybrid import fit_bounded_load_residual
    times = pd.date_range("2026-06-01", periods=4 * 24 * 12 + 1, freq="5min")
    load = np.where(times >= pd.Timestamp("2026-06-03"), 500.0, 200.0)
    fit = fit_bounded_load_residual(_archive_with_load_history(), _dense_load_truth(times, load),
        issue_time="2026-06-04", forecast_times=pd.date_range("2026-06-04", periods=3, freq="h"),
        load_mode="DC-Only", control_forecast_model_contract_id="v10", control_forecast_system_version="v10-control")
    assert fit.status == "active"
    assert fit.uncertainty_status == "blocked_holdout"
    assert fit.uncertainty_samples == 16
    # The final day's +300 W error was unseen by its training fit. The 48-row
    # shrink is 1/2, so upper residual spread is +150 W after shrink, rather
    # than the narrower fitted-on-all-days residual.
    np.testing.assert_allclose(fit.p90_correction_w - fit.p50_correction_w, 150, atol=1e-6)


def test_diagnostic_replays_label_hindsight_and_stop_on_missing_interval():
    power = _power(solar=300, load=200)
    times = pd.date_range("2026-09-01", periods=4, freq="1h")
    forecast = xr.Dataset({"ForecastSolarWatts": ("time", [300.0] * 4), "ForecastLoadWatts": ("time", [100.0] * 4)}, coords={"time": times}, attrs={
        "initial_soc_time": str(times[0]), "initial_soc_pct": "50", "battery_usable_capacity_kwh": "20",
        "battery_charge_efficiency": "1", "battery_discharge_efficiency": "1"})
    result = diagnostic_soc_replays(forecast, power)
    np.testing.assert_allclose(result.DiagnosticSOCObservedSolar, [50, 51, 52, 53])
    np.testing.assert_allclose(result.DiagnosticSOCObservedLoad, [50, 50.5, 51, 51.5])
    assert result.attrs["predictive_skill_eligible"] == "false"
    power["BatteryWatts"].values[16:20] = np.nan
    result = diagnostic_soc_replays(forecast, power)
    assert np.isnan(result.DiagnosticSOCObservedLoad.values[2:]).all()


def test_semantic_digest_allows_ui_sha_change_without_pooling_different_forecast_code():
    first = xr.Dataset(attrs={"forecast_implementation_digest": "sha256:" + "a" * 64, "forecast_code_revision": "ui-sha-1"})
    second = first.copy()
    second.attrs["forecast_code_revision"] = "ui-sha-2"
    contract = evaluation_contract_from_forecast(first)
    assert _matches_evaluation_contract(second, contract)
    second.attrs["forecast_implementation_digest"] = "sha256:" + "b" * 64
    assert not _matches_evaluation_contract(second, contract)
    legacy = xr.Dataset(attrs={"forecast_code_revision": "ui-sha-1"})
    assert not _matches_evaluation_contract(legacy, contract)


def test_non_monotonic_or_duplicate_forecast_intervals_fail():
    with pytest.raises(ValueError, match="increase"):
        forecast_interval_starts(["2026-09-01T01:00", "2026-09-01T01:00"], "2026-09-01")


def test_future_soc_endpoint_is_not_matured_by_nearby_latest_telemetry():
    power = _power()
    end = pd.Timestamp(power.time.values[-1]) + pd.Timedelta(minutes=5)
    truth = evaluate_power_observations(power, [pd.NaT], [end])
    assert np.isnan(truth.soc.iloc[0])
    assert truth.soc_status.iloc[0] == "endpoint_unavailable"
