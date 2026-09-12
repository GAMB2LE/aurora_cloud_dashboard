"""Regression checks for physically coherent candidate uncertainty."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from power_v12_ensemble import (
    _candidate_member_load, _reserve_episode_count,
    align_baseline_ensemble_grid, candidate_ensemble_contract_id,
)


def fixtures():
    times = pd.date_range("2026-06-21T06:00", periods=3, freq="3h")
    attrs = {"initial_soc_time": str(times[0]), "initial_soc_pct": "80",
             "ecmwf_cycle_time": str(times[0]), "forecast_model_contract_id": "v13",
             "forecast_system_version": "power-v13", "forecast_identity_id": "issue-a",
             "battery_usable_capacity_kwh": "26", "battery_charge_efficiency": ".92",
             "battery_discharge_efficiency": ".92", "battery_max_charge_w": "3000",
             "battery_max_discharge_w": "3000"}
    baseline = xr.Dataset({"ForecastLoadWatts": ("time", [100., 100., 100.]),
                           "ForecastSolarWatts": ("time", [0., 200., 400.]),
                           "BatterySOCForecast": ("time", [80., 80., 80.])},
                          coords={"time": times}, attrs=attrs)
    ensemble = xr.Dataset({
        "BatterySOCForecastEnsemble": (("member", "time"), np.full((10, 3), 80.)),
        "ECMWFSolarIrradianceEnsemble": (("member", "time"), np.tile([0., 100., 200.], (10, 1))),
        "ForecastSolarWattsEnsemble": (("member", "time"), np.tile([0., 200., 400.], (10, 1))),
        "ForecastLoadWattsEnsemble": (("member", "time"), np.full((10, 3), 100.)),
        "BatteryUsableCapacityKWhEnsemble": ("member", np.full(10, 26.)),
        "BatteryChargeEfficiencyEnsemble": ("member", np.full(10, .92)),
        "BatteryDischargeEfficiencyEnsemble": ("member", np.full(10, .92)),
    }, coords={"member": np.arange(10), "time": times}, attrs=attrs)
    return baseline, ensemble


def test_ensemble_campaign_contract_excludes_issue_state():
    baseline, ensemble = fixtures()
    contract = candidate_ensemble_contract_id(baseline, ensemble)
    changed = baseline.copy(deep=True)
    changed.attrs.update(forecast_identity_id="issue-b", battery_max_charge_w="2700")
    other = ensemble.copy(deep=True)
    other["ForecastSolarWattsEnsemble"] *= .8
    assert candidate_ensemble_contract_id(changed, other) == contract
    changed.attrs["forecast_model_contract_id"] = "new-algorithm"
    assert candidate_ensemble_contract_id(changed, other) != contract


def test_conservative_alignment_preserves_interval_energy_and_original():
    baseline, ensemble = fixtures()
    ensemble["BatterySOCForecastEnsemble"] = (("member", "time"),
        np.tile([80., 80. + 100 * .3 * .92 / 26,
                 80. + 100 * 1.2 * .92 / 26], (10, 1)))
    original = ensemble.copy(deep=True)
    target = baseline.reindex(time=pd.date_range(baseline.time.values[0], periods=7, freq="h"))
    out = align_baseline_ensemble_grid(target, ensemble)
    np.testing.assert_allclose(out.ForecastSolarWattsEnsemble.values[:, 1:].sum(axis=1), 1800.)
    assert np.isfinite(out.BatterySOCForecastEnsemble.values).all()
    assert out.attrs["original_baseline_ensemble_signature"]
    xr.testing.assert_identical(ensemble, original)


def test_alignment_rejects_changed_nonlinear_baseline_comparator():
    baseline, ensemble = fixtures()
    target = baseline.reindex(time=pd.date_range(baseline.time.values[0], periods=7, freq="h"))
    with pytest.raises(ValueError, match="differs from archived SOC"):
        align_baseline_ensemble_grid(target, ensemble)


def test_alignment_rejects_cycle_anchor_and_tail_changes():
    baseline, ensemble = fixtures()
    altered = baseline.copy()
    altered.attrs["ecmwf_cycle_time"] = "2026-06-21T00:00"
    with pytest.raises(ValueError, match="cycle"):
        align_baseline_ensemble_grid(altered, ensemble)
    altered = baseline.reindex(time=pd.date_range(baseline.time.values[0], periods=9, freq="h"))
    with pytest.raises(ValueError, match="horizon"):
        align_baseline_ensemble_grid(altered, ensemble)


def test_residual_quantiles_propagate_coherent_reproducible_spread():
    baseline, ensemble = fixtures()
    candidate = baseline.copy(deep=True)
    candidate["ForecastLoadWatts"] += 10
    profile = {"status": "active", "uncertainty_status": "blocked_holdout", "p10_correction_w": np.full(3, -10.),
               "p50_correction_w": np.full(3, 10.), "p90_correction_w": np.full(3, 40.)}
    def build():
        return np.asarray([_candidate_member_load(baseline, candidate, ensemble,
            member_index=i, times=pd.DatetimeIndex(baseline.time.values),
            apply_load_residual=True, load_residual_profile=profile) for i in range(10)])
    first = build()
    np.testing.assert_array_equal(first, build())
    assert np.ptp(first[:, 1]) > 40
    np.testing.assert_allclose(first[:, 1], first[:, 2])
    profile["p10_correction_w"] = np.full(3, 80.)
    with pytest.raises(ValueError, match="unordered"):
        build()


def test_reserve_events_count_crossings_not_issues_or_low_rows():
    times = pd.date_range("2026-06-01", periods=8, freq="h")
    obs = np.asarray([60., 35., 30., 20., 60., 35., 30., 60.])
    assert _reserve_episode_count(times.repeat(4), np.repeat(obs, 4)) == 2
    assert _reserve_episode_count(times, np.full(8, 30.)) == 0
    assert _reserve_episode_count(pd.DatetimeIndex([times[0], times[0] + pd.Timedelta(days=1)]),
                                  np.asarray([60., 30.])) == 0


def test_archive_verification_compares_datetime_instants_not_storage_units(tmp_path):
    from power_archive_io import write_forecast_archive
    data = xr.Dataset({"ECMWFCycleTime": ("issue_time", np.asarray(
        ["2026-06-21T00:00"], dtype="datetime64[us]"))},
        coords={"issue_time": np.asarray(["2026-06-21T06:00"], dtype="datetime64[ns]")})
    path = tmp_path / "archive.zarr"
    write_forecast_archive(data, path)
    with xr.open_zarr(path, chunks={}) as restored:
        np.testing.assert_array_equal(restored.ECMWFCycleTime.values.astype("datetime64[ns]"),
                                      data.ECMWFCycleTime.values.astype("datetime64[ns]"))


def test_implementation_digest_tracks_local_dependencies_not_unrelated_ui(tmp_path, monkeypatch):
    import power_implementation_identity as identity
    monkeypatch.setattr(identity, "FORECAST_IMPLEMENTATION_FILES", ("forecast.py",))
    (tmp_path / "forecast.py").write_text("from provider import forcing\n")
    (tmp_path / "provider.py").write_text("forcing = 1\n")
    before = identity.forecast_implementation_digest(tmp_path)
    (tmp_path / "dashboard.py").write_text("color = 'green'\n")
    assert identity.forecast_implementation_digest(tmp_path) == before
    (tmp_path / "provider.py").write_text("forcing = 2\n")
    assert identity.forecast_implementation_digest(tmp_path) != before
