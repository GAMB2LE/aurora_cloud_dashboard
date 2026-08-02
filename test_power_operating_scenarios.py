from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from power_operating_scenarios import (
    COMPONENT_INDEX,
    MODE_DC_ONLY,
    SCENARIO_CL61,
    SCENARIO_CURRENT,
    SCENARIO_DC_ONLY,
    SCENARIO_OPTIMIZED,
    _align_ensemble_solar_contract,
    _validate_scenario_invariants,
    build_observation_frame,
    build_operating_scenarios,
    evaluate_custom_schedule,
    fit_operating_model,
    integrate_soc_members,
    mode_from_code,
    mode_id,
    mode_kits,
    load_operating_events,
    optimize_cl61_schedule,
    _tier_profile_members,
)
from power_scenario_catalog import SUGGESTED_OPERATING_SCENARIOS
from generate_power_operating_scenarios import (
    _verification_for_record,
    _planning_forecast_provenance,
    _validate_operating_inputs,
    generate as generate_operating_products,
    scenario_publication_signature,
)


def _training_data() -> tuple[xr.Dataset, xr.Dataset]:
    times = pd.date_range("2026-07-15T00:00:00", periods=49, freq="15min")
    cl61_active = times >= pd.Timestamp("2026-07-15T06:00:00")
    solar = np.zeros(len(times), dtype=float)
    load = np.where(cl61_active, 420.0, 200.0)
    power = xr.Dataset(
        {
            "BatterySOC": (("time",), np.linspace(90.0, 86.0, len(times))),
            "BatteryWatts": (("time",), solar - load),
            "SolarWatts_East": (("time",), solar / 3.0),
            "SolarWatts_South": (("time",), solar / 3.0),
            "SolarWatts_West": (("time",), solar / 3.0),
            "ACOutputWatts": (("time",), np.where(cl61_active, 220.0, 0.0)),
            "DCInverterWatts": (("time",), np.full(len(times), 8.0)),
        },
        coords={"time": times},
    )
    pdu = xr.Dataset(
        {
            "PDUOutlet4Watts": (("time",), np.zeros(len(times))),
            "PDUOutlet4State": (("time",), np.zeros(len(times))),
            "PDUOutlet5Watts": (("time",), np.where(cl61_active, 220.0, 0.0)),
            "PDUOutlet5State": (("time",), cl61_active.astype(float)),
            "PDUOutlet6Watts": (("time",), np.zeros(len(times))),
            "PDUOutlet6State": (("time",), np.zeros(len(times))),
            "PDUOutlet8Watts": (("time",), np.zeros(len(times))),
            "PDUOutlet8State": (("time",), np.zeros(len(times))),
        },
        coords={"time": times},
    )
    return power, pdu


def _forecast_inputs(issue: pd.Timestamp, horizon_hours: int = 96) -> tuple[xr.Dataset, xr.Dataset]:
    times = pd.date_range(issue, issue + pd.Timedelta(hours=horizon_hours), freq="3h")
    solar = np.full(len(times), 520.0, dtype=float)
    load = np.full(len(times), 640.0, dtype=float)
    deterministic = xr.Dataset(
        {
            "ForecastSolarWatts": (("time",), solar),
            "ForecastLoadWatts": (("time",), load),
        },
        coords={"time": times},
        attrs={"battery_capacity_kwh": "26", "initial_soc_time": issue.isoformat()},
    )
    members = np.vstack([solar * factor for factor in np.linspace(0.75, 1.25, 20)])
    ensemble = xr.Dataset(
        {
            "ForecastSolarWattsEnsemble": (("member", "time"), members),
            "ForecastLoadWattsEnsemble": (("member", "time"), np.tile(load, (20, 1))),
        },
        coords={"member": np.arange(1, 21), "time": times},
    )
    return deterministic, ensemble


class OperatingScenarioTests(unittest.TestCase):
    def test_validation_reanchors_a_planning_forecast_with_an_old_soc_anchor(self) -> None:
        power, _ = _training_data()
        forecast, _ = _forecast_inputs(pd.Timestamp("2026-07-15T00:00:00"), horizon_hours=240)
        forecast.attrs["initial_soc_time"] = "2026-07-13T00:00:00"

        anchor_time, anchor_soc, _ = _validate_operating_inputs(
            power,
            forecast,
            planning_hours=96,
            max_power_age_minutes=None,
            now=pd.Timestamp("2026-07-15T12:00:00"),
        )

        self.assertEqual(anchor_time, pd.Timestamp(power.time.values[-1]))
        self.assertEqual(anchor_soc, float(power.BatterySOC.values[-1]))
    def test_archived_decision_verifies_against_actual_soc_and_mode(self) -> None:
        times = pd.date_range("2026-07-18T00:00:00", periods=3, freq="1h")
        record = {
            "forecast_trace": {
                "time_utc": [value.isoformat() for value in times],
                "soc_p50_pct": [80.0, 79.0, 78.0],
                "mode_code": [0, 0, 1],
            }
        }
        power = xr.Dataset(
            {"BatterySOC": (("time",), [80.0, 78.0, 77.0])},
            coords={"time": times},
        )
        state = xr.Dataset(
            {"OperatingModeCode": (("time",), [0, 0, 1])},
            coords={"time": times},
        )

        verification = _verification_for_record(record, power=power, operating_state=state)

        self.assertIsNotNone(verification)
        self.assertEqual(verification["status"], "complete")
        self.assertAlmostEqual(verification["soc_mae_pct"], 2.0 / 3.0)
        self.assertEqual(verification["mode_adherence_fraction"], 1.0)

    def test_planning_provenance_preserves_cached_cycle_identity(self) -> None:
        times = pd.date_range("2026-07-18T00:00:00", periods=3, freq="1h")
        forecast = xr.Dataset(
            {"ForecastSolarWatts": (("time",), [10.0, 20.0, 30.0])},
            coords={"time": times},
            attrs={
                "generated_at_utc": "2026-07-18T00:05:00+00:00",
                "initial_soc_time": "2026-07-18T00:00:00",
                "forecast_refresh_kind": "cached_reanchor",
                "forecast_verification_eligible": "false",
            },
        )

        provenance = _planning_forecast_provenance(forecast)

        self.assertEqual(provenance["planning_forecast_refresh_kind"], "cached_reanchor")
        self.assertEqual(provenance["planning_forecast_initial_soc_time"], "2026-07-18T00:00:00")
        self.assertEqual(provenance["planning_forecast_time_coverage_start"], "2026-07-18T00:00:00")
        self.assertEqual(provenance["planning_forecast_time_coverage_end"], "2026-07-18T02:00:00")

    def test_mode_code_round_trip_supports_combinations(self) -> None:
        value = mode_id(("CL61", "Radar"))
        code = 0
        for kit in mode_kits(value):
            code |= {"CL61": 1, "Radar": 2}[kit]
        self.assertEqual(mode_from_code(code), value)

    def test_model_learns_dc_baseline_and_cl61_increment(self) -> None:
        power, pdu = _training_data()

        result = fit_operating_model(power, pdu, lookback_days=2)

        self.assertEqual(result.current_mode, mode_id(("CL61",)))
        self.assertGreater(result.current_confidence, 0.90)
        self.assertIn(MODE_DC_ONLY, result.learned_modes)
        self.assertIn(mode_id(("CL61",)), result.learned_modes)
        self.assertIn(MODE_DC_ONLY, result.mode_load_profiles)
        self.assertIn(mode_id(("CL61",)), result.mode_load_profiles)
        self.assertIn("mode_load_profiles", result.state_dataset.attrs)
        self.assertAlmostEqual(result.component_mean[COMPONENT_INDEX["DC"]], 200.0, delta=25.0)
        self.assertAlmostEqual(result.component_mean[COMPONENT_INDEX["CL61"]], 220.0, delta=20.0)
        probabilities = result.state_dataset["OperatingModeProbability"].values
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_exact_state_confirmation_requires_all_assigned_pdu_outlets(self) -> None:
        power, pdu = _training_data()
        partial_pdu = pdu[["PDUOutlet5Watts", "PDUOutlet5State"]]

        observations = build_observation_frame(power, partial_pdu, lookback_days=2)

        self.assertFalse(observations["direct_state_confirmed"].any())

    def test_unconfirmed_high_load_does_not_contaminate_dc_only_profile(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=32, freq="15min")
        load = np.r_[np.full(12, 200.0), np.full(20, 750.0)]
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(90.0, 86.0, len(times))),
                "BatteryWatts": (("time",), -load),
                "SolarWatts_East": (("time",), np.zeros(len(times))),
                "SolarWatts_South": (("time",), np.zeros(len(times))),
                "SolarWatts_West": (("time",), np.zeros(len(times))),
                "ACOutputWatts": (("time",), np.zeros(len(times))),
                "DCInverterWatts": (("time",), np.full(len(times), 8.0)),
            },
            coords={"time": times},
        )
        pdu_times = times[:12]
        pdu = xr.Dataset(
            {
                f"PDUOutlet{outlet}{metric}": (("time",), np.zeros(len(pdu_times)))
                for outlet in (4, 5, 6, 8)
                for metric in ("Watts", "State")
            },
            coords={"time": pdu_times},
        )

        result = fit_operating_model(power, pdu, lookback_days=2)

        profile = result.mode_load_profiles[MODE_DC_ONLY]
        self.assertLess(max(value.p90_w for value in profile.phase_profiles.values()), 300.0)
        self.assertLess(result.component_mean[COMPONENT_INDEX["DC"]], 300.0)
        self.assertGreater(result.state["confirmed_state_observation_count"], 8)

    def test_incompatible_saved_phase_profile_is_relearned(self) -> None:
        power, pdu = _training_data()
        first = fit_operating_model(power, pdu, lookback_days=2)
        poisoned = json.loads(json.dumps(first.state))
        saved = poisoned["mode_load_profiles"][MODE_DC_ONLY]
        saved["schema_version"] = 1
        for phase in saved["phase_profiles"].values():
            phase.update({"p10_w": 850.0, "p50_w": 900.0, "p90_w": 950.0})

        second = fit_operating_model(power, pdu, raw_state=poisoned, lookback_days=2)

        rebuilt = second.state["mode_load_profiles"][MODE_DC_ONLY]
        self.assertEqual(second.state["new_observation_count"], 0)
        self.assertEqual(rebuilt["schema_version"], 2)
        self.assertLess(rebuilt["phase_profiles"]["steady"]["p90_w"], 300.0)

    def test_saved_state_does_not_retrain_the_same_observations(self) -> None:
        power, pdu = _training_data()
        first = fit_operating_model(power, pdu, lookback_days=2)

        second = fit_operating_model(power, pdu, raw_state=first.state, lookback_days=2)

        np.testing.assert_allclose(second.component_mean, first.component_mean, atol=1e-9)
        np.testing.assert_allclose(second.component_covariance, first.component_covariance, atol=1e-9)
        self.assertEqual(second.state["new_observation_count"], 0)

    def test_repeated_cl61_load_levels_become_two_regimes(self) -> None:
        power, pdu = _training_data()
        active = np.asarray(pdu["PDUOutlet5Watts"].values) >= 5.0
        cl61 = np.where(active, np.where(np.arange(len(active)) % 2 == 0, 39.0, 222.0), 0.0)
        pdu["PDUOutlet5Watts"] = (("time",), cl61)
        result = fit_operating_model(power, pdu, lookback_days=2)

        regimes = result.component_regimes["CL61"]
        self.assertEqual(len(regimes), 2)
        self.assertLess(regimes[0]["mean_w"], 50.0)
        self.assertGreater(regimes[1]["mean_w"], 200.0)
        self.assertGreater(result.component_covariance[COMPONENT_INDEX["CL61"], COMPONENT_INDEX["CL61"]], 1000.0)

    def test_short_pdu_combination_is_visible_with_observed_maturity(self) -> None:
        power, pdu = _training_data()
        radar = np.zeros(power.sizes["time"], dtype=float)
        radar[-3:] = 285.0
        pdu["PDUOutlet6Watts"] = (("time",), radar)
        pdu["PDUOutlet6State"] = (("time",), (radar > 0).astype(float))
        result = fit_operating_model(power, pdu, lookback_days=2)
        mode = mode_id(("CL61", "Radar"))

        self.assertIn(mode, result.observed_modes)
        self.assertEqual(result.mode_maturity[mode], "observed")
        self.assertNotIn(mode, result.learned_modes)

    def test_operator_event_file_is_loaded_without_overriding_pdu_mode(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.csv"
            path.write_text("time_utc,action,kit,note\n2026-07-15T06:00:00Z,on,CL61,test\n", encoding="utf-8")
            events = load_operating_events(path)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].active)
        self.assertEqual(events[0].kit, "CL61")

    def test_stale_pdu_evidence_does_not_keep_cl61_active(self) -> None:
        power, pdu = _training_data()
        later_times = pd.date_range("2026-07-15T12:15:00", periods=5, freq="15min")
        later = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(85.9, 85.5, len(later_times))),
                "BatteryWatts": (("time",), np.full(len(later_times), -430.0)),
                "SolarWatts_East": (("time",), np.zeros(len(later_times))),
                "SolarWatts_South": (("time",), np.zeros(len(later_times))),
                "SolarWatts_West": (("time",), np.zeros(len(later_times))),
                "ACOutputWatts": (("time",), np.full(len(later_times), 230.0)),
                "DCInverterWatts": (("time",), np.full(len(later_times), 8.0)),
            },
            coords={"time": later_times},
        )
        extended = xr.concat([power, later], dim="time")

        result = fit_operating_model(extended, pdu, lookback_days=2)

        self.assertEqual(result.current_mode, "unknown_ac")

    def test_operating_scenarios_reject_stale_power_or_expired_solar(self) -> None:
        power, _ = _training_data()
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, _ = _forecast_inputs(issue, horizon_hours=96)

        with self.assertRaisesRegex(ValueError, "stale SOC/load input"):
            _validate_operating_inputs(
                power,
                deterministic,
                planning_hours=96,
                max_power_age_minutes=20,
                now=issue + pd.Timedelta(minutes=21),
            )

        short_forecast, _ = _forecast_inputs(issue, horizon_hours=90)
        with self.assertRaisesRegex(ValueError, "minimum decision horizon"):
            _validate_operating_inputs(
                power,
                short_forecast,
                planning_hours=240,
                minimum_horizon_hours=96,
                max_power_age_minutes=None,
                now=issue,
            )

    def test_named_scenarios_replace_fixed_watt_curves(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
            optimization_hours=96,
        )

        scenario_ids = set(str(value) for value in scenarios["scenario"].values)
        self.assertTrue({SCENARIO_DC_ONLY, SCENARIO_CL61, SCENARIO_OPTIMIZED}.issubset(scenario_ids))
        self.assertNotIn("BatterySOCForecast_Load100W", scenarios)
        self.assertEqual(float(scenarios["ScenarioSOCP50"].isel(time=0).min()), 86.0)
        optimized = scenarios.sel(scenario=SCENARIO_OPTIMIZED)
        self.assertGreaterEqual(float(optimized["ScenarioMinimumP10SOC"]), 40.0)
        self.assertEqual(scenarios.attrs["control_authority"], "advisory_only")

    def test_ensemble_is_recalibrated_to_the_planning_solar_contract(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic["ECMWFSolarIrradiance"] = deterministic["ForecastSolarWatts"] / 2.0
        ensemble["ECMWFSolarIrradianceEnsemble"] = ensemble["ForecastSolarWattsEnsemble"] / 4.0
        deterministic.attrs["solar_calibration_factor_w_per_wm2"] = "2"
        deterministic.attrs["solar_calibration_contract_id"] = "solar-a"
        ensemble.attrs["solar_calibration_contract_id"] = "solar-b"

        aligned, metadata = _align_ensemble_solar_contract(deterministic, ensemble)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble)

        self.assertIsNotNone(aligned)
        np.testing.assert_allclose(
            aligned["ForecastSolarWattsEnsemble"].values,
            ensemble["ECMWFSolarIrradianceEnsemble"].values * 2.0,
        )
        self.assertEqual(metadata["solar_ensemble_recalibrated"], "true")
        self.assertEqual(scenarios.attrs["solar_calibration_contract_id"], "solar-a")
        self.assertEqual(scenarios.attrs["solar_ensemble_source_calibration_contract_id"], "solar-b")
        self.assertEqual(scenarios.attrs["solar_ensemble_recalibrated"], "true")

    def test_mismatched_solar_contract_without_raw_members_is_rejected(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic["ECMWFSolarIrradiance"] = deterministic["ForecastSolarWatts"] / 2.0
        deterministic.attrs["solar_calibration_contract_id"] = "solar-a"
        ensemble.attrs["solar_calibration_contract_id"] = "solar-b"

        with self.assertRaisesRegex(ValueError, "ensemble lacks raw irradiance members"):
            build_operating_scenarios(power, deterministic, model, ensemble=ensemble)

    def test_scenario_publication_signature_ignores_generation_time(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        first = scenario_publication_signature(scenarios)
        scenarios.attrs["generated_at_utc"] = "2099-01-01T00:00:00+00:00"

        self.assertEqual(scenario_publication_signature(scenarios), first)
        scenarios["ScenarioLoadP50Watts"].values[0, 0] += 50.0
        self.assertNotEqual(scenario_publication_signature(scenarios), first)

    def test_current_scenario_uses_the_finite_operating_state_and_soc_anchor(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        current = scenarios.sel(scenario=SCENARIO_CURRENT)

        self.assertEqual(pd.Timestamp(scenarios.time.values[0]), issue)
        self.assertEqual(float(current["ScenarioSOCP50"].isel(time=0)), 86.0)
        expected_load = float(scenarios.attrs["modeled_current_load_p50_w"])
        self.assertAlmostEqual(float(current["ScenarioLoadP50Watts"].isel(time=0)), expected_load, places=3)
        self.assertEqual(
            scenarios.attrs["load_baseline_source"],
            "finite_state_component_model_for_all_operational_scenarios",
        )
        self.assertEqual(scenarios.attrs["load_state_contract"], "finite_operating_state_phases_v2")
        self.assertIn("ScenarioLoadPhaseCode", scenarios)
        self.assertIn("ScenarioLoadPhaseEpoch", scenarios)
        np.testing.assert_allclose(np.diff(current["ScenarioLoadP50Watts"].values), 0.0)

    def test_current_scenario_reuses_the_shared_system_as_is_state_distribution(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic.attrs.update(
            {
                "load_state_contract": "finite_operating_state_phases_v2",
                "load_state_hold_policy": "hold_confirmed_state_allow_detected_phase_or_explicit_schedule_transition",
            }
        )
        ensemble["ForecastLoadWattsEnsemble"][:] = 515.0

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        current = scenarios.sel(scenario=SCENARIO_CURRENT)

        np.testing.assert_allclose(current["ScenarioLoadP50Watts"].values, 515.0)
        self.assertEqual(
            scenarios.attrs["current_system_load_source"],
            "shared_system_as_is_finite_state_contract",
        )

    def test_scenario_validation_rejects_load_drift_inside_one_mode(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble)
        current_index = int(np.flatnonzero(scenarios["scenario"].values == SCENARIO_CURRENT)[0])
        for name in ("ScenarioLoadP10Watts", "ScenarioLoadP50Watts", "ScenarioLoadP90Watts"):
            scenarios[name].values[current_index, 1] += 25.0

        with self.assertRaisesRegex(ValueError, "without an operating-state or phase transition"):
            _validate_scenario_invariants(scenarios)

    def test_all_instruments_uas_tier3_uses_tier_profile_and_stays_above_dc(self) -> None:
        power, pdu = _training_data()
        pdu["PDUOutlet4Watts"] = (("time",), np.full(power.sizes["time"], 108.0))
        pdu["PDUOutlet4State"] = (("time",), np.ones(power.sizes["time"]))
        pdu["PDUOutlet6Watts"] = (("time",), np.full(power.sizes["time"], 285.0))
        pdu["PDUOutlet6State"] = (("time",), np.ones(power.sizes["time"]))
        pdu["PDUOutlet8Watts"] = (("time",), np.full(power.sizes["time"], 230.0))
        pdu["PDUOutlet8State"] = (("time",), np.ones(power.sizes["time"]))
        tier = pd.Series(3.0, index=pd.DatetimeIndex(power["time"].values))
        model = fit_operating_model(power, pdu, uas_tier=tier, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )

        tier3 = scenarios.sel(scenario="suggested_all_uas_tier3")
        dc_only = scenarios.sel(scenario=SCENARIO_DC_ONLY)
        np.testing.assert_array_equal(tier3["ScenarioUASEffectiveTier"].values, 3)
        self.assertTrue(
            np.all(
                tier3["ScenarioLoadP50Watts"].values
                >= dc_only["ScenarioLoadP50Watts"].values
            )
        )
        self.assertEqual(
            str(tier3["scenario_mode_maturity"].item()),
            "provisional",
        )

    def test_provisional_uas_tier_profile_uses_conservative_fallback(self) -> None:
        members = _tier_profile_members(
            {
                "p10_w": 170.0,
                "p50_w": 175.0,
                "p90_w": 180.0,
                "maturity": "provisional",
            },
            1_000,
            seed=4,
        )

        self.assertAlmostEqual(float(np.median(members)), 108.0, delta=1.0)
        self.assertLess(float(np.quantile(members, 0.10)), 60.0)
        self.assertGreater(float(np.quantile(members, 0.90)), 295.0)

    def test_uas_tier_requires_three_separate_episodes_for_reliability(self) -> None:
        power, pdu = _training_data()
        sample_count = power.sizes["time"]
        pdu["PDUOutlet4Watts"] = (("time",), np.full(sample_count, 175.0))
        pdu["PDUOutlet4State"] = (("time",), np.ones(sample_count))
        tier_values = np.full(sample_count, 3.0)
        tier_values[8:10] = 2.0
        tier_values[18:20] = 2.0
        tiers = pd.Series(tier_values, index=pd.DatetimeIndex(power["time"].values))

        result = fit_operating_model(power, pdu, uas_tier=tiers, lookback_days=2)
        tier3 = result.uas_tier_profiles["3"]

        self.assertEqual(tier3["episode_count"], 3.0)
        self.assertGreaterEqual(tier3["observed_hours"], 6.0)
        self.assertEqual(tier3["maturity"], "reliable")

    def test_soc_integration_honours_member_specific_capacity(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=3, freq="1h")
        solar = np.zeros((2, len(times)))
        load = np.full((2, len(times)), 1_000.0)

        soc = integrate_soc_members(
            initial_soc=80.0,
            times=times,
            solar_members_w=solar,
            load_members_w=load,
            capacity_kwh=26.0,
            member_capacity_kwh=np.array([10.0, 20.0]),
            member_charge_efficiency=np.ones(2),
            member_discharge_efficiency=np.ones(2),
        )

        self.assertAlmostEqual(float(soc[0, -1]), 60.0)
        self.assertAlmostEqual(float(soc[1, -1]), 70.0)

    def test_optimizer_enforces_minimum_run_and_daily_start_limit(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=97, freq="1h")
        solar = np.full((20, len(times)), 500.0)
        components = np.tile(np.array([200.0, 220.0, 300.0, 250.0, 200.0, 250.0]), (20, 1))

        result = optimize_cl61_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=70.0,
            capacity_kwh=26.0,
            base_mode=MODE_DC_ONLY,
            horizon_hours=96,
        )

        on = np.asarray(["CL61" in mode_kits(value) for value in result.modes], dtype=bool)
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        for start in starts:
            stop_candidates = np.flatnonzero(~on[start:])
            stop = start + int(stop_candidates[0]) if stop_candidates.size else len(on)
            self.assertGreaterEqual(stop - start, 12)
        start_days = [times[index].date() for index in starts]
        self.assertEqual(len(start_days), len(set(start_days)))
        self.assertGreaterEqual(result.minimum_p10_soc, 40.0)

    def test_optimizer_protects_reserve_through_full_planning_horizon(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(np.array([40.0, 100.0, 0.0, 0.0, 0.0, 0.0]), (20, 1))

        result = optimize_cl61_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=80.0,
            capacity_kwh=26.0,
            base_mode=MODE_DC_ONLY,
            horizon_hours=96,
        )

        on = np.asarray(["CL61" in mode_kits(value) for value in result.modes], dtype=bool)
        self.assertEqual(len(result.modes), len(times))
        self.assertEqual(int(np.count_nonzero(on)), 0)
        self.assertTrue(result.safe)
        self.assertGreaterEqual(result.minimum_p10_soc, 40.0)
        self.assertLess(result.minimum_p10_soc, 45.0)

    def test_custom_schedule_reacts_to_start_and_duration(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble, horizon_hours=96)

        short = evaluate_custom_schedule(scenarios, start_time=issue + pd.Timedelta(hours=6), duration_hours=12)
        long = evaluate_custom_schedule(scenarios, start_time=issue + pd.Timedelta(hours=6), duration_hours=24)

        self.assertEqual(short["collection_hours"], 12.0)
        self.assertEqual(long["collection_hours"], 24.0)
        self.assertLessEqual(long["final_p10_soc"], short["final_p10_soc"] + 1e-6)

    def test_custom_schedule_supports_each_learned_instrument_load(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble, horizon_hours=96)

        radar = evaluate_custom_schedule(
            scenarios,
            start_time=issue + pd.Timedelta(hours=6),
            duration_hours=12,
            kit="Radar",
        )

        self.assertEqual(radar["kit"], "Radar")
        self.assertEqual(radar["collection_hours"], 12.0)
        self.assertTrue(any("Radar" in mode_kits(mode) for mode in radar["modes"]))
        self.assertGreater(float(np.nanmax(radar["load_p50_w"])), 0.0)

    def test_planning_horizon_extends_short_ensemble_with_deterministic_shape(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic_times = pd.date_range(issue, issue + pd.Timedelta(hours=240), freq="3h")
        deterministic_solar = np.maximum(600.0 * np.sin(np.arange(len(deterministic_times)) * np.pi / 8.0), 0.0)
        deterministic = xr.Dataset(
            {"ForecastSolarWatts": (("time",), deterministic_solar)},
            coords={"time": deterministic_times},
            attrs={"battery_capacity_kwh": "26"},
        )
        ensemble_times = deterministic_times[:33]
        ensemble = xr.Dataset(
            {
                "ForecastSolarWattsEnsemble": (
                    ("member", "time"),
                    np.vstack([deterministic_solar[:33] * factor for factor in np.linspace(0.8, 1.2, 20)]),
                )
            },
            coords={"member": np.arange(1, 21), "time": ensemble_times},
        )

        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble, horizon_hours=240)

        self.assertEqual(scenarios.sizes["time"], 241)
        self.assertEqual(scenarios.attrs["uncertainty_extrapolated"], "true")
        tail_solar = scenarios["SolarP50Watts"].isel(time=slice(-24, None)).values
        self.assertGreater(float(np.nanmax(tail_solar)), 0.0)
        self.assertGreater(float(np.nanmax(tail_solar) - np.nanmin(tail_solar)), 0.0)
        optimized_codes = scenarios.sel(scenario=SCENARIO_OPTIMIZED)["ScenarioModeCode"].values
        self.assertTrue(np.all((optimized_codes[97:] & 1) == 0))

    def test_generator_persists_versioned_state_scenarios_and_recommendation(self) -> None:
        power, pdu = _training_data()
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "power": root / "power.zarr",
                "pdu": root / "pdu.zarr",
                "forecast": root / "forecast.zarr",
                "ensemble": root / "ensemble.zarr",
                "state": root / "operating_state.zarr",
                "scenarios": root / "operating_scenarios.zarr",
                "model": root / "model.json",
                "recommendations": root / "recommendations.json",
            }
            power.to_zarr(paths["power"], mode="w", consolidated=True)
            pdu.to_zarr(paths["pdu"], mode="w", consolidated=True)
            deterministic.to_zarr(paths["forecast"], mode="w", consolidated=True)
            ensemble.to_zarr(paths["ensemble"], mode="w", consolidated=True)

            generate_operating_products(
                power_zarr=paths["power"],
                pdu_zarr=paths["pdu"],
                forecast_zarr=paths["forecast"],
                ensemble_zarr=paths["ensemble"],
                state_output=paths["state"],
                scenario_output=paths["scenarios"],
                model_state=paths["model"],
                bootstrap_state=None,
                recommendation_archive=paths["recommendations"],
                planning_hours=96,
                optimization_hours=96,
                lookback_days=2,
            )

            state = xr.open_zarr(paths["state"], chunks={})
            scenarios = xr.open_zarr(paths["scenarios"], chunks={})
            try:
                self.assertEqual(state.attrs["model_version"], "9")
                self.assertEqual(scenarios.attrs["control_authority"], "advisory_only")
                self.assertIn("optimized_cl61", set(str(value) for value in scenarios["scenario"].values))
                scenario_ids = [str(value) for value in scenarios["scenario"].values]
                scenario_labels = [str(value) for value in scenarios["scenario_label"].values]
                for definition in SUGGESTED_OPERATING_SCENARIOS:
                    self.assertIn(definition.scenario_id, scenario_ids)
                    index = scenario_ids.index(definition.scenario_id)
                    self.assertEqual(scenario_labels[index], definition.label)
                    codes = np.asarray(scenarios["ScenarioModeCode"].isel(scenario=index).values)
                    expected_mode = mode_id(definition.instruments)
                    self.assertTrue(all(mode_from_code(value) == expected_mode for value in codes))
            finally:
                state.close()
                scenarios.close()
            self.assertTrue(paths["model"].exists())
            self.assertTrue(paths["recommendations"].exists())
            archive = json.loads(paths["recommendations"].read_text(encoding="utf-8"))
            self.assertEqual(archive["schema_version"], 2)
            record = archive["recommendations"][-1]
            self.assertEqual(record["decision_horizon_hours"], 96)
            self.assertEqual(record["safety_constraint"], "P10 SOC must remain at or above 40%")
            self.assertEqual(len(record["forecast_trace"]["time_utc"]), 96)
            self.assertEqual(len(record["forecast_trace"]["soc_p50_pct"]), 96)
            self.assertTrue(record["recommended_mode_windows"])
            verification = record["verification"]
            self.assertIsNotNone(verification)
            self.assertGreaterEqual(verification["coverage_hours"], 0.0)
            self.assertIn("soc_mae_pct", verification)


if __name__ == "__main__":
    unittest.main()
