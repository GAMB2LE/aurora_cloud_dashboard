from __future__ import annotations

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
    SCENARIO_DC_ONLY,
    SCENARIO_OPTIMIZED,
    build_operating_scenarios,
    evaluate_custom_schedule,
    fit_operating_model,
    mode_from_code,
    mode_id,
    mode_kits,
    optimize_cl61_schedule,
)
from generate_power_operating_scenarios import (
    _validate_operating_inputs,
    generate as generate_operating_products,
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
            "PDUOutlet5Watts": (("time",), np.where(cl61_active, 220.0, 0.0)),
            "PDUOutlet5State": (("time",), cl61_active.astype(float)),
        },
        coords={"time": times},
    )
    return power, pdu


def _forecast_inputs(issue: pd.Timestamp, horizon_hours: int = 96) -> tuple[xr.Dataset, xr.Dataset]:
    times = pd.date_range(issue, issue + pd.Timedelta(hours=horizon_hours), freq="3h")
    solar = np.full(len(times), 520.0, dtype=float)
    deterministic = xr.Dataset(
        {"ForecastSolarWatts": (("time",), solar)},
        coords={"time": times},
        attrs={"battery_capacity_kwh": "26"},
    )
    members = np.vstack([solar * factor for factor in np.linspace(0.75, 1.25, 20)])
    ensemble = xr.Dataset(
        {"ForecastSolarWattsEnsemble": (("member", "time"), members)},
        coords={"member": np.arange(1, 21), "time": times},
    )
    return deterministic, ensemble


class OperatingScenarioTests(unittest.TestCase):
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
        self.assertAlmostEqual(result.component_mean[COMPONENT_INDEX["DC"]], 200.0, delta=25.0)
        self.assertAlmostEqual(result.component_mean[COMPONENT_INDEX["CL61"]], 220.0, delta=20.0)
        probabilities = result.state_dataset["OperatingModeProbability"].values
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_saved_state_does_not_retrain_the_same_observations(self) -> None:
        power, pdu = _training_data()
        first = fit_operating_model(power, pdu, lookback_days=2)

        second = fit_operating_model(power, pdu, raw_state=first.state, lookback_days=2)

        np.testing.assert_allclose(second.component_mean, first.component_mean, atol=1e-9)
        np.testing.assert_allclose(second.component_covariance, first.component_covariance, atol=1e-9)
        self.assertEqual(second.state["new_observation_count"], 0)

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

        with self.assertRaisesRegex(ValueError, "expired planning forecast"):
            _validate_operating_inputs(
                power,
                deterministic,
                planning_hours=240,
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
                self.assertEqual(state.attrs["model_version"], "5")
                self.assertEqual(scenarios.attrs["control_authority"], "advisory_only")
                self.assertIn("optimized_cl61", set(str(value) for value in scenarios["scenario"].values))
            finally:
                state.close()
                scenarios.close()
            self.assertTrue(paths["model"].exists())
            self.assertTrue(paths["recommendations"].exists())


if __name__ == "__main__":
    unittest.main()
