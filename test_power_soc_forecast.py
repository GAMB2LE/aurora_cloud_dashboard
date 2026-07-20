from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    _apply_soc_bias_corrections,
    _extend_irradiance_with_diurnal_persistence,
    _load_mode_signature,
    _mode_learning_status,
    _pdu_active_kits,
    _repair_dc_only_registry,
    _resolve_load_mode,
    append_forecast_archive,
    build_forecast_dataset,
    build_forecast_skill_dataset,
    build_historical_load_forecast,
    build_soc_hindcast_dataset,
    evaluate_forecast_archive,
    resolve_ecmwf_cycle_hour,
    solar_irradiance_from_ssrd,
    validate_power_input_freshness,
)
from generate_power_soc_ensemble import (
    _ensemble_refresh_reasons,
    append_ensemble_archive,
    apply_operational_soc_threshold,
    build_ensemble_dataset,
    build_ensemble_skill_dataset,
)
from grouped_timeseries import (
    PDU_WATT_FIELDS,
    SUMMARY_LAYOUTS,
    _active_panels,
    _trace_plot_values,
    build_power_display_summary_dataset,
    build_power_verification_guidance,
    build_summary_plotly,
    merge_operating_scenarios_into_display_summary,
    prepare_summary_dataset,
)
from power_soc_thresholds import (
    MINIMUM_OPERATIONAL_SOC_PCT,
    MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
    SOC_BELOW_THRESHOLD_BRIER_FIELD,
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,
)


class PowerSocForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_archive_path = Path(self._tmp.name) / "power_soc_forecast_archive.zarr"
        self.tmp_ensemble_archive_path = Path(self._tmp.name) / "power_soc_ensemble_archive.zarr"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_ssrd_accumulation_converts_to_irradiance(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="3h")
        ds = xr.Dataset(
            {"ssrd": (("time",), [0.0, 3 * 3600 * 100.0, 3 * 3600 * 250.0, 3 * 3600 * 400.0])},
            coords={"time": times},
        )

        irradiance = solar_irradiance_from_ssrd(ds)

        self.assertEqual(list(irradiance.index), list(times[1:]))
        np.testing.assert_allclose(irradiance.to_numpy(), [100.0, 150.0, 150.0])

    def test_auto_long_cycle_selects_the_latest_likely_complete_cycle(self) -> None:
        self.assertEqual(resolve_ecmwf_cycle_hour("auto", now=datetime(2026, 7, 16, 9, tzinfo=timezone.utc)), 0)
        self.assertEqual(resolve_ecmwf_cycle_hour("auto", now=datetime(2026, 7, 16, 21, tzinfo=timezone.utc)), 12)
        self.assertEqual(resolve_ecmwf_cycle_hour("auto", now=datetime(2026, 7, 16, 3, tzinfo=timezone.utc)), 12)

    def test_stale_power_input_is_rejected_before_forecast_publication(self) -> None:
        latest = pd.Timestamp("2026-07-16T12:00:00")
        power = xr.Dataset(
            {"BatterySOC": (("time",), [65.0])},
            coords={"time": [latest]},
        )

        anchor_time, anchor_soc = validate_power_input_freshness(
            power,
            max_age_minutes=20,
            now=latest + pd.Timedelta(minutes=19),
        )
        self.assertEqual(anchor_time, latest)
        self.assertEqual(anchor_soc, 65.0)

        with self.assertRaisesRegex(ValueError, "stale SOC/load input"):
            validate_power_input_freshness(
                power,
                max_age_minutes=20,
                now=latest + pd.Timedelta(minutes=21),
            )

    def test_long_forecast_tail_repeats_diurnal_shape_instead_of_flatlining(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=17, freq="3h")
        values = np.tile([0.0, 20.0, 100.0, 300.0, 500.0, 250.0, 50.0, 0.0], 3)[: len(times)]
        source = pd.Series(values, index=times)

        extended, hours = _extend_irradiance_with_diurnal_persistence(
            source,
            times[-1] + pd.Timedelta(hours=6),
        )

        self.assertEqual(hours, 6.0)
        self.assertEqual(float(extended.iloc[-2]), float(source.loc[times[-1] - pd.Timedelta(hours=21)]))
        self.assertEqual(float(extended.iloc[-1]), float(source.loc[times[-1] - pd.Timedelta(hours=18)]))
        self.assertNotEqual(float(extended.iloc[-2]), float(extended.iloc[-1]))

    def test_soc_bias_correction_preserves_actual_initial_anchor(self) -> None:
        issue = pd.Timestamp("2026-07-16T06:00:00")
        times = pd.DatetimeIndex([issue, issue + pd.Timedelta(hours=3), issue + pd.Timedelta(hours=6)])
        forecast = pd.DataFrame({"BatterySOCForecast": [63.0, 62.0, 61.0]}, index=times)

        corrected = _apply_soc_bias_corrections(
            forecast,
            {"0_6h": 5.0, "6_24h": -2.0},
            issue_time=issue,
        )

        self.assertEqual(float(corrected["BatterySOCForecast"].iloc[0]), 63.0)
        self.assertEqual(float(corrected["BatterySOCForecast"].iloc[1]), 67.0)
        self.assertEqual(float(corrected["BatterySOCForecast"].iloc[2]), 59.0)

    def test_24h_forecast_panel_is_future_model_output_not_observed_soc(self) -> None:
        panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "soc_24h_forecast")

        self.assertEqual(panel.label, "SOC Next 24 h Forecast")
        self.assertEqual([trace.var for trace in panel.traces], ["BatterySOCForecast"])
        trace = panel.traces[0]
        self.assertEqual(trace.display_horizon_hours, 24.0)

        times = pd.date_range("2026-07-16T06:00:00", periods=33, freq="3h")
        values = np.linspace(60.0, 40.0, len(times))
        rendered_times, _ = _trace_plot_values(times, values, max_time_samples=100, trace=trace)

        self.assertEqual(rendered_times.min(), times.min())
        self.assertLessEqual(rendered_times.max(), times.min() + pd.Timedelta(hours=24))

    def test_mobile_summary_maps_compact_cumulative_energy_fields(self) -> None:
        times = pd.date_range("2026-07-16T00:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.0, 68.0]),
                "PowerDisplaySolarYield_East": (("time",), [0.0, 0.1, 0.2]),
                "PowerDisplaySolarYield_South": (("time",), [0.0, 0.2, 0.4]),
                "PowerDisplaySolarYield_West": (("time",), [0.0, 0.1, 0.3]),
                "PowerDisplayCumulativePowerGeneratedTotal": (("time",), [0.0, 0.4, 0.9]),
                "PowerDisplayCumulativePowerUtilised": (("time",), [0.0, 0.05, 0.1]),
            },
            coords={"time": times},
        )

        prepared = prepare_summary_dataset(display, "power")
        panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "cumulative_power")

        self.assertEqual(panel.label, "Cumulative Energy & State of Charge")
        for field_name in (
            "SolarYield_East",
            "SolarYield_South",
            "SolarYield_West",
            "CumulativePowerGeneratedTotal",
            "CumulativePowerUtilised",
        ):
            self.assertIn(field_name, prepared)
            self.assertTrue(np.isfinite(prepared[field_name].values).any())

    def test_build_forecast_dataset_integrates_soc(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=12, capacity_kwh=26.0)

        self.assertIn("BatterySOCForecast", forecast)
        self.assertIn("ECMWFSolarIrradiance", forecast)
        self.assertEqual(pd.Timestamp(forecast["time"].values[0]), power_times[-1])
        self.assertAlmostEqual(float(forecast["BatterySOCForecast"].values[0]), 70.0)
        self.assertGreaterEqual(float(forecast["BatterySOCForecast"].min()), 0.0)
        self.assertLessEqual(float(forecast["BatterySOCForecast"].max()), 100.0)
        self.assertEqual(forecast.attrs["forecast_horizon_hours"], "12")
        self.assertEqual(forecast.attrs["load_model"], "kit_mode_persistence_v4")
        self.assertEqual(forecast.attrs["load_model_version"], "4")
        self.assertEqual(float(forecast.attrs["minimum_operational_soc_pct"]), 40.0)
        self.assertEqual(forecast.attrs["scenario_loads_w"], "100,200,300,400,500,600")
        self.assertEqual(forecast.attrs["scenario_solar_mode"], "ecmwf")
        self.assertIn("ForecastLoadMAERecent", forecast)
        self.assertIn("ForecastSOCMAE_0_6h", forecast)
        self.assertIn("ForecastSkillSampleCount", forecast)
        for load_w in (100, 200, 300, 400, 500, 600):
            self.assertIn(f"BatterySOCForecast_Load{load_w}W", forecast)
            self.assertAlmostEqual(float(forecast[f"BatterySOCForecast_Load{load_w}W"].values[0]), 70.0)

    def test_load_scenarios_decline_with_higher_loads(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.zeros(len(power_times))),
                "SolarWatts_South": (("time",), np.zeros(len(power_times))),
                "SolarWatts_West": (("time",), np.zeros(len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=6, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.zeros(len(forecast_times), dtype=float))},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=15, capacity_kwh=26.0)

        previous = None
        for load_w in (100, 200, 300, 400, 500, 600):
            values = forecast[f"BatterySOCForecast_Load{load_w}W"].values
            if previous is not None:
                self.assertTrue(np.all(values <= previous + 1e-6))
            previous = values

    def test_build_forecast_dataset_supports_96h_horizon(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=34, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=96, capacity_kwh=26.0)

        self.assertEqual(forecast.attrs["forecast_horizon_hours"], "96")
        self.assertEqual(pd.Timestamp(forecast["time"].values[0]), power_times[-1])
        self.assertGreaterEqual((pd.Timestamp(forecast["time"].values[-1]) - power_times[-1]) / pd.Timedelta(hours=1), 95.0)

    def test_historical_load_forecast_persists_current_regime_without_clock_aliasing(self) -> None:
        times = pd.date_range("2026-07-01T00:00:00", periods=10 * 24 * 4, freq="15min")
        total_load = np.full(len(times), 9.0)
        total_load[(times >= times[3 * 24 * 4]) & (times < times[7 * 24 * 4])] = 650.0
        frame = pd.DataFrame(
            {
                "ACOutputWatts": total_load * 0.75,
                "DCInverterWatts": total_load * 0.25,
            },
            index=times,
        )
        forecast_times = pd.date_range(times[-1], periods=33, freq="3h")

        load = build_historical_load_forecast(frame, forecast_times, end=times[-1], calibration_days=10)

        np.testing.assert_allclose(load.to_numpy(), 9.0)
        self.assertEqual(load.attrs["load_model"], "kit_mode_persistence_v4")
        self.assertEqual(load.attrs["load_model_version"], 4)
        self.assertEqual(load.attrs["load_mode"], "DC-Only")
        self.assertEqual(load.attrs["load_regime"], "DC-Only")
        self.assertGreater(float(load.attrs["load_regime_threshold_w"]), 9.0)
        self.assertLess(float(load.attrs["load_regime_threshold_w"]), 650.0)

    def test_dc_only_mode_uses_solar_battery_power_balance(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=24 * 4, freq="15min")
        solar_total = np.zeros(len(times))
        solar_total[24:72] = np.sin(np.linspace(0.0, np.pi, 48)) * 600.0
        frame = pd.DataFrame(
            {
                "SolarWatts_East": solar_total * 0.25,
                "SolarWatts_South": solar_total * 0.45,
                "SolarWatts_West": solar_total * 0.30,
                "BatteryWatts": solar_total - 220.0,
                "ACOutputWatts": np.zeros(len(times)),
                "DCInverterWatts": np.full(len(times), 9.0),
            },
            index=times,
        )
        forecast_times = pd.date_range(times[-1], periods=9, freq="3h")

        load = build_historical_load_forecast(frame, forecast_times, end=times[-1], calibration_days=1)

        np.testing.assert_allclose(load.to_numpy(), 220.0)
        self.assertEqual(load.attrs["load_mode"], "DC-Only")
        self.assertEqual(load.attrs["load_measurement"], "battery_discharge_when_solar_zero")
        self.assertEqual(load.attrs["load_balance_measurement"], "solar_generation_minus_battery_power")

    def test_active_pdu_kit_names_the_learned_mode(self) -> None:
        power_times = pd.date_range("2026-07-15T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 100.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 100.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 100.0)),
                "BatteryWatts": (("time",), np.full(len(power_times), -200.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 300.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 9.0)),
            },
            coords={"time": power_times},
        )
        pdu_times = pd.date_range(power_times[-1] - pd.Timedelta(minutes=30), periods=3, freq="15min")
        pdu = xr.Dataset(
            {
                "PDUOutlet6State": (("time",), np.ones(len(pdu_times))),
                "PDUOutlet6Watts": (("time",), np.full(len(pdu_times), 300.0)),
            },
            coords={"time": pdu_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, pdu=pdu, horizon_hours=12, capacity_kwh=26.0)

        self.assertEqual(forecast.attrs["load_mode"], "DC-Only + Radar")
        self.assertEqual(forecast.attrs["load_mode_source"], "pdu_signature")
        self.assertEqual(forecast.attrs["load_mode_signature"], "PDUOutlet6Watts>=5W")
        self.assertEqual(forecast.attrs["load_mode_learning_ready"], "true")
        self.assertAlmostEqual(float(forecast["ForecastLoadWatts"].median()), 500.0)
        registry = forecast.attrs["load_mode_registry"]
        self.assertIn("DC-Only + Radar", registry)

    def test_powered_cl61_pdu_signature_precedes_smoothed_ac_state(self) -> None:
        end = pd.Timestamp("2026-07-16T12:00:00")
        times = pd.date_range(end - pd.Timedelta(minutes=30), end, freq="5min")
        frame = pd.DataFrame({"ACOutputWatts": [0.0] * (len(times) - 1) + [175.0]}, index=times)
        pdu_times = pd.date_range(end - pd.Timedelta(minutes=10), end, freq="5min")
        pdu = xr.Dataset(
            {
                "PDUOutlet5State": (("time",), np.ones(len(pdu_times))),
                "PDUOutlet5Watts": (("time",), np.full(len(pdu_times), 223.0)),
            },
            coords={"time": pdu_times},
        )

        mode, source, active_kits, pdu_time, active_watts = _resolve_load_mode(
            frame,
            pdu,
            end=end,
            observed_level_w=450.0,
            raw_registry={},
            previous_mode="DC-Only",
        )

        self.assertEqual(mode, "Ceilometer-on-AC")
        self.assertEqual(source, "pdu_ac_signature")
        self.assertEqual(active_kits, ["CL61"])
        self.assertEqual(pdu_time, end)
        self.assertAlmostEqual(active_watts, 223.0)

        signature = _load_mode_signature(mode, source, active_kits)
        self.assertEqual(signature, "PDUOutlet5Watts>=5W+ACOutputWatts>25W")

    def test_pdu_relay_state_without_power_is_not_an_active_kit(self) -> None:
        end = pd.Timestamp("2026-07-16T12:00:00")
        times = pd.date_range(end - pd.Timedelta(minutes=30), end, freq="15min")
        pdu = xr.Dataset(
            {
                "PDUOutlet5State": (("time",), np.ones(len(times))),
                "PDUOutlet5Watts": (("time",), np.zeros(len(times))),
            },
            coords={"time": times},
        )

        active_kits, pdu_time, active_watts = _pdu_active_kits(pdu, end=end)

        self.assertEqual(active_kits, [])
        self.assertEqual(pdu_time, end)
        self.assertTrue(np.isnan(active_watts))

    def test_mode_learning_waits_for_a_stable_run(self) -> None:
        ready, reason = _mode_learning_status(
            {
                "load_mode_state": "ac-active",
                "load_regime_run_hours": 0.25,
                "load_regime_sample_count": 4,
            },
            "DC-Only + CL61",
        )

        self.assertFalse(ready)
        self.assertEqual(reason, "waiting_for_stable_duration")

    def test_dc_only_registry_rejects_transition_level_outlier(self) -> None:
        end = pd.Timestamp("2026-07-16T12:00:00")
        times = pd.date_range(end - pd.Timedelta(hours=48), end, freq="15min")
        frame = pd.DataFrame(
            {
                "SolarWatts_East": np.zeros(len(times)),
                "SolarWatts_South": np.zeros(len(times)),
                "SolarWatts_West": np.zeros(len(times)),
                "BatteryWatts": np.full(len(times), -220.0),
                "ACOutputWatts": np.zeros(len(times)),
            },
            index=times,
        )
        frame.loc[frame.index >= end - pd.Timedelta(minutes=30), "BatteryWatts"] = -450.0
        frame.loc[frame.index >= end - pd.Timedelta(minutes=30), "ACOutputWatts"] = 175.0
        raw_registry = {
            "DC-Only": {
                "observations": [{"time": end.isoformat(), "level_w": 450.0}],
                "learned_level_w": 450.0,
            }
        }

        registry, clean_level = _repair_dc_only_registry(raw_registry, frame, end=end)

        self.assertAlmostEqual(float(clean_level), 220.0)
        self.assertAlmostEqual(float(registry["DC-Only"]["learned_level_w"]), 220.0)
        self.assertEqual(registry["DC-Only"]["observation_count"], 1)
        self.assertGreater(registry["DC-Only"]["clean_dark_sample_count"], 100)

    def test_retired_hourly_model_bias_does_not_distort_regime_forecast(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 200.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 50.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )
        previous = xr.Dataset(
            {"ForecastLoadWatts": (("time",), np.full(len(power_times), 500.0))},
            coords={"time": power_times},
        )

        forecast = build_forecast_dataset(power, solar, previous_forecast=previous, horizon_hours=12, capacity_kwh=26.0)

        self.assertEqual(float(forecast.attrs["load_bias_correction_w"]), 0.0)
        self.assertAlmostEqual(float(forecast["ForecastLoadWatts"].median()), 250.0)
        self.assertEqual(forecast.attrs["load_model"], "kit_mode_persistence_v4")
        self.assertIn("ForecastLoadBiasRecent", forecast)

    def test_stale_negative_load_bias_cannot_zero_ac_dc_load_forecast(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=48, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 25.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 25.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 25.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 220.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 180.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 100.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(
            power,
            solar,
            state={"load_bias_correction_w": -2000.0},
            horizon_hours=12,
            capacity_kwh=26.0,
        )

        self.assertGreater(float(forecast["ForecastLoadWatts"].median()), 0.0)
        self.assertGreaterEqual(float(forecast["ForecastLoadWatts"].median()), 300.0)
        self.assertGreater(float(forecast.attrs["forecast_load_w"]), 0.0)

    def test_archive_skill_scores_lead_buckets(self) -> None:
        issue_time = pd.Timestamp("2026-07-10T00:00:00")
        times = [
            issue_time,
            issue_time + pd.Timedelta(hours=3),
            issue_time + pd.Timedelta(hours=12),
            issue_time + pd.Timedelta(hours=18),
            issue_time + pd.Timedelta(hours=30),
            issue_time + pd.Timedelta(hours=42),
            issue_time + pd.Timedelta(hours=60),
            issue_time + pd.Timedelta(hours=84),
        ]
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [50.0, 48.0, 45.0, 44.0, 40.0, 38.0, 35.0, 30.0]),
                "ForecastSolarWatts": (("time",), [0.0, 10.0, 20.0, 22.0, 30.0, 35.0, 40.0, 45.0]),
                "ForecastLoadWatts": (("time",), np.full(8, 100.0)),
            },
            coords={"time": times},
            attrs={"initial_soc_time": issue_time.isoformat()},
        )
        archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [49.0, 46.0, 42.0, 41.0, 38.0, 35.0, 30.0, 25.0]),
                "SolarWatts_East": (("time",), [0.0, 12.0, 18.0, 21.0, 31.0, 34.0, 39.0, 44.0]),
                "ACOutputWatts": (("time",), np.full(8, 95.0)),
                "DCInverterWatts": (("time",), np.full(8, 5.0)),
            },
            coords={"time": forecast["time"].values},
        )

        metrics = evaluate_forecast_archive(archive, pd.DataFrame({name: power[name].values for name in power.data_vars}, index=pd.DatetimeIndex(power["time"].values)))

        self.assertIn("soc_mae_0_6h", metrics)
        self.assertIn("soc_mae_6_24h", metrics)
        self.assertIn("soc_mae_24_48h", metrics)
        self.assertIn("soc_mae_48_96h", metrics)

    def test_forecast_skill_dataset_is_past_facing(self) -> None:
        issue_time = pd.Timestamp("2026-07-10T00:00:00")
        forecast_times = pd.date_range(issue_time, periods=5, freq="3h")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [50.0, 49.0, 48.0, 47.0, 46.0]),
                "ForecastSolarWatts": (("time",), [0.0, 10.0, 20.0, 10.0, 0.0]),
                "ForecastLoadWatts": (("time",), np.full(5, 120.0)),
            },
            coords={"time": forecast_times},
            attrs={"initial_soc_time": issue_time.isoformat()},
        )
        archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [50.0, 48.5, 47.5, 46.0, 45.0]),
                "SolarWatts_East": (("time",), [0.0, 11.0, 18.0, 9.0, 0.0]),
                "ACOutputWatts": (("time",), np.full(5, 100.0)),
                "DCInverterWatts": (("time",), np.full(5, 10.0)),
            },
            coords={"time": forecast_times},
        )

        skill = build_forecast_skill_dataset(archive, power, window_hours=24, retention_days=1, freq="3h")

        self.assertIn("ForecastSOCMAE_0_6h_Verified", skill)
        self.assertIn("ForecastLoadMAE24h", skill)
        self.assertIn("ForecastSolarMAE24h", skill)
        self.assertIn("ForecastSOCSkill_0_6h", skill)
        self.assertLessEqual(pd.Timestamp(skill["time"].values[-1]), forecast_times[-1])
        self.assertTrue(np.isfinite(skill["ForecastSOCMAE_0_6h_Verified"].values).any())

    def test_load_verification_only_scores_current_model_version(self) -> None:
        old_issue = pd.Timestamp("2026-07-10T00:00:00")
        old_times = pd.DatetimeIndex([old_issue, old_issue + pd.Timedelta(hours=3)])
        old_forecast = xr.Dataset(
            {"ForecastLoadWatts": (("time",), [500.0, 500.0])},
            coords={"time": old_times},
            attrs={
                "initial_soc_time": old_issue.isoformat(),
                "ecmwf_cycle_time": old_issue.isoformat(),
                "load_model_version": "0",
            },
        )
        append_forecast_archive(old_forecast, self.tmp_archive_path)

        new_issue = pd.Timestamp("2026-07-10T06:00:00")
        new_times = pd.DatetimeIndex([new_issue, new_issue + pd.Timedelta(hours=3)])
        new_forecast = xr.Dataset(
            {"ForecastLoadWatts": (("time",), [110.0, 110.0])},
            coords={"time": new_times},
            attrs={
                "initial_soc_time": new_issue.isoformat(),
                "ecmwf_cycle_time": new_issue.isoformat(),
                "load_model_version": "4",
            },
        )
        archive = append_forecast_archive(new_forecast, self.tmp_archive_path)
        power_times = pd.date_range(old_issue, new_times[-1], freq="15min")
        power = xr.Dataset(
            {
                "ACOutputWatts": (("time",), np.full(len(power_times), 90.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        frame = pd.DataFrame(
            {name: power[name].values for name in power.data_vars},
            index=pd.DatetimeIndex(power["time"].values),
        )

        metrics = evaluate_forecast_archive(archive, frame)
        skill = build_forecast_skill_dataset(archive, power, window_hours=24, retention_days=1, freq="1h")

        self.assertAlmostEqual(float(metrics["load_mae_w"]), 10.0)
        finite_mae = skill["ForecastLoadMAE24h"].dropna("time")
        self.assertTrue(len(finite_mae))
        self.assertAlmostEqual(float(finite_mae.values[-1]), 10.0)
        self.assertEqual(skill.attrs["load_model_version"], "4")

    def test_hindcast_selects_fixed_lead_forecasts(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        forecast_times = pd.DatetimeIndex([issue, issue + pd.Timedelta(hours=6), issue + pd.Timedelta(hours=24)])
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [60.0, 58.0, 50.0]),
                "ForecastSolarWatts": (("time",), [0.0, 10.0, 0.0]),
                "ForecastLoadWatts": (("time",), [100.0, 100.0, 100.0]),
            },
            coords={"time": forecast_times},
            attrs={"initial_soc_time": issue.isoformat(), "ecmwf_cycle_time": issue.isoformat()},
        )
        archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power_times = pd.date_range(issue, issue + pd.Timedelta(hours=24), freq="15min")
        power = xr.Dataset(
            {"BatterySOC": (("time",), np.linspace(60.0, 49.0, len(power_times)))},
            coords={"time": power_times},
        )

        hindcast = build_soc_hindcast_dataset(archive, power, retention_days=2)

        self.assertIn("BatterySOCObservedHindcast", hindcast)
        self.assertIn("BatterySOCHindcast_6h", hindcast)
        self.assertIn("BatterySOCHindcast_24h", hindcast)
        self.assertAlmostEqual(float(hindcast["BatterySOCHindcast_6h"].max(skipna=True)), 58.0)

    def test_skill_counts_independent_ecmwf_cycles(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        archive = None
        for minute in (0, 15):
            run_issue = issue + pd.Timedelta(minutes=minute)
            forecast_times = pd.DatetimeIndex([run_issue, issue + pd.Timedelta(hours=3)])
            forecast = xr.Dataset(
                {
                    "BatterySOCForecast": (("time",), [60.0, 58.0]),
                    "ForecastSolarWatts": (("time",), [0.0, 10.0]),
                    "ForecastLoadWatts": (("time",), [100.0, 100.0]),
                },
                coords={"time": forecast_times},
                attrs={"initial_soc_time": run_issue.isoformat(), "ecmwf_cycle_time": issue.isoformat()},
            )
            archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power_times = pd.date_range(issue, issue + pd.Timedelta(hours=3), freq="15min")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(60.0, 57.0, len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 90.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )

        skill = build_forecast_skill_dataset(archive, power, window_hours=24, retention_days=1, freq="1h")

        finite = skill["ForecastIndependentCycles"].dropna("time")
        self.assertTrue(len(finite))
        self.assertEqual(float(finite.values[-1]), 1.0)

    def test_build_ensemble_starts_every_member_at_actual_soc(self) -> None:
        power_times = pd.date_range("2026-07-10T00:00:00", periods=49, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(70.0, 66.0, len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 100.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 20.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=6, freq="3h")
        accumulated = np.stack(
            [np.arange(len(forecast_times), dtype=float) * 3 * 3600 * value for value in (100.0, 200.0, 300.0)]
        )
        solar = xr.Dataset(
            {"ssrd": (("number", "time"), accumulated)},
            coords={"number": [1, 2, 3], "time": forecast_times},
        )
        deterministic = xr.Dataset(
            attrs={
                "solar_calibration_factor_w_per_wm2": "1.0",
                "battery_capacity_kwh": "26",
                "load_bias_correction_w": "0",
                "forecast_load_w": "455.15",
                "load_model": "kit_mode_persistence_v4",
                "load_model_version": "4",
                "load_mode": "DC-Only + CL61",
                "load_mode_source": "pdu_signature",
                "load_mode_active_kits": "CL61",
                "load_mode_signature": "PDUOutlet5Watts>=5W",
            }
        )

        ensemble = build_ensemble_dataset(power, deterministic, solar, horizon_hours=15)

        np.testing.assert_allclose(ensemble["BatterySOCForecastEnsemble"].values[:, 0], 66.0)
        ensemble_loads = ensemble["ForecastLoadWattsEnsemble"].values
        np.testing.assert_allclose(
            ensemble_loads,
            np.repeat(ensemble_loads[[0], :], ensemble_loads.shape[0], axis=0),
        )
        self.assertTrue(np.all(ensemble["BatterySOCForecastP10"] <= ensemble["BatterySOCForecastP50"]))
        self.assertTrue(np.all(ensemble["BatterySOCForecastP50"] <= ensemble["BatterySOCForecastP90"]))
        self.assertTrue(
            np.all(
                (ensemble[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD] >= 0)
                & (ensemble[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD] <= 1)
            )
        )
        self.assertEqual(
            float(ensemble.attrs["minimum_operational_soc_pct"]),
            MINIMUM_OPERATIONAL_SOC_PCT,
        )
        self.assertEqual(ensemble.attrs["load_model"], "kit_mode_persistence_v4")
        self.assertEqual(ensemble.attrs["load_model_version"], "4")
        self.assertEqual(ensemble.attrs["load_mode"], "DC-Only + CL61")
        self.assertEqual(ensemble.attrs["load_mode_signature"], "PDUOutlet5Watts>=5W")
        self.assertEqual(float(ensemble.attrs["forecast_load_w"]), 455.15)
        self.assertEqual(ensemble.attrs["scenario_scope"], "current_system_only")
        self.assertEqual(
            ensemble.attrs["load_uncertainty"],
            "fixed current-system load; ECMWF solar ensemble only",
        )

    def test_ensemble_reanchors_when_soc_or_load_mode_changes_within_same_cycle(self) -> None:
        deterministic_attrs = {
            "initial_soc_time": "2026-07-16T12:29:34",
            "initial_soc_pct": "57",
            "solar_calibration_factor_w_per_wm2": "1.82",
            "battery_capacity_kwh": "26",
            "load_bias_correction_w": "0",
            "forecast_load_w": "455.15",
            "load_model": "kit_mode_persistence_v4",
            "load_model_version": "4",
            "load_mode": "DC-Only + CL61",
            "load_mode_source": "pdu_signature",
            "load_mode_active_kits": "CL61",
            "load_mode_signature": "PDUOutlet5Watts>=5W",
        }
        matching_attrs = dict(deterministic_attrs)

        self.assertEqual(_ensemble_refresh_reasons(matching_attrs, deterministic_attrs), [])

        stale_attrs = dict(matching_attrs)
        stale_attrs.update(
            {
                "initial_soc_time": "2026-07-16T08:36:12",
                "initial_soc_pct": "61",
                "forecast_load_w": "223",
                "load_model_version": "3",
                "load_mode": "DC-Only",
            }
        )
        reasons = _ensemble_refresh_reasons(stale_attrs, deterministic_attrs)

        self.assertIn("initial_soc_time", reasons)
        self.assertIn("initial_soc_pct", reasons)
        self.assertIn("forecast_load_w", reasons)
        self.assertIn("load_model_version", reasons)
        self.assertIn("load_mode", reasons)

    def test_operational_threshold_refresh_replaces_legacy_probability(self) -> None:
        ensemble = xr.Dataset(
            {
                "BatterySOCForecastEnsemble": (
                    ("member", "time"),
                    [[50.0, 35.0], [45.0, 30.0]],
                ),
                "BatterySOCBelow20Probability": (("time",), [0.0, 0.0]),
            },
            coords={"member": [1, 2], "time": pd.date_range("2026-07-10", periods=2, freq="3h")},
        )

        refreshed = apply_operational_soc_threshold(ensemble)

        self.assertNotIn("BatterySOCBelow20Probability", refreshed)
        np.testing.assert_allclose(refreshed[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD], [0.0, 1.0])
        self.assertEqual(
            float(refreshed.attrs["minimum_operational_soc_pct"]),
            MINIMUM_OPERATIONAL_SOC_PCT,
        )

    def test_ensemble_archive_produces_probabilistic_skill(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        times = pd.date_range(issue, periods=5, freq="3h")
        members = np.array(
            [
                [60.0, 58.0, 56.0, 54.0, 52.0],
                [60.0, 59.0, 57.0, 55.0, 53.0],
                [60.0, 57.0, 55.0, 53.0, 51.0],
            ],
            dtype=np.float32,
        )
        forecast = xr.Dataset(
            {"BatterySOCForecastEnsemble": (("member", "time"), members)},
            coords={"member": [1, 2, 3], "time": times},
            attrs={"initial_soc_time": issue.isoformat()},
        )
        archive = append_ensemble_archive(forecast, self.tmp_ensemble_archive_path)
        power_times = pd.date_range(issue, times[-1], freq="15min")
        power = xr.Dataset(
            {"BatterySOC": (("time",), np.linspace(60.0, 52.0, len(power_times)))},
            coords={"time": power_times},
        )

        skill = build_ensemble_skill_dataset(archive, power, retention_days=1)

        self.assertIn("ForecastSOCCRPS_0_6h", skill)
        self.assertIn("ForecastSOCIntervalCoverage80", skill)
        self.assertIn(SOC_BELOW_THRESHOLD_BRIER_FIELD, skill)
        self.assertIn("ForecastSOCCRPSSamples_0_6h", skill)
        self.assertIn("ForecastSOCCRPSCycles_0_6h", skill)
        self.assertIn("ForecastSOCCRPSSkill_0_6h", skill)
        self.assertNotIn("ForecastSOCBelow20Brier", skill)
        self.assertTrue(np.isfinite(skill["ForecastSOCCRPS_0_6h"].values).any())
        finite_coverage = skill["ForecastSOCIntervalCoverage80"].values
        finite_coverage = finite_coverage[np.isfinite(finite_coverage)]
        self.assertTrue(np.all((finite_coverage >= 0.0) & (finite_coverage <= 1.0)))

    def test_ensemble_guidance_marks_immature_long_range_scores_not_verified(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=2, freq="1h")
        summary = xr.Dataset(
            {
                "ForecastSOCCRPS_0_6h": (("time",), [1.0, 1.2]),
                "ForecastSOCCRPSSamples_0_6h": (("time",), [24.0, 24.0]),
                "ForecastSOCCRPSCycles_0_6h": (("time",), [12.0, 12.0]),
                "ForecastSOCCRPSSkill_0_6h": (("time",), [0.1, 0.1]),
                "ForecastSOCIntervalCoverage80": (("time",), [0.8, 0.8]),
                "ForecastSOCIntervalCoverage80Samples": (("time",), [24.0, 24.0]),
                "ForecastSOCIntervalCoverage80Cycles": (("time",), [12.0, 12.0]),
                SOC_BELOW_THRESHOLD_BRIER_FIELD: (("time",), [0.05, 0.05]),
                f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Samples": (("time",), [24.0, 24.0]),
                f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Cycles": (("time",), [12.0, 12.0]),
            },
            coords={"time": times},
        )

        guidance = build_power_verification_guidance("soc_ensemble_skill", summary)

        self.assertIsNotNone(guidance)
        metrics = {metric["id"]: metric for metric in guidance["metrics"]}
        self.assertEqual(metrics["soc-crps-0_6h"]["status"], "Better than persistence")
        self.assertEqual(metrics["soc-crps-48_96h"]["valueText"], "Not yet verified")
        self.assertEqual(metrics["soc-coverage"]["status"], "Consistent with 80% target")

    def test_display_summary_merges_forecast_fields(self) -> None:
        power_times = pd.date_range("2026-07-10T00:00:00", periods=3, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.5, 69.0]),
                "ACOutputWatts": (("time",), [10.0, 10.0, 10.0]),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range("2026-07-10T03:00:00", periods=3, freq="3h")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [68.0, 67.0, 66.0]),
                "BatterySOCForecast_Load100W": (("time",), [68.0, 67.5, 67.0]),
                "BatterySOCForecast_Load600W": (("time",), [68.0, 63.0, 58.0]),
                "ECMWFSolarIrradiance": (("time",), [100.0, 200.0, 100.0]),
                "ForecastLoadWatts": (("time",), [220.0, 220.0, 220.0]),
            },
            coords={"time": forecast_times},
            attrs={
                "load_mode": "DC-Only",
                "load_model": "kit_mode_persistence_v4",
                "load_model_version": "4",
                "load_mode_source": "ac_output",
                "load_mode_active_kits": "",
                "load_mode_signature": "ACOutputWatts<=25W",
                "load_mode_learning_ready": "true",
                "load_mode_learning_reason": "stable",
                "load_mode_learning_observations": "2",
                "load_mode_pdu_active_watts": "nan",
                "load_measurement": "battery_discharge_when_solar_zero",
                "load_balance_measurement": "solar_generation_minus_battery_power",
            },
        )
        skill = xr.Dataset(
            {
                "ForecastSOCMAE_0_6h_Verified": (("time",), [1.0, 1.5, 2.0]),
                "ForecastLoadMAE24h": (("time",), [10.0, 11.0, 12.0]),
                "ForecastSolarMAE24h": (("time",), [20.0, 21.0, 22.0]),
            },
            coords={"time": power_times},
        )
        hindcast = xr.Dataset(
            {
                "BatterySOCObservedHindcast": (("time",), [70.0, 69.5, 69.0]),
                "BatterySOCHindcast_6h": (("time",), [71.0, 70.0, 68.0]),
            },
            coords={"time": power_times},
        )
        ensemble = xr.Dataset(
            {
                "BatterySOCForecastP10": (("time",), [65.0, 60.0, 55.0]),
                "BatterySOCForecastP90": (("time",), [72.0, 75.0, 78.0]),
                SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: (("time",), [0.0, 0.0, 0.1]),
            },
            coords={"time": forecast_times},
        )
        operating = xr.Dataset(
            {
                "ScenarioSOCP10": (("scenario", "time"), [[67.0, 65.0, 63.0], [68.0, 67.0, 66.0], [66.0, 61.0, 56.0], [67.0, 66.0, 64.0], [66.0, 63.0, 60.0]]),
                "ScenarioSOCP50": (("scenario", "time"), [[68.0, 66.0, 64.0], [69.0, 68.0, 67.0], [67.0, 62.0, 57.0], [68.0, 67.0, 65.0], [67.0, 64.0, 61.0]]),
                "ScenarioSOCP90": (("scenario", "time"), [[69.0, 67.0, 65.0], [70.0, 69.0, 68.0], [68.0, 63.0, 58.0], [69.0, 68.0, 66.0], [68.0, 65.0, 62.0]]),
                "ScenarioLoadP50Watts": (("scenario", "time"), np.full((5, 3), 220.0)),
                "ScenarioBelow40Probability": (("scenario", "time"), np.zeros((5, 3))),
                "ScenarioModeCode": (("scenario", "time"), [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1], [2, 2, 2]]),
                "SolarP50Watts": (("time",), [100.0, 200.0, 100.0]),
                "scenario_label": (("scenario",), ["Current Mode", "DC-Only", "DC + CL61", "Optimized CL61", "DC + Radar"]),
            },
            coords={
                "scenario": ["current_mode", "dc_only", "cl61_continuous", "optimized_cl61", "learned_dc_radar"],
                "time": forecast_times,
            },
            attrs={
                "current_mode": "dc_cl61",
                "current_mode_label": "DC + CL61",
                "current_mode_confidence": "0.98",
                "model": "hybrid_state_space_v5",
                "model_version": "5",
            },
        )

        summary = build_power_display_summary_dataset(
            power,
            forecast_ds=forecast,
            forecast_skill_ds=skill,
            hindcast_ds=hindcast,
            ensemble_forecast_ds=ensemble,
            operating_scenarios_ds=operating,
            freq="1h",
        )

        self.assertIn("BatterySOCForecast", summary)
        self.assertIn("BatterySOCForecast_Load100W", summary)
        self.assertIn("BatterySOCForecast_Load600W", summary)
        self.assertIn("ECMWFSolarIrradiance", summary)
        self.assertIn("ForecastSOCMAE_0_6h_Verified", summary)
        self.assertIn("ForecastLoadMAE24h", summary)
        self.assertIn("ForecastSolarMAE24h", summary)
        self.assertIn("BatterySOCHindcast_6h", summary)
        self.assertIn("BatterySOCForecastP10", summary)
        self.assertIn(SOC_BELOW_THRESHOLD_PROBABILITY_FIELD, summary)
        self.assertIn("OperatingDCOnlySOCP50", summary)
        self.assertIn("OperatingCL61OptimizedSOCP10", summary)
        self.assertIn("OperatingCL61OptimizedModeCode", summary)
        self.assertIn("OperatingLearned1SOCP50", summary)
        self.assertEqual(summary.attrs["operating_learned_1_label"], "DC + Radar")
        self.assertEqual(summary.attrs["operating_current_mode_label"], "DC + CL61")
        self.assertEqual(summary.attrs["forecast_load_mode"], "DC-Only")
        self.assertEqual(summary.attrs["forecast_load_model"], "kit_mode_persistence_v4")
        self.assertEqual(summary.attrs["forecast_load_mode_signature"], "ACOutputWatts<=25W")
        self.assertEqual(summary.attrs["forecast_load_mode_learning_ready"], "true")
        self.assertEqual(summary.attrs["forecast_load_measurement"], "battery_discharge_when_solar_zero")
        self.assertEqual(summary.attrs["forecast_load_balance_measurement"], "solar_generation_minus_battery_power")
        self.assertEqual(float(summary.attrs["minimum_operational_soc_pct"]), 40.0)
        self.assertGreater(summary.sizes["time"], power.sizes["time"])

        merged = merge_operating_scenarios_into_display_summary(power, operating)
        self.assertIn("OperatingLearned1SOCP50", merged)
        self.assertEqual(merged.attrs["operating_learned_1_label"], "DC + Radar")
        self.assertEqual(pd.Timestamp(merged["time"].values[-1]), forecast_times[-1])

    def test_all_soc_decision_panels_draw_40_percent_operational_minimum(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="3h")
        ds = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [60.0, 55.0, 50.0, 45.0]),
                "BatterySOCForecastP10": (("time",), [58.0, 50.0, 43.0, 35.0]),
                "BatterySOCForecastP90": (("time",), [62.0, 60.0, 57.0, 52.0]),
                SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: (("time",), [0.0, 0.0, 0.0, 0.5]),
                "BatterySOCObservedHindcast": (("time",), [65.0, 60.0, 55.0, 50.0]),
                "BatterySOCHindcast_6h": (("time",), [64.0, 59.0, 54.0, 49.0]),
                "OperatingDCOnlySOCP50": (("time",), [60.0, 58.0, 56.0, 54.0]),
                "OperatingCL61ContinuousSOCP50": (("time",), [60.0, 50.0, 40.0, 30.0]),
                "OperatingCL61OptimizedSOCP50": (("time",), [60.0, 57.0, 53.0, 48.0]),
                "OperatingCL61OptimizedSOCP10": (("time",), [58.0, 54.0, 49.0, 42.0]),
            },
            coords={"time": times},
        )

        figure = build_summary_plotly(ds, "power")

        references = [trace for trace in figure.data if trace.name == MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL]
        self.assertEqual(len(references), 4)
        for trace in references:
            np.testing.assert_allclose(trace.y, MINIMUM_OPERATIONAL_SOC_PCT)

    def test_unavailable_operating_product_removes_baked_stale_recommendations(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {"OperatingCL61OptimizedSOCP50": (("time",), [80.0, 70.0, 60.0]), "BatterySOC": (("time",), [90.0, 89.0, 88.0])},
            coords={"time": times},
        )
        unavailable = xr.Dataset(
            coords={"scenario": np.asarray([], dtype=str), "time": np.asarray([], dtype="datetime64[ns]")},
            attrs={"planning_status": "unavailable", "planning_status_reason": "SOC anchor mismatch"},
        )

        merged = merge_operating_scenarios_into_display_summary(display, unavailable)

        self.assertNotIn("OperatingCL61OptimizedSOCP50", merged)
        self.assertEqual(merged.attrs["operating_planning_status"], "unavailable")
        self.assertEqual(merged.attrs["operating_planning_status_reason"], "SOC anchor mismatch")

    def test_only_assigned_pdu_outlet_loads_are_displayed(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="15min")
        ds = xr.Dataset(
            {field: (("time",), np.zeros(len(times))) for field in PDU_WATT_FIELDS},
            coords={"time": times},
        )

        panels = _active_panels(ds, "power")
        pdu_rows = next(rows for panel, rows in panels if panel.key == "pdu_outlet_power")
        figure = build_summary_plotly(ds, "power")

        self.assertEqual(len(pdu_rows), 4)
        self.assertTrue(all(np.allclose(values, 0.0) for _trace, values in pdu_rows))
        outlet_names = {trace.name for trace in figure.data}
        self.assertEqual({"UAS", "CL61", "Radar", "HATPRO"} & outlet_names, {"UAS", "CL61", "Radar", "HATPRO"})
        self.assertFalse({"Outlet 1", "Outlet 2", "Outlet 3", "Outlet 7"} & outlet_names)


if __name__ == "__main__":
    unittest.main()
