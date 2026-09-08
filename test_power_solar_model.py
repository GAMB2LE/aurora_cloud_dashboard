from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from power_solar_model import (
    PVArrayConfig,
    PhysicalSolarConfig,
    build_physical_solar_forecast,
    build_physical_solar_forecast_frames,
    load_physical_solar_config,
    physical_solar_config_digest,
    physical_solar_contract_id,
    solar_position,
)


SITE_LATITUDE = 64.829694
SITE_LONGITUDE = -23.248139


def config_for(*arrays: PVArrayConfig, status: str = "surveyed") -> PhysicalSolarConfig:
    return PhysicalSolarConfig(
        schema_version=1,
        configuration_status=status,
        source="unit-test fixture",
        arrays=tuple(arrays),
        ground_albedo=0.2,
        substep_minutes=10.0,
    ).validated()


class PhysicalSolarModelTests(unittest.TestCase):
    def test_noaa_geometry_is_southerly_near_local_solar_noon(self) -> None:
        # At 23.25 W, local solar noon is roughly 13:30 UTC.
        result = solar_position(
            pd.DatetimeIndex(["2026-06-21T13:30:00"]),
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
        ).iloc[0]

        self.assertLess(abs(float(result["SolarAzimuthDegrees"]) - 180.0), 5.0)
        self.assertGreater(float(result["SolarCosineZenith"]), 0.65)

    def test_horizontal_plane_recovers_interval_ghi_without_iam_loss(self) -> None:
        horizontal = PVArrayConfig(
            name="Horizontal",
            azimuth_deg=180.0,
            tilt_deg=0.0,
            nameplate_power_w=1000.0,
            controller_limit_w=5000.0,
            fixed_efficiency=1.0,
            temperature_coefficient_per_c=0.0,
            incidence_angle_modifier_b0=0.0,
        )
        times = pd.DatetimeIndex(["2026-06-21T12:00:00", "2026-06-21T15:00:00"])
        ghi = pd.Series([500.0, 700.0], index=times)

        result, metadata = build_physical_solar_forecast(
            ghi,
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(horizontal),
        )

        np.testing.assert_allclose(
            result["ForecastPlaneOfArrayIrradianceHorizontal"],
            ghi,
            rtol=0.0,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(result["ECMWFSolarIrradiance"], ghi)
        self.assertIn("direct_diffuse_erbs", metadata["solar_degradation_codes"])
        self.assertEqual(metadata["solar_interval_energy_conservation"], "ecmwf_ghi_exact_by_interval")

    def test_partial_first_interval_is_truncated_at_exact_issue_time(self) -> None:
        horizontal = PVArrayConfig(
            name="Horizontal",
            azimuth_deg=180.0,
            tilt_deg=0.0,
            nameplate_power_w=1000.0,
            controller_limit_w=5000.0,
            fixed_efficiency=1.0,
            temperature_coefficient_per_c=0.0,
            incidence_angle_modifier_b0=0.0,
        )
        times = pd.DatetimeIndex(["2026-08-28T09:00:00", "2026-08-28T12:00:00"])
        ghi = pd.Series([200.0, 300.0], index=times)
        issue_time = pd.Timestamp("2026-08-28T08:03:00")

        full_intervals, _full_substeps, _ = build_physical_solar_forecast_frames(
            ghi,
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(horizontal),
        )
        intervals, substeps, metadata = build_physical_solar_forecast_frames(
            ghi,
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(horizontal),
            forecast_start_time=issue_time,
        )

        first_end = times[0]
        first_substeps = substeps.loc[substeps.index <= first_end]
        weights = first_substeps["SolarIntervalHours"].to_numpy(dtype=np.float64)
        weighted_available = np.average(
            first_substeps["ForecastPVAvailableWattsHorizontal"].to_numpy(dtype=np.float64),
            weights=weights,
        )
        retained_ghi_energy = float(
            np.sum(
                first_substeps["ECMWFSolarIrradiance"].to_numpy(dtype=np.float64)
                * weights
            )
        )
        self.assertAlmostEqual(float(intervals.loc[first_end, "SolarIntervalHours"]), 0.95)
        self.assertAlmostEqual(float(intervals.loc[first_end, "ECMWFSourceIntervalHours"]), 3.0)
        self.assertAlmostEqual(float(weights.sum()), 0.95)
        self.assertGreater(substeps.index.min(), issue_time)
        self.assertAlmostEqual(
            float(
                intervals.loc[first_end, "ForecastEffectiveGlobalHorizontalIrradiance"]
                * intervals.loc[first_end, "SolarIntervalHours"]
            ),
            retained_ghi_energy,
        )
        self.assertAlmostEqual(
            float(intervals.loc[first_end, "ForecastPVAvailableWattsHorizontal"]),
            float(weighted_available),
        )
        self.assertGreater(
            float(intervals.loc[first_end, "ForecastPVAvailableWattsHorizontal"]),
            float(full_intervals.loc[first_end, "ForecastPVAvailableWattsHorizontal"]),
        )
        self.assertEqual(
            metadata["solar_soc_integration_resolution"],
            "physical_substeps_with_exact_issue_time_cutoff",
        )

    def test_anchor_at_interval_endpoint_has_no_previous_interval_effective_pv(self) -> None:
        horizontal = PVArrayConfig(
            "Horizontal",
            180.0,
            0.0,
            1000.0,
            5000.0,
            fixed_efficiency=1.0,
            temperature_coefficient_per_c=0.0,
            incidence_angle_modifier_b0=0.0,
        )
        times = pd.DatetimeIndex(["2026-08-28T09:00:00", "2026-08-28T12:00:00"])
        intervals, substeps, _ = build_physical_solar_forecast_frames(
            pd.Series([200.0, 300.0], index=times),
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(horizontal),
            forecast_start_time=times[0],
        )

        self.assertEqual(float(intervals.loc[times[0], "SolarIntervalHours"]), 0.0)
        self.assertEqual(float(intervals.loc[times[0], "ECMWFSolarIrradiance"]), 200.0)
        self.assertEqual(float(intervals.loc[times[0], "ECMWFSourceIntervalHours"]), 3.0)
        self.assertTrue(
            np.isnan(
                float(
                    intervals.loc[
                        times[0], "ForecastEffectiveGlobalHorizontalIrradiance"
                    ]
                )
            )
        )
        self.assertTrue(
            np.isnan(float(intervals.loc[times[0], "ForecastPVAvailableWattsHorizontal"]))
        )
        self.assertGreater(substeps.index.min(), times[0])

    def test_raw_and_iam_adjusted_poa_have_explicit_semantics(self) -> None:
        array = PVArrayConfig(
            "South",
            180.0,
            65.0,
            1000.0,
            5000.0,
            fixed_efficiency=1.0,
            temperature_coefficient_per_c=0.0,
            incidence_angle_modifier_b0=0.10,
        )
        times = pd.DatetimeIndex(["2026-06-21T09:00:00", "2026-06-21T12:00:00"])
        result, _ = build_physical_solar_forecast(
            pd.Series([500.0, 500.0], index=times),
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(array),
            direct_horizontal_w_m2=pd.Series([450.0, 450.0], index=times),
        )

        raw = result["ForecastPlaneOfArrayIrradianceSouth"]
        raw_components = (
            result["ForecastPlaneOfArrayDirectIrradianceSouth"]
            + result["ForecastPlaneOfArrayDiffuseIrradianceSouth"]
        )
        effective = result["ForecastEffectivePlaneOfArrayIrradianceSouth"]
        np.testing.assert_allclose(raw, raw_components)
        self.assertTrue(np.all(effective <= raw + 1.0e-10))
        self.assertTrue(np.any(effective < raw - 1.0))
        np.testing.assert_allclose(result["ForecastPVAvailableWattsSouth"], effective)

    def test_east_and_west_arrays_follow_solar_azimuth(self) -> None:
        east = PVArrayConfig("East", 90.0, 65.0, 1000.0, 2000.0, 1.0, 0.0, 0.0)
        west = PVArrayConfig("West", 270.0, 65.0, 1000.0, 2000.0, 1.0, 0.0, 0.0)
        times = pd.DatetimeIndex(
            [
                "2026-06-21T09:00:00",
                "2026-06-21T12:00:00",
                "2026-06-21T18:00:00",
                "2026-06-21T21:00:00",
            ]
        )
        result, _ = build_physical_solar_forecast(
            pd.Series(500.0, index=times),
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(east, west),
        )

        self.assertGreater(
            float(result.loc[times[0], "ForecastPVAvailableWattsEast"]),
            float(result.loc[times[0], "ForecastPVAvailableWattsWest"]),
        )
        self.assertGreater(
            float(result.loc[times[-1], "ForecastPVAvailableWattsWest"]),
            float(result.loc[times[-1], "ForecastPVAvailableWattsEast"]),
        )

    def test_night_is_zero_and_array_components_sum_to_total(self) -> None:
        east = PVArrayConfig("East", 90.0, 65.0, 1000.0, 1000.0)
        south = PVArrayConfig("South", 180.0, 65.0, 2000.0, 2000.0)
        west = PVArrayConfig("West", 270.0, 65.0, 1000.0, 1000.0)
        times = pd.DatetimeIndex(["2026-12-21T00:00:00", "2026-12-21T03:00:00"])
        result, _ = build_physical_solar_forecast(
            pd.Series([0.0, 0.0], index=times),
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(east, south, west),
        )

        components = sum(result[f"ForecastPVAvailableWatts{name}"] for name in ("East", "South", "West"))
        np.testing.assert_allclose(result["ForecastPVAvailableWatts"], components)
        np.testing.assert_allclose(result["ForecastPVAvailableWatts"], 0.0)
        np.testing.assert_allclose(result["ForecastSolarWatts"], result["ForecastPVAvailableWatts"])

    def test_controller_limit_and_temperature_inputs_are_bounded(self) -> None:
        array = PVArrayConfig(
            "South",
            180.0,
            65.0,
            10000.0,
            500.0,
            fixed_efficiency=1.0,
        )
        times = pd.DatetimeIndex(["2026-06-21T12:00:00", "2026-06-21T15:00:00"])
        result, metadata = build_physical_solar_forecast(
            pd.Series([1000.0, 1000.0], index=times),
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(array),
            air_temperature_c=pd.Series([20.0, 20.0], index=times),
            wind_speed_m_s=pd.Series([5.0, 5.0], index=times),
        )

        self.assertLessEqual(float(result["ForecastPVAvailableWattsSouth"].max()), 500.0)
        self.assertGreater(float(result["ForecastPVControllerClippingWattsSouth"].max()), 0.0)
        self.assertTrue(np.all(np.isfinite(result["ForecastPVCellTemperatureSouth"])))
        self.assertEqual(metadata["solar_temperature_model"], "faiman_u0_u1")

    def test_supplied_direct_beam_is_retained_at_polar_winter_low_sun(self) -> None:
        south = PVArrayConfig(
            "South",
            180.0,
            64.83,
            1000.0,
            5000.0,
            fixed_efficiency=1.0,
            temperature_coefficient_per_c=0.0,
            incidence_angle_modifier_b0=0.0,
        )
        times = pd.DatetimeIndex(["2026-12-21T13:30:00"])
        ghi = pd.Series([25.0], index=times)
        direct_horizontal = pd.Series([20.0], index=times)

        supplied, metadata = build_physical_solar_forecast(
            ghi,
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(south),
            direct_horizontal_w_m2=direct_horizontal,
        )
        erbs, _ = build_physical_solar_forecast(
            ghi,
            latitude=SITE_LATITUDE,
            longitude=SITE_LONGITUDE,
            config=config_for(south),
        )

        self.assertGreater(
            float(supplied["ForecastPlaneOfArrayIrradianceSouth"].iloc[0]),
            5.0 * float(erbs["ForecastPlaneOfArrayIrradianceSouth"].iloc[0]),
        )
        self.assertIn("supplied_direct_low_sun_dni_bounded", metadata["solar_degradation_codes"])

    def test_config_digest_is_stable_and_changes_with_geometry(self) -> None:
        first = config_for(PVArrayConfig("South", 180.0, 65.0, 2000.0, 2000.0))
        same = config_for(PVArrayConfig("South", 180.0, 65.0, 2000.0, 2000.0))
        changed = config_for(PVArrayConfig("South", 181.0, 65.0, 2000.0, 2000.0))

        self.assertEqual(physical_solar_config_digest(first), physical_solar_config_digest(same))
        self.assertNotEqual(physical_solar_config_digest(first), physical_solar_config_digest(changed))
        self.assertTrue(physical_solar_contract_id(first).startswith("solar-physical-v2-"))
        self.assertNotEqual(
            physical_solar_contract_id(first, latitude=64.83, longitude=-23.25),
            physical_solar_contract_id(first, latitude=50.0, longitude=0.0),
        )

    def test_repository_candidate_config_is_explicitly_provisional(self) -> None:
        path = Path(__file__).with_name("config") / "power_solar_physical_candidate_v1.json"
        config = load_physical_solar_config(path)

        self.assertEqual(config.configuration_status, "provisional")
        self.assertEqual([array.name for array in config.arrays], ["East", "South", "West"])


if __name__ == "__main__":
    unittest.main()
