"""Paired ablations must replay the battery used by each immutable issue."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecmwf_forecast_provider import ForecastProviderResult
from generate_power_soc_forecast import build_forecast_dataset, generate
from generate_power_soc_v12_candidate import LANES, run_candidate
from power_battery_model import BatteryModel
from test_power_solar_integration import CONFIG_PATH, synthetic_inputs
from test_power_v12_hybrid import _tree_digest


ARCHIVED_BATTERY = BatteryModel(
    usable_capacity_kwh=25.7609,
    charge_efficiency=0.905479,
    discharge_efficiency=0.925823,
    max_charge_w=3493.53,
    max_discharge_w=542.3,
    calibration_sample_count=92,
    calibration_confidence="calibrated",
)


def test_same_cycle_reanchors_replay_their_own_battery_in_every_lane(tmp_path):
    power, solar = synthetic_inputs()
    solar = solar.assign_coords(forecast_reference_time=solar.time.values[0])
    forcing = tmp_path / "forcing.grib2"
    forcing.write_bytes(b"same immutable weather cycle, different SOC anchors")
    power_path = tmp_path / "power.zarr"
    power.to_zarr(power_path, consolidated=True)
    provider = ForecastProviderResult(solar, {
        "effective_provider": "legacy", "requested_provider": "legacy", "fallback_reason": "",
    })
    baselines = []
    batteries = [BatteryModel(), ARCHIVED_BATTERY]
    with patch("generate_power_soc_forecast.open_provider_solar_forecast", return_value=provider):
        for index, battery in enumerate(batteries):
            baseline_root = tmp_path / f"baseline-{index}"
            issue_path = baseline_root / "issue" / "forecast.zarr"
            with patch("generate_power_soc_forecast.fit_battery_model", return_value=battery):
                generate(
                    power_zarr=power_path, pdu_zarr=tmp_path / "missing-pdu.zarr",
                    output_zarr=baseline_root / "forecast.zarr", input_forecast=forcing,
                    state_path=baseline_root / "state.json", archive_zarr=baseline_root / "archive.zarr",
                    skill_zarr=None, hindcast_zarr=None, horizon_hours=15,
                    max_power_age_minutes=None, archive_forecast=True,
                    issue_snapshot_zarr=issue_path,
                )
            baselines.append(issue_path)
            if index == 0:
                extra = power.isel(time=[-1]).assign_coords(
                    time=[pd.Timestamp(power.time.values[-1]) + pd.Timedelta(minutes=30)]
                )
                power = xr.concat([power, extra], dim="time")
                power.to_zarr(power_path, mode="w", consolidated=True)

    baseline_digests = [_tree_digest(path) for path in baselines]
    cycles = []
    anchors = []
    # On the unfixed implementation the first issue fits defaults successfully;
    # the second skips fitting (same weather cycle) and incorrectly uses defaults.
    with patch("generate_power_soc_forecast.fit_battery_model", return_value=BatteryModel()) as fit:
        for issue_path, battery in zip(baselines, batteries):
            results = run_candidate(
                baseline_issue_zarr=issue_path, baseline_archive_zarr=tmp_path / "missing-archive.zarr",
                baseline_ensemble_zarr=None, candidate_root=tmp_path / "candidate",
                power_zarr=power_path, pdu_zarr=tmp_path / "missing-pdu.zarr",
                physical_config=CONFIG_PATH, asfs_zarr=tmp_path / "missing-asfs.zarr",
                menapia_mqtt_log=tmp_path / "missing-menapia.log",
                public_source_manifest_root=tmp_path / "missing-public-models",
            )
            assert set(results) == set(LANES)
            for lane, path in results.items():
                with xr.open_zarr(path, chunks={}) as candidate:
                    for name, value in battery.attrs().items():
                        assert candidate.attrs[name] == value, (lane, name)
                    with xr.open_zarr(issue_path, chunks={}) as baseline:
                        assert candidate.attrs["initial_soc_time"] == baseline.attrs["initial_soc_time"]
                        if lane == "C_load_residual":
                            np.testing.assert_allclose(
                                candidate.BatterySOCForecast, baseline.BatterySOCForecast,
                                rtol=0, atol=1e-6,
                            )
                state = json.loads((path.parent / "power_soc_forecast_state.json").read_text())
                assert state["battery_model"]["battery_usable_capacity_kwh"] == battery.attrs()["battery_usable_capacity_kwh"]
            with xr.open_zarr(issue_path, chunks={}) as baseline:
                cycles.append(baseline.attrs["source_cycle_set_id"])
                anchors.append(baseline.attrs["initial_soc_time"])
        fit.assert_not_called()
    assert cycles[0] == cycles[1]
    assert anchors[0] != anchors[1]
    assert [_tree_digest(path) for path in baselines] == baseline_digests


@pytest.mark.parametrize("allow_update", [False, True])
def test_fixed_battery_wins_over_state_and_new_matured_verification(allow_update):
    power, solar = synthetic_inputs()
    with patch("generate_power_soc_forecast._matured_verification_id", return_value="new-verification"), \
         patch("generate_power_soc_forecast.fit_battery_model") as fit:
        result = build_forecast_dataset(
            power, solar, horizon_hours=15,
            state={"battery_model": BatteryModel().attrs(), "calibration_verification_id": "old-verification"},
            allow_calibration_update=allow_update, fixed_battery_model=ARCHIVED_BATTERY,
        )
    fit.assert_not_called()
    for name, value in ARCHIVED_BATTERY.attrs().items():
        assert result.attrs[name] == value


def test_paired_battery_reader_preserves_the_recorded_model():
    assert BatteryModel.from_paired_attrs(ARCHIVED_BATTERY.attrs()).attrs() == ARCHIVED_BATTERY.attrs()


@pytest.mark.parametrize("name,value", [
    ("battery_usable_capacity_kwh", None),
    ("battery_charge_efficiency", "nan"),
    ("battery_discharge_efficiency", "0.1"),
    ("battery_max_charge_w", "invalid"),
    ("battery_max_discharge_w", "30000"),
    ("battery_energy_model", "unknown_model"),
])
def test_paired_battery_reader_rejects_missing_invalid_or_clipped_parameters(name, value):
    attrs = ARCHIVED_BATTERY.attrs()
    if value is None:
        attrs.pop(name)
    else:
        attrs[name] = value
    with pytest.raises(ValueError, match="Paired baseline battery"):
        BatteryModel.from_paired_attrs(attrs)
