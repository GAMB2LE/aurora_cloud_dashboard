"""Grid measurements affect supply once and never disguise unknown station load."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from power_grid import station_load, continued_grid_rate, no_grid_intervals
from power_operating_scenarios import integrate_soc_members
from power_battery_model import BatteryModel
from grouped_timeseries import merge_operating_scenarios_into_display_summary


def observations():
    times = pd.date_range('2026-10-23T12:00:00', periods=900, freq='s')
    return pd.DataFrame({'GridWatts': 1100., 'MultiPlusDCWatts': 1000.,
                         'BatteryWatts': 900., 'SolarWatts_East': 100.,
                         'SolarWatts_South': 100., 'SolarWatts_West': 100.}, index=times)


def test_charger_supply_is_counted_once_and_missing_is_unknown():
    frame = observations()
    assert station_load(frame).iloc[-1] == 400
    frame['MultiPlusDCWatts'] = -10
    frame['BatteryWatts'] = -100
    assert station_load(frame).iloc[-1] == 400
    assert station_load(frame.drop(columns='MultiPlusDCWatts')).isna().all()
    frame['MultiPlusDCWatts'] = 0
    frame['BatteryWatts'] = 900
    assert station_load(frame).isna().all()


def test_rate_coverage_zero_and_freshness():
    frame = observations(); anchor = frame.index[-1]
    assert continued_grid_rate(frame, anchor)[0] == 1000
    frame.iloc[:91, frame.columns.get_loc('MultiPlusDCWatts')] = np.nan
    assert continued_grid_rate(frame, anchor)[0] is None
    frame['MultiPlusDCWatts'] = 0
    assert continued_grid_rate(frame, anchor)[0] == 0
    assert continued_grid_rate(frame, anchor, now=anchor + pd.Timedelta(minutes=21))[0] is None
    frame.iloc[-1, frame.columns.get_loc('GridWatts')] = np.nan
    assert continued_grid_rate(frame, anchor)[0] is None


def test_real_logger_cadence_does_not_look_like_an_outage():
    frame = observations().iloc[:820].copy()
    frame.index = pd.date_range('2026-10-23T12:00:00', periods=820, freq='1100ms')
    assert continued_grid_rate(frame, frame.index[-1])[0] == 1000


def test_soc_learning_rejects_charging_between_endpoints():
    frame = observations(); frame['MultiPlusDCWatts'] = 0
    starts, ends = [frame.index[0]], [frame.index[-1]]
    assert no_grid_intervals(frame, starts, ends).tolist() == [True]
    frame.iloc[450, frame.columns.get_loc('MultiPlusDCWatts')] = 1000
    assert no_grid_intervals(frame, starts, ends).tolist() == [False]
    assert no_grid_intervals(frame.drop(columns='MultiPlusDCWatts'), starts, ends).tolist() == [False]


def test_charging_uses_same_integrator_and_battery_limits():
    times = pd.date_range('2026-10-23', periods=4, freq='h')
    args = dict(initial_soc=50, times=times, solar_members_w=np.zeros((2,4)),
                load_members_w=np.full((2,4), 500.), capacity_kwh=20.,
                battery_model=BatteryModel(usable_capacity_kwh=20, charge_efficiency=1,
                    discharge_efficiency=1, max_charge_w=1000, max_discharge_w=1000))
    base = integrate_soc_members(**args)
    zero = integrate_soc_members(**args, external_charge_w=0)
    charge = integrate_soc_members(**args, external_charge_w=1000)
    np.testing.assert_array_equal(base, zero)
    np.testing.assert_allclose(base[0], [50,47.5,45,42.5])
    np.testing.assert_allclose(charge[0], [50,52.5,55,57.5])
    limited = integrate_soc_members(**args, external_charge_w=10000)
    np.testing.assert_allclose(limited[0], [50,55,60,65])
    with pytest.raises(ValueError): integrate_soc_members(**args, external_charge_w=np.nan)


def test_unavailable_plan_removes_old_grid_curve():
    ds = xr.Dataset({'GridContinuesSOCP50': ('time', [50.,60.])},
                    coords={'time': pd.date_range('2026-10-23', periods=2, freq='h')})
    assert 'GridContinuesSOCP50' not in merge_operating_scenarios_into_display_summary(ds, None)


def test_operating_product_retains_grid_curve_and_shared_anchor(tmp_path):
    from test_power_operating_scenarios import _training_data, _forecast_inputs
    from power_operating_scenarios import fit_operating_model, build_operating_scenarios
    from grouped_timeseries import _operating_scenario_frame, SUMMARY_LAYOUTS
    training, pdu = _training_data()
    model = fit_operating_model(training, pdu, lookback_days=2)
    frame = observations()
    frame.index = pd.date_range(end=pd.Timestamp.now(tz='UTC').tz_localize(None).floor('s'), periods=900, freq='s')
    frame['BatterySOC'] = 50.
    power = xr.Dataset.from_dataframe(frame.rename_axis('time'))
    deterministic, ensemble = _forecast_inputs(frame.index[-1], horizon_hours=6)
    product = build_operating_scenarios(power, deterministic, model, ensemble=ensemble,
                                       horizon_hours=6, optimization_hours=6)
    assert product.attrs['grid_charge_w'] == '1000.0'
    current = product.sel(scenario='current_mode').ScenarioSOCP50
    assert product.GridContinuesSOCP50.values[0] == current.values[0] == 50.
    assert np.all(product.GridContinuesSOCP50.values[1:] >= current.values[1:])
    path = tmp_path / 'scenarios.zarr'
    product.to_zarr(path)
    with xr.open_zarr(path) as restored:
        np.testing.assert_array_equal(restored.GridContinuesSOCP50, product.GridContinuesSOCP50)
        assert 'GridContinuesSOCP50' in _operating_scenario_frame(restored)
    for panel in SUMMARY_LAYOUTS['power']:
        if panel.key in {'soc_24h_forecast','soc_ecmwf_forecast'}:
            assert 'GridContinuesSOCP50' in [trace.var for trace in panel.traces]
