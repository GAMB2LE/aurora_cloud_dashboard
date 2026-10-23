from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import enclosure_sensors as enclosure
import mobile_catalog
from generate_power_display_summary import _section_subset
from grouped_timeseries import build_power_display_summary_dataset
from send_ops_alerts import evaluate_alerts

NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


def observations():
    return xr.Dataset({
        'APSBLETemperature': ('time', [10., np.nan, 12.]),
        'APSBLEHumidity': ('time', [100., np.nan, 60.]),
        'APSBLEBattery': ('time', [80., np.nan, 79.]),
        'APSBLERSSI': ('time', [-60., np.nan, -61.]),
        'ASSBLETemperature': ('time', [np.nan, 20., np.nan]),
        'ASSBLEHumidity': ('time', [np.nan, 50., np.nan]),
    }, coords={'time': pd.to_datetime(['2026-10-23T07:40:00', '2026-10-23T07:52:00', '2026-10-23T07:58:00'])})


def test_per_sensor_latest_pairs_and_freshness():
    aps, ass = enclosure.sensors_from_dataset(observations(), now=NOW)
    assert aps['temperatureC'] == 12
    assert aps['relativeHumidityPct'] == 60
    assert aps['batteryPct'] == 79 and aps['rssiDbm'] == -61
    assert aps['state'] == 'fresh' and aps['ageMinutes'] == 2
    assert ass['state'] == 'delayed' and ass['ageMinutes'] == 8
    assert aps['dewPointC'] == pytest.approx(4.47, abs=.02)
    assert ass['observedAt'] == '2026-10-23T07:52:00Z'


def test_pairing_never_uses_other_sensor_or_prior_humidity():
    ds = observations()
    ds['APSBLEHumidity'][-1] = np.nan
    aps, _ = enclosure.sensors_from_dataset(ds, now=NOW)
    assert aps['temperatureC'] == 10
    assert aps['relativeHumidityPct'] == 100
    assert aps['state'] == 'stale'
    assert aps['dewPointMarginC'] == pytest.approx(0)


@pytest.mark.parametrize('age,state,level', [(5,'fresh','ok'), (5.01,'delayed','warning'), (15,'delayed','warning'), (15.01,'stale','error')])
def test_freshness_boundaries_and_snapshot_age_recomputed(age, state, level):
    cards = enclosure.sensors_from_dataset(observations(), now=NOW)
    snapshot = enclosure.snapshot_metrics(cards)
    observed = datetime.fromisoformat(cards[0]['observedAt'].replace('Z','+00:00'))
    aps = enclosure.sensors_from_snapshot(snapshot, now=observed + timedelta(minutes=age))[0]
    assert (aps['state'], aps['level']) == (state, level)


def test_invalid_humidity_and_missing_product_unavailable(tmp_path):
    ds = observations()
    ds['APSBLEHumidity'][:] = [101., np.nan, -1.]
    assert enclosure.sensors_from_dataset(ds, now=NOW)[0]['state'] == 'unavailable'
    with patch.dict('os.environ', {'BLESENSORS_ZARR_PATH': str(tmp_path/'missing.zarr')}):
        assert all(row['state'] == 'unavailable' for row in enclosure.load_enclosure_sensors(now=NOW))


def test_zero_humidity_has_no_dew_point_but_valid_observation():
    ds = observations()
    ds['APSBLEHumidity'][:] = [0., np.nan, 0.]
    assert enclosure.sensors_from_dataset(ds, now=NOW)[0]['dewPointC'] is None
    display = enclosure.display_dataset(ds)
    assert np.isnan(display.APSBLEDewPoint.values).all()


def test_display_preserves_empty_minutes_and_power_independence():
    ds = observations()
    power = xr.Dataset({'InternalTemperature': ('time', [33.,34.])}, coords={'time': pd.to_datetime(['2026-10-23T05:00','2026-10-23T05:01'])})
    display = build_power_display_summary_dataset(power, ble_ds=ds)
    assert float(display.APSBLETemperature.sel(time='2026-10-23T07:58')) == 12.
    assert np.isnan(float(display.InternalTemperature.sel(time='2026-10-23T07:58')))
    ble = enclosure.display_dataset(ds)
    assert np.isnan(float(ble.APSBLETemperature.sel(time='2026-10-23T07:41')))
    assert 'APSBLEHumidity' in _section_subset(display, 'current')
    assert 'APSBLEHumidity' not in _section_subset(display, 'forecast')


def test_ble_snapshot_does_not_activate_existing_emails():
    ds = observations()
    ds['APSBLEHumidity'][-1] = 100
    metrics = enclosure.snapshot_metrics(enclosure.sensors_from_dataset(ds, now=NOW))
    assert metrics['aps_ble_dewpoint_margin_c'] == pytest.approx(0)
    assert not any(key.startswith('aps_internal_') for key in metrics)
    assert evaluate_alerts(metrics) == []


def test_power_panel_survives_missing_legacy_or_bundle_product_and_retains_gaps(tmp_path):
    path = tmp_path/'ble.zarr'
    observations().to_zarr(path, consolidated=True)
    with patch.dict('os.environ', {'BLESENSORS_ZARR_PATH': str(path)}), patch.object(enclosure, 'utc_now', return_value=NOW):
        panel = enclosure.power_panel(max_points=2)
        assert panel['id'] == 'aps_enclosure'
        points = panel['traces'][0]['points']
        assert len(points) == 2 and points[0]['segment'] != points[1]['segment']
        with patch.object(mobile_catalog, '_power_without_enclosure', return_value={'panels': [], 'warning':'Power unavailable'}):
            payload = mobile_catalog.power(group='current')
        assert payload['panels'][0]['id'] == 'aps_enclosure'
        assert payload['panels'][0]['traces'][0]['points'][-1]['value'] == 12


def test_forecast_api_does_not_add_observed_enclosure_panel():
    with patch.object(mobile_catalog, '_power_without_enclosure', return_value={'panels': []}):
        assert mobile_catalog.power(group='forecast')['panels'] == []


def test_cached_power_explanation_uses_absolute_reception_time(tmp_path):
    path = tmp_path/'ble.zarr'
    observations().to_zarr(path, consolidated=True)
    with patch.dict('os.environ', {'BLESENSORS_ZARR_PATH': str(path)}):
        fresh = enclosure.power_panel(now=NOW)['explanation']
        stale = enclosure.power_panel(now=NOW + timedelta(hours=1))['explanation']
    assert fresh == stale
    assert 'Last reception: 2026-10-23 07:58:00 UTC.' in fresh
    assert 'Fresh' not in fresh and 'minutes ago' not in fresh


def test_operations_contract_is_additive_and_json_safe():
    import json
    cards = enclosure.sensors_from_dataset(observations(), now=NOW)
    with patch.object(mobile_catalog, 'load_enclosure_sensors', return_value=cards), patch.object(mobile_catalog, 'read_json_file', return_value={}):
        payload = mobile_catalog.operations()
    assert payload['enclosureSensors'] == cards
    assert {'streamStates', 'alerts', 'trendCards'} <= payload.keys()
    json.dumps(payload, allow_nan=False)


def test_enclosure_survives_invalid_activated_immutable_bundle(tmp_path):
    import json
    ble = tmp_path/'ble.zarr'
    observations().to_zarr(ble, consolidated=True)
    manifest = tmp_path/'generation.json'
    manifest.write_text(json.dumps({'schemaVersion': 1, 'status': 'incomplete'}))
    environment = {
        'BLESENSORS_ZARR_PATH': str(ble),
        'AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED': 'true',
        'AURORA_POWER_FORECAST_PUBLICATION_ACTIVE': 'true',
        'AURORA_POWER_FORECAST_BUNDLE_READY_PATH': str(manifest),
        'AURORA_POWER_FORECAST_BUNDLE_STATUS_PATH': str(tmp_path/'status.json'),
    }
    with patch.dict('os.environ', environment), patch.object(enclosure, 'utc_now', return_value=NOW):
        payload = mobile_catalog.power(group='current')
    assert payload['status'] == 'unavailable'
    assert [panel['id'] for panel in payload['panels']] == ['aps_enclosure']
    assert payload['panels'][0]['traces'][0]['points'][-1]['value'] == 12


def test_latest_metadata_avoids_history_scan_and_recomputes_age(tmp_path):
    import json
    path = tmp_path/'ble.zarr'
    ds = observations()
    ds.attrs['latest_observations'] = json.dumps({'aps': {
        'time_utc': '2026-10-23T07:58:00Z', 'temperature_c': 12.,
        'humidity_pct': 60., 'battery_pct': 79., 'rssi_dbm': -61.,
    }})
    ds.to_zarr(path, consolidated=True)
    with patch.dict('os.environ', {'BLESENSORS_ZARR_PATH': str(path), 'BLESENSORS_SOURCE_LABEL': 'SwitchBot via ASS · direct-field continuity'}), patch.object(enclosure, 'sensors_from_dataset', side_effect=AssertionError('must not scan history')):
        cards = enclosure.load_enclosure_sensors(now=NOW + timedelta(minutes=5))
    assert cards[0]['state'] == 'delayed'
    assert cards[0]['ageMinutes'] == 7
    assert cards[0]['rssiDbm'] == -61
    assert 'direct-field continuity' in cards[0]['source']
    assert cards[1]['state'] == 'unavailable'
