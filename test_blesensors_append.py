import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from append_new_blesensors_to_zarr import append_new, read_increment

HEADER = ['time_utc','sensor','temperature_c','relative_humidity_pct','battery_pct','rssi_dbm']


def write_csv(path, rows, mode='w'):
    with path.open(mode, newline='') as handle:
        writer = csv.writer(handle)
        if mode == 'w':
            writer.writerow(HEADER)
        writer.writerows(rows)


def legacy(path, stamp, sensor='aps', temp=10, rh=60):
    with path.open('a') as handle:
        handle.write(json.dumps({'time': stamp, 'name':sensor, 'temperature':temp,'humidity':rh,'battery':0})+'\n')


def test_csv_takes_over_per_sensor_without_duplicate_legacy_seconds(tmp_path):
    root=tmp_path/'raw';root.mkdir(); target=tmp_path/'products'/'blesensors.zarr'
    log=root/'ble_temperatures_20261023.log'
    legacy(log,'2026-10-23 07:59:58')
    legacy(log,'2026-10-23 08:00:00')
    legacy(log,'2026-10-23 08:00:01',sensor='ass',temp=20)
    write_csv(root/'ble_readings_20261023_08.csv',[
        ['2026-10-23T08:00:00.123456Z','aps',11,62,0,-79],
        ['2026-10-23T08:00:02.123456Z','aps',11,62,'',''],
    ])
    append_new(root,target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert int(ds.APSBLETemperature.count()) == 3
        assert int(ds.ASSBLETemperature.count()) == 1
        assert ds.APSBLEBattery.sel(time='2026-10-23T08:00:00.123456').item() == 0
        assert np.isnan(ds.APSBLEBattery.sel(time='2026-10-23T08:00:02.123456').item())
        original=ds.load()
    result=append_new(root,target)
    assert result['input_records']==0
    with xr.open_zarr(target, chunks=None) as again:
        xr.testing.assert_equal(original,again)


def test_partial_row_is_retried_and_append_preserves_microseconds(tmp_path):
    path=tmp_path/'ble_readings_20261023_08.csv'
    write_csv(path,[['2026-10-23T08:00:00.123456Z','aps',11,62,0,-79]])
    with path.open('a') as h:h.write('2026-10-23T08:00:02.654321Z,aps,12,')
    rows,state,bad=read_increment(path)
    assert len(rows)==1 and bad==0
    with path.open('a') as h:h.write('63,0,-78\n')
    rows2,_,bad=read_increment(path,state)
    assert len(rows2)==1 and rows2[0]['time']==pd.Timestamp('2026-10-23T08:00:02.654321')
    target=tmp_path/'product'/'b.zarr';append_new(tmp_path,target)
    write_csv(path,[['2026-10-23T08:00:04.987654Z','aps',13,64,0,-77]],mode='a')
    append_new(tmp_path,target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.sizes['time']==3
        assert ds.time.values[-1]==np.datetime64('2026-10-23T08:00:04.987654')


def test_late_sensor_at_existing_timestamp_is_not_lost(tmp_path):
    path=tmp_path/'ble_readings_20261023_08.csv'
    write_csv(path,[['2026-10-23T08:00:00Z','aps',11,62,0,-79],
                    ['2026-10-23T08:00:04Z','aps',12,63,0,-78]])
    target=tmp_path/'product'/'b.zarr';append_new(tmp_path,target)
    write_csv(path,[['2026-10-23T08:00:00Z','ass',20,50,91,-60]],mode='a')
    append_new(tmp_path,target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.ASSBLETemperature.sel(time='2026-10-23T08:00:00').item()==20
        assert ds.APSBLETemperature.sel(time='2026-10-23T08:00:00').item()==11
        assert ds.sizes['time']==2
        latest=json.loads(ds.attrs['latest_observations'])
        assert latest['aps']['time_utc']=='2026-10-23T08:00:04Z'
        assert latest['ass']['temperature_c']==20
    write_csv(path,[['2026-10-23T08:00:08Z','aps',13,64,'',-77]],mode='a')
    append_new(tmp_path,target)
    with xr.open_zarr(target, chunks=None) as ds:
        latest=json.loads(ds.attrs['latest_observations'])
        assert latest['ass']['temperature_c']==20
        assert latest['aps']['battery_pct'] is None
        assert latest['aps']['temperature_c']==13


def test_replaced_file_reprocessed_and_old_observations_preserved(tmp_path):
    path=tmp_path/'ble_readings_20261023_08.csv'
    write_csv(path,[['2026-10-23T08:00:00Z','aps',11,62,0,-79]])
    target=tmp_path/'product'/'b.zarr';append_new(tmp_path,target)
    write_csv(path,[['2026-10-23T08:00:00Z','aps',12,62,0,-79]])
    append_new(tmp_path,target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.item()==12
    write_csv(path,[['2026-10-23T08:00:10Z','aps',13,62,0,-79]])
    append_new(tmp_path,target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.sizes['time']==2
        assert ds.time.values[-1]==np.datetime64('2026-10-23T08:00:10')


def test_invalid_rows_and_legacy_none_leave_valid_measurements(tmp_path):
    path=tmp_path/'ble_readings_20261023_08.csv'
    write_csv(path,[['2026-10-23T08:00:00Z','aps',11,162,0,-79],
                    ['2026-10-23T08:00:04Z','aps','nan',62,0,-79],
                    ['2026-10-23T08:00:08Z','unknown',11,62,0,-79]])
    log=tmp_path/'ble_temperatures_20261023.log'
    log.write_text('{"time":"2026-10-23 07:59:00","name":"aps","temperature":10,"humidity":60,"battery":None}\n')
    target=tmp_path/'product'/'b.zarr'; result=append_new(tmp_path,target)
    assert result['rejected_records']==3
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.item()==10
        assert np.isnan(ds.APSBLEBattery.item())


def test_state_loss_replay_is_idempotent(tmp_path):
    path=tmp_path/'ble_readings_20261023_08.csv'
    write_csv(path,[['2026-10-23T08:00:00Z','aps',11,62,0,-79]])
    target=tmp_path/'product'/'b.zarr';append_new(tmp_path,target)
    target.with_suffix('.ingest.json').unlink()
    append_new(tmp_path,target)
    with xr.open_zarr(target, chunks=None) as ds:assert ds.sizes['time']==1


def test_atomic_replacement_with_same_edges_updates_middle(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    rows = [[f'2026-10-23T08:00:{i:02d}Z', 'aps', 11, 62, 0, -79] for i in range(30)]
    write_csv(path, rows)
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    before = path.read_bytes()
    rows[15][2] = 99
    replacement = tmp_path / 'replacement.tmp'
    write_csv(replacement, rows)
    assert replacement.read_bytes()[:128] == before[:128]
    assert replacement.read_bytes()[-128:] == before[-128:]
    os.replace(replacement, path)
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.sel(time='2026-10-23T08:00:15').item() == 99


def test_append_migrates_existing_different_time_encoding(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        old = ds.load()
    old.time.encoding.update(units='days since 2026-10-23 08:00:00', dtype='int64')
    old.to_zarr(target, mode='w', consolidated=True)
    write_csv(path, [['2026-10-23T08:00:04.123456Z', 'aps', 12, 63, 0, -78]], mode='a')
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        np.testing.assert_array_equal(ds.time.values, np.array([
            '2026-10-23T08:00:00', '2026-10-23T08:00:04.123456'], dtype='datetime64[ns]'))
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]
    assert zarr.open_group(str(target), mode='r')['time'].attrs['units'].startswith('microseconds since 1970')


def test_replay_heals_partial_old_in_place_append(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    write_csv(path, [['2026-10-23T08:00:04Z', 'aps', 12, 63, 0, -78]], mode='a')
    group = zarr.open_group(str(target), mode='a')
    group['time'].resize(2)
    group['time'][1] = pd.Timestamp('2026-10-23T08:00:04').value // 1000
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.sizes['time'] == 2
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]


def test_failed_staged_append_preserves_product_and_cursor(tmp_path, monkeypatch):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    cursor = target.with_suffix('.ingest.json').read_bytes()
    original_chunks = {p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    write_csv(path, [['2026-10-23T08:00:04Z', 'aps', 12, 63, 0, -78]], mode='a')
    original = xr.Dataset.to_zarr

    def fail_after_write(self, store, *args, **kwargs):
        result = original(self, store, *args, **kwargs)
        if kwargs.get('append_dim'):
            assert Path(store) != target
            raise OSError('simulated interrupted append')
        return result

    with monkeypatch.context() as patch:
        patch.setattr(xr.Dataset, 'to_zarr', fail_after_write)
        with pytest.raises(OSError, match='interrupted append'):
            append_new(tmp_path, target)
    assert target.with_suffix('.ingest.json').read_bytes() == cursor
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()} == original_chunks
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]


def test_invalid_staged_shape_is_never_published(tmp_path, monkeypatch):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    cursor = target.with_suffix('.ingest.json').read_bytes()
    write_csv(path, [['2026-10-23T08:00:04Z', 'aps', 12, 63, 0, -78]], mode='a')
    original = xr.Dataset.to_zarr

    def corrupt_shape(self, store, *args, **kwargs):
        result = original(self, store, *args, **kwargs)
        if kwargs.get('append_dim'):
            zarr.open_group(str(store), mode='a')['APSBLEHumidity'].resize(3)
        return result

    monkeypatch.setattr(xr.Dataset, 'to_zarr', corrupt_shape)
    with pytest.raises(ValueError, match='candidate failed'):
        append_new(tmp_path, target)
    assert target.with_suffix('.ingest.json').read_bytes() == cursor
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.values.tolist() == [11.0]


def test_completed_chunks_reused_but_partial_chunk_detached(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    times = pd.date_range('2026-10-23T08:00:00', periods=4097, freq='s')
    write_csv(path, [[stamp.isoformat() + 'Z', 'aps', 11, 62, 0, -79] for stamp in times])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    completed = target / 'APSBLETemperature' / '0'
    partial = target / 'APSBLETemperature' / '1'
    completed_inode, partial_inode = completed.stat().st_ino, partial.stat().st_ino
    completed_bytes = completed.read_bytes()
    write_csv(path, [[(times[-1] + pd.Timedelta(seconds=4)).isoformat() + 'Z',
                      'aps', 12, 63, 0, -78]], mode='a')
    append_new(tmp_path, target)
    assert completed.stat().st_ino == completed_inode
    assert completed.read_bytes() == completed_bytes
    assert partial.stat().st_ino != partial_inode
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.sizes['time'] == 4098
        assert ds.APSBLETemperature.values[-2:].tolist() == [11.0, 12.0]


def test_startup_recovers_store_missing_between_publication_renames(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    target.rename(target.with_name('.' + target.name + '.previous.tmp'))
    staging = target.with_name('.' + target.name + '.staging.tmp')
    staging.mkdir()
    (staging / 'incomplete').write_text('unpublished')
    write_csv(path, [['2026-10-23T08:00:04Z', 'aps', 12, 63, 0, -78]], mode='a')
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]
    assert not staging.exists()


def test_published_product_replays_when_cursor_commit_fails(tmp_path, monkeypatch):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    cursor = target.with_suffix('.ingest.json').read_bytes()
    write_csv(path, [['2026-10-23T08:00:04Z', 'aps', 12, 63, 0, -78]], mode='a')
    original = os.replace

    def fail_cursor(source, destination):
        if Path(destination) == target.with_suffix('.ingest.json'):
            raise OSError('simulated cursor commit failure')
        return original(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', fail_cursor)
        with pytest.raises(OSError, match='cursor commit failure'):
            append_new(tmp_path, target)
    assert target.with_suffix('.ingest.json').read_bytes() == cursor
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.sizes['time'] == 2
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]
    assert not target.with_name('.' + target.name + '.previous.tmp').exists()


def test_unchanged_fully_consumed_source_is_not_reopened(tmp_path, monkeypatch):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    _, state, _ = read_increment(path)
    original = Path.open

    def fail_source_reopen(self, *args, **kwargs):
        if self == path:
            raise AssertionError('unchanged closed source was reread')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', fail_source_reopen)
    rows, next_state, rejected = read_increment(path, state)
    assert rows == [] and next_state == state and rejected == 0


def test_csv_gap_keeps_legacy_then_late_csv_replaces_only_matching_sensor_second(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    log = tmp_path / 'ble_temperatures_20261023.log'
    write_csv(path, [['2026-10-23T08:00:00.123456Z', 'aps', 11, 62, 0, -79]])
    legacy(log, '2026-10-23 08:00:00', temp=10)
    legacy(log, '2026-10-23 08:00:04', temp=12)
    legacy(log, '2026-10-23 08:00:04', sensor='ass', temp=20)
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.sizes['time'] == 2
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]
        assert ds.APSBLESourceFormat.values.tolist() == [1.0, 0.0]
    write_csv(path, [['2026-10-23T08:00:04.654321Z', 'aps', 13, 63, 0, -78]], mode='a')
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert np.isnan(ds.APSBLETemperature.sel(time=pd.Timestamp('2026-10-23T08:00:04')).item())
        assert ds.ASSBLETemperature.sel(time=pd.Timestamp('2026-10-23T08:00:04')).item() == 20
        assert ds.APSBLETemperature.sel(time='2026-10-23T08:00:04.654321').item() == 13
        assert int(ds.APSBLETemperature.count()) == 2


def test_policy_migration_replays_old_raw_to_recover_legacy_gap(tmp_path):
    path = tmp_path / 'ble_readings_20261023_08.csv'
    log = tmp_path / 'ble_temperatures_20261023.log'
    write_csv(path, [['2026-10-23T08:00:00Z', 'aps', 11, 62, 0, -79]])
    legacy(log, '2026-10-23 08:00:04', temp=12)
    target = tmp_path / 'product' / 'b.zarr'
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        old = ds.isel(time=slice(0, 1)).load()
    old.attrs.pop('deduplication')
    old.to_zarr(target, mode='w', consolidated=True)
    append_new(tmp_path, target)
    with xr.open_zarr(target, chunks=None) as ds:
        assert ds.APSBLETemperature.values.tolist() == [11.0, 12.0]
