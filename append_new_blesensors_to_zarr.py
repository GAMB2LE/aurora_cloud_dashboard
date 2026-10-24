#!/usr/bin/env python3
"""Incrementally ingest BLE companion CSV and pre-CSV legacy history."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import zarr

ROOT_DEFAULT = Path('/project/aurora/raw/blesensors')
ZARR_DEFAULT = Path('/data/aurora/products/blesensors/blesensors.zarr')
SENSORS = {'aps': 'APS', 'ass': 'ASS'}
METRICS = {'temperature': 'Temperature', 'humidity': 'Humidity', 'battery': 'Battery', 'rssi': 'RSSI'}
VARIABLES = {prefix + 'BLE' + suffix for prefix in SENSORS.values()
             for suffix in [*METRICS.values(), 'SourceFormat']}
TIME_ENCODING = {'units': 'microseconds since 1970-01-01 00:00:00',
                 'calendar': 'proleptic_gregorian', 'dtype': 'int64'}
DEDUPLICATION = 'prefer_csv_per_sensor_utc_second'


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _record(row, csv_source):
    sensor = str(row.get('sensor' if csv_source else 'name', '')).lower()
    if sensor not in SENSORS:
        return None
    stamp = pd.to_datetime(row.get('time_utc' if csv_source else 'time'), utc=True, errors='coerce')
    if pd.isna(stamp):
        return None
    temp = _number(row.get('temperature_c' if csv_source else 'temperature'))
    rh = _number(row.get('relative_humidity_pct' if csv_source else 'humidity'))
    if temp is None or rh is None or not -100 <= temp <= 100 or not 0 <= rh <= 100:
        return None
    battery = _number(row.get('battery_pct' if csv_source else 'battery'))
    rssi = _number(row.get('rssi_dbm'))
    return {'time': stamp.tz_localize(None), 'sensor': sensor, 'temperature': temp,
            'humidity': rh, 'battery': battery if battery is not None and 0 <= battery <= 100 else np.nan,
            'rssi': rssi if rssi is not None and -127 <= rssi <= 0 else np.nan,
            'csv_source': int(csv_source)}


def _file_signature(stat):
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def _prefix_digest(handle, length):
    handle.seek(0)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = handle.read(min(remaining, 1024 * 1024))
        if not block:
            raise OSError('BLE source was truncated while reading its consumed prefix')
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def read_increment(path, previous=None):
    """Read complete new lines, verifying the entire already-consumed prefix."""
    path = Path(path)
    previous = previous or {}
    signature = _file_signature(path.stat())
    offset = int(previous.get('offset', 0))
    if (signature == previous.get('stat') and offset == signature['size']
            and previous.get('consumed_sha256')):
        return [], previous, 0
    with path.open('rb') as handle:
        signature = _file_signature(os.fstat(handle.fileno()))
        if (offset < 0 or offset > signature['size'] or not previous.get('consumed_sha256')
                or _prefix_digest(handle, offset) != previous['consumed_sha256']):
            offset = 0
        handle.seek(0)
        header = handle.readline()
        handle.seek(offset)
        data = handle.read()
        complete_length = data.rfind(b'\n') + 1
        complete = data[:complete_length]
        end = offset + complete_length
        digest = _prefix_digest(handle, end)
        # Atomic rsync replacement leaves this open handle consistent. In-place
        # mutation must retry rather than commit a cursor over mixed contents.
        if _file_signature(os.fstat(handle.fileno())) != signature:
            raise OSError(f'BLE source changed while reading: {path}')
    state = {'offset': end, 'consumed_sha256': digest, 'stat': signature}
    rows = []
    rejected = 0
    csv_source = path.suffix == '.csv'
    text = complete.decode('utf-8', errors='replace')
    if csv_source:
        text = text if offset == 0 else header.decode('utf-8') + text
        if text:
            for row in csv.DictReader(io.StringIO(text)):
                record = _record(row, True)
                if record is None:
                    rejected += 1
                else:
                    rows.append(record)
    else:
        for line in text.splitlines():
            try:
                # Older logging prints Python None for a missing battery.
                row = json.loads(re.sub(r'(:\s*)None(?=\s*[,}])', r'\1null', line))
                record = _record(row, False)
                if record is None:
                    rejected += 1
                else:
                    rows.append(record)
            except (ValueError, TypeError, AttributeError):
                rejected += 1
    return rows, state, rejected


def normalize(records, cutovers):
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    for sensor in SENSORS:
        csv_rows = frame[(frame.sensor == sensor) & (frame.csv_source == 1)]
        if not csv_rows.empty:
            first = csv_rows.time.min().floor('s')
            previous = pd.Timestamp(cutovers[sensor]) if sensor in cutovers else first
            cutovers[sensor] = min(first, previous).isoformat()
    csv_buckets = set(zip(frame.loc[frame.csv_source == 1, 'sensor'],
                          frame.loc[frame.csv_source == 1, 'time'].dt.floor('s')))
    legacy_overlap = [source == 0 and (sensor, stamp.floor('s')) in csv_buckets
                      for source, sensor, stamp in zip(frame.csv_source, frame.sensor, frame.time)]
    frame = frame.loc[~np.asarray(legacy_overlap)]
    return frame.sort_values(['time', 'sensor', 'csv_source'], kind='stable').drop_duplicates(['time', 'sensor'], keep='last')


def to_dataset(frame, cutovers, latest=None):
    latest = dict(latest or {})
    times = pd.DatetimeIndex(sorted(frame.time.unique()))
    variables = {}
    for sensor, prefix in SENSORS.items():
        sensor_rows = frame[frame.sensor == sensor].sort_values('time')
        if not sensor_rows.empty:
            last = sensor_rows.iloc[-1]
            latest[sensor] = {
                'time_utc': pd.Timestamp(last.time).isoformat() + 'Z',
                'temperature_c': _number(last.temperature), 'humidity_pct': _number(last.humidity),
                'battery_pct': _number(last.battery), 'rssi_dbm': _number(last.rssi),
            }
        rows = sensor_rows.set_index('time').reindex(times)
        for key, suffix in {**METRICS, 'csv_source': 'SourceFormat'}.items():
            variables[prefix + 'BLE' + suffix] = ('time', rows[key].to_numpy(dtype='float32'))
    ds = xr.Dataset(variables, coords={'time': times.to_numpy(dtype='datetime64[ns]')}, attrs={
        'instrument': 'blesensors', 'title': 'AURORA enclosure Bluetooth sensors',
        'time_basis': 'UTC reception time; legacy naive timestamps interpreted as ASS UTC',
        'csv_cutovers': json.dumps(cutovers, sort_keys=True),
        'deduplication': DEDUPLICATION,
        'aps_sensor_mac': 'D2:35:34:38:1F:88', 'ass_sensor_mac': 'D4:35:34:38:1D:6B',
        'collector': 'ass', 'aps_location': 'APS enclosure', 'ass_location': 'unconfirmed',
        'latest_observations': json.dumps(latest, sort_keys=True, allow_nan=False),
    })
    for name in ds.data_vars:
        ds[name].attrs['units'] = next((unit for ending, unit in
            [('Temperature', 'degC'), ('Humidity', '%'), ('Battery', '%'), ('RSSI', 'dBm')]
            if name.endswith(ending)), '1')
    # A single-row bootstrap otherwise chooses days, corrupting later sub-day
    # appends when xarray infers a finer unit for the new rows.
    ds.time.encoding.update(TIME_ENCODING)
    return ds


def existing_records(ds):
    records = []
    for sensor, prefix in SENSORS.items():
        temp = ds[prefix + 'BLETemperature'].values
        rh = ds[prefix + 'BLEHumidity'].values
        mask = np.isfinite(temp) & np.isfinite(rh)
        for i in np.flatnonzero(mask):
            row = {'time': pd.Timestamp(ds.time.values[i]), 'sensor': sensor,
                   'csv_source': int(ds[prefix + 'BLESourceFormat'].values[i])}
            row.update({key: float(ds[prefix + 'BLE' + suffix].values[i]) for key, suffix in METRICS.items()})
            records.append(row)
    return records


def _sidecar(target, suffix):
    return target.with_name('.' + target.name + suffix + '.tmp')


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _recover_publication(target):
    """A killed two-rename publication leaves its previous store recoverable."""
    previous = _sidecar(target, '.previous')
    if not target.exists() and previous.exists():
        previous.rename(target)
        _fsync_directory(target.parent)
    staging = _sidecar(target, '.staging')
    if staging.exists():
        shutil.rmtree(staging)


def _canonical_layout(group):
    if set(group.array_keys()) != VARIABLES | {'time'} or list(group.group_keys()):
        return False
    for name, array in group.arrays():
        if (array.ndim != 1 or len(array.chunks) != 1 or array.chunks[0] <= 0
                or array.attrs.get('_ARRAY_DIMENSIONS') != ['time']
                or array.dtype != np.dtype('int64' if name == 'time' else 'float32')):
            return False
    time = group['time']
    unit, _, origin = time.attrs.get('units', '').partition(' since ')
    try:
        epoch = pd.Timestamp(origin)
    except ValueError:
        return False
    return (unit == 'microseconds' and epoch == pd.Timestamp('1970-01-01')
            and time.attrs.get('calendar') == 'proleptic_gregorian')


def _open_existing(target):
    """Inspect real array metadata, not a possibly stale consolidated index.

    Older in-place writers can leave unequal lengths. Retain their consistent
    committed prefix and replay raw sources before publishing a repaired store.
    """
    group = zarr.open_group(str(target), mode='r')
    expected = VARIABLES | {'time'}
    if not expected.issubset(group.array_keys()):
        raise ValueError(f'BLE store lacks required arrays: {target}')
    arrays = {name: group[name] for name in expected}
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError(f'BLE store has unsupported dimensions: {target}')
    lengths = [array.shape[0] for array in arrays.values()]
    consolidated = target / '.zmetadata'
    cached_lengths = []
    if consolidated.exists():
        metadata = json.loads(consolidated.read_text()).get('metadata', {})
        cached_lengths = [metadata.get(name + '/.zarray', {}).get('shape', [None])[0]
                          for name in expected]
    repair = len(set(lengths)) != 1 or (cached_lengths and cached_lengths != lengths)
    if repair:
        prefix = min(lengths + [n for n in cached_lengths if isinstance(n, int)])
        variables = {name: xr.Variable(('time',), array[:prefix], attrs=dict(array.attrs))
                     for name, array in arrays.items()}
        time = variables.pop('time')
        ds = xr.decode_cf(xr.Dataset(variables, coords={'time': time}, attrs=dict(group.attrs)))
    else:
        ds = xr.open_zarr(target, chunks=None, consolidated=False)
    times = ds.time.values
    ordered = not np.isnat(times).any() and (len(times) < 2 or np.all(times[1:] > times[:-1]))
    return ds, bool(not repair and ordered and _canonical_layout(group)), bool(repair)


def _validate_candidate(staging, expected_times):
    check, canonical, repair = _open_existing(staging)
    try:
        if repair or not canonical or not np.array_equal(check.time.values, expected_times):
            raise ValueError('BLE candidate failed raw shape, layout or timestamp validation')
        # Also require the published consolidated view to open with all variables.
        with xr.open_zarr(staging, chunks=None, consolidated=True) as consolidated:
            if any(consolidated[name].shape != (len(expected_times),) for name in VARIABLES):
                raise ValueError('BLE candidate has inconsistent consolidated shapes')
    finally:
        check.close()


def _sync_candidate(staging):
    # Historical hard-linked chunks were already synced on their first publish.
    for directory, _, files in os.walk(staging, topdown=False):
        for name in files:
            path = Path(directory) / name
            if path.stat().st_nlink == 1:
                with path.open('rb') as handle:
                    os.fsync(handle.fileno())
        _fsync_directory(directory)


def _publish_candidate(staging, target):
    previous = _sidecar(target, '.previous')
    _sync_candidate(staging)
    # A previous successful publication may have died before cursor commit.
    # Its canonical store has now been read and validated into this candidate.
    if previous.exists():
        shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
        _fsync_directory(target.parent)
    try:
        staging.rename(target)
        _fsync_directory(target.parent)
    except BaseException:
        if not target.exists() and previous.exists():
            previous.rename(target)
            _fsync_directory(target.parent)
        raise


def _write_replacement(ds, target):
    staging = _sidecar(target, '.staging')
    try:
        encoding = {k: {'chunks': (4096,)} for k in ds.variables}
        encoding['time'].update(ds.time.encoding)
        ds.to_zarr(staging, mode='w', consolidated=True, encoding=encoding)
        _validate_candidate(staging, ds.time.values)
        _publish_candidate(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_append(ds, target, existing_times):
    """Copy metadata; share immutable chunks; detach only a partial last chunk."""
    staging = _sidecar(target, '.staging')

    def copy_file(source, destination):
        source = Path(source)
        if source.parent.name in VARIABLES | {'time'} and re.fullmatch(r'\d+', source.name):
            os.link(source, destination)
            return destination
        return shutil.copy2(source, destination)

    try:
        shutil.copytree(target, staging, copy_function=copy_file)
        group = zarr.open_group(str(staging), mode='r')
        for name, array in group.arrays():
            size, chunk = array.shape[0], array.chunks[0]
            if size % chunk:
                path = staging / name / str(size // chunk)
                if path.exists():
                    detached = path.with_name(path.name + '.copy')
                    shutil.copy2(path, detached)
                    os.replace(detached, path)
        ds.to_zarr(staging, mode='a', append_dim='time', consolidated=True)
        _validate_candidate(staging, np.concatenate([existing_times, ds.time.values]))
        _publish_candidate(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def append_new(root=ROOT_DEFAULT, zarr_path=ZARR_DEFAULT):
    root, zarr_path = Path(root), Path(zarr_path)
    if not root.exists():
        raise FileNotFoundError(f'BLE source directory missing: {root}')
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = zarr_path.with_suffix('.ingest.json')
    lock_path = zarr_path.with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _recover_publication(zarr_path)
        state = json.loads(state_path.read_text()) if state_path.exists() and zarr_path.exists() else {}
        cutovers = dict(state.get('csv_cutovers', {}))
        file_states = dict(state.get('files', {}))
        existing = None
        canonical = repair = replay = False
        old_cutovers = {}
        if zarr_path.exists():
            existing, canonical, repair = _open_existing(zarr_path)
            replay = repair or existing.attrs.get('deduplication') != DEDUPLICATION
            old_cutovers = json.loads(existing.attrs.get('csv_cutovers', '{}'))
            for sensor, stamp in old_cutovers.items():
                cutovers[sensor] = min(pd.Timestamp(stamp), pd.Timestamp(cutovers.get(sensor, stamp))).isoformat()
            if replay:
                file_states = {}
        records = []
        rejected = 0
        files = sorted(root.glob('ble_readings_*.csv')) + sorted(root.glob('ble_temperatures_*.log'))
        try:
            for path in files:
                rows, file_state, bad = read_increment(path, file_states.get(path.name))
                records.extend(rows)
                rejected += bad
                file_states[path.name] = file_state
            frame = normalize(records, cutovers)
            if not frame.empty or (existing is not None and (replay or not canonical)):
                append_only = False
                latest = None
                if existing is not None:
                    existing_times = existing.time.values
                    append_only = (canonical and not replay and len(existing_times) > 0 and not frame.empty
                                   and frame.time.min().floor('s') > pd.Timestamp(existing_times[-1]).floor('s')
                                   and old_cutovers == cutovers)
                    if append_only:
                        latest = json.loads(existing.attrs.get('latest_observations', '{}'))
                    else:
                        frame = normalize(existing_records(existing) + frame.to_dict('records'), cutovers)
                if frame.empty:
                    raise ValueError('Cannot recover an empty BLE product; raw sources must be restored')
                ds = to_dataset(frame, cutovers, latest)
                if append_only:
                    _write_append(ds, zarr_path, existing_times)
                else:
                    _write_replacement(ds, zarr_path)
        finally:
            if existing is not None:
                existing.close()
        result = {'files': file_states, 'csv_cutovers': cutovers}
        temporary = state_path.with_suffix('.tmp')
        with temporary.open('w') as handle:
            handle.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state_path)
        _fsync_directory(state_path.parent)
        previous = _sidecar(zarr_path, '.previous')
        if previous.exists():
            shutil.rmtree(previous)
        print(json.dumps({'input_records': len(records), 'accepted_records': len(frame),
                          'rejected_records': rejected, 'source_files': len(files), 'zarr': str(zarr_path)}))
        return {'input_records': len(records), 'accepted_records': len(frame), 'rejected_records': rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--zarr', type=Path, default=ZARR_DEFAULT)
    args = parser.parse_args()
    append_new(args.root, args.zarr)


if __name__ == '__main__':
    main()
