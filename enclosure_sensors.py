"""Read-only BLE enclosure observations and display-only presentation.

These metrics deliberately do not populate the legacy APS internal-humidity
keys consumed by email alerts. Source times are receiver UTC timestamps.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BLESENSORS_ZARR_DEFAULT = Path('/data/aurora/products/blesensors/blesensors.zarr')
SENSORS = (('aps', 'APS enclosure', 'APSBLE'), ('ass', 'ASS enclosure', 'ASSBLE'))
METRICS = {'temperatureC': 'Temperature', 'relativeHumidityPct': 'Humidity', 'batteryPct': 'Battery', 'rssiDbm': 'RSSI'}
SNAPSHOT_METRICS = {'temperatureC': 'temp_c', 'relativeHumidityPct': 'humidity_pct', 'batteryPct': 'battery_pct', 'rssiDbm': 'rssi_dbm', 'dewPointC': 'dewpoint_c', 'dewPointMarginC': 'dewpoint_margin_c'}
BLE_DISPLAY_FIELDS = tuple(prefix + metric for _, _, prefix in SENSORS
                           for metric in ('Temperature', 'Humidity', 'DewPoint', 'DewPointMargin'))


def product_path() -> Path:
    return Path(os.environ.get('BLESENSORS_ZARR_PATH', str(BLESENSORS_ZARR_DEFAULT))).expanduser()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _utc(value):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    except (TypeError, ValueError):
        return None


def dew_point(temperature, humidity):
    temperature, humidity = _number(temperature), _number(humidity)
    if temperature is None or humidity is None or not -100 <= temperature <= 100 or not 0 < humidity <= 100:
        return None
    gamma = math.log(humidity / 100.0) + 17.62 * temperature / (243.12 + temperature)
    return 243.12 * gamma / (17.62 - gamma)


def _card(sensor_id, title, values, observed_at, now):
    stamp, current = _utc(observed_at), _utc(now)
    age = None if stamp is None else (current - stamp).total_seconds() / 60.0
    # Clock errors are not evidence of a fresh observation.
    if age is None or age < -1:
        state, level = 'unavailable', 'unknown'
        age = None
    else:
        age = max(0.0, age)
        state, level = ('fresh', 'ok') if age <= 5 else ('delayed', 'warning') if age <= 15 else ('stale', 'error')
    return {
        'id': sensor_id, 'title': title,
        **{key: _number(values.get(key)) for key in (*METRICS, 'dewPointC', 'dewPointMarginC')},
        'observedAt': stamp.isoformat().replace('+00:00', 'Z') if stamp is not None else None,
        'ageMinutes': age, 'state': state, 'level': level,
        'source': os.environ.get('BLESENSORS_SOURCE_LABEL', 'SwitchBot via ASS') + '; receiver UTC',
    }


def sensors_from_dataset(ds: xr.Dataset, now=None) -> list[dict]:
    now = now or utc_now()
    cards = []
    for sensor_id, title, prefix in SENSORS:
        values, observed_at = {}, None
        tvar, hvar = prefix + 'Temperature', prefix + 'Humidity'
        if 'time' in ds.coords and tvar in ds and hvar in ds and ds[tvar].dims == ds[hvar].dims == ('time',):
            temperature, humidity = np.asarray(ds[tvar].values), np.asarray(ds[hvar].values)
            valid = np.isfinite(temperature) & (temperature >= -100) & (temperature <= 100) & np.isfinite(humidity) & (humidity >= 0) & (humidity <= 100)
            indices = np.flatnonzero(valid & ~pd.isna(ds.time.values))
            if len(indices):
                index = indices[np.argmax(ds.time.values[indices])]
                observed_at = ds.time.values[index]
                for key, suffix in METRICS.items():
                    name = prefix + suffix
                    if name in ds and ds[name].dims == ('time',):
                        values[key] = _number(ds[name].values[index])
                point = dew_point(values.get('temperatureC'), values.get('relativeHumidityPct'))
                values['dewPointC'] = point
                values['dewPointMarginC'] = None if point is None else values['temperatureC'] - point
        cards.append(_card(sensor_id, title, values, observed_at, now))
    return cards


def _metadata_cards(ds: xr.Dataset, now=None) -> list[dict] | None:
    """Read the appender's latest paired rows without scanning historical arrays."""
    raw = ds.attrs.get("latest_observations")
    if raw is None:
        return None
    try:
        latest = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(latest, dict):
        return None
    cards = []
    now = now or utc_now()
    for sensor_id, title, _ in SENSORS:
        row = latest.get(sensor_id) or {}
        if not isinstance(row, dict):
            return None
        temperature, humidity = _number(row.get("temperature_c")), _number(row.get("humidity_pct"))
        values, observed_at = {}, None
        if temperature is not None and humidity is not None and -100 <= temperature <= 100 and 0 <= humidity <= 100:
            point = dew_point(temperature, humidity)
            values = {"temperatureC": temperature, "relativeHumidityPct": humidity,
                      "batteryPct": row.get("battery_pct"), "rssiDbm": row.get("rssi_dbm"),
                      "dewPointC": point, "dewPointMarginC": None if point is None else temperature - point}
            observed_at = row.get("time_utc")
        cards.append(_card(sensor_id, title, values, observed_at, now))
    return cards


def load_enclosure_sensors(now=None) -> list[dict]:
    try:
        with xr.open_zarr(product_path(), chunks=None, consolidated=True) as ds:
            cards = _metadata_cards(ds, now=now)
            return cards if cards is not None else sensors_from_dataset(ds, now=now)
    except (OSError, ValueError, KeyError) as exc:
        logging.getLogger(__name__).debug('BLE enclosure product unavailable: %s', exc)
        return sensors_from_dataset(xr.Dataset(), now=now)


def snapshot_metrics(cards: list[dict]) -> dict:
    result = {}
    for card in cards:
        prefix = card['id'] + '_ble_'
        for source, suffix in SNAPSHOT_METRICS.items():
            result[prefix + suffix] = card[source]
        result[prefix + 'time_utc'] = card['observedAt']
        result[prefix + 'age_min'] = card['ageMinutes']
        result[prefix + 'available_state'] = int(card['observedAt'] is not None)
    return result


def sensors_from_snapshot(snapshot: dict, now=None) -> list[dict]:
    now = now or utc_now()
    return [
        _card(sensor_id, title, {key: snapshot.get(sensor_id + '_ble_' + suffix) for key, suffix in SNAPSHOT_METRICS.items()}, snapshot.get(sensor_id + '_ble_time_utc'), now)
        for sensor_id, title, _ in SENSORS
    ]


def display_dataset(ds: xr.Dataset, freq='1min') -> xr.Dataset:
    """Average each sensor's paired readings independently, preserving gaps."""
    frames = []
    for _, _, prefix in SENSORS:
        temperature, humidity = prefix + 'Temperature', prefix + 'Humidity'
        if temperature not in ds or humidity not in ds:
            continue
        frame = ds[[temperature, humidity]].to_dataframe().sort_index()
        valid = np.isfinite(frame[temperature]) & frame[temperature].between(-100, 100) & np.isfinite(frame[humidity]) & frame[humidity].between(0, 100)
        frame = frame.loc[valid].copy()
        if frame.empty:
            continue
        frame[prefix + 'DewPoint'] = np.asarray([dew_point(t, h) for t, h in zip(frame[temperature], frame[humidity])], dtype=float)
        frame[prefix + 'DewPointMargin'] = frame[temperature] - frame[prefix + 'DewPoint']
        frames.append(frame.resample(freq).mean())
    if not frames:
        return xr.Dataset()
    output = xr.Dataset.from_dataframe(pd.concat(frames, axis=1).sort_index())
    for name in output:
        output[name].attrs['units'] = '%' if name.endswith('Humidity') else 'C'
    return output


def load_display_dataset(window='24h', now=None) -> xr.Dataset:
    now = _utc(now or utc_now()).tz_localize(None)
    start = now - pd.Timedelta(hours=96 if window == '96h' else 24)
    try:
        with xr.open_zarr(product_path(), chunks=None, consolidated=True) as ds:
            fields = [name for name in BLE_DISPLAY_FIELDS if name in ds and name.endswith(('Temperature', 'Humidity'))]
            return display_dataset(ds[fields].sel(time=slice(start, now)).load())
    except (OSError, ValueError, KeyError) as exc:
        logging.getLogger(__name__).debug('BLE enclosure history unavailable: %s', exc)
        return xr.Dataset()


def power_panel(window='24h', now=None, max_points=160) -> dict:
    """Use the existing generic Power panel contract, independently of power.zarr."""
    from grouped_timeseries import ENCLOSURE_CONDITIONS_PANEL
    panel = ENCLOSURE_CONDITIONS_PANEL
    ds = load_display_dataset(window, now=now)
    sensors = load_enclosure_sensors(now=now)
    # Generic native Power panels may be cached: relative age or a freshness
    # claim here would freeze at request time. Operations recomputes live age.
    receptions = []
    for sensor in sensors:
        stamp = _utc(sensor['observedAt'])
        receptions.append(f"{sensor['id'].upper()} last reception: {stamp:%Y-%m-%d %H:%M:%S} UTC."
                          if stamp is not None else f"{sensor['id'].upper()}: no readings received.")
    explanation = f"{panel.description} {sensors[0]['source']}. {' '.join(receptions)}"
    traces = []
    for trace in panel.traces:
        values = np.asarray(ds[trace.var].values, dtype=float) if trace.var in ds else np.array([], dtype=float)
        valid = np.isfinite(values)
        times = pd.DatetimeIndex(ds.time.values[valid]) if len(values) else pd.DatetimeIndex([])
        values = values[valid]
        # Segment before thinning: selected plotting points must not hide a gap.
        segments = np.cumsum(np.r_[0, np.diff(times.asi8) > 90_000_000_000])
        keep = np.unique(np.linspace(0, len(times) - 1, min(len(times), max_points), dtype=int)) if len(times) else []
        traces.append({
            'id': trace.var, 'label': trace.label, 'color': trace.color, 'axis': trace.axis,
            'unit': '%' if trace.var.endswith('Humidity') else 'C', 'step': False,
            'points': [{'time': times[i].isoformat() + 'Z', 'value': float(values[i]), 'segment': int(segments[i])} for i in keep],
        })
    return {'id': panel.key, 'title': panel.label, 'explanation': explanation,
            'leftAxisLabel': panel.left_axis_label, 'rightAxisLabel': panel.right_axis_label, 'traces': traces}
