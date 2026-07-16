#!/usr/bin/env python3
"""Generate an ECMWF-informed Aurora Power Supply SOC forecast product."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from power_soc_thresholds import MINIMUM_OPERATIONAL_SOC_PCT

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
POWER_SOC_FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
POWER_ECMWF_FORECAST_CACHE_DIR = Path(
    os.environ.get("POWER_ECMWF_FORECAST_CACHE_DIR", "/data/aurora/products/power/ecmwf_solar_forecast")
)
POWER_SOC_FORECAST_STATE_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_STATE_PATH", "/data/aurora/products/power/power_soc_forecast_state.json")
)
POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast_archive.zarr")
)
POWER_SOC_FORECAST_SKILL_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_SKILL_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast_skill.zarr")
)
POWER_SOC_HINDCAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_HINDCAST_ZARR_PATH", "/data/aurora/products/power/power_soc_hindcast.zarr")
)
POWER_PDU_ZARR_PATH = Path(os.environ.get("POWER_PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))

DEFAULT_LATITUDE = float(os.environ.get("AURORA_POWER_FORECAST_LATITUDE", "64.829694"))
DEFAULT_LONGITUDE = float(os.environ.get("AURORA_POWER_FORECAST_LONGITUDE", "-23.248139"))
DEFAULT_HORIZON_HOURS = int(os.environ.get("AURORA_POWER_SOC_FORECAST_HOURS", "96"))
DEFAULT_CALIBRATION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_CALIBRATION_DAYS", "7"))
DEFAULT_FALLBACK_CALIBRATION_HOURS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_FALLBACK_CALIBRATION_HOURS", "48"))
DEFAULT_BATTERY_CAPACITY_KWH = float(os.environ.get("APS_BATTERY_CAPACITY_KWH", "26"))
DEFAULT_SOLAR_CALIBRATION_FACTOR = float(os.environ.get("AURORA_POWER_SOLAR_CALIBRATION_FACTOR", "1.0"))
DEFAULT_LOAD_W = float(os.environ.get("AURORA_POWER_FORECAST_DEFAULT_LOAD_W", "0"))
DEFAULT_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_ADAPTIVE_ALPHA", "0.25"))
DEFAULT_MIN_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_MIN_ADAPTIVE_ALPHA", "0.08"))
DEFAULT_MAX_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_MAX_ADAPTIVE_ALPHA", "0.45"))
DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W = float(os.environ.get("AURORA_POWER_LOAD_BIAS_CORRECTION_LIMIT_W", "2000"))
DEFAULT_LOAD_NEGATIVE_CORRECTION_FRACTION_LIMIT = float(
    os.environ.get("AURORA_POWER_LOAD_NEGATIVE_CORRECTION_FRACTION_LIMIT", "0.25")
)
DEFAULT_LOAD_MODE_LEVEL_HOURS = float(os.environ.get("AURORA_POWER_LOAD_MODE_LEVEL_HOURS", "2"))
DEFAULT_LOAD_MODE_STATE_MINUTES = float(os.environ.get("AURORA_POWER_LOAD_MODE_STATE_MINUTES", "30"))
DEFAULT_LOAD_MODE_LEARN_INTERVAL_MINUTES = float(
    os.environ.get("AURORA_POWER_LOAD_MODE_LEARN_INTERVAL_MINUTES", "60")
)
DEFAULT_AC_MODE_THRESHOLD_W = float(os.environ.get("AURORA_POWER_AC_MODE_THRESHOLD_W", "25"))
DEFAULT_PDU_MODE_FRESHNESS_MINUTES = float(os.environ.get("AURORA_POWER_PDU_MODE_FRESHNESS_MINUTES", "60"))
DEFAULT_PDU_ACTIVE_W_THRESHOLD = float(os.environ.get("AURORA_POWER_PDU_ACTIVE_W_THRESHOLD", "5"))
DEFAULT_LOAD_MODE_MIN_STABLE_SAMPLES = int(os.environ.get("AURORA_POWER_LOAD_MODE_MIN_STABLE_SAMPLES", "2"))
DEFAULT_ZERO_SOLAR_THRESHOLD_W = float(os.environ.get("AURORA_POWER_ZERO_SOLAR_THRESHOLD_W", "10"))
DEFAULT_DARK_LOAD_LOOKBACK_HOURS = float(os.environ.get("AURORA_POWER_DARK_LOAD_LOOKBACK_HOURS", "48"))
DEFAULT_SOC_BIAS_CORRECTION_LIMIT = float(os.environ.get("AURORA_POWER_SOC_BIAS_CORRECTION_LIMIT", "8"))
DEFAULT_ARCHIVE_RETENTION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_ARCHIVE_RETENTION_DAYS", "21"))
DEFAULT_ECMWF_LOOKAHEAD_BUFFER_HOURS = int(os.environ.get("AURORA_POWER_ECMWF_LOOKAHEAD_BUFFER_HOURS", "24"))
DEFAULT_OPEN_DATA_SOURCE = os.environ.get("AURORA_POWER_ECMWF_OPEN_DATA_SOURCE", "azure")
DEFAULT_SKILL_WINDOW_HOURS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_SKILL_WINDOW_HOURS", "24"))
DEFAULT_SKILL_RETENTION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_SKILL_RETENTION_DAYS", "7"))
ECMWF_PARAM = "ssrd"
LOAD_MODEL_NAME = "kit_mode_persistence_v4"
LOAD_MODEL_VERSION = 4
PDU_OUTLET_KIT_NAMES = {
    4: "UAS",
    5: "CL61",
    6: "Radar",
    8: "HATPRO",
}
PDU_KIT_OUTLETS = {name: outlet for outlet, name in PDU_OUTLET_KIT_NAMES.items()}
LEAD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0_6h", 0.0, 6.0),
    ("6_24h", 6.0, 24.0),
    ("24_48h", 24.0, 48.0),
    ("48_96h", 48.0, 96.0),
)
ARCHIVE_FORECAST_FIELDS = (
    "BatterySOCForecast",
    "ECMWFSolarIrradiance",
    "ForecastSolarWatts",
    "ForecastLoadWatts",
)
SCENARIO_LOADS_W = (100, 200, 300, 400, 500, 600)
HINDCAST_LEAD_HOURS = (6, 24, 48, 72)
DEFAULT_HINDCAST_RETENTION_DAYS = float(os.environ.get("AURORA_POWER_SOC_HINDCAST_RETENTION_DAYS", "7"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_zarr(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.chunk({"time": min(max(ds.sizes.get("time", 1), 1), 288)}).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(path)


def _atomic_write_archive(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk_spec = {}
    if "issue_time" in ds.sizes:
        chunk_spec["issue_time"] = min(max(ds.sizes.get("issue_time", 1), 1), 64)
    if "forecast_step" in ds.sizes:
        chunk_spec["forecast_step"] = min(max(ds.sizes.get("forecast_step", 1), 1), 64)
    if "ForecastValidTime" in ds:
        ds["ForecastValidTime"].encoding["units"] = "nanoseconds since 1970-01-01"
        ds["ForecastValidTime"].encoding["dtype"] = "int64"
    ds.chunk(chunk_spec).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _atomic_write_skill(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk_spec = {}
    if "time" in ds.sizes:
        chunk_spec["time"] = min(max(ds.sizes.get("time", 1), 1), 288)
    ds.chunk(chunk_spec).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _atomic_write_time_product(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk = min(max(ds.sizes.get("time", 1), 1), 672)
    ds.chunk({"time": chunk}).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _balanced_alpha(sample_count: object, *, base: float = DEFAULT_ADAPTIVE_ALPHA) -> float:
    try:
        samples = max(float(sample_count), 0.0)
    except Exception:
        samples = 0.0
    if samples <= 0.0:
        return float(base)
    early = DEFAULT_MAX_ADAPTIVE_ALPHA / np.sqrt(samples)
    return float(np.clip(max(float(base), early), DEFAULT_MIN_ADAPTIVE_ALPHA, DEFAULT_MAX_ADAPTIVE_ALPHA))


def _latest_cached_forecast(cache_dir: Path, *, param: str = ECMWF_PARAM) -> Path:
    patterns = (f"*{param}*.grib2", f"*{param}*.grib", "*.grib2", "*.grib")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(cache_dir.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No cached ECMWF GRIB files found in {cache_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def retrieve_open_data_grib(
    output_grib: Path,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    lookahead_buffer_hours: int = DEFAULT_ECMWF_LOOKAHEAD_BUFFER_HOURS,
    param: str = ECMWF_PARAM,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
) -> Path:
    """Retrieve ECMWF open-data solar forecast GRIB for the requested horizon."""
    try:
        from ecmwf.opendata import Client
    except Exception as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError("Install ecmwf-opendata to retrieve ECMWF open-data forecasts") from exc

    requested_horizon = int(horizon_hours) + max(int(lookahead_buffer_hours), 0)
    steps = list(range(0, requested_horizon + 1, 3))
    if steps[-1] != requested_horizon:
        steps.append(requested_horizon)
    output_grib.parent.mkdir(parents=True, exist_ok=True)
    client = Client(source=source)
    client.retrieve(
        type="fc",
        stream="oper",
        levtype="sfc",
        param=param,
        step=steps,
        target=str(output_grib),
    )
    return output_grib


def open_solar_forecast(path: Path, *, latitude: float, longitude: float) -> xr.Dataset:
    """Open a GRIB/NetCDF solar forecast and select the nearest site point."""
    suffix = path.suffix.lower()
    if suffix in {".grib", ".grb", ".grib2", ".grb2"}:
        ds = xr.open_dataset(path, engine="cfgrib")
    else:
        ds = xr.open_dataset(path)

    if "ssrd" not in ds and "surface_solar_radiation_downwards" in ds:
        ds = ds.rename({"surface_solar_radiation_downwards": "ssrd"})
    if "ssrd" not in ds:
        raise KeyError("ECMWF solar file does not contain ssrd/surface_solar_radiation_downwards")

    lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
    lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
    if lat_name is not None and lon_name is not None:
        lon_values = ds[lon_name].values
        select_lon = longitude
        if np.nanmin(lon_values) >= 0.0 and select_lon < 0.0:
            select_lon = select_lon % 360.0
        ds = ds.sel({lat_name: latitude, lon_name: select_lon}, method="nearest")
    return ds


def _ecmwf_cycle_time(ds: xr.Dataset) -> pd.Timestamp | None:
    if "time" not in ds.coords:
        return None
    values = np.asarray(ds["time"].values).reshape(-1)
    if values.size == 0:
        return None
    cycle = pd.Timestamp(values[0])
    if pd.isna(cycle):
        return None
    if cycle.tz is not None:
        cycle = cycle.tz_convert("UTC").tz_localize(None)
    return cycle


def solar_irradiance_from_ssrd(ds: xr.Dataset) -> pd.Series:
    """Convert accumulated ECMWF SSRD J/m2 to interval W/m2."""
    da = ds["ssrd"]
    valid_time = ds["valid_time"] if "valid_time" in ds.coords else None
    if valid_time is None:
        if "time" in da.coords and "step" in da.coords:
            valid_time = da["time"] + da["step"]
        elif "time" in da.coords:
            valid_time = da["time"]
        else:
            raise KeyError("Solar forecast needs time or valid_time coordinates")

    values = np.asarray(da.values, dtype=np.float64).reshape(-1)
    times = pd.DatetimeIndex(np.asarray(valid_time.values).reshape(-1))
    frame = pd.DataFrame({"ssrd": values}, index=times).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[np.isfinite(frame["ssrd"])]
    if len(frame) < 2:
        return pd.Series(dtype=np.float64)

    delta_j_m2 = frame["ssrd"].diff()
    delta_seconds = frame.index.to_series().diff().dt.total_seconds()
    irradiance = delta_j_m2 / delta_seconds
    irradiance = irradiance.clip(lower=0.0)
    return irradiance.dropna()


def _power_frame(power: xr.Dataset) -> pd.DataFrame:
    fields = [
        name
        for name in (
            "BatterySOC",
            "SolarWatts_East",
            "SolarWatts_South",
            "SolarWatts_West",
            "BatteryWatts",
            "ACOutputWatts",
            "DCInverterWatts",
        )
        if name in power and power[name].dims == ("time",)
    ]
    if "time" not in power or not fields:
        return pd.DataFrame()
    times = pd.DatetimeIndex(power["time"].values)
    frame = pd.DataFrame({name: np.asarray(power[name].values, dtype=np.float64) for name in fields}, index=times)
    frame = frame[~frame.index.isna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def latest_finite(series: pd.Series) -> tuple[pd.Timestamp, float]:
    finite = series[np.isfinite(series)]
    if finite.empty:
        raise ValueError(f"No finite samples available for {series.name or 'series'}")
    return pd.Timestamp(finite.index[-1]), float(finite.iloc[-1])


def estimate_load_w(frame: pd.DataFrame, *, end: pd.Timestamp, calibration_days: float) -> float:
    start = end - pd.Timedelta(days=float(calibration_days))
    window = frame.loc[frame.index >= start]
    load = _observed_load_w(window)
    finite = load[np.isfinite(load)]
    if finite.empty:
        return float(DEFAULT_LOAD_W)
    return float(finite.median())


def build_historical_load_forecast(
    frame: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    *,
    end: pd.Timestamp,
    calibration_days: float,
    default_load_w: float = DEFAULT_LOAD_W,
) -> pd.Series:
    """Forecast total station load from the robust level of its current operating mode."""
    forecast_times = pd.DatetimeIndex(forecast_times)
    if len(forecast_times) == 0:
        return pd.Series(dtype=np.float64)
    start = end - pd.Timedelta(days=float(calibration_days))
    load = _observed_load_w(frame)
    load = load.loc[(load.index >= start) & (load.index <= end)]
    finite = load[np.isfinite(load)]
    if finite.empty:
        return pd.Series(np.full(len(forecast_times), float(default_load_w)), index=forecast_times)

    samples = finite.resample("15min").median().dropna()
    if samples.empty:
        samples = finite
    state_window_start = end - pd.Timedelta(minutes=float(DEFAULT_LOAD_MODE_STATE_MINUTES))
    state_window = samples[samples.index >= state_window_start]
    if state_window.empty:
        state_window = samples.iloc[-min(len(samples), 2) :]

    ac = frame.get("ACOutputWatts")
    if ac is not None:
        ac_samples = ac.loc[(ac.index >= start) & (ac.index <= end)].resample("15min").median()
        ac_samples = ac_samples.reindex(samples.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        recent_ac = ac_samples[ac_samples.index >= state_window_start].dropna()
        current_ac_w = float(recent_ac.median()) if not recent_ac.empty else float(ac_samples.dropna().iloc[-1])
        current_active = bool(current_ac_w > DEFAULT_AC_MODE_THRESHOLD_W)
        states = ac_samples.fillna(current_ac_w).to_numpy(dtype=np.float64) > DEFAULT_AC_MODE_THRESHOLD_W
        mode_state = "ac-active" if current_active else "dc-only"
        split_w = float(DEFAULT_AC_MODE_THRESHOLD_W)
    else:
        values = samples.to_numpy(dtype=np.float64)
        split_w = _load_regime_threshold(values)
        current_reference = float(state_window.median())
        current_active = bool(np.isfinite(split_w) and current_reference > split_w)
        states = values > split_w if np.isfinite(split_w) else np.zeros(len(values), dtype=bool)
        current_ac_w = np.nan
        mode_state = "unlabelled-active" if current_active else "dc-only"

    opposite = np.flatnonzero(states != current_active)
    run_start_index = int(opposite[-1] + 1) if opposite.size else 0
    current_run = samples.iloc[run_start_index:]
    same_mode = samples.iloc[np.flatnonzero(states == current_active)]

    level_start = end - pd.Timedelta(hours=float(DEFAULT_LOAD_MODE_LEVEL_HOURS))
    level_samples = current_run[current_run.index >= level_start]
    measurement = _load_measurement_name(frame)
    level_measurement = measurement
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if measurement == "solar_generation_minus_battery_power" and len(solar_fields) == 3:
        solar_total = frame[solar_fields].sum(axis=1, min_count=3).resample("15min").median()
        battery_power = frame["BatteryWatts"].resample("15min").median()
        solar_total = solar_total.reindex(samples.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        battery_power = battery_power.reindex(samples.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        dark_mask = (solar_total <= DEFAULT_ZERO_SOLAR_THRESHOLD_W) & (battery_power < 0.0)
        dark_load = (-battery_power[dark_mask]).dropna()
        dark_run = dark_load[dark_load.index >= current_run.index[0]]
        dark_run = dark_run[dark_run.index >= end - pd.Timedelta(hours=float(DEFAULT_DARK_LOAD_LOOKBACK_HOURS))]
        if len(dark_run) >= 4:
            level_samples = dark_run
            level_measurement = "battery_discharge_when_solar_zero"
    if len(level_samples) < 4:
        level_samples = same_mode[same_mode.index >= level_start]
    if level_samples.empty:
        level_samples = state_window
    level = float(level_samples.median()) if not level_samples.empty else float(default_load_w)
    level = max(level, 0.0) if np.isfinite(level) else float(default_load_w)
    run_hours = max(float((end - current_run.index[0]) / pd.Timedelta(hours=1)), 0.0) if len(current_run) else 0.0
    forecast = pd.Series(np.full(len(forecast_times), level, dtype=np.float64), index=forecast_times)
    mode_name = "AC Load (Unlabelled)" if current_active else "DC-Only"
    forecast.attrs.update(
        {
            "load_model": LOAD_MODEL_NAME,
            "load_model_version": LOAD_MODEL_VERSION,
            "load_mode": mode_name,
            "load_mode_state": mode_state,
            "load_mode_source": "ac_output" if ac is not None else "load_level",
            "load_measurement": level_measurement,
            "load_balance_measurement": measurement,
            "load_current_ac_w": current_ac_w,
            "load_regime": mode_name,
            "load_regime_threshold_w": split_w,
            "load_regime_level_w": level,
            "load_regime_run_hours": run_hours,
            "load_regime_sample_count": int(len(level_samples)),
        }
    )
    return forecast


def _load_regime_threshold(values: np.ndarray) -> float:
    """Find a material low/high load split using the largest gap in log power."""
    finite = np.sort(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite) & (finite >= 0.0)]
    if finite.size < 20:
        return np.nan
    low_quantile, high_quantile = np.nanquantile(finite, [0.10, 0.90])
    candidates = np.unique(finite[(finite >= low_quantile) & (finite <= high_quantile)])
    if candidates.size < 2:
        return np.nan
    log_values = np.log1p(candidates)
    gap_index = int(np.argmax(np.diff(log_values)))
    lower = float(candidates[gap_index])
    upper = float(candidates[gap_index + 1])
    if (upper + 1.0) / (lower + 1.0) < 3.0:
        return np.nan
    return float(np.expm1((np.log1p(lower) + np.log1p(upper)) / 2.0))


def _normalise_load_mode_registry(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    registry: dict[str, dict[str, object]] = {}
    for label, raw_entry in value.items():
        if not isinstance(label, str) or not label or not isinstance(raw_entry, dict):
            continue
        observations = []
        for raw_observation in raw_entry.get("observations", []):
            if not isinstance(raw_observation, dict):
                continue
            try:
                timestamp = pd.Timestamp(raw_observation["time"])
                level_w = float(raw_observation["level_w"])
            except Exception:
                continue
            if pd.isna(timestamp) or not np.isfinite(level_w) or level_w < 0.0:
                continue
            observations.append({"time": timestamp.isoformat(), "level_w": level_w})
        entry = dict(raw_entry)
        entry["observations"] = observations[-168:]
        registry[label] = entry
    return registry


def _pdu_active_kits(
    pdu: xr.Dataset | None,
    *,
    end: pd.Timestamp,
) -> tuple[list[str], pd.Timestamp | None, float]:
    if pdu is None or "time" not in pdu or pdu.sizes.get("time", 0) == 0:
        return [], None, np.nan
    times = pd.DatetimeIndex(pdu["time"].values)
    eligible = np.flatnonzero((~times.isna()) & (times <= end))
    if not eligible.size:
        return [], None, np.nan
    latest_index = int(eligible[np.argmax(times[eligible].to_numpy(dtype="datetime64[ns]"))])
    latest_time = pd.Timestamp(times[latest_index])
    if end - latest_time > pd.Timedelta(minutes=float(DEFAULT_PDU_MODE_FRESHNESS_MINUTES)):
        return [], latest_time, np.nan
    window_indices = eligible[times[eligible] >= latest_time - pd.Timedelta(minutes=30)]
    active: list[str] = []
    active_watts = 0.0
    for outlet in range(1, 9):
        is_active = False
        state_name = f"PDUOutlet{outlet}State"
        watts_name = f"PDUOutlet{outlet}Watts"
        watts_level = np.nan
        watts_available = False
        if watts_name in pdu:
            watts = np.asarray(pdu[watts_name].isel(time=window_indices).values, dtype=np.float64)
            watts = watts[np.isfinite(watts)]
            if watts.size:
                watts_level = float(np.nanmedian(watts))
                watts_available = True
                is_active = bool(watts_level >= DEFAULT_PDU_ACTIVE_W_THRESHOLD)
        if not watts_available and state_name in pdu:
            states = np.asarray(pdu[state_name].isel(time=window_indices).values, dtype=np.float64)
            states = states[np.isfinite(states)]
            is_active = bool(states.size and np.nanmedian(states) >= 0.5)
        if is_active:
            active.append(PDU_OUTLET_KIT_NAMES.get(outlet, f"AC Outlet {outlet}"))
            if np.isfinite(watts_level):
                active_watts += max(watts_level, 0.0)
    return active, latest_time, active_watts if active else np.nan


def _resolve_load_mode(
    frame: pd.DataFrame,
    pdu: xr.Dataset | None,
    *,
    end: pd.Timestamp,
    observed_level_w: float,
    raw_registry: object,
    previous_mode: object,
) -> tuple[str, str, list[str], pd.Timestamp | None, float]:
    state_start = end - pd.Timedelta(minutes=float(DEFAULT_LOAD_MODE_STATE_MINUTES))
    recent_ac = frame.loc[frame.index >= state_start, "ACOutputWatts"].dropna() if "ACOutputWatts" in frame else pd.Series(dtype=float)
    current_ac_w = float(recent_ac.median()) if not recent_ac.empty else 0.0
    latest_ac_w = float(recent_ac.iloc[-1]) if not recent_ac.empty else current_ac_w

    active_kits, pdu_time, pdu_active_watts = _pdu_active_kits(pdu, end=end)
    if active_kits:
        # Use the latest AC sample for immediate PDU classification. The
        # smoothed level remains responsible for sustained-mode learning.
        if "CL61" in active_kits and latest_ac_w > DEFAULT_AC_MODE_THRESHOLD_W:
            return "Ceilometer-on-AC", "pdu_ac_signature", active_kits, pdu_time, pdu_active_watts
        return f"DC-Only + {' + '.join(active_kits)}", "pdu_signature", active_kits, pdu_time, pdu_active_watts
    if current_ac_w <= DEFAULT_AC_MODE_THRESHOLD_W:
        return "DC-Only", "ac_output", [], pdu_time, np.nan

    registry = _normalise_load_mode_registry(raw_registry)
    candidates: list[tuple[float, str]] = []
    for label, entry in registry.items():
        if label == "DC-Only":
            continue
        try:
            learned_level = float(entry.get("learned_level_w", np.nan))
        except Exception:
            continue
        if np.isfinite(learned_level):
            candidates.append((abs(learned_level - observed_level_w), label))
    if candidates:
        difference, label = min(candidates)
        if difference <= max(75.0, 0.20 * max(observed_level_w, 1.0)):
            return label, "learned_power_match", [], pdu_time, np.nan
    if isinstance(previous_mode, str) and previous_mode != "DC-Only" and previous_mode in registry:
        return previous_mode, "persisted_mode", [], pdu_time, np.nan
    return "AC Load (Unlabelled)", "ac_output", [], pdu_time, np.nan


def _load_mode_signature(mode: str, mode_source: str, active_kits: list[str]) -> str:
    if mode == "Ceilometer-on-AC" and "CL61" in active_kits:
        return (
            f"PDUOutlet{PDU_KIT_OUTLETS['CL61']}Watts>={DEFAULT_PDU_ACTIVE_W_THRESHOLD:g}W+"
            f"ACOutputWatts>{DEFAULT_AC_MODE_THRESHOLD_W:g}W"
        )
    if mode_source == "pdu_signature" and active_kits:
        parts = []
        for kit in active_kits:
            outlet = PDU_KIT_OUTLETS.get(kit)
            if outlet is None:
                parts.append(kit)
            else:
                parts.append(f"PDUOutlet{outlet}Watts>={DEFAULT_PDU_ACTIVE_W_THRESHOLD:g}W")
        return "+".join(parts)
    if mode == "DC-Only":
        return f"ACOutputWatts<={DEFAULT_AC_MODE_THRESHOLD_W:g}W"
    if mode_source == "learned_power_match":
        return "learned_total_load_match"
    return f"ACOutputWatts>{DEFAULT_AC_MODE_THRESHOLD_W:g}W"


def _mode_learning_status(load_diagnostics: dict[str, object], mode: str) -> tuple[bool, str]:
    state = str(load_diagnostics.get("load_mode_state", ""))
    expects_ac = mode != "DC-Only"
    active_states = {"ac-active", "unlabelled-active"}
    if expects_ac and state not in active_states:
        return False, "waiting_for_ac_state"
    if not expects_ac and state != "dc-only":
        return False, "waiting_for_dc_state"
    run_hours = float(load_diagnostics.get("load_regime_run_hours", 0.0) or 0.0)
    if run_hours * 60.0 < float(DEFAULT_LOAD_MODE_STATE_MINUTES):
        return False, "waiting_for_stable_duration"
    sample_count = int(load_diagnostics.get("load_regime_sample_count", 0) or 0)
    if sample_count < DEFAULT_LOAD_MODE_MIN_STABLE_SAMPLES:
        return False, "waiting_for_stable_samples"
    return True, "stable"


def _update_load_mode_registry(
    raw_registry: object,
    *,
    mode: str,
    observed_level_w: float,
    observed_at: pd.Timestamp,
    active_kits: list[str] | None = None,
    mode_source: str | None = None,
    signature: str | None = None,
) -> tuple[dict[str, dict[str, object]], float]:
    registry = _normalise_load_mode_registry(raw_registry)
    entry = dict(registry.get(mode, {}))
    observations = list(entry.get("observations", []))
    observation = {"time": pd.Timestamp(observed_at).isoformat(), "level_w": float(observed_level_w)}
    if observations:
        last_time = pd.Timestamp(observations[-1]["time"])
        if observed_at - last_time < pd.Timedelta(minutes=float(DEFAULT_LOAD_MODE_LEARN_INTERVAL_MINUTES)):
            observations[-1] = observation
        else:
            observations.append(observation)
    else:
        observations.append(observation)
    observations = observations[-168:]
    levels = np.asarray([item["level_w"] for item in observations], dtype=np.float64)
    learned_level = float(np.nanmedian(levels))
    entry.update(
        {
            "learned_level_w": learned_level,
            "latest_observed_level_w": float(observed_level_w),
            "observation_count": int(len(observations)),
            "last_seen": pd.Timestamp(observed_at).isoformat(),
            "observations": observations,
        }
    )
    if active_kits is not None:
        entry["active_kits"] = list(active_kits)
    if mode_source:
        entry["mode_source"] = str(mode_source)
    if signature:
        entry["signature"] = str(signature)
    registry[mode] = entry
    return registry, learned_level


def _observed_solar_w(frame: pd.DataFrame) -> pd.Series:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if not solar_fields:
        return pd.Series(dtype=np.float64)
    return frame[solar_fields].sum(axis=1, min_count=1).clip(lower=0.0)


def _observed_load_w(frame: pd.DataFrame) -> pd.Series:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if "BatteryWatts" in frame and len(solar_fields) == 3:
        solar = frame[solar_fields].sum(axis=1, min_count=len(solar_fields))
        # APS BatteryWatts is positive while charging and negative while
        # discharging, so generation minus battery flow is total station load.
        balanced = (solar - frame["BatteryWatts"]).clip(lower=0.0)
        if np.isfinite(balanced.to_numpy(dtype=np.float64)).any():
            return balanced
    load_fields = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in frame]
    if not load_fields:
        return pd.Series(dtype=np.float64)
    return frame[load_fields].sum(axis=1, min_count=1).clip(lower=0.0)


def _clean_dc_only_observation(
    frame: pd.DataFrame,
    *,
    end: pd.Timestamp,
) -> tuple[float, pd.Timestamp, int] | None:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    required = {"BatteryWatts", "ACOutputWatts"}
    if len(solar_fields) != 3 or not required.issubset(frame.columns):
        return None
    start = end - pd.Timedelta(hours=float(DEFAULT_DARK_LOAD_LOOKBACK_HOURS))
    samples = pd.DataFrame(
        {
            "load_w": _observed_load_w(frame),
            "solar_w": frame[solar_fields].sum(axis=1, min_count=3),
            "battery_w": frame["BatteryWatts"],
            "ac_w": frame["ACOutputWatts"],
        }
    ).loc[start:end]
    samples = samples.resample("15min").median()
    clean = samples.loc[
        (samples["solar_w"] <= DEFAULT_ZERO_SOLAR_THRESHOLD_W)
        & (samples["battery_w"] < 0.0)
        & (samples["ac_w"] <= DEFAULT_AC_MODE_THRESHOLD_W),
        "load_w",
    ].dropna()
    if len(clean) < 4:
        return None
    return float(clean.median()), pd.Timestamp(clean.index[-1]), int(len(clean))


def _repair_dc_only_registry(
    raw_registry: object,
    frame: pd.DataFrame,
    *,
    end: pd.Timestamp,
) -> tuple[dict[str, dict[str, object]], float | None]:
    registry = _normalise_load_mode_registry(raw_registry)
    clean = _clean_dc_only_observation(frame, end=end)
    if clean is None:
        return registry, None
    level_w, observed_at, sample_count = clean
    tolerance_w = max(75.0, 0.35 * max(level_w, 1.0))
    entry = dict(registry.get("DC-Only", {}))
    observations = [
        observation
        for observation in entry.get("observations", [])
        if abs(float(observation["level_w"]) - level_w) <= tolerance_w
    ]
    replacement = {"time": observed_at.isoformat(), "level_w": level_w}
    replaced = False
    for index, observation in enumerate(observations):
        if abs(pd.Timestamp(observation["time"]) - observed_at) < pd.Timedelta(
            minutes=float(DEFAULT_LOAD_MODE_LEARN_INTERVAL_MINUTES)
        ):
            observations[index] = replacement
            replaced = True
            break
    if not replaced:
        observations.append(replacement)
    observations.sort(key=lambda observation: pd.Timestamp(observation["time"]))
    observations = observations[-168:]
    levels = np.asarray([observation["level_w"] for observation in observations], dtype=np.float64)
    latest = observations[-1]
    entry.update(
        {
            "learned_level_w": float(np.nanmedian(levels)),
            "latest_observed_level_w": float(latest["level_w"]),
            "observation_count": int(len(observations)),
            "last_seen": str(latest["time"]),
            "observations": observations,
            "active_kits": [],
            "mode_source": "battery_discharge_when_solar_zero",
            "signature": (
                f"ACOutputWatts<={DEFAULT_AC_MODE_THRESHOLD_W:g}W+"
                f"SolarTotalWatts<={DEFAULT_ZERO_SOLAR_THRESHOLD_W:g}W"
            ),
            "clean_dark_sample_count": sample_count,
        }
    )
    registry["DC-Only"] = entry
    return registry, level_w


def _load_measurement_name(frame: pd.DataFrame) -> str:
    solar_fields = {"SolarWatts_East", "SolarWatts_South", "SolarWatts_West"}
    if "BatteryWatts" in frame and solar_fields.issubset(frame.columns):
        return "solar_generation_minus_battery_power"
    return "ac_plus_dc_output_fallback"


def evaluate_previous_forecast(previous: xr.Dataset | None, frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Score the previous forecast against newly arrived APS observations."""
    if previous is None or "time" not in previous or previous.sizes.get("time", 0) == 0 or frame.empty:
        return {}
    forecast_times = pd.DatetimeIndex(previous["time"].values)
    observed_end = pd.Timestamp(frame.index.max())
    valid_forecast = forecast_times <= observed_end
    if not np.any(valid_forecast):
        return {}

    metrics: dict[str, float | int | str] = {}
    if "BatterySOCForecast" in previous and "BatterySOC" in frame:
        forecast_soc = pd.Series(np.asarray(previous["BatterySOCForecast"].values, dtype=np.float64), index=forecast_times)
        forecast_soc = forecast_soc.loc[valid_forecast]
        observed_soc = frame["BatterySOC"].reindex(forecast_soc.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        valid = np.isfinite(forecast_soc.to_numpy()) & np.isfinite(observed_soc.to_numpy())
        if np.count_nonzero(valid) >= 2:
            errors = forecast_soc.to_numpy()[valid] - observed_soc.to_numpy()[valid]
            metrics["soc_mae_pct_points"] = float(np.mean(np.abs(errors)))
            metrics["soc_bias_pct_points"] = float(np.mean(errors))
            metrics["soc_sample_count"] = int(np.count_nonzero(valid))

    if "ForecastSolarWatts" in previous:
        forecast_solar = pd.Series(np.asarray(previous["ForecastSolarWatts"].values, dtype=np.float64), index=forecast_times)
        forecast_solar = forecast_solar.loc[valid_forecast]
        observed_solar = _observed_solar_w(frame).reindex(forecast_solar.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        valid = np.isfinite(forecast_solar.to_numpy()) & np.isfinite(observed_solar.to_numpy())
        if np.count_nonzero(valid) >= 2:
            errors = forecast_solar.to_numpy()[valid] - observed_solar.to_numpy()[valid]
            metrics["solar_mae_w"] = float(np.mean(np.abs(errors)))
            metrics["solar_bias_w"] = float(np.mean(errors))
            metrics["solar_sample_count"] = int(np.count_nonzero(valid))

    if "ForecastLoadWatts" in previous:
        forecast_load = pd.Series(np.asarray(previous["ForecastLoadWatts"].values, dtype=np.float64), index=forecast_times)
        forecast_load = forecast_load.loc[valid_forecast]
        observed_load = _observed_load_w(frame).reindex(forecast_load.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        valid = np.isfinite(forecast_load.to_numpy()) & np.isfinite(observed_load.to_numpy())
        if np.count_nonzero(valid) >= 2:
            errors = forecast_load.to_numpy()[valid] - observed_load.to_numpy()[valid]
            metrics["load_mae_w"] = float(np.mean(np.abs(errors)))
            metrics["load_bias_w"] = float(np.mean(errors))
            metrics["load_sample_count"] = int(np.count_nonzero(valid))

    if metrics:
        metrics["evaluated_at_utc"] = _utc_now()
    return metrics


def _metric_bucket_name(metric: str, bucket: str) -> str:
    return f"{metric}_{bucket}"


def scenario_soc_field(load_w: int | float) -> str:
    return f"BatterySOCForecast_Load{int(load_w)}W"


def scenario_load_label(load_w: int | float) -> str:
    return f"{int(load_w)} W Load"


def _errors_for_archive_variable(
    archive: xr.Dataset,
    frame: pd.DataFrame,
    *,
    forecast_var: str,
    observed: pd.Series,
    tolerance: pd.Timedelta,
    load_model_version: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if forecast_var not in archive or "ForecastValidTime" not in archive or "ForecastLeadHours" not in archive:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    forecast_values = np.asarray(archive[forecast_var].values, dtype=np.float64).reshape(-1)
    valid_times = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    lead_hours = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    model_versions = None
    if load_model_version is not None:
        if "LoadModelVersion" not in archive:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        model_versions = np.repeat(
            np.asarray(archive["LoadModelVersion"].values, dtype=np.float64).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    observed_end = pd.Timestamp(frame.index.max())
    valid_mask = (
        np.isfinite(forecast_values)
        & np.isfinite(lead_hours)
        & ~valid_times.isna()
        & (valid_times <= observed_end)
    )
    if model_versions is not None:
        valid_mask &= np.isfinite(model_versions) & (model_versions == float(load_model_version))
    if not np.any(valid_mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    forecast_values = forecast_values[valid_mask]
    valid_times = valid_times[valid_mask]
    lead_hours = lead_hours[valid_mask]
    observed_values = observed.reindex(valid_times, method="nearest", tolerance=tolerance).to_numpy(dtype=np.float64)
    paired = np.isfinite(observed_values)
    if not np.any(paired):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return forecast_values[paired] - observed_values[paired], lead_hours[paired]


def _archive_verification_frame(
    archive: xr.Dataset,
    observed: pd.Series,
    *,
    forecast_var: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    if forecast_var not in archive or "ForecastValidTime" not in archive or "ForecastLeadHours" not in archive:
        return pd.DataFrame()
    forecast_values = np.asarray(archive[forecast_var].values, dtype=np.float64).reshape(-1)
    valid_times = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    lead_hours = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    issue_grid = np.repeat(
        pd.DatetimeIndex(archive["issue_time"].values).to_numpy(dtype="datetime64[ns]"),
        int(archive.sizes.get("forecast_step", 0)),
    )
    issue_times = pd.DatetimeIndex(issue_grid)
    if "LoadModelVersion" in archive:
        model_versions = np.repeat(
            np.asarray(archive["LoadModelVersion"].values, dtype=np.float64).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    else:
        model_versions = np.full(len(forecast_values), np.nan, dtype=np.float64)
    if "ECMWFCycleTime" in archive:
        cycle_grid = np.repeat(
            pd.DatetimeIndex(archive["ECMWFCycleTime"].values).to_numpy(dtype="datetime64[ns]"),
            int(archive.sizes.get("forecast_step", 0)),
        )
        cycle_times = pd.DatetimeIndex(cycle_grid)
    else:
        cycle_times = issue_times.floor("3h")
    observed_end = pd.Timestamp(observed.index.max()) if not observed.empty else pd.NaT
    valid_mask = (
        np.isfinite(forecast_values)
        & np.isfinite(lead_hours)
        & ~valid_times.isna()
        & ~issue_times.isna()
        & (valid_times <= observed_end)
    )
    if not np.any(valid_mask):
        return pd.DataFrame()
    forecast_values = forecast_values[valid_mask]
    valid_times = valid_times[valid_mask]
    issue_times = issue_times[valid_mask]
    cycle_times = cycle_times[valid_mask]
    model_versions = model_versions[valid_mask]
    lead_hours = lead_hours[valid_mask]
    observed_valid = observed.reindex(valid_times, method="nearest", tolerance=tolerance).to_numpy(dtype=np.float64)
    reference_values = observed.reindex(issue_times, method="nearest", tolerance=tolerance).to_numpy(dtype=np.float64)
    paired = np.isfinite(observed_valid) & np.isfinite(reference_values)
    if not np.any(paired):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "issue_time": issue_times[paired],
            "cycle_time": cycle_times[paired],
            "load_model_version": model_versions[paired],
            "valid_time": valid_times[paired],
            "lead_hour": lead_hours[paired],
            "forecast_value": forecast_values[paired],
            "observed_value": observed_valid[paired],
            "reference_value": reference_values[paired],
            "error": forecast_values[paired] - observed_valid[paired],
            "reference_error": reference_values[paired] - observed_valid[paired],
        }
    ).sort_values("valid_time")


def _independent_verification_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Keep one forecast per ECMWF cycle and valid time.

    Cached learning runs re-anchor the same ECMWF cycle every 15 minutes. They
    are operationally useful but are not independent evidence for skill.
    """
    if table.empty:
        return table
    keys = [name for name in ("cycle_time", "valid_time") if name in table]
    if not keys:
        return table
    return table.sort_values("issue_time").drop_duplicates(keys, keep="last")


def _add_error_metrics(metrics: dict[str, float | int | str], prefix: str, errors: np.ndarray, lead_hours: np.ndarray) -> None:
    valid = np.isfinite(errors) & np.isfinite(lead_hours)
    if np.count_nonzero(valid) < 2:
        return
    errors = errors[valid]
    lead_hours = lead_hours[valid]
    metrics[f"{prefix}_mae"] = float(np.mean(np.abs(errors)))
    metrics[f"{prefix}_bias"] = float(np.mean(errors))
    metrics[f"{prefix}_sample_count"] = int(errors.size)
    for bucket, start, end in LEAD_BUCKETS:
        in_bucket = (lead_hours >= start) & (lead_hours < end)
        if np.count_nonzero(in_bucket) >= 2:
            bucket_errors = errors[in_bucket]
            metrics[_metric_bucket_name(f"{prefix}_mae", bucket)] = float(np.mean(np.abs(bucket_errors)))
            metrics[_metric_bucket_name(f"{prefix}_bias", bucket)] = float(np.mean(bucket_errors))
            metrics[_metric_bucket_name(f"{prefix}_sample_count", bucket)] = int(bucket_errors.size)


def evaluate_forecast_archive(archive: xr.Dataset | None, frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Score archived forecast runs against APS observations by lead time."""
    if archive is None or frame.empty or "issue_time" not in archive.sizes or archive.sizes.get("issue_time", 0) == 0:
        return {}
    metrics: dict[str, float | int | str] = {}
    tolerance = pd.Timedelta(minutes=10)
    if "BatterySOC" in frame:
        errors, lead_hours = _errors_for_archive_variable(
            archive,
            frame,
            forecast_var="BatterySOCForecast",
            observed=frame["BatterySOC"],
            tolerance=tolerance,
        )
        _add_error_metrics(metrics, "soc", errors, lead_hours)
    observed_solar = _observed_solar_w(frame)
    if not observed_solar.empty:
        errors, lead_hours = _errors_for_archive_variable(
            archive,
            frame,
            forecast_var="ForecastSolarWatts",
            observed=observed_solar,
            tolerance=tolerance,
        )
        _add_error_metrics(metrics, "solar", errors, lead_hours)
    observed_load = _observed_load_w(frame)
    if not observed_load.empty:
        errors, lead_hours = _errors_for_archive_variable(
            archive,
            frame,
            forecast_var="ForecastLoadWatts",
            observed=observed_load,
            tolerance=tolerance,
            load_model_version=LOAD_MODEL_VERSION,
        )
        _add_error_metrics(metrics, "load", errors, lead_hours)
    if metrics:
        aliases = {
            "soc_mae_pct_points": "soc_mae",
            "soc_bias_pct_points": "soc_bias",
            "soc_sample_count": "soc_sample_count",
            "solar_mae_w": "solar_mae",
            "solar_bias_w": "solar_bias",
            "solar_sample_count": "solar_sample_count",
            "load_mae_w": "load_mae",
            "load_bias_w": "load_bias",
            "load_sample_count": "load_sample_count",
        }
        for alias, source in aliases.items():
            if source in metrics:
                metrics[alias] = metrics[source]
        metrics["evaluated_at_utc"] = _utc_now()
    return metrics


def _rolling_error_stats(errors: np.ndarray, reference_errors: np.ndarray) -> tuple[float, float, float, float, float, float, int]:
    valid = np.isfinite(errors)
    if np.count_nonzero(valid) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, int(np.count_nonzero(valid)))
    errors = errors[valid]
    ref = reference_errors[valid & np.isfinite(reference_errors)] if reference_errors.shape == valid.shape else np.array([])
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    ref_mae = float(np.mean(np.abs(ref))) if ref.size >= 2 else np.nan
    skill = float(1.0 - (mae / ref_mae)) if np.isfinite(ref_mae) and ref_mae > 0.0 else np.nan
    return (mae, bias, rmse, ref_mae, skill, float(errors.size), int(errors.size))


def _guarded_skill(mae: float, reference_mae: float, *, minimum_reference_mae: float) -> float:
    if not np.isfinite(mae) or not np.isfinite(reference_mae) or reference_mae < minimum_reference_mae:
        return np.nan
    return float(1.0 - mae / reference_mae)


def _empty_skill_dataset() -> xr.Dataset:
    empty_time = np.array([], dtype="datetime64[ns]")
    fields = (
        "ForecastVerificationSamples",
        "ForecastIndependentCycles",
        "ForecastSOCMAE_0_6h_Verified",
        "ForecastSOCMAE_6_24h_Verified",
        "ForecastSOCMAE_24_48h_Verified",
        "ForecastSOCMAE_48_96h_Verified",
        "ForecastSOCBias_0_6h_Verified",
        "ForecastSOCSkill_0_6h",
        "ForecastLoadMAE24h",
        "ForecastLoadBias24h",
        "ForecastLoadSkill24h",
        "ForecastSolarMAE24h",
        "ForecastSolarBias24h",
        "ForecastSolarSkill24h",
    )
    return xr.Dataset(
        {name: (("time",), np.array([], dtype=np.float32)) for name in fields},
        coords={"time": empty_time},
        attrs={
            "power_soc_forecast_skill_product": "true",
            "source": "verification of archived APS SOC forecasts against observed APS power history",
            "generated_at_utc": _utc_now(),
        }
    )


def build_forecast_skill_dataset(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    window_hours: float = DEFAULT_SKILL_WINDOW_HOURS,
    retention_days: float = DEFAULT_SKILL_RETENTION_DAYS,
    freq: str = "1h",
) -> xr.Dataset:
    """Build a past-facing verification product from archived forecasts and observations."""
    frame = _power_frame(power)
    if archive is None or frame.empty or "issue_time" not in archive.sizes or archive.sizes.get("issue_time", 0) == 0:
        return _empty_skill_dataset()

    tolerance = pd.Timedelta(minutes=10)
    pieces: dict[str, pd.DataFrame] = {}
    if "BatterySOC" in frame:
        pieces["soc"] = _archive_verification_frame(
            archive,
            frame["BatterySOC"],
            forecast_var="BatterySOCForecast",
            tolerance=tolerance,
        )
    observed_solar = _observed_solar_w(frame)
    if not observed_solar.empty:
        pieces["solar"] = _archive_verification_frame(
            archive,
            observed_solar,
            forecast_var="ForecastSolarWatts",
            tolerance=tolerance,
        )
    observed_load = _observed_load_w(frame)
    if not observed_load.empty:
        load_table = _archive_verification_frame(
            archive,
            observed_load,
            forecast_var="ForecastLoadWatts",
            tolerance=tolerance,
        )
        if not load_table.empty:
            load_table = load_table[load_table["load_model_version"] == float(LOAD_MODEL_VERSION)]
        pieces["load"] = load_table
    pieces = {name: table for name, table in pieces.items() if not table.empty}
    if not pieces:
        return _empty_skill_dataset()

    observed_end = pd.Timestamp(frame.index.max())
    start = observed_end - pd.Timedelta(days=float(retention_days))
    time_index = pd.date_range(start.floor(freq), observed_end.ceil(freq), freq=freq)
    if len(time_index) == 0:
        return _empty_skill_dataset()

    columns: dict[str, np.ndarray] = {}
    window = pd.Timedelta(hours=float(window_hours))
    for metric_name in (
        "ForecastVerificationSamples",
        "ForecastIndependentCycles",
        "ForecastSOCMAE_0_6h_Verified",
        "ForecastSOCMAE_6_24h_Verified",
        "ForecastSOCMAE_24_48h_Verified",
        "ForecastSOCMAE_48_96h_Verified",
        "ForecastSOCBias_0_6h_Verified",
        "ForecastSOCSkill_0_6h",
        "ForecastLoadMAE24h",
        "ForecastLoadBias24h",
        "ForecastLoadSkill24h",
        "ForecastSolarMAE24h",
        "ForecastSolarBias24h",
        "ForecastSolarSkill24h",
    ):
        columns[metric_name] = np.full(len(time_index), np.nan, dtype=np.float32)

    for idx, now in enumerate(time_index):
        window_start = now - window
        soc = pieces.get("soc")
        if soc is not None:
            selected = _independent_verification_rows(
                soc[(soc["valid_time"] > window_start) & (soc["valid_time"] <= now)]
            )
            columns["ForecastVerificationSamples"][idx] = float(len(selected))
            columns["ForecastIndependentCycles"][idx] = float(selected["cycle_time"].nunique())
            for bucket, start_hour, end_hour in LEAD_BUCKETS:
                bucketed = selected[(selected["lead_hour"] >= start_hour) & (selected["lead_hour"] < end_hour)]
                mae, bias, _rmse, ref_mae, _skill, _sample_float, sample_count = _rolling_error_stats(
                    bucketed["error"].to_numpy(dtype=np.float64),
                    bucketed["reference_error"].to_numpy(dtype=np.float64),
                )
                if sample_count >= 2:
                    columns[f"ForecastSOCMAE_{bucket}_Verified"][idx] = mae
                    if bucket == "0_6h":
                        columns["ForecastSOCBias_0_6h_Verified"][idx] = bias
                        columns["ForecastSOCSkill_0_6h"][idx] = _guarded_skill(
                            mae, ref_mae, minimum_reference_mae=0.5
                        )
        load = pieces.get("load")
        if load is not None:
            selected = _independent_verification_rows(
                load[(load["valid_time"] > window_start) & (load["valid_time"] <= now)]
            )
            mae, bias, _rmse, ref_mae, _skill, _sample_float, sample_count = _rolling_error_stats(
                selected["error"].to_numpy(dtype=np.float64),
                selected["reference_error"].to_numpy(dtype=np.float64),
            )
            if sample_count >= 2:
                columns["ForecastLoadMAE24h"][idx] = mae
                columns["ForecastLoadBias24h"][idx] = bias
                columns["ForecastLoadSkill24h"][idx] = _guarded_skill(
                    mae, ref_mae, minimum_reference_mae=5.0
                )
        solar = pieces.get("solar")
        if solar is not None:
            selected = _independent_verification_rows(
                solar[(solar["valid_time"] > window_start) & (solar["valid_time"] <= now)]
            )
            mae, bias, _rmse, ref_mae, _skill, _sample_float, sample_count = _rolling_error_stats(
                selected["error"].to_numpy(dtype=np.float64),
                selected["reference_error"].to_numpy(dtype=np.float64),
            )
            if sample_count >= 2:
                columns["ForecastSolarMAE24h"][idx] = mae
                columns["ForecastSolarBias24h"][idx] = bias
                columns["ForecastSolarSkill24h"][idx] = _guarded_skill(
                    mae, ref_mae, minimum_reference_mae=5.0
                )

    out = xr.Dataset(
        {name: (("time",), values) for name, values in columns.items()},
        coords={"time": time_index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_forecast_skill_product": "true",
            "source": "verification of archived APS SOC forecasts against observed APS power history",
            "generated_at_utc": _utc_now(),
            "verification_window_hours": str(float(window_hours)),
            "retention_days": str(float(retention_days)),
            "reference_model": "persistence from observed value at forecast issue time",
            "skill_score": "1 - forecast_mae / persistence_mae",
            "description": "Past-facing forecast verification history; future forecast curves remain in power_soc_forecast.zarr.",
            "sample_independence": "one forecast per ECMWF cycle and valid time",
            "load_model": LOAD_MODEL_NAME,
            "load_model_version": str(LOAD_MODEL_VERSION),
        },
    )
    for name in out.data_vars:
        if name.endswith("Samples"):
            out[name].attrs["units"] = "samples"
        elif "Skill" in name:
            out[name].attrs["units"] = "1"
        elif "SOC" in name:
            out[name].attrs["units"] = "percentage points"
        else:
            out[name].attrs["units"] = "W"
    return out


def _archive_row_from_forecast(forecast: xr.Dataset) -> xr.Dataset:
    issue_time = pd.Timestamp(forecast.attrs.get("initial_soc_time", forecast.attrs.get("generated_at_utc", _utc_now())))
    if issue_time.tz is not None:
        issue_time = issue_time.tz_convert("UTC").tz_localize(None)
    times = pd.DatetimeIndex(forecast["time"].values)
    lead_hours = ((times - issue_time) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32)
    step = np.arange(len(times), dtype=np.int32)
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {
        "ForecastValidTime": (("issue_time", "forecast_step"), times.to_numpy(dtype="datetime64[ns]")[None, :]),
        "ForecastLeadHours": (("issue_time", "forecast_step"), lead_hours[None, :]),
    }
    cycle_time = pd.Timestamp(forecast.attrs.get("ecmwf_cycle_time", issue_time))
    if cycle_time.tz is not None:
        cycle_time = cycle_time.tz_convert("UTC").tz_localize(None)
    data_vars["ECMWFCycleTime"] = (("issue_time",), np.array([cycle_time.to_datetime64()], dtype="datetime64[ns]"))
    data_vars["LoadModelVersion"] = (
        ("issue_time",),
        np.array([float(forecast.attrs.get("load_model_version", LOAD_MODEL_VERSION))], dtype=np.float32),
    )
    for name in ARCHIVE_FORECAST_FIELDS:
        if name in forecast:
            data_vars[name] = (("issue_time", "forecast_step"), np.asarray(forecast[name].values, dtype=np.float32)[None, :])
    return xr.Dataset(
        data_vars,
        coords={"issue_time": np.array([issue_time.to_datetime64()], dtype="datetime64[ns]"), "forecast_step": step},
        attrs={
            "power_soc_forecast_archive": "true",
            "source": "archived rows from latest ECMWF-informed APS SOC forecasts",
        },
    )


def build_soc_hindcast_dataset(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    retention_days: float = DEFAULT_HINDCAST_RETENTION_DAYS,
    lead_hours: tuple[int, ...] = HINDCAST_LEAD_HOURS,
    lead_tolerance_hours: float = 1.5,
) -> xr.Dataset:
    """Build observed SOC plus fixed-lead archived forecasts for the dashboard."""
    frame = _power_frame(power)
    if archive is None or frame.empty or "BatterySOC" not in frame:
        return xr.Dataset(coords={"time": np.array([], dtype="datetime64[ns]")})
    observed_end = pd.Timestamp(frame.index.max())
    observed_start = observed_end - pd.Timedelta(days=float(retention_days))
    observed = frame.loc[frame.index >= observed_start, "BatterySOC"].resample("15min").last().dropna()
    records = _archive_verification_frame(
        archive,
        frame["BatterySOC"],
        forecast_var="BatterySOCForecast",
        tolerance=pd.Timedelta(minutes=10),
    )
    series: list[pd.Series] = [observed.rename("BatterySOCObservedHindcast")]
    for target in lead_hours:
        selected = records[
            (records["valid_time"] >= observed_start)
            & ((records["lead_hour"] - float(target)).abs() <= float(lead_tolerance_hours))
        ].copy()
        if selected.empty:
            continue
        selected["lead_delta"] = (selected["lead_hour"] - float(target)).abs()
        selected = selected.sort_values(["valid_time", "lead_delta", "issue_time"])
        selected = selected.drop_duplicates("valid_time", keep="first")
        values = pd.Series(
            selected["forecast_value"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(selected["valid_time"]),
            name=f"BatterySOCHindcast_{int(target)}h",
        )
        series.append(values)
    merged = pd.concat(series, axis=1).sort_index().dropna(how="all")
    out = xr.Dataset(
        {name: (("time",), merged[name].to_numpy(dtype=np.float32)) for name in merged.columns},
        coords={"time": merged.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_hindcast_product": "true",
            "generated_at_utc": _utc_now(),
            "retention_days": str(float(retention_days)),
            "lead_tolerance_hours": str(float(lead_tolerance_hours)),
            "source": "archived operational SOC forecasts matched to later APS observations",
        },
    )
    for name in out.data_vars:
        out[name].attrs["units"] = "%"
    return out


def append_forecast_archive(
    forecast: xr.Dataset,
    archive_zarr: Path,
    *,
    retention_days: float = DEFAULT_ARCHIVE_RETENTION_DAYS,
) -> xr.Dataset:
    """Append a latest forecast run to the forecast-run archive."""
    row = _archive_row_from_forecast(forecast)
    archive = None
    if archive_zarr.exists():
        try:
            archive = xr.open_zarr(archive_zarr, chunks={}).load()
        except Exception:
            archive = None
    if archive is not None and archive.sizes.get("issue_time", 0):
        max_steps = max(int(archive.sizes.get("forecast_step", 0)), int(row.sizes.get("forecast_step", 0)))
        steps = np.arange(max_steps, dtype=np.int32)
        archive = archive.reindex(forecast_step=steps)
        row = row.reindex(forecast_step=steps)
        combined = xr.concat([archive, row], dim="issue_time")
        combined = combined.sortby("issue_time")
        combined = combined.isel(issue_time=~combined.indexes["issue_time"].duplicated(keep="last"))
    else:
        combined = row
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=float(retention_days))
    issue_times = pd.DatetimeIndex(combined["issue_time"].values)
    combined = combined.isel(issue_time=np.asarray(issue_times >= cutoff))
    _atomic_write_archive(combined, archive_zarr)
    return combined


def _adaptive_value(raw_value: float, state_value: object, *, alpha: float = DEFAULT_ADAPTIVE_ALPHA) -> float:
    try:
        previous = float(state_value)
    except Exception:
        return float(raw_value)
    if not np.isfinite(previous):
        return float(raw_value)
    return float((1.0 - alpha) * previous + alpha * raw_value)


def _load_bias_correction(previous_correction: object, previous_bias: object, *, alpha: float = DEFAULT_ADAPTIVE_ALPHA) -> float:
    try:
        correction = float(previous_correction)
    except Exception:
        correction = 0.0
    if not np.isfinite(correction):
        correction = 0.0
    try:
        bias = float(previous_bias)
    except Exception:
        return float(correction)
    if not np.isfinite(bias):
        return float(correction)
    # Positive bias means the previous forecast load was too high, so reduce the
    # next load profile; negative bias means it was too low.
    updated = correction - float(alpha) * bias
    return float(np.clip(updated, -DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W, DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W))


def _bounded_load_profile(raw_load_profile: pd.Series, correction_w: float) -> tuple[pd.Series, float]:
    """Apply adaptive load bias without allowing it to erase a load profile."""
    if raw_load_profile.empty:
        return raw_load_profile, float(correction_w)
    raw_values = raw_load_profile.to_numpy(dtype=np.float64)
    finite_raw = raw_values[np.isfinite(raw_values)]
    if finite_raw.size == 0:
        return raw_load_profile.clip(lower=0.0), float(correction_w)
    fraction_limit = float(np.clip(DEFAULT_LOAD_NEGATIVE_CORRECTION_FRACTION_LIMIT, 0.0, 0.95))
    raw_median = float(np.nanmedian(finite_raw))
    bounded_correction = float(correction_w)
    if bounded_correction < 0.0 and raw_median > 0.0:
        bounded_correction = max(bounded_correction, -fraction_limit * raw_median)
    adjusted = (raw_load_profile + bounded_correction).clip(lower=0.0)
    if fraction_limit > 0.0:
        floor = raw_load_profile.clip(lower=0.0) * (1.0 - fraction_limit)
        adjusted = adjusted.where(adjusted >= floor, floor)
    return adjusted, bounded_correction


def _soc_bias_corrections(
    previous_corrections: object,
    metrics: dict[str, float | int | str],
    *,
    alpha: float,
) -> dict[str, float]:
    corrections: dict[str, float] = {}
    if isinstance(previous_corrections, dict):
        for key, value in previous_corrections.items():
            try:
                parsed = float(value)
            except Exception:
                continue
            if np.isfinite(parsed):
                corrections[str(key)] = parsed
    for bucket, _, _ in LEAD_BUCKETS:
        bias_key = _metric_bucket_name("soc_bias", bucket)
        try:
            bias = float(metrics.get(bias_key, np.nan))
        except Exception:
            continue
        if not np.isfinite(bias):
            continue
        previous = float(corrections.get(bucket, 0.0))
        # Positive SOC bias means the forecast was too high; subtract it from
        # the next forecast for that lead bucket.
        updated = previous - float(alpha) * bias
        corrections[bucket] = float(np.clip(updated, -DEFAULT_SOC_BIAS_CORRECTION_LIMIT, DEFAULT_SOC_BIAS_CORRECTION_LIMIT))
    return corrections


def _apply_soc_bias_corrections(
    forecast: pd.DataFrame,
    corrections: dict[str, float],
    *,
    issue_time: pd.Timestamp,
) -> pd.DataFrame:
    if not corrections or "BatterySOCForecast" not in forecast:
        return forecast
    out = forecast.copy()
    lead_hours = (pd.DatetimeIndex(out.index) - pd.Timestamp(issue_time)) / pd.Timedelta(hours=1)
    initial_soc = float(out["BatterySOCForecast"].iloc[0])
    soc = out["BatterySOCForecast"].to_numpy(dtype=np.float64).copy()
    for bucket, start, end in LEAD_BUCKETS:
        correction = float(corrections.get(bucket, 0.0))
        if correction == 0.0:
            continue
        mask = (lead_hours >= start) & (lead_hours < end)
        soc[mask] = np.clip(soc[mask] + correction, 0.0, 100.0)
    soc[0] = initial_soc
    out["BatterySOCForecast"] = soc
    return out


def calibrate_solar_factor(
    frame: pd.DataFrame,
    irradiance: pd.Series,
    *,
    end: pd.Timestamp,
    calibration_days: float = DEFAULT_CALIBRATION_DAYS,
    fallback_hours: float = DEFAULT_FALLBACK_CALIBRATION_HOURS,
) -> float:
    """Estimate APS solar watts per ECMWF W/m2 from recent observations."""
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if not solar_fields or irradiance.empty:
        return float(DEFAULT_SOLAR_CALIBRATION_FACTOR)

    for start in (end - pd.Timedelta(days=float(calibration_days)), end - pd.Timedelta(hours=float(fallback_hours))):
        observed = frame.loc[frame.index >= start, solar_fields].sum(axis=1, min_count=1).clip(lower=0.0)
        if observed.empty:
            continue
        model = irradiance.reindex(observed.index, method="nearest", tolerance=pd.Timedelta(hours=2))
        valid = np.isfinite(observed.to_numpy(dtype=np.float64)) & np.isfinite(model.to_numpy(dtype=np.float64)) & (model.to_numpy(dtype=np.float64) > 20.0)
        if np.count_nonzero(valid) >= 6:
            ratios = observed.to_numpy(dtype=np.float64)[valid] / model.to_numpy(dtype=np.float64)[valid]
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                return float(np.clip(np.nanmedian(ratios), 0.0, 20.0))
        observed_finite = observed.to_numpy(dtype=np.float64)
        observed_finite = observed_finite[np.isfinite(observed_finite) & (observed_finite > 0.0)]
        model_finite = irradiance.to_numpy(dtype=np.float64)
        model_finite = model_finite[np.isfinite(model_finite) & (model_finite > 20.0)]
        if observed_finite.size >= 6 and model_finite.size >= 2:
            observed_scale = np.nanpercentile(observed_finite, 95)
            model_scale = np.nanpercentile(model_finite, 95)
            if np.isfinite(observed_scale) and np.isfinite(model_scale) and model_scale > 0.0:
                return float(np.clip(observed_scale / model_scale, 0.0, 20.0))
    return float(DEFAULT_SOLAR_CALIBRATION_FACTOR)


def integrate_soc_forecast(
    *,
    initial_soc: float,
    initial_time: pd.Timestamp | None = None,
    irradiance: pd.Series,
    solar_factor: float,
    load_w: float | pd.Series,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
) -> pd.DataFrame:
    """Integrate SOC forward from ECMWF solar and expected load."""
    if irradiance.empty:
        return pd.DataFrame()
    forecast_times = pd.DatetimeIndex(irradiance.index)
    forecast_irradiance = irradiance.to_numpy(dtype=np.float64)
    forecast_solar_w = np.clip(forecast_irradiance * float(solar_factor), 0.0, None)
    if initial_time is not None:
        initial_time = pd.Timestamp(initial_time)
        if initial_time.tz is not None:
            initial_time = initial_time.tz_convert("UTC").tz_localize(None)
        if initial_time < forecast_times[0]:
            times = pd.DatetimeIndex([initial_time]).append(forecast_times)
            irradiance_values = np.concatenate(([np.nan], forecast_irradiance))
            solar_w = np.concatenate(([np.nan], forecast_solar_w))
        else:
            times = forecast_times
            irradiance_values = forecast_irradiance
            solar_w = forecast_solar_w
    else:
        times = forecast_times
        irradiance_values = forecast_irradiance
        solar_w = forecast_solar_w
    if isinstance(load_w, pd.Series):
        load_series = load_w.reindex(times, method="nearest", tolerance=pd.Timedelta(hours=2))
        if load_series.isna().all():
            load_values = np.full(len(times), DEFAULT_LOAD_W, dtype=np.float64)
        else:
            load_values = load_series.ffill().bfill().to_numpy(dtype=np.float64)
        load = np.clip(load_values, 0.0, None)
    else:
        load = np.full(len(times), max(float(load_w), 0.0), dtype=np.float64)
    soc = np.full(len(times), np.nan, dtype=np.float64)
    soc[0] = float(np.clip(initial_soc, 0.0, 100.0))
    for idx in range(1, len(times)):
        dt_hours = max((times[idx] - times[idx - 1]) / pd.Timedelta(hours=1), 0.0)
        interval_solar_w = solar_w[idx]
        if not np.isfinite(interval_solar_w):
            interval_solar_w = 0.0
        net_kwh = (interval_solar_w - load[idx]) * dt_hours / 1000.0
        soc[idx] = np.clip(soc[idx - 1] + 100.0 * net_kwh / float(capacity_kwh), 0.0, 100.0)
    return pd.DataFrame(
        {
            "BatterySOCForecast": soc,
            "ECMWFSolarIrradiance": irradiance_values,
            "ForecastSolarWatts": solar_w,
            "ForecastLoadWatts": load,
        },
        index=times,
    )


def build_forecast_dataset(
    power: xr.Dataset,
    solar: xr.Dataset,
    *,
    pdu: xr.Dataset | None = None,
    previous_forecast: xr.Dataset | None = None,
    forecast_archive: xr.Dataset | None = None,
    state: dict[str, object] | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    calibration_days: float = DEFAULT_CALIBRATION_DAYS,
    fallback_calibration_hours: float = DEFAULT_FALLBACK_CALIBRATION_HOURS,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
) -> xr.Dataset:
    frame = _power_frame(power)
    if frame.empty or "BatterySOC" not in frame:
        raise ValueError("Power dataset needs BatterySOC to initialize the SOC forecast")
    latest_time, latest_soc = latest_finite(frame["BatterySOC"])
    state = dict(state or {})
    previous_metrics = evaluate_forecast_archive(forecast_archive, frame)
    if not previous_metrics:
        previous_metrics = evaluate_previous_forecast(previous_forecast, frame)
    irradiance = solar_irradiance_from_ssrd(solar)
    if irradiance.empty:
        raise ValueError("No ECMWF solar forecast samples could be converted from ssrd")
    horizon_end = latest_time + pd.Timedelta(hours=horizon_hours)
    irradiance = irradiance[(irradiance.index >= latest_time) & (irradiance.index <= horizon_end)]
    if not irradiance.empty and irradiance.index[-1] < horizon_end:
        irradiance = pd.concat([irradiance, pd.Series([float(irradiance.iloc[-1])], index=pd.DatetimeIndex([horizon_end]))])
        irradiance = irradiance[~irradiance.index.duplicated(keep="last")].sort_index()
    if len(irradiance) < 2:
        raise ValueError("ECMWF solar forecast does not overlap the requested SOC forecast horizon")

    factor_raw = calibrate_solar_factor(
        frame,
        irradiance,
        end=latest_time,
        calibration_days=calibration_days,
        fallback_hours=fallback_calibration_hours,
    )
    adaptive_alpha = _balanced_alpha(
        max(
            int(previous_metrics.get("soc_sample_count", 0) or 0),
            int(previous_metrics.get("solar_sample_count", 0) or 0),
            int(previous_metrics.get("load_sample_count", 0) or 0),
        )
    )
    factor = _adaptive_value(factor_raw, state.get("solar_calibration_factor_w_per_wm2"), alpha=adaptive_alpha)
    raw_load_profile = build_historical_load_forecast(
        frame,
        pd.DatetimeIndex(irradiance.index),
        end=latest_time,
        calibration_days=calibration_days,
        default_load_w=DEFAULT_LOAD_W,
    )
    load_diagnostics = dict(raw_load_profile.attrs)
    raw_load_w = float(raw_load_profile.median()) if not raw_load_profile.empty else float(DEFAULT_LOAD_W)
    load_mode_registry, clean_dc_only_level_w = _repair_dc_only_registry(
        state.get("load_mode_registry"),
        frame,
        end=latest_time,
    )
    load_mode, mode_source, active_kits, pdu_time, pdu_active_watts = _resolve_load_mode(
        frame,
        pdu,
        end=latest_time,
        observed_level_w=raw_load_w,
        raw_registry=load_mode_registry,
        previous_mode=state.get("current_load_mode"),
    )
    mode_signature = _load_mode_signature(load_mode, mode_source, active_kits)
    learning_ready, learning_reason = _mode_learning_status(load_diagnostics, load_mode)
    if learning_ready:
        load_mode_registry, load_w = _update_load_mode_registry(
            load_mode_registry,
            mode=load_mode,
            observed_level_w=raw_load_w,
            observed_at=latest_time,
            active_kits=active_kits,
            mode_source=mode_source,
            signature=mode_signature,
        )
    else:
        mode_entry = load_mode_registry.get(load_mode, {})
        try:
            learned_level_w = float(mode_entry.get("learned_level_w", np.nan))
        except Exception:
            learned_level_w = np.nan
        if np.isfinite(learned_level_w):
            load_w = learned_level_w
        elif mode_source == "pdu_signature" and np.isfinite(pdu_active_watts) and clean_dc_only_level_w is not None:
            load_w = float(clean_dc_only_level_w) + float(pdu_active_watts)
        else:
            load_w = raw_load_w
    mode_entry = load_mode_registry.get(load_mode, {})
    learning_observations = int(mode_entry.get("observation_count", 0) or 0)
    load_diagnostics.update(
        {
            "load_mode": load_mode,
            "load_mode_source": mode_source,
            "load_mode_active_kits": active_kits,
            "load_mode_pdu_time": pdu_time,
            "load_mode_pdu_active_watts": pdu_active_watts,
            "load_mode_signature": mode_signature,
            "load_mode_learning_ready": learning_ready,
            "load_mode_learning_reason": learning_reason,
            "load_mode_learning_observations": learning_observations,
            "load_regime": load_mode,
            "load_regime_level_w": load_w,
        }
    )
    # Bias learned by retired load models is not transferable across operating
    # modes. Each named mode learns its own robust power-balance level.
    load_bias_correction = 0.0
    load_profile = pd.Series(np.full(len(raw_load_profile), load_w), index=raw_load_profile.index)
    forecast = integrate_soc_forecast(
        initial_soc=latest_soc,
        initial_time=latest_time,
        irradiance=irradiance,
        solar_factor=factor,
        load_w=load_profile,
        capacity_kwh=capacity_kwh,
    )
    for scenario_load_w in SCENARIO_LOADS_W:
        scenario = integrate_soc_forecast(
            initial_soc=latest_soc,
            initial_time=latest_time,
            irradiance=irradiance,
            solar_factor=factor,
            load_w=float(scenario_load_w),
            capacity_kwh=capacity_kwh,
        )
        forecast[scenario_soc_field(scenario_load_w)] = scenario["BatterySOCForecast"]
    soc_bias_corrections = _soc_bias_corrections(
        state.get("soc_bias_correction_pct_points_by_bucket"),
        previous_metrics,
        alpha=adaptive_alpha,
    )
    forecast = _apply_soc_bias_corrections(forecast, soc_bias_corrections, issue_time=latest_time)
    soc_mae = float(previous_metrics.get("soc_mae_pct_points", np.nan))
    solar_mae = float(previous_metrics.get("solar_mae_w", np.nan))
    load_mae = float(previous_metrics.get("load_mae_w", np.nan))
    load_bias = float(previous_metrics.get("load_bias_w", np.nan))
    evaluation_samples = max(
        int(previous_metrics.get("soc_sample_count", 0) or 0),
        int(previous_metrics.get("solar_sample_count", 0) or 0),
        int(previous_metrics.get("load_sample_count", 0) or 0),
    )
    forecast["ForecastSOCMAERecent"] = soc_mae
    for bucket, _, _ in LEAD_BUCKETS:
        forecast[f"ForecastSOCMAE_{bucket}"] = float(previous_metrics.get(_metric_bucket_name("soc_mae", bucket), np.nan))
    forecast["ForecastSolarMAERecent"] = solar_mae
    forecast["ForecastLoadMAERecent"] = load_mae
    forecast["ForecastLoadBiasRecent"] = load_bias
    forecast["ForecastEvaluationSamples"] = float(evaluation_samples)
    forecast["ForecastSkillSampleCount"] = float(evaluation_samples)
    out = xr.Dataset(
        {name: (("time",), forecast[name].to_numpy(dtype=np.float32)) for name in forecast.columns},
        coords={"time": forecast.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_forecast_product": "true",
            "source": "derived from ECMWF ssrd forecast and APS power history",
            "ecmwf_param": ECMWF_PARAM,
            "generated_at_utc": _utc_now(),
            "initial_soc_pct": f"{latest_soc:.6g}",
            "initial_soc_time": latest_time.isoformat(),
            "forecast_horizon_hours": str(int(horizon_hours)),
            "calibration_days": str(float(calibration_days)),
            "solar_calibration_factor_w_per_wm2": f"{factor:.6g}",
            "raw_solar_calibration_factor_w_per_wm2": f"{factor_raw:.6g}",
            "forecast_load_w": f"{load_w:.6g}",
            "raw_forecast_load_w": f"{raw_load_w:.6g}",
            "load_bias_correction_w": f"{load_bias_correction:.6g}",
            "soc_bias_correction_pct_points_by_bucket": json.dumps(soc_bias_corrections, sort_keys=True),
            "load_model": LOAD_MODEL_NAME,
            "load_model_version": str(LOAD_MODEL_VERSION),
            "load_mode": load_mode,
            "load_mode_source": mode_source,
            "load_mode_active_kits": ",".join(active_kits),
            "load_mode_pdu_time": pdu_time.isoformat() if pdu_time is not None else "",
            "load_mode_pdu_active_watts": f"{float(pdu_active_watts):.6g}",
            "load_mode_signature": mode_signature,
            "load_mode_learning_ready": str(bool(learning_ready)).lower(),
            "load_mode_learning_reason": learning_reason,
            "load_mode_learning_observations": str(learning_observations),
            "load_measurement": str(load_diagnostics.get("load_measurement", "unknown")),
            "load_balance_measurement": str(load_diagnostics.get("load_balance_measurement", "unknown")),
            "load_mode_registry": json.dumps(load_mode_registry, sort_keys=True),
            "load_regime": str(load_diagnostics.get("load_regime", "unknown")),
            "load_regime_threshold_w": f"{float(load_diagnostics.get('load_regime_threshold_w', np.nan)):.6g}",
            "load_regime_level_w": f"{float(load_diagnostics.get('load_regime_level_w', load_w)):.6g}",
            "load_regime_run_hours": f"{float(load_diagnostics.get('load_regime_run_hours', 0.0)):.6g}",
            "load_regime_sample_count": str(int(load_diagnostics.get("load_regime_sample_count", 0))),
            "battery_capacity_kwh": f"{float(capacity_kwh):.6g}",
            "adaptive_alpha": f"{adaptive_alpha:.6g}",
            "previous_forecast_metrics": json.dumps(previous_metrics, sort_keys=True),
            "scenario_loads_w": ",".join(str(load_w) for load_w in SCENARIO_LOADS_W),
            "scenario_solar_mode": "ecmwf",
            "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
        },
    )
    out["BatterySOCForecast"].attrs["units"] = "%"
    for scenario_load_w in SCENARIO_LOADS_W:
        field = scenario_soc_field(scenario_load_w)
        out[field].attrs["units"] = "%"
        out[field].attrs["scenario_load_w"] = str(int(scenario_load_w))
        out[field].attrs["scenario_solar_mode"] = "ecmwf"
    out["ECMWFSolarIrradiance"].attrs["units"] = "W m-2"
    out["ForecastSolarWatts"].attrs["units"] = "W"
    out["ForecastLoadWatts"].attrs["units"] = "W"
    out["ForecastSOCMAERecent"].attrs["units"] = "percentage points"
    for bucket, _, _ in LEAD_BUCKETS:
        out[f"ForecastSOCMAE_{bucket}"].attrs["units"] = "percentage points"
    out["ForecastSolarMAERecent"].attrs["units"] = "W"
    out["ForecastLoadMAERecent"].attrs["units"] = "W"
    out["ForecastLoadBiasRecent"].attrs["units"] = "W"
    out["ForecastEvaluationSamples"].attrs["units"] = "samples"
    out["ForecastSkillSampleCount"].attrs["units"] = "samples"
    return out


def generate(
    power_zarr: Path = POWER_ZARR_PATH,
    output_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    *,
    pdu_zarr: Path = POWER_PDU_ZARR_PATH,
    input_forecast: Path | None = None,
    cache_dir: Path = POWER_ECMWF_FORECAST_CACHE_DIR,
    state_path: Path = POWER_SOC_FORECAST_STATE_PATH,
    archive_zarr: Path = POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH,
    skill_zarr: Path | None = POWER_SOC_FORECAST_SKILL_ZARR_PATH,
    hindcast_zarr: Path | None = POWER_SOC_HINDCAST_ZARR_PATH,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    refresh_from_cache: bool = False,
) -> Path:
    if input_forecast is None:
        if refresh_from_cache:
            input_forecast = _latest_cached_forecast(cache_dir)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            input_forecast = cache_dir / f"ecmwf_ssrd_{stamp}.grib2"
            retrieve_open_data_grib(input_forecast, horizon_hours=horizon_hours)
    power = xr.open_zarr(power_zarr, chunks={})
    pdu = None
    if pdu_zarr.exists():
        try:
            pdu = xr.open_zarr(pdu_zarr, chunks={})
        except Exception:
            pdu = None
    solar = open_solar_forecast(input_forecast, latitude=latitude, longitude=longitude)
    previous_forecast = None
    if output_zarr.exists():
        try:
            previous_forecast = xr.open_zarr(output_zarr, chunks={})
        except Exception:
            previous_forecast = None
    forecast_archive = None
    if archive_zarr.exists():
        try:
            forecast_archive = xr.open_zarr(archive_zarr, chunks={}).load()
        except Exception:
            forecast_archive = None
    state = _load_state(state_path)
    forecast = build_forecast_dataset(
        power,
        solar,
        pdu=pdu,
        previous_forecast=previous_forecast,
        forecast_archive=forecast_archive,
        state=state,
        horizon_hours=horizon_hours,
    )
    forecast.attrs["ecmwf_input_file"] = str(input_forecast)
    forecast.attrs["site_latitude"] = str(float(latitude))
    forecast.attrs["site_longitude"] = str(float(longitude))
    forecast.attrs["refresh_from_cache"] = str(bool(refresh_from_cache)).lower()
    cycle_time = _ecmwf_cycle_time(solar)
    if cycle_time is not None:
        forecast.attrs["ecmwf_cycle_time"] = cycle_time.isoformat()
    _atomic_write_zarr(forecast, output_zarr)
    updated_archive = append_forecast_archive(forecast, archive_zarr)
    if skill_zarr is not None:
        skill = build_forecast_skill_dataset(updated_archive, power)
        _atomic_write_skill(skill, skill_zarr)
    if hindcast_zarr is not None:
        hindcast = build_soc_hindcast_dataset(updated_archive, power)
        _atomic_write_time_product(hindcast, hindcast_zarr)
    next_state = dict(state)
    next_state.update(
        {
            "updated_at_utc": _utc_now(),
            "solar_calibration_factor_w_per_wm2": float(forecast.attrs["solar_calibration_factor_w_per_wm2"]),
            "forecast_load_w": float(forecast.attrs["forecast_load_w"]),
            "load_bias_correction_w": float(forecast.attrs["load_bias_correction_w"]),
            "load_model": forecast.attrs["load_model"],
            "load_model_version": int(forecast.attrs["load_model_version"]),
            "load_regime": forecast.attrs["load_regime"],
            "load_regime_threshold_w": float(forecast.attrs["load_regime_threshold_w"]),
            "load_regime_level_w": float(forecast.attrs["load_regime_level_w"]),
            "load_regime_run_hours": float(forecast.attrs["load_regime_run_hours"]),
            "load_regime_sample_count": int(forecast.attrs["load_regime_sample_count"]),
            "current_load_mode": forecast.attrs["load_mode"],
            "current_load_mode_source": forecast.attrs["load_mode_source"],
            "current_load_mode_signature": forecast.attrs["load_mode_signature"],
            "current_load_mode_learning_ready": forecast.attrs["load_mode_learning_ready"] == "true",
            "current_load_mode_learning_reason": forecast.attrs["load_mode_learning_reason"],
            "minimum_operational_soc_pct": float(forecast.attrs["minimum_operational_soc_pct"]),
            "load_measurement": forecast.attrs["load_measurement"],
            "load_balance_measurement": forecast.attrs["load_balance_measurement"],
            "load_mode_registry": json.loads(forecast.attrs["load_mode_registry"]),
            "soc_bias_correction_pct_points_by_bucket": json.loads(
                forecast.attrs["soc_bias_correction_pct_points_by_bucket"]
            ),
            "latest_metrics": json.loads(forecast.attrs["previous_forecast_metrics"]),
            "latest_ecmwf_input_file": str(input_forecast),
            "latest_refresh_from_cache": bool(refresh_from_cache),
        }
    )
    _write_state(state_path, next_state)
    print(f"Wrote {output_zarr} with {forecast.sizes.get('time', 0)} forecast samples")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ECMWF-informed APS SOC forecast Zarr")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=POWER_PDU_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--input-forecast", type=Path, help="Existing ECMWF GRIB/NetCDF file to use instead of downloading")
    parser.add_argument("--cache-dir", type=Path, default=POWER_ECMWF_FORECAST_CACHE_DIR)
    parser.add_argument("--state", type=Path, default=POWER_SOC_FORECAST_STATE_PATH)
    parser.add_argument("--archive-zarr", type=Path, default=POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH)
    parser.add_argument("--skill-zarr", type=Path, default=POWER_SOC_FORECAST_SKILL_ZARR_PATH)
    parser.add_argument("--hindcast-zarr", type=Path, default=POWER_SOC_HINDCAST_ZARR_PATH)
    parser.add_argument("--no-skill-zarr", action="store_true", help="Do not write the forecast verification skill Zarr")
    parser.add_argument("--no-hindcast-zarr", action="store_true", help="Do not write the fixed-lead SOC hindcast Zarr")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument(
        "--refresh-from-cache",
        action="store_true",
        help="Reuse the latest cached ECMWF GRIB instead of downloading a new forecast",
    )
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        pdu_zarr=args.pdu_zarr,
        output_zarr=args.output_zarr,
        input_forecast=args.input_forecast,
        cache_dir=args.cache_dir,
        state_path=args.state,
        archive_zarr=args.archive_zarr,
        skill_zarr=None if args.no_skill_zarr else args.skill_zarr,
        hindcast_zarr=None if args.no_hindcast_zarr else args.hindcast_zarr,
        latitude=args.latitude,
        longitude=args.longitude,
        horizon_hours=args.horizon_hours,
        refresh_from_cache=args.refresh_from_cache,
    )


if __name__ == "__main__":
    main()
