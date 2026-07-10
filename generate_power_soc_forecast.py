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

DEFAULT_LATITUDE = float(os.environ.get("AURORA_POWER_FORECAST_LATITUDE", "64.829694"))
DEFAULT_LONGITUDE = float(os.environ.get("AURORA_POWER_FORECAST_LONGITUDE", "-23.248139"))
DEFAULT_HORIZON_HOURS = int(os.environ.get("AURORA_POWER_SOC_FORECAST_HOURS", "48"))
DEFAULT_CALIBRATION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_CALIBRATION_DAYS", "7"))
DEFAULT_FALLBACK_CALIBRATION_HOURS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_FALLBACK_CALIBRATION_HOURS", "48"))
DEFAULT_BATTERY_CAPACITY_KWH = float(os.environ.get("APS_BATTERY_CAPACITY_KWH", "26"))
DEFAULT_SOLAR_CALIBRATION_FACTOR = float(os.environ.get("AURORA_POWER_SOLAR_CALIBRATION_FACTOR", "1.0"))
DEFAULT_LOAD_W = float(os.environ.get("AURORA_POWER_FORECAST_DEFAULT_LOAD_W", "0"))
DEFAULT_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_ADAPTIVE_ALPHA", "0.25"))
DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W = float(os.environ.get("AURORA_POWER_LOAD_BIAS_CORRECTION_LIMIT_W", "2000"))
DEFAULT_OPEN_DATA_SOURCE = os.environ.get("AURORA_POWER_ECMWF_OPEN_DATA_SOURCE", "azure")
ECMWF_PARAM = "ssrd"


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


def retrieve_open_data_grib(
    output_grib: Path,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    param: str = ECMWF_PARAM,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
) -> Path:
    """Retrieve ECMWF open-data solar forecast GRIB for the requested horizon."""
    try:
        from ecmwf.opendata import Client
    except Exception as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError("Install ecmwf-opendata to retrieve ECMWF open-data forecasts") from exc

    steps = list(range(0, int(horizon_hours) + 1, 3))
    if steps[-1] != int(horizon_hours):
        steps.append(int(horizon_hours))
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
    pieces = []
    for name in ("ACOutputWatts", "DCInverterWatts"):
        if name in window:
            pieces.append(window[name])
    if not pieces:
        return float(DEFAULT_LOAD_W)
    load = pd.concat(pieces, axis=1).sum(axis=1, min_count=1).clip(lower=0.0)
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
    """Forecast station load from recent historical UTC-hour medians."""
    forecast_times = pd.DatetimeIndex(forecast_times)
    if len(forecast_times) == 0:
        return pd.Series(dtype=np.float64)
    start = end - pd.Timedelta(days=float(calibration_days))
    load = _observed_load_w(frame)
    load = load.loc[(load.index >= start) & (load.index <= end)]
    finite = load[np.isfinite(load)]
    if finite.empty:
        return pd.Series(np.full(len(forecast_times), float(default_load_w)), index=forecast_times)

    # Collapse dense APS samples before grouping, otherwise one-second bursts can
    # dominate the hourly historical profile.
    hourly_samples = finite.resample("15min").median().dropna()
    if hourly_samples.empty:
        hourly_samples = finite
    fallback = float(hourly_samples.median()) if not hourly_samples.empty else float(default_load_w)
    by_hour = hourly_samples.groupby(hourly_samples.index.hour).median()
    values = [float(by_hour.get(ts.hour, fallback)) for ts in forecast_times]
    return pd.Series(np.clip(values, 0.0, None), index=forecast_times)


def _observed_solar_w(frame: pd.DataFrame) -> pd.Series:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if not solar_fields:
        return pd.Series(dtype=np.float64)
    return frame[solar_fields].sum(axis=1, min_count=1).clip(lower=0.0)


def _observed_load_w(frame: pd.DataFrame) -> pd.Series:
    load_fields = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in frame]
    if not load_fields:
        return pd.Series(dtype=np.float64)
    return frame[load_fields].sum(axis=1, min_count=1).clip(lower=0.0)


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
    previous_forecast: xr.Dataset | None = None,
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
    previous_metrics = evaluate_previous_forecast(previous_forecast, frame)
    irradiance = solar_irradiance_from_ssrd(solar)
    if irradiance.empty:
        raise ValueError("No ECMWF solar forecast samples could be converted from ssrd")
    irradiance = irradiance[(irradiance.index >= latest_time) & (irradiance.index <= latest_time + pd.Timedelta(hours=horizon_hours))]
    if len(irradiance) < 2:
        raise ValueError("ECMWF solar forecast does not overlap the requested SOC forecast horizon")

    factor_raw = calibrate_solar_factor(
        frame,
        irradiance,
        end=latest_time,
        calibration_days=calibration_days,
        fallback_hours=fallback_calibration_hours,
    )
    factor = _adaptive_value(factor_raw, state.get("solar_calibration_factor_w_per_wm2"))
    raw_load_profile = build_historical_load_forecast(
        frame,
        pd.DatetimeIndex(irradiance.index),
        end=latest_time,
        calibration_days=calibration_days,
        default_load_w=DEFAULT_LOAD_W,
    )
    load_bias_correction = _load_bias_correction(
        state.get("load_bias_correction_w"),
        previous_metrics.get("load_bias_w"),
    )
    load_profile = (raw_load_profile + load_bias_correction).clip(lower=0.0)
    raw_load_w = float(raw_load_profile.median()) if not raw_load_profile.empty else float(DEFAULT_LOAD_W)
    load_w = float(load_profile.median()) if not load_profile.empty else float(DEFAULT_LOAD_W)
    forecast = integrate_soc_forecast(
        initial_soc=latest_soc,
        initial_time=latest_time,
        irradiance=irradiance,
        solar_factor=factor,
        load_w=load_profile,
        capacity_kwh=capacity_kwh,
    )
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
    forecast["ForecastSolarMAERecent"] = solar_mae
    forecast["ForecastLoadMAERecent"] = load_mae
    forecast["ForecastLoadBiasRecent"] = load_bias
    forecast["ForecastEvaluationSamples"] = float(evaluation_samples)
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
            "load_model": "historical_utc_hour_median",
            "battery_capacity_kwh": f"{float(capacity_kwh):.6g}",
            "previous_forecast_metrics": json.dumps(previous_metrics, sort_keys=True),
        },
    )
    out["BatterySOCForecast"].attrs["units"] = "%"
    out["ECMWFSolarIrradiance"].attrs["units"] = "W m-2"
    out["ForecastSolarWatts"].attrs["units"] = "W"
    out["ForecastLoadWatts"].attrs["units"] = "W"
    out["ForecastSOCMAERecent"].attrs["units"] = "percentage points"
    out["ForecastSolarMAERecent"].attrs["units"] = "W"
    out["ForecastLoadMAERecent"].attrs["units"] = "W"
    out["ForecastLoadBiasRecent"].attrs["units"] = "W"
    out["ForecastEvaluationSamples"].attrs["units"] = "samples"
    return out


def generate(
    power_zarr: Path = POWER_ZARR_PATH,
    output_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    *,
    input_forecast: Path | None = None,
    cache_dir: Path = POWER_ECMWF_FORECAST_CACHE_DIR,
    state_path: Path = POWER_SOC_FORECAST_STATE_PATH,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> Path:
    if input_forecast is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        input_forecast = cache_dir / f"ecmwf_ssrd_{stamp}.grib2"
        retrieve_open_data_grib(input_forecast, horizon_hours=horizon_hours)
    power = xr.open_zarr(power_zarr, chunks={})
    solar = open_solar_forecast(input_forecast, latitude=latitude, longitude=longitude)
    previous_forecast = None
    if output_zarr.exists():
        try:
            previous_forecast = xr.open_zarr(output_zarr, chunks={})
        except Exception:
            previous_forecast = None
    state = _load_state(state_path)
    forecast = build_forecast_dataset(
        power,
        solar,
        previous_forecast=previous_forecast,
        state=state,
        horizon_hours=horizon_hours,
    )
    forecast.attrs["ecmwf_input_file"] = str(input_forecast)
    forecast.attrs["site_latitude"] = str(float(latitude))
    forecast.attrs["site_longitude"] = str(float(longitude))
    _atomic_write_zarr(forecast, output_zarr)
    next_state = dict(state)
    next_state.update(
        {
            "updated_at_utc": _utc_now(),
            "solar_calibration_factor_w_per_wm2": float(forecast.attrs["solar_calibration_factor_w_per_wm2"]),
            "forecast_load_w": float(forecast.attrs["forecast_load_w"]),
            "load_bias_correction_w": float(forecast.attrs["load_bias_correction_w"]),
            "latest_metrics": json.loads(forecast.attrs["previous_forecast_metrics"]),
        }
    )
    _write_state(state_path, next_state)
    print(f"Wrote {output_zarr} with {forecast.sizes.get('time', 0)} forecast samples")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ECMWF-informed APS SOC forecast Zarr")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--input-forecast", type=Path, help="Existing ECMWF GRIB/NetCDF file to use instead of downloading")
    parser.add_argument("--cache-dir", type=Path, default=POWER_ECMWF_FORECAST_CACHE_DIR)
    parser.add_argument("--state", type=Path, default=POWER_SOC_FORECAST_STATE_PATH)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        output_zarr=args.output_zarr,
        input_forecast=args.input_forecast,
        cache_dir=args.cache_dir,
        state_path=args.state,
        latitude=args.latitude,
        longitude=args.longitude,
        horizon_hours=args.horizon_hours,
    )


if __name__ == "__main__":
    main()
