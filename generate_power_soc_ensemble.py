#!/usr/bin/env python3
"""Generate a 50-member ECMWF solar-driven APS SOC ensemble forecast."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_AC_MODE_THRESHOLD_W,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_OPEN_DATA_SOURCE,
    _bounded_load_profile,
    _observed_load_w,
    _power_frame,
    build_historical_load_forecast,
    integrate_soc_forecast,
    latest_finite,
    solar_irradiance_from_ssrd,
)
from power_soc_thresholds import (
    MINIMUM_OPERATIONAL_SOC_PCT,
    SOC_BELOW_THRESHOLD_BRIER_FIELD,
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,
)

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
POWER_SOC_FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
POWER_SOC_ENSEMBLE_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_forecast.zarr")
)
POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH = Path(
    os.environ.get(
        "POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_archive.zarr"
    )
)
POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_skill.zarr")
)
POWER_ECMWF_ENSEMBLE_TMP_DIR = Path(
    os.environ.get("POWER_ECMWF_ENSEMBLE_TMP_DIR", "/data/aurora/products/power/ecmwf_solar_ensemble_tmp")
)

ENSEMBLE_MEMBERS = tuple(range(1, 51))
LEAD_BUCKETS = (("0_6h", 0.0, 6.0), ("6_24h", 6.0, 24.0), ("24_48h", 24.0, 48.0), ("48_96h", 48.0, 96.0))
ENSEMBLE_INPUT_ATTRS = (
    "initial_soc_time",
    "initial_soc_pct",
    "solar_calibration_factor_w_per_wm2",
    "battery_capacity_kwh",
    "load_bias_correction_w",
    "forecast_load_w",
    "load_model",
    "load_model_version",
    "load_mode",
    "load_mode_source",
    "load_mode_active_kits",
    "load_mode_signature",
)
NUMERIC_ENSEMBLE_INPUT_ATTRS = {
    "initial_soc_pct",
    "solar_calibration_factor_w_per_wm2",
    "battery_capacity_kwh",
    "load_bias_correction_w",
    "forecast_load_w",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensemble_refresh_reasons(
    current_attrs: Mapping[str, object],
    deterministic_attrs: Mapping[str, object],
) -> list[str]:
    """Return deterministic inputs that require a same-cycle re-anchoring run."""
    reasons: list[str] = []
    for name in ENSEMBLE_INPUT_ATTRS:
        desired = deterministic_attrs.get(name)
        if desired is None:
            continue
        current = current_attrs.get(name)
        if name in NUMERIC_ENSEMBLE_INPUT_ATTRS:
            try:
                matches = bool(np.isclose(float(current), float(desired), rtol=0.0, atol=1e-6))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(current) == str(desired)
        if not matches:
            reasons.append(name)
    return reasons


def latest_ensemble_cycle(*, source: str = DEFAULT_OPEN_DATA_SOURCE) -> pd.Timestamp:
    from ecmwf.opendata import Client

    client = Client(source=source)
    candidates = []
    for hour in (0, 12):
        value = client.latest(stream="enfo", type="pf", time=hour)
        candidates.append(pd.Timestamp(value).tz_localize(None))
    return max(candidates)


def retrieve_ensemble_grib(
    target: Path,
    *,
    cycle: pd.Timestamp,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
) -> Path:
    from ecmwf.opendata import Client

    requested_horizon = int(horizon_hours) + 24
    steps = list(range(0, requested_horizon + 1, 3))
    target.parent.mkdir(parents=True, exist_ok=True)
    Client(source=source, preserve_request_order=True).retrieve(
        date=cycle.strftime("%Y%m%d"),
        time=int(cycle.hour),
        stream="enfo",
        type="pf",
        number=list(ENSEMBLE_MEMBERS),
        levtype="sfc",
        param="ssrd",
        step=steps,
        target=str(target),
    )
    return target


def open_ensemble_site(path: Path, *, latitude: float, longitude: float) -> xr.Dataset:
    if path.suffix.lower() in {".grib", ".grib2", ".grb", ".grb2"}:
        return _open_grib_ensemble_site(path, latitude=latitude, longitude=longitude)
    ds = xr.open_dataset(path)
    if "surface_solar_radiation_downwards" in ds and "ssrd" not in ds:
        ds = ds.rename({"surface_solar_radiation_downwards": "ssrd"})
    if "ssrd" not in ds:
        raise KeyError("ECMWF ensemble does not contain ssrd")
    lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
    lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
    if lat_name and lon_name:
        select_lon = longitude
        if float(ds[lon_name].min()) >= 0.0 and select_lon < 0.0:
            select_lon %= 360.0
        # cfgrib's label-based nearest selection can cause its backend to allocate
        # the complete member x step x global-grid array. Resolve the two scalar
        # indices from the small coordinate vectors and slice before loading.
        latitude_index = int(np.abs(np.asarray(ds[lat_name].values) - latitude).argmin())
        longitude_index = int(np.abs(np.asarray(ds[lon_name].values) - select_lon).argmin())
        ds = ds.isel({lat_name: latitude_index, lon_name: longitude_index})
    return ds.load()


def _open_grib_ensemble_site(path: Path, *, latitude: float, longitude: float) -> xr.Dataset:
    """Stream one site from a multi-member GRIB without allocating its global grid."""
    from eccodes import (
        codes_get,
        codes_grib_find_nearest,
        codes_grib_new_from_file,
        codes_release,
    )

    records: list[tuple[int, int, float]] = []
    cycle: pd.Timestamp | None = None
    selected_latitude: float | None = None
    selected_longitude: float | None = None
    with path.open("rb") as handle:
        while True:
            grib_id = codes_grib_new_from_file(handle)
            if grib_id is None:
                break
            try:
                nearest = codes_grib_find_nearest(grib_id, latitude, longitude)[0]
                member = int(codes_get(grib_id, "number"))
                step = int(codes_get(grib_id, "step"))
                records.append((member, step, float(nearest["value"])))
                selected_latitude = float(nearest["lat"])
                selected_longitude = float(nearest["lon"])
                if cycle is None:
                    date = int(codes_get(grib_id, "dataDate"))
                    time = int(codes_get(grib_id, "dataTime"))
                    cycle = pd.Timestamp(datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M"))
            finally:
                codes_release(grib_id)

    if not records or cycle is None:
        raise ValueError(f"No ECMWF ensemble messages found in {path}")
    frame = pd.DataFrame(records, columns=["number", "step", "ssrd"])
    frame = frame.drop_duplicates(["number", "step"], keep="last")
    values = frame.pivot(index="number", columns="step", values="ssrd").sort_index().sort_index(axis=1)
    steps = pd.to_timedelta(values.columns.to_numpy(dtype=np.int64), unit="h")
    return xr.Dataset(
        {"ssrd": (("number", "step"), values.to_numpy(dtype=np.float64))},
        coords={
            "number": values.index.to_numpy(dtype=np.int64),
            "step": steps.to_numpy(),
            "time": cycle.to_datetime64(),
            "valid_time": ("step", (cycle + steps).to_numpy()),
            "latitude": selected_latitude,
            "longitude": selected_longitude,
        },
    )


def _member_dimension(ds: xr.Dataset) -> str:
    for name in ("number", "member", "realization"):
        if name in ds["ssrd"].dims:
            return name
    raise ValueError("ECMWF ensemble ssrd has no member dimension")


def _load_residual_offsets(frame: pd.DataFrame, *, end: pd.Timestamp, members: int) -> np.ndarray:
    observed = _observed_load_w(frame)
    observed = observed[(observed.index >= end - pd.Timedelta(days=7)) & (observed.index <= end)]
    sampled = observed.resample("15min").median().dropna()
    if sampled.empty:
        return np.zeros(members, dtype=np.float64)
    if "ACOutputWatts" in frame:
        ac = frame.loc[(frame.index >= observed.index.min()) & (frame.index <= end), "ACOutputWatts"]
        ac = ac.resample("15min").median()
        ac = ac.reindex(sampled.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        recent_ac = ac[ac.index >= end - pd.Timedelta(minutes=30)].dropna()
        current_active = bool(not recent_ac.empty and float(recent_ac.median()) > DEFAULT_AC_MODE_THRESHOLD_W)
        sampled = sampled[(ac.fillna(0.0) > DEFAULT_AC_MODE_THRESHOLD_W) == current_active]
    expected = float(sampled.median())
    residuals = (sampled - expected).replace([np.inf, -np.inf], np.nan).dropna().clip(-500.0, 500.0)
    if len(residuals) < 10:
        return np.zeros(members, dtype=np.float64)
    quantiles = (np.arange(members, dtype=np.float64) + 0.5) / members
    return np.quantile(residuals.to_numpy(dtype=np.float64), quantiles)


def apply_operational_soc_threshold(ds: xr.Dataset) -> xr.Dataset:
    """Refresh threshold-derived fields without rerunning the ECMWF ensemble."""
    if "BatterySOCForecastEnsemble" not in ds:
        raise KeyError("Ensemble forecast does not contain BatterySOCForecastEnsemble")
    out = ds.copy()
    obsolete = [
        name
        for name in out.data_vars
        if name.startswith("BatterySOCBelow")
        and name.endswith("Probability")
        and name != SOC_BELOW_THRESHOLD_PROBABILITY_FIELD
    ]
    if obsolete:
        out = out.drop_vars(obsolete)
    probability = np.mean(
        np.asarray(out["BatterySOCForecastEnsemble"].values, dtype=np.float64)
        < MINIMUM_OPERATIONAL_SOC_PCT,
        axis=0,
    ).astype(np.float32)
    out[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD] = (("time",), probability)
    out[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD].attrs["units"] = "1"
    out.attrs["minimum_operational_soc_pct"] = f"{MINIMUM_OPERATIONAL_SOC_PCT:g}"
    return out


def build_ensemble_dataset(
    power: xr.Dataset,
    deterministic: xr.Dataset,
    solar_ensemble: xr.Dataset,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> xr.Dataset:
    frame = _power_frame(power)
    latest_time, latest_soc = latest_finite(frame["BatterySOC"])
    member_dim = _member_dimension(solar_ensemble)
    member_values = np.asarray(solar_ensemble[member_dim].values)
    solar_factor = float(deterministic.attrs.get("solar_calibration_factor_w_per_wm2", 1.0))
    capacity_kwh = float(deterministic.attrs.get("battery_capacity_kwh", DEFAULT_BATTERY_CAPACITY_KWH))
    correction_w = float(deterministic.attrs.get("load_bias_correction_w", 0.0))

    member_irradiance: list[pd.Series] = []
    for member in member_values:
        series = solar_irradiance_from_ssrd(solar_ensemble.sel({member_dim: member}))
        series = series[(series.index >= latest_time) & (series.index <= latest_time + pd.Timedelta(hours=horizon_hours))]
        if len(series) >= 2:
            member_irradiance.append(series)
    if not member_irradiance:
        raise ValueError("No ECMWF ensemble members overlap the requested SOC horizon")
    common_times = member_irradiance[0].index
    common_times = common_times[(common_times >= latest_time) & (common_times <= latest_time + pd.Timedelta(hours=horizon_hours))]
    if "ForecastLoadWatts" in deterministic and "time" in deterministic:
        central_load = pd.Series(
            np.asarray(deterministic["ForecastLoadWatts"].values, dtype=np.float64),
            index=pd.DatetimeIndex(deterministic["time"].values),
        ).reindex(common_times, method="nearest", tolerance=pd.Timedelta(hours=2))
        central_load = central_load.ffill().bfill().clip(lower=0.0)
    else:
        raw_load = build_historical_load_forecast(frame, common_times, end=latest_time, calibration_days=7)
        central_load, _ = _bounded_load_profile(raw_load, correction_w)
    offsets = _load_residual_offsets(frame, end=latest_time, members=len(member_irradiance))

    soc_rows = []
    irr_rows = []
    solar_rows = []
    load_rows = []
    output_times: pd.DatetimeIndex | None = None
    for index, irradiance in enumerate(member_irradiance):
        irradiance = irradiance.reindex(common_times).interpolate().ffill().bfill()
        load = (central_load.reindex(common_times).ffill().bfill() + offsets[index]).clip(lower=0.0)
        result = integrate_soc_forecast(
            initial_soc=latest_soc,
            initial_time=latest_time,
            irradiance=irradiance,
            solar_factor=solar_factor,
            load_w=load,
            capacity_kwh=capacity_kwh,
        )
        output_times = pd.DatetimeIndex(result.index)
        soc_rows.append(result["BatterySOCForecast"].to_numpy(dtype=np.float32))
        irr_rows.append(result["ECMWFSolarIrradiance"].to_numpy(dtype=np.float32))
        solar_rows.append(result["ForecastSolarWatts"].to_numpy(dtype=np.float32))
        load_rows.append(result["ForecastLoadWatts"].to_numpy(dtype=np.float32))
    assert output_times is not None
    soc = np.asarray(soc_rows, dtype=np.float32)
    out = xr.Dataset(
        {
            "BatterySOCForecastEnsemble": (("member", "time"), soc),
            "ECMWFSolarIrradianceEnsemble": (("member", "time"), np.asarray(irr_rows, dtype=np.float32)),
            "ForecastSolarWattsEnsemble": (("member", "time"), np.asarray(solar_rows, dtype=np.float32)),
            "ForecastLoadWattsEnsemble": (("member", "time"), np.asarray(load_rows, dtype=np.float32)),
            "BatterySOCForecastP10": (("time",), np.nanquantile(soc, 0.10, axis=0).astype(np.float32)),
            "BatterySOCForecastP50": (("time",), np.nanquantile(soc, 0.50, axis=0).astype(np.float32)),
            "BatterySOCForecastP90": (("time",), np.nanquantile(soc, 0.90, axis=0).astype(np.float32)),
            "BatterySOCForecastMinimum": (("time",), np.nanmin(soc, axis=0).astype(np.float32)),
            "BatterySOCForecastMaximum": (("time",), np.nanmax(soc, axis=0).astype(np.float32)),
        },
        coords={"member": np.arange(1, soc.shape[0] + 1, dtype=np.int16), "time": output_times.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_ensemble_forecast_product": "true",
            "generated_at_utc": _utc_now(),
            "initial_soc_time": latest_time.isoformat(),
            "initial_soc_pct": f"{latest_soc:.6g}",
            "forecast_horizon_hours": str(int(horizon_hours)),
            "ensemble_members": str(int(soc.shape[0])),
            "solar_calibration_factor_w_per_wm2": f"{solar_factor:.6g}",
            "battery_capacity_kwh": f"{capacity_kwh:.6g}",
            "load_bias_correction_w": f"{correction_w:.6g}",
            "forecast_load_w": str(deterministic.attrs.get("forecast_load_w", "")),
            "load_model": str(deterministic.attrs.get("load_model", "kit_mode_persistence_v4")),
            "load_model_version": str(deterministic.attrs.get("load_model_version", "4")),
            "load_mode": str(deterministic.attrs.get("load_mode", "unknown")),
            "load_mode_source": str(deterministic.attrs.get("load_mode_source", "unknown")),
            "load_mode_active_kits": str(deterministic.attrs.get("load_mode_active_kits", "")),
            "load_mode_signature": str(deterministic.attrs.get("load_mode_signature", "")),
            "load_uncertainty": "recent same-regime residual quantiles",
            "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
            "source": "ECMWF IFS perturbed ssrd members plus APS power history",
        },
    )
    out = apply_operational_soc_threshold(out)
    for name in out.data_vars:
        out[name].attrs["units"] = "1" if name.endswith("Probability") else "%" if "SOC" in name else "W m-2" if "Irradiance" in name else "W"
    return out


def _write_forecast(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.chunk({"member": min(ds.sizes["member"], 10), "time": min(ds.sizes["time"], 64)}).to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def _ensemble_site_cache_path(cache_dir: Path, cycle: pd.Timestamp, horizon_hours: int) -> Path:
    stamp = cycle.strftime("%Y%m%dT%H%M%SZ")
    return cache_dir / f"ecmwf_ens_ssrd_{stamp}_h{int(horizon_hours)}.site.zarr"


def _write_ensemble_site_cache(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def _prune_ensemble_site_cache(cache_dir: Path, *, keep: int = 4) -> None:
    paths = sorted(cache_dir.glob("ecmwf_ens_ssrd_*_h*.site.zarr"), key=lambda path: path.name)
    for path in paths[:-keep]:
        shutil.rmtree(path)


def _archive_row(forecast: xr.Dataset) -> xr.Dataset:
    issue = pd.Timestamp(forecast.attrs["initial_soc_time"])
    times = pd.DatetimeIndex(forecast["time"].values)
    return xr.Dataset(
        {
            "ForecastValidTime": (("issue_time", "forecast_step"), times.to_numpy(dtype="datetime64[ns]")[None, :]),
            "ForecastLeadHours": (("issue_time", "forecast_step"), (((times - issue) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32))[None, :]),
            "BatterySOCForecastEnsemble": (("issue_time", "member", "forecast_step"), forecast["BatterySOCForecastEnsemble"].values[None, :, :]),
        },
        coords={
            "issue_time": np.array([issue.to_datetime64()], dtype="datetime64[ns]"),
            "member": forecast["member"].values,
            "forecast_step": np.arange(len(times), dtype=np.int16),
        },
    )


def append_ensemble_archive(forecast: xr.Dataset, path: Path, *, retention_days: float = 21.0) -> xr.Dataset:
    row = _archive_row(forecast)
    previous = xr.open_zarr(path, chunks={}).load() if path.exists() else None
    if previous is not None:
        steps = np.arange(max(previous.sizes["forecast_step"], row.sizes["forecast_step"]), dtype=np.int16)
        combined = xr.concat([previous.reindex(forecast_step=steps), row.reindex(forecast_step=steps)], dim="issue_time")
        combined = combined.sortby("issue_time")
        combined = combined.isel(issue_time=~combined.indexes["issue_time"].duplicated(keep="last"))
    else:
        combined = row
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=retention_days)
    combined = combined.isel(issue_time=pd.DatetimeIndex(combined.issue_time.values) >= cutoff)
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    combined.chunk({"issue_time": 8, "member": 10, "forecast_step": 64}).to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    return combined


def _crps_ensemble(members: np.ndarray, observation: float) -> float:
    members = members[np.isfinite(members)]
    if members.size < 2 or not np.isfinite(observation):
        return np.nan
    first = np.mean(np.abs(members - observation))
    second = 0.5 * np.mean(np.abs(members[:, None] - members[None, :]))
    return float(first - second)


def build_ensemble_skill_dataset(archive: xr.Dataset, power: xr.Dataset, *, retention_days: float = 7.0) -> xr.Dataset:
    frame = _power_frame(power)
    observed = frame.get("BatterySOC", pd.Series(dtype=np.float64))
    end = pd.Timestamp(frame.index.max())
    times = pd.date_range((end - pd.Timedelta(days=retention_days)).floor("1h"), end.ceil("1h"), freq="1h")
    fields = [f"ForecastSOCCRPS_{bucket}" for bucket, _, _ in LEAD_BUCKETS]
    fields += ["ForecastSOCIntervalCoverage80", SOC_BELOW_THRESHOLD_BRIER_FIELD, "ForecastEnsembleCycles"]
    values = {name: np.full(len(times), np.nan, dtype=np.float32) for name in fields}
    valid_times = pd.DatetimeIndex(archive["ForecastValidTime"].values.reshape(-1))
    leads = archive["ForecastLeadHours"].values.reshape(-1)
    ensembles = archive["BatterySOCForecastEnsemble"].values.transpose(0, 2, 1).reshape(-1, archive.sizes["member"])
    issues = np.repeat(pd.DatetimeIndex(archive.issue_time.values), archive.sizes["forecast_step"])
    observed_values = observed.reindex(valid_times, method="nearest", tolerance=pd.Timedelta(minutes=10)).to_numpy(dtype=np.float64)
    rows = pd.DataFrame({"valid_time": valid_times, "issue_time": issues, "lead": leads, "observed": observed_values})
    rows["sample_index"] = np.arange(len(rows))
    rows = rows[np.isfinite(rows["observed"]) & (rows["valid_time"] <= end)]
    for index, now in enumerate(times):
        selected = rows[(rows.valid_time > now - pd.Timedelta(hours=24)) & (rows.valid_time <= now)]
        if selected.empty:
            continue
        values["ForecastEnsembleCycles"][index] = float(selected.issue_time.nunique())
        coverage = []
        brier = []
        for bucket, start, stop in LEAD_BUCKETS:
            bucket_rows = selected[(selected.lead >= start) & (selected.lead < stop)]
            scores = []
            for row in bucket_rows.itertuples(index=False):
                member_values = ensembles[int(row.sample_index)]
                scores.append(_crps_ensemble(member_values, float(row.observed)))
                coverage.append(float(np.nanquantile(member_values, 0.1) <= row.observed <= np.nanquantile(member_values, 0.9)))
                probability = float(np.mean(member_values < MINIMUM_OPERATIONAL_SOC_PCT))
                brier.append((probability - float(row.observed < MINIMUM_OPERATIONAL_SOC_PCT)) ** 2)
            if scores:
                values[f"ForecastSOCCRPS_{bucket}"][index] = float(np.nanmean(scores))
        if coverage:
            values["ForecastSOCIntervalCoverage80"][index] = float(np.mean(coverage))
        if brier:
            values[SOC_BELOW_THRESHOLD_BRIER_FIELD][index] = float(np.mean(brier))
    out = xr.Dataset({name: (("time",), data) for name, data in values.items()}, coords={"time": times.values})
    out.attrs = {
        "power_soc_ensemble_skill_product": "true",
        "generated_at_utc": _utc_now(),
        "source": "archived ECMWF SOC ensemble forecasts verified against APS BatterySOC",
        "verification_window_hours": "24",
        "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
    }
    for name in out.data_vars:
        out[name].attrs["units"] = "cycles" if name.endswith("Cycles") else "1" if name.endswith(("Coverage80", "Brier")) else "percentage points"
    return out


def _write_time_product(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.chunk({"time": min(ds.sizes.get("time", 1), 168)}).to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def generate(
    *,
    power_zarr: Path = POWER_ZARR_PATH,
    deterministic_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    output_zarr: Path = POWER_SOC_ENSEMBLE_ZARR_PATH,
    archive_zarr: Path = POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH,
    skill_zarr: Path = POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH,
    input_forecast: Path | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
    cache_dir: Path = POWER_ECMWF_ENSEMBLE_TMP_DIR,
) -> Path:
    cycle = latest_ensemble_cycle(source=source) if input_forecast is None else None
    if cycle is not None and output_zarr.exists():
        deterministic_metadata = xr.open_zarr(deterministic_zarr, chunks={})
        deterministic_attrs = dict(deterministic_metadata.attrs)
        deterministic_metadata.close()
        current = xr.open_zarr(output_zarr, chunks={})
        if current.attrs.get("ecmwf_cycle_time") == cycle.isoformat():
            refresh_reasons = _ensemble_refresh_reasons(current.attrs, deterministic_attrs)
            obsolete = [
                name
                for name in current.data_vars
                if name.startswith("BatterySOCBelow")
                and name.endswith("Probability")
                and name != SOC_BELOW_THRESHOLD_PROBABILITY_FIELD
            ]
            threshold_changed = (
                current.attrs.get("minimum_operational_soc_pct") != f"{MINIMUM_OPERATIONAL_SOC_PCT:g}"
                or SOC_BELOW_THRESHOLD_PROBABILITY_FIELD not in current
                or bool(obsolete)
            )
            if refresh_reasons:
                print(
                    f"Re-anchoring ECMWF ensemble cycle {cycle.isoformat()} for updated inputs: "
                    f"{', '.join(refresh_reasons)}"
                )
                current.close()
            elif threshold_changed:
                current.load()
                current = apply_operational_soc_threshold(current)
                current.attrs["threshold_metrics_updated_at_utc"] = _utc_now()
                _write_forecast(current, output_zarr)
                if archive_zarr.exists():
                    archive = xr.open_zarr(archive_zarr, chunks={}).load()
                    power = xr.open_zarr(power_zarr, chunks={})
                    skill = build_ensemble_skill_dataset(archive, power)
                    _write_time_product(skill, skill_zarr)
                    power.close()
                print(
                    f"Refreshed ensemble diagnostics for the "
                    f"{MINIMUM_OPERATIONAL_SOC_PCT:g}% operational threshold"
                )
                current.close()
                return output_zarr
            else:
                current.close()
                print(f"ECMWF ensemble cycle {cycle.isoformat()} already matches current forecast inputs")
                return output_zarr
        else:
            current.close()

    downloaded_input: Path | None = None
    solar: xr.Dataset | None = None
    power: xr.Dataset | None = None
    deterministic: xr.Dataset | None = None
    try:
        if input_forecast is None:
            assert cycle is not None
            site_cache = _ensemble_site_cache_path(cache_dir, cycle, horizon_hours)
            if site_cache.exists():
                solar = xr.open_zarr(site_cache, chunks={}).load()
            else:
                stamp = cycle.strftime("%Y%m%dT%H%M%SZ")
                downloaded_input = cache_dir / f"ecmwf_ens_ssrd_{stamp}.grib2"
                retrieve_ensemble_grib(downloaded_input, cycle=cycle, horizon_hours=horizon_hours, source=source)
                solar = open_ensemble_site(downloaded_input, latitude=latitude, longitude=longitude)
                _write_ensemble_site_cache(solar, site_cache)
                _prune_ensemble_site_cache(cache_dir)
        else:
            solar = open_ensemble_site(input_forecast, latitude=latitude, longitude=longitude)
        assert solar is not None
        power = xr.open_zarr(power_zarr, chunks={})
        deterministic = xr.open_zarr(deterministic_zarr, chunks={})
        forecast = build_ensemble_dataset(power, deterministic, solar, horizon_hours=horizon_hours)
        cycle_time = cycle or pd.Timestamp(np.asarray(solar["time"].values).reshape(-1)[0])
        forecast.attrs["ecmwf_cycle_time"] = pd.Timestamp(cycle_time).isoformat()
        _write_forecast(forecast, output_zarr)
        archive = append_ensemble_archive(forecast, archive_zarr)
        skill = build_ensemble_skill_dataset(archive, power)
        _write_time_product(skill, skill_zarr)
    finally:
        if deterministic is not None:
            deterministic.close()
        if power is not None:
            power.close()
        if solar is not None:
            solar.close()
        if downloaded_input is not None and downloaded_input.exists():
            downloaded_input.unlink()
            for sidecar in downloaded_input.parent.glob(f"{downloaded_input.name}.*.idx"):
                sidecar.unlink(missing_ok=True)
    print(f"Wrote {output_zarr} with {forecast.sizes['member']} members")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ECMWF 50-member APS SOC ensemble forecast")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--deterministic-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_SOC_ENSEMBLE_ZARR_PATH)
    parser.add_argument("--archive-zarr", type=Path, default=POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH)
    parser.add_argument("--skill-zarr", type=Path, default=POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH)
    parser.add_argument("--input-forecast", type=Path)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--source", default=DEFAULT_OPEN_DATA_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=POWER_ECMWF_ENSEMBLE_TMP_DIR)
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        deterministic_zarr=args.deterministic_zarr,
        output_zarr=args.output_zarr,
        archive_zarr=args.archive_zarr,
        skill_zarr=args.skill_zarr,
        input_forecast=args.input_forecast,
        latitude=args.latitude,
        longitude=args.longitude,
        horizon_hours=args.horizon_hours,
        source=args.source,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
