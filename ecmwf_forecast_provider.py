"""ECMWF forecast retrieval and decoding providers for APS forecasts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

PROVIDER_LEGACY = "legacy"
PROVIDER_EARTHKIT = "earthkit"
PROVIDER_SHADOW = "shadow"
VALID_PROVIDERS = (PROVIDER_LEGACY, PROVIDER_EARTHKIT, PROVIDER_SHADOW)

DEFAULT_PROVIDER = os.environ.get("AURORA_ECMWF_PROVIDER", PROVIDER_LEGACY).strip().lower()
DEFAULT_SHADOW_REPORT_PATH = Path(
    os.environ.get(
        "AURORA_ECMWF_SHADOW_REPORT_PATH",
        "/data/aurora/products/power/ecmwf_provider_shadow.json",
    )
)


@dataclass
class ForecastProviderResult:
    dataset: xr.Dataset
    diagnostics: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def validate_provider(provider: str) -> str:
    provider = str(provider).strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown ECMWF provider {provider!r}; expected one of {', '.join(VALID_PROVIDERS)}")
    return provider


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _steps(horizon_hours: int, lookahead_buffer_hours: int) -> list[int]:
    requested_horizon = min(max(int(horizon_hours) + max(int(lookahead_buffer_hours), 0), 0), 240)
    if requested_horizon <= 144:
        final_step = min(((requested_horizon + 2) // 3) * 3, 144)
        return list(range(0, final_step + 1, 3))
    values = list(range(0, 145, 3))
    final_step = min(150 + ((requested_horizon - 150 + 5) // 6) * 6, 240)
    values.extend(range(150, final_step + 1, 6))
    return values


def retrieve_open_data_grib(
    output_grib: Path,
    *,
    provider: str,
    horizon_hours: int,
    lookahead_buffer_hours: int,
    param: str,
    source: str,
    cycle_hour: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Retrieve an ECMWF Open Data forecast with the selected client."""
    provider = validate_provider(provider)
    requested_provider = PROVIDER_EARTHKIT if provider == PROVIDER_EARTHKIT else PROVIDER_LEGACY
    output_grib.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "type": "fc",
        "stream": "oper",
        "levtype": "sfc",
        "param": param,
        "step": _steps(horizon_hours, lookahead_buffer_hours),
    }
    if cycle_hour is not None:
        if int(cycle_hour) not in {0, 12}:
            raise ValueError("Long-range ECMWF deterministic cycles must be 00 or 12 UTC")
        request["time"] = int(cycle_hour)
    started = time.perf_counter()
    fallback_reason = ""
    try:
        if requested_provider == PROVIDER_EARTHKIT:
            import earthkit.data as ekd

            data = ekd.from_source("ecmwf-open-data", source=source, request=request)
            data.to_target("file", str(output_grib))
        else:
            from ecmwf.opendata import Client

            Client(source=source).retrieve(**request, target=str(output_grib))
        effective_provider = requested_provider
    except Exception as exc:
        if requested_provider != PROVIDER_EARTHKIT:
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        from ecmwf.opendata import Client

        Client(source=source).retrieve(**request, target=str(output_grib))
        effective_provider = PROVIDER_LEGACY
    return output_grib, {
        "retrieval_requested_provider": requested_provider,
        "retrieval_effective_provider": effective_provider,
        "retrieval_fallback_reason": fallback_reason,
        "retrieval_duration_seconds": round(time.perf_counter() - started, 6),
        "retrieval_source": source,
        "retrieval_step_count": len(request["step"]),
        "retrieval_cycle_hour": "latest" if cycle_hour is None else int(cycle_hour),
    }


def _rename_ssrd(ds: xr.Dataset) -> xr.Dataset:
    for name in ("surface_solar_radiation_downwards", "surface_solar_radiation_downward"):
        if "ssrd" not in ds and name in ds:
            return ds.rename({name: "ssrd"})
    if "ssrd" not in ds:
        raise KeyError("ECMWF solar file does not contain ssrd/surface_solar_radiation_downwards")
    return ds


def _select_site(ds: xr.Dataset, latitude: float, longitude: float) -> tuple[xr.Dataset, float, float]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
    lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
    if lat_name is None or lon_name is None:
        return ds, float("nan"), float("nan")

    lat_values = np.asarray(ds[lat_name].values, dtype=np.float64)
    lon_values = np.asarray(ds[lon_name].values, dtype=np.float64)
    if lat_values.ndim == 0 or lon_values.ndim == 0:
        return ds, float(lat_values.reshape(-1)[0]), float(lon_values.reshape(-1)[0])
    select_lon = float(longitude)
    if np.nanmin(lon_values) >= 0.0 and select_lon < 0.0:
        select_lon %= 360.0
    lat_index = int(np.nanargmin(np.abs(lat_values - float(latitude))))
    lon_index = int(np.nanargmin(np.abs(lon_values - select_lon)))
    selected = ds.isel({lat_name: lat_index, lon_name: lon_index})
    return selected, float(lat_values[lat_index]), float(lon_values[lon_index])


def _grid_distance_km(latitude: float, longitude: float, grid_latitude: float, grid_longitude: float) -> float:
    if not all(np.isfinite(value) for value in (latitude, longitude, grid_latitude, grid_longitude)):
        return float("nan")
    lat1, lat2 = np.radians([latitude, grid_latitude])
    delta_lat = lat2 - lat1
    delta_lon = np.radians(((grid_longitude - longitude + 180.0) % 360.0) - 180.0)
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    return float(6371.0 * 2.0 * np.arcsin(np.sqrt(a)))


def _normalise_site_dataset(
    ds: xr.Dataset,
    *,
    provider: str,
    input_path: Path,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    ds = _rename_ssrd(ds)
    ds, selected_latitude, selected_longitude = _select_site(ds, latitude, longitude)
    ds = ds.squeeze(drop=True)

    if "forecast_reference_time" in ds.coords:
        cycle_values = np.asarray(ds["forecast_reference_time"].values).reshape(-1)
    elif "time" in ds.coords:
        cycle_values = np.asarray(ds["time"].values).reshape(-1)
    else:
        cycle_values = np.array([], dtype="datetime64[ns]")
    cycle = pd.Timestamp(cycle_values[0]) if cycle_values.size else None
    if cycle is not None and cycle.tz is not None:
        cycle = cycle.tz_convert("UTC").tz_localize(None)

    if "lead_time" in ds.coords:
        lead = ds["lead_time"]
    elif "step" in ds.coords:
        lead = ds["step"]
    else:
        lead = None
    if "valid_time" in ds.coords:
        valid = ds["valid_time"]
    elif lead is not None and cycle is not None:
        valid = xr.DataArray(cycle.to_datetime64() + np.asarray(lead.values), dims=lead.dims)
    elif "time" in ds.coords:
        valid = ds["time"]
    else:
        raise KeyError("ECMWF solar forecast needs valid_time, step, lead_time, or time coordinates")

    lead_dim = valid.dims[0] if valid.dims else ds["ssrd"].dims[-1]
    if lead is None:
        if cycle is None:
            lead_values = pd.DatetimeIndex(np.asarray(valid.values).reshape(-1)) - pd.Timestamp(valid.values[0])
        else:
            lead_values = pd.DatetimeIndex(np.asarray(valid.values).reshape(-1)) - cycle
        lead = xr.DataArray(np.asarray(lead_values), dims=(lead_dim,))

    if lead_dim != "lead_time":
        ds = ds.rename({lead_dim: "lead_time"})
    valid_values = np.asarray(valid.values).reshape(-1)
    lead_values = np.asarray(lead.values).reshape(-1)
    ds = ds.assign_coords(
        lead_time=("lead_time", lead_values),
        valid_time=("lead_time", valid_values),
    )
    if cycle is not None:
        ds = ds.assign_coords(forecast_reference_time=cycle.to_datetime64())
    ds.attrs.update(
        {
            "ecmwf_provider": provider,
            "ecmwf_input_file": str(input_path),
            "requested_site_latitude": str(float(latitude)),
            "requested_site_longitude": str(float(longitude)),
            "selected_grid_latitude": str(selected_latitude),
            "selected_grid_longitude": str(selected_longitude),
            "selected_grid_distance_km": str(
                _grid_distance_km(latitude, longitude, selected_latitude, selected_longitude)
            ),
        }
    )
    return ds


def _open_legacy(path: Path) -> xr.Dataset:
    if path.suffix.lower() in {".grib", ".grb", ".grib2", ".grb2"}:
        return xr.open_dataset(path, engine="cfgrib")
    return xr.open_dataset(path)


def _open_earthkit(path: Path) -> xr.Dataset:
    import earthkit.data as ekd

    data = ekd.from_source("file", str(path))
    if path.suffix.lower() in {".grib", ".grb", ".grib2", ".grb2"}:
        fields = data.to_fieldlist()
        ds = fields.to_xarray(add_earthkit_attrs=False)
        if len(fields):
            cycle = fields[0].metadata("base_datetime")
            if cycle is not None:
                ds = ds.assign_coords(forecast_reference_time=pd.Timestamp(cycle).to_datetime64())
        return ds
    return data.to_xarray()


def _open_timed(
    opener: Callable[[Path], xr.Dataset],
    path: Path,
    *,
    provider: str,
    latitude: float,
    longitude: float,
) -> tuple[xr.Dataset, float]:
    started = time.perf_counter()
    raw = opener(path)
    try:
        normalised = _normalise_site_dataset(
            raw,
            provider=provider,
            input_path=path,
            latitude=latitude,
            longitude=longitude,
        ).load()
    finally:
        raw.close()
    return normalised, time.perf_counter() - started


def _shadow_comparison(legacy: xr.Dataset, earthkit: xr.Dataset) -> dict[str, Any]:
    legacy_ssrd, earthkit_ssrd = xr.align(legacy["ssrd"], earthkit["ssrd"], join="inner")
    differences = np.abs(
        np.asarray(legacy_ssrd.values, dtype=np.float64) - np.asarray(earthkit_ssrd.values, dtype=np.float64)
    )
    finite = differences[np.isfinite(differences)]
    return {
        "common_sample_count": int(finite.size),
        "legacy_sample_count": int(legacy["ssrd"].size),
        "earthkit_sample_count": int(earthkit["ssrd"].size),
        "ssrd_max_abs_difference_j_m2": float(np.max(finite)) if finite.size else None,
        "ssrd_mean_abs_difference_j_m2": float(np.mean(finite)) if finite.size else None,
        "valid_times_match": bool(np.array_equal(legacy["valid_time"].values, earthkit["valid_time"].values)),
    }


def open_solar_forecast(
    path: Path,
    *,
    provider: str,
    latitude: float,
    longitude: float,
    shadow_report_path: Path | None = DEFAULT_SHADOW_REPORT_PATH,
) -> ForecastProviderResult:
    """Open and normalize a solar forecast, optionally comparing Earthkit in shadow mode."""
    requested_provider = validate_provider(provider)
    diagnostics: dict[str, Any] = {
        "requested_provider": requested_provider,
        "input_file": str(path),
        "opened_at_utc": _utc_now(),
        "fallback_reason": "",
        "peak_rss_mb": round(_peak_rss_mb(), 3),
        "earthkit_data_version": _package_version("earthkit-data")
        if requested_provider in {PROVIDER_EARTHKIT, PROVIDER_SHADOW}
        else "",
    }
    if requested_provider == PROVIDER_LEGACY:
        dataset, duration = _open_timed(
            _open_legacy, path, provider=PROVIDER_LEGACY, latitude=latitude, longitude=longitude
        )
        diagnostics.update(effective_provider=PROVIDER_LEGACY, legacy_open_seconds=round(duration, 6))
        diagnostics["peak_rss_mb"] = round(_peak_rss_mb(), 3)
        return ForecastProviderResult(dataset, diagnostics)

    if requested_provider == PROVIDER_EARTHKIT:
        try:
            dataset, duration = _open_timed(
                _open_earthkit, path, provider=PROVIDER_EARTHKIT, latitude=latitude, longitude=longitude
            )
            diagnostics.update(effective_provider=PROVIDER_EARTHKIT, earthkit_open_seconds=round(duration, 6))
            diagnostics["peak_rss_mb"] = round(_peak_rss_mb(), 3)
            return ForecastProviderResult(dataset, diagnostics)
        except Exception as exc:
            diagnostics["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            dataset, duration = _open_timed(
                _open_legacy, path, provider=PROVIDER_LEGACY, latitude=latitude, longitude=longitude
            )
            diagnostics.update(effective_provider=PROVIDER_LEGACY, legacy_open_seconds=round(duration, 6))
            diagnostics["peak_rss_mb"] = round(_peak_rss_mb(), 3)
            return ForecastProviderResult(dataset, diagnostics)

    legacy, legacy_duration = _open_timed(
        _open_legacy, path, provider=PROVIDER_LEGACY, latitude=latitude, longitude=longitude
    )
    diagnostics.update(effective_provider=PROVIDER_LEGACY, legacy_open_seconds=round(legacy_duration, 6))
    try:
        earthkit, earthkit_duration = _open_timed(
            _open_earthkit, path, provider=PROVIDER_EARTHKIT, latitude=latitude, longitude=longitude
        )
        try:
            diagnostics.update(_shadow_comparison(legacy, earthkit))
            diagnostics["earthkit_open_seconds"] = round(earthkit_duration, 6)
            diagnostics["shadow_status"] = "compared"
        finally:
            earthkit.close()
    except Exception as exc:
        diagnostics["shadow_status"] = "earthkit_failed"
        diagnostics["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    if shadow_report_path is not None:
        diagnostics["peak_rss_mb"] = round(_peak_rss_mb(), 3)
        _atomic_write_json(shadow_report_path, diagnostics)
    return ForecastProviderResult(legacy, diagnostics)


def _latest_cached_forecast(cache_dir: Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("*ssrd*.grib2", "*ssrd*.grib", "*.grib2", "*.grib"):
        candidates.extend(path for path in cache_dir.glob(pattern) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No cached ECMWF GRIB files found in {cache_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Earthkit and legacy ECMWF decoders")
    parser.add_argument("--input-forecast", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/aurora/products/power/ecmwf_solar_forecast"))
    parser.add_argument("--report", type=Path, default=DEFAULT_SHADOW_REPORT_PATH)
    parser.add_argument("--latitude", type=float, default=64.829694)
    parser.add_argument("--longitude", type=float, default=-23.248139)
    args = parser.parse_args()
    input_forecast = args.input_forecast or _latest_cached_forecast(args.cache_dir)
    result = open_solar_forecast(
        input_forecast,
        provider=PROVIDER_SHADOW,
        latitude=args.latitude,
        longitude=args.longitude,
        shadow_report_path=args.report,
    )
    result.dataset.close()
    print(f"Wrote {args.report}: {result.diagnostics.get('shadow_status', 'unknown')}")


if __name__ == "__main__":
    main()
