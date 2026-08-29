#!/usr/bin/env python3
"""Physically interpretable plane-of-array PV model for the APS forecast.

The operational baseline converts ECMWF surface solar radiation to electrical
power with one fitted multiplier.  This module keeps the ECMWF field as global
horizontal irradiance (GHI), resolves solar geometry, separates direct and
diffuse radiation, transposes it to each APS array, and applies a bounded
PVWatts-style DC model.  It deliberately has no dependency on pvlib so the
candidate can run in the resource-constrained forecast service.

All times are interpreted as UTC.  Irradiance values are interval means ending
at their timestamp.  Each interval is internally disaggregated to short
substeps using the extraterrestrial-horizontal solar shape; the substep GHI is
renormalised to conserve the original ECMWF interval energy exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SOLAR_MODEL_NAME = "three_array_poa_pv_v2"
SOLAR_MODEL_VERSION = 2
SOLAR_FEATURE_SET_VERSION = "ecmwf_ssrd_geometry_optional_fdir_t2m_wind_substep_v2"
SOLAR_CONSTANT_W_M2 = 1361.0
MIN_COS_ZENITH = 0.065
MAX_DNI_ZENITH_DEG = 87.0


@dataclass(frozen=True)
class PVArrayConfig:
    """Fixed physical/electrical configuration for one APS PV array."""

    name: str
    azimuth_deg: float
    tilt_deg: float
    nameplate_power_w: float
    controller_limit_w: float
    fixed_efficiency: float = 0.90
    temperature_coefficient_per_c: float = -0.0037
    incidence_angle_modifier_b0: float = 0.05

    def validated(self) -> "PVArrayConfig":
        name = str(self.name).strip()
        if not name or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid PV array name {self.name!r}")
        if not 0.0 <= float(self.azimuth_deg) < 360.0:
            raise ValueError(f"{name} azimuth must be in [0, 360) degrees")
        if not 0.0 <= float(self.tilt_deg) <= 90.0:
            raise ValueError(f"{name} tilt must be in [0, 90] degrees")
        if float(self.nameplate_power_w) <= 0.0:
            raise ValueError(f"{name} nameplate power must be positive")
        if float(self.controller_limit_w) <= 0.0:
            raise ValueError(f"{name} controller limit must be positive")
        if not 0.0 < float(self.fixed_efficiency) <= 1.0:
            raise ValueError(f"{name} fixed efficiency must be in (0, 1]")
        if not -0.02 <= float(self.temperature_coefficient_per_c) <= 0.0:
            raise ValueError(f"{name} temperature coefficient is implausible")
        if not 0.0 <= float(self.incidence_angle_modifier_b0) <= 0.25:
            raise ValueError(f"{name} incidence-angle modifier is implausible")
        return PVArrayConfig(
            name=name,
            azimuth_deg=float(self.azimuth_deg),
            tilt_deg=float(self.tilt_deg),
            nameplate_power_w=float(self.nameplate_power_w),
            controller_limit_w=float(self.controller_limit_w),
            fixed_efficiency=float(self.fixed_efficiency),
            temperature_coefficient_per_c=float(self.temperature_coefficient_per_c),
            incidence_angle_modifier_b0=float(self.incidence_angle_modifier_b0),
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "azimuth_deg": self.azimuth_deg,
            "tilt_deg": self.tilt_deg,
            "nameplate_power_w": self.nameplate_power_w,
            "controller_limit_w": self.controller_limit_w,
            "fixed_efficiency": self.fixed_efficiency,
            "temperature_coefficient_per_c": self.temperature_coefficient_per_c,
            "incidence_angle_modifier_b0": self.incidence_angle_modifier_b0,
        }


@dataclass(frozen=True)
class PhysicalSolarConfig:
    """Versioned configuration for the three-array physical candidate."""

    schema_version: int
    configuration_status: str
    source: str
    arrays: tuple[PVArrayConfig, ...]
    ground_albedo: float = 0.20
    faiman_u0_w_m2_k: float = 25.0
    faiman_u1_w_s_m3_k: float = 6.84
    substep_minutes: float = 10.0

    def validated(self) -> "PhysicalSolarConfig":
        if int(self.schema_version) != 1:
            raise ValueError(f"Unsupported physical solar config schema {self.schema_version!r}")
        arrays = tuple(array.validated() for array in self.arrays)
        if not arrays:
            raise ValueError("Physical solar config must contain at least one array")
        names = [array.name for array in arrays]
        if len(names) != len(set(names)):
            raise ValueError("Physical solar array names must be unique")
        if not 0.0 <= float(self.ground_albedo) <= 1.0:
            raise ValueError("Ground albedo must be in [0, 1]")
        if float(self.faiman_u0_w_m2_k) <= 0.0 or float(self.faiman_u1_w_s_m3_k) < 0.0:
            raise ValueError("Faiman heat-loss coefficients must be non-negative")
        if not 1.0 <= float(self.substep_minutes) <= 60.0:
            raise ValueError("Solar substep must be between 1 and 60 minutes")
        status = str(self.configuration_status).strip().lower()
        if status not in {"surveyed", "provisional", "fitted"}:
            raise ValueError("configuration_status must be surveyed, provisional, or fitted")
        return PhysicalSolarConfig(
            schema_version=1,
            configuration_status=status,
            source=str(self.source).strip(),
            arrays=arrays,
            ground_albedo=float(self.ground_albedo),
            faiman_u0_w_m2_k=float(self.faiman_u0_w_m2_k),
            faiman_u1_w_s_m3_k=float(self.faiman_u1_w_s_m3_k),
            substep_minutes=float(self.substep_minutes),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "configuration_status": self.configuration_status,
            "source": self.source,
            "ground_albedo": self.ground_albedo,
            "faiman_u0_w_m2_k": self.faiman_u0_w_m2_k,
            "faiman_u1_w_s_m3_k": self.faiman_u1_w_s_m3_k,
            "substep_minutes": self.substep_minutes,
            "arrays": [array.to_dict() for array in self.arrays],
        }


def _number(mapping: Mapping[str, object], name: str, default: float | None = None) -> float:
    value = mapping.get(name, default)
    if value is None:
        raise ValueError(f"Physical solar config is missing {name}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Physical solar config field {name} is not numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Physical solar config field {name} is not finite")
    return parsed


def physical_solar_config_from_mapping(payload: Mapping[str, object]) -> PhysicalSolarConfig:
    raw_arrays = payload.get("arrays")
    if not isinstance(raw_arrays, Sequence) or isinstance(raw_arrays, (str, bytes)):
        raise ValueError("Physical solar config arrays must be a list")
    arrays: list[PVArrayConfig] = []
    for raw in raw_arrays:
        if not isinstance(raw, Mapping):
            raise ValueError("Each physical solar array must be an object")
        arrays.append(
            PVArrayConfig(
                name=str(raw.get("name", "")),
                azimuth_deg=_number(raw, "azimuth_deg"),
                tilt_deg=_number(raw, "tilt_deg"),
                nameplate_power_w=_number(raw, "nameplate_power_w"),
                controller_limit_w=_number(raw, "controller_limit_w"),
                fixed_efficiency=_number(raw, "fixed_efficiency", 0.90),
                temperature_coefficient_per_c=_number(raw, "temperature_coefficient_per_c", -0.0037),
                incidence_angle_modifier_b0=_number(raw, "incidence_angle_modifier_b0", 0.05),
            )
        )
    return PhysicalSolarConfig(
        schema_version=int(payload.get("schema_version", 1)),
        configuration_status=str(payload.get("configuration_status", "")),
        source=str(payload.get("source", "")),
        arrays=tuple(arrays),
        ground_albedo=_number(payload, "ground_albedo", 0.20),
        faiman_u0_w_m2_k=_number(payload, "faiman_u0_w_m2_k", 25.0),
        faiman_u1_w_s_m3_k=_number(payload, "faiman_u1_w_s_m3_k", 6.84),
        substep_minutes=_number(payload, "substep_minutes", 10.0),
    ).validated()


def load_physical_solar_config(path: Path) -> PhysicalSolarConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read physical solar config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Physical solar config root must be an object")
    return physical_solar_config_from_mapping(payload)


def physical_solar_config_digest(config: PhysicalSolarConfig) -> str:
    payload = json.dumps(config.validated().to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def physical_solar_contract_id(
    config: PhysicalSolarConfig,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> str:
    payload = {
        "schema": 2,
        "solar_model_name": SOLAR_MODEL_NAME,
        "solar_model_version": SOLAR_MODEL_VERSION,
        "solar_feature_set_version": SOLAR_FEATURE_SET_VERSION,
        "physical_config_sha256": physical_solar_config_digest(config),
        "latitude": None if latitude is None else round(float(latitude), 8),
        "longitude": None if longitude is None else round(float(longitude), 8),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"solar-physical-v{SOLAR_MODEL_VERSION}-{digest}"


def _utc_naive_index(values: pd.DatetimeIndex | Sequence[object]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if index.hasnans:
        raise ValueError("Physical solar forecast times contain NaT")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Physical solar forecast times must be unique and increasing")
    return index


def solar_position(
    times: pd.DatetimeIndex | Sequence[object],
    *,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Return NOAA-style true solar geometry for UTC timestamps.

    Azimuth is degrees clockwise from true north.  Refraction is intentionally
    omitted because irradiance transposition needs the geometric solar vector.
    """

    index = _utc_naive_index(times)
    if not -90.0 <= float(latitude) <= 90.0:
        raise ValueError("Latitude must be in [-90, 90]")
    if not -180.0 <= float(longitude) <= 180.0:
        raise ValueError("Longitude must be in [-180, 180]")
    day = index.dayofyear.to_numpy(dtype=np.float64)
    days_in_year = np.where(index.is_leap_year, 366.0, 365.0)
    hour = (
        index.hour.to_numpy(dtype=np.float64)
        + index.minute.to_numpy(dtype=np.float64) / 60.0
        + index.second.to_numpy(dtype=np.float64) / 3600.0
        + index.microsecond.to_numpy(dtype=np.float64) / 3.6e9
    )
    gamma = 2.0 * np.pi / days_in_year * (day - 1.0 + (hour - 12.0) / 24.0)
    equation_minutes = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    true_solar_minutes = (hour * 60.0 + equation_minutes + 4.0 * float(longitude)) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = math.radians(float(latitude))
    cos_zenith = (
        math.sin(latitude_rad) * np.sin(declination)
        + math.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.rad2deg(np.arccos(cos_zenith))
    azimuth = (
        np.rad2deg(
            np.arctan2(
                np.sin(hour_angle),
                np.cos(hour_angle) * math.sin(latitude_rad)
                - np.tan(declination) * math.cos(latitude_rad),
            )
        )
        + 180.0
    ) % 360.0
    extraterrestrial_normal = SOLAR_CONSTANT_W_M2 * (
        1.0 + 0.033 * np.cos(2.0 * np.pi * day / days_in_year)
    )
    return pd.DataFrame(
        {
            "SolarZenithDegrees": zenith,
            "SolarAzimuthDegrees": azimuth,
            "SolarCosineZenith": np.maximum(cos_zenith, 0.0),
            "SolarExtraterrestrialNormalIrradiance": extraterrestrial_normal,
            "SolarExtraterrestrialHorizontalIrradiance": extraterrestrial_normal
            * np.maximum(cos_zenith, 0.0),
        },
        index=index,
    )


def _erbs_direct_diffuse(
    ghi_w_m2: np.ndarray,
    geometry: pd.DataFrame,
    direct_horizontal_w_m2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded DNI, DHI and clearness index from interval GHI."""

    ghi = np.clip(np.asarray(ghi_w_m2, dtype=np.float64), 0.0, None)
    cos_zenith = geometry["SolarCosineZenith"].to_numpy(dtype=np.float64)
    zenith = geometry["SolarZenithDegrees"].to_numpy(dtype=np.float64)
    extra_normal = geometry["SolarExtraterrestrialNormalIrradiance"].to_numpy(dtype=np.float64)
    denominator = extra_normal * np.maximum(cos_zenith, MIN_COS_ZENITH)
    kt = np.divide(ghi, denominator, out=np.zeros_like(ghi), where=denominator > 0.0)
    kt = np.clip(kt, 0.0, 1.2)

    if direct_horizontal_w_m2 is not None:
        # A supplied direct-horizontal forecast remains informative at the very
        # low winter Sun experienced by Aurora.  Divide by the actual cosine
        # and cap to extraterrestrial DNI; the 87-degree Erbs validity limit
        # below must not erase this physically important beam component.
        usable_direct = (cos_zenith > 0.0) & (ghi > 0.0)
        beam_horizontal = np.clip(np.asarray(direct_horizontal_w_m2, dtype=np.float64), 0.0, ghi)
        dni = np.divide(
            beam_horizontal,
            np.maximum(cos_zenith, 1.0e-6),
            out=np.zeros_like(ghi),
            where=usable_direct,
        )
    else:
        usable_direct = (
            (zenith <= MAX_DNI_ZENITH_DEG)
            & (cos_zenith > 0.0)
            & (ghi > 0.0)
        )
        diffuse_fraction = np.where(
            kt <= 0.22,
            1.0 - 0.09 * kt,
            np.where(
                kt <= 0.80,
                0.9511
                - 0.1604 * kt
                + 4.388 * kt**2
                - 16.638 * kt**3
                + 12.336 * kt**4,
                0.165,
            ),
        )
        diffuse_fraction = np.clip(diffuse_fraction, 0.0, 1.0)
        dhi_initial = diffuse_fraction * ghi
        dni = np.divide(
            ghi - dhi_initial,
            np.maximum(cos_zenith, MIN_COS_ZENITH),
            out=np.zeros_like(ghi),
            where=usable_direct,
        )
    dni = np.where(usable_direct, np.clip(dni, 0.0, extra_normal), 0.0)
    # Recompute diffuse after the DNI cap so GHI = DNI*cos(zenith) + DHI.
    dhi = np.clip(ghi - dni * cos_zenith, 0.0, ghi)
    return dni, dhi, kt


def _incidence_cosine(geometry: pd.DataFrame, array: PVArrayConfig) -> np.ndarray:
    zenith = np.deg2rad(geometry["SolarZenithDegrees"].to_numpy(dtype=np.float64))
    solar_azimuth = np.deg2rad(geometry["SolarAzimuthDegrees"].to_numpy(dtype=np.float64))
    tilt = math.radians(array.tilt_deg)
    surface_azimuth = math.radians(array.azimuth_deg)
    return np.clip(
        np.cos(zenith) * math.cos(tilt)
        + np.sin(zenith) * math.sin(tilt) * np.cos(solar_azimuth - surface_azimuth),
        -1.0,
        1.0,
    )


def _array_power_at_substeps(
    ghi: np.ndarray,
    geometry: pd.DataFrame,
    array: PVArrayConfig,
    *,
    dni: np.ndarray,
    dhi: np.ndarray,
    albedo: np.ndarray,
    air_temperature_c: np.ndarray | None,
    wind_speed_m_s: np.ndarray | None,
    config: PhysicalSolarConfig,
) -> dict[str, np.ndarray]:
    cos_incidence = _incidence_cosine(geometry, array)
    beam_cosine = np.maximum(cos_incidence, 0.0)
    iam = np.zeros_like(beam_cosine)
    illuminated = beam_cosine > 1.0e-6
    iam[illuminated] = np.clip(
        1.0 - array.incidence_angle_modifier_b0 * (1.0 / beam_cosine[illuminated] - 1.0),
        0.0,
        1.0,
    )
    tilt = math.radians(array.tilt_deg)
    poa_direct = dni * beam_cosine
    effective_poa_direct = poa_direct * iam
    poa_sky = dhi * (1.0 + math.cos(tilt)) / 2.0
    poa_ground = ghi * albedo * (1.0 - math.cos(tilt)) / 2.0
    poa_diffuse = poa_sky + poa_ground
    poa = np.clip(poa_direct + poa_diffuse, 0.0, None)
    effective_poa = np.clip(effective_poa_direct + poa_diffuse, 0.0, None)

    if air_temperature_c is None or wind_speed_m_s is None:
        cell_temperature = np.full_like(poa, np.nan)
        temperature_factor = np.ones_like(poa)
    else:
        denominator = config.faiman_u0_w_m2_k + config.faiman_u1_w_s_m3_k * np.maximum(
            wind_speed_m_s, 0.0
        )
        cell_temperature = air_temperature_c + np.divide(
            poa,
            np.maximum(denominator, 1.0),
            out=np.zeros_like(poa),
        )
        temperature_factor = np.clip(
            1.0 + array.temperature_coefficient_per_c * (cell_temperature - 25.0),
            0.0,
            1.25,
        )
    unconstrained = (
        array.nameplate_power_w
        * effective_poa
        / 1000.0
        * temperature_factor
        * array.fixed_efficiency
    )
    available = np.clip(unconstrained, 0.0, array.controller_limit_w)
    return {
        "poa": poa,
        "poa_direct": poa_direct,
        "poa_diffuse": poa_diffuse,
        "effective_poa": effective_poa,
        "effective_poa_direct": effective_poa_direct,
        "cell_temperature": cell_temperature,
        "available_power": available,
        "controller_clipping": np.clip(unconstrained - available, 0.0, None),
    }


def _aligned_optional_series(
    values: pd.Series | None,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray | None, bool]:
    if values is None or values.empty:
        return None, False
    source = values.copy()
    source.index = _utc_naive_index(source.index)
    source = source[~source.index.duplicated(keep="last")].sort_index()
    aligned = source.reindex(index, method="nearest", tolerance=pd.Timedelta(hours=3))
    array = aligned.to_numpy(dtype=np.float64)
    if not np.any(np.isfinite(array)):
        return None, False
    return array, bool(np.all(np.isfinite(array)))


def _interval_durations(index: pd.DatetimeIndex) -> pd.Series:
    differences = index.to_series().diff()
    finite = differences[differences > pd.Timedelta(0)]
    fallback = finite.iloc[0] if not finite.empty else pd.Timedelta(hours=1)
    differences.iloc[0] = fallback
    lower = pd.Timedelta(minutes=1)
    upper = pd.Timedelta(hours=12)
    return differences.clip(lower=lower, upper=upper)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(array) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(array[valid], weights=weight[valid]))


def build_physical_solar_forecast_frames(
    ghi_w_m2: pd.Series,
    *,
    latitude: float,
    longitude: float,
    config: PhysicalSolarConfig,
    direct_horizontal_w_m2: pd.Series | None = None,
    air_temperature_c: pd.Series | None = None,
    wind_speed_m_s: pd.Series | None = None,
    ground_albedo: pd.Series | None = None,
    forecast_start_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return ECMWF-interval and PV-substep physical solar forecasts."""

    config = config.validated()
    if ghi_w_m2.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    index = _utc_naive_index(ghi_w_m2.index)
    ghi = pd.Series(np.asarray(ghi_w_m2.values, dtype=np.float64), index=index)
    if not np.all(np.isfinite(ghi.values)):
        raise ValueError("Physical solar model requires finite ECMWF GHI")
    ghi = ghi.clip(lower=0.0)
    durations = _interval_durations(index)
    cutoff = None
    if forecast_start_time is not None:
        cutoff = pd.Timestamp(forecast_start_time)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)

    direct_interval, direct_complete = _aligned_optional_series(direct_horizontal_w_m2, index)
    air_interval, air_complete = _aligned_optional_series(air_temperature_c, index)
    wind_interval, wind_complete = _aligned_optional_series(wind_speed_m_s, index)
    albedo_interval, albedo_complete = _aligned_optional_series(ground_albedo, index)

    degradation: set[str] = set()
    if direct_interval is None:
        degradation.add("direct_diffuse_erbs")
    elif not direct_complete:
        degradation.add("direct_horizontal_partial")
    if air_interval is None or wind_interval is None:
        degradation.add("temperature_correction_disabled")
    elif not air_complete or not wind_complete:
        degradation.add("temperature_meteorology_partial")
    if albedo_interval is None:
        degradation.add("fixed_ground_albedo")
    elif not albedo_complete:
        degradation.add("ground_albedo_partial")
    if config.configuration_status != "surveyed":
        degradation.add(f"{config.configuration_status}_physical_configuration")

    field_names = [
        "SolarZenithDegrees",
        "SolarAzimuthDegrees",
        "SolarCosineZenith",
        "SolarExtraterrestrialNormalIrradiance",
        "SolarExtraterrestrialHorizontalIrradiance",
        "ECMWFDirectNormalIrradiance",
        "ECMWFDirectHorizontalIrradiance",
        "ECMWFDiffuseHorizontalIrradiance",
        "ECMWFClearnessIndex",
        "SolarForcingConsistencyFlag",
    ]
    values: dict[str, np.ndarray] = {
        name: np.full(len(index), np.nan, dtype=np.float64) for name in field_names
    }
    values["ECMWFSolarIrradiance"] = ghi.to_numpy(dtype=np.float64)
    values["ECMWFSourceIntervalHours"] = (
        durations / pd.Timedelta(hours=1)
    ).to_numpy(dtype=np.float64)
    values["ForecastEffectiveGlobalHorizontalIrradiance"] = np.full(
        len(index), np.nan, dtype=np.float64
    )
    values["SolarIntervalHours"] = np.zeros(len(index), dtype=np.float64)
    for array in config.arrays:
        for prefix in (
            "ForecastPlaneOfArrayIrradiance",
            "ForecastPlaneOfArrayDirectIrradiance",
            "ForecastPlaneOfArrayDiffuseIrradiance",
            "ForecastEffectivePlaneOfArrayIrradiance",
            "ForecastEffectivePlaneOfArrayDirectIrradiance",
            "ForecastPVCellTemperature",
            "ForecastPVAvailableWatts",
            "ForecastPVControllerClippingWatts",
        ):
            values[f"{prefix}{array.name}"] = np.full(len(index), np.nan, dtype=np.float64)

    total_available = np.full(len(index), np.nan, dtype=np.float64)
    total_clipping = np.full(len(index), np.nan, dtype=np.float64)
    substep_rows: list[dict[str, float]] = []
    substep_end_times: list[pd.Timestamp] = []
    for interval_index, end in enumerate(index):
        duration = pd.Timedelta(durations.iloc[interval_index])
        start = pd.Timestamp(end) - duration
        substeps = max(1, int(math.ceil(duration / pd.Timedelta(minutes=config.substep_minutes))))
        boundaries = [start + duration * (idx / float(substeps)) for idx in range(substeps + 1)]
        if cutoff is not None and start < cutoff < pd.Timestamp(end):
            boundaries.append(cutoff)
        boundaries = sorted(set(pd.Timestamp(value) for value in boundaries))
        segment_starts = pd.DatetimeIndex(boundaries[:-1])
        segment_ends = pd.DatetimeIndex(boundaries[1:])
        segment_seconds = np.asarray(
            [(right - left) / pd.Timedelta(seconds=1) for left, right in zip(segment_starts, segment_ends)],
            dtype=np.float64,
        )
        sub_times = pd.DatetimeIndex(
            [left + (right - left) / 2.0 for left, right in zip(segment_starts, segment_ends)]
        )
        geometry = solar_position(sub_times, latitude=latitude, longitude=longitude)
        solar_shape = geometry["SolarExtraterrestrialHorizontalIrradiance"].to_numpy(dtype=np.float64)
        shape_mean = _weighted_mean(solar_shape, segment_seconds)
        target_ghi = float(ghi.iloc[interval_index])
        consistency_flag = 0
        if np.isfinite(shape_mean) and shape_mean > 0.0:
            sub_ghi = target_ghi * solar_shape / shape_mean
        elif target_ghi <= 1.0e-6:
            sub_ghi = np.zeros(len(sub_times), dtype=np.float64)
        else:
            # Preserve input energy but make the forcing inconsistency explicit.
            sub_ghi = np.full(len(sub_times), target_ghi, dtype=np.float64)
            degradation.add("positive_ghi_below_geometric_horizon")
            consistency_flag |= 1

        sub_direct = None
        if direct_interval is not None and np.isfinite(direct_interval[interval_index]):
            if float(direct_interval[interval_index]) > target_ghi + 1.0e-6:
                degradation.add("direct_horizontal_exceeds_ghi_clipped")
                consistency_flag |= 2
            target_direct = float(np.clip(direct_interval[interval_index], 0.0, target_ghi))
            if np.isfinite(shape_mean) and shape_mean > 0.0:
                sub_direct = target_direct * solar_shape / shape_mean
            else:
                sub_direct = np.zeros(len(sub_times), dtype=np.float64)
            sub_direct = np.minimum(np.clip(sub_direct, 0.0, None), sub_ghi)
            if np.any(
                (geometry["SolarZenithDegrees"].to_numpy(dtype=np.float64) > MAX_DNI_ZENITH_DEG)
                & (sub_direct > 0.0)
            ):
                degradation.add("supplied_direct_low_sun_dni_bounded")
        dni, dhi, kt = _erbs_direct_diffuse(sub_ghi, geometry, sub_direct)
        beam_horizontal = dni * geometry["SolarCosineZenith"].to_numpy(dtype=np.float64)
        if sub_direct is None and np.any(
            (geometry["SolarZenithDegrees"].to_numpy(dtype=np.float64) > MAX_DNI_ZENITH_DEG)
            & (geometry["SolarCosineZenith"].to_numpy(dtype=np.float64) > 0.0)
            & (sub_ghi > 0.0)
        ):
            degradation.add("erbs_low_sun_out_of_domain")

        if albedo_interval is not None and np.isfinite(albedo_interval[interval_index]):
            sub_albedo = np.full(len(sub_times), np.clip(albedo_interval[interval_index], 0.0, 1.0))
        else:
            sub_albedo = np.full(len(sub_times), config.ground_albedo)
        sub_air = None
        sub_wind = None
        if (
            air_interval is not None
            and wind_interval is not None
            and np.isfinite(air_interval[interval_index])
            and np.isfinite(wind_interval[interval_index])
        ):
            sub_air = np.full(len(sub_times), air_interval[interval_index], dtype=np.float64)
            sub_wind = np.full(len(sub_times), max(wind_interval[interval_index], 0.0), dtype=np.float64)

        remaining = np.ones(len(sub_times), dtype=bool)
        if cutoff is not None:
            remaining = segment_ends > cutoff
        has_remaining = bool(np.any(remaining))
        values["SolarIntervalHours"][interval_index] = (
            float(np.sum(segment_seconds[remaining]) / 3600.0) if has_remaining else 0.0
        )
        values["SolarForcingConsistencyFlag"][interval_index] = float(consistency_flag)
        if not has_remaining:
            # This row is the SOC anchor itself.  The preceding ECMWF source
            # interval remains available through the separately named raw
            # fields, but it has no future forcing duration and must not expose
            # previous-interval POA or PV as a candidate forecast.
            anchor_geometry = solar_position(
                pd.DatetimeIndex([pd.Timestamp(end)]),
                latitude=latitude,
                longitude=longitude,
            ).iloc[0]
            for name in (
                "SolarZenithDegrees",
                "SolarAzimuthDegrees",
                "SolarCosineZenith",
                "SolarExtraterrestrialNormalIrradiance",
                "SolarExtraterrestrialHorizontalIrradiance",
            ):
                values[name][interval_index] = float(anchor_geometry[name])
            continue

        aggregate = remaining
        aggregate_weights = segment_seconds[aggregate]
        values["ForecastEffectiveGlobalHorizontalIrradiance"][interval_index] = _weighted_mean(
            sub_ghi[aggregate], aggregate_weights
        )

        effective_start = segment_starts[np.flatnonzero(aggregate)[0]]
        effective_end = segment_ends[np.flatnonzero(aggregate)[-1]]
        midpoint_geometry = solar_position(
            pd.DatetimeIndex([effective_start + (effective_end - effective_start) / 2.0]),
            latitude=latitude,
            longitude=longitude,
        ).iloc[0]
        for name in (
            "SolarZenithDegrees",
            "SolarAzimuthDegrees",
            "SolarCosineZenith",
            "SolarExtraterrestrialNormalIrradiance",
            "SolarExtraterrestrialHorizontalIrradiance",
        ):
            values[name][interval_index] = float(midpoint_geometry[name])
        values["ECMWFDirectNormalIrradiance"][interval_index] = _weighted_mean(
            dni[aggregate], aggregate_weights
        )
        values["ECMWFDirectHorizontalIrradiance"][interval_index] = _weighted_mean(
            beam_horizontal[aggregate], aggregate_weights
        )
        values["ECMWFDiffuseHorizontalIrradiance"][interval_index] = _weighted_mean(
            dhi[aggregate], aggregate_weights
        )
        values["ECMWFClearnessIndex"][interval_index] = _weighted_mean(
            kt[aggregate], aggregate_weights
        )
        total_available[interval_index] = 0.0
        total_clipping[interval_index] = 0.0

        modeled_arrays: dict[str, dict[str, np.ndarray]] = {}
        for array in config.arrays:
            modeled = _array_power_at_substeps(
                sub_ghi,
                geometry,
                array,
                dni=dni,
                dhi=dhi,
                albedo=sub_albedo,
                air_temperature_c=sub_air,
                wind_speed_m_s=sub_wind,
                config=config,
            )
            modeled_arrays[array.name] = modeled
            values[f"ForecastPlaneOfArrayIrradiance{array.name}"][interval_index] = float(
                _weighted_mean(modeled["poa"][aggregate], aggregate_weights)
            )
            values[f"ForecastPlaneOfArrayDirectIrradiance{array.name}"][interval_index] = float(
                _weighted_mean(modeled["poa_direct"][aggregate], aggregate_weights)
            )
            values[f"ForecastPlaneOfArrayDiffuseIrradiance{array.name}"][interval_index] = float(
                _weighted_mean(modeled["poa_diffuse"][aggregate], aggregate_weights)
            )
            values[f"ForecastEffectivePlaneOfArrayIrradiance{array.name}"][interval_index] = float(
                _weighted_mean(modeled["effective_poa"][aggregate], aggregate_weights)
            )
            values[
                f"ForecastEffectivePlaneOfArrayDirectIrradiance{array.name}"
            ][interval_index] = float(
                _weighted_mean(modeled["effective_poa_direct"][aggregate], aggregate_weights)
            )
            cell = modeled["cell_temperature"]
            values[f"ForecastPVCellTemperature{array.name}"][interval_index] = (
                _weighted_mean(cell[aggregate], aggregate_weights)
            )
            available = _weighted_mean(modeled["available_power"][aggregate], aggregate_weights)
            clipping = _weighted_mean(modeled["controller_clipping"][aggregate], aggregate_weights)
            values[f"ForecastPVAvailableWatts{array.name}"][interval_index] = available
            values[f"ForecastPVControllerClippingWatts{array.name}"][interval_index] = clipping
            total_available[interval_index] += available
            total_clipping[interval_index] += clipping

        for sub_index in np.flatnonzero(remaining):
            row: dict[str, float] = {
                "ECMWFSolarIrradiance": float(sub_ghi[sub_index]),
                "SolarIntervalHours": float(segment_seconds[sub_index] / 3600.0),
                "SolarZenithDegrees": float(geometry["SolarZenithDegrees"].iloc[sub_index]),
                "SolarAzimuthDegrees": float(geometry["SolarAzimuthDegrees"].iloc[sub_index]),
                "SolarCosineZenith": float(geometry["SolarCosineZenith"].iloc[sub_index]),
                "SolarExtraterrestrialNormalIrradiance": float(
                    geometry["SolarExtraterrestrialNormalIrradiance"].iloc[sub_index]
                ),
                "SolarExtraterrestrialHorizontalIrradiance": float(
                    geometry["SolarExtraterrestrialHorizontalIrradiance"].iloc[sub_index]
                ),
                "ECMWFDirectNormalIrradiance": float(dni[sub_index]),
                "ECMWFDirectHorizontalIrradiance": float(beam_horizontal[sub_index]),
                "ECMWFDiffuseHorizontalIrradiance": float(dhi[sub_index]),
                "ECMWFClearnessIndex": float(kt[sub_index]),
                "SolarForcingConsistencyFlag": float(consistency_flag),
            }
            row_total_available = 0.0
            row_total_clipping = 0.0
            for array in config.arrays:
                modeled = modeled_arrays[array.name]
                row[f"ForecastPlaneOfArrayIrradiance{array.name}"] = float(modeled["poa"][sub_index])
                row[f"ForecastPlaneOfArrayDirectIrradiance{array.name}"] = float(
                    modeled["poa_direct"][sub_index]
                )
                row[f"ForecastPlaneOfArrayDiffuseIrradiance{array.name}"] = float(
                    modeled["poa_diffuse"][sub_index]
                )
                row[f"ForecastEffectivePlaneOfArrayIrradiance{array.name}"] = float(
                    modeled["effective_poa"][sub_index]
                )
                row[f"ForecastEffectivePlaneOfArrayDirectIrradiance{array.name}"] = float(
                    modeled["effective_poa_direct"][sub_index]
                )
                row[f"ForecastPVCellTemperature{array.name}"] = float(
                    modeled["cell_temperature"][sub_index]
                )
                row[f"ForecastPVAvailableWatts{array.name}"] = float(
                    modeled["available_power"][sub_index]
                )
                row[f"ForecastPVControllerClippingWatts{array.name}"] = float(
                    modeled["controller_clipping"][sub_index]
                )
                row_total_available += row[f"ForecastPVAvailableWatts{array.name}"]
                row_total_clipping += row[f"ForecastPVControllerClippingWatts{array.name}"]
            row["ForecastPVAvailableWatts"] = max(row_total_available, 0.0)
            row["ForecastSolarWatts"] = row["ForecastPVAvailableWatts"]
            row["ForecastPVControllerClippingWatts"] = max(row_total_clipping, 0.0)
            substep_rows.append(row)
            substep_end_times.append(pd.Timestamp(segment_ends[sub_index]))

    values["ForecastPVAvailableWatts"] = np.clip(total_available, 0.0, None)
    # Compatibility field: this remains the available PV forcing.  Battery/MPPT
    # acceptance and curtailment are added by the SOC energy-balance layer.
    values["ForecastSolarWatts"] = values["ForecastPVAvailableWatts"].copy()
    values["ForecastPVControllerClippingWatts"] = np.clip(total_clipping, 0.0, None)
    frame = pd.DataFrame(values, index=index)
    substep_frame = pd.DataFrame(substep_rows, index=pd.DatetimeIndex(substep_end_times))
    if not substep_frame.empty:
        substep_frame = substep_frame[~substep_frame.index.duplicated(keep="last")].sort_index()
    metadata: dict[str, object] = {
        "solar_model_name": SOLAR_MODEL_NAME,
        "solar_model_version": str(SOLAR_MODEL_VERSION),
        "solar_feature_set_version": SOLAR_FEATURE_SET_VERSION,
        "solar_model_contract_id": physical_solar_contract_id(
            config,
            latitude=latitude,
            longitude=longitude,
        ),
        "solar_physical_config_sha256": physical_solar_config_digest(config),
        "solar_physical_config_status": config.configuration_status,
        "solar_physical_config_source": config.source,
        "solar_physical_config": json.dumps(config.to_dict(), sort_keys=True),
        "solar_direct_diffuse_source": "forecast_direct_horizontal" if direct_interval is not None else "erbs_from_ssrd",
        "solar_temperature_model": (
            "faiman_u0_u1" if air_interval is not None and wind_interval is not None else "disabled_missing_forecast_met"
        ),
        "solar_ground_albedo_source": "forecast_or_observed" if albedo_interval is not None else "fixed_config",
        "solar_degradation_codes": ",".join(sorted(degradation)) if degradation else "none",
        "solar_substep_minutes": f"{config.substep_minutes:.6g}",
        "solar_interval_energy_conservation": "ecmwf_ghi_exact_by_interval",
        "solar_soc_integration_resolution": "physical_substeps_with_exact_issue_time_cutoff",
        "solar_first_interval_handling": "truncate_at_soc_anchor_and_preserve_full_ecmwf_interval_shape",
        "solar_power_semantics": "available_dc_before_battery_acceptance",
    }
    return frame, substep_frame, metadata


def build_physical_solar_forecast(
    ghi_w_m2: pd.Series,
    *,
    latitude: float,
    longitude: float,
    config: PhysicalSolarConfig,
    direct_horizontal_w_m2: pd.Series | None = None,
    air_temperature_c: pd.Series | None = None,
    wind_speed_m_s: pd.Series | None = None,
    ground_albedo: pd.Series | None = None,
    forecast_start_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert interval-mean GHI into interval physical-PV diagnostics."""

    interval, _substeps, metadata = build_physical_solar_forecast_frames(
        ghi_w_m2,
        latitude=latitude,
        longitude=longitude,
        config=config,
        direct_horizontal_w_m2=direct_horizontal_w_m2,
        air_temperature_c=air_temperature_c,
        wind_speed_m_s=wind_speed_m_s,
        ground_albedo=ground_albedo,
        forecast_start_time=forecast_start_time,
    )
    return interval, metadata
