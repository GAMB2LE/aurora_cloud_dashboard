#!/usr/bin/env python3
"""Learn APS operating states and evaluate state-aware SOC scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from power_soc_thresholds import MINIMUM_OPERATIONAL_SOC_PCT

MODEL_NAME = "hybrid_state_space_v5"
MODEL_VERSION = 5
STATE_SCHEMA_VERSION = 1
SCENARIO_SCHEMA_VERSION = 1

KIT_ORDER = ("CL61", "Radar", "HATPRO", "UAS")
KIT_BITS = {name: 1 << index for index, name in enumerate(KIT_ORDER)}
KIT_OUTLETS = {"UAS": 4, "CL61": 5, "Radar": 6, "HATPRO": 8}
COMPONENTS = ("DC",) + KIT_ORDER + ("UnknownAC",)
COMPONENT_INDEX = {name: index for index, name in enumerate(COMPONENTS)}

MODE_DC_ONLY = "dc_only"
MODE_UNKNOWN_AC = "unknown_ac"
PDU_ACTIVE_W = 5.0
AC_ACTIVE_W = 25.0
PDU_FRESHNESS_MINUTES = 60.0
OBSERVATION_FREQUENCY = "15min"
MIN_LEARNED_SAMPLES = 4
MIN_RUN_HOURS = 12
MAX_STARTS_PER_UTC_DAY = 1

SCENARIO_CURRENT = "current_mode"
SCENARIO_DC_ONLY = "dc_only"
SCENARIO_CL61 = "cl61_continuous"
SCENARIO_OPTIMIZED = "optimized_cl61"
CORE_SCENARIOS = (SCENARIO_CURRENT, SCENARIO_DC_ONLY, SCENARIO_CL61, SCENARIO_OPTIMIZED)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mode_id(active_kits: Iterable[str], *, unknown_ac: bool = False) -> str:
    active_set = set(active_kits)
    kits = tuple(name for name in KIT_ORDER if name in active_set)
    if kits:
        return "dc_" + "_".join(name.lower() for name in kits)
    return MODE_UNKNOWN_AC if unknown_ac else MODE_DC_ONLY


def mode_kits(value: str) -> tuple[str, ...]:
    text = str(value or "").lower()
    if text in {MODE_DC_ONLY, MODE_UNKNOWN_AC}:
        return ()
    return tuple(name for name in KIT_ORDER if name.lower() in text.split("_"))


def mode_label(value: str) -> str:
    if value == MODE_DC_ONLY:
        return "DC-Only"
    if value == MODE_UNKNOWN_AC:
        return "Unknown AC Load"
    kits = mode_kits(value)
    if kits:
        return "DC + " + " + ".join(kits)
    return str(value).replace("_", " ").title()


def mode_code(value: str) -> int:
    if value == MODE_UNKNOWN_AC:
        return 1 << len(KIT_ORDER)
    result = 0
    for kit in mode_kits(value):
        result |= KIT_BITS[kit]
    return result


def mode_from_code(value: int) -> str:
    if int(value) & (1 << len(KIT_ORDER)):
        return MODE_UNKNOWN_AC
    return mode_id(name for name, bit in KIT_BITS.items() if int(value) & bit)


def _power_frame(
    power: xr.Dataset,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    fields = (
        "BatterySOC",
        "BatteryWatts",
        "SolarWatts_East",
        "SolarWatts_South",
        "SolarWatts_West",
        "ACOutputWatts",
        "DCInverterWatts",
    )
    if "time" not in power:
        return pd.DataFrame()
    source = power
    if start is not None or end is not None:
        source = power.sel(time=slice(start, end))
    values = {
        name: np.asarray(source[name].values, dtype=np.float64)
        for name in fields
        if name in source and source[name].dims == ("time",)
    }
    if not values:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(source["time"].values)).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def observed_total_load(frame: pd.DataFrame) -> pd.Series:
    solar_names = ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West")
    if "BatteryWatts" in frame and all(name in frame for name in solar_names):
        solar = frame[list(solar_names)].sum(axis=1, min_count=len(solar_names))
        balanced = (solar - frame["BatteryWatts"]).clip(lower=0.0)
        if np.isfinite(balanced.to_numpy(dtype=np.float64)).any():
            balanced.name = "ObservedLoadWatts"
            return balanced
    names = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in frame]
    if not names:
        return pd.Series(dtype=np.float64, name="ObservedLoadWatts")
    result = frame[names].sum(axis=1, min_count=1).clip(lower=0.0)
    result.name = "ObservedLoadWatts"
    return result


def _pdu_frame(
    pdu: xr.Dataset | None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if pdu is None or "time" not in pdu:
        return pd.DataFrame()
    source = pdu
    if start is not None or end is not None:
        source = pdu.sel(time=slice(start, end))
    values: dict[str, np.ndarray] = {}
    for kit, outlet in KIT_OUTLETS.items():
        for metric in ("Watts", "State"):
            name = f"PDUOutlet{outlet}{metric}"
            if name in source and source[name].dims == ("time",):
                values[f"{kit}_{metric.lower()}"] = np.asarray(source[name].values, dtype=np.float64)
    if not values:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(source["time"].values)).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def build_observation_frame(
    power: xr.Dataset,
    pdu: xr.Dataset | None,
    *,
    end: pd.Timestamp | None = None,
    lookback_days: float = 7.0,
    frequency: str = OBSERVATION_FREQUENCY,
) -> pd.DataFrame:
    if "time" not in power or power.sizes.get("time", 0) == 0:
        return pd.DataFrame()
    power_times = pd.DatetimeIndex(power["time"].values)
    end = pd.Timestamp(end if end is not None else power_times.max())
    start = end - pd.Timedelta(days=float(lookback_days))
    power_frame = _power_frame(power, start=start, end=end)
    if power_frame.empty:
        return pd.DataFrame()
    observed = pd.DataFrame(index=power_frame.resample(frequency).median().index)
    observed["load_w"] = observed_total_load(power_frame).resample(frequency).median()
    for name in ("BatterySOC", "ACOutputWatts"):
        if name in power_frame:
            observed[name] = power_frame[name].resample(frequency).median()

    pdu_frame = _pdu_frame(pdu, start=start, end=end)
    if not pdu_frame.empty:
        pdu_frame = pdu_frame.loc[(pdu_frame.index >= start) & (pdu_frame.index <= end)]
        pdu_samples = pdu_frame.resample(frequency).median()
        for name in pdu_samples:
            observed[name] = pdu_samples[name].reindex(observed.index, method="nearest", tolerance=pd.Timedelta(frequency))

    mode_values: list[str] = []
    evidence_values: list[str] = []
    confidence_values: list[float] = []
    for _, row in observed.iterrows():
        active: list[str] = []
        has_pdu_evidence = False
        for kit in KIT_ORDER:
            watts = row.get(f"{kit}_watts", np.nan)
            state = row.get(f"{kit}_state", np.nan)
            if np.isfinite(watts):
                has_pdu_evidence = True
                if float(watts) >= PDU_ACTIVE_W:
                    active.append(kit)
            elif np.isfinite(state):
                has_pdu_evidence = True
                if float(state) >= 0.5:
                    active.append(kit)
        ac_active = bool(np.isfinite(row.get("ACOutputWatts", np.nan)) and row["ACOutputWatts"] > AC_ACTIVE_W)
        if active:
            selected = mode_id(active)
            evidence = "pdu_signature"
            confidence = 0.995 if ac_active else 0.90
        elif has_pdu_evidence and not ac_active:
            selected = MODE_DC_ONLY
            evidence = "pdu_and_ac"
            confidence = 0.995
        elif ac_active:
            selected = MODE_UNKNOWN_AC
            evidence = "ac_output"
            confidence = 0.80
        else:
            selected = MODE_DC_ONLY
            evidence = "ac_output"
            confidence = 0.90
        mode_values.append(selected)
        evidence_values.append(evidence)
        confidence_values.append(confidence)
    observed["direct_mode"] = mode_values
    observed["mode_evidence"] = evidence_values
    observed["direct_confidence"] = confidence_values
    subset = [name for name in ("load_w", "BatterySOC", "ACOutputWatts") if name in observed]
    return observed.dropna(how="all", subset=subset) if subset else observed


def _default_component_means(observations: pd.DataFrame) -> np.ndarray:
    means = np.array([200.0, 220.0, 300.0, 250.0, 200.0, 250.0], dtype=np.float64)
    if observations.empty:
        return means
    dc = observations.loc[observations["direct_mode"] == MODE_DC_ONLY, "load_w"].dropna()
    if not dc.empty:
        means[COMPONENT_INDEX["DC"]] = max(float(dc.median()), 0.0)
    for kit in KIT_ORDER:
        field = f"{kit}_watts"
        if field in observations:
            values = observations.loc[observations[field] >= PDU_ACTIVE_W, field].dropna()
            if not values.empty:
                means[COMPONENT_INDEX[kit]] = max(float(values.median()), 0.0)
    return means


def _bootstrap_components(raw_state: Mapping[str, Any] | None, observations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = _default_component_means(observations)
    covariance = np.diag(np.array([60.0, 100.0, 150.0, 120.0, 100.0, 150.0]) ** 2)
    counts = np.zeros(len(COMPONENTS), dtype=np.int64)
    if not isinstance(raw_state, Mapping):
        return means, covariance, counts
    component_state = raw_state.get("components")
    if isinstance(component_state, Mapping):
        for name, index in COMPONENT_INDEX.items():
            entry = component_state.get(name)
            if not isinstance(entry, Mapping):
                continue
            try:
                value = float(entry.get("mean_w", np.nan))
                variance = float(entry.get("variance_w2", np.nan))
                count = int(entry.get("observation_count", 0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= 0.0:
                means[index] = value
            if np.isfinite(variance) and variance > 0.0:
                covariance[index, index] = variance
            counts[index] = max(count, 0)
        saved_covariance = raw_state.get("covariance_w2")
        try:
            candidate = np.asarray(saved_covariance, dtype=np.float64)
        except (TypeError, ValueError):
            candidate = np.empty((0, 0), dtype=np.float64)
        if candidate.shape == covariance.shape and np.isfinite(candidate).all():
            candidate = (candidate + candidate.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(candidate)
            covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
        return means, covariance, counts

    registry = raw_state.get("load_mode_registry")
    if isinstance(registry, Mapping):
        dc_entry = registry.get("DC-Only")
        if isinstance(dc_entry, Mapping):
            try:
                means[0] = max(float(dc_entry.get("learned_level_w", means[0])), 0.0)
            except (TypeError, ValueError):
                pass
    return means, covariance, counts


def _mode_design(value: str) -> np.ndarray:
    design = np.zeros(len(COMPONENTS), dtype=np.float64)
    design[COMPONENT_INDEX["DC"]] = 1.0
    if value == MODE_UNKNOWN_AC:
        design[COMPONENT_INDEX["UnknownAC"]] = 1.0
    for kit in mode_kits(value):
        design[COMPONENT_INDEX[kit]] = 1.0
    return design


def _kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    design: np.ndarray,
    observation: float,
    measurement_variance: float,
    *,
    innovation_limit_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    predicted = float(design @ mean)
    innovation = float(observation - predicted)
    innovation_variance = float(design @ covariance @ design + measurement_variance)
    if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
        return mean, covariance, innovation, False
    limit = float(innovation_limit_sigma * np.sqrt(innovation_variance))
    clipped = bool(abs(innovation) > limit)
    innovation_used = float(np.clip(innovation, -limit, limit))
    gain = covariance @ design / innovation_variance
    updated_mean = np.clip(mean + gain * innovation_used, 0.0, None)
    identity = np.eye(len(mean), dtype=np.float64)
    update_matrix = identity - np.outer(gain, design)
    updated_covariance = update_matrix @ covariance @ update_matrix.T + np.outer(gain, gain) * measurement_variance
    updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
    return updated_mean, updated_covariance, innovation, clipped


@dataclass
class OperatingModelResult:
    state_dataset: xr.Dataset
    state: dict[str, Any]
    component_mean: np.ndarray
    component_covariance: np.ndarray
    learned_modes: tuple[str, ...]
    current_mode: str
    current_confidence: float


def fit_operating_model(
    power: xr.Dataset,
    pdu: xr.Dataset | None,
    *,
    raw_state: Mapping[str, Any] | None = None,
    end: pd.Timestamp | None = None,
    lookback_days: float = 7.0,
) -> OperatingModelResult:
    observations = build_observation_frame(power, pdu, end=end, lookback_days=lookback_days)
    if observations.empty:
        raise ValueError("No APS/PDU observations are available for operating-state learning")
    mean, covariance, counts = _bootstrap_components(raw_state, observations)
    process_variance = np.array([1.0, 4.0, 6.0, 5.0, 4.0, 8.0], dtype=np.float64)
    last_trained = pd.NaT
    if isinstance(raw_state, Mapping):
        candidate = pd.to_datetime(raw_state.get("last_observation_time_utc"), errors="coerce")
        if not pd.isna(candidate):
            last_trained = pd.Timestamp(candidate)
            if last_trained.tzinfo is not None:
                last_trained = last_trained.tz_convert("UTC").tz_localize(None)
    train_mask = np.ones(len(observations), dtype=bool) if pd.isna(last_trained) else observations.index > last_trained

    known_modes = {MODE_DC_ONLY, MODE_UNKNOWN_AC, mode_id(("CL61",))}
    known_modes.update(str(value) for value in observations["direct_mode"].dropna().unique())
    mode_names = tuple(sorted(known_modes, key=lambda value: (mode_code(value), value)))
    posterior = np.full(len(mode_names), 1.0 / len(mode_names), dtype=np.float64)
    previous_mode = str(raw_state.get("current_mode", MODE_DC_ONLY)) if isinstance(raw_state, Mapping) else MODE_DC_ONLY
    if previous_mode in mode_names:
        posterior[:] = 0.02 / max(len(mode_names) - 1, 1)
        posterior[mode_names.index(previous_mode)] = 0.98

    selected_modes: list[str] = []
    confidences: list[float] = []
    estimated_loads: list[float] = []
    innovations: list[float] = []
    outliers: list[float] = []
    probabilities = np.zeros((len(observations), len(mode_names)), dtype=np.float64)

    for row_index, (observation_time, row) in enumerate(observations.iterrows()):
        should_train = bool(train_mask[row_index])
        if should_train:
            covariance = covariance + np.diag(process_variance)
        direct_mode = str(row["direct_mode"])
        direct_confidence = float(row["direct_confidence"])
        prior = 0.985 * posterior + 0.015 / len(mode_names)
        emissions = np.ones(len(mode_names), dtype=np.float64)
        finite_load = np.isfinite(row.get("load_w", np.nan))
        for index, candidate in enumerate(mode_names):
            if direct_mode == candidate:
                emissions[index] *= max(direct_confidence, 0.5)
            else:
                emissions[index] *= max((1.0 - direct_confidence) / max(len(mode_names) - 1, 1), 1e-5)
            if finite_load:
                design = _mode_design(candidate)
                expected = float(design @ mean)
                variance = max(float(design @ covariance @ design + 75.0**2), 1.0)
                residual = float(row["load_w"] - expected)
                emissions[index] *= float(np.exp(-0.5 * residual * residual / variance) / np.sqrt(variance))
        posterior = prior * emissions
        if not np.isfinite(posterior).all() or posterior.sum() <= 0.0:
            posterior = np.full(len(mode_names), 1.0 / len(mode_names), dtype=np.float64)
        else:
            posterior /= posterior.sum()
        selected_index = int(np.argmax(posterior))
        selected_mode = mode_names[selected_index]
        probabilities[row_index] = posterior

        design = _mode_design(selected_mode)
        estimated_load = float(design @ mean)
        innovation = np.nan
        clipped = False
        if finite_load and should_train:
            mean, covariance, innovation, clipped = _kalman_update(
                mean,
                covariance,
                design,
                float(row["load_w"]),
                75.0**2,
            )
            counts[np.flatnonzero(design)] += 1
        for kit in KIT_ORDER:
            field = f"{kit}_watts"
            if not should_train or field not in row or not np.isfinite(row[field]) or float(row[field]) < PDU_ACTIVE_W:
                continue
            direct_design = np.zeros(len(COMPONENTS), dtype=np.float64)
            direct_design[COMPONENT_INDEX[kit]] = 1.0
            mean, covariance, _, _ = _kalman_update(
                mean,
                covariance,
                direct_design,
                float(row[field]),
                25.0**2,
            )
            counts[COMPONENT_INDEX[kit]] += 1

        selected_modes.append(selected_mode)
        confidences.append(float(posterior[selected_index]))
        estimated_loads.append(estimated_load)
        innovations.append(float(innovation))
        outliers.append(float(clipped))

    mode_counts = pd.Series(selected_modes).value_counts()
    learned_modes = tuple(
        value
        for value in mode_names
        if value != MODE_UNKNOWN_AC and int(mode_counts.get(value, 0)) >= MIN_LEARNED_SAMPLES
    )
    current_mode = selected_modes[-1]
    current_confidence = confidences[-1]
    new_observation_count = int(np.count_nonzero(train_mask))
    latest_observation_time = pd.Timestamp(observations.index[-1])
    if not pd.isna(last_trained) and new_observation_count == 0:
        latest_observation_time = last_trained
    state_ds = xr.Dataset(
        {
            "OperatingModeCode": (("time",), np.asarray([mode_code(value) for value in selected_modes], dtype=np.int16)),
            "OperatingModeConfidence": (("time",), np.asarray(confidences, dtype=np.float32)),
            "ObservedLoadWatts": (("time",), observations["load_w"].to_numpy(dtype=np.float32)),
            "EstimatedModeLoadWatts": (("time",), np.asarray(estimated_loads, dtype=np.float32)),
            "LoadInnovationWatts": (("time",), np.asarray(innovations, dtype=np.float32)),
            "LoadObservationOutlier": (("time",), np.asarray(outliers, dtype=np.float32)),
            "OperatingModeProbability": (("time", "mode"), probabilities.astype(np.float32)),
        },
        coords={
            "time": observations.index.to_numpy(dtype="datetime64[ns]"),
            "mode": np.asarray(mode_names, dtype=str),
        },
        attrs={
            "power_operating_state_product": "true",
            "schema_version": str(STATE_SCHEMA_VERSION),
            "model": MODEL_NAME,
            "model_version": str(MODEL_VERSION),
            "generated_at_utc": _utc_now(),
            "current_mode": current_mode,
            "current_mode_label": mode_label(current_mode),
            "current_mode_confidence": f"{current_confidence:.6g}",
            "learned_modes": json.dumps(list(learned_modes)),
            "component_names": json.dumps(list(COMPONENTS)),
            "component_mean_w": json.dumps({name: float(mean[index]) for index, name in enumerate(COMPONENTS)}),
            "component_std_w": json.dumps(
                {name: float(np.sqrt(max(covariance[index, index], 0.0))) for index, name in enumerate(COMPONENTS)}
            ),
            "observation_frequency": OBSERVATION_FREQUENCY,
            "last_observation_time_utc": latest_observation_time.isoformat(),
            "new_observation_count": str(new_observation_count),
        },
    )
    state_ds["OperatingModeCode"].attrs["mode_mapping"] = json.dumps(
        {str(mode_code(value)): mode_label(value) for value in mode_names}, sort_keys=True
    )
    state_ds["OperatingModeConfidence"].attrs["units"] = "1"
    for name in ("ObservedLoadWatts", "EstimatedModeLoadWatts", "LoadInnovationWatts"):
        state_ds[name].attrs["units"] = "W"

    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "updated_at_utc": _utc_now(),
        "current_mode": current_mode,
        "current_mode_confidence": current_confidence,
        "last_observation_time_utc": latest_observation_time.isoformat(),
        "new_observation_count": new_observation_count,
        "learned_modes": list(learned_modes),
        "components": {
            name: {
                "mean_w": float(mean[index]),
                "variance_w2": float(max(covariance[index, index], 0.0)),
                "observation_count": int(counts[index]),
            }
            for index, name in enumerate(COMPONENTS)
        },
        "covariance_w2": covariance.tolist(),
        "mode_sample_counts": {str(name): int(value) for name, value in mode_counts.items()},
    }
    return OperatingModelResult(
        state_dataset=state_ds,
        state=state,
        component_mean=mean,
        component_covariance=covariance,
        learned_modes=learned_modes,
        current_mode=current_mode,
        current_confidence=current_confidence,
    )


def _latest_soc(power: xr.Dataset) -> tuple[pd.Timestamp, float]:
    if "time" not in power or "BatterySOC" not in power or power.sizes.get("time", 0) == 0:
        raise ValueError("Power data do not contain BatterySOC")
    count = int(power.sizes["time"])
    starts = tuple(dict.fromkeys((max(count - 100_000, 0), 0)))
    for start_index in starts:
        view = power[["BatterySOC"]].isel(time=slice(start_index, None))
        values = np.asarray(view["BatterySOC"].values, dtype=np.float64)
        finite_indices = np.flatnonzero(np.isfinite(values))
        if finite_indices.size:
            index = int(finite_indices[-1])
            return pd.Timestamp(view["time"].values[index]), float(values[index])
    raise ValueError("Power data do not contain a finite BatterySOC sample")


def _hourly_solar_members(
    deterministic: xr.Dataset,
    ensemble: xr.Dataset | None,
    *,
    issue_time: pd.Timestamp,
    horizon_hours: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, str]]:
    times = pd.date_range(issue_time.floor("h"), issue_time.floor("h") + pd.Timedelta(hours=horizon_hours), freq="1h")
    if "ForecastSolarWatts" not in deterministic or "time" not in deterministic:
        raise ValueError("Deterministic forecast does not contain ForecastSolarWatts")
    deterministic_times = pd.DatetimeIndex(deterministic["time"].values)
    deterministic_series = pd.Series(
        np.asarray(deterministic["ForecastSolarWatts"].values, dtype=np.float64),
        index=deterministic_times,
    )
    deterministic_hourly = (
        deterministic_series.reindex(times.union(deterministic_times))
        .interpolate("time", limit_area="inside")
        .reindex(times)
        .ffill()
        .bfill()
        .clip(lower=0.0)
    )
    if ensemble is not None and "ForecastSolarWattsEnsemble" in ensemble and "time" in ensemble:
        source_times = pd.DatetimeIndex(ensemble["time"].values)
        rows = []
        native = np.asarray(
            ensemble["ForecastSolarWattsEnsemble"].transpose("member", "time").values,
            dtype=np.float64,
        )
        valid_native_times = source_times[np.any(np.isfinite(native), axis=0)] if native.size else pd.DatetimeIndex([])
        native_end = valid_native_times.max() if len(valid_native_times) else pd.NaT
        rank_factors = np.linspace(0.75, 1.25, max(native.shape[0], 1))
        deterministic_values = deterministic_hourly.to_numpy(dtype=np.float64)
        for member_index, values in enumerate(native):
            series = (
                pd.Series(values, index=source_times)
                .reindex(times.union(source_times))
                .interpolate("time", limit_area="inside")
                .reindex(times)
            )
            native_values = series.to_numpy(dtype=np.float64)
            ratio_mask = np.isfinite(native_values) & (deterministic_values >= 25.0)
            ratio = float(np.nanmedian(native_values[ratio_mask] / deterministic_values[ratio_mask])) if ratio_mask.any() else float(rank_factors[member_index])
            ratio = float(np.clip(ratio, 0.20, 1.80))
            fallback = deterministic_values * ratio
            if not pd.isna(native_end):
                lead = np.maximum((times - native_end) / pd.Timedelta(hours=1), 0.0).to_numpy(dtype=np.float64)
                widened_ratio = 1.0 + (ratio - 1.0) * (1.0 + np.minimum(lead / 96.0, 0.5))
                fallback = deterministic_values * np.clip(widened_ratio, 0.10, 2.00)
            combined = np.where(np.isfinite(native_values), native_values, fallback)
            rows.append(np.clip(combined, 0.0, None))
        if rows:
            return times, np.asarray(rows, dtype=np.float64), {
                "solar_member_source": "native_ensemble_with_deterministic_extension",
                "native_ensemble_end_time": "" if pd.isna(native_end) else pd.Timestamp(native_end).isoformat(),
                "uncertainty_extrapolated": str(bool(not pd.isna(native_end) and times[-1] > native_end)).lower(),
            }
    return times, deterministic_hourly.to_numpy(dtype=np.float64)[None, :], {
        "solar_member_source": "deterministic_only",
        "native_ensemble_end_time": "",
        "uncertainty_extrapolated": "false",
    }


def _component_members(mean: np.ndarray, covariance: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    count = max(int(count), 1)
    if count == 1:
        return mean[None, :]
    safe_covariance = np.asarray(covariance, dtype=np.float64).copy()
    safe_covariance = (safe_covariance + safe_covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(safe_covariance)
    safe_covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    values = np.random.default_rng(seed).multivariate_normal(mean, safe_covariance, size=count)
    return np.clip(values, 0.0, None)


def _load_members_for_modes(component_members: np.ndarray, modes: Sequence[str]) -> np.ndarray:
    components = np.asarray(component_members, dtype=np.float64)
    design = np.asarray([_mode_design(value) for value in modes], dtype=np.float64)
    if components.ndim != 2 or components.shape[1] != len(COMPONENTS):
        raise ValueError("Component members must be a member x component array")
    if not np.isfinite(components).all() or not np.isfinite(design).all():
        raise ValueError("Operating-mode load inputs must be finite")
    # The explicit contraction avoids spurious Accelerate/BLAS warnings seen
    # with a transposed, non-contiguous design matrix on macOS.
    loads = np.einsum("mc,tc->mt", components, design, optimize=False)
    return np.clip(loads, 0.0, None)


def integrate_soc_members(
    *,
    initial_soc: float,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    load_members_w: np.ndarray,
    capacity_kwh: float,
) -> np.ndarray:
    solar = np.asarray(solar_members_w, dtype=np.float64)
    load = np.asarray(load_members_w, dtype=np.float64)
    if solar.ndim != 2 or load.ndim != 2 or solar.shape != load.shape:
        raise ValueError("Solar and load members must be matching member x time arrays")
    soc = np.full(solar.shape, np.nan, dtype=np.float64)
    soc[:, 0] = float(np.clip(initial_soc, 0.0, 100.0))
    for index in range(1, len(times)):
        hours = max(float((times[index] - times[index - 1]) / pd.Timedelta(hours=1)), 0.0)
        net_kwh = (solar[:, index] - load[:, index]) * hours / 1000.0
        soc[:, index] = np.clip(soc[:, index - 1] + 100.0 * net_kwh / float(capacity_kwh), 0.0, 100.0)
    return soc


@dataclass
class ScheduleResult:
    modes: tuple[str, ...]
    collection_hours: float
    start_time: pd.Timestamp | None
    stop_time: pd.Timestamp | None
    minimum_p10_soc: float
    final_p10_soc: float
    safe: bool
    starts: int


@dataclass
class _Candidate:
    modes: tuple[str, ...]
    soc: np.ndarray
    on: bool
    run_hours: int
    last_start_day: object | None
    starts: int
    on_hours: int
    minimum_p10: float
    first_start_index: int | None


def _candidate_score(candidate: _Candidate) -> tuple[float, float, float, float]:
    first = candidate.first_start_index if candidate.first_start_index is not None else 10**6
    return (float(candidate.on_hours), float(-candidate.starts), float(candidate.minimum_p10), float(-first))


def optimize_cl61_schedule(
    *,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
    base_mode: str,
    horizon_hours: int = 96,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
    minimum_run_hours: int = MIN_RUN_HOURS,
    max_starts_per_day: int = MAX_STARTS_PER_UTC_DAY,
    beam_width: int = 300,
) -> ScheduleResult:
    full_times = pd.DatetimeIndex(times)
    full_solar = np.asarray(solar_members_w, dtype=np.float64)
    if len(full_times) == 0:
        raise ValueError("CL61 optimization requires at least one forecast time")
    if full_solar.ndim != 2 or full_solar.shape[1] < len(full_times):
        raise ValueError("Solar members must be a member x time array covering the forecast")
    full_solar = full_solar[:, : len(full_times)]
    decision_count = min(len(full_times), int(horizon_hours) + 1)
    decision_times = full_times[:decision_count]
    base_kits = set(mode_kits(base_mode))
    base_kits.discard("CL61")
    off_mode = mode_id(base_kits)
    on_mode = mode_id(base_kits | {"CL61"})
    off_load = _load_members_for_modes(component_members, [off_mode])[:, 0]
    on_load = _load_members_for_modes(component_members, [on_mode])[:, 0]
    initial_members = np.full(full_solar.shape[0], float(initial_soc), dtype=np.float64)
    candidates = [
        _Candidate(
            modes=(off_mode,),
            soc=initial_members,
            on=False,
            run_hours=0,
            last_start_day=None,
            starts=0,
            on_hours=0,
            minimum_p10=float(initial_soc),
            first_start_index=None,
        )
    ]
    search_complete = decision_count == 1
    for index in range(1, decision_count):
        next_candidates: list[_Candidate] = []
        day = decision_times[index].date()
        for candidate in candidates:
            choices = (True,) if candidate.on and candidate.run_hours < minimum_run_hours else (False, True)
            for turn_on in choices:
                is_start = bool(turn_on and not candidate.on)
                if is_start and candidate.last_start_day == day:
                    continue
                if is_start and max_starts_per_day <= 0:
                    continue
                load = on_load if turn_on else off_load
                hours = max(
                    float((decision_times[index] - decision_times[index - 1]) / pd.Timedelta(hours=1)),
                    0.0,
                )
                soc = np.clip(
                    candidate.soc
                    + 100.0 * (full_solar[:, index] - load) * hours / (1000.0 * capacity_kwh),
                    0.0,
                    100.0,
                )
                p10 = float(np.nanquantile(soc, 0.10))
                if p10 < float(minimum_soc) - 1e-9:
                    continue
                next_candidates.append(
                    _Candidate(
                        modes=candidate.modes + (on_mode if turn_on else off_mode,),
                        soc=soc,
                        on=turn_on,
                        run_hours=candidate.run_hours + 1 if turn_on else 0,
                        last_start_day=day if is_start else candidate.last_start_day,
                        starts=candidate.starts + int(is_start),
                        on_hours=candidate.on_hours + int(turn_on),
                        minimum_p10=min(candidate.minimum_p10, p10),
                        first_start_index=index if is_start and candidate.first_start_index is None else candidate.first_start_index,
                    )
                )
        if not next_candidates:
            candidates = []
            break
        grouped: dict[tuple[bool, int, object | None, int], list[_Candidate]] = {}
        for candidate in next_candidates:
            key = (
                candidate.on,
                min(candidate.run_hours, minimum_run_hours),
                candidate.last_start_day,
                int(np.nanmedian(candidate.soc) // 2),
            )
            grouped.setdefault(key, []).append(candidate)
        reduced = [max(values, key=_candidate_score) for values in grouped.values()]
        ranked = sorted(reduced, key=_candidate_score, reverse=True)
        baseline = next(
            (candidate for candidate in next_candidates if candidate.starts == 0 and candidate.on_hours == 0),
            None,
        )
        width = max(int(beam_width), 1)
        candidates = ranked[:width]
        if baseline is not None and not any(value.starts == 0 and value.on_hours == 0 for value in candidates):
            candidates = ranked[: width - 1] + [baseline]
        search_complete = index == decision_count - 1

    valid = [] if not search_complete else [
        candidate for candidate in candidates if not candidate.on or candidate.run_hours >= minimum_run_hours
    ]

    def extend_candidate(candidate: _Candidate) -> tuple[_Candidate, tuple[str, ...], float, float]:
        modes = list(candidate.modes)
        soc = candidate.soc.copy()
        minimum_p10 = float(candidate.minimum_p10)
        for index in range(decision_count, len(full_times)):
            hours = max(float((full_times[index] - full_times[index - 1]) / pd.Timedelta(hours=1)), 0.0)
            soc = np.clip(
                soc + 100.0 * (full_solar[:, index] - off_load) * hours / (1000.0 * capacity_kwh),
                0.0,
                100.0,
            )
            minimum_p10 = min(minimum_p10, float(np.nanquantile(soc, 0.10)))
            modes.append(off_mode)
        return candidate, tuple(modes), minimum_p10, float(np.nanquantile(soc, 0.10))

    evaluated = [extend_candidate(candidate) for candidate in valid]
    safe_evaluated = [value for value in evaluated if value[2] >= float(minimum_soc) - 1e-9]
    if safe_evaluated:
        best, modes, minimum_p10, final_p10 = max(
            safe_evaluated,
            key=lambda value: (
                float(value[0].on_hours),
                float(-value[0].starts),
                float(value[2]),
                float(-(value[0].first_start_index if value[0].first_start_index is not None else 10**6)),
            ),
        )
    else:
        modes = tuple(off_mode for _ in full_times)
        off_loads = np.repeat(off_load[:, np.newaxis], len(full_times), axis=1)
        off_soc = integrate_soc_members(
            initial_soc=initial_soc,
            times=full_times,
            solar_members_w=full_solar,
            load_members_w=off_loads,
            capacity_kwh=capacity_kwh,
        )
        off_p10 = np.nanquantile(off_soc, 0.10, axis=0)
        minimum_p10 = float(np.nanmin(off_p10))
        final_p10 = float(off_p10[-1])
        best = _Candidate(
            modes=modes[:decision_count],
            soc=off_soc[:, decision_count - 1],
            on=False,
            run_hours=0,
            last_start_day=None,
            starts=0,
            on_hours=0,
            minimum_p10=minimum_p10,
            first_start_index=None,
        )

    on_indices = [index for index, value in enumerate(modes) if "CL61" in mode_kits(value)]
    start_time = full_times[on_indices[0]] if on_indices else None
    stop_time = None
    if on_indices and on_indices[-1] + 1 < len(full_times):
        stop_time = full_times[on_indices[-1] + 1]
    return ScheduleResult(
        modes=modes,
        collection_hours=float(best.on_hours),
        start_time=start_time,
        stop_time=stop_time,
        minimum_p10_soc=float(minimum_p10),
        final_p10_soc=float(final_p10),
        safe=bool(minimum_p10 >= minimum_soc),
        starts=int(best.starts),
    )


def _schedule_modes(times: pd.DatetimeIndex, base_mode: str, start: pd.Timestamp, duration_hours: int) -> tuple[str, ...]:
    base_kits = set(mode_kits(base_mode))
    base_kits.discard("CL61")
    off_mode = mode_id(base_kits)
    on_mode = mode_id(base_kits | {"CL61"})
    stop = pd.Timestamp(start) + pd.Timedelta(hours=int(duration_hours))
    return tuple(on_mode if pd.Timestamp(start) <= value < stop else off_mode for value in times)


def _scenario_members(
    modes: Sequence[str],
    *,
    times: pd.DatetimeIndex,
    solar_members: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
) -> tuple[np.ndarray, np.ndarray]:
    loads = _load_members_for_modes(component_members, modes)
    soc = integrate_soc_members(
        initial_soc=initial_soc,
        times=times,
        solar_members_w=solar_members,
        load_members_w=loads,
        capacity_kwh=capacity_kwh,
    )
    return loads, soc


def build_operating_scenarios(
    power: xr.Dataset,
    deterministic: xr.Dataset,
    model: OperatingModelResult,
    *,
    ensemble: xr.Dataset | None = None,
    horizon_hours: int = 240,
    optimization_hours: int = 96,
    capacity_kwh: float | None = None,
) -> xr.Dataset:
    issue_time, initial_soc = _latest_soc(power)
    available_end = pd.Timestamp(deterministic["time"].values[-1]) if "time" in deterministic else issue_time
    available_hours = max(int((available_end - issue_time) / pd.Timedelta(hours=1)), 1)
    actual_horizon = min(int(horizon_hours), available_hours)
    capacity = float(capacity_kwh or deterministic.attrs.get("battery_capacity_kwh", 26.0))
    times, solar_members, solar_metadata = _hourly_solar_members(
        deterministic,
        ensemble,
        issue_time=issue_time,
        horizon_hours=actual_horizon,
    )
    member_count = solar_members.shape[0]
    seed = int(issue_time.value % (2**32 - 1))
    component_members = _component_members(model.component_mean, model.component_covariance, member_count, seed=seed)
    base_mode = model.current_mode if model.current_mode != MODE_UNKNOWN_AC else MODE_DC_ONLY
    optimized = optimize_cl61_schedule(
        times=times,
        solar_members_w=solar_members,
        component_members=component_members,
        initial_soc=initial_soc,
        capacity_kwh=capacity,
        base_mode=base_mode,
        horizon_hours=min(optimization_hours, actual_horizon),
    )
    optimized_modes = list(optimized.modes)
    if len(optimized_modes) < len(times):
        tail_kits = set(mode_kits(base_mode))
        tail_kits.discard("CL61")
        tail_mode = mode_id(tail_kits)
        optimized_modes.extend([tail_mode] * (len(times) - len(optimized_modes)))

    scenario_modes: dict[str, tuple[str, ...]] = {
        SCENARIO_CURRENT: tuple(base_mode for _ in times),
        SCENARIO_DC_ONLY: tuple(MODE_DC_ONLY for _ in times),
        SCENARIO_CL61: tuple(mode_id(("CL61",)) for _ in times),
        SCENARIO_OPTIMIZED: tuple(optimized_modes),
    }
    for learned_mode in model.learned_modes:
        if learned_mode in {MODE_DC_ONLY, mode_id(("CL61",))}:
            continue
        scenario_modes.setdefault(f"learned_{learned_mode}", tuple(learned_mode for _ in times))

    scenario_ids = tuple(scenario_modes)
    labels = {
        SCENARIO_CURRENT: f"Current: {mode_label(base_mode)}",
        SCENARIO_DC_ONLY: "DC-Only",
        SCENARIO_CL61: "DC + CL61 Continuously On",
        SCENARIO_OPTIMIZED: "Optimized CL61 Schedule",
    }
    load_p10: list[np.ndarray] = []
    load_p50: list[np.ndarray] = []
    load_p90: list[np.ndarray] = []
    soc_p10: list[np.ndarray] = []
    soc_p50: list[np.ndarray] = []
    soc_p90: list[np.ndarray] = []
    below_probability: list[np.ndarray] = []
    mode_codes: list[np.ndarray] = []
    collection_hours: list[float] = []
    minimum_p10: list[float] = []
    final_p10: list[float] = []
    starts: list[int] = []
    start_times: list[np.datetime64] = []
    stop_times: list[np.datetime64] = []
    safe_values: list[float] = []
    for scenario_id, modes in scenario_modes.items():
        loads, soc = _scenario_members(
            modes,
            times=times,
            solar_members=solar_members,
            component_members=component_members,
            initial_soc=initial_soc,
            capacity_kwh=capacity,
        )
        load_p10.append(np.nanquantile(loads, 0.10, axis=0))
        load_p50.append(np.nanquantile(loads, 0.50, axis=0))
        load_p90.append(np.nanquantile(loads, 0.90, axis=0))
        p10 = np.nanquantile(soc, 0.10, axis=0)
        soc_p10.append(p10)
        soc_p50.append(np.nanquantile(soc, 0.50, axis=0))
        soc_p90.append(np.nanquantile(soc, 0.90, axis=0))
        below_probability.append(np.mean(soc < MINIMUM_OPERATIONAL_SOC_PCT, axis=0))
        mode_codes.append(np.asarray([mode_code(value) for value in modes], dtype=np.int16))
        on = np.asarray(["CL61" in mode_kits(value) for value in modes], dtype=bool)
        collection_hours.append(float(np.count_nonzero(on[1:])))
        minimum_p10.append(float(np.nanmin(p10)))
        final_p10.append(float(p10[-1]))
        safe_values.append(float(np.nanmin(p10) >= MINIMUM_OPERATIONAL_SOC_PCT))
        transitions = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        stops_found = np.flatnonzero(~on & np.r_[False, on[:-1]])
        starts.append(int(len(transitions)))
        start_times.append(times[transitions[0]].to_datetime64() if transitions.size else np.datetime64("NaT"))
        stop_times.append(times[stops_found[-1]].to_datetime64() if stops_found.size else np.datetime64("NaT"))
        labels.setdefault(scenario_id, mode_label(modes[0]))

    output = xr.Dataset(
        {
            "ScenarioLoadP10Watts": (("scenario", "time"), np.asarray(load_p10, dtype=np.float32)),
            "ScenarioLoadP50Watts": (("scenario", "time"), np.asarray(load_p50, dtype=np.float32)),
            "ScenarioLoadP90Watts": (("scenario", "time"), np.asarray(load_p90, dtype=np.float32)),
            "ScenarioSOCP10": (("scenario", "time"), np.asarray(soc_p10, dtype=np.float32)),
            "ScenarioSOCP50": (("scenario", "time"), np.asarray(soc_p50, dtype=np.float32)),
            "ScenarioSOCP90": (("scenario", "time"), np.asarray(soc_p90, dtype=np.float32)),
            "ScenarioBelow40Probability": (("scenario", "time"), np.asarray(below_probability, dtype=np.float32)),
            "ScenarioModeCode": (("scenario", "time"), np.asarray(mode_codes, dtype=np.int16)),
            "ScenarioCollectionHours": (("scenario",), np.asarray(collection_hours, dtype=np.float32)),
            "ScenarioMinimumP10SOC": (("scenario",), np.asarray(minimum_p10, dtype=np.float32)),
            "ScenarioFinalP10SOC": (("scenario",), np.asarray(final_p10, dtype=np.float32)),
            "ScenarioSafe": (("scenario",), np.asarray(safe_values, dtype=np.float32)),
            "ScenarioStarts": (("scenario",), np.asarray(starts, dtype=np.int16)),
            "ScenarioStartTime": (("scenario",), np.asarray(start_times, dtype="datetime64[ns]")),
            "ScenarioStopTime": (("scenario",), np.asarray(stop_times, dtype="datetime64[ns]")),
            "SolarEnsembleWatts": (("member", "time"), solar_members.astype(np.float32)),
            "SolarP10Watts": (("time",), np.nanquantile(solar_members, 0.10, axis=0).astype(np.float32)),
            "SolarP50Watts": (("time",), np.nanquantile(solar_members, 0.50, axis=0).astype(np.float32)),
            "SolarP90Watts": (("time",), np.nanquantile(solar_members, 0.90, axis=0).astype(np.float32)),
            "ComponentLoadWatts": (("member", "component"), component_members.astype(np.float32)),
        },
        coords={
            "scenario": np.asarray(scenario_ids, dtype=str),
            "scenario_label": (("scenario",), np.asarray([labels[value] for value in scenario_ids], dtype=str)),
            "time": times.to_numpy(dtype="datetime64[ns]"),
            "member": np.arange(1, member_count + 1, dtype=np.int16),
            "component": np.asarray(COMPONENTS, dtype=str),
        },
        attrs={
            "power_operating_scenarios_product": "true",
            "schema_version": str(SCENARIO_SCHEMA_VERSION),
            "model": MODEL_NAME,
            "model_version": str(MODEL_VERSION),
            "generated_at_utc": _utc_now(),
            "initial_soc_time": issue_time.isoformat(),
            "initial_soc_pct": f"{initial_soc:.6g}",
            "battery_capacity_kwh": f"{capacity:.6g}",
            "current_mode": model.current_mode,
            "current_mode_label": mode_label(model.current_mode),
            "current_mode_confidence": f"{model.current_confidence:.6g}",
            "scenario_base_mode": base_mode,
            "forecast_horizon_hours": str(actual_horizon),
            "optimization_horizon_hours": str(min(optimization_hours, actual_horizon)),
            "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
            "minimum_cl61_run_hours": str(MIN_RUN_HOURS),
            "max_cl61_starts_per_utc_day": str(MAX_STARTS_PER_UTC_DAY),
            "control_authority": "advisory_only",
            "optimized_safe": str(bool(optimized.safe)).lower(),
            "optimized_collection_hours": f"{optimized.collection_hours:.6g}",
            "optimized_minimum_p10_soc": f"{optimized.minimum_p10_soc:.6g}",
            **solar_metadata,
        },
    )
    for name in ("ScenarioSOCP10", "ScenarioSOCP50", "ScenarioSOCP90", "ScenarioMinimumP10SOC", "ScenarioFinalP10SOC"):
        output[name].attrs["units"] = "%"
    for name in (
        "ScenarioLoadP10Watts",
        "ScenarioLoadP50Watts",
        "ScenarioLoadP90Watts",
        "SolarEnsembleWatts",
        "SolarP10Watts",
        "SolarP50Watts",
        "SolarP90Watts",
        "ComponentLoadWatts",
    ):
        output[name].attrs["units"] = "W"
    output["ScenarioBelow40Probability"].attrs["units"] = "1"
    output["ScenarioModeCode"].attrs["mode_mapping"] = json.dumps(
        {str(mode_code(value)): mode_label(value) for value in {mode for modes in scenario_modes.values() for mode in modes}},
        sort_keys=True,
    )
    return output


def evaluate_custom_schedule(
    scenarios: xr.Dataset,
    *,
    start_time: pd.Timestamp,
    duration_hours: int,
) -> dict[str, Any]:
    times = pd.DatetimeIndex(scenarios["time"].values)
    components = tuple(str(value) for value in scenarios["component"].values)
    if components != COMPONENTS:
        raise ValueError("Scenario component schema is not compatible with this model version")
    solar = np.asarray(scenarios["SolarEnsembleWatts"].values, dtype=np.float64)
    component_members = np.asarray(scenarios["ComponentLoadWatts"].values, dtype=np.float64)
    base_mode = str(scenarios.attrs.get("scenario_base_mode", scenarios.attrs.get("current_mode", MODE_DC_ONLY)))
    modes = _schedule_modes(times, base_mode, pd.Timestamp(start_time), int(duration_hours))
    loads = _load_members_for_modes(component_members, modes)
    soc = integrate_soc_members(
        initial_soc=float(scenarios.attrs["initial_soc_pct"]),
        times=times,
        solar_members_w=solar,
        load_members_w=loads,
        capacity_kwh=float(scenarios.attrs["battery_capacity_kwh"]),
    )
    p10 = np.nanquantile(soc, 0.10, axis=0)
    return {
        "time": times,
        "modes": modes,
        "load_p50_w": np.nanquantile(loads, 0.50, axis=0),
        "soc_p10": p10,
        "soc_p50": np.nanquantile(soc, 0.50, axis=0),
        "soc_p90": np.nanquantile(soc, 0.90, axis=0),
        "below_40_probability": np.mean(soc < MINIMUM_OPERATIONAL_SOC_PCT, axis=0),
        "collection_hours": float(np.count_nonzero(["CL61" in mode_kits(value) for value in modes[1:]])),
        "minimum_p10_soc": float(np.nanmin(p10)),
        "final_p10_soc": float(p10[-1]),
        "safe": bool(np.nanmin(p10) >= MINIMUM_OPERATIONAL_SOC_PCT),
    }
