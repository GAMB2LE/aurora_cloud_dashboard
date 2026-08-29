"""Issue-time-safe utilities for the isolated v12 hybrid power candidate.

This module deliberately has no operational writer.  It learns only from
archived forecast issues whose valid observations were available before the
new issue-time cutoff, then returns bounded residual paths for the candidate
runner.  The same module rebuilds campaign evidence from immutable pair
bundles; it never reads or mutates the v10 adaptive state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


V12_FORECAST_SYSTEM_VERSION = "power-v12-hybrid-candidate"
V12_FEATURE_SET_VERSION = "issue_safe_physical_pv_bounded_load_residual_v2"
V12_POWER_HISTORY_DAYS = 21.0
LOAD_RESIDUAL_MODEL_NAME = "bounded_ridge_load_residual_v1"
LOAD_RESIDUAL_MIN_SAMPLES = 48
LOAD_RESIDUAL_MIN_CYCLES = 3
LOAD_RESIDUAL_MIN_UTC_DAYS = 3
LOAD_RESIDUAL_BOUND_W = 500.0
LEAD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0_6h", 0.0, 6.0),
    ("6_24h", 6.0, 24.0),
    ("24_48h", 24.0, 48.0),
    ("48_96h", 48.0, 96.0),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _as_utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _observed_load_w(power: xr.Dataset) -> pd.Series:
    """Calculate station load using the same energy-balance convention as v10."""
    if "time" not in power.coords:
        return pd.Series(dtype=np.float64)
    frame = pd.DataFrame(
        {name: np.asarray(power[name].values, dtype=np.float64) for name in power.data_vars if power[name].dims == ("time",)},
        index=pd.DatetimeIndex(power["time"].values),
    )
    solar_fields = [
        name
        for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West")
        if name in frame
    ]
    if "BatteryWatts" in frame and len(solar_fields) == 3:
        solar = frame[solar_fields].sum(axis=1, min_count=3)
        balanced = (solar - frame["BatteryWatts"]).clip(lower=0.0)
        if np.isfinite(balanced.to_numpy(dtype=np.float64)).any():
            return balanced
    load_fields = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in frame]
    if not load_fields:
        return pd.Series(dtype=np.float64)
    return frame[load_fields].sum(axis=1, min_count=1).clip(lower=0.0)


def _repeat_issue_field(
    archive: xr.Dataset,
    name: str,
    length: int,
    *,
    default: object,
) -> np.ndarray:
    if name not in archive:
        return np.full(length, default)
    steps = int(archive.sizes.get("forecast_step", 0))
    values = np.asarray(archive[name].values).reshape(-1)
    return np.repeat(values, steps)


def _load_training_rows(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    cutoff: pd.Timestamp,
    load_mode: str,
) -> pd.DataFrame:
    """Return one independent, pre-cutoff load residual row per cycle/valid time."""
    if archive is None or not {"ForecastLoadWatts", "ForecastValidTime", "ForecastLeadHours"}.issubset(archive):
        return pd.DataFrame()
    if archive.sizes.get("issue_time", 0) == 0 or archive.sizes.get("forecast_step", 0) == 0:
        return pd.DataFrame()
    values = np.asarray(archive["ForecastLoadWatts"].values, dtype=np.float64).reshape(-1)
    valid_times = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    lead_hours = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    steps = int(archive.sizes["forecast_step"])
    issue_times = pd.DatetimeIndex(
        np.repeat(np.asarray(archive["issue_time"].values).reshape(-1), steps)
    )
    cycles = pd.DatetimeIndex(
        _repeat_issue_field(archive, "ECMWFCycleTime", len(values), default=np.datetime64("NaT"))
    )
    cycles = cycles.where(~cycles.isna(), issue_times.floor("3h"))
    modes = _repeat_issue_field(archive, "LoadMode", len(values), default="unknown").astype(str)
    model_contracts = _repeat_issue_field(
        archive, "ForecastModelContractID", len(values), default="legacy"
    ).astype(str)
    observed = _observed_load_w(power)
    if observed.empty:
        return pd.DataFrame()
    mask = (
        np.isfinite(values)
        & np.isfinite(lead_hours)
        & ~valid_times.isna()
        & ~issue_times.isna()
        & (issue_times < cutoff)
        & (valid_times <= cutoff)
        & (modes == str(load_mode))
    )
    if not np.any(mask):
        return pd.DataFrame()
    valid_times = valid_times[mask]
    observed_values = observed.reindex(
        valid_times,
        method="nearest",
        tolerance=pd.Timedelta(minutes=10),
    ).to_numpy(dtype=np.float64)
    paired = np.isfinite(observed_values)
    if not np.any(paired):
        return pd.DataFrame()
    rows = pd.DataFrame(
        {
            "issue_time": issue_times[mask][paired],
            "cycle_time": cycles[mask][paired],
            "valid_time": valid_times[paired],
            "lead_hour": lead_hours[mask][paired],
            "forecast_load_w": values[mask][paired],
            "observed_load_w": observed_values[paired],
            "forecast_model_contract_id": model_contracts[mask][paired],
        }
    )
    rows["residual_w"] = rows["observed_load_w"] - rows["forecast_load_w"]
    # Cached re-anchors share forcing and are not independent training records.
    return (
        rows.sort_values("issue_time")
        .drop_duplicates(["cycle_time", "valid_time"], keep="last")
        .sort_values(["valid_time", "issue_time"])
        .reset_index(drop=True)
    )


def _feature_matrix(times: pd.DatetimeIndex, lead_hours: np.ndarray) -> np.ndarray:
    lead = np.asarray(lead_hours, dtype=np.float64) / 96.0
    hour = np.asarray(times.hour + times.minute / 60.0, dtype=np.float64)
    angle = 2.0 * np.pi * hour / 24.0
    return np.column_stack(
        (
            np.ones(len(times), dtype=np.float64),
            lead,
            np.square(lead),
            np.sin(angle),
            np.cos(angle),
        )
    )


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0  # Never shrink the residual intercept.
    try:
        return np.linalg.solve(X.T @ X + penalty, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X.T @ X + penalty) @ X.T @ y


@dataclass(frozen=True)
class LoadResidualFit:
    status: str
    contract_id: str
    p10_correction_w: pd.Series
    p50_correction_w: pd.Series
    p90_correction_w: pd.Series
    training_samples: int
    training_cycles: int
    training_days: int
    bound_w: float
    selection: str

    def as_profile(self) -> dict[str, object]:
        return {
            "status": self.status,
            "contract_id": self.contract_id,
            "p10_correction_w": self.p10_correction_w,
            "p50_correction_w": self.p50_correction_w,
            "p90_correction_w": self.p90_correction_w,
            "training_samples": self.training_samples,
            "training_cycles": self.training_cycles,
            "training_days": self.training_days,
            "bound_w": self.bound_w,
            "selection": self.selection,
        }


def fit_bounded_load_residual(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    issue_time: pd.Timestamp | str,
    forecast_times: Iterable[object],
    load_mode: str,
    bound_w: float = LOAD_RESIDUAL_BOUND_W,
) -> LoadResidualFit:
    """Fit a small ridge residual model using only data available at issue time.

    It is intentionally fail-closed: insufficient same-mode independent
    evidence produces an explicit zero correction rather than borrowing a
    different operating state or future observation.
    """
    cutoff = _as_utc_naive(issue_time)
    times = pd.DatetimeIndex(forecast_times)
    rows = _load_training_rows(archive, power, cutoff=cutoff, load_mode=load_mode)
    samples = int(len(rows))
    cycles = int(rows["cycle_time"].nunique()) if not rows.empty else 0
    days = int(rows["valid_time"].dt.floor("D").nunique()) if not rows.empty else 0
    zero = pd.Series(np.zeros(len(times), dtype=np.float64), index=times)
    contracts = (
        sorted({str(value) for value in rows.get("forecast_model_contract_id", []) if str(value)})
        if not rows.empty
        else []
    )
    payload = {
        "schema": 1,
        "name": LOAD_RESIDUAL_MODEL_NAME,
        "bound_w": float(bound_w),
        "feature_columns": ["intercept", "lead", "lead_squared", "utc_hour_sin", "utc_hour_cos"],
        "load_mode": str(load_mode),
        "source_contracts": contracts,
    }
    contract_id = "load-residual-v1-" + stable_json_digest(payload)[:16]
    if (
        samples < LOAD_RESIDUAL_MIN_SAMPLES
        or cycles < LOAD_RESIDUAL_MIN_CYCLES
        or days < LOAD_RESIDUAL_MIN_UTC_DAYS
    ):
        status = f"insufficient_issue_time_evidence:samples={samples};cycles={cycles};days={days}"
        return LoadResidualFit(
            status,
            contract_id,
            zero,
            zero,
            zero,
            samples,
            cycles,
            days,
            float(bound_w),
            "disabled_fail_closed",
        )

    X = _feature_matrix(pd.DatetimeIndex(rows["valid_time"]), rows["lead_hour"].to_numpy())
    y = rows["residual_w"].to_numpy(dtype=np.float64)
    split = max(int(np.floor(len(rows) * 0.8)), 1)
    alphas = (0.1, 1.0, 10.0, 100.0)
    if len(rows) - split >= 12:
        validation = []
        for alpha in alphas:
            coefficients = _fit_ridge(X[:split], y[:split], alpha)
            validation.append((float(np.mean(np.abs((X[split:] @ coefficients) - y[split:]))), alpha))
        _, selected_alpha = min(validation, key=lambda item: (item[0], item[1]))
        selection = f"blocked_rolling_origin_alpha={selected_alpha:g}"
    else:
        selected_alpha = 10.0
        selection = "insufficient_holdout_default_alpha=10"
    coefficients = _fit_ridge(X, y, selected_alpha)
    train_prediction = X @ coefficients
    residual_noise = y - train_prediction
    lower_noise, upper_noise = np.nanquantile(residual_noise, (0.10, 0.90))
    lead = (times - cutoff) / pd.Timedelta(hours=1)
    predicted = _feature_matrix(times, lead.to_numpy(dtype=np.float64)) @ coefficients
    shrink = float(samples / (samples + LOAD_RESIDUAL_MIN_SAMPLES))
    p50 = np.clip(predicted * shrink, -bound_w, bound_w)
    p10 = np.clip((predicted + lower_noise) * shrink, -bound_w, bound_w)
    p90 = np.clip((predicted + upper_noise) * shrink, -bound_w, bound_w)
    return LoadResidualFit(
        "active",
        contract_id,
        pd.Series(p10, index=times),
        pd.Series(p50, index=times),
        pd.Series(p90, index=times),
        samples,
        cycles,
        days,
        float(bound_w),
        selection,
    )


def v12_feature_digest(
    *,
    physical_config_digest: str,
    load_residual_contract_id: str,
    power_history_days: float = V12_POWER_HISTORY_DAYS,
) -> str:
    return stable_json_digest(
        {
            "schema": 1,
            "feature_set_version": V12_FEATURE_SET_VERSION,
            "physical_config_digest": str(physical_config_digest),
            "load_residual_contract_id": str(load_residual_contract_id),
            "power_history_days": float(power_history_days),
        }
    )


def v12_forecast_identity(
    *,
    lane: str,
    issue_time: pd.Timestamp | str,
    source_cycle_set_id: str,
    source_manifest_digest: str,
    physical_config_digest: str,
    load_residual: LoadResidualFit | None,
    code_revision: str,
    power_history_days: float = V12_POWER_HISTORY_DAYS,
) -> dict[str, str]:
    residual = load_residual or LoadResidualFit(
        "not_requested",
        "",
        pd.Series(dtype=np.float64),
        pd.Series(dtype=np.float64),
        pd.Series(dtype=np.float64),
        0,
        0,
        0,
        LOAD_RESIDUAL_BOUND_W,
        "not_requested",
    )
    degraded = ["hardware_geometry_unverified", "solar_residual_disabled_until_mpp_active_history"]
    if residual.status != "active":
        degraded.append("load_residual_" + residual.status.split(":", 1)[0])
    return {
        "forecast_model_name": "aps_soc_energy_balance_v12_hybrid_candidate",
        "forecast_model_version": "12",
        "forecast_model_status": "candidate",
        "forecast_system_version": V12_FORECAST_SYSTEM_VERSION,
        "feature_set_version": V12_FEATURE_SET_VERSION,
        "feature_set_digest": v12_feature_digest(
            physical_config_digest=physical_config_digest,
            load_residual_contract_id=residual.contract_id,
            power_history_days=power_history_days,
        ),
        "training_cutoff_utc": _as_utc_naive(issue_time).isoformat(),
        "forecast_code_revision": str(code_revision),
        "source_cycle_set_id": str(source_cycle_set_id),
        "source_manifest_digest": str(source_manifest_digest),
        "degraded_mode_code": "+".join(degraded),
        "candidate_lane": str(lane),
    }


def completed_pair_bundles(pairs_root: Path) -> list[tuple[dict[str, object], Path]]:
    """Return only immutable, complete two-level pair bundles."""
    root = Path(pairs_root)
    if not root.exists():
        return []
    bundles: list[tuple[dict[str, object], Path]] = []
    for family in sorted(root.iterdir()):
        if not family.is_dir() or family.name.startswith("."):
            continue
        for bundle in sorted(family.iterdir()):
            if not bundle.is_dir() or bundle.name.startswith("."):
                continue
            manifest_path = bundle / "pair_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("pair_status") != "complete":
                continue
            if not (bundle / "baseline_forecast.zarr").exists() or not (
                bundle / "candidate_forecast.zarr"
            ).exists():
                continue
            bundles.append((manifest, bundle))
    return bundles


def _power_series(power: xr.Dataset, name: str) -> pd.Series:
    if name not in power or "time" not in power.coords or power[name].dims != ("time",):
        return pd.Series(dtype=np.float64)
    return pd.Series(
        np.asarray(power[name].values, dtype=np.float64),
        index=pd.DatetimeIndex(power["time"].values),
    )


def _pair_text(dataset: xr.Dataset, name: str, fallback: str = "") -> str:
    return str(dataset.attrs.get(name, fallback) or fallback)


def _irradiance_regime(value: float) -> str:
    if not np.isfinite(value) or value <= 1.0:
        return "dark"
    if value < 100.0:
        return "low_irradiance"
    if value < 350.0:
        return "moderate_irradiance"
    return "high_irradiance"


def build_campaign_evidence(
    pairs_root: Path,
    power: xr.Dataset,
    *,
    lane: str,
) -> xr.Dataset:
    """Materialise paired evidence rows from immutable bundles and observations.

    The product includes unevaluated future rows for provenance but all summary
    routines explicitly select ``EvaluationAvailable``.  No data from a
    delayed Cloudnet/HATPRO/radar product are used as a predictor here.
    """
    observed_soc = _power_series(power, "BatterySOC")
    observed_load = _observed_load_w(power)
    records: list[dict[str, object]] = []
    for manifest, bundle in completed_pair_bundles(pairs_root):
        try:
            with xr.open_zarr(bundle / "baseline_forecast.zarr", chunks={}) as opened:
                baseline = opened.load()
            with xr.open_zarr(bundle / "candidate_forecast.zarr", chunks={}) as opened:
                candidate = opened.load()
        except Exception:
            continue
        if "time" not in baseline or "time" not in candidate:
            continue
        times = pd.DatetimeIndex(candidate["time"].values)
        if not np.array_equal(times.to_numpy(dtype="datetime64[ns]"), np.asarray(baseline["time"].values)):
            continue
        issue = _as_utc_naive(candidate.attrs.get("initial_soc_time", times[0]))
        lead_hours = (times - issue) / pd.Timedelta(hours=1)
        candidate_soc = np.asarray(candidate.get("BatterySOCForecast", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        baseline_soc = np.asarray(baseline.get("BatterySOCForecast", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        candidate_load = np.asarray(candidate.get("ForecastLoadWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        baseline_load = np.asarray(baseline.get("ForecastLoadWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        candidate_solar = np.asarray(candidate.get("ForecastSolarWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        baseline_solar = np.asarray(baseline.get("ForecastSolarWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        ghi = np.asarray(candidate.get("ECMWFSolarIrradiance", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        observed_soc_values = observed_soc.reindex(
            times, method="nearest", tolerance=pd.Timedelta(minutes=10)
        ).to_numpy(dtype=np.float64)
        observed_load_values = observed_load.reindex(
            times, method="nearest", tolerance=pd.Timedelta(minutes=10)
        ).to_numpy(dtype=np.float64)
        initial_soc = float(candidate.attrs.get("initial_soc_pct", np.nan))
        for index, valid_time in enumerate(times):
            available = bool(np.isfinite(observed_soc_values[index]))
            records.append(
                {
                    "IssueTime": issue.to_datetime64(),
                    "ValidTime": valid_time.to_datetime64(),
                    "LeadHours": float(lead_hours[index]),
                    "SOCAuthoringAnchor": initial_soc,
                    "CandidateLane": str(lane),
                    "EvaluationPairID": str(manifest.get("evaluation_pair_id", "")),
                    "ForecastIdentityID": _pair_text(candidate, "forecast_identity_id"),
                    "ForecastSystemVersion": _pair_text(candidate, "forecast_system_version"),
                    "ForecastModelContractID": _pair_text(candidate, "forecast_model_contract_id"),
                    "SourceCycleSetID": _pair_text(candidate, "source_cycle_set_id"),
                    "LoadMode": _pair_text(candidate, "load_mode", "unknown"),
                    "CloudRegime": _irradiance_regime(float(ghi[index])),
                    "CloudRegimeMethod": "ecmwf_ghi_proxy_not_delayed_cloud_product",
                    "SourceAvailability": _pair_text(candidate, "ecmwf_provider_effective", "unknown"),
                    "DegradedModeCode": _pair_text(candidate, "degraded_mode_code", "none"),
                    "CandidateSOC": float(candidate_soc[index]),
                    "BaselineSOC": float(baseline_soc[index]),
                    "ObservedSOC": float(observed_soc_values[index]),
                    "CandidateLoadWatts": float(candidate_load[index]),
                    "BaselineLoadWatts": float(baseline_load[index]),
                    "ObservedLoadWatts": float(observed_load_values[index]),
                    "CandidateSolarWatts": float(candidate_solar[index]),
                    "BaselineSolarWatts": float(baseline_solar[index]),
                    "ECMWFGHI": float(ghi[index]),
                    "EvaluationAvailable": available,
                }
            )
    if not records:
        return xr.Dataset(
            coords={"record": np.array([], dtype=np.int64)},
            attrs={
                "power_campaign_evidence_product": "true",
                "candidate_lane": str(lane),
                "generated_at_utc": utc_now_iso(),
                "evidence_status": "no_complete_pair_bundles",
            },
        )
    columns = {name: [record[name] for record in records] for name in records[0]}
    data_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for name, values in columns.items():
        if isinstance(values[0], np.datetime64):
            array = np.asarray(values, dtype="datetime64[ns]")
        elif isinstance(values[0], (bool, np.bool_)):
            array = np.asarray(values, dtype=bool)
        elif isinstance(values[0], (float, np.floating)):
            array = np.asarray(values, dtype=np.float64)
        else:
            array = np.asarray([str(value) for value in values], dtype="U512")
        data_vars[name] = (("record",), array)
    return xr.Dataset(
        data_vars,
        coords={"record": np.arange(len(records), dtype=np.int64)},
        attrs={
            "power_campaign_evidence_product": "true",
            "candidate_lane": str(lane),
            "generated_at_utc": utc_now_iso(),
            "evidence_status": "complete_pair_bundles_materialised",
            "solar_metric_status": "excluded_until_mpp_active_observed_power_is_available",
            "ensemble_metric_status": "not_generated_in_bounded_initial_candidate",
            "reserve_event_status": "insufficient_events",
        },
    )


def _metric_summary(rows: pd.DataFrame) -> dict[str, float | int | str]:
    if rows.empty:
        return {"status": "insufficient_evidence", "samples": 0, "cycles": 0, "utc_days": 0}
    candidate_error = rows["CandidateSOC"].to_numpy(dtype=np.float64) - rows["ObservedSOC"].to_numpy(dtype=np.float64)
    baseline_error = rows["BaselineSOC"].to_numpy(dtype=np.float64) - rows["ObservedSOC"].to_numpy(dtype=np.float64)
    persistence_error = rows["SOCAuthoringAnchor"].to_numpy(dtype=np.float64) - rows["ObservedSOC"].to_numpy(dtype=np.float64)
    candidate_mae = float(np.mean(np.abs(candidate_error)))
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    persistence_mae = float(np.mean(np.abs(persistence_error)))
    return {
        "status": "evidence" if len(rows) >= 2 else "diagnostic_sparse",
        "samples": int(len(rows)),
        "cycles": int(rows["IssueTime"].nunique()),
        "utc_days": int(pd.DatetimeIndex(rows["IssueTime"]).floor("D").nunique()),
        "candidate_soc_mae": candidate_mae,
        "candidate_soc_bias": float(np.mean(candidate_error)),
        "baseline_soc_mae": baseline_mae,
        "baseline_soc_bias": float(np.mean(baseline_error)),
        "paired_mae_improvement_fraction": float(1.0 - candidate_mae / baseline_mae)
        if baseline_mae > 0.0
        else np.nan,
        "candidate_persistence_skill": float(1.0 - candidate_mae / persistence_mae)
        if persistence_mae > 0.0
        else np.nan,
    }


def campaign_score_surfaces(evidence: xr.Dataset) -> dict[str, object]:
    """Return cumulative campaign and last-24h diagnostic score surfaces."""
    if evidence.sizes.get("record", 0) == 0 or "EvaluationAvailable" not in evidence:
        empty = {bucket: _metric_summary(pd.DataFrame()) for bucket, _, _ in LEAD_BUCKETS}
        return {
            "generated_at_utc": utc_now_iso(),
            "campaign_evidence": {"lead_buckets": empty, "status": "insufficient_evidence"},
            "daily_diagnostic": {"lead_buckets": empty, "status": "insufficient_evidence"},
            "solar": "excluded_until_mpp_active_observed_power_is_available",
            "ensemble": "not_generated_in_bounded_initial_candidate",
            "reserve_events": "insufficient_events",
        }
    frame = evidence.to_dataframe().reset_index(drop=True)
    frame["ValidTime"] = pd.to_datetime(frame["ValidTime"])
    frame["IssueTime"] = pd.to_datetime(frame["IssueTime"])
    available = frame.loc[frame["EvaluationAvailable"].astype(bool)].copy()
    latest = available["ValidTime"].max() if not available.empty else pd.NaT

    def surface(selected: pd.DataFrame) -> dict[str, object]:
        buckets: dict[str, object] = {}
        for bucket, start, end in LEAD_BUCKETS:
            buckets[bucket] = _metric_summary(
                selected.loc[(selected["LeadHours"] >= start) & (selected["LeadHours"] < end)]
            )
        strata: dict[str, dict[str, int | str]] = {}
        for field in ("LoadMode", "CloudRegime", "SourceAvailability", "DegradedModeCode"):
            if field not in selected:
                continue
            for value, group in selected.groupby(field, dropna=False):
                cycles = int(group["IssueTime"].nunique())
                strata[f"{field}:{value}"] = {
                    "samples": int(len(group)),
                    "cycles": cycles,
                    "status": "diagnostic_sparse" if cycles < 30 else "evidence",
                }
        return {"lead_buckets": buckets, "strata": strata, "status": "evidence" if len(selected) else "insufficient_evidence"}

    daily = (
        available.loc[available["ValidTime"] > latest - pd.Timedelta(hours=24)]
        if not pd.isna(latest)
        else available.iloc[0:0]
    )
    return {
        "generated_at_utc": utc_now_iso(),
        "campaign_evidence": surface(available),
        "daily_diagnostic": surface(daily),
        "solar": "excluded_until_mpp_active_observed_power_is_available",
        "ensemble": "not_generated_in_bounded_initial_candidate",
        "reserve_events": "insufficient_events",
    }
