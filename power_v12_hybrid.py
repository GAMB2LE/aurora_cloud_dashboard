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
from power_observation_truth import (
    TRUTH_CONTRACT_VERSION, electrical_observation_series, evaluate_power_observations,
    forecast_interval_starts, prepare_observation_truth,
)
from power_implementation_identity import semantic_code_key


V12_FORECAST_SYSTEM_VERSION = "power-v12-hybrid-candidate"
V12_FEATURE_SET_VERSION = "issue_safe_physical_pv_bounded_load_residual_v5"
V12_POWER_HISTORY_DAYS = 21.0
LOAD_RESIDUAL_MODEL_NAME = "bounded_ridge_interval_state_load_residual_v2"
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
EVALUATION_CONTRACT_ATTRS = (
    "forecast_model_contract_id",
    "forecast_system_version",
    "feature_set_version",
    "feature_set_digest",
    "forecast_code_revision",
    "candidate_lane",
    "baseline_control_contract_id",
    "baseline_control_system_version",
    "local_feature_contract_id",
)
SOLAR_MPP_MODE_FIELDS = (
    "SolarMPPMode_East",
    "SolarMPPMode_South",
    "SolarMPPMode_West",
)
MPP_ACTIVE_MODE = 2.0
PAIR_ARTIFACT_DIGEST_ALGORITHM = "sha256-relative-path-nul-content-nul-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def immutable_artifact_record(path: Path, *, relative_path: str) -> dict[str, object]:
    """Return a deterministic, symlink-free content record for one artifact."""

    root = Path(path)
    if not root.exists() or root.is_symlink():
        raise ValueError(f"Immutable pair artifact is missing or symbolic: {root}")
    if root.is_file():
        files = [root]
    else:
        descendants = list(root.rglob("*"))
        if any(candidate.is_symlink() for candidate in descendants):
            raise ValueError(f"Immutable pair artifact contains a symbolic link: {root}")
        files = sorted(
            (candidate for candidate in descendants if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
    if not files:
        raise ValueError(f"Immutable pair artifact is empty: {root}")
    digest = hashlib.sha256()
    byte_count = 0
    for candidate in files:
        if candidate.is_symlink():
            raise ValueError(f"Immutable pair artifact contains a symbolic link: {candidate}")
        relative = candidate.name if root.is_file() else candidate.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                byte_count += len(block)
                digest.update(block)
        digest.update(b"\0")
    return {
        "relative_path": str(relative_path),
        "sha256": f"sha256:{digest.hexdigest()}",
        "file_count": len(files),
        "byte_count": byte_count,
    }


def pair_artifacts_valid(
    manifest: Mapping[str, object],
    bundle: Path,
    required: Mapping[str, str],
) -> bool:
    """Fail closed unless every named immutable pair artifact still matches."""

    if manifest.get("artifact_digest_algorithm") != PAIR_ARTIFACT_DIGEST_ALGORITHM:
        return False
    checksums = manifest.get("artifact_checksums")
    if not isinstance(checksums, Mapping):
        return False
    bundle = Path(bundle)
    if bundle.is_symlink():
        return False
    try:
        bundle_root = bundle.resolve(strict=True)
    except OSError:
        return False
    for logical_name, relative_path in required.items():
        record = checksums.get(logical_name)
        if not isinstance(record, Mapping):
            return False
        if str(record.get("relative_path") or "") != relative_path:
            return False
        target = bundle / relative_path
        try:
            resolved = target.resolve(strict=True)
            if resolved.parent != bundle_root:
                return False
            actual = immutable_artifact_record(target, relative_path=relative_path)
        except (OSError, ValueError):
            return False
        for field in ("sha256", "file_count", "byte_count"):
            if record.get(field) != actual[field]:
                return False
    return True


def _as_utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _observed_load_w(power: xr.Dataset) -> pd.Series:
    """Total station load; partial AC/inverter fallback is diagnostic only."""
    return electrical_observation_series(power).load_w


def _observed_solar_w(power: xr.Dataset) -> pd.Series:
    return electrical_observation_series(power).solar_delivered_w


def _mpp_active_available_power_mask(power: xr.Dataset) -> pd.Series:
    """Return valid available-PV truth targets from Victron mode 791.

    Every array must report mode 2 (MPPT active).  A missing or limited charger
    means delivered watts could be battery-limited and is excluded rather than
    masquerading as a solar-model miss.
    """
    if "time" not in power.coords:
        return pd.Series(dtype=bool)
    index = pd.DatetimeIndex(power["time"].values)
    if not all(
        name in power and power[name].dims == ("time",) for name in SOLAR_MPP_MODE_FIELDS
    ):
        return pd.Series(False, index=index, dtype=bool)
    values = np.column_stack(
        [np.asarray(power[name].values, dtype=np.float64) for name in SOLAR_MPP_MODE_FIELDS]
    )
    return pd.Series(
        np.all(values == MPP_ACTIVE_MODE, axis=1),
        index=index,
    )


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
    if archive[name].dims == ("issue_time", "forecast_step"):
        if len(values) != length:
            raise ValueError(f"Archived {name} has an inconsistent forecast grid")
        return values
    return np.repeat(values, steps)


def _normalise_identity_values(values: np.ndarray, *, fallback: str) -> np.ndarray:
    """Normalise archive text fields without treating NaN as an identity."""
    normalised: list[str] = []
    for value in np.asarray(values).reshape(-1):
        text = str(value if value is not None else "").strip()
        normalised.append(fallback if text.lower() in {"", "nan", "none"} else text)
    return np.asarray(normalised, dtype=object)


def _load_training_rows(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    cutoff: pd.Timestamp,
    load_mode: str,
    control_forecast_model_contract_id: str | None,
    control_forecast_system_version: str | None,
    operating_state: xr.Dataset | None = None,
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
    starts = pd.DatetimeIndex(np.concatenate([
        forecast_interval_starts(archive["ForecastValidTime"].values[index], issue).values
        for index, issue in enumerate(archive["issue_time"].values)
    ]))
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
    )
    system_versions = _repeat_issue_field(
        archive,
        "ForecastSystemVersion",
        len(values),
        default="unversioned_control",
    )
    model_contracts = _normalise_identity_values(model_contracts, fallback="unqualified_control")
    system_versions = _normalise_identity_values(system_versions, fallback="unversioned_control")
    required_contract = str(control_forecast_model_contract_id or "").strip()
    required_system = str(control_forecast_system_version or "").strip() or "unversioned_control"
    # A residual is adaptive learning.  Rows with an opaque or changed
    # baseline identity are not comparable training evidence.
    if not required_contract:
        return pd.DataFrame()
    mask = (
        np.isfinite(values)
        & np.isfinite(lead_hours)
        & ~valid_times.isna()
        & ~issue_times.isna()
        & (issue_times < cutoff)
        & (valid_times <= cutoff)
        & (modes == str(load_mode))
        & (model_contracts == required_contract)
        & (system_versions == required_system)
    )
    for flag in ("IndependentCycle", "ForecastVerificationEligible"):
        if flag in archive:
            mask &= _repeat_issue_field(archive, flag, len(values), default=False).astype(bool)
    if not np.any(mask):
        return pd.DataFrame()
    valid_times = valid_times[mask]
    history = power.sel(time=slice(None, cutoff))
    state_history = operating_state.sel(time=slice(None, cutoff)) if operating_state is not None and "time" in operating_state.coords else operating_state
    prepared = prepare_observation_truth(history, state_history)
    truth = evaluate_power_observations(history, starts[mask], valid_times,
        expected_modes=modes[mask], prepared=prepared)
    observed_values = truth.load_w.to_numpy(dtype=float)
    paired = np.isfinite(observed_values)
    state_status = truth.operating_state_status
    paired &= state_status.eq("unchanged").to_numpy()
    current_state = evaluate_power_observations(history, [pd.NaT], [cutoff],
        expected_modes=[load_mode], prepared=prepared)
    current_exact = str(current_state.realized_exact_state.iloc[0])
    current_known = current_state.operating_state_status.iloc[0] == "unchanged"
    exact_matches = truth.realized_exact_state.eq(current_exact).to_numpy() & current_known
    # When a composed state (tier/charging/heater) exists, it is a conditioning
    # key, not just display text. Same-mode evidence from another phase cannot
    # tune the current phase's correction. Raw PDU evidence stays labelled as
    # direct PDU scope, never silently asserted to include instrument phases.
    paired &= exact_matches
    if "ForecastOperatingLoadState" in archive:
        archived_exact = _repeat_issue_field(archive, "ForecastOperatingLoadState", len(values), default="")[mask].astype(str)
        paired &= (archived_exact == truth.realized_exact_state.to_numpy()) & (archived_exact != "")
    exclusions = {
        "missing_total_interval_load": int(np.count_nonzero(~np.isfinite(observed_values))),
        "realized_state": {str(name): int(count) for name, count in state_status.value_counts().items()},
        "current_exact_state_unavailable": not bool(current_known),
        "different_current_exact_state": int(np.count_nonzero(~exact_matches)),
        "accepted": int(np.count_nonzero(paired)),
    }
    if not np.any(paired):
        empty = pd.DataFrame()
        empty.attrs["training_exclusions"] = exclusions
        return empty
    rows = pd.DataFrame(
        {
            "issue_time": issue_times[mask][paired],
            "cycle_time": cycles[mask][paired],
            "valid_time": valid_times[paired],
            "lead_hour": lead_hours[mask][paired],
            "forecast_load_w": values[mask][paired],
            "observed_load_w": observed_values[paired],
            "forecast_model_contract_id": model_contracts[mask][paired],
            "forecast_system_version": system_versions[mask][paired],
            "realized_state_status": state_status.to_numpy()[paired],
            "realized_exact_state": truth.realized_exact_state.to_numpy()[paired],
            "realized_state_scope": truth.operating_state_scope.to_numpy()[paired],
            "observation_coverage": truth.load_coverage.to_numpy()[paired],
        }
    )
    rows["residual_w"] = rows["observed_load_w"] - rows["forecast_load_w"]
    # Cached re-anchors share forcing and are not independent training records.
    result = (
        rows.sort_values("issue_time")
        .drop_duplicates(["cycle_time", "valid_time"], keep="last")
        .sort_values(["valid_time", "issue_time"])
        .reset_index(drop=True)
    )
    result.attrs["training_exclusions"] = exclusions
    return result


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
    uncertainty_status: str = "not_calibrated"
    uncertainty_samples: int = 0
    training_state_status: str = "metadata_unavailable"
    training_exclusions_json: str = "{}"

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
            "uncertainty_status": self.uncertainty_status,
            "uncertainty_samples": self.uncertainty_samples,
            "training_state_status": self.training_state_status,
            "training_exclusions_json": self.training_exclusions_json,
            "observation_truth_contract": TRUTH_CONTRACT_VERSION,
        }


def fit_bounded_load_residual(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    issue_time: pd.Timestamp | str,
    forecast_times: Iterable[object],
    load_mode: str,
    control_forecast_model_contract_id: str | None = None,
    control_forecast_system_version: str | None = None,
    bound_w: float = LOAD_RESIDUAL_BOUND_W,
    operating_state: xr.Dataset | None = None,
) -> LoadResidualFit:
    """Fit a small ridge residual model using only data available at issue time.

    It is intentionally fail-closed: insufficient same-mode independent
    evidence produces an explicit zero correction rather than borrowing a
    different operating state or future observation.
    """
    cutoff = _as_utc_naive(issue_time)
    times = pd.DatetimeIndex(forecast_times)
    rows = _load_training_rows(
        archive,
        power,
        cutoff=cutoff,
        load_mode=load_mode,
        control_forecast_model_contract_id=control_forecast_model_contract_id,
        control_forecast_system_version=control_forecast_system_version,
        operating_state=operating_state,
    )
    samples = int(len(rows))
    cycles = int(rows["cycle_time"].nunique()) if not rows.empty else 0
    days = int(rows["valid_time"].dt.floor("D").nunique()) if not rows.empty else 0
    zero = pd.Series(np.zeros(len(times), dtype=np.float64), index=times)
    payload = {
        "schema": 1,
        "name": LOAD_RESIDUAL_MODEL_NAME,
        "observation_truth_contract": TRUTH_CONTRACT_VERSION,
        "realized_state_policy": "require_direct_match_conditioned_on_current_exact_state",
        "uncertainty_policy": "latest_utc_day_blocked_holdout_v1",
        "bound_w": float(bound_w),
        "feature_columns": ["intercept", "lead", "lead_squared", "utc_hour_sin", "utc_hour_cos"],
        # The current operating mode and learned coefficients are adaptive
        # issue state, not changes to the residual-model algorithm. Keeping
        # them out of this contract allows one campaign cohort to contain the
        # multiple operating states required by the promotion gate.
        "load_mode_conditioning": "exact_issue_load_mode",
        "control_forecast_model_contract_id": str(control_forecast_model_contract_id or ""),
        "control_forecast_system_version": str(
            control_forecast_system_version or "unversioned_control"
        ),
    }
    contract_id = "load-residual-v2-" + stable_json_digest(payload)[:16]
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
            training_exclusions_json=json.dumps(rows.attrs.get("training_exclusions", {}), sort_keys=True),
        )

    X = _feature_matrix(pd.DatetimeIndex(rows["valid_time"]), rows["lead_hour"].to_numpy())
    y = rows["residual_w"].to_numpy(dtype=np.float64)
    utc_days = rows.valid_time.dt.floor("D")
    holdout = (utc_days == utc_days.max()).to_numpy()
    prior = ~holdout
    tuning_day = utc_days.loc[prior].max()
    tune = (utc_days == tuning_day).to_numpy()
    train = (utc_days < tuning_day).to_numpy()
    alphas = (0.1, 1.0, 10.0, 100.0)
    if np.count_nonzero(train) >= 12 and np.count_nonzero(tune) >= 12:
        validation = []
        for alpha in alphas:
            coefficients = _fit_ridge(X[train], y[train], alpha)
            validation.append((float(np.mean(np.abs((X[tune] @ coefficients) - y[tune]))), alpha))
        _, selected_alpha = min(validation, key=lambda item: (item[0], item[1]))
        selection = f"blocked_rolling_origin_alpha={selected_alpha:g}"
    else:
        selected_alpha = 10.0
        selection = "insufficient_holdout_default_alpha=10"
    if np.count_nonzero(prior) >= 24 and np.count_nonzero(holdout) >= 12:
        heldout_coefficients = _fit_ridge(X[prior], y[prior], selected_alpha)
        residual_noise = y[holdout] - X[holdout] @ heldout_coefficients
        lower_noise, upper_noise = np.nanquantile(residual_noise, (0.10, 0.90))
        lower_noise, upper_noise = min(lower_noise, 0.0), max(upper_noise, 0.0)
        uncertainty_status, uncertainty_samples = "blocked_holdout", int(np.count_nonzero(holdout))
    else:
        lower_noise = upper_noise = 0.0
        uncertainty_status, uncertainty_samples = "insufficient_blocked_holdout", 0
    coefficients = _fit_ridge(X, y, selected_alpha)
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
        uncertainty_status,
        uncertainty_samples,
        "+".join(sorted(set(rows.realized_state_scope.astype(str)))),
        json.dumps(rows.attrs.get("training_exclusions", {}), sort_keys=True),
    )


def v12_feature_digest(
    *,
    physical_config_digest: str,
    load_residual_contract_id: str,
    issue_feature_contract_id: str = "",
    power_history_days: float = V12_POWER_HISTORY_DAYS,
) -> str:
    return stable_json_digest(
        {
            "schema": 1,
            "feature_set_version": V12_FEATURE_SET_VERSION,
            "physical_config_digest": str(physical_config_digest),
            "load_residual_contract_id": str(load_residual_contract_id),
            "issue_feature_contract_id": str(issue_feature_contract_id),
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
    issue_feature_contract_id: str = "",
    baseline_control_contract_id: str = "",
    baseline_control_system_version: str = "unversioned_control",
    source_availability_code: str = "",
    feature_degradation_codes: Iterable[str] = (),
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
    degraded.extend(str(value) for value in feature_degradation_codes if str(value))
    return {
        "forecast_model_name": "aps_soc_energy_balance_v12_hybrid_candidate",
        "forecast_model_version": "12",
        "forecast_model_status": "candidate",
        "forecast_system_version": V12_FORECAST_SYSTEM_VERSION,
        "feature_set_version": V12_FEATURE_SET_VERSION,
        "feature_set_digest": v12_feature_digest(
            physical_config_digest=physical_config_digest,
            load_residual_contract_id=residual.contract_id,
            issue_feature_contract_id=issue_feature_contract_id,
            power_history_days=power_history_days,
        ),
        "training_cutoff_utc": _as_utc_naive(issue_time).isoformat(),
        "observation_cutoff_utc": _as_utc_naive(issue_time).isoformat(),
        "soc_anchor_time_utc": _as_utc_naive(issue_time).isoformat(),
        "forecast_code_revision": str(code_revision),
        "source_cycle_set_id": str(source_cycle_set_id),
        "source_manifest_digest": str(source_manifest_digest),
        "degraded_mode_code": "+".join(degraded),
        "candidate_lane": str(lane),
        "local_feature_contract_id": str(issue_feature_contract_id),
        "baseline_control_contract_id": str(baseline_control_contract_id),
        "baseline_control_system_version": str(baseline_control_system_version),
        "source_availability_code": str(source_availability_code),
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
            if not pair_artifacts_valid(
                manifest,
                bundle,
                {
                    "baseline": "baseline_forecast.zarr",
                    "candidate": "candidate_forecast.zarr",
                },
            ):
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


def evaluation_contract_from_forecast(forecast: xr.Dataset) -> dict[str, str]:
    """Return the semantic contract that defines one comparable campaign.

    Source-cycle, observation-cutoff and input-snapshot values intentionally
    remain issue identities, not campaign partitions.  Any algorithm, feature,
    code, baseline-control or local-feature-contract change creates a new
    campaign cohort.
    """
    contract = {name: _pair_text(forecast, name) for name in EVALUATION_CONTRACT_ATTRS}
    if _pair_text(forecast, "forecast_implementation_digest"):
        contract["forecast_code_revision"] = semantic_code_key(forecast.attrs)
    return contract


def _matches_evaluation_contract(
    forecast: xr.Dataset,
    contract: Mapping[str, object] | None,
) -> bool:
    if contract is None:
        return True
    actual = evaluation_contract_from_forecast(forecast)
    for name in EVALUATION_CONTRACT_ATTRS:
        expected = str(contract.get(name, "") or "")
        if actual[name] != expected:
            return False
    return True


def _irradiance_regime(value: float) -> str:
    if not np.isfinite(value) or value <= 1.0:
        return "dark"
    if value < 100.0:
        return "low_irradiance"
    if value < 350.0:
        return "moderate_irradiance"
    return "high_irradiance"


def _clearness_cloud_regime(ghi: float, clearness_index: float) -> str:
    """Classify cloud only when an issue-time clearness index is available."""

    if not np.isfinite(ghi) or ghi <= 1.0:
        return "dark"
    if not np.isfinite(clearness_index):
        return "unknown"
    if clearness_index >= 0.65:
        return "clear"
    if clearness_index <= 0.35:
        return "cloudy"
    return "transitional"


def build_campaign_evidence(
    pairs_root: Path,
    power: xr.Dataset,
    *,
    lane: str,
    evaluation_contract: Mapping[str, object] | None = None,
    operating_state: xr.Dataset | None = None,
) -> xr.Dataset:
    """Materialise paired evidence rows from immutable bundles and observations.

    The product includes unevaluated future rows for provenance but all summary
    routines explicitly select ``EvaluationAvailable``.  No data from a
    delayed Cloudnet/HATPRO/radar product are used as a predictor here.
    """
    prepared_truth = prepare_observation_truth(power, operating_state)
    records: list[dict[str, object]] = []
    incompatible_pair_count = 0
    for manifest, bundle in completed_pair_bundles(pairs_root):
        try:
            with xr.open_zarr(bundle / "baseline_forecast.zarr", chunks={}) as opened:
                baseline = opened.load()
            with xr.open_zarr(bundle / "candidate_forecast.zarr", chunks={}) as opened:
                candidate = opened.load()
        except Exception:
            continue
        if not _matches_evaluation_contract(candidate, evaluation_contract):
            incompatible_pair_count += 1
            continue
        if "time" not in baseline or "time" not in candidate:
            continue
        times = pd.DatetimeIndex(candidate["time"].values)
        if not np.array_equal(times.to_numpy(dtype="datetime64[ns]"), np.asarray(baseline["time"].values)):
            continue
        issue = _as_utc_naive(candidate.attrs.get("initial_soc_time", times[0]))
        cycle_time = _as_utc_naive(candidate.attrs.get("ecmwf_cycle_time", issue.floor("3h")))
        source_cycle_id = _pair_text(candidate, "source_cycle_set_id").strip()
        if not source_cycle_id:
            source_cycle_id = "ecmwf-cycle:" + cycle_time.isoformat()
        lead_hours = (times - issue) / pd.Timedelta(hours=1)
        candidate_soc = np.asarray(candidate.get("BatterySOCForecast", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        baseline_soc = np.asarray(baseline.get("BatterySOCForecast", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        candidate_load = np.asarray(candidate.get("ForecastLoadWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        baseline_load = np.asarray(baseline.get("ForecastLoadWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        candidate_solar = np.asarray(candidate.get("ForecastSolarWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        baseline_solar = np.asarray(baseline.get("ForecastSolarWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=np.float64)
        forcing_name = (
            "ForecastInputGlobalHorizontalIrradiance"
            if "ForecastInputGlobalHorizontalIrradiance" in candidate
            else "ECMWFSolarIrradiance"
        )
        ghi = np.asarray(
            candidate.get(forcing_name, xr.DataArray(np.full(len(times), np.nan))).values,
            dtype=np.float64,
        )
        clearness = np.asarray(
            candidate.get(
                "ECMWFClearnessIndex",
                xr.DataArray(np.full(len(times), np.nan)),
            ).values,
            dtype=np.float64,
        )
        cloud_regime_method = (
            "issue_time_ecmwf_clearness_index_thresholds_v1"
            if "ECMWFClearnessIndex" in candidate
            else "unavailable_no_issue_time_clearness_index"
        )
        starts = forecast_interval_starts(times, issue)
        truth = evaluate_power_observations(power, starts, times,
            expected_modes=[_pair_text(candidate, "load_mode", "unknown")] * len(times),
            prepared=prepared_truth)
        observed_soc_values = truth.soc.to_numpy(dtype=float)
        observed_load_values = truth.load_w.to_numpy(dtype=float)
        candidate_available_pv = "available" in str(
            candidate.attrs.get("solar_power_semantics", "")
        ).lower()
        baseline_available_pv = "available" in str(baseline.attrs.get("solar_power_semantics", "")).lower()
        observed_solar_values = truth[
            "solar_available_w" if candidate_available_pv or baseline_available_pv else "solar_delivered_w"
        ].to_numpy(dtype=float)
        candidate_delivered = np.asarray(candidate.get("ForecastPVDeliveredWatts",
            xr.DataArray(np.full(len(times), np.nan) if candidate_available_pv else candidate_solar)).values, dtype=float)
        baseline_delivered = np.asarray(baseline.get("ForecastPVDeliveredWatts",
            xr.DataArray(np.full(len(times), np.nan) if baseline_available_pv else baseline_solar)).values, dtype=float)
        initial_soc = float(candidate.attrs.get("initial_soc_pct", np.nan))
        for index, valid_time in enumerate(times):
            available = bool(np.isfinite(observed_soc_values[index]))
            records.append(
                {
                    "IssueTime": issue.to_datetime64(),
                    "ValidTime": valid_time.to_datetime64(),
                    "IntervalStartTime": starts[index].to_datetime64(),
                    "LeadHours": float(lead_hours[index]),
                    "SOCAuthoringAnchor": initial_soc,
                    "CandidateLane": str(lane),
                    "EvaluationPairID": str(manifest.get("evaluation_pair_id", "")),
                    "ForecastIdentityID": _pair_text(candidate, "forecast_identity_id"),
                    "ForecastSystemVersion": _pair_text(candidate, "forecast_system_version"),
                    "ForecastModelContractID": _pair_text(candidate, "forecast_model_contract_id"),
                    "BaselineControlContractID": _pair_text(candidate, "baseline_control_contract_id"),
                    "BaselineControlSystemVersion": _pair_text(candidate, "baseline_control_system_version"),
                    "LocalFeatureContractID": _pair_text(candidate, "local_feature_contract_id"),
                    "SourceCycleSetID": source_cycle_id,
                    "ECMWFCycleTime": cycle_time.to_datetime64(),
                    "LoadMode": _pair_text(candidate, "load_mode", "unknown"),
                    "RealizedLoadMode": str(truth.realized_mode.iloc[index]),
                    "RealizedExactState": str(truth.realized_exact_state.iloc[index]),
                    "RealizedStateStatus": str(truth.operating_state_status.iloc[index]),
                    "RealizedStateScope": str(truth.operating_state_scope.iloc[index]),
                    "CloudRegime": _clearness_cloud_regime(
                        float(ghi[index]), float(clearness[index])
                    ),
                    "CloudRegimeMethod": cloud_regime_method,
                    "ECMWFClearnessIndex": float(clearness[index]),
                    "SourceAvailability": _pair_text(
                        candidate,
                        "source_availability_code",
                        _pair_text(candidate, "ecmwf_provider_effective", "unknown"),
                    ),
                    "DegradedModeCode": _pair_text(candidate, "degraded_mode_code", "none"),
                    "CandidateSOC": float(candidate_soc[index]),
                    "BaselineSOC": float(baseline_soc[index]),
                    "ObservedSOC": float(observed_soc_values[index]),
                    "CandidateLoadWatts": float(candidate_load[index]),
                    "BaselineLoadWatts": float(baseline_load[index]),
                    "ObservedLoadWatts": float(observed_load_values[index]),
                    "ObservedLoadEnergyWh": float(truth.load_energy_wh.iloc[index]),
                    "ObservedLoadCoverage": float(truth.load_coverage.iloc[index]),
                    "ObservedLoadStatus": str(truth.load_status.iloc[index]),
                    "ObservedLoadMeasurement": str(truth.load_measurement.iloc[index]),
                    "ObservedLoadFallbackWatts": float(truth.load_fallback_w.iloc[index]),
                    "CandidateSolarWatts": float(candidate_solar[index]),
                    "BaselineSolarWatts": float(baseline_solar[index]),
                    "ObservedSolarWatts": float(observed_solar_values[index]),
                    "ObservedSolarDeliveredWatts": float(truth.solar_delivered_w.iloc[index]),
                    "ObservedSolarAvailableWatts": float(truth.solar_available_w.iloc[index]),
                    "ObservedSolarDeliveredEnergyWh": float(truth.solar_delivered_energy_wh.iloc[index]),
                    "ObservedSolarAvailableEnergyWh": float(truth.solar_available_energy_wh.iloc[index]),
                    "ObservedSolarCoverage": float(truth.solar_delivered_coverage.iloc[index]),
                    "ObservedSolarAvailableCoverage": float(truth.solar_available_coverage.iloc[index]),
                    "ObservedSolarStatus": str(truth.solar_available_status.iloc[index]),
                    "CandidateSolarDeliveredWatts": float(candidate_delivered[index]),
                    "BaselineSolarDeliveredWatts": float(baseline_delivered[index]),
                    "SolarEvaluationAvailable": bool(np.isfinite(observed_solar_values[index])),
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
                "evaluation_contract": json.dumps(dict(evaluation_contract or {}), sort_keys=True),
                "incompatible_pair_count": int(incompatible_pair_count),
            },
        )
    # Retries and cached re-anchors remain auditable operational samples, but
    # only the latest exact pair for each source-cycle/valid-time key may
    # contribute to independent campaign scores or promotion gates.
    records.sort(
        key=lambda value: (
            np.datetime64(value["IssueTime"]),
            np.datetime64(value["ValidTime"]),
            str(value["EvaluationPairID"]),
        )
    )
    selected_by_cycle_valid: dict[tuple[str, int], int] = {}
    for index, record in enumerate(records):
        valid_ns = int(np.datetime64(record["ValidTime"], "ns").astype(np.int64))
        selected_by_cycle_valid[(str(record["SourceCycleSetID"]), valid_ns)] = index
    independent_indices = set(selected_by_cycle_valid.values())
    for index, record in enumerate(records):
        record["IndependentEvaluationSample"] = index in independent_indices
        record["SOCAnchorTime"] = record["IssueTime"]
    duplicate_pair_rows = len(records) - len(independent_indices)
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
            "observation_truth_contract": TRUTH_CONTRACT_VERSION,
            "evaluation_contract": json.dumps(dict(evaluation_contract or {}), sort_keys=True),
            "incompatible_pair_count": int(incompatible_pair_count),
            "duplicate_cycle_valid_rows_discarded": 0,
            "duplicate_cycle_valid_rows_retained": int(duplicate_pair_rows),
            "solar_metric_status": "mpp_active_available_power_only",
            "ensemble_metric_status": "not_generated_in_bounded_initial_candidate",
            "reserve_event_status": "insufficient_events",
        },
    )


def _independent_cycle_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    if "SourceCycleSetID" in rows:
        values = rows["SourceCycleSetID"].astype(str).str.strip()
        values = values.loc[~values.str.lower().isin({"", "nan", "none"})]
        if not values.empty:
            return int(values.nunique())
    return int(rows["IssueTime"].nunique()) if "IssueTime" in rows else 0


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
        "cycles": _independent_cycle_count(rows),
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


def _solar_metric_summary(rows: pd.DataFrame) -> dict[str, float | int | str]:
    """Compare forecast solar only against uncensored available-PV samples."""
    if rows.empty or "SolarEvaluationAvailable" not in rows:
        return {
            "status": "insufficient_mpp_active_available_power_evidence",
            "samples": 0,
            "cycles": 0,
        }
    eligible = rows.loc[rows["SolarEvaluationAvailable"].astype(bool)].copy()
    metrics = _error_metrics(
        eligible,
        candidate="CandidateSolarWatts",
        baseline="BaselineSolarWatts",
        observed="ObservedSolarWatts",
    )
    if metrics is None:
        return {
            "status": "insufficient_mpp_active_available_power_evidence",
            "samples": int(len(eligible)),
            "cycles": _independent_cycle_count(eligible),
        }
    return {
        **metrics,
        "cycles": _independent_cycle_count(eligible),
        "status": "evidence" if len(eligible) >= 2 else "diagnostic_sparse",
    }


def campaign_score_surfaces(evidence: xr.Dataset) -> dict[str, object]:
    """Return cumulative campaign and last-24h diagnostic score surfaces."""
    try:
        evaluation_contract = json.loads(str(evidence.attrs.get("evaluation_contract", "{}")))
    except json.JSONDecodeError:
        evaluation_contract = {}
    incompatible_pair_count = int(evidence.attrs.get("incompatible_pair_count", 0) or 0)
    if evidence.sizes.get("record", 0) == 0 or "EvaluationAvailable" not in evidence:
        empty = {bucket: _metric_summary(pd.DataFrame()) for bucket, _, _ in LEAD_BUCKETS}
        return {
            "generated_at_utc": utc_now_iso(),
            "campaign_evidence": {
                "lead_buckets": empty,
                "solar": _solar_metric_summary(pd.DataFrame()),
                "status": "insufficient_evidence",
            },
            "daily_diagnostic": {
                "lead_buckets": empty,
                "solar": _solar_metric_summary(pd.DataFrame()),
                "status": "insufficient_evidence",
            },
            "solar": "mpp_active_available_power_only",
            "ensemble": "not_generated_in_bounded_initial_candidate",
            "reserve_events": "insufficient_events",
            "evaluation_contract": evaluation_contract,
            "incompatible_pair_count": incompatible_pair_count,
        }
    frame = evidence.to_dataframe().reset_index(drop=True)
    frame["ValidTime"] = pd.to_datetime(frame["ValidTime"])
    frame["IssueTime"] = pd.to_datetime(frame["IssueTime"])
    available = frame.loc[frame["EvaluationAvailable"].astype(bool)].copy()
    independent = (
        available.loc[available["IndependentEvaluationSample"].astype(bool)].copy()
        if "IndependentEvaluationSample" in available
        else available
    )
    latest = available["ValidTime"].max() if not available.empty else pd.NaT

    def surface(selected: pd.DataFrame) -> dict[str, object]:
        buckets: dict[str, object] = {}
        for bucket, start, end in LEAD_BUCKETS:
            buckets[bucket] = _metric_summary(
                selected.loc[(selected["LeadHours"] >= start) & (selected["LeadHours"] < end)]
            )
        strata: dict[str, dict[str, int | str]] = {}
        for field in ("LoadMode", "RealizedStateStatus", "RealizedExactState", "CloudRegime", "SourceAvailability", "DegradedModeCode"):
            if field not in selected:
                continue
            for value, group in selected.groupby(field, dropna=False):
                cycles = _independent_cycle_count(group)
                strata[f"{field}:{value}"] = {
                    "samples": int(len(group)),
                    "cycles": cycles,
                    "status": "diagnostic_sparse" if cycles < 30 else "evidence",
                    "soc": _metric_summary(group),
                    "load": _error_metrics(group, candidate="CandidateLoadWatts", baseline="BaselineLoadWatts", observed="ObservedLoadWatts"),
                }
        return {
            "lead_buckets": buckets,
            "strata": strata,
            "solar": _solar_metric_summary(selected),
            "solar_delivered": _error_metrics(selected, candidate="CandidateSolarDeliveredWatts", baseline="BaselineSolarDeliveredWatts", observed="ObservedSolarDeliveredWatts"),
            "load": _error_metrics(selected, candidate="CandidateLoadWatts", baseline="BaselineLoadWatts", observed="ObservedLoadWatts"),
            "truth_exclusions": {
                field: {str(value): int(count) for value, count in selected[field].value_counts(dropna=False).items()}
                for field in ("ObservedLoadStatus", "ObservedLoadMeasurement", "ObservedSolarStatus", "RealizedStateStatus")
                if field in selected
            },
            "status": "evidence" if len(selected) else "insufficient_evidence",
        }

    daily = (
        available.loc[available["ValidTime"] > latest - pd.Timedelta(hours=24)]
        if not pd.isna(latest)
        else available.iloc[0:0]
    )
    return {
        "generated_at_utc": utc_now_iso(),
        "campaign_evidence": surface(independent),
        "daily_diagnostic": surface(daily),
        "solar": "mpp_active_available_power_only",
        "ensemble": "not_generated_in_bounded_initial_candidate",
        "reserve_events": "insufficient_events",
        "evaluation_contract": evaluation_contract,
        "incompatible_pair_count": incompatible_pair_count,
    }


def _error_metrics(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    observed: str,
) -> dict[str, float] | None:
    required = (candidate, baseline, observed)
    if any(name not in frame for name in required):
        return None
    values = frame.loc[:, list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if values.empty:
        return None
    candidate_error = values[candidate].to_numpy(dtype=np.float64) - values[observed].to_numpy(dtype=np.float64)
    baseline_error = values[baseline].to_numpy(dtype=np.float64) - values[observed].to_numpy(dtype=np.float64)
    candidate_mae = float(np.mean(np.abs(candidate_error)))
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    return {
        "samples": float(len(values)),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "candidate_bias": float(np.mean(candidate_error)),
        "baseline_bias": float(np.mean(baseline_error)),
        "mae_improvement_fraction": (
            float(1.0 - candidate_mae / baseline_mae) if baseline_mae > 0.0 else 0.0
        ),
    }


def _promotion_diversity(rows: pd.DataFrame) -> dict[str, object]:
    """Return fail-closed weather/load diversity for one lead bucket."""

    regimes = set()
    if "CloudRegime" in rows:
        regimes = {
            str(value).strip().lower()
            for value in rows["CloudRegime"]
            if str(value).strip().lower() not in {"", "nan", "none", "unknown"}
        }
    modes = set()
    if "LoadMode" in rows:
        modes = {
            str(value).strip()
            for value in rows["LoadMode"]
            if str(value).strip().lower()
            not in {"", "nan", "none", "unknown", "unavailable", "unspecified"}
        }
    clear_covered = "clear" in regimes
    cloudy_covered = "cloudy" in regimes
    multiple_modes = len({value.lower() for value in modes}) >= 2
    eligible = clear_covered and cloudy_covered and multiple_modes
    return {
        "status": "eligible" if eligible else "insufficient_diversity",
        "eligible": eligible,
        "clear_covered": clear_covered,
        "cloudy_covered": cloudy_covered,
        "cloud_regimes": sorted(regimes),
        "operating_state_count": len({value.lower() for value in modes}),
        "operating_states": sorted(modes),
        "required_operating_state_count": 2,
    }


def promotion_gate_review(
    evidence: xr.Dataset,
    *,
    ensemble_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Materialise the specified manual-promotion gates without auto-promoting.

    A missing eligible solar/ensemble/release-test input is intentionally a
    blocking *unavailable* gate, not a score of zero or a silently omitted
    metric.  This lets the development UI describe exactly why a candidate is
    retained without confusing a rolling diagnostic with campaign evidence.
    """
    base: dict[str, object] = {
        "status": "not_eligible",
        "decision": "manual_review_required",
        "minimum_evidence": {
            "independent_cycles_per_lead_bucket": 30,
            "utc_days_per_lead_bucket": 10,
        },
        "solar": "blocked_mpp_active_available_power_evidence_unavailable",
        "ensemble": "blocked_memberwise_candidate_not_generated",
        "reserve_events": "insufficient_events",
        "release_tests": "external_reproducibility_resource_and_api_checks_required",
    }
    if ensemble_gate is not None:
        base["ensemble"] = dict(ensemble_gate)
        base["reserve_events"] = str(
            ensemble_gate.get("reserve_events", "insufficient_events")
        )
    if evidence.sizes.get("record", 0) == 0 or "EvaluationAvailable" not in evidence:
        base["evidence"] = "insufficient_evidence"
        return base
    frame = evidence.to_dataframe().reset_index(drop=True)
    if frame.empty or "EvaluationAvailable" not in frame:
        base["evidence"] = "insufficient_evidence"
        return base
    frame = frame.loc[frame["EvaluationAvailable"].astype(bool)].copy()
    if "IndependentEvaluationSample" in frame:
        frame = frame.loc[frame["IndependentEvaluationSample"].astype(bool)].copy()
    if frame.empty:
        base["evidence"] = "no_matured_observations"
        return base
    frame["IssueTime"] = pd.to_datetime(frame["IssueTime"])
    lead_gates: dict[str, object] = {}
    evidence_ready = True
    for label, start, end in LEAD_BUCKETS:
        rows = frame.loc[(frame["LeadHours"] >= start) & (frame["LeadHours"] < end)]
        cycles = _independent_cycle_count(rows)
        days = int(rows["IssueTime"].dt.floor("D").nunique())
        diversity = _promotion_diversity(rows)
        sample_count_eligible = cycles >= 30 and days >= 10
        eligible = sample_count_eligible and bool(diversity["eligible"])
        lead_gates[label] = {
            "cycles": cycles,
            "utc_days": days,
            "sample_count_eligible": sample_count_eligible,
            "diversity": diversity,
            "status": (
                "eligible"
                if eligible
                else "insufficient_diversity"
                if sample_count_eligible
                else "insufficient_evidence"
            ),
        }
        evidence_ready &= eligible
    base["independent_evidence"] = lead_gates
    if not evidence_ready:
        base["evidence"] = "insufficient_evidence"
        return base

    soc_0_6 = _error_metrics(
        frame.loc[(frame["LeadHours"] >= 0.0) & (frame["LeadHours"] < 6.0)],
        candidate="CandidateSOC",
        baseline="BaselineSOC",
        observed="ObservedSOC",
    )
    soc_6_24 = _error_metrics(
        frame.loc[(frame["LeadHours"] >= 6.0) & (frame["LeadHours"] < 24.0)],
        candidate="CandidateSOC",
        baseline="BaselineSOC",
        observed="ObservedSOC",
    )
    soc_0_24 = _error_metrics(
        frame.loc[(frame["LeadHours"] >= 0.0) & (frame["LeadHours"] < 24.0)],
        candidate="CandidateSOC",
        baseline="BaselineSOC",
        observed="ObservedSOC",
    )
    if None in (soc_0_6, soc_6_24, soc_0_24):
        base["evidence"] = "insufficient_soc_observations"
        return base
    assert soc_0_6 is not None and soc_6_24 is not None and soc_0_24 is not None
    def persistence_skill(rows: pd.DataFrame) -> float | None:
        errors = _error_metrics(rows, candidate="CandidateSOC", baseline="BaselineSOC", observed="ObservedSOC")
        values = rows.loc[:, ["SOCAuthoringAnchor", "ObservedSOC"]].apply(pd.to_numeric, errors="coerce").dropna()
        if errors is None or values.empty:
            return None
        persistence_mae = float(
            np.mean(np.abs(values["SOCAuthoringAnchor"].to_numpy() - values["ObservedSOC"].to_numpy()))
        )
        return float(1.0 - errors["candidate_mae"] / persistence_mae) if persistence_mae > 0.0 else None

    skill_0_6 = persistence_skill(frame.loc[(frame["LeadHours"] >= 0.0) & (frame["LeadHours"] < 6.0)])
    skill_6_24 = persistence_skill(frame.loc[(frame["LeadHours"] >= 6.0) & (frame["LeadHours"] < 24.0)])
    base["soc"] = {
        "combined_0_24_improvement_fraction": soc_0_24["mae_improvement_fraction"],
        "required_combined_0_24_improvement_fraction": 0.10,
        "zero_to_six_mae_change": soc_0_6["candidate_mae"] - soc_0_6["baseline_mae"],
        "six_to_twentyfour_mae_change": soc_6_24["candidate_mae"] - soc_6_24["baseline_mae"],
        "zero_to_six_persistence_skill": skill_0_6,
        "six_to_twentyfour_persistence_skill": skill_6_24,
        "status": (
            "pass"
            if (
                soc_0_24["mae_improvement_fraction"] >= 0.10
                and soc_0_6["candidate_mae"] <= soc_0_6["baseline_mae"] + 2.0
                and soc_6_24["candidate_mae"] <= soc_6_24["baseline_mae"] + 2.0
                and skill_0_6 is not None
                and skill_0_6 > 0.0
                and skill_6_24 is not None
                and skill_6_24 > 0.0
            )
            else "fail"
        ),
    }
    long_gate: dict[str, object] = {}
    for label, start, end in LEAD_BUCKETS[2:]:
        metrics = _error_metrics(
            frame.loc[(frame["LeadHours"] >= start) & (frame["LeadHours"] < end)],
            candidate="CandidateSOC",
            baseline="BaselineSOC",
            observed="ObservedSOC",
        )
        if metrics is None:
            long_gate[label] = {"status": "insufficient_evidence"}
            continue
        skill = persistence_skill(frame.loc[(frame["LeadHours"] >= start) & (frame["LeadHours"] < end)])
        low_mae = metrics["mae_improvement_fraction"] >= 0.25
        no_worse_bias = abs(metrics["candidate_bias"]) <= abs(metrics["baseline_bias"])
        long_gate[label] = {
            **metrics,
            "persistence_skill": skill,
            "status": "pass" if (skill is not None and skill >= 0.0) or (low_mae and no_worse_bias) else "fail",
        }
    base["long_lead_soc"] = long_gate
    solar_metrics = _solar_metric_summary(frame)
    if solar_metrics.get("status") == "evidence":
        solar_metrics["required_mae_improvement_fraction"] = 0.10
        solar_metrics["status"] = (
            "pass"
            if float(solar_metrics["mae_improvement_fraction"]) >= 0.10
            else "fail"
        )
    else:
        solar_metrics["status"] = "blocked_mpp_active_available_power_evidence_unavailable"
    base["solar"] = solar_metrics

    load_metrics = _error_metrics(
        frame,
        candidate="CandidateLoadWatts",
        baseline="BaselineLoadWatts",
        observed="ObservedLoadWatts",
    )
    if load_metrics is None:
        base["load"] = {"status": "insufficient_evidence"}
    else:
        baseline_abs_bias = abs(load_metrics["baseline_bias"])
        candidate_abs_bias = abs(load_metrics["candidate_bias"])
        load_metrics["status"] = (
            "pass"
            if baseline_abs_bias > 0.0 and candidate_abs_bias <= 0.90 * baseline_abs_bias
            else "fail"
        )
        load_metrics["required_absolute_bias_improvement_fraction"] = 0.10
        load_metrics["absolute_bias_improvement_fraction"] = (
            float(1.0 - candidate_abs_bias / baseline_abs_bias) if baseline_abs_bias > 0.0 else np.nan
        )
        base["load"] = load_metrics
    gate_statuses = [
        str(base["soc"].get("status", "blocked")),
        *(str(value.get("status", "blocked")) for value in long_gate.values()),
        str(base["solar"].get("status", "blocked")),
        str(base["load"].get("status", "blocked")),
    ]
    if isinstance(base["ensemble"], Mapping):
        gate_statuses.append(str(base["ensemble"].get("status", "blocked")))
    base["quantitative_gates"] = (
        "pass" if gate_statuses and all(status == "pass" for status in gate_statuses) else "not_all_pass"
    )
    base["evidence"] = "eligible_for_manual_gate_review"
    # The physical solar and memberwise ensemble gates remain explicit blocks,
    # so this function can never accidentally accept a candidate by itself.
    return base
