"""Exact-intersection evidence for production versus development power forecasts.

The evaluator is deliberately read-only with respect to both forecast trees.
It pairs only identical issue/cycle/valid-time rows, rejects SOC-anchor or lead
mismatches, deduplicates cached/retried source cycles, and materialises a
separate development evidence product.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import _write_state
from power_v12_hybrid import LEAD_BUCKETS, stable_json_digest, utc_now_iso
from power_observation_truth import (
    POWER_OBSERVATION_FIELDS, TRUTH_CONTRACT_VERSION, bounded_observation_view,
    electrical_observation_series, evaluate_power_observations, forecast_interval_starts,
)
from power_implementation_identity import semantic_code_key


FORECAST_FIELDS = {
    "soc": "BatterySOCForecast",
    "solar": "ForecastSolarWatts",
    "load": "ForecastLoadWatts",
    "solar_delivered": "ForecastPVDeliveredWatts",
}
OBSERVATION_MATCH_TOLERANCE = pd.Timedelta(minutes=10)


def _atomic_write_evidence_zarr(ds: xr.Dataset, output_zarr: Path) -> None:
    """Replace paired evidence while retaining the prior tree until promotion."""

    output_zarr = Path(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    staging = output_zarr.with_name(f".{output_zarr.name}.tmp")
    previous = output_zarr.with_name(f".{output_zarr.name}.previous")
    for candidate in (output_zarr, staging, previous):
        if candidate.is_symlink():
            raise ValueError("Paired evidence paths must not be symlinks")
        if candidate.exists() and not candidate.is_dir():
            raise ValueError("Paired evidence paths must be directories")
    if previous.exists():
        if output_zarr.exists():
            shutil.rmtree(previous)
        else:
            previous.rename(output_zarr)
    if staging.exists():
        shutil.rmtree(staging)
    chunk_dim = "time" if "time" in ds.sizes else next(iter(ds.sizes), None)
    chunked = (
        ds.chunk({chunk_dim: min(max(ds.sizes.get(chunk_dim, 1), 1), 288)})
        if chunk_dim is not None
        else ds
    )
    try:
        chunked.to_zarr(staging, mode="w", consolidated=True)
        try:
            if output_zarr.exists():
                output_zarr.rename(previous)
            staging.rename(output_zarr)
        except BaseException:
            if previous.exists() and not output_zarr.exists():
                previous.rename(output_zarr)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if previous.exists():
        shutil.rmtree(previous)


def _issue_text(archive: xr.Dataset, name: str, issue_count: int) -> np.ndarray:
    if name not in archive:
        return np.full(issue_count, "", dtype=object)
    return np.asarray(archive[name].values, dtype=str).reshape(-1)


def _normalise_pair_key_text(values: pd.Series) -> pd.Series:
    """Return canonical non-null text without inventing pairing provenance."""

    normalised = values.astype(str).str.strip()
    return normalised.mask(normalised.str.lower().isin({"", "nan", "none", "nat"}), "")


def _empty_evidence(
    *,
    status: str = "no_exact_intersection",
    rejection_counts: dict[str, int] | None = None,
    candidate_rows: int = 0,
) -> xr.Dataset:
    counts = dict(rejection_counts or {})
    return xr.Dataset(
        coords={"record": np.array([], dtype=np.int64)},
        attrs={
            "power_prod_dev_paired_evidence": "true",
            "status": status,
            "generated_at_utc": utc_now_iso(),
            "candidate_base_intersection_rows": int(candidate_rows),
            "mismatched_rows_rejected": int(counts.get("total", 0)),
            "rejection_counts_json": json.dumps(counts, sort_keys=True),
        },
    )


def _archive_rows(archive: xr.Dataset, *, label: str) -> pd.DataFrame:
    required = {"ForecastValidTime", "ForecastLeadHours", "issue_time"}
    if not required.issubset(set(archive.variables) | set(archive.coords)):
        return pd.DataFrame()
    issue_count = int(archive.sizes.get("issue_time", 0))
    steps = int(archive.sizes.get("forecast_step", 0))
    if issue_count == 0 or steps == 0:
        return pd.DataFrame()
    issue_values = pd.DatetimeIndex(archive["issue_time"].values)
    issues = pd.DatetimeIndex(np.repeat(issue_values.values, steps))
    valid = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    interval_starts = np.concatenate([
        forecast_interval_starts(archive["ForecastValidTime"].values[index], issue).values
        for index, issue in enumerate(issue_values)
    ])
    leads = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    if "ECMWFCycleTime" in archive:
        cycle_issue = pd.DatetimeIndex(archive["ECMWFCycleTime"].values)
    else:
        cycle_issue = issue_values.floor("3h")
    cycles = pd.DatetimeIndex(np.repeat(cycle_issue.values, steps))
    source_cycles = np.repeat(_issue_text(archive, "SourceCycleSetID", issue_count), steps)
    source_cycles = np.asarray([str(value).strip() for value in source_cycles], dtype=object)
    anchors = np.full(issue_count, np.nan, dtype=np.float64)
    if "SOCAuthoringAnchorPct" in archive:
        anchors = np.asarray(
            pd.to_numeric(
                np.asarray(archive["SOCAuthoringAnchorPct"].values).reshape(-1),
                errors="coerce",
            ),
            dtype=np.float64,
        )
    frame: dict[str, Any] = {
        "issue_time": issues,
        "cycle_time": cycles,
        "source_cycle_set_id": source_cycles,
        "valid_time": valid,
        "interval_start_time": interval_starts,
        "lead_hours": leads,
        "soc_anchor_pct": np.repeat(anchors, steps),
        "forecast_system_version": np.repeat(
            _issue_text(archive, "ForecastSystemVersion", issue_count), steps
        ),
        "forecast_model_contract_id": np.repeat(
            _issue_text(archive, "ForecastModelContractID", issue_count), steps
        ),
        "forecast_identity_id": np.repeat(
            _issue_text(archive, "ForecastIdentityID", issue_count), steps
        ),
        "feature_set_version": np.repeat(
            _issue_text(archive, "FeatureSetVersion", issue_count), steps
        ),
        "forecast_code_revision": np.repeat(
            _issue_text(archive, "ForecastCodeRevision", issue_count), steps
        ),
        "forecast_implementation_digest": np.repeat(
            _issue_text(archive, "ForecastImplementationDigest", issue_count), steps
        ),
        "load_mode": np.repeat(_issue_text(archive, "LoadMode", issue_count), steps),
        "solar_power_semantics": np.repeat(_issue_text(archive, "SolarPowerSemantics", issue_count), steps),
        "clearness_index": (
            np.asarray(archive["ECMWFClearnessIndex"].values, dtype=np.float64).reshape(-1)
            if "ECMWFClearnessIndex" in archive
            else np.full(issue_count * steps, np.nan, dtype=np.float64)
        ),
    }
    for short, field in FORECAST_FIELDS.items():
        frame[f"{label}_{short}"] = (
            np.asarray(archive[field].values, dtype=np.float64).reshape(-1)
            if field in archive
            else np.full(issue_count * steps, np.nan, dtype=np.float64)
        )
    available_solar = np.asarray(["available" in str(value).lower() for value in frame["solar_power_semantics"]])
    if "ForecastPVDeliveredWatts" not in archive:
        frame[f"{label}_solar_delivered"] = np.where(available_solar, np.nan, frame[f"{label}_solar"])
    out = pd.DataFrame(frame)
    return out.loc[
        ~out["issue_time"].isna()
        & ~out["cycle_time"].isna()
        & ~out["valid_time"].isna()
        & np.isfinite(out["lead_hours"])
    ].copy()


def _observations(power: xr.Dataset | None) -> dict[str, pd.Series]:
    if power is None or "time" not in power.coords:
        return {}
    truth = electrical_observation_series(power)
    return {"soc": truth.soc, "solar": truth.solar_delivered_w, "load": truth.load_w}


def paired_observation_view(
    power: xr.Dataset,
    evidence: xr.Dataset,
    *,
    tolerance: pd.Timedelta = OBSERVATION_MATCH_TOLERANCE,
) -> xr.Dataset:
    """Keep bounded interval telemetry, including battery and MPPT mode truth."""
    targets = evidence["valid_time"].values if "valid_time" in evidence else []
    starts = evidence["interval_start_time"].values if "interval_start_time" in evidence else np.full(len(targets), np.datetime64("NaT"))
    return bounded_observation_view(power, starts, targets, tolerance=tolerance)


def attach_paired_observations(
    evidence: xr.Dataset,
    power: xr.Dataset | None,
    *,
    tolerance: pd.Timedelta = OBSERVATION_MATCH_TOLERANCE,
    operating_state: xr.Dataset | None = None,
) -> xr.Dataset:
    """Attach retrospective observations without changing the paired cohort."""

    if power is None or evidence.sizes.get("record", 0) == 0:
        return evidence
    paired = evidence.copy()
    paired.attrs.update(
        {
            name: value
            for name, value in power.attrs.items()
            if name.startswith("paired_observation_")
        }
    )
    from power_observation_truth import ObservationPolicy
    targets = pd.DatetimeIndex(paired["valid_time"].values)
    starts = paired["interval_start_time"].values if "interval_start_time" in paired else np.full(len(targets), np.datetime64("NaT"))
    truth = evaluate_power_observations(power, starts, targets,
        expected_modes=paired["load_mode"].values if "load_mode" in paired else None,
        operating_state=operating_state, policy=ObservationPolicy(point_tolerance=tolerance))
    available = np.zeros(len(targets), dtype=bool)
    for label in ("production", "development"):
        field = f"solar_power_semantics_{label}"
        if field in paired:
            available |= np.asarray(["available" in str(value).lower() for value in paired[field].values])
    truth["solar_w"] = np.where(available, truth.solar_available_w, truth.solar_delivered_w)
    mapping = {"soc": "soc", "load": "load_w", "solar": "solar_w", "solar_delivered": "solar_delivered_w"}
    for name, field in mapping.items():
        paired[f"observed_{name}"] = (("record",), truth[field].to_numpy(dtype=float))
    for name in truth.columns:
        if name in {"interval_start", "valid_time", *mapping.values()}:
            continue
        values = truth[name]
        paired[f"observation_{name}"] = (("record",), values.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(values) else values.astype(str).to_numpy(dtype="U128"))
    paired.attrs["observation_truth_contract"] = TRUTH_CONTRACT_VERSION
    return paired


def build_exact_intersection_evidence(
    production: xr.Dataset,
    development: xr.Dataset,
    *,
    power: xr.Dataset | None = None,
    anchor_tolerance_pct: float = 0.05,
    lead_tolerance_hours: float = 1.0e-4,
    operating_state: xr.Dataset | None = None,
) -> xr.Dataset:
    prod = _archive_rows(production, label="production")
    dev = _archive_rows(development, label="development")
    if prod.empty or dev.empty:
        return _empty_evidence()
    base_keys = ["issue_time", "cycle_time", "valid_time"]
    strict_keys = [*base_keys, "source_cycle_set_id", "load_mode"]
    for frame in (prod, dev):
        frame["source_cycle_set_id"] = _normalise_pair_key_text(
            frame["source_cycle_set_id"]
        )
        frame["load_mode"] = _normalise_pair_key_text(frame["load_mode"])
    prod = prod.sort_values(strict_keys).drop_duplicates(strict_keys, keep="last")
    dev = dev.sort_values(strict_keys).drop_duplicates(strict_keys, keep="last")
    paired = prod.merge(
        dev,
        on=base_keys,
        how="inner",
        suffixes=("_production", "_development"),
    )
    if paired.empty:
        return _empty_evidence(
            rejection_counts={"no_common_issue_cycle_valid_time": 1, "total": 0}
        )
    source_production = _normalise_pair_key_text(paired["source_cycle_set_id_production"])
    source_development = _normalise_pair_key_text(paired["source_cycle_set_id_development"])
    source_known = source_production.ne("") & source_development.ne("")
    source_ok = source_known & source_production.eq(source_development)
    mode_production = _normalise_pair_key_text(paired["load_mode_production"])
    mode_development = _normalise_pair_key_text(paired["load_mode_development"])
    mode_known = mode_production.ne("") & mode_development.ne("")
    mode_ok = mode_known & mode_production.eq(mode_development)
    lead_ok = np.isclose(
        paired["lead_hours_production"],
        paired["lead_hours_development"],
        atol=float(lead_tolerance_hours),
        rtol=0.0,
    )
    interval_ok = paired["interval_start_time_production"].eq(paired["interval_start_time_development"]) | (
        paired["interval_start_time_production"].isna() & paired["interval_start_time_development"].isna())
    prod_anchor = paired["soc_anchor_pct_production"].to_numpy(dtype=float)
    dev_anchor = paired["soc_anchor_pct_development"].to_numpy(dtype=float)
    anchor_known = np.isfinite(prod_anchor) & np.isfinite(dev_anchor)
    anchor_ok = anchor_known & np.isclose(
        prod_anchor, dev_anchor, atol=float(anchor_tolerance_pct), rtol=0.0
    )
    accepted = source_ok.to_numpy(dtype=bool) & mode_ok.to_numpy(dtype=bool) & lead_ok & anchor_ok & interval_ok.to_numpy(dtype=bool)
    rejection_counts = {
        "source_cycle_missing": int((~source_known).sum()),
        "source_cycle_mismatch": int((source_known & ~source_ok).sum()),
        "load_mode_missing": int((~mode_known).sum()),
        "load_mode_mismatch": int((mode_known & ~mode_ok).sum()),
        "soc_anchor_unknown": int(np.count_nonzero(~anchor_known)),
        "soc_anchor_mismatch": int(np.count_nonzero(anchor_known & ~anchor_ok)),
        "lead_mismatch": int(np.count_nonzero(~lead_ok)),
        "interval_bounds_mismatch": int((~interval_ok).sum()),
        "total": int(np.count_nonzero(~accepted)),
    }
    candidate_rows = int(len(paired))
    paired = paired.loc[accepted].copy()
    if paired.empty:
        return _empty_evidence(
            status="all_candidate_pairs_rejected",
            rejection_counts=rejection_counts,
            candidate_rows=candidate_rows,
        )
    paired["source_cycle_set_id"] = source_production.loc[paired.index]
    paired["load_mode"] = mode_production.loc[paired.index]
    paired["interval_start_time"] = paired["interval_start_time_production"]
    clearness_production = paired["clearness_index_production"].to_numpy(dtype=float)
    clearness_development = paired["clearness_index_development"].to_numpy(dtype=float)
    clearness_known = np.isfinite(clearness_production) & np.isfinite(clearness_development)
    clearness_consistent = clearness_known & np.isclose(
        clearness_production,
        clearness_development,
        atol=0.02,
        rtol=0.0,
    )
    clearness = np.where(
        clearness_consistent,
        0.5 * (clearness_production + clearness_development),
        np.nan,
    )
    paired["cloud_regime"] = np.where(
        clearness >= 0.65,
        "clear",
        np.where(clearness <= 0.35, "cloudy", "transitional"),
    )
    paired.loc[~np.isfinite(clearness), "cloud_regime"] = ""
    # One source cycle contributes only one row at each valid time, even if an
    # old archive contains multiple cached anchors for that meteorological run.
    pre_deduplication_rows = int(len(paired))
    paired = paired.sort_values("issue_time").drop_duplicates(
        ["source_cycle_set_id", "load_mode", "cycle_time", "valid_time"],
        keep="last",
    )
    duplicate_rows = pre_deduplication_rows - int(len(paired))
    cohort_columns = (
        "forecast_system_version_production",
        "forecast_model_contract_id_production",
        "feature_set_version_production",
        "forecast_code_revision_production",
        "forecast_system_version_development",
        "forecast_model_contract_id_development",
        "feature_set_version_development",
        "forecast_code_revision_development",
    )
    paired["evaluation_cohort_id"] = [
        "prod-dev-cohort-v1-"
        + stable_json_digest(
            {
                column: semantic_code_key({
                    "forecast_implementation_digest": row[f"forecast_implementation_digest_{column.removeprefix('forecast_code_revision_')}"],
                    "forecast_code_revision": row[column],
                }) if column.startswith("forecast_code_revision_") else str(row[column] or "")
                for column in cohort_columns
            }
        )[:20]
        for _, row in paired.iterrows()
    ]
    paired["lead_hours"] = paired["lead_hours_development"]
    output_columns = [
        "issue_time",
        "cycle_time",
        "valid_time",
        "interval_start_time",
        "lead_hours",
        "soc_anchor_pct_production",
        "soc_anchor_pct_development",
        "source_cycle_set_id_production",
        "source_cycle_set_id_development",
        "source_cycle_set_id",
        "forecast_system_version_production",
        "forecast_system_version_development",
        "forecast_model_contract_id_production",
        "forecast_model_contract_id_development",
        "forecast_identity_id_production",
        "forecast_identity_id_development",
        "feature_set_version_production",
        "feature_set_version_development",
        "forecast_code_revision_production",
        "forecast_code_revision_development",
        "forecast_implementation_digest_production",
        "forecast_implementation_digest_development",
        "evaluation_cohort_id",
        "load_mode_production",
        "load_mode_development",
        "load_mode",
        "solar_power_semantics_production",
        "solar_power_semantics_development",
        "clearness_index_production",
        "clearness_index_development",
        "cloud_regime",
        *(f"production_{name}" for name in FORECAST_FIELDS),
        *(f"development_{name}" for name in FORECAST_FIELDS),
    ]
    variables: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for name in output_columns:
        values = paired[name]
        if name.endswith("_time"):
            array = pd.DatetimeIndex(values).to_numpy(dtype="datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(values):
            array = values.to_numpy(dtype=np.float64)
        else:
            array = values.fillna("").astype(str).to_numpy(dtype="U512")
        variables[name] = (("record",), array)
    evidence = xr.Dataset(
        variables,
        coords={"record": np.arange(len(paired), dtype=np.int64)},
        attrs={
            "power_prod_dev_paired_evidence": "true",
            "status": "complete" if len(paired) else "no_exact_intersection",
            "generated_at_utc": utc_now_iso(),
            "anchor_tolerance_pct": float(anchor_tolerance_pct),
            "lead_tolerance_hours": float(lead_tolerance_hours),
            "candidate_base_intersection_rows": candidate_rows,
            "mismatched_rows_rejected": rejection_counts["total"],
            "rejection_counts_json": json.dumps(rejection_counts, sort_keys=True),
            "duplicate_pair_rows_discarded": duplicate_rows,
            "pair_key": (
                "issue_time+ecmwf_cycle_time+valid_time+source_cycle_set_id+load_mode;"
                "known_equal_soc_anchor;dedupe=source_cycle_set_id+load_mode+"
                "ecmwf_cycle_time+valid_time"
            ),
        },
    )
    return attach_paired_observations(evidence, power, operating_state=operating_state)


def _paired_metric(
    frame: pd.DataFrame,
    name: str,
    *,
    bootstrap_samples: int,
    diversity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    columns = [f"production_{name}", f"development_{name}", f"observed_{name}"]
    if any(column not in frame for column in columns):
        return {
            "status": "observations_unavailable",
            "campaignReady": False,
            "samples": 0,
            "cycles": 0,
        }
    rows = frame.dropna(subset=columns).copy()
    if rows.empty:
        return {
            "status": "insufficient_evidence",
            "campaignReady": False,
            "samples": 0,
            "cycles": 0,
        }
    prod_error = rows[columns[0]].to_numpy(dtype=float) - rows[columns[2]].to_numpy(dtype=float)
    dev_error = rows[columns[1]].to_numpy(dtype=float) - rows[columns[2]].to_numpy(dtype=float)
    prod_mae = float(np.mean(np.abs(prod_error)))
    dev_mae = float(np.mean(np.abs(dev_error)))
    cycles = _normalise_pair_key_text(rows["source_cycle_set_id"]).unique()
    improvements: list[float] = []
    if len(cycles) >= 2 and bootstrap_samples > 0:
        seed = int(stable_json_digest({"metric": name, "cycles": [str(v) for v in cycles]})[:16], 16)
        rng = np.random.default_rng(seed)
        for _ in range(int(bootstrap_samples)):
            sampled = rng.choice(cycles, size=len(cycles), replace=True)
            blocks = [rows.loc[rows["source_cycle_set_id"] == cycle] for cycle in sampled]
            sample = pd.concat(blocks, ignore_index=True)
            p = np.mean(np.abs(sample[columns[0]] - sample[columns[2]]))
            d = np.mean(np.abs(sample[columns[1]] - sample[columns[2]]))
            if p > 0:
                improvements.append(float(1.0 - d / p))
    utc_days = int(pd.DatetimeIndex(rows["issue_time"]).floor("D").nunique())
    sample_count_ready = len(cycles) >= 30 and utc_days >= 10
    diversity_ready = bool((diversity or {}).get("campaignReady", False))
    campaign_ready = sample_count_ready and diversity_ready
    return {
        "status": "campaign_evidence" if campaign_ready else "diagnostic_sparse",
        "campaignReady": campaign_ready,
        "requiredIndependentCycles": 30,
        "requiredUTCDays": 10,
        "sampleCountReady": sample_count_ready,
        "diversityReady": diversity_ready,
        "samples": int(len(rows)),
        "cycles": int(len(cycles)),
        "utcDays": utc_days,
        "productionMAE": prod_mae,
        "developmentMAE": dev_mae,
        "developmentBias": float(np.mean(dev_error)),
        "productionBias": float(np.mean(prod_error)),
        "developmentMAEImprovementFraction": float(1.0 - dev_mae / prod_mae)
        if prod_mae > 0
        else None,
        "improvement95CI": (
            [float(value) for value in np.quantile(improvements, (0.025, 0.975))]
            if improvements
            else None
        ),
    }


def _campaign_diversity(rows: pd.DataFrame) -> dict[str, Any]:
    """Require explicit clear/cloudy metadata and multiple operating states."""

    regimes = (
        set(_normalise_pair_key_text(rows["cloud_regime"])) - {""}
        if "cloud_regime" in rows
        else set()
    )
    modes = (
        set(_normalise_pair_key_text(rows["load_mode"])) - {""}
        if "load_mode" in rows
        else set()
    )
    modes = {
        mode
        for mode in modes
        if mode.lower() not in {"unknown", "unavailable", "unspecified"}
    }
    clear_covered = "clear" in regimes
    cloudy_covered = "cloudy" in regimes
    multiple_states = len(modes) >= 2
    ready = clear_covered and cloudy_covered and multiple_states
    return {
        "status": "campaign_diversity_met" if ready else "diagnostic_incomplete_diversity",
        "campaignReady": ready,
        "cloudRegimeMetadataAvailable": bool(regimes),
        "clearCovered": clear_covered,
        "cloudyCovered": cloudy_covered,
        "requiredOperatingStateCount": 2,
        "operatingStateCount": len(modes),
        "operatingStates": sorted(modes),
        "cloudRegimes": sorted(regimes),
    }


def paired_score_surface(
    evidence: xr.Dataset,
    *,
    bootstrap_samples: int = 500,
) -> dict[str, Any]:
    if evidence.sizes.get("record", 0) == 0:
        return {
            "schemaVersion": 1,
            "status": "insufficient_evidence",
            "campaignReady": False,
            "dataUpdatedAt": utc_now_iso(),
            "pairedRows": 0,
            "pairedIndependentCycles": 0,
            "evidenceStatus": str(evidence.attrs.get("status", "no_exact_intersection")),
            "candidateBaseIntersectionRows": int(
                evidence.attrs.get("candidate_base_intersection_rows", 0)
            ),
            "campaignDiversity": _campaign_diversity(pd.DataFrame()),
            "leadBuckets": {},
            "rejectionCounts": json.loads(
                str(evidence.attrs.get("rejection_counts_json", "{}"))
            ),
            "mismatchedRowsRejected": int(
                evidence.attrs.get("mismatched_rows_rejected", 0)
            ),
        }
    frame = evidence.to_dataframe().reset_index(drop=True)
    frame["cycle_time"] = pd.to_datetime(frame["cycle_time"])
    frame["issue_time"] = pd.to_datetime(frame["issue_time"])
    if "evaluation_cohort_id" in frame:
        contract_columns = [
            column
            for column in (
                "forecast_model_contract_id_production",
                "forecast_model_contract_id_development",
            )
            if column in frame
        ]
        contract_known = pd.Series(False, index=frame.index)
        for column in contract_columns:
            contract_known |= _normalise_pair_key_text(frame[column]).ne("")
        active_candidates = frame.loc[contract_known]
        latest_row = (
            active_candidates if not active_candidates.empty else frame
        ).sort_values("issue_time").iloc[-1]
        active_cohort = str(latest_row["evaluation_cohort_id"])
        cohort_frame = frame.loc[frame["evaluation_cohort_id"].astype(str) == active_cohort].copy()
        cohort_inventory = {
            str(cohort): {
                "pairedRows": int(len(rows)),
                "pairedIndependentCycles": int(rows["source_cycle_set_id"].nunique()),
                "pairedUTCDays": int(rows["issue_time"].dt.floor("D").nunique()),
                "status": "active" if str(cohort) == active_cohort else "historical",
            }
            for cohort, rows in frame.groupby("evaluation_cohort_id", dropna=False)
        }
    else:
        active_cohort = "legacy-unpartitioned"
        cohort_frame = frame
        cohort_inventory = {
            active_cohort: {
                "pairedRows": int(len(frame)),
                "pairedIndependentCycles": int(frame["source_cycle_set_id"].nunique()),
                "pairedUTCDays": int(frame["issue_time"].dt.floor("D").nunique()),
                "status": "active",
            }
        }
    lead_buckets: dict[str, Any] = {}
    for label, start, end in LEAD_BUCKETS:
        rows = cohort_frame.loc[
            (cohort_frame["lead_hours"] >= start) & (cohort_frame["lead_hours"] < end)
        ]
        diversity = _campaign_diversity(rows)
        lead_buckets[label] = {
            **{
                name: _paired_metric(
                    rows,
                    name,
                    bootstrap_samples=bootstrap_samples,
                    diversity=diversity,
                )
                for name in FORECAST_FIELDS
            },
            "diversity": diversity,
            "campaignReady": False,
        }
        lead_buckets[label]["campaignReady"] = bool(
            lead_buckets[label]["soc"].get("campaignReady") is True
            and diversity["campaignReady"] is True
        )
    campaign_ready = all(
        bucket["campaignReady"] is True
        for bucket in lead_buckets.values()
    )
    return {
        "schemaVersion": 1,
        "status": "campaign_evidence" if campaign_ready else "diagnostic_only",
        "campaignReady": campaign_ready,
        "dataUpdatedAt": utc_now_iso(),
        "pairedRows": int(len(cohort_frame)),
        "pairedIndependentCycles": int(cohort_frame["source_cycle_set_id"].nunique()),
        "pairedUTCDays": int(cohort_frame["issue_time"].dt.floor("D").nunique()),
        "totalExactIntersectionRows": int(len(frame)),
        "evidenceStatus": str(evidence.attrs.get("status", "complete")),
        "candidateBaseIntersectionRows": int(
            evidence.attrs.get("candidate_base_intersection_rows", len(frame))
        ),
        "activeCohortID": active_cohort,
        "cohorts": cohort_inventory,
        "campaignDiversity": _campaign_diversity(cohort_frame),
        "observationTruthContract": str(evidence.attrs.get("observation_truth_contract", "legacy_point_truth")),
        "truthDiagnostics": {
            field: {str(value): int(count) for value, count in cohort_frame[field].value_counts(dropna=False).items()}
            for field in ("observation_load_status", "observation_load_measurement", "observation_solar_available_status", "observation_operating_state_status")
            if field in cohort_frame
        },
        "realizedStateStrata": {
            str(value): {
                "samples": int(len(rows)),
                "soc": _paired_metric(rows, "soc", bootstrap_samples=0),
                "load": _paired_metric(rows, "load", bootstrap_samples=0),
            }
            for value, rows in cohort_frame.groupby("observation_operating_state_status")
        } if "observation_operating_state_status" in cohort_frame else {},
        "leadBuckets": lead_buckets,
        "pairKey": str(evidence.attrs.get("pair_key", "")),
        "mismatchedRowsRejected": int(evidence.attrs.get("mismatched_rows_rejected", 0)),
        "rejectionCounts": json.loads(
            str(evidence.attrs.get("rejection_counts_json", "{}"))
        ),
        "duplicatePairRowsDiscarded": int(
            evidence.attrs.get("duplicate_pair_rows_discarded", 0)
        ),
    }


def write_paired_products(
    evidence: xr.Dataset,
    *,
    output_zarr: Path,
    status_json: Path,
    history_jsonl: Path,
    bootstrap_samples: int = 500,
    status_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _atomic_write_evidence_zarr(evidence, Path(output_zarr))
    summary = paired_score_surface(evidence, bootstrap_samples=bootstrap_samples)
    if status_context:
        summary = {**summary, **status_context}
    write_evaluation_status_event(
        status_json=Path(status_json),
        history_jsonl=Path(history_jsonl),
        event=summary,
    )
    return summary


def write_evaluation_status_event(
    *,
    status_json: Path,
    history_jsonl: Path,
    event: dict[str, Any],
) -> None:
    """Atomically replace current status and append an immutable run event."""

    status_json = Path(status_json)
    history_jsonl = Path(history_jsonl)
    _write_state(status_json, event)
    history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    history_event = {**event, "eventDigest": stable_json_digest(event)}
    with history_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
