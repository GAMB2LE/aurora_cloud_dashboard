#!/usr/bin/env python3
"""Generate the isolated, evaluation-first v12 hybrid power candidate.

The runner is deliberately conservative.  It accepts only a full, independent
ECMWF baseline issue, snapshots its forcing and SOC anchor, and writes solely
under the candidate root.  Operational v10/v11 products, archives and
adaptive state are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
    LEGACY_SOLAR_MODEL_NAME,
    PHYSICAL_SOLAR_MODEL_NAME,
    POWER_PDU_ZARR_PATH,
    POWER_ZARR_PATH,
    _atomic_write_zarr,
    _paths_overlap,
    _write_state,
    generate,
    validate_paired_candidate,
    validate_provider,
)
from generate_power_soc_physical_candidate import _baseline_seed_state
from power_solar_model import (
    load_physical_solar_config,
    physical_solar_config_digest,
    physical_solar_contract_id,
)
from power_v12_hybrid import (
    build_campaign_evidence,
    campaign_score_surfaces,
    fit_bounded_load_residual,
    stable_json_digest,
    utc_now_iso,
    v12_forecast_identity,
)


BASELINE_FORECAST_ZARR_PATH = Path(
    os.environ.get(
        "AURORA_POWER_BASELINE_FORECAST_ZARR",
        "/data/aurora/dev-products/power/power_soc_forecast.zarr",
    )
)
BASELINE_ARCHIVE_ZARR_PATH = Path(
    os.environ.get(
        "AURORA_POWER_BASELINE_FORECAST_ARCHIVE_ZARR",
        "/data/aurora/dev-products/power/power_soc_forecast_archive.zarr",
    )
)
CANDIDATE_ROOT = Path(
    os.environ.get(
        "AURORA_POWER_V12_CANDIDATE_ROOT",
        "/data/aurora/dev-products/power/candidates/v12",
    )
)

LANE_PHYSICAL_SOLAR = "B_physical_solar"
LANE_LOAD_RESIDUAL = "C_load_residual"
LANE_HYBRID = "D_physical_solar_load_residual"
LANES = (LANE_PHYSICAL_SOLAR, LANE_LOAD_RESIDUAL, LANE_HYBRID)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_history(path: Path, value: dict[str, object]) -> None:
    """Append a compact immutable event; its digest chains the previous event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_digest = ""
    if path.exists():
        try:
            last = path.read_text(encoding="utf-8").splitlines()[-1]
            previous_digest = str(json.loads(last).get("event_digest", ""))
        except (OSError, IndexError, json.JSONDecodeError):
            previous_digest = "invalid_prior_history"
    event = {"previous_event_digest": previous_digest, **value}
    event["event_digest"] = stable_json_digest(event)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _code_revision() -> str:
    configured = os.environ.get("AURORA_FORECAST_CODE_REVISION", "").strip()
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def _model_evaluation_active() -> bool:
    if os.environ.get("AURORA_POWER_CANDIDATE_DEFER_MODEL_EVALUATION", "true").lower() != "true":
        return False
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", "aurora-model-evaluation-daily.service"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _baseline_archive_before(
    path: Path,
    issue_time: pd.Timestamp,
) -> xr.Dataset | None:
    if not path.exists():
        return None
    try:
        with xr.open_zarr(path, chunks={}) as opened:
            archive = opened.load()
    except Exception:
        return None
    if "issue_time" not in archive.coords:
        return archive
    # A candidate may use only earlier forecast issues, never the baseline row
    # it is about to compare against.
    times = pd.DatetimeIndex(archive["issue_time"].values)
    return archive.isel(issue_time=np.asarray(times < issue_time))


def _source_manifest(
    *,
    baseline_attrs: dict[str, object],
    baseline_signature: str,
    input_forecast: Path,
    input_digest: str,
    issue_time: pd.Timestamp,
    physical_config_digest: str,
    physical_contract_id: str,
) -> tuple[dict[str, object], str, str]:
    cycle = str(baseline_attrs.get("ecmwf_cycle_time", ""))
    provider = str(
        baseline_attrs.get(
            "ecmwf_provider_effective",
            baseline_attrs.get("ecmwf_provider_requested", "legacy"),
        )
    )
    source_cycle_set = f"ecmwf:{provider}:{cycle}:sha256:{input_digest[:20]}"
    manifest = {
        "schema_version": 1,
        "baseline_publication_signature": baseline_signature,
        "initial_soc_time": issue_time.isoformat(),
        "ecmwf_cycle_time": cycle,
        "source_cycle_set_id": source_cycle_set,
        "ecmwf_provider_effective": provider,
        "ecmwf_input_file": str(input_forecast),
        "ecmwf_input_sha256": input_digest,
        "physical_solar_config_sha256": physical_config_digest,
        "physical_solar_contract_id": physical_contract_id,
        "site_latitude": str(baseline_attrs.get("site_latitude", "")),
        "site_longitude": str(baseline_attrs.get("site_longitude", "")),
        "forecast_horizon_hours": str(baseline_attrs.get("forecast_horizon_hours", "")),
        "observation_cutoff": issue_time.isoformat(),
    }
    return manifest, stable_json_digest(manifest), source_cycle_set


def _write_immutable_manifest(root: Path, manifest: dict[str, object], digest: str) -> Path:
    path = root / "source_manifests" / f"sha256-{digest}.json"
    if path.exists():
        existing = _read_json(path)
        if existing != manifest:
            raise RuntimeError(f"Immutable source manifest does not match: {path}")
    else:
        _atomic_json(path, manifest)
    return path


def _fixed_bias_from_baseline(attrs: dict[str, object]) -> dict[str, float]:
    try:
        raw = json.loads(str(attrs.get("soc_bias_correction_pct_points_by_bucket", "{}")))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            out[str(key)] = parsed
    return out


def _validate_load_change_pair(candidate: xr.Dataset, baseline: xr.Dataset, *, solar_must_match: bool) -> None:
    """Validate identical source/state while intentionally allowing load changes."""
    failures: list[str] = []
    for name in ("initial_soc_time", "ecmwf_cycle_time", "ecmwf_input_file", "forecast_horizon_hours"):
        if str(candidate.attrs.get(name, "")) != str(baseline.attrs.get(name, "")):
            failures.append(name)
    for name in ("initial_soc_pct", "site_latitude", "site_longitude"):
        try:
            matches = np.isclose(
                float(candidate.attrs.get(name, np.nan)),
                float(baseline.attrs.get(name, np.nan)),
                rtol=0.0,
                atol=1.0e-8,
                equal_nan=True,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            failures.append(name)
    if "time" not in candidate or "time" not in baseline or not np.array_equal(
        np.asarray(candidate["time"].values), np.asarray(baseline["time"].values)
    ):
        failures.append("forecast_time_grid")
    fields = ["ECMWFSolarIrradiance"]
    if solar_must_match:
        fields.append("ForecastSolarWatts")
    for name in fields:
        if name not in candidate or name not in baseline or not np.allclose(
            np.asarray(candidate[name].values, dtype=np.float64),
            np.asarray(baseline[name].values, dtype=np.float64),
            rtol=1.0e-6,
            atol=1.0e-5,
            equal_nan=True,
        ):
            failures.append(name)
    for name in (
        "load_model",
        "load_model_version",
        "load_state_contract",
        "battery_energy_model",
        "battery_usable_capacity_kwh",
        "battery_charge_efficiency",
        "battery_discharge_efficiency",
        "battery_parasitic_load_w",
        "battery_max_charge_w",
        "battery_max_discharge_w",
        "soc_bias_correction_pct_points_by_bucket",
    ):
        if str(candidate.attrs.get(name, "")) != str(baseline.attrs.get(name, "")):
            failures.append(name)
    if failures:
        raise ValueError("v12 candidate source/state pair validation failed: " + ", ".join(dict.fromkeys(failures)))


def _pair_id(
    *,
    lane: str,
    baseline_signature: str,
    input_digest: str,
    issue_time: pd.Timestamp,
    physical_contract_id: str,
) -> str:
    payload = {
        "schema": 1,
        "lane": lane,
        "baseline_publication_signature": baseline_signature,
        "input_sha256": input_digest,
        "issue_time": issue_time.isoformat(),
        "physical_solar_contract_id": physical_contract_id,
    }
    return "power-v12-pair-v1-" + stable_json_digest(payload)[:20]


def _write_pair_bundle(
    lane_root: Path,
    *,
    pair_id: str,
    baseline: xr.Dataset,
    candidate: xr.Dataset,
    manifest: dict[str, object],
) -> Path:
    family = lane_root / "pairs" / pair_id
    signature = str(candidate.attrs.get("publication_signature", "")).strip()
    if not signature:
        raise RuntimeError("Candidate publication has no signature")
    bundle = family / signature
    manifest = {**manifest, "pair_status": "complete", "candidate_publication_signature": signature}
    if bundle.exists():
        existing = _read_json(bundle / "pair_manifest.json")
        if existing != manifest:
            raise RuntimeError(f"Existing immutable pair bundle does not match: {bundle}")
        return bundle
    family.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pair-staging-", dir=family) as temporary:
        staging = Path(temporary)
        _atomic_write_zarr(baseline, staging / "baseline_forecast.zarr")
        _atomic_write_zarr(candidate, staging / "candidate_forecast.zarr")
        _write_state(staging / "pair_manifest.json", manifest)
        staging.replace(bundle)
    return bundle


def _lane_result_path(root: Path, lane: str) -> Path:
    return root / "lanes" / lane / "power_soc_forecast.zarr"


def run_candidate(
    *,
    baseline_forecast_zarr: Path = BASELINE_FORECAST_ZARR_PATH,
    baseline_archive_zarr: Path = BASELINE_ARCHIVE_ZARR_PATH,
    candidate_root: Path = CANDIDATE_ROOT,
    power_zarr: Path = POWER_ZARR_PATH,
    pdu_zarr: Path = POWER_PDU_ZARR_PATH,
    physical_config: Path = DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
) -> dict[str, Path]:
    """Generate lanes B/C/D for one verified v10/v11 ECMWF baseline issue."""
    baseline_forecast_zarr = Path(baseline_forecast_zarr)
    baseline_archive_zarr = Path(baseline_archive_zarr)
    candidate_root = Path(candidate_root)
    if any(path.suffix.lower() == ".zarr" for path in (candidate_root, *candidate_root.parents)):
        raise ValueError("v12 candidate root cannot be inside a Zarr store")
    protected = (baseline_forecast_zarr, baseline_archive_zarr, power_zarr, pdu_zarr)
    if any(_paths_overlap(candidate_root, path) for path in protected):
        raise ValueError("v12 candidate root overlaps a protected baseline or input product")
    if not baseline_forecast_zarr.exists():
        raise FileNotFoundError(f"Baseline forecast is missing: {baseline_forecast_zarr}")
    if _model_evaluation_active():
        status = {
            "schema_version": 1,
            "environment": "development",
            "status": "deferred_model_evaluation_active",
            "updated_at_utc": utc_now_iso(),
        }
        _atomic_json(candidate_root / "status.json", status)
        return {}
    with xr.open_zarr(baseline_forecast_zarr, chunks={}) as opened:
        baseline = opened.load()
    attrs = dict(baseline.attrs)
    if str(attrs.get("forecast_verification_eligible", "")).lower() != "true":
        raise ValueError("Baseline is not an archive-eligible independent forecast issue")
    if str(attrs.get("forecast_refresh_kind", "")) != "ecmwf_cycle":
        raise ValueError("Baseline refresh is not a full ECMWF cycle")
    baseline_signature = str(attrs.get("publication_signature", "")).strip()
    if not baseline_signature:
        raise ValueError("Baseline forecast does not have a publication signature")
    existing_status = _read_json(candidate_root / "status.json")
    if (
        existing_status
        and existing_status.get("status") == "complete"
        and existing_status.get("baseline_publication_signature") == baseline_signature
    ):
        return {
            lane: _lane_result_path(candidate_root, lane)
            for lane in LANES
            if _lane_result_path(candidate_root, lane).exists()
        }
    input_forecast = Path(str(attrs.get("ecmwf_input_file", "")).strip())
    if not input_forecast.is_file():
        raise FileNotFoundError("Baseline ECMWF forcing is no longer available")
    issue_time = _utc_naive(attrs.get("initial_soc_time", ""))
    try:
        latitude = float(attrs["site_latitude"])
        longitude = float(attrs["site_longitude"])
        horizon_hours = int(float(attrs["forecast_horizon_hours"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Baseline is missing site or horizon provenance") from exc
    configuration = load_physical_solar_config(physical_config)
    config_digest = physical_solar_config_digest(configuration)
    physical_contract = physical_solar_contract_id(configuration, latitude=latitude, longitude=longitude)
    input_digest = _sha256_file(input_forecast)
    source_manifest, source_manifest_digest, source_cycle_set_id = _source_manifest(
        baseline_attrs=attrs,
        baseline_signature=baseline_signature,
        input_forecast=input_forecast,
        input_digest=input_digest,
        issue_time=issue_time,
        physical_config_digest=config_digest,
        physical_contract_id=physical_contract,
    )
    _write_immutable_manifest(candidate_root, source_manifest, source_manifest_digest)
    try:
        provider = validate_provider(str(attrs.get("ecmwf_provider_effective", "legacy")))
    except ValueError:
        provider = "legacy"
    reference_archive = _baseline_archive_before(baseline_archive_zarr, issue_time)
    with xr.open_zarr(power_zarr, chunks={}) as opened:
        power_for_fit = opened.sel(time=slice(None, issue_time.to_datetime64())).load()
    residual = fit_bounded_load_residual(
        reference_archive,
        power_for_fit,
        issue_time=issue_time,
        forecast_times=pd.DatetimeIndex(baseline["time"].values),
        load_mode=str(attrs.get("load_mode", "unknown")),
    )
    seed_state = _baseline_seed_state(attrs)
    fixed_bias = _fixed_bias_from_baseline(attrs)
    results: dict[str, Path] = {}
    lane_signatures: dict[str, str] = {}
    lane_summaries: dict[str, dict[str, object]] = {}
    lane_specs = (
        (LANE_PHYSICAL_SOLAR, PHYSICAL_SOLAR_MODEL_NAME, None, True, False),
        (LANE_LOAD_RESIDUAL, LEGACY_SOLAR_MODEL_NAME, residual.as_profile(), False, True),
        (LANE_HYBRID, PHYSICAL_SOLAR_MODEL_NAME, residual.as_profile(), False, False),
    )
    for lane, solar_model, load_profile, exact_pair, solar_must_match in lane_specs:
        lane_root = candidate_root / "lanes" / lane
        pair_id = _pair_id(
            lane=lane,
            baseline_signature=baseline_signature,
            input_digest=input_digest,
            issue_time=issue_time,
            physical_contract_id=physical_contract,
        )
        identity = v12_forecast_identity(
            lane=lane,
            issue_time=issue_time,
            source_cycle_set_id=source_cycle_set_id,
            source_manifest_digest=source_manifest_digest,
            physical_config_digest=config_digest,
            load_residual=residual if load_profile is not None else None,
            code_revision=_code_revision(),
        )
        output = generate(
            power_zarr=power_zarr,
            pdu_zarr=pdu_zarr,
            output_zarr=lane_root / "power_soc_forecast.zarr",
            input_forecast=input_forecast,
            state_path=lane_root / "power_soc_forecast_state.json",
            archive_zarr=lane_root / "power_soc_forecast_archive.zarr",
            skill_zarr=lane_root / "daily_diagnostic_skill.zarr",
            hindcast_zarr=lane_root / "power_soc_hindcast.zarr",
            latitude=latitude,
            longitude=longitude,
            horizon_hours=horizon_hours,
            provider=provider,
            shadow_report_path=lane_root / "ecmwf_provider_shadow.json",
            max_power_age_minutes=None,
            archive_forecast=True,
            solar_model=solar_model,
            physical_solar_config_path=physical_config,
            power_cutoff_time=issue_time,
            evaluation_pair_id=pair_id,
            input_snapshot_id=f"sha256:{input_digest}",
            expected_input_sha256=input_digest,
            expected_physical_config_sha256=config_digest if solar_model == PHYSICAL_SOLAR_MODEL_NAME else None,
            pair_reference=baseline if exact_pair else None,
            state_override=seed_state,
            forecast_identity=identity,
            load_residual_profile=load_profile,
            reference_forecast_archive=reference_archive,
            fixed_soc_bias_corrections_override=fixed_bias,
        )
        with xr.open_zarr(output, chunks={}) as opened:
            candidate = opened.load()
        if exact_pair:
            validate_paired_candidate(candidate, baseline)
        else:
            _validate_load_change_pair(candidate, baseline, solar_must_match=solar_must_match)
        if _sha256_file(input_forecast) != input_digest:
            raise RuntimeError("ECMWF forcing changed during candidate generation")
        pair_manifest = {
            "schema_version": 1,
            "evaluation_pair_id": pair_id,
            "candidate_lane": lane,
            "baseline_publication_signature": baseline_signature,
            "input_snapshot_id": f"sha256:{input_digest}",
            "source_manifest_digest": source_manifest_digest,
            "initial_soc_time": issue_time.isoformat(),
            "source_cycle_set_id": source_cycle_set_id,
            "forecast_model_contract_id": str(candidate.attrs.get("forecast_model_contract_id", "")),
            "forecast_identity_id": str(candidate.attrs.get("forecast_identity_id", "")),
            "solar_model_contract_id": str(candidate.attrs.get("solar_model_contract_id", "")),
            "solar_physical_config_sha256": config_digest,
            "baseline_snapshot": "baseline_forecast.zarr",
            "candidate_snapshot": "candidate_forecast.zarr",
        }
        _write_pair_bundle(
            lane_root,
            pair_id=pair_id,
            baseline=baseline,
            candidate=candidate,
            manifest=pair_manifest,
        )
        evidence = build_campaign_evidence(lane_root / "pairs", power_for_fit, lane=lane)
        _atomic_write_zarr(evidence, lane_root / "campaign_evidence.zarr")
        summary = campaign_score_surfaces(evidence)
        _atomic_json(lane_root / "evaluation_summary.json", summary)
        results[lane] = output
        lane_signatures[lane] = str(candidate.attrs.get("publication_signature", ""))
        lane_summaries[lane] = summary
    status = {
        "schema_version": 1,
        "environment": "development",
        "authority": "candidate",
        "status": "complete",
        "updated_at_utc": utc_now_iso(),
        "baseline_publication_signature": baseline_signature,
        "source_manifest_digest": source_manifest_digest,
        "source_cycle_set_id": source_cycle_set_id,
        "input_snapshot_id": f"sha256:{input_digest}",
        "training_cutoff_utc": issue_time.isoformat(),
        "load_residual": {
            "status": residual.status,
            "contract_id": residual.contract_id,
            "training_samples": residual.training_samples,
            "training_cycles": residual.training_cycles,
            "training_days": residual.training_days,
        },
        "lanes": {
            lane: {
                "path": str(path),
                "publication_signature": lane_signatures[lane],
            }
            for lane, path in results.items()
        },
        "promotion_status": "not_eligible_requires_campaign_evidence",
    }
    _atomic_json(candidate_root / "status.json", status)
    _atomic_json(
        candidate_root / "acceptance_record.json",
        {
            "schema_version": 1,
            "environment": "development",
            "authority": "candidate",
            "status": "not_accepted",
            "decision": "retain_unpublished_candidate",
            "updated_at_utc": utc_now_iso(),
            "baseline_publication_signature": baseline_signature,
            "source_manifest_digest": source_manifest_digest,
            "source_cycle_set_id": source_cycle_set_id,
            "reason": (
                "Promotion is manual and requires cumulative paired campaign evidence; "
                "this runner never changes an operational forecast product."
            ),
            "required_gates": {
                "paired_independent_cycles": "minimum 30 per lead bucket across 10 UTC days",
                "soc_skill": "review campaign evidence against v10 and persistence",
                "solar_and_load": "review only issue-time-safe, uncensored metrics",
                "ensemble": "not_generated_in_bounded_initial_candidate",
                "reserve_events": "insufficient_events unless an event sample is available",
                "operational_safety": "memory, runtime, reproducibility and API compatibility must pass",
            },
        },
    )
    _atomic_json(
        candidate_root / "review_summary.json",
        {
            "schema_version": 1,
            "environment": "development",
            "authority": "candidate",
            "status": "pending_campaign_review",
            "updated_at_utc": utc_now_iso(),
            "baseline_publication_signature": baseline_signature,
            "source_manifest_digest": source_manifest_digest,
            "source_cycle_set_id": source_cycle_set_id,
            "load_residual": status["load_residual"],
            "lanes": lane_summaries,
            "next_action": "Accumulate paired independent ECMWF-cycle evidence; do not promote from rolling diagnostics.",
        },
    )
    _append_history(
        candidate_root / "evaluation_history.jsonl",
        {
            "event": "candidate_cycle_complete",
            "at_utc": utc_now_iso(),
            "baseline_publication_signature": baseline_signature,
            "source_manifest_digest": source_manifest_digest,
            "lanes": sorted(results),
            "load_residual_status": residual.status,
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate issue-time-paired v12 hybrid power candidates")
    parser.add_argument("--baseline-forecast-zarr", type=Path, default=BASELINE_FORECAST_ZARR_PATH)
    parser.add_argument("--baseline-archive-zarr", type=Path, default=BASELINE_ARCHIVE_ZARR_PATH)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=POWER_PDU_ZARR_PATH)
    parser.add_argument("--physical-config", type=Path, default=DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH)
    args = parser.parse_args()
    results = run_candidate(
        baseline_forecast_zarr=args.baseline_forecast_zarr,
        baseline_archive_zarr=args.baseline_archive_zarr,
        candidate_root=args.candidate_root,
        power_zarr=args.power_zarr,
        pdu_zarr=args.pdu_zarr,
        physical_config=args.physical_config,
    )
    if results:
        print("Verified isolated v12 candidate lanes: " + ", ".join(sorted(results)))
    else:
        print("v12 candidate deferred; no operational product was changed")


if __name__ == "__main__":
    main()
