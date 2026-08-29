#!/usr/bin/env python3
"""Replay the live v10 inputs into an isolated physical-PV candidate.

The runner intentionally reuses the already-published deterministic ECMWF file
and APS issue-time anchor.  It never writes the operational latest product,
state, archive, skill, hindcast, or ensemble artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
    PHYSICAL_SOLAR_MODEL_NAME,
    POWER_PDU_ZARR_PATH,
    POWER_ZARR_PATH,
    _atomic_write_zarr,
    _paths_overlap,
    _write_state,
    generate,
    validate_provider,
)
from power_solar_model import (
    load_physical_solar_config,
    physical_solar_config_digest,
    physical_solar_contract_id,
)


BASELINE_FORECAST_ZARR_PATH = Path(
    os.environ.get(
        "AURORA_POWER_BASELINE_FORECAST_ZARR",
        "/data/aurora/dev-products/power/power_soc_forecast.zarr",
    )
)
CANDIDATE_ROOT = Path(
    os.environ.get(
        "AURORA_POWER_PHYSICAL_CANDIDATE_ROOT",
        "/data/aurora/dev-products/power/evaluations/solar_physical_v1",
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_seed_state(attrs: dict[str, object]) -> dict[str, object]:
    """Seed only shared non-solar adaptive state for the first paired run."""

    state: dict[str, object] = {}
    text_json_fields = {
        "load_mode_registry": "load_mode_registry",
        "soc_bias_correction_pct_points_by_bucket": "soc_bias_correction_pct_points_by_bucket",
        "latest_metrics": "previous_forecast_metrics",
    }
    for state_name, attr_name in text_json_fields.items():
        try:
            state[state_name] = json.loads(str(attrs.get(attr_name, "{}")))
        except json.JSONDecodeError:
            state[state_name] = {}
    scalar_fields = {
        "current_load_mode": "load_mode",
        "current_load_mode_source": "load_mode_source",
        "current_load_mode_signature": "load_mode_signature",
        "current_load_mode_learning_reason": "load_mode_learning_reason",
        "load_regime": "load_regime",
    }
    for state_name, attr_name in scalar_fields.items():
        if attr_name in attrs:
            state[state_name] = str(attrs[attr_name])
    state["current_load_mode_learning_ready"] = (
        str(attrs.get("load_mode_learning_ready", "false")).lower() == "true"
    )
    state["seed_source"] = "paired_operational_baseline_latest_product"
    state["seed_baseline_publication_signature"] = str(attrs.get("publication_signature", ""))
    return state


def run_candidate(
    *,
    baseline_forecast_zarr: Path = BASELINE_FORECAST_ZARR_PATH,
    candidate_root: Path = CANDIDATE_ROOT,
    power_zarr: Path = POWER_ZARR_PATH,
    pdu_zarr: Path = POWER_PDU_ZARR_PATH,
    physical_config: Path = DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
) -> Path:
    baseline_forecast_zarr = Path(baseline_forecast_zarr)
    candidate_root = Path(candidate_root)
    if any(path.suffix.lower() == ".zarr" for path in (candidate_root, *candidate_root.parents)):
        raise ValueError("Physical candidate root cannot be inside a Zarr store")
    if _paths_overlap(candidate_root, baseline_forecast_zarr):
        raise ValueError("Physical candidate root must not contain or be inside the baseline forecast")
    if not baseline_forecast_zarr.exists():
        raise FileNotFoundError(f"Baseline forecast is missing: {baseline_forecast_zarr}")

    with xr.open_zarr(baseline_forecast_zarr, chunks={}) as baseline:
        baseline_snapshot = baseline.load()
        baseline_attrs = dict(baseline_snapshot.attrs)
    baseline_signature = str(baseline_attrs.get("publication_signature", "")).strip()
    if not baseline_signature:
        raise ValueError("Baseline forecast does not have a publication signature")
    input_value = str(baseline_attrs.get("ecmwf_input_file", "")).strip()
    if not input_value:
        raise ValueError("Baseline forecast does not identify its ECMWF input file")
    input_forecast = Path(input_value)
    if not input_forecast.is_file():
        raise FileNotFoundError(f"Baseline ECMWF input is no longer available: {input_forecast}")
    issue_value = str(baseline_attrs.get("initial_soc_time", "")).strip()
    if not issue_value:
        raise ValueError("Baseline forecast does not identify its APS issue-time anchor")
    issue_time = pd.Timestamp(issue_value)
    if issue_time.tz is not None:
        issue_time = issue_time.tz_convert("UTC").tz_localize(None)
    try:
        latitude = float(baseline_attrs["site_latitude"])
        longitude = float(baseline_attrs["site_longitude"])
        horizon_hours = int(float(baseline_attrs["forecast_horizon_hours"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Baseline forecast is missing site or horizon provenance") from exc

    physical_configuration = load_physical_solar_config(physical_config)
    physical_config_digest = physical_solar_config_digest(physical_configuration)
    physical_contract_id = physical_solar_contract_id(
        physical_configuration,
        latitude=latitude,
        longitude=longitude,
    )
    input_digest = _sha256_file(input_forecast)
    pair_payload = {
        "schema": 2,
        "baseline_publication_signature": baseline_signature,
        "ecmwf_sha256": input_digest,
        "issue_time": issue_time.isoformat(),
        "physical_solar_contract_id": physical_contract_id,
    }
    pair_id = "power-solar-pair-v2-" + hashlib.sha256(
        json.dumps(pair_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    input_snapshot_id = f"sha256:{input_digest}"

    provider_value = str(
        baseline_attrs.get(
            "ecmwf_provider_effective",
            baseline_attrs.get("ecmwf_provider_requested", "legacy"),
        )
    )
    try:
        provider = validate_provider(provider_value)
    except ValueError:
        provider = "legacy"

    output_zarr = candidate_root / "power_soc_forecast.zarr"
    state_path = candidate_root / "power_soc_forecast_state.json"

    result = generate(
        power_zarr=power_zarr,
        pdu_zarr=pdu_zarr,
        output_zarr=output_zarr,
        input_forecast=input_forecast,
        state_path=state_path,
        archive_zarr=candidate_root / "power_soc_forecast_archive.zarr",
        skill_zarr=candidate_root / "power_soc_forecast_skill.zarr",
        hindcast_zarr=candidate_root / "power_soc_hindcast.zarr",
        latitude=latitude,
        longitude=longitude,
        horizon_hours=horizon_hours,
        provider=provider,
        shadow_report_path=candidate_root / "ecmwf_provider_shadow.json",
        max_power_age_minutes=None,
        archive_forecast=True,
        solar_model=PHYSICAL_SOLAR_MODEL_NAME,
        physical_solar_config_path=physical_config,
        power_cutoff_time=issue_time,
        evaluation_pair_id=pair_id,
        input_snapshot_id=input_snapshot_id,
        expected_input_sha256=input_digest,
        expected_physical_config_sha256=physical_config_digest,
        pair_reference=baseline_snapshot,
        state_override=_baseline_seed_state(baseline_attrs),
    )
    with xr.open_zarr(result, chunks={}) as candidate:
        candidate_snapshot = candidate.load()
        if candidate.attrs.get("solar_model_name") != PHYSICAL_SOLAR_MODEL_NAME:
            raise RuntimeError("Candidate publication did not use the requested physical solar model")
        if candidate.attrs.get("evaluation_pair_id") != pair_id:
            raise RuntimeError("Candidate publication lost its baseline pairing identity")
    if _sha256_file(input_forecast) != input_digest:
        raise RuntimeError("Baseline ECMWF input changed while the paired candidate was generated")
    candidate_signature = str(candidate_snapshot.attrs.get("publication_signature", "")).strip()
    if not candidate_signature:
        raise RuntimeError("Candidate publication does not have a publication signature")
    pair_family = candidate_root / "pairs" / pair_id
    pair_bundle = pair_family / candidate_signature
    pair_manifest = {
        "schema_version": 2,
        "pair_status": "complete",
        "evaluation_pair_id": pair_id,
        "baseline_publication_signature": baseline_signature,
        "candidate_publication_signature": candidate_signature,
        "input_snapshot_id": input_snapshot_id,
        "ecmwf_input_file": str(input_forecast),
        "initial_soc_time": issue_time.isoformat(),
        "site_latitude": latitude,
        "site_longitude": longitude,
        "forecast_horizon_hours": horizon_hours,
        "solar_model_contract_id": str(
            candidate_snapshot.attrs.get("solar_model_contract_id", "")
        ),
        "forecast_model_contract_id": str(
            candidate_snapshot.attrs.get("forecast_model_contract_id", "")
        ),
        "solar_physical_config_sha256": physical_config_digest,
        "baseline_snapshot": "baseline_forecast.zarr",
        "candidate_snapshot": "candidate_forecast.zarr",
    }
    if pair_bundle.exists():
        try:
            existing_manifest = json.loads(
                (pair_bundle / "pair_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing pair bundle is incomplete: {pair_bundle}") from exc
        if existing_manifest != pair_manifest:
            raise RuntimeError(f"Existing immutable pair bundle does not match: {pair_bundle}")
    else:
        pair_family.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".pair-staging-", dir=pair_family) as temporary:
            staging = Path(temporary)
            _atomic_write_zarr(baseline_snapshot, staging / "baseline_forecast.zarr")
            _atomic_write_zarr(candidate_snapshot, staging / "candidate_forecast.zarr")
            # The complete manifest is written last inside staging; one atomic
            # directory rename then makes the entire immutable bundle visible.
            _write_state(staging / "pair_manifest.json", pair_manifest)
            staging.replace(pair_bundle)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an isolated, issue-time-paired physical APS solar/SOC candidate"
    )
    parser.add_argument("--baseline-forecast-zarr", type=Path, default=BASELINE_FORECAST_ZARR_PATH)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=POWER_PDU_ZARR_PATH)
    parser.add_argument("--physical-config", type=Path, default=DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH)
    args = parser.parse_args()
    output = run_candidate(
        baseline_forecast_zarr=args.baseline_forecast_zarr,
        candidate_root=args.candidate_root,
        power_zarr=args.power_zarr,
        pdu_zarr=args.pdu_zarr,
        physical_config=args.physical_config,
    )
    print(f"Verified isolated physical candidate {output}")


if __name__ == "__main__":
    main()
