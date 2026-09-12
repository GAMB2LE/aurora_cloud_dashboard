#!/usr/bin/env python3
"""Build a NEW archive solely from verified immutable forecast issues.

Example:
  python repair_power_forecast_archive.py --issues-root /data/power/forecast-issues \
    --output-zarr /data/power/recovery/run-1/recovered.zarr \
    --report /data/power/recovery/run-1/report.json

Unverifiable snapshots are recorded in the report, never edited or guessed.
The existing forecast archive is not an input to reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
    ISSUE_SNAPSHOT_DIGEST_MARKER,
    _archive_row_from_forecast,
    _verify_issue_snapshot,
)
from power_archive_io import write_forecast_archive


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def verified_issue_snapshot(directory: Path) -> tuple[xr.Dataset, dict[str, str]]:
    """Use the persisted manifest and existing byte-checksum verifier."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Issue directory is not a direct directory")
    manifest_path = directory / "issue_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Issue manifest is a symbolic link")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1 or manifest.get("status") != "complete":
        raise ValueError("Issue manifest is incomplete or unsupported")
    if manifest.get("relativePath") != "forecast.zarr":
        raise ValueError("Issue manifest has an unsupported snapshot path")
    if manifest.get("digestAlgorithm") != ISSUE_SNAPSHOT_DIGEST_ALGORITHM:
        raise ValueError("Issue manifest has an unsupported digest algorithm")
    signature = str(manifest.get("publicationSignature") or "")
    if not signature:
        raise ValueError("Issue manifest lacks a publication signature")
    snapshot = directory / "forecast.zarr"
    # The deployed publisher's artifact_digest binds filename + NUL + bytes
    # + NUL, including for a single file. A bare file-byte hash is not its
    # manifest contract. Keep this identical to the publisher, not permissive.
    marker_digest = "sha256:" + hashlib.sha256(
        ISSUE_SNAPSHOT_DIGEST_MARKER.encode("utf-8") + b"\0"
        + (snapshot / ISSUE_SNAPSHOT_DIGEST_MARKER).read_bytes() + b"\0"
    ).hexdigest()
    if marker_digest != manifest.get("snapshotMarkerDigest"):
        raise ValueError("Issue manifest does not match the snapshot marker")
    digest = _verify_issue_snapshot(snapshot, expected_signature=signature)
    if manifest.get("contentDigest") != f"sha256:{digest}":
        raise ValueError("Issue manifest does not match snapshot content")
    with xr.open_zarr(snapshot, chunks={}, consolidated=True) as opened:
        forecast = opened.load()
    attrs = forecast.attrs
    required = (
        "forecast_model_contract_id", "forecast_identity_id", "forecast_system_version",
        "feature_set_version", "feature_set_digest", "forecast_code_revision",
        "source_cycle_set_id", "source_manifest_digest", "initial_soc_time",
        "initial_soc_pct", "soc_anchor_time_utc", "observation_cutoff_utc",
        "training_cutoff_utc", "ecmwf_cycle_time", "load_model_version", "load_mode",
        "load_mode_learning_ready",
    )
    missing = [name for name in required if str(attrs.get(name, "")).strip().lower() in {"", "nan", "none"}]
    if missing:
        raise ValueError("Incomplete issue provenance: " + ", ".join(missing))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(attrs["feature_set_digest"])):
        raise ValueError("Feature digest is not complete")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(attrs["forecast_code_revision"])):
        raise ValueError("Code revision is not complete")
    if not re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", str(attrs["source_manifest_digest"])):
        raise ValueError("Source manifest digest is not complete")
    if len(str(attrs["source_cycle_set_id"])) < 16:
        raise ValueError("Source cycle identity is not complete")
    for manifest_key, attr_key in (
        ("forecastIdentityID", "forecast_identity_id"),
        ("sourceCycleSetID", "source_cycle_set_id"),
    ):
        if manifest.get(manifest_key) != attrs[attr_key]:
            raise ValueError(f"Issue manifest identity mismatch: {manifest_key}")
    if pd.Timestamp(manifest["sourceCycleUTC"]) != pd.Timestamp(attrs["ecmwf_cycle_time"]):
        raise ValueError("Issue manifest source cycle mismatch")
    if attrs.get("forecast_refresh_kind") != "ecmwf_cycle" or any(
        str(attrs.get(name, "")).lower() != "true"
        for name in ("forecast_verification_eligible", "independent_cycle")
    ):
        raise ValueError("Issue is not an independent eligible forecast")
    if pd.Timestamp(attrs["initial_soc_time"]) != pd.Timestamp(attrs["soc_anchor_time_utc"]):
        raise ValueError("Issue SOC authoring time is inconsistent")
    anchor = float(attrs["initial_soc_pct"])
    if not np.isfinite(anchor) or not 0.0 <= anchor <= 100.0:
        raise ValueError("Issue SOC anchor is invalid")
    if not {"BatterySOCForecast", "ForecastSolarWatts", "ForecastLoadWatts"}.issubset(forecast):
        raise ValueError("Snapshot lacks complete forecast variables")
    row = _archive_row_from_forecast(forecast)
    evidence = {
        "snapshot": str(snapshot), "issue_time": str(row.issue_time.values[0]),
        "forecast_identity_id": str(attrs["forecast_identity_id"]),
        "source_cycle_set_id": str(attrs["source_cycle_set_id"]),
        "publication_signature": signature, "content_digest": f"sha256:{digest}",
    }
    return row, evidence


def recover_archive(
    issues_root: Path, output_zarr: Path, report_path: Path, *, maximum_snapshots: int = 10000
) -> dict[str, object]:
    """Reconstruct only provable issues; reject conflicting duplicate issues."""
    source = Path(issues_root)
    output = Path(output_zarr)
    report_file = Path(report_path)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Issues root must be a direct directory")
    if output.suffix != ".zarr" or report_file.suffix != ".json":
        raise ValueError("Use a new .zarr archive and .json recovery report")
    resolved = [path.resolve() for path in (source, output, report_file)]
    if any(_overlap(resolved[i], resolved[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Recovery outputs must not overlap the immutable source or each other")
    if output.exists() or output.is_symlink() or report_file.exists() or report_file.is_symlink():
        raise FileExistsError("Recovery requires new output and report paths")
    if maximum_snapshots < 1:
        raise ValueError("maximum_snapshots must be positive")
    directories = sorted(source.iterdir())
    if len(directories) > maximum_snapshots:
        raise ValueError("Snapshot inventory exceeds the configured recovery bound")
    by_issue: dict[str, list[tuple[xr.Dataset, dict[str, str]]]] = defaultdict(list)
    skipped: list[dict[str, str]] = []
    for directory in directories:
        try:
            row, evidence = verified_issue_snapshot(directory)
            by_issue[evidence["issue_time"]].append((row, evidence))
        except Exception as exc:
            skipped.append({"source": str(directory), "reason": f"{type(exc).__name__}: {exc}"})
    rows = []
    accepted = []
    for issue_time, candidates in sorted(by_issue.items()):
        digests = {evidence["content_digest"] for _, evidence in candidates}
        if len(digests) != 1:
            for _, evidence in candidates:
                skipped.append({"source": evidence["snapshot"], "reason": f"Conflicting immutable snapshots for issue {issue_time}"})
            continue
        row, evidence = candidates[0]
        rows.append(row)
        accepted.append(evidence)
        for _, duplicate in candidates[1:]:
            skipped.append({"source": duplicate["snapshot"], "reason": "Identical duplicate snapshot; retained once"})
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "recovered" if rows else "no_verified_issues",
        "source_root": str(source.resolve()), "output_zarr": str(output.resolve()),
        "recovery_policy": "checksum_verified_immutable_issues_only_no_metadata_inference",
        "source_snapshots_examined": len(directories), "recovered_issue_count": len(rows),
        "distinct_source_cycle_count": len({e["source_cycle_set_id"] for e in accepted}),
        "accepted": accepted, "excluded": skipped,
    }
    if rows:
        max_steps = max(row.sizes["forecast_step"] for row in rows)
        archive = xr.concat(
            [row.reindex(forecast_step=np.arange(max_steps, dtype=np.int32)) for row in rows],
            dim="issue_time", join="outer",
        ).sortby("issue_time")
        archive.attrs.update(
            recovery_policy=report["recovery_policy"],
            recovery_source_root=str(source.resolve()),
            recovery_report=str(report_file.resolve()),
        )
        write_forecast_archive(archive, output)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues-root", required=True, type=Path)
    parser.add_argument("--output-zarr", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maximum-snapshots", type=int, default=10000)
    args = parser.parse_args()
    result = recover_archive(args.issues_root, args.output_zarr, args.report, maximum_snapshots=args.maximum_snapshots)
    print(json.dumps({key: result[key] for key in ("status", "recovered_issue_count", "distinct_source_cycle_count", "output_zarr")}))
    if result["status"] != "recovered":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
