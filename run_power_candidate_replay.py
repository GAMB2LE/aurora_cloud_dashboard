#!/usr/bin/env python3
"""Replay a bounded chronological queue of verified, local forecast issues.

This collector calls the existing candidate with embedded site irradiance and
explicit local observation stores. It never retrieves weather or enrolls public
models. Install the optional systemd unit for OS-enforced resource/network limits.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

from generate_power_soc_forecast import _issue_snapshot_tree_digest
from generate_power_soc_v12_candidate import LANES, run_candidate
from power_implementation_identity import forecast_implementation_digest
from repair_power_forecast_archive import verified_issue_snapshot


MODEL_EVALUATION_UNIT = "aurora-model-evaluation-daily.service"
RECOVERY_POLICY = "checksum_verified_immutable_issues_only_no_metadata_inference"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_evaluation_state() -> str:
    try:
        result = subprocess.run(
            ["systemctl", "show", MODEL_EVALUATION_UNIT, "--property=ActiveState", "--value"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _append_event(path: Path, **event: object) -> None:
    payload = {"schema_version": 1, "updated_at_utc": _now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _successful_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Replay history is invalid at line {number}; preserve it for review") from exc
            if event.get("status") == "succeeded" and event.get("replay_key"):
                keys.add(str(event["replay_key"]))
    return keys


def _overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _replay_key(evidence: dict[str, str], code_digest: str, config_digest: str, archive_digest: str, ensemble_digest: str) -> str:
    payload = {
        "issue_content_digest": evidence["content_digest"],
        "implementation_digest": code_digest,
        "physical_config_digest": config_digest,
        "recovered_archive_digest": archive_digest,
        "ensemble_digest": ensemble_digest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _completed_candidate(results: dict[str, Path], root: Path, evidence: dict[str, str], *, require_ensemble: bool = False) -> bool:
    if set(results) != set(LANES):
        return False
    for path in results.values():
        resolved = Path(path).resolve()
        if root.resolve() not in resolved.parents or not resolved.is_dir():
            return False
    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if require_ensemble and any(
        status.get("lanes", {}).get(lane, {}).get("memberwise_ensemble", {}).get("status") != "complete"
        for lane in LANES
    ):
        return False
    return (
        status.get("status") == "complete"
        and status.get("baseline_issue_content_digest") == evidence["content_digest"]
        and status.get("baseline_publication_signature") == evidence["publication_signature"]
    )


def run_replay(
    *, issues_root: Path, recovered_archive_zarr: Path, candidate_root: Path,
    power_zarr: Path, physical_config: Path, max_issues: int = 1,
    max_seconds: float = 1200.0, maximum_snapshots: int = 10000,
    pdu_zarr: Path | None = None, asfs_zarr: Path | None = None,
    menapia_mqtt_log: Path | None = None, baseline_ensemble_zarr: Path | None = None,
) -> dict[str, object]:
    """Run up to max_issues; deferred and failed issues remain unprocessed."""
    if max_issues < 1 or max_seconds <= 0 or maximum_snapshots < 1:
        raise ValueError("Replay issue, runtime and inventory bounds must be positive")
    source, archive, root = map(Path, (issues_root, recovered_archive_zarr, candidate_root))
    inputs = [source, archive, Path(power_zarr), Path(physical_config)] + [
        Path(path) for path in (pdu_zarr, asfs_zarr, menapia_mqtt_log, baseline_ensemble_zarr) if path is not None
    ]
    if any(path.is_symlink() for path in (source, root)):
        raise ValueError("Replay roots must be direct directories")
    if any(_overlap(root, path) for path in inputs):
        raise ValueError("Replay candidate root overlaps a read-only input")
    if any(path.suffix == ".zarr" for path in (root, *root.parents)):
        raise ValueError("Replay candidate root must not be inside a Zarr store")
    if not source.is_dir() or not archive.is_dir():
        raise ValueError("Replay requires immutable issues and an explicit recovered archive")
    with xr.open_zarr(archive, chunks={}, consolidated=True) as opened:
        if opened.attrs.get("recovery_policy") != RECOVERY_POLICY:
            raise ValueError("Baseline archive must be produced by verified snapshot recovery")
        recovered_issues = set(zip(
            opened.issue_time.values.astype(str), opened.ForecastIdentityID.values.astype(str),
            opened.SourceCycleSetID.values.astype(str),
        ))
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / "replay_history.jsonl"
    if history_path.is_symlink():
        raise ValueError("Replay history must not be a symbolic link")
    lock_path = root / ".replay.lock"
    if lock_path.is_symlink():
        raise ValueError("Replay lock must not be a symbolic link")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "deferred", "reason": "replay_already_running", "processed": 0}
        started = time.monotonic()
        state = _model_evaluation_state()
        if state not in {"inactive", "failed"}:
            _append_event(history_path, status="deferred", reason="model_evaluation_not_idle", model_evaluation_state=state)
            return {"status": "deferred", "reason": "model_evaluation_not_idle", "processed": 0}
        code_digest = forecast_implementation_digest()
        config_digest = hashlib.sha256(Path(physical_config).read_bytes()).hexdigest()
        archive_digest = "sha256:" + _issue_snapshot_tree_digest(archive)[0]
        ensemble_digest = "absent"
        if baseline_ensemble_zarr is not None:
            ensemble_digest = "sha256:" + _issue_snapshot_tree_digest(Path(baseline_ensemble_zarr))[0]
        prior_successes = _successful_keys(history_path)
        directories = sorted(source.iterdir())
        if len(directories) > maximum_snapshots:
            raise ValueError("Immutable issue inventory exceeds maximum_snapshots")
        queue = []
        excluded = 0
        for directory in directories:
            try:
                _, evidence = verified_issue_snapshot(directory)
            except Exception as exc:
                excluded += 1
                _append_event(history_path, status="snapshot_rejected", source=str(directory), reason=f"{type(exc).__name__}: {exc}")
                continue
            if (evidence["issue_time"], evidence["forecast_identity_id"], evidence["source_cycle_set_id"]) not in recovered_issues:
                excluded += 1
                _append_event(history_path, status="snapshot_rejected", source=str(directory), reason="issue_not_in_recovered_archive")
                continue
            key = _replay_key(evidence, code_digest, config_digest, archive_digest, ensemble_digest)
            if key not in prior_successes:
                queue.append((evidence["issue_time"], directory, evidence, key))
        queue.sort(key=lambda entry: (entry[0], str(entry[1])))
        # A separate implementation tree prevents a code/config change from
        # mixing new replay results with the previous candidate campaign.
        work_root = root / "implementations" / code_digest.removeprefix("sha256:") / config_digest
        disabled = work_root / "absent_optional_sources"
        disabled.mkdir(parents=True, exist_ok=True)
        public_sources = disabled / "public_models"
        if public_sources.exists():
            raise ValueError("Offline replay public-model directory must remain absent")
        summary: dict[str, object] = {
            "status": "complete", "processed": 0, "failed": 0,
            "queued": len(queue), "rejected_snapshots": excluded,
            "implementation_digest": code_digest, "candidate_work_root": str(work_root),
            "physical_config_digest": config_digest,
        }
        for _, directory, evidence, key in queue[:max_issues]:
            if time.monotonic() - started >= max_seconds:
                summary.update(status="deferred", reason="runtime_budget_reached")
                break
            state = _model_evaluation_state()
            if state not in {"inactive", "failed"}:
                _append_event(history_path, status="deferred", replay_key=key, reason="model_evaluation_not_idle", model_evaluation_state=state)
                summary.update(status="deferred", reason="model_evaluation_not_idle")
                break
            context = {
                "replay_key": key, "implementation_digest": code_digest,
                "physical_config_digest": config_digest,
                "recovered_archive_digest": archive_digest, "ensemble_digest": ensemble_digest,
                **evidence,
            }
            _append_event(history_path, status="started", **context)
            try:
                # No latest-forecast default or source retrieval is used here.
                results = run_candidate(
                    baseline_issue_zarr=directory / "forecast.zarr",
                    baseline_archive_zarr=archive,
                    baseline_ensemble_zarr=baseline_ensemble_zarr,
                    candidate_root=work_root, power_zarr=Path(power_zarr),
                    physical_config=Path(physical_config),
                    pdu_zarr=Path(pdu_zarr) if pdu_zarr is not None else disabled / "pdu.zarr",
                    asfs_zarr=Path(asfs_zarr) if asfs_zarr is not None else disabled / "asfs.zarr",
                    menapia_mqtt_log=Path(menapia_mqtt_log) if menapia_mqtt_log is not None else disabled / "menapia.log",
                    public_source_manifest_root=public_sources,
                )
                if not results:
                    _append_event(history_path, status="deferred", reason="candidate_deferred", **context)
                    summary.update(status="deferred", reason="candidate_deferred")
                    break
                if not _completed_candidate(results, work_root, evidence, require_ensemble=baseline_ensemble_zarr is not None):
                    raise ValueError("Candidate did not confirm all lanes complete for this immutable issue")
            except Exception as exc:
                _append_event(history_path, status="failed", reason=f"{type(exc).__name__}: {exc}", **context)
                summary.update(status="failed", failed=1, reason=str(exc))
                break
            _append_event(history_path, status="succeeded", candidate_work_root=str(work_root), **context)
            summary["processed"] = int(summary["processed"]) + 1
        summary["pending"] = max(0, len(queue) - int(summary["processed"]))
        _append_event(history_path, status="run_summary", **{key: value for key, value in summary.items() if key != "status"}, run_status=summary["status"])
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues-root", required=True, type=Path)
    parser.add_argument("--recovered-archive-zarr", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--power-zarr", required=True, type=Path)
    parser.add_argument("--physical-config", required=True, type=Path)
    parser.add_argument("--pdu-zarr", type=Path)
    parser.add_argument("--asfs-zarr", type=Path)
    parser.add_argument("--menapia-mqtt-log", type=Path)
    parser.add_argument("--baseline-ensemble-zarr", type=Path)
    parser.add_argument("--max-issues", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--maximum-snapshots", type=int, default=10000)
    result = run_replay(**vars(parser.parse_args()))
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
