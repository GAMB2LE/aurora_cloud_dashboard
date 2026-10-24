"""Content-based forecast implementation identity, independent of UI releases.

The full repository revision remains issue provenance. This narrower digest
is a campaign boundary, not a replacement for source or configuration hashes.
Missing dependencies fail closed; no environment-supplied override is accepted.
"""
from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path


FORECAST_IMPLEMENTATION_FILES = (
    "generate_power_soc_forecast.py", "generate_power_soc_ensemble.py",
    "generate_power_soc_v12_candidate.py", "generate_power_soc_physical_candidate.py",
    "power_archive_io.py", "power_battery_model.py", "power_solar_model.py",
    "power_load_contract.py", "power_load_dynamics.py", "power_state_catalog.py",
    "power_soc_thresholds.py", "power_issue_time_features.py", "power_v12_hybrid.py",
    "power_v12_ensemble.py", "power_observation_truth.py", "power_prod_dev_evaluation.py",
    "power_implementation_identity.py", "power_public_source_ablation.py",
    "run_power_candidate_replay.py", "repair_power_forecast_archive.py",
    "evaluate_power_diagnostic_replays.py",
    "generate_power_prod_dev_evaluation.py",
    "requirements-runtime.txt",
)


def forecast_implementation_digest(root: Path | None = None) -> str:
    directory = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = {}
    pending = list(FORECAST_IMPLEMENTATION_FILES)
    while pending:
        name = pending.pop()
        if name in manifest:
            continue
        content = (directory / name).read_bytes()
        manifest[name] = hashlib.sha256(content).hexdigest()
        if name.endswith(".py"):
            # Include transitive local imports, including imports inside
            # functions. A provider/load-model change is a new implementation
            # even if the top-level forecast module itself did not change.
            for node in ast.walk(ast.parse(content, filename=name)):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for module in modules:
                    local = module.replace(".", "/") + ".py"
                    if (directory / local).is_file() and local not in manifest:
                        pending.append(local)
    return "sha256:" + hashlib.sha256(json.dumps(
        {"schema": 2, "files": manifest}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def semantic_code_key(attrs: dict | object) -> str:
    """Never pool legacy issues without their original complete code identity."""
    digest = str(attrs.get("forecast_implementation_digest", "")).strip()
    if digest:
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("Malformed forecast implementation digest")
        int(digest[7:], 16)
        return digest
    return "legacy-revision:" + str(attrs.get("forecast_code_revision", "")).strip()
