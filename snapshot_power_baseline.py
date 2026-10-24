#!/usr/bin/env python3
"""Preserve a bounded forecast baseline in a new, checksum-verified tree.

Only explicit forecast products are copied. Raw telemetry, weather retrieval
caches, locks, temporary stores, and unrelated products are never included.
Configuration references are recorded as labels; no environment file is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from generate_power_soc_forecast import (
    ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
    ISSUE_SNAPSHOT_DIGEST_MARKER,
    _issue_snapshot_tree_digest,
)


BASELINE_PRODUCTS = (
    "power_soc_forecast.zarr", "power_soc_forecast_archive.zarr",
    "power_soc_forecast_state.json", "power_soc_forecast_skill.zarr",
    "power_soc_hindcast.zarr", "power_soc_ensemble_forecast.zarr",
    "power_soc_ensemble_archive.zarr", "power_soc_ensemble_skill.zarr",
    "power_soc_planning_forecast.zarr", "power_soc_planning_archive.zarr",
    "power_soc_planning_state.json", "power_soc_planning_skill.zarr",
    "power_soc_planning_hindcast.zarr", "power_operating_model_state.json",
    "power_operating_state.zarr", "power_operating_scenarios.zarr",
    "power_operating_recommendations.json", "forecast_bundle_status.json",
)
REQUIRED_PRODUCTS = (
    "power_soc_forecast.zarr", "power_soc_forecast_archive.zarr",
    "power_soc_forecast_state.json",
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def product_record(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"Baseline product must not be a symbolic link: {path.name}")
    if path.is_dir():
        digest, count, size = _issue_snapshot_tree_digest(path)
        record: dict[str, object] = {
            "type": "directory", "digest_algorithm": ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
            "content_digest": "sha256:" + digest, "file_count": count, "byte_count": size,
        }
        # The reused snapshot-tree algorithm excludes its own marker. Include
        # that file independently if a selected product happens to have one.
        marker = path / ISSUE_SNAPSHOT_DIGEST_MARKER
        if marker.exists():
            record["snapshot_marker_digest"] = _file_digest(marker)
            record["file_count"] = count + 1
            record["byte_count"] = size + marker.stat().st_size
        return record
    if path.is_file():
        return {
            "type": "file", "digest_algorithm": "sha256-file-bytes-v1",
            "content_digest": _file_digest(path), "file_count": 1,
            "byte_count": path.stat().st_size,
        }
    raise ValueError(f"Baseline product is missing or unsupported: {path.name}")


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        name: product_record(root / name)
        for name in BASELINE_PRODUCTS
        if (root / name).exists() or (root / name).is_symlink()
    }


def _copy_product(source: Path, destination: Path) -> None:
    if source.is_dir():
        # Copy any link created during a race as a link, never dereference it.
        # The post-copy checksum verification will then reject it.
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def snapshot_baseline(
    source_root: Path, output_root: Path, *, code_revision: str | None = None,
    config_references: list[str] | None = None,
) -> dict[str, object]:
    source, output = Path(source_root), Path(output_root)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Baseline source must be a direct directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Baseline snapshot requires a new output directory")
    source_resolved, output_resolved = source.resolve(), output.resolve()
    if (source_resolved == output_resolved or source_resolved in output_resolved.parents
            or output_resolved in source_resolved.parents):
        raise ValueError("Baseline snapshot output must not overlap its source")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    manifest: dict[str, object] = {
        "schema_version": 1, "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_resolved), "output_root": str(output_resolved),
        "staging_root": str(staging.resolve()), "code_revision": code_revision,
        "configuration_references": list(config_references or []),
        "configuration_reference_policy": "labels_only_no_environment_files_read",
        "allowlisted_products": list(BASELINE_PRODUCTS), "products": {},
    }
    try:
        before = _inventory(source)
        manifest["products"] = before
        manifest["missing_optional_products"] = sorted(set(BASELINE_PRODUCTS) - set(before))
        missing = sorted(set(REQUIRED_PRODUCTS) - set(before))
        if missing:
            raise ValueError("Missing required baseline products: " + ", ".join(missing))
        for name in before:
            _copy_product(source / name, staging / name)
        # Check the entire source again after all copies, not merely each file
        # immediately after copying; this detects cross-product publication races.
        after = _inventory(source)
        copied = _inventory(staging)
        if before != after:
            changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
            manifest["changed_source_products"] = changed
            raise ValueError("Baseline source changed during snapshot: " + ", ".join(changed))
        if before != copied:
            changed = sorted(name for name in set(before) | set(copied) if before.get(name) != copied.get(name))
            manifest["copy_mismatch_products"] = changed
            raise ValueError("Copied baseline bytes do not match source: " + ", ".join(changed))
        manifest.update(
            status="complete", verification="source_before_equals_source_after_equals_copy",
            verified_at_utc=datetime.now(timezone.utc).isoformat(),
            product_count=len(before), total_bytes=sum(int(row["byte_count"]) for row in before.values()),
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Baseline output appeared during staging; refusing replacement")
        staging.rename(output)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", reason=f"{type(exc).__name__}: {exc}", diagnostic_staging_preserved=True)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--code-revision")
    parser.add_argument("--config-reference", action="append", default=[])
    args = parser.parse_args()
    result = snapshot_baseline(
        args.source_root, args.output_root, code_revision=args.code_revision,
        config_references=args.config_reference,
    )
    print(json.dumps({key: result.get(key) for key in ("status", "output_root", "staging_root", "product_count", "reason")}, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
