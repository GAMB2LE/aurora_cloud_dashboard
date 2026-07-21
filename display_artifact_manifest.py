"""Create and read bounded manifests for browser-ready dashboard artifacts.

The authoritative raw data and Zarr products remain outside this module.  A
manifest describes only derived media and precomputed browser artifacts, which
can safely be published to a CDN or object store without exposing raw inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile
from typing import Iterable


UTC = timezone.utc
MANIFEST_VERSION = 1
DEFAULT_SUFFIXES = frozenset({".json", ".png", ".jpg", ".jpeg", ".webp", ".avif", ".mp4"})


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def default_manifest_path() -> Path:
    return Path(
        os.environ.get(
            "AURORA_DISPLAY_ARTIFACT_MANIFEST",
            "/data/aurora/products/dashboard/display_artifacts/latest.json",
        )
    )


def _file_record(path: Path, root: Path) -> dict[str, object]:
    stat_result = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "sizeBytes": stat_result.st_size,
        "modifiedAt": datetime.fromtimestamp(stat_result.st_mtime, UTC).isoformat().replace("+00:00", "Z"),
    }


def _iter_files(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    if not root.exists() or not root.is_dir():
        return ()
    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in allowed
    )


def build_manifest(
    groups: dict[str, Path],
    *,
    max_files_per_group: int = 20_000,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
) -> dict[str, object]:
    """Return a bounded, deterministic manifest for publishable artifacts."""
    payload_groups: dict[str, object] = {}
    digest = hashlib.sha256()
    total_files = 0
    total_bytes = 0
    for name, root in sorted(groups.items()):
        records: list[dict[str, object]] = []
        truncated = False
        for path in _iter_files(root, suffixes):
            if len(records) >= max_files_per_group:
                truncated = True
                break
            record = _file_record(path, root)
            records.append(record)
            total_files += 1
            total_bytes += int(record["sizeBytes"])
            digest.update(name.encode("utf-8"))
            digest.update(str(record["path"]).encode("utf-8"))
            digest.update(str(record["sizeBytes"]).encode("ascii"))
            digest.update(str(record["modifiedAt"]).encode("ascii"))
        payload_groups[name] = {
            "sourceRoot": str(root),
            "exists": root.is_dir(),
            "truncated": truncated,
            "files": records,
        }
    return {
        "schemaVersion": MANIFEST_VERSION,
        "generatedAt": utc_now(),
        "contentRevision": digest.hexdigest(),
        "artifactCount": total_files,
        "totalBytes": total_bytes,
        "groups": payload_groups,
    }


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish a complete manifest without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_manifest(path: Path | None = None) -> dict[str, object]:
    target = path or default_manifest_path()
    try:
        with target.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {"available": False, "path": str(target)}
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "path": str(target), "error": str(exc)}
    if not isinstance(value, dict):
        return {"available": False, "path": str(target), "error": "Manifest root is not an object"}
    return {"available": True, "path": str(target), **value}
