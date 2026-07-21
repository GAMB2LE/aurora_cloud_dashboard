#!/usr/bin/env python3
"""Publish a bounded manifest of derived artifacts suitable for CDN delivery."""

from __future__ import annotations

import argparse
from pathlib import Path

from display_artifact_manifest import build_manifest, default_manifest_path, write_manifest


def _group(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("groups must use NAME=PATH")
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_manifest_path())
    parser.add_argument("--group", action="append", type=_group, default=[])
    parser.add_argument("--max-files-per-group", type=int, default=20_000)
    args = parser.parse_args()
    groups = dict(args.group)
    if not groups:
        parser.error("at least one --group NAME=PATH is required")
    payload = build_manifest(groups, max_files_per_group=max(1, args.max_files_per_group))
    write_manifest(args.output, payload)
    print(f"wrote {args.output} with {payload['artifactCount']} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
