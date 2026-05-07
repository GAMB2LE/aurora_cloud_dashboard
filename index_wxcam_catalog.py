#!/usr/bin/env python3
"""Index Aurora wxcam HDR images into a lightweight SQLite catalog."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from wxcam_catalog import build_bootstrap_record, build_record, ensure_schema, existing_file_state, iter_raw_images, open_catalog, upsert_record

ROOT_DEFAULT = Path("/project/aurora/raw/wxcam")
CATALOG_DEFAULT = Path("/data/aurora/products/wxcam/wxcam_catalog.sqlite")


def _bootstrap_from_remote(
    root: Path,
    conn,
    source_user: str,
    source_host: str,
    source_path: str,
    fish_pattern: str,
    pano_pattern: str,
) -> int:
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityFile=none",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/home/aurora/.ssh/known_hosts",
        f"{source_user}@{source_host}",
        "bash",
        "-s",
        "--",
        source_path,
        fish_pattern,
        pano_pattern,
    ]
    remote_script = """\
set -euo pipefail
source_path="$1"
fish_pattern="$2"
pano_pattern="$3"
cd "$source_path"
find FISH -type f -name "$fish_pattern" -printf '%p\\t%s\\t%T@\\0'
find PANO -type f -name "$pano_pattern" -printf '%p\\t%s\\t%T@\\0'
"""
    inserted = 0
    process = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(remote_script.encode())
    process.stdin.close()

    buffer = b""
    while True:
        chunk = process.stdout.read(1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        while b"\0" in buffer:
            raw_record, buffer = buffer.split(b"\0", 1)
            if not raw_record:
                continue
            relative_path, size_bytes, mtime_seconds = raw_record.decode().split("\t")
            record = build_bootstrap_record(
                root,
                relative_path=relative_path,
                size_bytes=int(size_bytes),
                mtime_ns=int(float(mtime_seconds) * 1_000_000_000),
            )
            upsert_record(conn, record)
            inserted += 1
            if inserted % 1000 == 0:
                conn.commit()
    stderr = process.stderr.read().decode()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ssh_cmd, stderr=stderr)
    if buffer:
        relative_path, size_bytes, mtime_seconds = buffer.decode().split("\t")
        record = build_bootstrap_record(
            root,
            relative_path=relative_path,
            size_bytes=int(size_bytes),
            mtime_ns=int(float(mtime_seconds) * 1_000_000_000),
        )
        upsert_record(conn, record)
        inserted += 1
    conn.commit()
    return inserted


def index_catalog(
    root: Path,
    catalog_path: Path,
    bootstrap_source_user: str | None = None,
    bootstrap_source_host: str | None = None,
    bootstrap_source_path: str | None = None,
    bootstrap_fish_pattern: str = "HDR_*.jpg",
    bootstrap_pano_pattern: str = "HDR_*_PANO.jpg",
) -> None:
    catalog_exists = catalog_path.exists()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrapped = 0
    inserted = 0
    updated = 0
    skipped = 0

    with open_catalog(catalog_path) as conn:
        ensure_schema(conn)
        if (
            not catalog_exists
            and bootstrap_source_user
            and bootstrap_source_host
            and bootstrap_source_path
        ):
            print("Bootstrapping wxcam catalog from remote metadata...")
            bootstrapped = _bootstrap_from_remote(
                root,
                conn,
                source_user=bootstrap_source_user,
                source_host=bootstrap_source_host,
                source_path=bootstrap_source_path,
                fish_pattern=bootstrap_fish_pattern,
                pano_pattern=bootstrap_pano_pattern,
            )
        known = existing_file_state(conn)
        for image_type, path in iter_raw_images(root):
            absolute_path = str(path.resolve())
            stat_result = path.stat()
            current_state = (stat_result.st_size, stat_result.st_mtime_ns)
            if known.get(absolute_path) == current_state:
                skipped += 1
                continue
            try:
                record = build_record(root, image_type, path)
            except Exception as exc:
                print(f"Skipping unreadable wxcam image {path}: {exc}")
                continue
            upsert_record(conn, record)
            if absolute_path in known:
                updated += 1
            else:
                inserted += 1
        conn.commit()

    print(
        "wxcam catalog updated: "
        f"bootstrapped={bootstrapped} "
        f"inserted={inserted} updated={updated} skipped={skipped} "
        f"path={catalog_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Aurora wxcam HDR images into SQLite.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    parser.add_argument("--bootstrap-source-user", default=None)
    parser.add_argument("--bootstrap-source-host", default=None)
    parser.add_argument("--bootstrap-source-path", default=None)
    parser.add_argument("--bootstrap-fish-pattern", default="HDR_*.jpg")
    parser.add_argument("--bootstrap-pano-pattern", default="HDR_*_PANO.jpg")
    parser.add_argument("--rebuild", action="store_true", help="Delete the existing catalog before re-indexing.")
    args = parser.parse_args()

    if args.rebuild and args.catalog.exists():
        args.catalog.unlink()
        wal_path = args.catalog.with_suffix(args.catalog.suffix + "-wal")
        shm_path = args.catalog.with_suffix(args.catalog.suffix + "-shm")
        for extra in (wal_path, shm_path):
            if extra.exists():
                extra.unlink()
        print(f"Removed existing catalog: {args.catalog}")

    index_catalog(
        args.root,
        args.catalog,
        bootstrap_source_user=args.bootstrap_source_user,
        bootstrap_source_host=args.bootstrap_source_host,
        bootstrap_source_path=args.bootstrap_source_path,
        bootstrap_fish_pattern=args.bootstrap_fish_pattern,
        bootstrap_pano_pattern=args.bootstrap_pano_pattern,
    )


if __name__ == "__main__":
    main()
