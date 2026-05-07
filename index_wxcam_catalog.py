#!/usr/bin/env python3
"""Index Aurora wxcam HDR images into a lightweight SQLite catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from wxcam_catalog import build_record, ensure_schema, existing_file_state, iter_raw_images, open_catalog, upsert_record

ROOT_DEFAULT = Path("/project/aurora/raw/wxcam")
CATALOG_DEFAULT = Path("/data/aurora/products/wxcam/wxcam_catalog.sqlite")


def index_catalog(root: Path, catalog_path: Path) -> None:
    if not root.exists():
        print(f"wxcam raw root does not exist: {root}")
        return

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    inserted = 0
    updated = 0
    skipped = 0

    with open_catalog(catalog_path) as conn:
        ensure_schema(conn)
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
        f"inserted={inserted} updated={updated} skipped={skipped} "
        f"path={catalog_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Aurora wxcam HDR images into SQLite.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
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

    index_catalog(args.root, args.catalog)


if __name__ == "__main__":
    main()
