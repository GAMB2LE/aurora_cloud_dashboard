#!/usr/bin/env python3
"""Shared catalog helpers for Aurora wxcam image products."""

from __future__ import annotations

import calendar
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

WXCAM_IMAGE_TYPES: dict[str, dict[str, str]] = {
    "fish_hdr": {
        "label": "FISH HDR",
        "stream": "FISH",
        "glob": "HDR_*.jpg",
        "width": "3120",
        "height": "3040",
    },
    "pano_hdr": {
        "label": "PANO HDR",
        "stream": "PANO",
        "glob": "HDR_*_PANO.jpg",
        "width": "2880",
        "height": "750",
    },
}

_TIMESTAMP_REGEX = re.compile(r"^HDR_(\d{8})_(\d{6})(?:_PANO)?\.jpg$")


@dataclass(frozen=True)
class WxcamRecord:
    image_type: str
    stream: str
    label: str
    time_utc: str
    time_epoch_ns: int
    day_utc: str
    raw_path: str
    relative_path: str
    filename: str
    width: int
    height: int
    size_bytes: int
    mtime_ns: int
    indexed_at: str


def wxcam_label(image_type: str) -> str:
    return WXCAM_IMAGE_TYPES[image_type]["label"]


def wxcam_stream(image_type: str) -> str:
    return WXCAM_IMAGE_TYPES[image_type]["stream"]


def wxcam_shape(image_type: str) -> tuple[int, int]:
    spec = WXCAM_IMAGE_TYPES[image_type]
    return int(spec["width"]), int(spec["height"])


def open_catalog(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            raw_path TEXT PRIMARY KEY,
            image_type TEXT NOT NULL,
            stream TEXT NOT NULL,
            label TEXT NOT NULL,
            time_utc TEXT NOT NULL,
            time_epoch_ns INTEGER NOT NULL,
            day_utc TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_images_type_time ON images (image_type, time_epoch_ns);
        CREATE INDEX IF NOT EXISTS idx_images_type_day ON images (image_type, day_utc);
        """
    )


def parse_timestamp(path: Path) -> datetime | None:
    match = _TIMESTAMP_REGEX.match(path.name)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def iter_raw_images(root: Path, image_type: str | None = None) -> Iterable[tuple[str, Path]]:
    items = [(image_type, WXCAM_IMAGE_TYPES[image_type])] if image_type else list(WXCAM_IMAGE_TYPES.items())
    for current_type, spec in items:
        stream_root = root / spec["stream"]
        if not stream_root.exists():
            continue
        for path in sorted(stream_root.rglob(spec["glob"])):
            if path.is_file():
                yield current_type, path


def image_type_from_relative_path(relative_path: str) -> str:
    rel = relative_path.strip("/")
    if rel.startswith("FISH/"):
        return "fish_hdr"
    if rel.startswith("PANO/"):
        return "pano_hdr"
    raise ValueError(f"Could not infer wxcam image type from {relative_path}")


def build_record(root: Path, image_type: str, path: Path) -> WxcamRecord:
    timestamp = parse_timestamp(path)
    if timestamp is None:
        raise ValueError(f"Could not parse wxcam timestamp from {path.name}")

    with Image.open(path) as image:
        width, height = image.size

    stat_result = path.stat()
    absolute_path = str(path.resolve())
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    time_epoch_ns = calendar.timegm(timestamp.utctimetuple()) * 1_000_000_000
    indexed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return WxcamRecord(
        image_type=image_type,
        stream=wxcam_stream(image_type),
        label=wxcam_label(image_type),
        time_utc=timestamp.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds"),
        time_epoch_ns=time_epoch_ns,
        day_utc=timestamp.strftime("%Y-%m-%d"),
        raw_path=absolute_path,
        relative_path=relative_path,
        filename=path.name,
        width=width,
        height=height,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        indexed_at=indexed_at,
    )


def build_bootstrap_record(root: Path, relative_path: str, size_bytes: int, mtime_ns: int) -> WxcamRecord:
    rel_path = Path(relative_path)
    image_type = image_type_from_relative_path(relative_path)
    timestamp = parse_timestamp(rel_path)
    if timestamp is None:
        raise ValueError(f"Could not parse wxcam timestamp from {relative_path}")
    width, height = wxcam_shape(image_type)
    local_path = (root / rel_path).resolve()
    indexed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    time_epoch_ns = calendar.timegm(timestamp.utctimetuple()) * 1_000_000_000
    return WxcamRecord(
        image_type=image_type,
        stream=wxcam_stream(image_type),
        label=wxcam_label(image_type),
        time_utc=timestamp.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds"),
        time_epoch_ns=time_epoch_ns,
        day_utc=timestamp.strftime("%Y-%m-%d"),
        raw_path=str(local_path),
        relative_path=rel_path.as_posix(),
        filename=rel_path.name,
        width=width,
        height=height,
        size_bytes=int(size_bytes),
        mtime_ns=int(mtime_ns),
        indexed_at=indexed_at,
    )


def upsert_record(conn: sqlite3.Connection, record: WxcamRecord) -> None:
    conn.execute(
        """
        INSERT INTO images (
            raw_path, image_type, stream, label, time_utc, time_epoch_ns, day_utc,
            relative_path, filename, width, height, size_bytes, mtime_ns, indexed_at
        ) VALUES (
            :raw_path, :image_type, :stream, :label, :time_utc, :time_epoch_ns, :day_utc,
            :relative_path, :filename, :width, :height, :size_bytes, :mtime_ns, :indexed_at
        )
        ON CONFLICT(raw_path) DO UPDATE SET
            image_type=excluded.image_type,
            stream=excluded.stream,
            label=excluded.label,
            time_utc=excluded.time_utc,
            time_epoch_ns=excluded.time_epoch_ns,
            day_utc=excluded.day_utc,
            relative_path=excluded.relative_path,
            filename=excluded.filename,
            width=excluded.width,
            height=excluded.height,
            size_bytes=excluded.size_bytes,
            mtime_ns=excluded.mtime_ns,
            indexed_at=excluded.indexed_at
        """,
        record.__dict__,
    )


def existing_file_state(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    return {
        row["raw_path"]: (int(row["size_bytes"]), int(row["mtime_ns"]))
        for row in conn.execute("SELECT raw_path, size_bytes, mtime_ns FROM images")
    }


def catalog_time_bounds(path: Path) -> tuple[datetime | None, datetime | None]:
    if not path.exists():
        return None, None
    with open_catalog(path) as conn:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT MIN(time_epoch_ns) AS min_ns, MAX(time_epoch_ns) AS max_ns FROM images"
        ).fetchone()
    if row is None or row["min_ns"] is None or row["max_ns"] is None:
        return None, None
    return ns_to_datetime(int(row["min_ns"])), ns_to_datetime(int(row["max_ns"]))


def latest_record(path: Path, image_type: str) -> sqlite3.Row | None:
    if not path.exists():
        return None
    with open_catalog(path) as conn:
        ensure_schema(conn)
        return conn.execute(
            """
            SELECT *
            FROM images
            WHERE image_type = ?
            ORDER BY time_epoch_ns DESC
            LIMIT 1
            """,
            (image_type,),
        ).fetchone()


def latest_record_before(path: Path, image_type: str, end: datetime | None) -> sqlite3.Row | None:
    if end is None or not path.exists():
        return latest_record(path, image_type)
    with open_catalog(path) as conn:
        ensure_schema(conn)
        return conn.execute(
            """
            SELECT *
            FROM images
            WHERE image_type = ? AND time_epoch_ns <= ?
            ORDER BY time_epoch_ns DESC
            LIMIT 1
            """,
            (image_type, datetime_to_ns(end)),
        ).fetchone()


def latest_record_in_window(path: Path, image_type: str, start: datetime | None, end: datetime | None) -> sqlite3.Row | None:
    if not path.exists():
        return None
    if start is None or end is None:
        return latest_record(path, image_type)
    with open_catalog(path) as conn:
        ensure_schema(conn)
        row = conn.execute(
            """
            SELECT *
            FROM images
            WHERE image_type = ?
              AND time_epoch_ns >= ?
              AND time_epoch_ns <= ?
            ORDER BY time_epoch_ns DESC
            LIMIT 1
            """,
            (image_type, datetime_to_ns(start), datetime_to_ns(end)),
        ).fetchone()
    return row


def daily_latest_records(path: Path, image_type: str) -> list[sqlite3.Row]:
    if not path.exists():
        return []
    with open_catalog(path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT i.*
            FROM images AS i
            JOIN (
                SELECT day_utc, MAX(time_epoch_ns) AS max_ns
                FROM images
                WHERE image_type = ?
                GROUP BY day_utc
            ) AS latest
              ON i.day_utc = latest.day_utc
             AND i.time_epoch_ns = latest.max_ns
            WHERE i.image_type = ?
            ORDER BY i.day_utc
            """,
            (image_type, image_type),
        ).fetchall()
    return rows


def catalog_frontier(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with open_catalog(path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT image_type, MAX(time_epoch_ns) AS max_ns
            FROM images
            GROUP BY image_type
            """
        ).fetchall()
    return {row["image_type"]: int(row["max_ns"]) for row in rows if row["max_ns"] is not None}


def records_after(path: Path, image_type: str, time_epoch_ns: int) -> list[sqlite3.Row]:
    if not path.exists():
        return []
    with open_catalog(path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM images
            WHERE image_type = ? AND time_epoch_ns > ?
            ORDER BY time_epoch_ns ASC
            """,
            (image_type, int(time_epoch_ns)),
        ).fetchall()
    return rows


def ns_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).replace(tzinfo=None)


def datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return calendar.timegm(value.utctimetuple()) * 1_000_000_000 + value.microsecond * 1_000
