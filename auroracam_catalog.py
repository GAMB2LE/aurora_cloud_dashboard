#!/usr/bin/env python3
"""Catalog helpers for AURORACam/MX4 camera JPEG archives."""

from __future__ import annotations

import calendar
from functools import lru_cache
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

AURORACAM_CAMERAS: dict[str, dict[str, str]] = {
    "end-south-array-cam": {
        "label": "End South Array",
        "ip": "192.168.1.27",
    },
    "fence-post-cam": {
        "label": "Fence Post",
        "ip": "192.168.1.28",
    },
    "radar-cam": {
        "label": "Radar",
        "ip": "192.168.1.29",
    },
    "mid-south-array-cam": {
        "label": "Mid South Array",
        "ip": "192.168.1.30",
    },
}

_FILENAME_REGEX = re.compile(
    r"^(?P<camera>[a-z0-9-]+)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})-(?P<minute>\d{2})\.jpe?g$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuroracamRecord:
    camera_id: str
    label: str
    ip: str
    time_utc: str
    time_epoch_ns: int
    day_utc: str
    raw_path: str
    relative_path: str
    filename: str
    size_bytes: int
    mtime_ns: int


def parse_timestamp(path: Path) -> datetime | None:
    match = _FILENAME_REGEX.match(path.name)
    if not match:
        return None
    day = match.group("day")
    hour = match.group("hour")
    minute = match.group("minute")
    try:
        dt = datetime.strptime(f"{day} {hour}:{minute}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def camera_from_filename(path: Path) -> str | None:
    match = _FILENAME_REGEX.match(path.name)
    if not match:
        return None
    camera_id = match.group("camera")
    return camera_id if camera_id in AURORACAM_CAMERAS else None


def _iter_camera_files(root: Path, camera_id: str) -> Iterable[Path]:
    camera_root = root / camera_id
    if not camera_root.exists():
        return
    for path in sorted(camera_root.rglob("*.jpg")):
        if path.is_file():
            yield path


def _root_key(root: Path) -> str:
    root = Path(root)
    if root.is_absolute():
        return str(root)
    return str(root.resolve())


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _available_days_fingerprint(root: Path, camera_id: str | None) -> tuple[tuple[str, int], ...]:
    cameras = [camera_id] if camera_id else list(AURORACAM_CAMERAS)
    return tuple(
        (current_camera, _path_mtime_ns(root / current_camera))
        for current_camera in cameras
        if current_camera in AURORACAM_CAMERAS
    )


@lru_cache(maxsize=64)
def _available_days_cached(root_key: str, camera_id: str | None, fingerprint: tuple[tuple[str, int], ...]) -> tuple[str, ...]:
    root = Path(root_key)
    days: set[str] = set()
    cameras = [camera_id] if camera_id else list(AURORACAM_CAMERAS)
    for current_camera in cameras:
        if current_camera not in AURORACAM_CAMERAS:
            continue
        camera_root = root / current_camera
        if not camera_root.exists():
            continue
        for day_dir in camera_root.iterdir():
            if not day_dir.is_dir():
                continue
            day = day_dir.name
            if len(day) == 10 and day[4] == "-" and day[7] == "-":
                days.add(day)
    return tuple(sorted(days))


@lru_cache(maxsize=256)
def _day_records_cached(root_key: str, camera_id: str, day_utc: str, day_mtime_ns: int) -> tuple[AuroracamRecord, ...]:
    root = Path(root_key)
    day_dir = root / camera_id / day_utc
    if not day_dir.exists():
        return ()
    records: list[AuroracamRecord] = []
    for path in sorted(day_dir.glob("*.jpg")):
        if not path.is_file():
            continue
        try:
            records.append(build_record(root, path))
        except ValueError:
            continue
    return tuple(sorted(records, key=lambda record: (record.time_epoch_ns, record.filename)))


def iter_image_paths(root: Path, camera_id: str | None = None) -> Iterable[Path]:
    cameras = [camera_id] if camera_id else list(AURORACAM_CAMERAS)
    for current_camera in cameras:
        if current_camera not in AURORACAM_CAMERAS:
            continue
        yield from _iter_camera_files(root, current_camera)


def build_record(root: Path, path: Path) -> AuroracamRecord:
    root = Path(root)
    path = Path(path)
    root_abs = root if root.is_absolute() else root.resolve()
    path_abs = path if path.is_absolute() else path.resolve()
    timestamp = parse_timestamp(path)
    if timestamp is None:
        raise ValueError(f"Could not parse AURORACam timestamp from {path.name}")
    camera_id = camera_from_filename(path)
    if camera_id is None:
        raise ValueError(f"Could not parse AURORACam camera from {path.name}")
    if path.parent.parent.name != camera_id:
        raise ValueError(f"Camera folder {path.parent.parent.name} does not match filename {path.name}")
    day_utc = timestamp.strftime("%Y-%m-%d")
    if path.parent.name != day_utc:
        raise ValueError(f"Day folder {path.parent.name} does not match filename {path.name}")

    spec = AURORACAM_CAMERAS[camera_id]
    stat_result = path.stat()
    try:
        relative_path = path_abs.relative_to(root_abs).as_posix()
    except ValueError:
        relative_path = path_abs.resolve().relative_to(root_abs.resolve()).as_posix()
    time_epoch_ns = calendar.timegm(timestamp.utctimetuple()) * 1_000_000_000
    return AuroracamRecord(
        camera_id=camera_id,
        label=spec["label"],
        ip=spec["ip"],
        time_utc=timestamp.replace(tzinfo=None).isoformat(sep=" ", timespec="minutes"),
        time_epoch_ns=time_epoch_ns,
        day_utc=day_utc,
        raw_path=str(path_abs),
        relative_path=relative_path,
        filename=path.name,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )


def iter_image_records(root: Path, camera_id: str | None = None) -> Iterable[AuroracamRecord]:
    for path in iter_image_paths(root, camera_id):
        try:
            yield build_record(root, path)
        except ValueError:
            continue


def available_days(root: Path, camera_id: str | None = None) -> list[str]:
    root = Path(root)
    return list(_available_days_cached(_root_key(root), camera_id, _available_days_fingerprint(root, camera_id)))


def day_records(root: Path, camera_id: str, day_utc: str) -> list[AuroracamRecord]:
    if camera_id not in AURORACAM_CAMERAS:
        return []
    root = Path(root)
    day_dir = root / camera_id / day_utc
    return list(_day_records_cached(_root_key(root), camera_id, day_utc, _path_mtime_ns(day_dir)))


def latest_records(root: Path, day_utc: str | None = None) -> dict[str, AuroracamRecord]:
    latest: dict[str, AuroracamRecord] = {}
    for camera_id in AURORACAM_CAMERAS:
        record = latest_record(root, camera_id, day_utc)
        if record is not None:
            latest[camera_id] = record
    return {
        camera_id: latest[camera_id]
        for camera_id in AURORACAM_CAMERAS
        if camera_id in latest
    }


def latest_record(root: Path, camera_id: str, day_utc: str | None = None) -> AuroracamRecord | None:
    if camera_id not in AURORACAM_CAMERAS:
        return None
    if day_utc is None:
        days = available_days(root, camera_id)
        if not days:
            return None
        day_utc = days[-1]
    records = day_records(root, camera_id, day_utc)
    if not records:
        return None
    return max(records, key=lambda record: (record.time_epoch_ns, record.filename))


def representative_hourly_records(root: Path, camera_id: str, day_utc: str) -> dict[int, AuroracamRecord]:
    rows = day_records(root, camera_id, day_utc)
    if not rows:
        return {}
    by_hour: dict[int, list[AuroracamRecord]] = {}
    for record in rows:
        dt = datetime.fromisoformat(record.time_utc).replace(tzinfo=timezone.utc)
        by_hour.setdefault(dt.hour, []).append(record)

    representatives: dict[int, AuroracamRecord] = {}
    for hour, records in by_hour.items():
        target_minute = 30
        representatives[hour] = min(
            records,
            key=lambda record: (
                abs(datetime.fromisoformat(record.time_utc).minute - target_minute),
                -record.time_epoch_ns,
            ),
        )
    return representatives


def record_dict(record: AuroracamRecord) -> dict[str, object]:
    return {
        "camera_id": record.camera_id,
        "label": record.label,
        "ip": record.ip,
        "time_utc": record.time_utc,
        "time_epoch_ns": record.time_epoch_ns,
        "day_utc": record.day_utc,
        "raw_path": record.raw_path,
        "relative_path": record.relative_path,
        "filename": record.filename,
        "size_bytes": record.size_bytes,
        "mtime_ns": record.mtime_ns,
    }
