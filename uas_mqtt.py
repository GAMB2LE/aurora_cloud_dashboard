"""Parse Menapia UAS MQTT tier logs for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


UAS_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s+"
    r"(?:(?P<event>Tier change)\s+)?"
    r"(?P<reported_tier>-?\d+)\s+"
    r"(?P<effective_tier>-?\d+)\s*$"
)


@dataclass(frozen=True)
class UASMqttRecord:
    timestamp: datetime
    event_type: str
    reported_tier: int
    effective_tier: int
    raw: str
    line_number: int


@dataclass(frozen=True)
class UASMqttParseResult:
    path: Path
    records: tuple[UASMqttRecord, ...]
    malformed_lines: tuple[str, ...]
    missing: bool = False
    error: str | None = None


def parse_uas_mqtt_line(line: str, line_number: int = 0) -> UASMqttRecord | None:
    """Parse one Menapia MQTT log line, returning None for malformed input."""
    raw = line.rstrip("\n")
    match = UAS_LOG_LINE_RE.match(raw.strip())
    if match is None:
        return None
    timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    return UASMqttRecord(
        timestamp=timestamp,
        event_type="tier_change" if match.group("event") else "sample",
        reported_tier=int(match.group("reported_tier")),
        effective_tier=int(match.group("effective_tier")),
        raw=raw,
        line_number=line_number,
    )


def load_uas_mqtt_log(path: Path | str, max_lines: int = 5000) -> UASMqttParseResult:
    """Load and parse the tail of the UAS MQTT log."""
    log_path = Path(path)
    if not log_path.exists():
        return UASMqttParseResult(path=log_path, records=(), malformed_lines=(), missing=True)
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return UASMqttParseResult(path=log_path, records=(), malformed_lines=(), error=str(exc))

    if max_lines > 0 and len(lines) > max_lines:
        start_line = len(lines) - max_lines + 1
        selected = lines[-max_lines:]
    else:
        start_line = 1
        selected = lines

    records: list[UASMqttRecord] = []
    malformed: list[str] = []
    for offset, line in enumerate(selected):
        line_number = start_line + offset
        if not line.strip():
            continue
        record = parse_uas_mqtt_line(line, line_number=line_number)
        if record is None:
            malformed.append(f"{line_number}: {line}")
        else:
            records.append(record)
    records.sort(key=lambda item: item.timestamp)
    return UASMqttParseResult(path=log_path, records=tuple(records), malformed_lines=tuple(malformed))
