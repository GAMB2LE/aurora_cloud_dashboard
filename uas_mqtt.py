"""Parse Menapia UAS MQTT dock-tier logs for the dashboard.

The producer writes two fields: the current tier of Dock 1 followed by the
current tier of Dock 2.  Older dashboard code called these ``reported`` and
``effective`` tier, which turned a two-dock state into one fictitious tier and
made mixed-dock power impossible to learn safely.  The compatibility aliases
below keep existing clients readable while all forecasting code uses the two
explicit dock fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


UAS_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s+"
    r"(?:(?P<event>Tier change)\s+)?"
    r"(?P<dock1_tier>-?\d+)\s+"
    r"(?P<dock2_tier>-?\d+)\s*$"
)


@dataclass(frozen=True)
class UASMqttRecord:
    timestamp: datetime
    event_type: str
    dock1_tier: int
    dock2_tier: int
    raw: str
    line_number: int

    @property
    def reported_tier(self) -> int:
        """Deprecated alias retained for old mobile clients: Dock 1 tier."""
        return self.dock1_tier

    @property
    def effective_tier(self) -> int:
        """Deprecated alias retained for old mobile clients: Dock 2 tier."""
        return self.dock2_tier

    @property
    def shared_tier(self) -> int | None:
        """Return a trainable single tier only when both docks agree exactly."""
        return self.dock1_tier if self.dock1_tier == self.dock2_tier else None

    @property
    def dock_pair_state(self) -> str:
        """Stable raw pair identity used for diagnostics and mixed-state exclusion."""
        return f"dock1_{self.dock1_tier}__dock2_{self.dock2_tier}"


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
        dock1_tier=int(match.group("dock1_tier")),
        dock2_tier=int(match.group("dock2_tier")),
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
