#!/usr/bin/env python3
"""Summarize Aurora dashboard performance logs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path


DEFAULT_LOG = Path(os.environ.get("AURORA_DASHBOARD_PERF_LOG", "/data/aurora/products/dashboard/dashboard_perf.jsonl"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Aurora dashboard performance logs.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--hours", type=float, default=24.0, help="Only include events from the last N hours.")
    parser.add_argument("--event", default=None, help="Optional exact event name filter.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum slow examples to print.")
    return parser.parse_args()


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _load_records(path: Path, hours: float, event_filter: str | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(str(row.get("ts_utc", "")))
            if ts is None or ts < cutoff:
                continue
            if event_filter and row.get("event") != event_filter:
                continue
            rows.append(row)
    return rows


def _print_group_summary(records: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        duration = row.get("duration_ms")
        if duration is None:
            continue
        key = (str(row.get("event", "")), str(row.get("instrument", "")))
        grouped[key].append(float(duration))
    if not grouped:
        print("No timing events matched.")
        return
    print("Event summary")
    print("=============")
    for (event, instrument), durations in sorted(grouped.items()):
        avg = sum(durations) / len(durations)
        p50 = _quantile(durations, 0.50)
        p95 = _quantile(durations, 0.95)
        max_v = max(durations)
        print(
            f"{event:28} instrument={instrument:18} "
            f"count={len(durations):5d} avg={avg:8.1f}ms p50={p50:8.1f}ms p95={p95:8.1f}ms max={max_v:8.1f}ms"
        )


def _print_session_summary(records: list[dict]) -> None:
    session_ids = {row.get("session_id") for row in records if row.get("session_id")}
    loaded = [row for row in records if row.get("event") == "session_loaded"]
    destroyed = [row for row in records if row.get("event") == "session_destroyed"]
    heartbeats = [row for row in records if row.get("event") == "session_heartbeat"]
    live_counts = [int(row["live_sessions"]) for row in records if row.get("live_sessions") is not None]
    server_counts = [int(row["server_sessions"]) for row in records if row.get("server_sessions") is not None]
    control_changes: Counter[tuple[str, str]] = Counter(
        (str(row.get("instrument", "")), str(row.get("control", "")))
        for row in records
        if row.get("event") == "ui_selection_change"
    )
    print("\nSession summary")
    print("===============")
    print(f"unique session ids: {len(session_ids)}")
    print(f"sessions loaded:   {len(loaded)}")
    print(f"sessions closed:   {len(destroyed)}")
    print(f"heartbeats:        {len(heartbeats)}")
    if live_counts:
        print(f"max live sessions:   {max(live_counts)}")
    if server_counts:
        print(f"max server sessions: {max(server_counts)}")
    if control_changes:
        print("\nTop control changes")
        for (instrument, control), count in control_changes.most_common(10):
            print(f"{instrument:18} {control:24} {count}")


def _print_instrument_coverage(records: list[dict]) -> None:
    browse_counts: Counter[str] = Counter()
    for row in records:
        event = str(row.get("event", ""))
        instrument = str(row.get("instrument", ""))
        # The WXcam Science Quicklook grid kept its original internal
        # wxcam_calendar_* event names so older performance logs remain
        # comparable.
        if event in {
            "interactive_view_update",
            "stacked_timeseries_render",
            "science_quicklook_render",
            "housekeeping_quicklook_render",
            "wxcam_interactive_render",
            "wxcam_calendar_options",
            "wxcam_calendar_day_view",
            "operations_dashboard_render",
        }:
            browse_counts[instrument] += 1
    if not browse_counts:
        return
    print("\nInstrument coverage")
    print("===================")
    for instrument, count in browse_counts.most_common():
        print(f"{instrument:18} {count}")


def _print_slowest(records: list[dict], limit: int) -> None:
    timed = [row for row in records if row.get("duration_ms") is not None]
    if not timed:
        return
    slowest = sorted(timed, key=lambda row: float(row["duration_ms"]), reverse=True)[:limit]
    print("\nSlowest examples")
    print("================")
    for row in slowest:
        details = []
        for key in ("event", "instrument", "view_type", "status", "selection", "selected", "day_token", "path"):
            value = row.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        print(
            f"{row.get('ts_utc')} duration={float(row['duration_ms']):8.1f}ms "
            + " ".join(details)
        )


def main() -> None:
    args = _parse_args()
    try:
        records = _load_records(args.log, args.hours, args.event)
    except FileNotFoundError:
        print(f"No performance log found at {args.log}.")
        return
    print(f"Loaded {len(records)} events from {args.log} over the last {args.hours:g} hours.")
    _print_group_summary(records)
    _print_session_summary(records)
    _print_instrument_coverage(records)
    _print_slowest(records, args.limit)


if __name__ == "__main__":
    main()
