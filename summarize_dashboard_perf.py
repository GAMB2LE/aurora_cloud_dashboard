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
SUMMARY_COVERAGE_EVENTS = {
    "interactive_view_update",
    "stacked_timeseries_render",
    "science_quicklook_render",
    "housekeeping_quicklook_render",
    "wxcam_interactive_render",
    "wxcam_calendar_options",
    "wxcam_calendar_day_view",
    "operations_dashboard_render",
    "auroracam_render",
    "uas_dashboard_render",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Aurora dashboard performance logs.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--hours", type=float, default=24.0, help="Only include events from the last N hours.")
    parser.add_argument("--event", default=None, help="Optional exact event name filter.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum slow examples to print.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write the latest summary as JSON.")
    parser.add_argument("--output-markdown", type=Path, default=None, help="Write the latest summary as Markdown.")
    parser.add_argument("--history-jsonl", type=Path, default=None, help="Append each generated summary to a JSONL history file.")
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


def _stats(values: list[float], unit: str = "ms") -> dict[str, float | int | str]:
    avg = sum(values) / len(values)
    return {
        "count": len(values),
        "avg": avg,
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values),
        "unit": unit,
    }


def _timing_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        duration = row.get("duration_ms")
        if duration is None:
            continue
        key = (str(row.get("event", "")), str(row.get("instrument", "")))
        grouped[key].append(float(duration))
    rows = []
    for (event, instrument), durations in sorted(grouped.items()):
        item = {"event": event, "instrument": instrument}
        item.update(_stats(durations, unit="ms"))
        rows.append(item)
    return rows


def _phase_summary_data(records: list[dict]) -> dict[str, dict[str, dict[str, float | int | str]]]:
    phases = ("source_open_ms", "combine_ms", "figure_build_ms", "plot_points_total", "plot_json_bytes")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("event") != "stacked_timeseries_render":
            continue
        instrument = str(row.get("instrument", ""))
        for phase in phases:
            value = row.get(phase)
            if value is None:
                continue
            try:
                grouped[instrument][phase].append(float(value))
            except (TypeError, ValueError):
                continue
    summary: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for instrument, phase_values in sorted(grouped.items()):
        summary[instrument] = {}
        for phase, values in phase_values.items():
            unit = "bytes" if phase == "plot_json_bytes" else "points" if phase == "plot_points_total" else "ms"
            summary[instrument][phase] = _stats(values, unit=unit)
    return summary


def _client_family(user_agent: object) -> str:
    value = str(user_agent or "").lower()
    if not value:
        return "unknown"
    if "bot" in value or "spider" in value or "crawler" in value:
        return "bot"
    if "ipad" in value or "tablet" in value:
        return "tablet"
    if "mobi" in value or "iphone" in value or "android" in value:
        return "mobile"
    return "desktop"


def _session_summary_data(records: list[dict]) -> dict:
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
    client_families: Counter[str] = Counter(_client_family(row.get("user_agent")) for row in loaded)
    request_paths: Counter[str] = Counter(str(row.get("request_path", "")) for row in loaded if row.get("request_path"))
    return {
        "unique_session_ids": len(session_ids),
        "sessions_loaded": len(loaded),
        "sessions_closed": len(destroyed),
        "heartbeats": len(heartbeats),
        "max_live_sessions": max(live_counts) if live_counts else 0,
        "max_server_sessions": max(server_counts) if server_counts else 0,
        "top_control_changes": [
            {"instrument": instrument, "control": control, "count": count}
            for (instrument, control), count in control_changes.most_common(10)
        ],
        "client_families": [{"client": family, "count": count} for family, count in client_families.most_common()],
        "request_paths": [{"path": path, "count": count} for path, count in request_paths.most_common(10)],
    }


def _instrument_coverage_data(records: list[dict]) -> list[dict]:
    browse_counts: Counter[str] = Counter()
    for row in records:
        event = str(row.get("event", ""))
        instrument = str(row.get("instrument", ""))
        if event in SUMMARY_COVERAGE_EVENTS:
            browse_counts[instrument] += 1
    return [{"instrument": instrument, "count": count} for instrument, count in browse_counts.most_common()]


def _slowest_rows(records: list[dict], limit: int) -> list[dict]:
    timed = [row for row in records if row.get("duration_ms") is not None]
    keys = (
        "ts_utc",
        "duration_ms",
        "event",
        "instrument",
        "view_type",
        "status",
        "selection",
        "selected",
        "day_token",
        "power_display_summary",
        "power_display_energy",
        "source_open_ms",
        "combine_ms",
        "figure_build_ms",
        "trace_count",
        "plot_points_total",
        "plot_json_bytes",
        "path",
    )
    rows = []
    for row in sorted(timed, key=lambda item: float(item["duration_ms"]), reverse=True)[:limit]:
        rows.append({key: row.get(key) for key in keys if row.get(key) not in (None, "")})
    return rows


def _build_summary(path: Path, hours: float, event_filter: str | None, records: list[dict], limit: int) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(path),
        "window_hours": hours,
        "event_filter": event_filter,
        "event_count": len(records),
        "timing": _timing_summary(records),
        "stacked_timeseries_phases": _phase_summary_data(records),
        "sessions": _session_summary_data(records),
        "instrument_coverage": _instrument_coverage_data(records),
        "slowest": _slowest_rows(records, limit),
        "status": "ok",
    }


def _missing_summary(path: Path, hours: float, event_filter: str | None) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(path),
        "window_hours": hours,
        "event_filter": event_filter,
        "event_count": 0,
        "timing": [],
        "stacked_timeseries_phases": {},
        "sessions": {},
        "instrument_coverage": [],
        "slowest": [],
        "status": "missing_log",
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _summary_markdown(summary: dict) -> str:
    lines = [
        "# Dashboard Performance Summary",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Window: `{summary['window_hours']}` h",
        f"- Events: `{summary['event_count']}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Sessions",
    ]
    sessions = summary.get("sessions") or {}
    if sessions:
        lines.extend(
            [
                f"- Unique sessions: `{sessions.get('unique_session_ids', 0)}`",
                f"- Sessions loaded: `{sessions.get('sessions_loaded', 0)}`",
                f"- Sessions closed: `{sessions.get('sessions_closed', 0)}`",
                f"- Max live sessions: `{sessions.get('max_live_sessions', 0)}`",
                f"- Max server sessions: `{sessions.get('max_server_sessions', 0)}`",
            ]
        )
        if sessions.get("client_families"):
            clients = ", ".join(f"{row['client']}={row['count']}" for row in sessions["client_families"])
            lines.append(f"- Clients: `{clients}`")
        if sessions.get("request_paths"):
            paths = ", ".join(f"{row['path']}={row['count']}" for row in sessions["request_paths"][:5])
            lines.append(f"- Paths: `{paths}`")
    else:
        lines.append("- No session events found.")

    lines.extend(["", "## Slowest Timed Events"])
    for row in summary.get("slowest", [])[:10]:
        lines.append(
            f"- `{row.get('duration_ms', 0):.1f} ms` `{row.get('event', '')}` "
            f"`{row.get('instrument', '')}` `{row.get('ts_utc', '')}`"
        )
    if not summary.get("slowest"):
        lines.append("- No timed events found.")

    lines.extend(["", "## Timing By Event"])
    for row in summary.get("timing", [])[:30]:
        lines.append(
            f"- `{row['event']}` `{row['instrument']}` count `{row['count']}` "
            f"p50 `{row['p50']:.1f} {row['unit']}` p95 `{row['p95']:.1f} {row['unit']}`"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(summary: dict, output_json: Path | None, output_markdown: Path | None, history_jsonl: Path | None) -> None:
    if output_json:
        _atomic_write_text(output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if output_markdown:
        _atomic_write_text(output_markdown, _summary_markdown(summary))
    if history_jsonl:
        _append_jsonl(history_jsonl, summary)


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


def _print_phase_summary(records: list[dict]) -> None:
    phases = ("source_open_ms", "combine_ms", "figure_build_ms", "plot_points_total", "plot_json_bytes")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("event") != "stacked_timeseries_render":
            continue
        instrument = str(row.get("instrument", ""))
        for phase in phases:
            value = row.get(phase)
            if value is None:
                continue
            try:
                grouped[instrument][phase].append(float(value))
            except (TypeError, ValueError):
                continue
    if not grouped:
        return
    print("\nStacked-timeseries phase summary")
    print("================================")
    for instrument, phase_values in sorted(grouped.items()):
        print(instrument)
        for phase in phases:
            values = phase_values.get(phase)
            if not values:
                continue
            avg = sum(values) / len(values)
            p95 = _quantile(values, 0.95)
            max_v = max(values)
            suffix = " bytes" if phase == "plot_json_bytes" else " points" if phase == "plot_points_total" else "ms"
            print(f"  {phase:22} count={len(values):4d} avg={avg:9.1f}{suffix} p95={p95:9.1f}{suffix} max={max_v:9.1f}{suffix}")


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
        if event in SUMMARY_COVERAGE_EVENTS:
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
        for key in (
            "event",
            "instrument",
            "view_type",
            "status",
            "selection",
            "selected",
            "day_token",
            "power_display_summary",
            "power_display_energy",
            "source_open_ms",
            "combine_ms",
            "figure_build_ms",
            "trace_count",
            "plot_points_total",
            "plot_json_bytes",
            "path",
        ):
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
        summary = _missing_summary(args.log, args.hours, args.event)
        _write_outputs(summary, args.output_json, args.output_markdown, args.history_jsonl)
        return
    summary = _build_summary(args.log, args.hours, args.event, records, args.limit)
    print(f"Loaded {len(records)} events from {args.log} over the last {args.hours:g} hours.")
    _print_group_summary(records)
    _print_phase_summary(records)
    _print_session_summary(records)
    _print_instrument_coverage(records)
    _print_slowest(records, args.limit)
    _write_outputs(summary, args.output_json, args.output_markdown, args.history_jsonl)


if __name__ == "__main__":
    main()
