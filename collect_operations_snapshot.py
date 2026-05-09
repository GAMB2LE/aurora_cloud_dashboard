#!/usr/bin/env python3
"""Collect Aurora source-host and transfer health metrics into daily JSONL snapshots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import numpy as np
import os
from pathlib import Path
import subprocess
from typing import Any
import pandas as pd
import xarray as xr


RAW_ROOT_DEFAULT = Path("/project/aurora/raw/ops_monitor")
MANIFEST_ROOT_DEFAULT = Path("/data/aurora/internal/mirror_manifests")
GWS_PATH_DEFAULT = Path("/gws/ssde/j25b/gamb2le")
POWER_ZARR_DEFAULT = Path("/data/aurora/products/power/power.zarr")
KNOWN_HOSTS = Path("/home/aurora/.ssh/known_hosts")

SOURCE_HOSTS = {
    "host_celine_source": {
        "host_id": "celine",
        "user_env": "CL61_SOURCE_USER",
        "host_env": "CL61_SOURCE_HOST",
        "path_default": "/",
        "auth": "cl61",
    },
    "host_celine_data": {
        "host_id": "celine",
        "user_env": "CL61_SOURCE_USER",
        "host_env": "CL61_SOURCE_HOST",
        "path_default": "/home/aurora/data",
        "auth": "cl61",
    },
    "host_ass_data": {
        "host_id": "ass",
        "user_env": "RADAR_SOURCE_USER",
        "host_env": "RADAR_SOURCE_HOST",
        "path_default": "/home/aurora/data",
        "auth": "tailscale",
    },
    "host_ass_root": {
        "host_id": "ass",
        "user_env": "RADAR_SOURCE_USER",
        "host_env": "RADAR_SOURCE_HOST",
        "path_default": "/",
        "auth": "tailscale",
    },
    "host_aps_source": {
        "host_id": "aps",
        "user_env": "POWER_SOURCE_USER",
        "host_env": "POWER_SOURCE_HOST",
        "path_default": "/",
        "auth": "tailscale",
    },
    "host_aps_data": {
        "host_id": "aps",
        "user_env": "POWER_SOURCE_USER",
        "host_env": "POWER_SOURCE_HOST",
        "path_default": "/data",
        "auth": "tailscale",
    },
    "host_aps_root": {
        "host_id": "aps",
        "user_env": "POWER_SOURCE_USER",
        "host_env": "POWER_SOURCE_HOST",
        "path_default": "/",
        "auth": "tailscale",
    },
}

STREAM_PREFIXES = {
    "cl61": "cl61",
    "rpgfmcw94": "radar",
    "vaisalamet": "vaisalamet",
    "asfs_logger": "asfs_logger",
    "asfs_fast_sonic": "asfs_fast_sonic",
    "power": "power",
    "wxcam": "wxcam",
}

BACKFILL_STREAMS = {"wxcam"}

SOURCE_SYNC_UNITS = (
    "aurora-cl61-source-sync.timer",
    "aurora-cl61-source-sync.service",
    "aurora-radar-source-sync.timer",
    "aurora-radar-source-sync.service",
    "aurora-vaisalamet-source-sync.timer",
    "aurora-vaisalamet-source-sync.service",
    "aurora-asfs-logger-source-sync.timer",
    "aurora-asfs-logger-source-sync.service",
    "aurora-asfs-fast-sonic-source-sync.timer",
    "aurora-asfs-fast-sonic-source-sync.service",
    "aurora-power-source-sync.timer",
    "aurora-power-source-sync.service",
    "aurora-wxcam-source-sync.timer",
    "aurora-wxcam-source-sync.service",
)

PROCESSING_UNITS = (
    "aurora-ceilometer-append.timer",
    "aurora-ceilometer-append.service",
    "aurora-ceilometer-quicklooks.timer",
    "aurora-ceilometer-quicklooks.service",
    "aurora-radar-append.timer",
    "aurora-radar-append.service",
    "aurora-radar-quicklooks.timer",
    "aurora-radar-quicklooks.service",
    "aurora-vaisalamet-append.timer",
    "aurora-vaisalamet-append.service",
    "aurora-vaisalamet-quicklooks.timer",
    "aurora-vaisalamet-quicklooks.service",
    "aurora-asfs-logger-append.timer",
    "aurora-asfs-logger-append.service",
    "aurora-asfs-logger-quicklooks.timer",
    "aurora-asfs-logger-quicklooks.service",
    "aurora-asfs-fast-sonic-append.timer",
    "aurora-asfs-fast-sonic-append.service",
    "aurora-asfs-fast-sonic-quicklooks.timer",
    "aurora-asfs-fast-sonic-quicklooks.service",
    "aurora-power-append.timer",
    "aurora-power-append.service",
    "aurora-power-quicklooks.timer",
    "aurora-power-quicklooks.service",
    "aurora-wxcam-catalog.timer",
    "aurora-wxcam-catalog.service",
    "aurora-wxcam-append.timer",
    "aurora-wxcam-append.service",
    "aurora-wxcam-daily-videos.timer",
    "aurora-wxcam-daily-videos.service",
    "aurora-ops-monitor-append.timer",
    "aurora-ops-monitor-append.service",
    "aurora-ops-monitor-quicklooks.timer",
    "aurora-ops-monitor-quicklooks.service",
)

TRANSFER_UNITS = (
    "aurora-gws-rsync-raw.timer",
    "aurora-gws-rsync-raw.service",
    "aurora-gws-rsync-products.timer",
    "aurora-gws-rsync-products.service",
    "aurora-gws-rsync-manifests.timer",
    "aurora-gws-rsync-manifests.service",
    "aurora-mirror-verify.timer",
    "aurora-mirror-verify.service",
)
SOURCE_RECENT_THRESHOLD_MINUTES = 90.0


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _tailscale_ssh_base() -> list[str]:
    return [
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
        f"UserKnownHostsFile={KNOWN_HOSTS}",
    ]


def _cl61_ssh_base() -> list[str]:
    key_path = os.environ.get("CL61_SOURCE_KEY_PATH", "/home/aurora/.ssh/id_ed25519_celine")
    return [
        "ssh",
        "-i",
        key_path,
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
    ]


def _gws_ssh_base() -> list[str]:
    key_path = os.environ.get("GWS_SSH_PRIVATE_KEY", "/home/aurora/.ssh/id_rsa_jasmin_20200514")
    return [
        "ssh",
        "-i",
        key_path,
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
    ]


def _remote_df(base_cmd: list[str], target: str, path: str) -> dict[str, Any]:
    quoted = json.dumps(path)
    remote = (
        f"cd {quoted} && pwd -P && "
        "df -PB1 . | tail -1; "
        "df -Pi . | tail -1"
    )
    proc = _run(base_cmd + [target, remote])
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Unexpected df output for {target}:{path}")
    metrics = _parse_df_lines(lines[1], lines[2])
    metrics["resolved_path"] = lines[0].strip()
    return metrics


def _local_df(path: str | Path) -> dict[str, Any]:
    quoted = json.dumps(str(path))
    proc = _run(
        [
            "bash",
            "-lc",
            f"cd {quoted} && pwd -P && df -PB1 . | tail -1 && df -Pi . | tail -1",
        ]
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Unexpected local df output for {path}")
    metrics = _parse_df_lines(lines[1], lines[2])
    metrics["resolved_path"] = lines[0].strip()
    return metrics


def _parse_df_lines(space_line: str, inode_line: str) -> dict[str, float]:
    fields = space_line.split()
    inode_fields = inode_line.split()
    if len(fields) < 6 or len(inode_fields) < 6:
        raise ValueError("Could not parse df output")
    total = float(fields[1])
    used = float(fields[2])
    available = float(fields[3])
    inode_total = float(inode_fields[1]) if inode_fields[1].isdigit() else math.nan
    inode_used = float(inode_fields[2]) if inode_fields[2].isdigit() else math.nan
    return {
        "total_gb": total / (1024.0 ** 3),
        "used_gb": used / (1024.0 ** 3),
        "free_gb": available / (1024.0 ** 3),
        "used_pct": float(fields[4].rstrip("%")),
        "inode_used_pct": float(inode_fields[4].rstrip("%")),
        "inode_total": inode_total,
        "inode_used": inode_used,
    }


def _read_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_stats(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {"count": 0, "size_bytes": 0, "latest_mtime": None}
    count = 0
    size_bytes = 0
    latest_mtime: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            count += 1
            try:
                size = int(row.get("size", "0") or 0)
            except Exception:
                size = 0
            try:
                mtime = int(row.get("mtime", "0") or 0)
            except Exception:
                mtime = 0
            size_bytes += size
            if mtime and (latest_mtime is None or mtime > latest_mtime):
                latest_mtime = mtime
    return {"count": count, "size_bytes": size_bytes, "latest_mtime": latest_mtime}


def _coverage_pct(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return 100.0 * float(numerator) / float(denominator)


def _lag_minutes(source_mtime: int | None, mirror_mtime: int | None) -> float | None:
    if source_mtime is None or mirror_mtime is None:
        return None
    return max(float(source_mtime - mirror_mtime), 0.0) / 60.0


def _age_minutes(now_epoch: float, sample_epoch: int | None) -> float | None:
    if sample_epoch is None:
        return None
    return max(now_epoch - float(sample_epoch), 0.0) / 60.0


def _recent_state(age_minutes: float | None, threshold_minutes: float = SOURCE_RECENT_THRESHOLD_MINUTES) -> int:
    if age_minutes is None:
        return 0
    return 1 if age_minutes <= threshold_minutes else 0


def _latest_finite_zarr_value(
    zarr_path: Path,
    var_name: str,
    *,
    time_name: str = "time",
) -> tuple[float | None, datetime | None]:
    if not zarr_path.exists():
        return None, None
    ds = xr.open_zarr(zarr_path)
    try:
        if var_name not in ds or time_name not in ds:
            return None, None
        data = ds[var_name]
        if time_name not in data.dims:
            return None, None
        total = int(data.sizes.get(time_name, 0))
        if total <= 0:
            return None, None
        time_coord = ds[time_name]
        for window in (2048, 16384, None):
            selector = slice(None) if window is None or total <= window else slice(-window, None)
            values = np.asarray(data.isel({time_name: selector}).values)
            times = np.asarray(time_coord.isel({time_name: selector}).values)
            finite_idx = np.flatnonzero(np.isfinite(values))
            if finite_idx.size == 0:
                continue
            idx = int(finite_idx[-1])
            timestamp = pd.Timestamp(times[idx])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return float(values[idx]), timestamp.to_pydatetime(warn=False)
        return None, None
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()


def _unit_slug(unit: str) -> str:
    return unit.replace("aurora-", "").replace(".", "_").replace("-", "_")


def _systemd_show(unit: str) -> dict[str, str]:
    props = ("ActiveState", "UnitFileState", "Result", "ExecMainExitTimestamp", "LastTriggerUSec", "NextElapseUSecRealtime")
    proc = _run(
        ["systemctl", "show", unit, *sum((["-p", prop] for prop in props), [])],
        check=False,
    )
    info: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key] = value
    info["_exists"] = "1" if proc.returncode == 0 and bool(info) else "0"
    return info


def _service_healthy(info: dict[str, str]) -> int:
    if info.get("_exists") != "1":
        return 0
    active = info.get("ActiveState", "")
    result = info.get("Result", "")
    if active == "failed":
        return 0
    if result in {"success", "", "done"}:
        return 1
    return 0


def _timer_active(info: dict[str, str]) -> int:
    if info.get("_exists") != "1":
        return 0
    return 1 if info.get("ActiveState") == "active" else 0


def _unit_enabled(info: dict[str, str]) -> int:
    if info.get("_exists") != "1":
        return 0
    return 1 if info.get("UnitFileState") == "enabled" else 0


def _collect_unit_metrics(units: tuple[str, ...], record: dict[str, Any]) -> tuple[int, int]:
    failures = 0
    enabled_count = 0
    for unit in units:
        info = _systemd_show(unit)
        slug = _unit_slug(unit)
        is_timer = unit.endswith(".timer")
        if is_timer:
            active = _timer_active(info)
            enabled = _unit_enabled(info)
            record[f"{slug}_active_state"] = active
            record[f"{slug}_enabled_state"] = enabled
            failures += 0 if active else 1
            enabled_count += enabled
        else:
            healthy = _service_healthy(info)
            record[f"{slug}_healthy_state"] = healthy
            failures += 0 if healthy else 1
    return failures, enabled_count


def _probe_gws(gws_path: Path) -> tuple[str | None, dict[str, float] | None]:
    hosts = [host.strip() for host in os.environ.get("GWS_TRANSFER_HOSTS", "xfer-vm-03.jasmin.ac.uk,xfer-vm-01.jasmin.ac.uk,xfer-vm-02.jasmin.ac.uk").split(",") if host.strip()]
    username = os.environ.get("GWS_USERNAME", "rrniii")
    base_cmd = _gws_ssh_base()
    for host in hosts:
        target = f"{username}@{host}"
        try:
            metrics = _remote_df(base_cmd, target, str(gws_path))
            return host, metrics
        except Exception:
            continue
    return None, None


def build_snapshot(manifest_root: Path, gws_path: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    now_epoch = now.timestamp()
    record: dict[str, Any] = {
        "time_utc": now.isoformat(),
        "snapshot_time_utc": now.isoformat(),
        "snapshot_epoch": now_epoch,
    }

    summary_path = manifest_root / "latest" / "summary.json"
    summary = _read_summary(summary_path)
    summary_generated = summary.get("generated_at")
    if summary_generated:
        try:
            summary_time = datetime.fromisoformat(summary_generated)
            record["mirror_summary_age_min"] = max((now - summary_time).total_seconds(), 0.0) / 60.0
        except Exception:
            pass

    host_reachability: dict[str, bool] = {}
    for prefix, cfg in SOURCE_HOSTS.items():
        host_id = cfg.get("host_id", prefix)
        user = os.environ.get(cfg["user_env"], "aurora")
        host = os.environ.get(cfg["host_env"], "")
        path = os.environ.get(cfg.get("path_env", ""), cfg["path_default"])
        if not host:
            record[f"{prefix}_probe_ok_state"] = 0
            host_reachability.setdefault(host_id, False)
            continue
        base_cmd = _cl61_ssh_base() if cfg["auth"] == "cl61" else _tailscale_ssh_base()
        target = f"{user}@{host}"
        try:
            metrics = _remote_df(base_cmd, target, path)
            record[f"{prefix}_probe_ok_state"] = 1
            host_reachability[host_id] = True
            for key, value in metrics.items():
                record[f"{prefix}_{key}"] = value
        except Exception:
            record[f"{prefix}_probe_ok_state"] = 0
            host_reachability.setdefault(host_id, False)
    record["source_host_probe_fail_count"] = sum(1 for ok in host_reachability.values() if not ok)

    for prefix, path in (
        ("aurora_project", "/project/aurora"),
        ("aurora_data", "/data/aurora"),
        ("aurora_root", "/"),
    ):
        try:
            metrics = _local_df(path)
            for key, value in metrics.items():
                record[f"{prefix}_{key}"] = value
        except Exception:
            continue

    gws_host, gws_metrics = _probe_gws(gws_path)
    record["gws_available_state"] = 1 if gws_metrics else 0
    if gws_metrics:
        for key, value in gws_metrics.items():
            record[f"gws_storage_{key}"] = value

    battery_voltage, battery_time = _latest_finite_zarr_value(
        _path_from_env("POWER_ZARR_PATH", POWER_ZARR_DEFAULT),
        "DCInverterVolts",
    )
    record["aps_battery_voltage_v"] = battery_voltage
    if battery_time is not None:
        record["aps_battery_voltage_time_utc"] = battery_time.isoformat()
        record["aps_battery_voltage_age_min"] = max((now - battery_time).total_seconds(), 0.0) / 60.0
    internal_temp, internal_temp_time = _latest_finite_zarr_value(
        _path_from_env("POWER_ZARR_PATH", POWER_ZARR_DEFAULT),
        "InternalTemperature",
    )
    record["aps_internal_temp_c"] = internal_temp
    if internal_temp_time is not None:
        record["aps_internal_temp_time_utc"] = internal_temp_time.isoformat()
        record["aps_internal_temp_age_min"] = max((now - internal_temp_time).total_seconds(), 0.0) / 60.0

    streams = summary.get("streams", {})
    local_issue_count = 0
    gws_issue_count = 0
    prune_ready_count = 0
    product_gate_ok_count = 0
    backfill_pending_count = 0
    source_stale_count = 0
    source_recent_count = 0
    for stream_name, prefix in STREAM_PREFIXES.items():
        stream_dir = manifest_root / "latest" / stream_name
        source_stats = _manifest_stats(stream_dir / "source.tsv")
        local_stats = _manifest_stats(stream_dir / "local.tsv")
        gws_stats = _manifest_stats(stream_dir / "gws.tsv")
        stream_summary = streams.get(stream_name, {})

        source_count = int(source_stats["count"])
        local_count = int(local_stats["count"])
        gws_count = int(gws_stats["count"]) if (stream_dir / "gws.tsv").exists() else None

        record[f"{prefix}_source_count"] = source_count
        record[f"{prefix}_local_count"] = local_count
        record[f"{prefix}_gws_count"] = gws_count

        record[f"{prefix}_source_size_gb"] = float(source_stats["size_bytes"]) / (1024.0 ** 3)
        record[f"{prefix}_local_size_gb"] = float(local_stats["size_bytes"]) / (1024.0 ** 3)
        if gws_count is not None:
            record[f"{prefix}_gws_size_gb"] = float(gws_stats["size_bytes"]) / (1024.0 ** 3)

        record[f"{prefix}_local_coverage_pct"] = _coverage_pct(local_count, source_count)
        record[f"{prefix}_gws_coverage_pct"] = _coverage_pct(gws_count, source_count) if gws_count is not None else None
        record[f"{prefix}_local_lag_min"] = _lag_minutes(source_stats["latest_mtime"], local_stats["latest_mtime"])
        if gws_count is not None:
            record[f"{prefix}_gws_lag_min"] = _lag_minutes(source_stats["latest_mtime"], gws_stats["latest_mtime"])
        source_age_min = _age_minutes(now_epoch, source_stats["latest_mtime"])
        source_recent_state = _recent_state(source_age_min)
        record[f"{prefix}_source_age_min"] = source_age_min
        record[f"{prefix}_source_recent_state"] = source_recent_state
        if source_recent_state:
            source_recent_count += 1
        else:
            source_stale_count += 1

        local_missing = stream_summary.get("local_missing_count")
        local_mismatch = stream_summary.get("local_mismatch_count")
        gws_missing = stream_summary.get("gws_missing_count")
        gws_mismatch = stream_summary.get("gws_mismatch_count")
        prune_ready = stream_summary.get("prune_ready")
        product_gate_ok = stream_summary.get("product_gate_ok")

        record[f"{prefix}_local_missing_count"] = local_missing
        record[f"{prefix}_local_mismatch_count"] = local_mismatch
        record[f"{prefix}_gws_missing_count"] = gws_missing
        record[f"{prefix}_gws_mismatch_count"] = gws_mismatch
        record[f"{prefix}_prune_ready_state"] = 1 if prune_ready else 0
        record[f"{prefix}_product_gate_ok_state"] = 1 if product_gate_ok else 0

        backfill_pending = False
        if stream_name in BACKFILL_STREAMS:
            local_coverage = record.get(f"{prefix}_local_coverage_pct")
            local_lag = record.get(f"{prefix}_local_lag_min")
            if (
                local_coverage is not None
                and local_coverage < 99.9
                and local_lag is not None
                and local_lag <= 1.0
            ):
                backfill_pending = True
        record[f"{prefix}_backfill_pending_state"] = 1 if backfill_pending else 0
        if backfill_pending:
            backfill_pending_count += 1
            continue

        if (local_missing or 0) > 0 or (local_mismatch or 0) > 0:
            local_issue_count += 1
        if gws_metrics and (((gws_missing or 0) > 0) or ((gws_mismatch or 0) > 0)):
            gws_issue_count += 1
        if prune_ready:
            prune_ready_count += 1
        if product_gate_ok:
            product_gate_ok_count += 1

    record["streams_local_issue_count"] = local_issue_count
    record["streams_gws_issue_count"] = gws_issue_count
    record["streams_prune_ready_count"] = prune_ready_count
    record["streams_product_gate_ok_count"] = product_gate_ok_count
    record["streams_backfill_pending_count"] = backfill_pending_count
    record["streams_source_stale_count"] = source_stale_count
    record["streams_source_recent_count"] = source_recent_count
    record["streams_target_count"] = max(0, len(STREAM_PREFIXES) - backfill_pending_count)

    failed_source_sync, source_sync_timer_enabled = _collect_unit_metrics(SOURCE_SYNC_UNITS, record)
    failed_processing, processing_timer_enabled = _collect_unit_metrics(PROCESSING_UNITS, record)
    failed_transfer, transfer_timer_enabled = _collect_unit_metrics(TRANSFER_UNITS, record)
    record["failed_source_sync_unit_count"] = failed_source_sync
    record["failed_processing_unit_count"] = failed_processing
    record["failed_transfer_unit_count"] = failed_transfer
    record["source_sync_enabled_count"] = source_sync_timer_enabled
    record["processing_timer_enabled_count"] = processing_timer_enabled
    record["transfer_timer_enabled_count"] = transfer_timer_enabled
    record["gws_probe_ok_state"] = 1 if gws_host else 0

    return {key: _float_or_none(value) if isinstance(value, (int, float)) and key != "time_utc" else value for key, value in record.items()}


def write_snapshot(output_root: Path, snapshot: dict[str, Any]) -> Path:
    stamp = datetime.fromisoformat(snapshot["time_utc"].replace("Z", "+00:00"))
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"ops_monitor_{stamp:%Y%m%d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True))
        handle.write("\n")
    latest_path = output_root / "latest.json"
    latest_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Aurora operations monitoring snapshots.")
    parser.add_argument("--output-root", type=Path, default=_path_from_env("OPS_MONITOR_RAW_ROOT", RAW_ROOT_DEFAULT))
    parser.add_argument("--manifest-root", type=Path, default=_path_from_env("GWS_MANIFEST_ROOT", MANIFEST_ROOT_DEFAULT))
    parser.add_argument("--gws-path", type=Path, default=_path_from_env("GWS_PATH", GWS_PATH_DEFAULT))
    args = parser.parse_args()

    snapshot = build_snapshot(args.manifest_root, args.gws_path)
    path = write_snapshot(args.output_root, snapshot)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
