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
import time
from typing import Any
import urllib.error
import urllib.request
import pandas as pd
import xarray as xr


RAW_ROOT_DEFAULT = Path("/project/aurora/raw/ops_monitor")
HEALTH_OUTPUT_ROOT_DEFAULT = Path("/data/aurora/products/ops_monitor/health")
MANIFEST_ROOT_DEFAULT = Path("/data/aurora/internal/mirror_manifests")
GWS_PATH_DEFAULT = Path("/gws/ssde/j25b/gamb2le")
POWER_ZARR_DEFAULT = Path("/data/aurora/products/power/power.zarr")
DASHBOARD_PERF_LOG_DEFAULT = Path("/data/aurora/products/dashboard/dashboard_perf.jsonl")
DASHBOARD_HTTP_URL_DEFAULT = "http://127.0.0.1:5006/app"
INFRA_REPO_DEFAULT = Path("/tmp/aurora-cloud-infra-codex")
ENV_FILE_DEFAULT = Path("/etc/aurora-dashboard.env")
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
    "hatprog5": "hatpro",
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
    "aurora-hatpro-source-sync.timer",
    "aurora-hatpro-source-sync.service",
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
    "aurora-hatpro-append.timer",
    "aurora-hatpro-append.service",
    "aurora-hatpro-quicklooks.timer",
    "aurora-hatpro-quicklooks.service",
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
    "aurora-ops-monitor-alerts.timer",
    "aurora-ops-monitor-alerts.service",
    "aurora-ops-monitor-quicklooks.timer",
    "aurora-ops-monitor-quicklooks.service",
)

TRANSFER_UNITS = (
    "aurora-gws-rsync-raw.timer",
    "aurora-gws-rsync-raw.service",
    "aurora-gws-rsync-products.timer",
    "aurora-gws-rsync-products.service",
    "aurora-gws-rsync-products-wxcam.timer",
    "aurora-gws-rsync-products-wxcam.service",
    "aurora-gws-rsync-manifests.timer",
    "aurora-gws-rsync-manifests.service",
    "aurora-mirror-verify.timer",
    "aurora-mirror-verify.service",
)
SOURCE_RECENT_THRESHOLD_MINUTES = 90.0
HEALTH_LEVELS = {"green": 0, "amber": 1, "red": 2, "gray": -1}


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


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


def _run(cmd: list[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


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


def _file_freshness(path: Path, now_epoch: float, *, recent_threshold_minutes: float) -> dict[str, float | int | str | None]:
    if not path.exists():
        return {
            "path": str(path),
            "exists_state": 0,
            "age_min": None,
            "size_mb": None,
            "recent_state": 0,
        }
    stat_result = path.stat()
    age_min = _age_minutes(now_epoch, int(stat_result.st_mtime))
    return {
        "path": str(path),
        "exists_state": 1,
        "age_min": age_min,
        "size_mb": float(stat_result.st_size) / (1024.0 ** 2),
        "recent_state": _recent_state(age_min, recent_threshold_minutes),
    }


def _probe_http(url: str) -> dict[str, float | int | str | None]:
    start = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "aurora-ops-monitor/1.0"}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = int(response.status)
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        return {
            "ok_state": 1 if 200 <= status_code < 400 or status_code == 405 else 0,
            "status_code": status_code,
            "response_ms": (time.monotonic() - start) * 1000.0,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok_state": 0,
            "status_code": None,
            "response_ms": (time.monotonic() - start) * 1000.0,
            "error": str(exc),
        }
    return {
        "ok_state": 1 if 200 <= status_code < 400 or status_code == 405 else 0,
        "status_code": status_code,
        "response_ms": (time.monotonic() - start) * 1000.0,
        "error": "",
    }


def _git_value(repo: Path, args: list[str]) -> str | None:
    try:
        proc = _run(["git", "-C", str(repo), *args], check=False, timeout=5.0)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _collect_git_metrics(prefix: str, repo: Path, record: dict[str, Any]) -> None:
    record[f"{prefix}_repo_path"] = str(repo)
    if not repo.exists():
        record[f"{prefix}_repo_exists_state"] = 0
        return
    inside = _git_value(repo, ["rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        record[f"{prefix}_repo_exists_state"] = 0
        return

    record[f"{prefix}_repo_exists_state"] = 1
    record[f"{prefix}_git_branch"] = _git_value(repo, ["rev-parse", "--abbrev-ref", "HEAD"]) or ""
    record[f"{prefix}_git_commit"] = _git_value(repo, ["rev-parse", "--short", "HEAD"]) or ""
    upstream = _git_value(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    record[f"{prefix}_git_upstream"] = upstream or ""

    status = _git_value(repo, ["status", "--porcelain"])
    dirty_count = len([line for line in (status or "").splitlines() if line.strip()])
    record[f"{prefix}_git_dirty_count"] = dirty_count
    record[f"{prefix}_git_clean_state"] = 1 if dirty_count == 0 else 0

    commit_time = _git_value(repo, ["log", "-1", "--format=%ct"])
    if commit_time:
        try:
            record[f"{prefix}_git_commit_epoch"] = int(commit_time)
        except ValueError:
            pass

    if upstream:
        divergence = _git_value(repo, ["rev-list", "--left-right", "--count", "@{u}...HEAD"])
        if divergence:
            try:
                behind, ahead = divergence.split()
                record[f"{prefix}_git_behind_count"] = int(behind)
                record[f"{prefix}_git_ahead_count"] = int(ahead)
            except ValueError:
                pass


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

    perf_log_stats = _file_freshness(
        _path_from_env("AURORA_DASHBOARD_PERF_LOG", DASHBOARD_PERF_LOG_DEFAULT),
        now_epoch,
        recent_threshold_minutes=30.0,
    )
    for key, value in perf_log_stats.items():
        record[f"dashboard_perf_log_{key}"] = value

    dashboard_probe = _probe_http(os.environ.get("AURORA_DASHBOARD_HEALTH_URL", DASHBOARD_HTTP_URL_DEFAULT))
    for key, value in dashboard_probe.items():
        record[f"dashboard_http_{key}"] = value

    _collect_git_metrics(
        "dashboard_code",
        _path_from_env("AURORA_DASHBOARD_REPO", Path(__file__).resolve().parent),
        record,
    )
    _collect_git_metrics(
        "infra_code",
        _path_from_env("AURORA_INFRA_REPO", INFRA_REPO_DEFAULT),
        record,
    )

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
    battery_soc, battery_soc_time = _latest_finite_zarr_value(
        _path_from_env("POWER_ZARR_PATH", POWER_ZARR_DEFAULT),
        "BatterySOC",
    )
    record["aps_battery_soc_pct"] = battery_soc
    if battery_soc_time is not None:
        record["aps_battery_soc_time_utc"] = battery_soc_time.isoformat()
        record["aps_battery_soc_age_min"] = max((now - battery_soc_time).total_seconds(), 0.0) / 60.0
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


def _value(snapshot: dict[str, Any], key: str) -> float | None:
    try:
        number = float(snapshot[key])
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _state(snapshot: dict[str, Any], key: str) -> bool | None:
    number = _value(snapshot, key)
    if number is None:
        return None
    return bool(round(number))


def _worst_level(levels: list[str]) -> str:
    meaningful = [level for level in levels if level != "gray"]
    if not meaningful:
        return "gray"
    return max(meaningful, key=lambda level: HEALTH_LEVELS.get(level, -1))


def _level_from_bool(value: bool | None) -> str:
    if value is None:
        return "gray"
    return "green" if value else "red"


def _level_from_count(value: float | None, *, amber_at: float = 1.0) -> str:
    if value is None:
        return "gray"
    if value <= 0:
        return "green"
    if value <= amber_at:
        return "amber"
    return "red"


def _level_from_used_pct(value: float | None) -> str:
    if value is None:
        return "gray"
    if value < 75.0:
        return "green"
    if value < 90.0:
        return "amber"
    return "red"


def _level_from_battery_voltage(value: float | None) -> str:
    if value is None:
        return "gray"
    if value > 52.0:
        return "green"
    if value >= 50.0:
        return "amber"
    return "red"


def _level_from_battery_soc(value: float | None) -> str:
    if value is None:
        return "gray"
    if value >= 50.0:
        return "green"
    if value >= 25.0:
        return "amber"
    return "red"


def _level_from_internal_temp(value: float | None) -> str:
    if value is None:
        return "gray"
    if value < 35.0:
        return "green"
    if value < 40.0:
        return "amber"
    return "red"


def _fmt(value: float | None, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}f}{suffix}"


def _health_check(
    checks: list[dict[str, Any]],
    level: str,
    component: str,
    message: str,
    *,
    details: str = "",
    metrics: dict[str, Any] | None = None,
    affects_overall: bool = True,
) -> None:
    checks.append(
        {
            "level": level,
            "component": component,
            "message": message,
            "details": details,
            "metrics": metrics or {},
            "affects_overall": affects_overall,
        }
    )


def build_health_assessment(snapshot: dict[str, Any], raw_snapshot_path: Path | None = None) -> dict[str, Any]:
    """Convert the raw metric snapshot into an observe-only health assessment."""

    checks: list[dict[str, Any]] = []

    http_level = _level_from_bool(_state(snapshot, "dashboard_http_ok_state"))
    _health_check(
        checks,
        http_level,
        "dashboard",
        "Dashboard HTTP endpoint",
        details=f"status={snapshot.get('dashboard_http_status_code', 'unknown')}, response={_fmt(_value(snapshot, 'dashboard_http_response_ms'), ' ms', 0)}",
    )

    perf_log_recent = _state(snapshot, "dashboard_perf_log_recent_state")
    perf_level = _level_from_bool(perf_log_recent)
    _health_check(
        checks,
        perf_level,
        "dashboard",
        "Dashboard performance log freshness",
        details=f"age={_fmt(_value(snapshot, 'dashboard_perf_log_age_min'), ' min')}",
        affects_overall=False,
    )

    for prefix, label in (
        ("host_celine_source", "CL61 root disk"),
        ("host_celine_data", "CL61 data disk"),
        ("host_ass_root", "ASS root disk"),
        ("host_ass_data", "ASS data disk"),
        ("host_aps_root", "APS root disk"),
        ("host_aps_data", "APS data disk"),
        ("aurora_project", "Aurora raw mirror disk"),
        ("aurora_data", "AURORA Cloud product disk"),
        ("aurora_root", "AURORA Cloud root disk"),
        ("gws_storage", "JASMIN GWS"),
    ):
        probe_level = "green"
        if prefix.startswith("host_"):
            probe_level = _level_from_bool(_state(snapshot, f"{prefix}_probe_ok_state"))
        elif prefix == "gws_storage":
            probe_level = _level_from_bool(_state(snapshot, "gws_probe_ok_state"))
        used_level = _level_from_used_pct(_value(snapshot, f"{prefix}_used_pct"))
        _health_check(
            checks,
            _worst_level([probe_level, used_level]),
            "storage",
            label,
            details=(
                f"used={_fmt(_value(snapshot, f'{prefix}_used_pct'), '%', 0)}, "
                f"free={_fmt(_value(snapshot, f'{prefix}_free_gb'), ' GB')}, "
                f"path={snapshot.get(f'{prefix}_resolved_path', '')}"
            ),
        )

    battery_level = _level_from_battery_voltage(_value(snapshot, "aps_battery_voltage_v"))
    _health_check(
        checks,
        battery_level,
        "power",
        "APS battery voltage",
        details=f"DC inverter voltage={_fmt(_value(snapshot, 'aps_battery_voltage_v'), ' V', 2)}, age={_fmt(_value(snapshot, 'aps_battery_voltage_age_min'), ' min')}",
    )
    soc_level = _level_from_battery_soc(_value(snapshot, "aps_battery_soc_pct"))
    _health_check(
        checks,
        soc_level,
        "power",
        "APS battery state of charge",
        details=f"SOC={_fmt(_value(snapshot, 'aps_battery_soc_pct'), '%', 0)}, age={_fmt(_value(snapshot, 'aps_battery_soc_age_min'), ' min')}",
    )
    temp_level = _level_from_internal_temp(_value(snapshot, "aps_internal_temp_c"))
    _health_check(
        checks,
        temp_level,
        "power",
        "APS internal temperature",
        details=f"temperature={_fmt(_value(snapshot, 'aps_internal_temp_c'), ' C')}, age={_fmt(_value(snapshot, 'aps_internal_temp_age_min'), ' min')}",
    )

    for stream_name, prefix in STREAM_PREFIXES.items():
        label = stream_name.replace("_", " ")
        source_level = _level_from_bool(_state(snapshot, f"{prefix}_source_recent_state"))
        local_level = _level_from_count(_value(snapshot, f"{prefix}_local_missing_count") or 0.0)
        local_mismatch_level = _level_from_count(_value(snapshot, f"{prefix}_local_mismatch_count") or 0.0)
        gws_level = _level_from_count(_value(snapshot, f"{prefix}_gws_missing_count") or 0.0)
        gws_mismatch_level = _level_from_count(_value(snapshot, f"{prefix}_gws_mismatch_count") or 0.0)
        product_level = _level_from_bool(_state(snapshot, f"{prefix}_product_gate_ok_state"))
        prune_level = _level_from_bool(_state(snapshot, f"{prefix}_prune_ready_state"))
        _health_check(
            checks,
            _worst_level([source_level, local_level, local_mismatch_level, gws_level, gws_mismatch_level, product_level, prune_level]),
            "stream",
            f"{label} stream",
            details=(
                f"source_age={_fmt(_value(snapshot, f'{prefix}_source_age_min'), ' min')}, "
                f"local_coverage={_fmt(_value(snapshot, f'{prefix}_local_coverage_pct'), '%', 2)}, "
                f"gws_coverage={_fmt(_value(snapshot, f'{prefix}_gws_coverage_pct'), '%', 2)}"
            ),
        )

    for unit in (*SOURCE_SYNC_UNITS, *PROCESSING_UNITS, *TRANSFER_UNITS):
        slug = _unit_slug(unit)
        if unit.endswith(".timer"):
            level = _level_from_bool(_state(snapshot, f"{slug}_active_state"))
            message = "Timer active"
        else:
            level = _level_from_bool(_state(snapshot, f"{slug}_healthy_state"))
            message = "Service healthy"
        _health_check(checks, level, "systemd", unit, details=message)

    for prefix, label in (
        ("dashboard_code", "Dashboard repository"),
        ("infra_code", "Infrastructure repository"),
    ):
        exists_level = _level_from_bool(_state(snapshot, f"{prefix}_repo_exists_state"))
        dirty_count = _value(snapshot, f"{prefix}_git_dirty_count")
        dirty_level = _level_from_count(dirty_count, amber_at=10.0)
        behind_count = _value(snapshot, f"{prefix}_git_behind_count")
        ahead_count = _value(snapshot, f"{prefix}_git_ahead_count")
        behind_level = "gray" if behind_count is None else ("amber" if behind_count > 0 else "green")
        ahead_level = "gray" if ahead_count is None else ("amber" if ahead_count > 0 else "green")
        _health_check(
            checks,
            _worst_level([exists_level, dirty_level, behind_level, ahead_level]),
            "code",
            label,
            details=(
                f"branch={snapshot.get(f'{prefix}_git_branch', '')}, "
                f"commit={snapshot.get(f'{prefix}_git_commit', '')}, "
                f"dirty={int(dirty_count or 0)}, "
                f"behind={int(_value(snapshot, f'{prefix}_git_behind_count') or 0)}, "
                f"ahead={int(_value(snapshot, f'{prefix}_git_ahead_count') or 0)}"
            ),
        )

    # Performance/logging telemetry is diagnostic. Keep it in the report, but
    # keep the top-level action status focused on data flow, storage, power,
    # transfer, service, and code health.
    actionable_levels = [check["level"] for check in checks if check.get("affects_overall", True)]
    overall_level = _worst_level(actionable_levels)
    observations = [check for check in checks if check["level"] in {"amber", "red", "gray"}]
    return {
        "generated_at_utc": snapshot.get("time_utc"),
        "phase": "observe-only",
        "automated_repairs_enabled": False,
        "overall_level": overall_level,
        "summary": {
            "checks": len(checks),
            "red": sum(1 for check in checks if check["level"] == "red"),
            "amber": sum(1 for check in checks if check["level"] == "amber"),
            "gray": sum(1 for check in checks if check["level"] == "gray"),
            "green": sum(1 for check in checks if check["level"] == "green"),
        },
        "raw_snapshot_path": str(raw_snapshot_path) if raw_snapshot_path else "",
        "observations": observations,
        "checks": checks,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_value(row, key) for row in rows]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def render_daily_report(snapshot: dict[str, Any], health: dict[str, Any], raw_snapshot_path: Path) -> str:
    rows = _read_jsonl(raw_snapshot_path)
    current_observations = health["observations"]
    red_observations = [item for item in current_observations if item["level"] == "red"]
    amber_observations = [item for item in current_observations if item["level"] == "amber"]
    generated = snapshot.get("time_utc", "")
    day = datetime.fromisoformat(str(generated).replace("Z", "+00:00")).strftime("%Y-%m-%d") if generated else "unknown"

    first_time = rows[0].get("time_utc", "unknown") if rows else "unknown"
    last_time = rows[-1].get("time_utc", "unknown") if rows else "unknown"
    lines = [
        f"# Aurora Health Report - {day} UTC",
        "",
        f"Generated: `{generated}`",
        "",
        f"Overall status: **{health['overall_level'].upper()}**",
        "",
        "This is an observe-only report. No automated repair, restart, deletion, or code change was attempted by this collector.",
        "",
        "## Current Issues",
    ]
    if not current_observations:
        lines.append("- No current red, amber, or unknown checks.")
    else:
        for item in [*red_observations, *amber_observations, *[obs for obs in current_observations if obs["level"] == "gray"]][:20]:
            details = f" - {item['details']}" if item.get("details") else ""
            scope = "" if item.get("affects_overall", True) else " _(diagnostic only)_"
            lines.append(f"- **{item['level'].upper()}** `{item['component']}`: {item['message']}{scope}{details}")

    lines.extend(
        [
            "",
            "## Today So Far",
            "",
            f"- Samples collected: `{len(rows)}`",
            f"- First sample: `{first_time}`",
            f"- Latest sample: `{last_time}`",
            f"- Max source sync failures: `{_fmt(_max_metric(rows, 'failed_source_sync_unit_count'), '', 0)}`",
            f"- Max processing failures: `{_fmt(_max_metric(rows, 'failed_processing_unit_count'), '', 0)}`",
            f"- Max transfer failures: `{_fmt(_max_metric(rows, 'failed_transfer_unit_count'), '', 0)}`",
            f"- Max stale source streams: `{_fmt(_max_metric(rows, 'streams_source_stale_count'), '', 0)}`",
            f"- Max local mirror issue streams: `{_fmt(_max_metric(rows, 'streams_local_issue_count'), '', 0)}`",
            f"- Max GWS mirror issue streams: `{_fmt(_max_metric(rows, 'streams_gws_issue_count'), '', 0)}`",
            "",
            "## Stream Freshness",
            "",
            "| Stream | Source age | Local coverage | GWS coverage | Product gate | Prune gate |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for stream_name, prefix in STREAM_PREFIXES.items():
        product = "ok" if _state(snapshot, f"{prefix}_product_gate_ok_state") else "blocked"
        prune = "ok" if _state(snapshot, f"{prefix}_prune_ready_state") else "blocked"
        lines.append(
            f"| {stream_name.replace('_', ' ')} | "
            f"{_fmt(_value(snapshot, f'{prefix}_source_age_min'), ' min')} | "
            f"{_fmt(_value(snapshot, f'{prefix}_local_coverage_pct'), '%', 2)} | "
            f"{_fmt(_value(snapshot, f'{prefix}_gws_coverage_pct'), '%', 2)} | "
            f"{product} | {prune} |"
        )

    lines.extend(
        [
            "",
            "## Code State",
            "",
            "| Repository | Branch | Commit | Dirty | Behind | Ahead |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for prefix, label in (("dashboard_code", "dashboard"), ("infra_code", "infra")):
        lines.append(
            f"| {label} | `{snapshot.get(f'{prefix}_git_branch', '')}` | "
            f"`{snapshot.get(f'{prefix}_git_commit', '')}` | "
            f"{int(_value(snapshot, f'{prefix}_git_dirty_count') or 0)} | "
            f"{int(_value(snapshot, f'{prefix}_git_behind_count') or 0)} | "
            f"{int(_value(snapshot, f'{prefix}_git_ahead_count') or 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_health_outputs(output_root: Path, snapshot: dict[str, Any], raw_snapshot_path: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(snapshot["time_utc"].replace("Z", "+00:00"))
    health = build_health_assessment(snapshot, raw_snapshot_path)

    latest_json = output_root / "latest_health.json"
    day_json = output_root / f"health_{stamp:%Y%m%d}.json"
    health_json = json.dumps(health, indent=2, sort_keys=True)
    latest_json.write_text(health_json, encoding="utf-8")
    day_json.write_text(health_json, encoding="utf-8")

    report = render_daily_report(snapshot, health, raw_snapshot_path)
    latest_report = output_root / "latest_report.md"
    day_report = output_root / f"health_report_{stamp:%Y%m%d}.md"
    latest_report.write_text(report, encoding="utf-8")
    day_report.write_text(report, encoding="utf-8")
    return latest_json, latest_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Aurora operations monitoring snapshots.")
    parser.add_argument("--env-file", type=Path, default=_path_from_env("AURORA_DASHBOARD_ENV_FILE", ENV_FILE_DEFAULT))
    parser.add_argument("--output-root", type=Path, default=_path_from_env("OPS_MONITOR_RAW_ROOT", RAW_ROOT_DEFAULT))
    parser.add_argument("--health-output-root", type=Path, default=_path_from_env("OPS_MONITOR_HEALTH_ROOT", HEALTH_OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--manifest-root", type=Path, default=_path_from_env("GWS_MANIFEST_ROOT", MANIFEST_ROOT_DEFAULT))
    parser.add_argument("--gws-path", type=Path, default=_path_from_env("GWS_PATH", GWS_PATH_DEFAULT))
    args = parser.parse_args()

    _load_env_file(args.env_file)

    snapshot = build_snapshot(args.manifest_root, args.gws_path)
    path = write_snapshot(args.output_root, snapshot)
    print(f"Wrote {path}")
    health_json, health_report = write_health_outputs(args.health_output_root, snapshot, path)
    print(f"Wrote {health_json}")
    print(f"Wrote {health_report}")


if __name__ == "__main__":
    main()
