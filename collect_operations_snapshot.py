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
import re
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
APS_BATTERY_CAPACITY_KWH = float(os.environ.get("APS_BATTERY_CAPACITY_KWH", "26"))
APS_BATTERY_DEPLETION_DEADBAND_W = float(os.environ.get("APS_BATTERY_DEPLETION_DEADBAND_W", "50"))
APS_INTERNAL_TEMP_LOW_AMBER_C = float(os.environ.get("APS_INTERNAL_TEMP_LOW_AMBER_C", "10"))
APS_INTERNAL_TEMP_LOW_RED_C = float(os.environ.get("APS_INTERNAL_TEMP_LOW_RED_C", "5"))
APS_INTERNAL_TEMP_HIGH_AMBER_C = float(os.environ.get("APS_INTERNAL_TEMP_HIGH_AMBER_C", "40"))
APS_INTERNAL_TEMP_HIGH_RED_C = float(os.environ.get("APS_INTERNAL_TEMP_HIGH_RED_C", "45"))
APS_DEWPOINT_RED_MARGIN_C = float(os.environ.get("APS_DEWPOINT_RED_MARGIN_C", "0"))
DASHBOARD_PERF_LOG_DEFAULT = Path("/data/aurora/products/dashboard/dashboard_perf.jsonl")
DASHBOARD_HTTP_URL_DEFAULT = "http://127.0.0.1:5006/app"
PRIMARY_DASHBOARD_URL_DEFAULT = "https://data.gamb2le.co.uk/app"
STANDBY_DASHBOARD_URL_DEFAULT = "https://data-ocean.gamb2le.co.uk/app"
DEV_LIVE_MIRROR_STAMP_DEFAULT = Path("/data/aurora/internal/dev-live-mirror/last_success.json")
REMOTE_DF_TIMEOUT_SECONDS = 45.0
LOCAL_COMMAND_TIMEOUT_SECONDS = 30.0
INFRA_REPO_DEFAULT = Path("/tmp/aurora-cloud-infra-codex")
ENV_FILE_DEFAULT = Path("/etc/aurora-dashboard.env")
KNOWN_HOSTS = Path("/home/aurora/.ssh/known_hosts")
AURORA_GUARD_STATUS_ROOT_DEFAULT = Path(os.environ.get("AURORA_GUARD_STATUS_ROOT", "/run/aurora/guarded"))
BATCH_SLICE_UNIT = "aurora-batch.slice"
GUARDED_HEAVY_UNITS = (
    "aurora-power-quicklooks.service",
    "aurora-power-soc-ensemble.service",
    "aurora-radar-daily-quicklooks.service",
    "aurora-radar-quicklooks.service",
    "aurora-ops-monitor-quicklooks.service",
    "aurora-wxcam-daily-videos.service",
    "aurora-asfs-fast-gas-append.service",
    "aurora-asfs-fast-sonic-append.service",
    "aurora-power-append.service",
    "aurora-radar-append.service",
    "aurora-wxcam-append.service",
    "aurora-gws-rsync-products-wxcam.service",
    "aurora-gws-rsync-products.service",
    "aurora-gws-rsync-raw.service",
    "aurora-mirror-verify.service",
)

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
    "asfs_fast_gas": "asfs_fast_gas",
    "power": "power",
    "pdu": "pdu",
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
    "aurora-asfs-fast-gas-source-sync.timer",
    "aurora-asfs-fast-gas-source-sync.service",
    "aurora-power-source-sync.timer",
    "aurora-power-source-sync.service",
    "aurora-pdu-source-sync.timer",
    "aurora-pdu-source-sync.service",
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
    "aurora-radar-daily-quicklooks.timer",
    "aurora-radar-daily-quicklooks.service",
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
    "aurora-asfs-fast-gas-append.timer",
    "aurora-asfs-fast-gas-append.service",
    "aurora-power-append.timer",
    "aurora-power-append.service",
    "aurora-pdu-append.timer",
    "aurora-pdu-append.service",
    "aurora-power-quicklooks.timer",
    "aurora-power-quicklooks.service",
    "aurora-power-soc-forecast.timer",
    "aurora-power-soc-forecast.service",
    "aurora-power-soc-forecast-learn.timer",
    "aurora-power-soc-forecast-learn.service",
    "aurora-power-soc-ensemble.timer",
    "aurora-power-soc-ensemble.service",
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
SOURCE_RECENT_THRESHOLD_OVERRIDES_MINUTES = {
    # HATPRO publishes as hourly batches. Wait for two missed batches before
    # marking the source stale so normal batch/manifest timing does not alert.
    "hatprog5": 180.0,
}
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
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
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
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
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
    proc = _run(base_cmd + [target, remote], timeout=REMOTE_DF_TIMEOUT_SECONDS)
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
        ],
        timeout=LOCAL_COMMAND_TIMEOUT_SECONDS,
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


def _source_recent_threshold_minutes(stream_name: str) -> float:
    return SOURCE_RECENT_THRESHOLD_OVERRIDES_MINUTES.get(stream_name, SOURCE_RECENT_THRESHOLD_MINUTES)


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


def _html_title(body: bytes) -> str:
    try:
        text = body[:64 * 1024].decode("utf-8", errors="ignore")
    except Exception:
        return ""
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _dashboard_document_state(status_code: int, body: bytes) -> int:
    if not (200 <= status_code < 400):
        return 0
    # A crashed Panel handler can still return a small Bokeh shell with title
    # "Bokeh Application". Require the actual dashboard document marker.
    return 1 if b"AURORA Data Viewer" in body and len(body) > 100_000 else 0


def _probe_http(url: str) -> dict[str, float | int | str | None]:
    start = time.monotonic()
    # Panel serves the dashboard route for GET requests but returns 405 for
    # HEAD. Use a normal GET so the service log only records real failures.
    request = urllib.request.Request(url, headers={"User-Agent": "aurora-ops-monitor/1.0"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = int(response.status)
            body = response.read(1024 * 1024)
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        try:
            body = exc.read(1024 * 1024)
        except Exception:
            body = b""
        return {
            "ok_state": 1 if 200 <= status_code < 400 or status_code == 405 else 0,
            "status_code": status_code,
            "response_ms": (time.monotonic() - start) * 1000.0,
            "content_bytes": len(body),
            "title": _html_title(body),
            "full_document_state": _dashboard_document_state(status_code, body),
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok_state": 0,
            "status_code": None,
            "response_ms": (time.monotonic() - start) * 1000.0,
            "content_bytes": 0,
            "title": "",
            "full_document_state": 0,
            "error": str(exc),
        }
    return {
        "ok_state": 1 if 200 <= status_code < 400 or status_code == 405 else 0,
        "status_code": status_code,
        "response_ms": (time.monotonic() - start) * 1000.0,
        "content_bytes": len(body),
        "title": _html_title(body),
        "full_document_state": _dashboard_document_state(status_code, body),
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
    record[f"{prefix}_git_tag"] = _git_value(repo, ["describe", "--tags", "--exact-match"]) or ""
    record[f"{prefix}_git_describe"] = _git_value(repo, ["describe", "--tags", "--always", "--dirty"]) or ""
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


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime(warn=False)


def _collect_dev_live_mirror_metrics(record: dict[str, Any], now_epoch: int) -> None:
    stamp_path = _path_from_env("AURORA_DEV_LIVE_MIRROR_STAMP", DEV_LIVE_MIRROR_STAMP_DEFAULT)
    record["dev_live_mirror_stamp_path"] = str(stamp_path)
    if not stamp_path.exists():
        record["dev_live_mirror_stamp_exists_state"] = 0
        return

    record["dev_live_mirror_stamp_exists_state"] = 1
    try:
        stat = stamp_path.stat()
        record["dev_live_mirror_stamp_age_min"] = max(now_epoch - stat.st_mtime, 0.0) / 60.0
    except OSError:
        pass

    payload: dict[str, Any] = {}
    text = ""
    try:
        text = stamp_path.read_text(encoding="utf-8").strip()
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"last_success_utc": text}
    except Exception as exc:
        record["dev_live_mirror_error"] = str(exc)
        return

    if isinstance(payload, dict):
        for key in ("source_host", "source_user", "site_env", "paths_replicated", "rsync_exit_code"):
            if key in payload:
                record[f"dev_live_mirror_{key}"] = payload[key]
        last_success = _parse_utc_timestamp(str(payload.get("last_success_utc") or ""))
    else:
        last_success = _parse_utc_timestamp(text)

    if last_success is None:
        record["dev_live_mirror_recent_state"] = 0
        record["dev_live_mirror_error"] = "No valid last_success_utc in mirror stamp"
        return

    now_dt = datetime.fromtimestamp(now_epoch, timezone.utc)
    age_min = max((now_dt - last_success).total_seconds(), 0.0) / 60.0
    threshold_min = float(os.environ.get("AURORA_DEV_LIVE_MIRROR_RECENT_MINUTES", "7.5"))
    record["dev_live_mirror_last_success_utc"] = last_success.isoformat().replace("+00:00", "Z")
    record["dev_live_mirror_age_min"] = age_min
    record["dev_live_mirror_recent_threshold_min"] = threshold_min
    record["dev_live_mirror_recent_state"] = 1 if age_min <= threshold_min else 0


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


def _latest_finite_zarr_row(
    zarr_path: Path,
    var_names: tuple[str, ...],
    *,
    time_name: str = "time",
) -> tuple[dict[str, float], datetime | None]:
    if not zarr_path.exists() or not var_names:
        return {}, None
    ds = xr.open_zarr(zarr_path)
    try:
        if time_name not in ds:
            return {}, None
        for name in var_names:
            if name not in ds or time_name not in ds[name].dims:
                return {}, None
        total = int(ds[var_names[0]].sizes.get(time_name, 0))
        if total <= 0:
            return {}, None
        time_coord = ds[time_name]
        for window in (2048, 16384, None):
            selector = slice(None) if window is None or total <= window else slice(-window, None)
            arrays = {name: np.asarray(ds[name].isel({time_name: selector}).values, dtype=float) for name in var_names}
            mask = np.ones(len(next(iter(arrays.values()))), dtype=bool)
            for values in arrays.values():
                mask &= np.isfinite(values)
            finite_idx = np.flatnonzero(mask)
            if finite_idx.size == 0:
                continue
            idx = int(finite_idx[-1])
            times = np.asarray(time_coord.isel({time_name: selector}).values)
            timestamp = pd.Timestamp(times[idx])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return {name: float(values[idx]) for name, values in arrays.items()}, timestamp.to_pydatetime(warn=False)
        return {}, None
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()


def _dewpoint_c(temperature_c: float | None, humidity_pct: float | None) -> float | None:
    temperature = _float_or_none(temperature_c)
    humidity = _float_or_none(humidity_pct)
    if temperature is None or humidity is None or humidity <= 0.0 or humidity > 100.0:
        return None
    a = 17.625
    b = 243.04
    gamma = math.log(humidity / 100.0) + (a * temperature) / (b + temperature)
    return (b * gamma) / (a - gamma)


def _unit_slug(unit: str) -> str:
    return unit.replace("aurora-", "").replace(".", "_").replace("-", "_")


def _systemd_show(unit: str, props: tuple[str, ...] | None = None) -> dict[str, str]:
    if props is None:
        props = (
            "ActiveState",
            "UnitFileState",
            "Result",
            "ExecMainExitTimestamp",
            "LastTriggerUSec",
            "NextElapseUSecRealtime",
        )
    prop_args = [arg for prop in props for arg in ("-p", prop)]
    proc = _run(
        ["systemctl", "show", unit, *prop_args],
        check=False,
        timeout=LOCAL_COMMAND_TIMEOUT_SECONDS,
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
    if result in {"success", "", "done", "exec-condition", "condition"}:
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


def _parse_systemd_bytes(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"infinity", "[not set]", "n/a", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?)", text.lower())
        if not match:
            return None
        number = float(match.group(1))
        multiplier = {"": 1.0, "k": 1024.0, "m": 1024.0 ** 2, "g": 1024.0 ** 3, "t": 1024.0 ** 4}[match.group(2)]
        number *= multiplier
    if number >= 2**63:
        return None
    return number


def _parse_systemd_duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"infinity", "[not set]", "n/a", "none"}:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s|min|h)?", text.lower())
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "us":
        return number / 1_000_000.0
    if unit == "ms":
        return number / 1000.0
    if unit == "min":
        return number * 60.0
    if unit == "h":
        return number * 3600.0
    if unit == "s":
        return number
    return number / 1_000_000.0 if number > 10000 else number


def _parse_event_time_utc(value: Any) -> float | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.timestamp()


def _pid_alive(pid_value: Any) -> bool:
    try:
        pid = int(pid_value)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _collect_guard_events(status_root: Path, now_epoch: float) -> dict[str, int]:
    counts = {
        "event_count_24h": 0,
        "skip_count_24h": 0,
        "acquired_count_24h": 0,
        "released_count_24h": 0,
        "quicklook_skip_count_24h": 0,
        "video_skip_count_24h": 0,
        "append_skip_count_24h": 0,
    }
    event_path = status_root / "events.jsonl"
    if not event_path.exists():
        return counts
    cutoff_epoch = now_epoch - 24.0 * 3600.0
    try:
        lines = event_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return counts
    for line in lines[-2000:]:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_epoch = _parse_event_time_utc(event.get("time_utc"))
        if event_epoch is None or event_epoch < cutoff_epoch or event_epoch > now_epoch + 300.0:
            continue
        counts["event_count_24h"] += 1
        event_name = str(event.get("event", ""))
        guard_class = str(event.get("class", ""))
        if event_name == "skipped":
            counts["skip_count_24h"] += 1
            if guard_class == "quicklook-heavy":
                counts["quicklook_skip_count_24h"] += 1
            elif guard_class == "video-heavy":
                counts["video_skip_count_24h"] += 1
            elif guard_class == "append-io":
                counts["append_skip_count_24h"] += 1
        elif event_name == "acquired":
            counts["acquired_count_24h"] += 1
        elif event_name == "released":
            counts["released_count_24h"] += 1
    return counts


def _collect_batch_resource_metrics(record: dict[str, Any], now_epoch: float) -> None:
    info = _systemd_show(
        BATCH_SLICE_UNIT,
        (
            "ActiveState",
            "CPUQuotaPerSecUSec",
            "CPUWeight",
            "IOWeight",
            "MemoryCurrent",
            "MemoryHigh",
        ),
    )
    record["aurora_batch_active_state"] = 1 if info.get("ActiveState") == "active" else 0
    cpu_quota_seconds = _parse_systemd_duration_seconds(info.get("CPUQuotaPerSecUSec"))
    if cpu_quota_seconds is not None:
        record["aurora_batch_cpu_quota_cores"] = cpu_quota_seconds
    for key, prop in (
        ("aurora_batch_cpu_weight", "CPUWeight"),
        ("aurora_batch_io_weight", "IOWeight"),
    ):
        try:
            record[key] = float(info[prop])
        except Exception:
            pass
    memory_current = _parse_systemd_bytes(info.get("MemoryCurrent"))
    memory_high = _parse_systemd_bytes(info.get("MemoryHigh"))
    if memory_current is not None:
        record["aurora_batch_memory_current_mb"] = memory_current / (1024.0 ** 2)
    if memory_high is not None:
        record["aurora_batch_memory_high_mb"] = memory_high / (1024.0 ** 2)
    if memory_current is not None and memory_high and memory_high > 0:
        record["aurora_batch_memory_pressure_pct"] = memory_current / memory_high * 100.0

    active_units: list[str] = []
    for unit in GUARDED_HEAVY_UNITS:
        unit_info = _systemd_show(unit, ("ActiveState", "SubState", "MainPID"))
        active = unit_info.get("ActiveState") in {"active", "activating"} and unit_info.get("_exists") == "1"
        main_pid = unit_info.get("MainPID")
        if main_pid and main_pid != "0":
            active = active or _pid_alive(main_pid)
        if active:
            active_units.append(unit)
    record["aurora_batch_active_heavy_job_count"] = len(active_units)
    record["aurora_batch_active_heavy_jobs"] = ", ".join(active_units[:12])

    current_root = AURORA_GUARD_STATUS_ROOT_DEFAULT / "current"
    lock_units: list[str] = []
    stale_locks = 0
    if current_root.exists():
        for path in sorted(current_root.glob("*.json")):
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _pid_alive(current.get("holder_pid")):
                lock_units.append(str(current.get("unit") or path.stem))
            else:
                stale_locks += 1
    record["aurora_guard_lock_active_count"] = len(lock_units)
    record["aurora_guard_lock_active_units"] = ", ".join(lock_units[:12])
    record["aurora_guard_stale_lock_count"] = stale_locks

    for key, value in _collect_guard_events(AURORA_GUARD_STATUS_ROOT_DEFAULT, now_epoch).items():
        record[f"aurora_guard_{key}"] = value


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
    _collect_dev_live_mirror_metrics(record, now_epoch)

    dashboard_probe = _probe_http(os.environ.get("AURORA_DASHBOARD_HEALTH_URL", DASHBOARD_HTTP_URL_DEFAULT))
    for key, value in dashboard_probe.items():
        record[f"dashboard_http_{key}"] = value

    record["site_env"] = os.environ.get("AURORA_SITE_ENV", "")
    record["failover_collector_role"] = os.environ.get("AURORA_FAILOVER_ROLE", "")
    record["failover_collector_domain"] = os.environ.get("AURORA_DOMAIN", "")
    for endpoint, default_url in (
        ("primary", PRIMARY_DASHBOARD_URL_DEFAULT),
        ("standby", STANDBY_DASHBOARD_URL_DEFAULT),
    ):
        env_name = f"AURORA_{endpoint.upper()}_DASHBOARD_URL"
        url = os.environ.get(env_name, default_url)
        record[f"failover_{endpoint}_dashboard_url"] = url
        endpoint_probe = _probe_http(url)
        for key, value in endpoint_probe.items():
            record[f"failover_{endpoint}_dashboard_http_{key}"] = value

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
    battery_power, battery_power_time = _latest_finite_zarr_value(
        _path_from_env("POWER_ZARR_PATH", POWER_ZARR_DEFAULT),
        "BatteryWatts",
    )
    record["aps_battery_power_w"] = battery_power
    record["aps_battery_capacity_kwh"] = APS_BATTERY_CAPACITY_KWH
    record["aps_battery_depletion_deadband_w"] = APS_BATTERY_DEPLETION_DEADBAND_W
    if battery_power_time is not None:
        record["aps_battery_power_time_utc"] = battery_power_time.isoformat()
        record["aps_battery_power_age_min"] = max((now - battery_power_time).total_seconds(), 0.0) / 60.0
    if battery_soc is not None:
        remaining_kwh = max(float(battery_soc), 0.0) / 100.0 * APS_BATTERY_CAPACITY_KWH
        record["aps_battery_remaining_kwh"] = remaining_kwh
        if battery_power is not None:
            if battery_power < -APS_BATTERY_DEPLETION_DEADBAND_W:
                discharge_kw = abs(float(battery_power)) / 1000.0
                record["aps_battery_depleting_state"] = 1
                record["aps_battery_charging_state"] = 0
                record["aps_battery_discharge_power_w"] = abs(float(battery_power))
                record["aps_battery_depletion_hours"] = remaining_kwh / discharge_kw if discharge_kw > 0 else None
            elif battery_power > APS_BATTERY_DEPLETION_DEADBAND_W:
                record["aps_battery_depleting_state"] = 0
                record["aps_battery_charging_state"] = 1
                record["aps_battery_discharge_power_w"] = 0.0
            else:
                record["aps_battery_depleting_state"] = 0
                record["aps_battery_charging_state"] = 0
                record["aps_battery_discharge_power_w"] = 0.0
    power_zarr_path = _path_from_env("POWER_ZARR_PATH", POWER_ZARR_DEFAULT)
    internal_temp, internal_temp_time = _latest_finite_zarr_value(
        power_zarr_path,
        "InternalTemperature",
    )
    record["aps_internal_temp_c"] = internal_temp
    if internal_temp_time is not None:
        record["aps_internal_temp_time_utc"] = internal_temp_time.isoformat()
        record["aps_internal_temp_age_min"] = max((now - internal_temp_time).total_seconds(), 0.0) / 60.0
    humidity_row, humidity_time = _latest_finite_zarr_row(power_zarr_path, ("InternalTemperature", "InternalHumidity"))
    if humidity_row:
        humidity = humidity_row.get("InternalHumidity")
        dewpoint_temp = humidity_row.get("InternalTemperature")
        dewpoint = _dewpoint_c(dewpoint_temp, humidity)
        record["aps_internal_humidity_available_state"] = 1
        record["aps_internal_humidity_pct"] = humidity
        record["aps_internal_dewpoint_temp_c"] = dewpoint_temp
        if humidity_time is not None:
            record["aps_internal_humidity_time_utc"] = humidity_time.isoformat()
            record["aps_internal_humidity_age_min"] = max((now - humidity_time).total_seconds(), 0.0) / 60.0
            record["aps_internal_dewpoint_time_utc"] = humidity_time.isoformat()
        if dewpoint is not None:
            margin = float(dewpoint_temp) - dewpoint
            record["aps_internal_dewpoint_c"] = dewpoint
            record["aps_internal_dewpoint_margin_c"] = margin
            record["aps_internal_dewpoint_risk_state"] = 1 if margin <= APS_DEWPOINT_RED_MARGIN_C else 0
    else:
        record["aps_internal_humidity_available_state"] = 0

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
        source_recent_state = _recent_state(source_age_min, _source_recent_threshold_minutes(stream_name))
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
    _collect_batch_resource_metrics(record, now_epoch)
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


def _level_from_battery_depletion(snapshot: dict[str, Any]) -> str:
    power_w = _value(snapshot, "aps_battery_power_w")
    soc = _value(snapshot, "aps_battery_soc_pct")
    if power_w is None or soc is None:
        return "gray"
    deadband_w = _value(snapshot, "aps_battery_depletion_deadband_w") or APS_BATTERY_DEPLETION_DEADBAND_W
    if power_w >= -deadband_w:
        return "green"
    hours = _value(snapshot, "aps_battery_depletion_hours")
    if hours is None:
        capacity_kwh = _value(snapshot, "aps_battery_capacity_kwh") or APS_BATTERY_CAPACITY_KWH
        remaining_kwh = max(soc, 0.0) / 100.0 * capacity_kwh
        discharge_kw = abs(power_w) / 1000.0
        hours = remaining_kwh / discharge_kw if discharge_kw > 0 else None
    if hours is None:
        return "gray"
    if hours >= 24.0:
        return "green"
    if hours >= 12.0:
        return "amber"
    return "red"


def _level_from_internal_temp(value: float | None) -> str:
    if value is None:
        return "gray"
    if value < APS_INTERNAL_TEMP_LOW_RED_C or value >= APS_INTERNAL_TEMP_HIGH_RED_C:
        return "red"
    if value < APS_INTERNAL_TEMP_LOW_AMBER_C or value >= APS_INTERNAL_TEMP_HIGH_AMBER_C:
        return "amber"
    return "green"


def _level_from_dewpoint_margin(snapshot: dict[str, Any]) -> str:
    if _state(snapshot, "aps_internal_humidity_available_state") is False:
        return "gray"
    margin = _value(snapshot, "aps_internal_dewpoint_margin_c")
    if margin is None:
        return "gray"
    if margin <= APS_DEWPOINT_RED_MARGIN_C:
        return "red"
    return "green"


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
    site_env = str(snapshot.get("site_env") or "").strip().lower()

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
    if site_env:
        _health_check(
            checks,
            "green" if site_env in {"production", "development"} else "amber",
            "deployment",
            "Dashboard site environment",
            details=f"site_env={site_env}",
            affects_overall=False,
        )
    if site_env == "development":
        mirror_level = _level_from_bool(_state(snapshot, "dev_live_mirror_recent_state"))
        _health_check(
            checks,
            mirror_level,
            "deployment",
            "Development live mirror freshness",
            details=(
                f"age={_fmt(_value(snapshot, 'dev_live_mirror_age_min'), ' min')}, "
                f"threshold={_fmt(_value(snapshot, 'dev_live_mirror_recent_threshold_min'), ' min')}, "
                f"stamp={snapshot.get('dev_live_mirror_stamp_path', '')}"
            ),
        )

    batch_level = _level_from_used_pct(_value(snapshot, "aurora_batch_memory_pressure_pct"))
    _health_check(
        checks,
        batch_level,
        "dashboard",
        "Batch resource pressure",
        details=(
            f"memory={_fmt(_value(snapshot, 'aurora_batch_memory_pressure_pct'), '%', 0)}, "
            f"current={_fmt(_value(snapshot, 'aurora_batch_memory_current_mb'), ' MB', 0)}, "
            f"limit={_fmt(_value(snapshot, 'aurora_batch_memory_high_mb'), ' MB', 0)}, "
            f"active_jobs={_fmt(_value(snapshot, 'aurora_batch_active_heavy_job_count'), '', 0)}"
        ),
        affects_overall=False,
    )

    guard_level = _level_from_count(_value(snapshot, "aurora_guard_skip_count_24h"), amber_at=5.0)
    _health_check(
        checks,
        guard_level,
        "systemd",
        "Guarded job skips",
        details=(
            f"skips_24h={_fmt(_value(snapshot, 'aurora_guard_skip_count_24h'), '', 0)}, "
            f"active_locks={_fmt(_value(snapshot, 'aurora_guard_lock_active_count'), '', 0)}, "
            f"quicklook_skips={_fmt(_value(snapshot, 'aurora_guard_quicklook_skip_count_24h'), '', 0)}, "
            f"append_skips={_fmt(_value(snapshot, 'aurora_guard_append_skip_count_24h'), '', 0)}"
        ),
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
    depletion_level = _level_from_battery_depletion(snapshot)
    _health_check(
        checks,
        depletion_level,
        "power",
        "APS estimated time until depleted",
        details=(
            f"time={_fmt(_value(snapshot, 'aps_battery_depletion_hours'), ' h')}, "
            f"remaining={_fmt(_value(snapshot, 'aps_battery_remaining_kwh'), ' kWh')}, "
            f"battery_power={_fmt(_value(snapshot, 'aps_battery_power_w'), ' W', 0)}, "
            f"capacity={_fmt(_value(snapshot, 'aps_battery_capacity_kwh'), ' kWh', 0)}"
        ),
    )
    temp_level = _level_from_internal_temp(_value(snapshot, "aps_internal_temp_c"))
    _health_check(
        checks,
        temp_level,
        "power",
        "APS internal temperature",
        details=(
            f"temperature={_fmt(_value(snapshot, 'aps_internal_temp_c'), ' C')}, "
            f"age={_fmt(_value(snapshot, 'aps_internal_temp_age_min'), ' min')}, "
            f"green={APS_INTERNAL_TEMP_LOW_AMBER_C:.0f}-{APS_INTERNAL_TEMP_HIGH_AMBER_C:.0f} C, "
            f"red <{APS_INTERNAL_TEMP_LOW_RED_C:.0f} C or >={APS_INTERNAL_TEMP_HIGH_RED_C:.0f} C"
        ),
    )
    dewpoint_level = _level_from_dewpoint_margin(snapshot)
    _health_check(
        checks,
        dewpoint_level,
        "power",
        "APS internal dew point margin",
        details=(
            f"humidity={_fmt(_value(snapshot, 'aps_internal_humidity_pct'), '%', 0)}, "
            f"dewpoint={_fmt(_value(snapshot, 'aps_internal_dewpoint_c'), ' C')}, "
            f"margin={_fmt(_value(snapshot, 'aps_internal_dewpoint_margin_c'), ' C')}, "
            f"age={_fmt(_value(snapshot, 'aps_internal_humidity_age_min'), ' min')}"
        ),
        affects_overall=dewpoint_level != "gray",
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
            f"- Max batch memory pressure: `{_fmt(_max_metric(rows, 'aurora_batch_memory_pressure_pct'), '%', 0)}`",
            f"- Max active guarded jobs: `{_fmt(_max_metric(rows, 'aurora_batch_active_heavy_job_count'), '', 0)}`",
            f"- Guarded job skips in latest 24 h: `{_fmt(_value(snapshot, 'aurora_guard_skip_count_24h'), '', 0)}`",
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
            "## Deployment State",
            "",
            f"- Site environment: `{snapshot.get('site_env', '') or 'unknown'}`",
            f"- Served domain: `{snapshot.get('failover_collector_domain', '') or 'unknown'}`",
            f"- Collector role: `{snapshot.get('failover_collector_role', '') or 'unknown'}`",
            f"- Development mirror lag: `{_fmt(_value(snapshot, 'dev_live_mirror_age_min'), ' min')}`",
            f"- Development mirror last success: `{snapshot.get('dev_live_mirror_last_success_utc', '') or 'unknown'}`",
            "",
            "## Code State",
            "",
            "| Repository | Branch | Commit | Tag | Describe | Dirty | Behind | Ahead |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for prefix, label in (("dashboard_code", "dashboard"), ("infra_code", "infra")):
        lines.append(
            f"| {label} | `{snapshot.get(f'{prefix}_git_branch', '')}` | "
            f"`{snapshot.get(f'{prefix}_git_commit', '')}` | "
            f"`{snapshot.get(f'{prefix}_git_tag', '') or 'none'}` | "
            f"`{snapshot.get(f'{prefix}_git_describe', '')}` | "
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
