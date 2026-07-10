#!/usr/bin/env python3
"""Rebuild Aurora dashboard product Zarrs from a shared UTC cutoff.

This script stages derived Zarr products into temporary sibling paths, validates
them, backs up existing product stores, swaps validated stores into place, then
refreshes quicklooks and restarts dashboard processing services. It never writes
raw mirror inputs under /project/aurora/raw.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from rebuild_cutoff import parse_from_time


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PRODUCT_ROOT = Path("/data/aurora/products")
DEFAULT_RAW_ROOT = Path("/project/aurora/raw")
DEFAULT_STATE_ROOT = Path("/var/lib/aurora-cloud")

PROCESSING_TIMERS = [
    "aurora-ceilometer-append.timer",
    "aurora-ceilometer-last24h.timer",
    "aurora-ceilometer-quicklooks.timer",
    "aurora-radar-append.timer",
    "aurora-radar-quicklooks.timer",
    "aurora-hatpro-append.timer",
    "aurora-hatpro-quicklooks.timer",
    "aurora-vaisalamet-append.timer",
    "aurora-vaisalamet-quicklooks.timer",
    "aurora-asfs-logger-append.timer",
    "aurora-asfs-logger-quicklooks.timer",
    "aurora-asfs-fast-sonic-append.timer",
    "aurora-asfs-fast-sonic-quicklooks.timer",
    "aurora-asfs-fast-gas-append.timer",
    "aurora-power-append.timer",
    "aurora-power-quicklooks.timer",
    "aurora-ops-monitor-collect.timer",
    "aurora-ops-monitor-append.timer",
    "aurora-ops-monitor-alerts.timer",
    "aurora-ops-monitor-quicklooks.timer",
    "aurora-les-operational-run.timer",
    "aurora-wxcam-catalog.timer",
    "aurora-wxcam-daily-videos.timer",
    "aurora-wxcam-append.timer",
]

PROCESSING_SERVICES = [unit.replace(".timer", ".service") for unit in PROCESSING_TIMERS]
DASHBOARD_SERVICE = "aurora-dashboard.service"


@dataclass
class Artifact:
    name: str
    target: Path
    temp: Path


@dataclass
class ProductSpec:
    name: str
    target: Path
    temp: Path
    command: list[str]
    source: str
    required_vars: tuple[str, ...] = ()
    wxcam_groups: tuple[str, ...] = ()
    extra_artifacts: list[Artifact] = field(default_factory=list)
    skip_run: bool = False


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return _iso_z(value)
    return str(value)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _tmp_store_path(target: Path, stamp: str) -> Path:
    return target.with_name(f".{target.name}.rebuild-{stamp}.tmp")


def _relative_backup_path(path: Path, product_root: Path) -> Path:
    try:
        return path.relative_to(product_root)
    except ValueError:
        return Path("external") / path.name


def _backup_path(target: Path, backup_root: Path, stamp: str, product_root: Path) -> Path:
    return backup_root / stamp / _relative_backup_path(target, product_root)


def _failed_install_path(target: Path, backup_root: Path, stamp: str, product_root: Path) -> Path:
    return backup_root / stamp / "failed-installs" / _relative_backup_path(target, product_root)


def _remove_temp(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "command": command,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "dry_run": dry_run,
    }
    if dry_run:
        log_path.write_text("DRY RUN: " + " ".join(command) + "\n", encoding="utf-8")
        record["returncode"] = 0
        return record

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        record["error"] = f"command failed with exit code {completed.returncode}"
    return record


def _systemctl(args: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["systemctl", *args]
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="dry-run\n", stderr="")
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _unit_state(unit: str, *, dry_run: bool) -> dict[str, Any]:
    active = _systemctl(["is-active", unit], dry_run=dry_run)
    enabled = _systemctl(["is-enabled", unit], dry_run=dry_run)
    return {
        "active": active.stdout.strip(),
        "active_returncode": active.returncode,
        "enabled": enabled.stdout.strip(),
        "enabled_returncode": enabled.returncode,
    }


def _stop_units(units: list[str], *, dry_run: bool) -> dict[str, dict[str, Any]]:
    states = {unit: _unit_state(unit, dry_run=dry_run) for unit in units}
    for unit in units:
        result = _systemctl(["stop", unit], dry_run=dry_run)
        if result.returncode != 0:
            raise RuntimeError(f"systemctl stop {unit} failed: {result.stderr.strip()}")
    return states


def _restore_active_units(states: dict[str, dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    restored: dict[str, Any] = {}
    for unit, state in states.items():
        if unit.endswith(".service"):
            restored[unit] = {"skipped": "one-shot service; timer will trigger the next run"}
            continue
        if state.get("active") != "active":
            restored[unit] = {"skipped": "was not active"}
            continue
        result = _systemctl(["start", unit], dry_run=dry_run)
        restored[unit] = {"returncode": result.returncode, "stderr": result.stderr.strip()}
    return restored


def _open_zarr(path: Path, *, group: str | None = None) -> xr.Dataset:
    try:
        return xr.open_zarr(path, group=group, chunks={}, consolidated=True)
    except Exception:
        return xr.open_zarr(path, group=group, chunks={}, consolidated=False)


def _time_validation(ds: xr.Dataset, cutoff: datetime, required_vars: tuple[str, ...]) -> dict[str, Any]:
    if "time" not in ds.coords:
        raise ValueError("missing time coordinate")
    if required_vars:
        missing = [name for name in required_vars if name not in ds.data_vars]
        if missing:
            raise ValueError(f"missing required variable(s): {', '.join(missing)}")
    if not ds.data_vars:
        raise ValueError("dataset has no data variables")

    times = np.asarray(ds["time"].values).astype("datetime64[ns]")
    times = times[~np.isnat(times)]
    if times.size == 0:
        raise ValueError("dataset has no valid time samples")
    cutoff64 = np.datetime64(pd.Timestamp(cutoff).tz_convert("UTC").tz_localize(None).to_datetime64(), "ns")
    min_time = times.min()
    max_time = times.max()
    if min_time < cutoff64:
        raise ValueError(f"minimum time {min_time} is before cutoff {cutoff64}")
    ints = times.astype("int64")
    diffs = np.diff(ints)
    if diffs.size and not np.all(diffs > 0):
        raise ValueError("time coordinate is not strictly sorted and unique")
    return {
        "min_time": str(min_time),
        "max_time": str(max_time),
        "sample_count": int(times.size),
        "dims": {key: int(value) for key, value in ds.sizes.items()},
        "data_vars": sorted(str(name) for name in ds.data_vars),
    }


def _consolidate(path: Path) -> None:
    try:
        zarr.consolidate_metadata(str(path))
    except Exception as exc:
        print(f"Could not consolidate {path}: {exc}")


def validate_regular_zarr(spec: ProductSpec, cutoff: datetime) -> dict[str, Any]:
    _consolidate(spec.temp)
    ds = _open_zarr(spec.temp)
    try:
        return _time_validation(ds, cutoff, spec.required_vars)
    finally:
        ds.close()


def validate_wxcam_zarr(spec: ProductSpec, cutoff: datetime) -> dict[str, Any]:
    _consolidate(spec.temp)
    root = zarr.open_group(str(spec.temp), mode="r")
    groups = sorted(root.group_keys())
    missing = [group for group in spec.wxcam_groups if group not in groups]
    if missing:
        raise ValueError(f"missing wxcam group(s): {', '.join(missing)}")
    group_results: dict[str, Any] = {}
    total = 0
    for group in spec.wxcam_groups:
        ds = _open_zarr(spec.temp, group=group)
        try:
            result = _time_validation(ds, cutoff, ("image",))
        finally:
            ds.close()
        group_results[group] = result
        total += int(result["sample_count"])
    return {
        "groups": group_results,
        "sample_count": total,
        "dims": {group: result["dims"] for group, result in group_results.items()},
    }


def swap_product(
    spec: ProductSpec,
    *,
    backup_root: Path,
    product_root: Path,
    stamp: str,
) -> dict[str, Any]:
    artifacts = [*spec.extra_artifacts, Artifact(spec.name, spec.target, spec.temp)]
    swapped: list[tuple[Artifact, Path | None]] = []
    installed: list[Artifact] = []
    try:
        results = []
        for artifact in artifacts:
            backup = _backup_path(artifact.target, backup_root, stamp, product_root)
            if artifact.target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(artifact.target), str(backup))
                swapped.append((artifact, backup))
            else:
                swapped.append((artifact, None))
            artifact.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(artifact.temp), str(artifact.target))
            installed.append(artifact)
            results.append(
                {
                    "name": artifact.name,
                    "target": str(artifact.target),
                    "backup": str(backup) if backup.exists() else None,
                }
            )
        return {"status": "swapped", "artifacts": results}
    except Exception as exc:
        for artifact in reversed(installed):
            if artifact.target.exists():
                failed_path = _failed_install_path(artifact.target, backup_root, stamp, product_root)
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(artifact.target), str(failed_path))
        for artifact, backup in reversed(swapped):
            if backup is not None and backup.exists():
                artifact.target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup), str(artifact.target))
        return {"status": "swap_failed", "error": str(exc)}


def build_product_specs(args: argparse.Namespace, stamp: str, cutoff_text: str) -> list[ProductSpec]:
    py = str(args.python)
    product_root = args.product_root
    raw_root = args.raw_root
    state_root = args.state_root

    targets = {
        "ceilometer": product_root / "cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr",
        "cloud_radar": product_root / "rpgfmcw94/cloud_radar.zarr",
        "hatpro": product_root / "hatprog5/hatpro.zarr",
        "vaisalamet": product_root / "vaisalamet/vaisalamet.zarr",
        "asfs_logger": product_root / "asfs_logger/asfs_logger.zarr",
        "asfs_fast_sonic": product_root / "asfs_fast_sonic/asfs_fast_sonic.zarr",
        "asfs_fast_gas": product_root / "asfs_fast_gas/asfs_fast_gas.zarr",
        "power": product_root / "power/power.zarr",
        "ops_monitor": product_root / "ops_monitor/ops_monitor.zarr",
        "wxcam": product_root / "wxcam/wxcam.zarr",
    }
    temps = {name: _tmp_store_path(target, stamp) for name, target in targets.items()}

    specs = [
        ProductSpec(
            "ceilometer",
            targets["ceilometer"],
            temps["ceilometer"],
            [
                py,
                "append_new_netcdf_to_zarr.py",
                "--input-dir",
                str(raw_root / "cl61"),
                "--zarr",
                str(temps["ceilometer"]),
                "--chunk-time",
                "30",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "cl61"),
            required_vars=("beta_att", "linear_depol_ratio"),
        ),
        ProductSpec(
            "cloud_radar",
            targets["cloud_radar"],
            temps["cloud_radar"],
            [
                py,
                "append_new_cloud_radar_to_zarr.py",
                "--root",
                str(raw_root / "rpgfmcw94"),
                "--zarr",
                str(temps["cloud_radar"]),
                "--chunk-time",
                "400",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "rpgfmcw94"),
            required_vars=("ZE_dBZ",),
        ),
        ProductSpec(
            "hatpro",
            targets["hatpro"],
            temps["hatpro"],
            [
                py,
                "hatpro_to_zarr.py",
                "--root",
                str(raw_root / "hatprog5"),
                "--zarr",
                str(temps["hatpro"]),
                "--chunk-time",
                "600",
                "--chunk-range",
                "48",
                "--rebuild",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "hatprog5"),
            required_vars=("T_PROF",),
        ),
        ProductSpec(
            "vaisalamet",
            targets["vaisalamet"],
            temps["vaisalamet"],
            [
                py,
                "append_new_vaisalamet_to_zarr.py",
                "--root",
                str(raw_root / "vaisalamet"),
                "--zarr",
                str(temps["vaisalamet"]),
                "--chunk-time",
                "1200",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "vaisalamet"),
        ),
        ProductSpec(
            "asfs_logger",
            targets["asfs_logger"],
            temps["asfs_logger"],
            [
                py,
                "append_new_asfs_logger_to_zarr.py",
                "--root",
                str(raw_root / "asfs"),
                "--zarr",
                str(temps["asfs_logger"]),
                "--chunk-time",
                "1200",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "asfs"),
        ),
        ProductSpec(
            "asfs_fast_sonic",
            targets["asfs_fast_sonic"],
            temps["asfs_fast_sonic"],
            [
                py,
                "append_new_asfs_fast_sonic_to_zarr.py",
                "--root",
                str(raw_root / "asfs"),
                "--zarr",
                str(temps["asfs_fast_sonic"]),
                "--chunk-time",
                "24000",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "asfs"),
        ),
        ProductSpec(
            "asfs_fast_gas",
            targets["asfs_fast_gas"],
            temps["asfs_fast_gas"],
            [
                py,
                "append_new_asfs_fast_gas_to_zarr.py",
                "--root",
                str(raw_root / "asfs"),
                "--zarr",
                str(temps["asfs_fast_gas"]),
                "--chunk-time",
                "24000",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "asfs"),
        ),
        ProductSpec(
            "power",
            targets["power"],
            temps["power"],
            [
                py,
                "append_new_power_to_zarr.py",
                "--root",
                str(raw_root / "power/level1"),
                "--zarr",
                str(temps["power"]),
                "--chunk-time",
                "1200",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "power/level1"),
        ),
        ProductSpec(
            "ops_monitor",
            targets["ops_monitor"],
            temps["ops_monitor"],
            [
                py,
                "append_new_ops_monitor_to_zarr.py",
                "--root",
                str(raw_root / "ops_monitor"),
                "--zarr",
                str(temps["ops_monitor"]),
                "--chunk-time",
                "720",
                "--lookback-days",
                "3",
                "--from-time",
                cutoff_text,
            ],
            str(raw_root / "ops_monitor"),
        ),
        ProductSpec(
            "wxcam",
            targets["wxcam"],
            temps["wxcam"],
            [
                py,
                "append_new_wxcam_to_zarr.py",
                "--catalog",
                str(product_root / "wxcam/wxcam_catalog.sqlite"),
                "--zarr",
                str(temps["wxcam"]),
                "--state",
                str(args.temp_root / f"wxcam-zarr-state-{stamp}.json"),
                "--batch-size",
                "4",
                "--rebuild-from-time",
                "--from-time",
                cutoff_text,
            ],
            str(product_root / "wxcam/wxcam_catalog.sqlite"),
            wxcam_groups=("fish_hdr", "pano_hdr"),
            extra_artifacts=[
                Artifact(
                    "wxcam_state",
                    state_root / "wxcam-zarr-state.json",
                    args.temp_root / f"wxcam-zarr-state-{stamp}.json",
                )
            ],
        ),
    ]
    return specs


def build_power_display_specs(args: argparse.Namespace, stamp: str, first_level: dict[str, ProductSpec]) -> list[ProductSpec]:
    product_root = args.product_root
    py = str(args.python)
    summary_target = product_root / "power/power_display_summary.zarr"
    energy_target = product_root / "power/power_display_energy.zarr"
    summary_temp = _tmp_store_path(summary_target, stamp)
    energy_temp = _tmp_store_path(energy_target, stamp)
    summary = ProductSpec(
        "power_display_summary",
        summary_target,
        summary_temp,
        [
            py,
            "generate_power_display_summary.py",
            "--power-zarr",
            str(first_level["power"].temp),
            "--asfs-logger-zarr",
            str(first_level["asfs_logger"].temp),
            "--output-zarr",
            str(summary_temp),
            "--energy-output-zarr",
            str(energy_temp),
        ],
        f"{first_level['power'].temp}; {first_level['asfs_logger'].temp}",
    )
    energy = ProductSpec(
        "power_display_energy",
        energy_target,
        energy_temp,
        [py, "generate_power_display_summary.py", "--energy-output-zarr", str(energy_temp)],
        str(first_level["power"].temp),
        skip_run=True,
    )
    return [summary, energy]


def quicklook_commands(py: str) -> list[list[str]]:
    return [
        [py, "plot_ceilometer_zarr_last24h.py"],
        [py, "generate_daily_quicklooks.py", "--force"],
        [py, "plot_cloud_radar_last24h.py"],
        [py, "generate_cloud_radar_quicklooks.py", "--force"],
        [py, "generate_hatpro_quicklooks.py", "--force"],
        [py, "plot_vaisalamet_last24h.py"],
        [py, "generate_vaisalamet_quicklooks.py", "--force"],
        [py, "plot_asfs_logger_last24h.py"],
        [py, "generate_asfs_logger_quicklooks.py", "--force"],
        [py, "plot_asfs_fast_sonic_last24h.py"],
        [py, "generate_asfs_fast_sonic_quicklooks.py", "--force"],
        [py, "plot_power_last24h.py"],
        [py, "generate_power_quicklooks.py", "--force"],
        [py, "generate_ops_monitor_quicklooks.py", "--force"],
    ]


def _build_and_validate(
    specs: list[ProductSpec],
    *,
    cutoff: datetime,
    manifest: dict[str, Any],
    backup_root: Path,
    stamp: str,
    dry_run: bool,
) -> dict[str, ProductSpec]:
    valid: dict[str, ProductSpec] = {}
    for spec in specs:
        print(f"[build] {spec.name}")
        if not spec.skip_run:
            _remove_temp(spec.temp)
            for artifact in spec.extra_artifacts:
                _remove_temp(artifact.temp)
        log_path = backup_root / stamp / "logs" / f"{spec.name}.log"
        product_record = {
            "target": spec.target,
            "temp": spec.temp,
            "source": spec.source,
            "command": spec.command,
        }
        manifest["products"][spec.name] = product_record
        if spec.skip_run:
            run = {
                "command": spec.command,
                "cwd": str(APP_DIR),
                "log_path": str(log_path),
                "returncode": 0,
                "skipped": "created by paired product command",
            }
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Created by paired product command; validating existing temp output.\n", encoding="utf-8")
        else:
            run = _run_command(spec.command, cwd=APP_DIR, log_path=log_path, dry_run=dry_run)
        product_record["run"] = run
        if run.get("returncode") != 0:
            product_record["status"] = "build_failed"
            continue
        if dry_run:
            product_record["status"] = "dry_run"
            valid[spec.name] = spec
            continue
        try:
            validation = validate_wxcam_zarr(spec, cutoff) if spec.wxcam_groups else validate_regular_zarr(spec, cutoff)
        except Exception as exc:
            product_record["status"] = "validation_failed"
            product_record["validation_error"] = str(exc)
            continue
        product_record["status"] = "validated"
        product_record["validation"] = validation
        valid[spec.name] = spec
    return valid


def run_quicklooks(args: argparse.Namespace, manifest: dict[str, Any], stamp: str) -> bool:
    if args.skip_quicklooks:
        manifest["quicklooks"] = [{"status": "skipped"}]
        return True
    entries = []
    ok = True
    for idx, command in enumerate(quicklook_commands(str(args.python)), start=1):
        name = Path(command[1]).stem
        log_path = args.backup_root / stamp / "logs" / f"quicklook-{idx:02d}-{name}.log"
        print(f"[quicklook] {name}")
        result = _run_command(command, cwd=APP_DIR, log_path=log_path, dry_run=args.dry_run)
        if result.get("returncode") != 0:
            ok = False
        entries.append(result)
    manifest["quicklooks"] = entries
    return ok


def run_dashboard_smoke(args: argparse.Namespace, manifest: dict[str, Any], stamp: str) -> dict[str, Any]:
    if args.skip_dashboard_smoke:
        manifest["dashboard_smoke"] = {"status": "skipped"}
        return {"returncode": 0, "status": "skipped"}
    command = [str(args.python), "-c", "import app; print('dashboard import ok')"]
    log_path = args.backup_root / stamp / "logs" / "dashboard-import-smoke.log"
    print("[smoke] dashboard import")
    result = _run_command(command, cwd=APP_DIR, log_path=log_path, dry_run=args.dry_run)
    manifest["dashboard_smoke"] = result
    return result


def run_perf_summary(args: argparse.Namespace, manifest: dict[str, Any], stamp: str) -> None:
    if args.skip_perf_summary:
        manifest["performance_summary"] = {"status": "skipped"}
        return
    command = [str(args.python), "summarize_dashboard_perf.py", "--hours", "2", "--limit", "10"]
    log_path = args.backup_root / stamp / "logs" / "dashboard-performance-summary.log"
    print("[perf] summarize dashboard log")
    manifest["performance_summary"] = _run_command(command, cwd=APP_DIR, log_path=log_path, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanly rebuild Aurora dashboard product Zarrs from a UTC cutoff.")
    parser.add_argument("--from-time", required=True, help="UTC ISO cutoff, e.g. 2026-07-04T00:00:00Z")
    parser.add_argument("--product-root", type=Path, default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_PRODUCT_ROOT / ".zarr-rebuild-backups")
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_PRODUCT_ROOT / ".zarr-rebuild-temp")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/logs but do not run commands or move stores.")
    parser.add_argument("--skip-timer-control", action="store_true")
    parser.add_argument("--skip-dashboard-restart", action="store_true")
    parser.add_argument("--skip-quicklooks", action="store_true")
    parser.add_argument("--skip-dashboard-smoke", action="store_true")
    parser.add_argument("--skip-perf-summary", action="store_true")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp stores after failed validations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cutoff = parse_from_time(args.from_time)
    if cutoff is None:
        raise SystemExit("--from-time is required")
    cutoff_text = _iso_z(cutoff)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.backup_root.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.backup_root / stamp / "manifest.json"
    manifest: dict[str, Any] = {
        "started_at": _iso_z(datetime.now(timezone.utc)),
        "from_time": cutoff_text,
        "product_root": args.product_root,
        "raw_root": args.raw_root,
        "backup_root": args.backup_root,
        "temp_root": args.temp_root,
        "dry_run": args.dry_run,
        "products": {},
        "service_control": {},
    }

    timer_states: dict[str, dict[str, Any]] = {}
    try:
        if not args.skip_timer_control:
            print("[services] stopping processing timers and services")
            timer_states = _stop_units(PROCESSING_TIMERS + PROCESSING_SERVICES, dry_run=args.dry_run)
            manifest["service_control"]["stopped"] = timer_states

        first_level = build_product_specs(args, stamp, cutoff_text)
        valid = _build_and_validate(
            first_level,
            cutoff=cutoff,
            manifest=manifest,
            backup_root=args.backup_root,
            stamp=stamp,
            dry_run=args.dry_run,
        )

        if {"power", "asfs_logger"}.issubset(valid):
            display_specs = build_power_display_specs(args, stamp, {name: valid[name] for name in ("power", "asfs_logger")})
            display_valid = _build_and_validate(
                display_specs,
                cutoff=cutoff,
                manifest=manifest,
                backup_root=args.backup_root,
                stamp=stamp,
                dry_run=args.dry_run,
            )
            valid.update(display_valid)
        else:
            manifest["products"]["power_display_summary"] = {"status": "skipped", "reason": "power or asfs_logger did not validate"}
            manifest["products"]["power_display_energy"] = {"status": "skipped", "reason": "power or asfs_logger did not validate"}

        if args.dry_run:
            manifest["status"] = "dry_run"
            _write_manifest(manifest_path, manifest)
            print(f"Dry run manifest written to {manifest_path}")
            return 0

        if not args.skip_dashboard_restart:
            print("[services] stopping dashboard for swap")
            result = _systemctl(["stop", DASHBOARD_SERVICE], dry_run=False)
            manifest["service_control"]["dashboard_stop"] = {"returncode": result.returncode, "stderr": result.stderr.strip()}
            if result.returncode != 0:
                raise RuntimeError(f"systemctl stop {DASHBOARD_SERVICE} failed: {result.stderr.strip()}")

        print("[swap] installing validated products")
        for name, spec in valid.items():
            swap_result = swap_product(spec, backup_root=args.backup_root, product_root=args.product_root, stamp=stamp)
            manifest["products"][name]["swap"] = swap_result
            if swap_result.get("status") != "swapped":
                manifest["products"][name]["status"] = "swap_failed"
            else:
                manifest["products"][name]["status"] = "active"

        quicklooks_ok = run_quicklooks(args, manifest, stamp)
        smoke_result = run_dashboard_smoke(args, manifest, stamp)

        if not args.skip_dashboard_restart:
            print("[services] restarting dashboard")
            result = _systemctl(["restart", DASHBOARD_SERVICE], dry_run=False)
            manifest["service_control"]["dashboard_restart"] = {"returncode": result.returncode, "stderr": result.stderr.strip()}
            if result.returncode != 0:
                raise RuntimeError(f"systemctl restart {DASHBOARD_SERVICE} failed: {result.stderr.strip()}")

        run_perf_summary(args, manifest, stamp)
        product_failures = {
            name: record.get("status")
            for name, record in manifest["products"].items()
            if record.get("status") != "active"
        }
        smoke_ok = smoke_result.get("returncode") == 0
        if product_failures or not quicklooks_ok or not smoke_ok:
            manifest["status"] = "completed_with_failures"
            manifest["product_failures"] = product_failures
            manifest["quicklooks_ok"] = quicklooks_ok
            manifest["dashboard_smoke_ok"] = smoke_ok
            return 1
        manifest["status"] = "complete"
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.skip_timer_control and timer_states:
            print("[services] restoring timers/services that were active")
            manifest["service_control"]["restored"] = _restore_active_units(timer_states, dry_run=args.dry_run)
        if not args.keep_temp and not args.dry_run:
            for record in manifest.get("products", {}).values():
                temp_value = record.get("temp")
                if temp_value:
                    _remove_temp(Path(temp_value))
        manifest["finished_at"] = _iso_z(datetime.now(timezone.utc))
        _write_manifest(manifest_path, manifest)
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
