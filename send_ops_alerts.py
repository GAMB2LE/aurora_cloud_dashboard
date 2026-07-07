#!/usr/bin/env python3
"""Evaluate Aurora operations snapshots and send threshold email alerts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import json
import math
import os
from pathlib import Path
import shutil
import smtplib
import socket
import subprocess
import sys
from typing import Any


SNAPSHOT_DEFAULT = Path("/project/aurora/raw/ops_monitor/latest.json")
STATE_DEFAULT = Path("/data/aurora/products/ops_monitor/alerts/state.json")
LOG_DEFAULT = Path("/data/aurora/products/ops_monitor/alerts/alerts.jsonl")
RECIPIENT_DEFAULT = "gamb2le@ncas.ac.uk"
FROM_DEFAULT = f"aurora-ops@{socket.gethostname() or 'localhost'}"
DASHBOARD_URL_DEFAULT = "https://data.gamb2le.co.uk/app?tab=operations"

STORAGE_THRESHOLD_PCT = 80.0
BATTERY_SOC_THRESHOLD_PCT = 20.0
INTERNAL_TEMP_THRESHOLD_C = 45.0
BATTERY_VOLTAGE_THRESHOLD_V = 50.0
STREAM_HOLD_MINUTES = 180.0
REPEAT_AFTER_HOURS = 12.0

STREAM_PREFIXES = {
    "CL61": "cl61",
    "Cloud Radar": "radar",
    "Scanning Microwave Radiometer": "hatpro",
    "Meteorology": "vaisalamet",
    "Radiation": "asfs_logger",
    "ASFS Fast Sonic": "asfs_fast_sonic",
    "Aurora Power Supply": "power",
    "ASS PDU": "pdu",
    "WXcam": "wxcam",
}

STREAM_SERVICE_KEYS = {
    "CL61": ("cl61_source_sync_service_healthy_state", "ceilometer_append_service_healthy_state", "ceilometer_quicklooks_service_healthy_state"),
    "Cloud Radar": ("radar_source_sync_service_healthy_state", "radar_append_service_healthy_state", "radar_quicklooks_service_healthy_state"),
    "Scanning Microwave Radiometer": ("hatpro_source_sync_service_healthy_state", "hatpro_append_service_healthy_state", "hatpro_quicklooks_service_healthy_state"),
    "Meteorology": ("vaisalamet_source_sync_service_healthy_state", "vaisalamet_append_service_healthy_state", "vaisalamet_quicklooks_service_healthy_state"),
    "Radiation": ("asfs_logger_source_sync_service_healthy_state", "asfs_logger_append_service_healthy_state", "asfs_logger_quicklooks_service_healthy_state"),
    "ASFS Fast Sonic": (
        "asfs_fast_sonic_source_sync_service_healthy_state",
        "asfs_fast_sonic_append_service_healthy_state",
        "asfs_fast_sonic_quicklooks_service_healthy_state",
    ),
    "Aurora Power Supply": ("power_source_sync_service_healthy_state", "power_append_service_healthy_state", "power_quicklooks_service_healthy_state"),
    "ASS PDU": ("pdu_source_sync_service_healthy_state", "pdu_append_service_healthy_state"),
    "WXcam": (
        "wxcam_source_sync_service_healthy_state",
        "wxcam_catalog_service_healthy_state",
        "wxcam_append_service_healthy_state",
        "wxcam_daily_videos_service_healthy_state",
    ),
}

STORAGE_LABELS = {
    "host_celine_source": "CL61 root disk",
    "host_celine_data": "CL61 data disk",
    "host_ass_root": "ASS root disk",
    "host_ass_data": "ASS data disk",
    "host_aps_root": "APS root disk",
    "host_aps_data": "APS data disk",
    "aurora_project": "Aurora raw mirror disk",
    "aurora_data": "AURORA Cloud product disk",
    "aurora_root": "AURORA Cloud root disk",
    "gws_storage": "JASMIN GWS",
}


@dataclass(frozen=True)
class AlertRule:
    id: str
    title: str
    message: str
    value: float | str
    threshold: str
    hold_minutes: float = 0.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(stamp: datetime | None) -> str | None:
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if stamp else None


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True))
        handle.write("\n")


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _bool_state(value: Any) -> bool | None:
    number = _float(value)
    if number is None:
        return None
    return bool(round(number))


def _fmt_value(value: float | str, unit: str = "") -> str:
    if isinstance(value, str):
        return value
    if abs(value) >= 100:
        return f"{value:.0f}{unit}"
    if abs(value) >= 10:
        return f"{value:.1f}{unit}"
    return f"{value:.2f}{unit}"


def _human_metric_key(key: str) -> str:
    return key.replace("_", " ").replace(" pct", "").title()


def _storage_prefix(key: str) -> str:
    return key[: -len("_used_pct")]


def _service_label(key: str) -> str:
    label = key
    for suffix in ("_service_healthy_state", "_timer_active_state", "_source_recent_state"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    return label.replace("_", " ")


def evaluate_alerts(snapshot: dict[str, Any]) -> list[AlertRule]:
    alerts: list[AlertRule] = []

    for key, raw_value in sorted(snapshot.items()):
        if not key.endswith("_used_pct") or key.endswith("_inode_used_pct"):
            continue
        value = _float(raw_value)
        if value is None or value < STORAGE_THRESHOLD_PCT:
            continue
        prefix = _storage_prefix(key)
        label = STORAGE_LABELS.get(prefix, _human_metric_key(prefix))
        path = snapshot.get(f"{prefix}_resolved_path", "")
        free_gb = _float(snapshot.get(f"{prefix}_free_gb"))
        free_detail = f", free={_fmt_value(free_gb, ' GB')}" if free_gb is not None else ""
        alerts.append(
            AlertRule(
                id=f"storage:{prefix}",
                title=f"{label} storage at {_fmt_value(value, '%')}",
                message=f"{label} is using {_fmt_value(value, '%')} of capacity{free_detail}. Path: {path or 'unknown'}.",
                value=value,
                threshold=f">= {STORAGE_THRESHOLD_PCT:.0f}%",
            )
        )

    soc = _float(snapshot.get("aps_battery_soc_pct"))
    if soc is not None and soc <= BATTERY_SOC_THRESHOLD_PCT:
        alerts.append(
            AlertRule(
                id="power:battery_soc",
                title=f"Battery SOC at {_fmt_value(soc, '%')}",
                message=f"Aurora Power Supply battery state of charge is {_fmt_value(soc, '%')}.",
                value=soc,
                threshold=f"<= {BATTERY_SOC_THRESHOLD_PCT:.0f}%",
            )
        )

    temp = _float(snapshot.get("aps_internal_temp_c"))
    if temp is not None and temp >= INTERNAL_TEMP_THRESHOLD_C:
        alerts.append(
            AlertRule(
                id="power:internal_temp",
                title=f"APS internal temperature at {_fmt_value(temp, ' C')}",
                message=f"Aurora Power Supply internal temperature is {_fmt_value(temp, ' C')}.",
                value=temp,
                threshold=f">= {INTERNAL_TEMP_THRESHOLD_C:.0f} C",
            )
        )

    volts = _float(snapshot.get("aps_battery_voltage_v"))
    if volts is not None and volts < BATTERY_VOLTAGE_THRESHOLD_V:
        alerts.append(
            AlertRule(
                id="power:battery_voltage",
                title=f"Battery voltage at {_fmt_value(volts, ' V')}",
                message=f"Aurora Power Supply battery voltage from DC inverter voltage is {_fmt_value(volts, ' V')}.",
                value=volts,
                threshold=f"< {BATTERY_VOLTAGE_THRESHOLD_V:.0f} V",
            )
        )

    for stream_label, prefix in STREAM_PREFIXES.items():
        source_age = _float(snapshot.get(f"{prefix}_source_age_min"))
        if source_age is not None and source_age >= STREAM_HOLD_MINUTES:
            alerts.append(
                AlertRule(
                    id=f"stream:{prefix}:source_stale",
                    title=f"{stream_label} source stale for {_fmt_value(source_age, ' min')}",
                    message=f"{stream_label} source data age is {_fmt_value(source_age, ' min')}.",
                    value=source_age,
                    threshold=f">= {STREAM_HOLD_MINUTES:.0f} min",
                )
            )

        for key in STREAM_SERVICE_KEYS.get(stream_label, ()):
            state = _bool_state(snapshot.get(key))
            if state is False:
                alerts.append(
                    AlertRule(
                        id=f"stream:{prefix}:{key}",
                        title=f"{stream_label} {_service_label(key)} is unhealthy",
                        message=f"{stream_label} stream health component `{key}` is reporting off/unhealthy.",
                        value="off",
                        threshold=f"continuous for {STREAM_HOLD_MINUTES:.0f} min",
                        hold_minutes=STREAM_HOLD_MINUTES,
                    )
                )

    return alerts


def _recipient_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _msmtp_config_present() -> bool:
    if os.environ.get("OPS_ALERT_MSMTP_CONFIG"):
        return Path(os.environ["OPS_ALERT_MSMTP_CONFIG"]).expanduser().exists()
    candidates = (
        Path.home() / ".msmtprc",
        Path("/etc/msmtprc"),
        Path("/etc/msmtp/msmtprc"),
    )
    return any(path.exists() for path in candidates)


def _sendmail_is_msmtp(sendmail: str | None) -> bool:
    if not sendmail:
        return False
    try:
        resolved = Path(sendmail).resolve()
    except OSError:
        resolved = Path(sendmail)
    return "msmtp" in {resolved.name, Path(sendmail).name}


def _sendmail_ready(sendmail: str | None) -> bool:
    if not sendmail:
        return False
    if _sendmail_is_msmtp(sendmail):
        return _msmtp_config_present() or _truthy_env("OPS_ALERT_ASSUME_SENDMAIL_CONFIGURED")
    return True


def _mailx_ready(mailx: str | None, sendmail: str | None) -> bool:
    if not mailx:
        return False
    if _truthy_env("OPS_ALERT_ASSUME_MAILX_CONFIGURED"):
        return True
    if _sendmail_is_msmtp(sendmail or shutil.which("sendmail")):
        return _msmtp_config_present()
    return True


def _transport_configured() -> bool:
    transport = os.environ.get("OPS_ALERT_TRANSPORT", "auto").lower()
    mailx = os.environ.get("OPS_ALERT_MAILX") or shutil.which("mailx") or shutil.which("mail")
    sendmail = os.environ.get("OPS_ALERT_SENDMAIL") or shutil.which("sendmail")
    smtp = os.environ.get("OPS_ALERT_SMTP_HOST")
    if transport == "mailx":
        return _mailx_ready(mailx, sendmail)
    if transport == "sendmail":
        return _sendmail_ready(sendmail)
    if transport == "smtp":
        return bool(smtp)
    return _mailx_ready(mailx, sendmail) or _sendmail_ready(sendmail) or bool(smtp)


def _send_email(subject: str, body: str, recipients: list[str]) -> str:
    sender = os.environ.get("OPS_ALERT_FROM", FROM_DEFAULT)
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    transport = os.environ.get("OPS_ALERT_TRANSPORT", "auto").lower()
    mailx = os.environ.get("OPS_ALERT_MAILX") or shutil.which("mailx") or shutil.which("mail")
    sendmail = os.environ.get("OPS_ALERT_SENDMAIL") or shutil.which("sendmail")
    if transport in {"auto", "mailx"} and _mailx_ready(mailx, sendmail):
        cmd = [mailx, "-s", subject]
        if os.environ.get("OPS_ALERT_FROM"):
            cmd.extend(["-r", os.environ["OPS_ALERT_FROM"]])
        subprocess.run([*cmd, *recipients], input=body, text=True, check=True)
        return f"mailx:{mailx}"

    if transport in {"auto", "sendmail"} and _sendmail_ready(sendmail):
        subprocess.run([sendmail, "-t"], input=msg.as_string(), text=True, check=True)
        return f"sendmail:{sendmail}"

    host = os.environ.get("OPS_ALERT_SMTP_HOST")
    if not host:
        raise RuntimeError("No mail transport configured. Set OPS_ALERT_SMTP_HOST or install/configure sendmail.")
    port = int(os.environ.get("OPS_ALERT_SMTP_PORT", "465" if os.environ.get("OPS_ALERT_SMTP_SSL", "").lower() in {"1", "true", "yes"} else "587"))
    timeout = float(os.environ.get("OPS_ALERT_SMTP_TIMEOUT", "20"))
    username = os.environ.get("OPS_ALERT_SMTP_USER")
    password = os.environ.get("OPS_ALERT_SMTP_PASSWORD")
    use_ssl = os.environ.get("OPS_ALERT_SMTP_SSL", "").lower() in {"1", "true", "yes"}
    use_starttls = os.environ.get("OPS_ALERT_SMTP_STARTTLS", "true").lower() in {"1", "true", "yes"}

    smtp_class = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    with smtp_class(host, port, timeout=timeout) as smtp:
        if not use_ssl and use_starttls:
            smtp.starttls()
        if username:
            smtp.login(username, password or "")
        smtp.send_message(msg)
    return f"smtp:{host}:{port}"


def _format_alert_list(alerts: list[AlertRule]) -> str:
    lines: list[str] = []
    for alert in alerts:
        lines.extend(
            [
                f"- {alert.title}",
                f"  Condition: {alert.threshold}",
                f"  Details: {alert.message}",
            ]
        )
    return "\n".join(lines)


def _build_email(kind: str, alerts: list[AlertRule], snapshot: dict[str, Any], now: datetime) -> tuple[str, str]:
    dashboard_url = os.environ.get("OPS_ALERT_DASHBOARD_URL", DASHBOARD_URL_DEFAULT)
    snapshot_time = snapshot.get("time_utc", "unknown")
    prefix = "RECOVERED" if kind == "recovery" else "ALERT"
    subject = f"AURORA Ops {prefix}: {len(alerts)} condition{'s' if len(alerts) != 1 else ''}"
    body = "\n".join(
        [
            f"AURORA operations {kind}",
            "",
            f"Evaluation time: {_iso(now)}",
            f"Snapshot time: {snapshot_time}",
            f"Dashboard: {dashboard_url}",
            "",
            _format_alert_list(alerts),
            "",
            "This message was generated by send_ops_alerts.py from the operations snapshot.",
        ]
    )
    return subject, body


def process_alerts(
    snapshot: dict[str, Any],
    *,
    state_path: Path,
    log_path: Path,
    dry_run: bool = False,
    repeat_after: timedelta = timedelta(hours=REPEAT_AFTER_HOURS),
) -> dict[str, Any]:
    now = _parse_time(snapshot.get("time_utc")) or _utc_now()
    active_rules = {rule.id: rule for rule in evaluate_alerts(snapshot)}
    state = _load_json(state_path, {"alerts": {}})
    state_alerts = state.setdefault("alerts", {})
    recipients = _recipient_list(os.environ.get("OPS_ALERT_RECIPIENTS", RECIPIENT_DEFAULT))
    transport_ready = _transport_configured()

    new_or_repeat: list[AlertRule] = []
    recovered: list[AlertRule] = []
    active_ids = set(active_rules)

    for alert_id, rule in active_rules.items():
        entry = state_alerts.setdefault(alert_id, {})
        first_seen = _parse_time(entry.get("first_seen_utc")) or now
        last_sent = _parse_time(entry.get("last_sent_utc"))
        entry.update(
            {
                "active": True,
                "title": rule.title,
                "threshold": rule.threshold,
                "last_seen_utc": _iso(now),
                "last_value": rule.value,
            }
        )
        entry.setdefault("first_seen_utc", _iso(first_seen))
        hold_satisfied = now - first_seen >= timedelta(minutes=rule.hold_minutes)
        last_attempt = _parse_time(entry.get("last_send_attempt_utc"))
        last_contact = last_sent or last_attempt
        repeat_due = last_contact is not None and now - last_contact >= repeat_after
        if hold_satisfied and (last_contact is None or repeat_due):
            new_or_repeat.append(rule)

    for alert_id, entry in list(state_alerts.items()):
        if alert_id in active_ids or not entry.get("active"):
            continue
        if entry.get("last_sent_utc"):
            recovered.append(
                AlertRule(
                    id=alert_id,
                    title=f"{entry.get('title', alert_id)} recovered",
                    message=f"Condition cleared. Previous threshold: {entry.get('threshold', 'unknown')}.",
                    value=entry.get("last_value", "recovered"),
                    threshold="recovered",
                )
            )
        entry["active"] = False
        entry["recovered_utc"] = _iso(now)

    sent_events: list[dict[str, Any]] = []
    had_send_failure = False
    for kind, alerts in (("alert", new_or_repeat), ("recovery", recovered)):
        if not alerts:
            continue
        subject, body = _build_email(kind, alerts, snapshot, now)
        event = {
            "time_utc": _iso(now),
            "kind": kind,
            "alert_ids": [alert.id for alert in alerts],
            "recipient_count": len(recipients),
            "dry_run": dry_run,
        }
        if dry_run:
            event["status"] = "dry_run"
            print(f"\n--- {subject} ---\n{body}\n")
        elif not transport_ready:
            event["status"] = "no_transport"
            print("No mail transport configured; alert state updated but no email sent.", file=sys.stderr)
        else:
            try:
                transport = _send_email(subject, body, recipients)
            except Exception as exc:
                event["status"] = "send_failed"
                event["error"] = str(exc)
                had_send_failure = True
            else:
                event["status"] = "sent"
                event["transport"] = transport
        sent_events.append(event)
        _append_log(log_path, event)
        if event["status"] == "sent" and kind == "alert":
            for alert in alerts:
                state_alerts[alert.id]["last_sent_utc"] = _iso(now)
        elif event["status"] in {"no_transport", "send_failed"} and kind == "alert":
            for alert in alerts:
                state_alerts[alert.id]["last_send_attempt_utc"] = _iso(now)

    state["updated_utc"] = _iso(now)
    _save_json(state_path, state)
    result = {
        "active_alerts": len(active_rules),
        "emails_or_events": len(sent_events),
        "transport_configured": transport_ready,
        "events": sent_events,
    }
    if had_send_failure:
        result["send_failure"] = True
    return result


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Aurora operations threshold alerts.")
    parser.add_argument("--env-file", type=Path, default=Path(os.environ.get("AURORA_DASHBOARD_ENV_FILE", "/etc/aurora-dashboard.env")))
    parser.add_argument("--snapshot", type=Path, default=Path(os.environ.get("OPS_ALERT_SNAPSHOT", SNAPSHOT_DEFAULT)))
    parser.add_argument("--state", type=Path, default=Path(os.environ.get("OPS_ALERT_STATE", STATE_DEFAULT)))
    parser.add_argument("--log", type=Path, default=Path(os.environ.get("OPS_ALERT_LOG", LOG_DEFAULT)))
    parser.add_argument("--dry-run", action="store_true", help="Evaluate alerts and print emails without sending or marking sent.")
    parser.add_argument("--test-email", metavar="RECIPIENT", help="Send a one-off transport test email to RECIPIENT.")
    args = parser.parse_args()

    _load_env_file(args.env_file)

    if args.test_email:
        subject = "AURORA Ops Alert test"
        body = f"Test email from send_ops_alerts.py at {_iso(_utc_now())}."
        transport = _send_email(subject, body, _recipient_list(args.test_email))
        print(f"Sent test email via {transport}")
        return

    snapshot = _load_json(args.snapshot, None)
    if not isinstance(snapshot, dict):
        raise SystemExit(f"No readable operations snapshot at {args.snapshot}")

    result = process_alerts(snapshot, state_path=args.state, log_path=args.log, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("send_failure"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
