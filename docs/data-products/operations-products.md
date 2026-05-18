# Operations Products

Operations monitoring has its own raw snapshots, monitoring Zarr, and archived
quicklook products.

## Raw snapshots

- rolling latest snapshot:
  `/project/aurora/raw/ops_monitor/latest.json`
- time-stamped JSONL snapshots:
  `/project/aurora/raw/ops_monitor/ops_monitor_YYYYMMDD.jsonl`
- observe-only health assessment:
  `/data/aurora/products/ops_monitor/health/latest_health.json`
- human-readable daily health report:
  `/data/aurora/products/ops_monitor/health/health_report_YYYYMMDD.md`

These snapshots capture:

- snapshot collection time as both `time_utc` and `snapshot_time_utc`
- source-host disk usage and probe reachability
- resolved `pwd -P` paths for the filesystem locations that were actually probed
- per-stream source recency, with a red operational state if a source has not
  produced data in the last 1.5 hours
- Aurora Power Supply battery voltage from the latest `DCInverterVolts` sample
  in the power Zarr, scored green above `52 V`, amber from `50-52 V`, and red
  below `50 V`
- Aurora Power Supply battery state of charge from the latest `BatterySOC`
  sample in the power Zarr, scored green at or above `50 %`, amber from
  `25-50 %`, and red below `25 %`
- Aurora Power Supply internal temperature from the latest
  `InternalTemperature` sample in the power Zarr, scored green below `40 C`,
  amber from `40-45 C`, and red at `45 C` or above
- local `/project`, `/data`, and `/` filesystem usage
- GWS usage and reachability
- per-stream mirror coverage, lag, and mismatch counts
- prune-gate and product-gate summaries
- systemd health for source sync, processing, and transfer units
- dashboard HTTP health and response time
- dashboard and infrastructure git branch, commit, dirty state, and local
  ahead/behind counts

The health assessment is deliberately observe-only. It summarizes the raw
snapshot into green/amber/red checks, but it does not restart services, delete
data, rebuild stores, or modify code.

## Operations Zarr

Path:

- `/data/aurora/products/ops_monitor/ops_monitor.zarr`

When checked on `2026-05-18`:

- dimension: `time`
- shape: `time=2431`
- data variables: `374`
- time coverage when checked: `2026-05-09 16:01:00.181057` to
  `2026-05-18 18:32:04.299819`
- useful attrs:
  - `instrument = "ops-monitor"`
  - `source = "ops_monitor_YYYYMMDD.jsonl"`
  - `title = "Aurora operations monitoring"`

The appender can rebuild automatically if the collector schema expands.

## Quicklooks

Archived operations products live under:

- `/data/aurora/products/quicklooks/ops_monitor`

They currently include:

- summary operations quicklooks
- `HK_Operations`

`HK_Operations` is intentionally curated diagnostics rather than an exhaustive
plot of every numeric monitoring field.

## Live dashboard relationship

The top-level **Operations Dashboard** tab reads the latest snapshot directly,
so the live status view can be useful even before enough archived samples exist
to generate meaningful historical PNGs.

## Phase 1 sentinel outputs

The collector now acts as the Phase 1 operations sentinel:

- every run writes the raw JSONL record and `latest.json`
- every run writes `latest_health.json`, a compact machine-readable health
  assessment with `overall_level`, check counts, and current observations
- every run refreshes `latest_report.md` and the date-stamped daily report
- the report explicitly states that no automated healing actions were taken

## Email Alerts

`send_ops_alerts.py` evaluates the latest operations snapshot and sends email
alerts to `gamb2le@ncas.ac.uk` when operational thresholds are crossed.

Alert rules:

- any monitored storage filesystem reaches `80 %` used
- Aurora Power Supply battery state of charge is at or below `20 %`
- Aurora Power Supply internal temperature is at or above `45 C`
- Aurora Power Supply battery voltage from `DCInverterVolts` is below `50 V`
- any stream source is stale for at least `3 h`
- any stream service-health component remains off/unhealthy for at least `3 h`

Alert state and logs:

- state: `/data/aurora/products/ops_monitor/alerts/state.json`
- event log: `/data/aurora/products/ops_monitor/alerts/alerts.jsonl`

The state file records active alerts, first-seen time, last-seen time, and last
sent time so the system does not send an email every five minutes. Alerts send
when they first become active after any hold period, repeat every `12 h` while
still active, and send a recovery email after a previously emailed alert clears.

The recommended delivery stack for this headless VM is:

- `bsd-mailx` as the script-facing command-line mail interface
- `msmtp` / `msmtp-mta` as the lightweight outbound SMTP delivery layer

In other words, dashboard scripts use `mailx` because it is a simple command
line interface, while `msmtp` does the actual SMTP delivery through a relay. A
full Sendmail MTA is intentionally not required unless the VM later needs to run
a real mail server.

The script prefers `mailx` when it is available, then a sendmail-compatible
interface, then direct SMTP from `OPS_ALERT_SMTP_*` environment variables. Set
`OPS_ALERT_TRANSPORT` to `mailx`, `sendmail`, `smtp`, or `auto` to force a
specific path. The VM still needs a real SMTP relay configured for `msmtp`
before real email can be delivered through the recommended `mailx` path. When
`sendmail` is the `msmtp` wrapper, the alert script only treats mail delivery as
configured if an msmtp config exists at `/home/aurora/.msmtprc`, `/etc/msmtprc`,
`/etc/msmtp/msmtprc`, or a path named by `OPS_ALERT_MSMTP_CONFIG`.

Useful checks:

```bash
python send_ops_alerts.py --dry-run
python send_ops_alerts.py --test-email gamb2le@ncas.ac.uk
```
