# Operations Dashboard

Operations Dashboard is the live operational status tab for the whole Aurora
stack.

## What it shows

- source-host probe health
- local and remote storage pressure
- Aurora Power Supply battery voltage from `DCInverterVolts`, scored green
  above `52 V`, amber from `50-52 V`, and red below `50 V`
- Aurora Power Supply battery state of charge from `BatterySOC`, scored green
  at or above `50 %`, amber from `25-50 %`, and red below `25 %`
- Aurora Power Supply internal temperature from `InternalTemperature`,
  scored green below `35 C`, amber from `35-40 C`, and red at `40 C` or above
- source sync and processing health
- HATPRO source, local mirror, GWS mirror, Zarr-build, and quicklook health
  alongside the other science streams
- dashboard performance-log freshness, including whether browser activity is
  still being written to `/data/aurora/products/dashboard/dashboard_perf.jsonl`
- dashboard HTTP endpoint health and response time
- dashboard and infrastructure git cleanliness and local ahead/behind counts
- recent dashboard render-performance statistics, including p50, p95, slowest
  timed event, and live-session counts
- GWS transfer status
- mirror verification and prune-readiness indicators
- per-stream archive state, including WXcam backfill progress

The storage cards are intentionally broken out as:

- CL61 root and CL61 data
- ASS data and ASS root
- APS data and APS root
- AURORA Cloud product and AURORA Cloud root
- JASMIN GWS

Each card subtitle uses the resolved `pwd -P` path that was actually probed for
filesystem usage.

## Display model

This tab reads the latest operations snapshot directly rather than waiting for
an archived quicklook to exist. That means a fresh deployment can show the live
Operations tab before the archived operations PNGs have accumulated enough
samples to plot. Archive traffic lights are based on settled mirror health, so
a stream stays green when the verified GWS archive has no missing or mismatched
files even if the newest just-arrived source file has not yet landed in the
next transfer batch.

## Archived products

The archived operations products live under:

- `/data/aurora/products/quicklooks/ops_monitor`
- `/data/aurora/products/ops_monitor/health`

These include:

- summary quicklooks
- `HK_Operations`
- observe-only health JSON and daily Markdown reports

## Email Alerts

Operations alert email is handled by `send_ops_alerts.py`, normally from
`aurora-ops-monitor-alerts.timer`. It evaluates the same latest snapshot used by
the dashboard and emails `gamb2le@ncas.ac.uk` for storage pressure at `80 %`,
battery SOC at or below `20 %`, APS internal temperature at or above `45 C`,
battery voltage below `50 V`, and stream-health problems that persist for
`3 h`.

The service keeps state under `/data/aurora/products/ops_monitor/alerts` so it
can send initial, repeat, and recovery messages without spamming every timer
tick. The deployed delivery path is intended to be `mailx` backed by `msmtp` or
another sendmail-compatible outbound relay.

Detailed product documentation:

- [Operations products](../data-products/operations-products.md)
