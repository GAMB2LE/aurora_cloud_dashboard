# Operations Products

Operations monitoring has its own raw snapshots, monitoring Zarr, and archived
quicklook products.

## Raw snapshots

- rolling latest snapshot:
  `/project/aurora/raw/ops_monitor/latest.json`
- time-stamped JSONL snapshots:
  `/project/aurora/raw/ops_monitor/ops_monitor_YYYYMMDD.jsonl`

These snapshots capture:

- source-host disk usage and probe reachability
- per-stream source recency, with a red operational state if a source has not
  produced data in the last 1.5 hours
- Aurora Power Supply battery voltage from the latest `DCInverterVolts` sample
  in the power Zarr, scored green above `52 V`, amber from `50-52 V`, and red
  below `50 V`
- local `/project`, `/data`, and `/` filesystem usage
- GWS usage and reachability
- per-stream mirror coverage, lag, and mismatch counts
- prune-gate and product-gate summaries
- systemd health for source sync, processing, and transfer units

## Operations Zarr

Path:

- `/data/aurora/products/ops_monitor/ops_monitor.zarr`

When checked on `2026-05-09`:

- dimension: `time`
- shape: `time=49`
- data variables: `296`
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
