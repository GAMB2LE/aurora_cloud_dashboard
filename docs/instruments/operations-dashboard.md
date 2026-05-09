# Operations Dashboard

Operations Dashboard is the live operational status tab for the whole Aurora
stack.

## What it shows

- source-host probe health
- local and remote storage pressure
- Aurora Power Supply battery voltage from `DCInverterVolts`, scored green
  above `52 V`, amber from `50-52 V`, and red below `50 V`
- source sync and processing health
- GWS transfer status
- mirror verification and prune-readiness indicators
- per-stream archive state, including WXcam backfill progress

## Display model

This tab reads the latest operations snapshot directly rather than waiting for
an archived quicklook to exist. That means a fresh deployment can show the live
Operations tab before the archived operations PNGs have accumulated enough
samples to plot.

## Archived products

The archived operations products live under:

- `/data/aurora/products/quicklooks/ops_monitor`

These include:

- summary quicklooks
- `HK_Operations`

Detailed product documentation:

- [Operations products](../data-products/operations-products.md)
