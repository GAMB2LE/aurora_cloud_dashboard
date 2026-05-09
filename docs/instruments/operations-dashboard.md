# Operations Dashboard

Operations Dashboard is the live operational status tab for the whole Aurora
stack.

## What it shows

- source-host probe health
- local and remote storage pressure
- Aurora Power Supply battery voltage from `DCInverterVolts`, scored green
  above `52 V`, amber from `50-52 V`, and red below `50 V`
- Aurora Power Supply internal temperature from `InternalTemperature`,
  scored green below `35 C`, amber from `35-40 C`, and red at `40 C` or above
- source sync and processing health
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

These include:

- summary quicklooks
- `HK_Operations`

Detailed product documentation:

- [Operations products](../data-products/operations-products.md)
