# Aurora Cloud Dashboard

The Aurora Cloud Dashboard is the live Panel application and data-product
tooling for the Aurora observing stack.

## What this repo covers

- the public dashboard served at `data.gamb2le.co.uk`
- Zarr appenders and quicklook generation for the deployed instruments
- WXcam cataloging, daily-video generation, thumbnails, and pixel Zarr writes
- operations monitoring, archived monitoring quicklooks, and dashboard
  performance logging

## Dashboard structure

The deployed interface is organized into four top-level views:

- **Interactive Data Browser**
- **Science Quicklooks**
- **House Keeping Quicklooks**
- **Operations Dashboard**

## Main instruments

- **Ceilometer**
- **Cloud Radar**
- **Meteorology**
- **Radiation**
- **Aurora Power Supply**
- **WXcam**

The Operations Dashboard is also backed by this repo through the `ops_monitor`
collector, Zarr, and quicklook pipeline.

## Storage model

The deployed host deliberately separates raw mirrored inputs from derived
products:

- `/project/aurora/raw` contains raw mirrored source data from the upstream
  instrument hosts
- `/data/aurora/products` contains Zarrs, quicklooks, WXcam media products,
  the WXcam SQLite catalog, operations products, and performance logs

This split keeps regeneratable products separate from the raw mirror and lets
the dashboard serve smaller local products without reading directly from the
source-style trees.

## Important paths

- Dashboard application: `/opt/aurora-cloud-dashboard`
- Raw data root: `/project/aurora/raw`
- Product root: `/data/aurora/products`
- Quicklook root: `/data/aurora/products/quicklooks`
- Dashboard environment file: `/etc/aurora-dashboard.env`

## Key entry points

- `app.py` - main Panel application
- `grouped_timeseries.py` - shared 1D summary plot and quicklook helpers
- `append_new_*_to_zarr.py` - numeric instrument appenders
- `generate_*_quicklooks.py` - archived quicklook generators
- `wxcam_catalog.py` - shared WXcam catalog helpers
- `collect_operations_snapshot.py` - operations snapshot collector

## Operations and performance

This repo also contains:

- performance logging to
  `/data/aurora/products/dashboard/dashboard_perf.jsonl`
- operations snapshots in `/project/aurora/raw/ops_monitor`
- operations Zarr and quicklooks under `/data/aurora/products/ops_monitor`
  and `/data/aurora/products/quicklooks/ops_monitor`

## Related documentation

- Infrastructure and rebuild docs live in the
  **Aurora Cloud Infrastructure** documentation section
- Source sync and deployment details live in the infra repo, not here

## Source repository

- GitHub: <https://github.com/GAMB2LE/aurora_cloud_dashboard>
