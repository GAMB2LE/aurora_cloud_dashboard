# Deployed Paths

Primary runtime paths come from `/etc/aurora-dashboard.env`. Production and
development share the same read-only mirror layout, but only production owns
the normal raw and product writers.

## Main application paths

- dashboard app checkout: `/opt/aurora-cloud-dashboard`
- raw data root: `/project/aurora/raw`
- product root: `/data/aurora/products`
- quicklook root: `/data/aurora/products/quicklooks`
- interactive prewarm root: `/data/aurora/products/dashboard/prewarm`

Development-only experiments must use:

- raw/test inputs: `/project/aurora/dev-raw`
- derived products: `/data/aurora/dev-products`

The development dashboard may read mirrored products from
`/data/aurora/products`; it must not write experimental output there.

## Important deployed products

- CL61 Zarr:
  `/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr`
- Cloud Radar Zarr: `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`
- HATPRO Zarr: `/data/aurora/products/hatprog5/hatpro.zarr`
- HATPRO quicklooks: `/data/aurora/products/quicklooks/hatpro`
- Meteorology Zarr: `/data/aurora/products/vaisalamet/vaisalamet.zarr`
- ASFS Logger Zarr: `/data/aurora/products/asfs_logger/asfs_logger.zarr`
- ASFS Fast Sonic Zarr:
  `/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr`
- ASFS Fast Gas Zarr:
  `/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr`
- Power Zarr: `/data/aurora/products/power/power.zarr`
- ASS PDU raw mirror: `/project/aurora/raw/pdu`
- ASS PDU Zarr: `/data/aurora/products/power/pdu.zarr`
- WXcam Zarr:
  `/data/aurora/products/wxcam/wxcam.zarr` (mutable and excluded from archives)
- WXcam catalog: `/data/aurora/products/wxcam/wxcam_catalog.sqlite`
- AURORACam raw mirror: `/project/aurora/raw/auroracam`
- AURORACam Zarr: `/data/aurora/products/auroracam/auroracam.zarr`
- Menapia flight catalog and profiles:
  `/data/aurora/products/menapia/catalog.json` and
  `/data/aurora/products/menapia/flights`
- Menapia per-flight plots: `/data/aurora/products/menapia/plots`
- UAS daily Science Quicklooks: `/data/aurora/products/quicklooks/uas`
- Operations Zarr: `/data/aurora/products/ops_monitor/ops_monitor.zarr`
- Operations health outputs:
  `/data/aurora/products/ops_monitor/health/latest_health.json` and
  `/data/aurora/products/ops_monitor/health/latest_report.md`
- Infrastructure archive-health contract:
  `/data/aurora/internal/archive_status/health-v1.json`

## Related docs

- [Storage layout](storage-layout.md)
- [Services and timers](services-and-timers.md)
