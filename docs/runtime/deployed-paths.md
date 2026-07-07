# Deployed Paths

Primary runtime paths come from `/etc/aurora-dashboard.env`.

## Main application paths

- dashboard app checkout: `/opt/aurora-cloud-dashboard`
- raw data root: `/project/aurora/raw`
- product root: `/data/aurora/products`
- quicklook root: `/data/aurora/products/quicklooks`
- interactive prewarm root: `/data/aurora/products/dashboard/prewarm`

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
- WXcam Zarr: `/data/aurora/products/wxcam/wxcam.zarr`
- WXcam catalog: `/data/aurora/products/wxcam/wxcam_catalog.sqlite`
- Operations Zarr: `/data/aurora/products/ops_monitor/ops_monitor.zarr`
- Operations health outputs:
  `/data/aurora/products/ops_monitor/health/latest_health.json` and
  `/data/aurora/products/ops_monitor/health/latest_report.md`

## Related docs

- [Storage layout](storage-layout.md)
- [Services and timers](services-and-timers.md)
