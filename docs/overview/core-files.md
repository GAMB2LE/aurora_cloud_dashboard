# Core Files

This repo is a mix of application code, appenders, quicklook generators, and
operations tooling. The most important files are grouped below by role.

## Dashboard application

- `app.py` - main Panel application and UI wiring
- `grouped_timeseries.py` - shared summary-layout, labeling, and downsampling
  helpers for the curated 1D instruments

## Numeric data appenders

- `append_new_cloud_radar_to_zarr.py`
- `append_new_vaisalamet_to_zarr.py`
- `append_new_asfs_logger_to_zarr.py`
- `append_new_asfs_fast_sonic_to_zarr.py`
- `append_new_power_to_zarr.py`
- `append_new_netcdf_to_zarr.py`

These scripts ingest raw mirrored files and append them into the deployed Zarr
stores.

## Quicklooks and latest plots

- `generate_cloud_radar_quicklooks.py`
- `generate_vaisalamet_quicklooks.py`
- `generate_asfs_logger_quicklooks.py`
- `generate_asfs_fast_sonic_quicklooks.py`
- `generate_power_quicklooks.py`
- `generate_ops_monitor_quicklooks.py`
- `plot_*_last24h.py`

These scripts generate the archived PNG products and the latest-view assets
used by the dashboard.

## WXcam tooling

- `wxcam_catalog.py` - shared catalog helpers
- `index_wxcam_catalog.py` - builds or refreshes the SQLite catalog
- `build_wxcam_daily_videos.py` - builds daily MP4s, `latest.mp4`, and hourly
  thumbnails
- `append_new_wxcam_to_zarr.py` - appends HDR JPG image data to the WXcam Zarr

## Operations monitoring

- `collect_operations_snapshot.py` - collects source-host, storage, mirror,
  systemd, dashboard endpoint, and git health into raw JSONL snapshots plus
  observe-only health JSON and Markdown reports
- `send_ops_alerts.py` - evaluates the latest operations snapshot and sends
  threshold email alerts with stateful repeat and recovery handling
- `append_new_ops_monitor_to_zarr.py` - appends or rebuilds the monitoring Zarr
- `extra_housekeeping.py` - extra housekeeping quicklook helpers, including the
  Ceilometer, Cloud Radar, and WXcam HK products

## Support and maintenance

- `summarize_dashboard_perf.py` - summarizes JSONL performance timing logs
- `consolidate_zarr_metadata.py` - metadata consolidation helper

The repo also contains older experimental app variants preserved as reference:

- `ceilo_app_dask_slow_20251028_212300.py`
- `ceilo_app_no_dask_works_20251028_195400.py`
