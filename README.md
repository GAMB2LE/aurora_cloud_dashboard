# Aurora Cloud Dashboard

Panel dashboard and data-product scripts for the Aurora observing stack.

The default landing view is now the `Aurora Power Supply` instrument on the
`Interactive Data Browser`, because power health is the quickest first check
for whether the observing stack is behaving.

## Instruments

- `Aurora Power Supply` - fixed multi-panel 1D summary view on `Interactive Data Browser`, science quicklooks on `Science Quicklooks`, and `HK_APS` products on `House Keeping Quicklooks`. The science layout includes renewables, cumulative power, battery charging, separate output power / output voltage views with grouped units, and a bottom thermal-state panel for internal, heatsink, and auxiliary temperature sensors.
- `Ceilometer` - CL61 backscatter/depolarization Zarr with height-time plots on the `Interactive Data Browser` tab and daily science quicklooks on `Science Quicklooks`.
- `Cloud Radar` - RPG FMCW 94 GHz Zarr with height-time plots on the `Interactive Data Browser` tab and daily science quicklooks on `Science Quicklooks`.
- `Scanning Microwave Radiometer` - HATPRO radiometer Zarr with LWP/IWV, infrared surface temperature, temperature-profile plots, and Science Quicklooks.
- `Meteorology` - fixed multi-panel 1D summary view on `Interactive Data Browser`, science quicklooks on `Science Quicklooks`, and `HK_Met` products on `House Keeping Quicklooks`. The summary view combines the Meteorology Zarr with selected ASFS logger met traces at display time only, including ASFS Vaisala temperature, relative humidity, and pressure when present.
- `Radiation` - fixed multi-panel 1D summary view on `Interactive Data Browser`, science quicklooks on `Science Quicklooks`, and `HK_ASFS` products on `House Keeping Quicklooks`. The science layout uses the current ASFS logger schema, including SR30 shortwave, IR20 longwave, flux plates, KT15 surface temperature, and SR50 distance.
- `WXcam` - interactive stitched HDR video browser for the deployed `FISH HDR` stream, backed by a SQLite media catalog plus an HDR image Zarr. `Interactive Data Browser` uses a camera-style layout with obvious `Latest`, `Previous`, and `Next` controls plus `Updated X min ago` freshness metadata above the player. MP4 files are served through the dashboard static media route instead of being embedded in the Panel websocket payload. `Science Quicklooks` shows a `3 x 8` grid of hourly HDR JPG thumbnails for both today and past days, using the image nearest `:30` in each hour. PANO and AUTO/LONG/SHORT files remain on the camera host and are not part of the current local product stream.
- `Operations Dashboard` - dedicated top-level status tab with traffic-light indicators for source-host reachability, storage pressure, Aurora Power Supply battery voltage, Aurora Power Supply internal temperature, processing health, dashboard render performance, GWS transfer status, mirror verification, and prune readiness, driven from the locally collected operations snapshot stream. The collector also writes Phase 1 observe-only health JSON and daily Markdown reports for service/code monitoring. The storage section now breaks out the CL61 root/data views, ASS data/root views, APS data/root views, and the AURORA Cloud product/root views separately, with subtitles showing the resolved `pwd -P` path being measured. Archived `HK_Operations` quicklooks are still available from `House Keeping Quicklooks` and now use a curated diagnostics layout rather than plotting every numeric operations field.

Additional housekeeping products now exist for:

- `Ceilometer` as `HK_Ceilometer`, using CL61 time-only diagnostics such as sample cadence, receiver gain, backscatter noise/sum, alignment, visibility, and weather flags rather than duplicating the main backscatter/depolarization science fields.
- `Cloud Radar` as `HK_Radar`, using raw RPG LV1 housekeeping variables such as status, DD voltage, IF power, transmitter/receiver/PC temperatures, pointing, surface met, rain, LWP, and cloud-base diagnostics rather than duplicating the radar science moments.
- `ASFS Logger` as `HK_ASFS`, using a curated support layout for logger power, logger temperature/scan timing, SR30 orientation/fans/heaters, IR20 support, and sensor variability.
- `WXcam` as `HK_WXcam`

## Core files

- `app.py` - main Panel application.
- `wxcam_catalog.py` - shared helpers for the wxcam SQLite catalog.
- `index_wxcam_catalog.py` - indexes local WXcam HDR images and videos, with optional one-shot remote metadata bootstrap if explicitly requested.
- `build_wxcam_daily_videos.py` - builds daily wxcam MP4 products and hourly thumbnails from raw hourly clips.
- `extra_housekeeping.py` - housekeeping quicklook helpers for Ceilometer, Cloud Radar, and WXcam. Radar housekeeping is read from the raw RPG LV1 files because those support channels are not part of the radar science Zarr.
- `append_new_wxcam_to_zarr.py` - appends local WXcam HDR JPGs into per-stream pixel Zarr groups.
- `collect_operations_snapshot.py` - samples source-host disk state, local/GWS storage, mirror manifests, systemd health, dashboard HTTP health, and repo git state into raw Operations snapshots plus observe-only health reports.
- `append_new_ops_monitor_to_zarr.py` - appends Operations snapshots into the monitoring Zarr used for archived monitoring products and rebuilds automatically if the schema expands.
- `generate_ops_monitor_quicklooks.py` - builds summary and housekeeping quicklooks for the archived Operations monitor products.
- `append_new_*_to_zarr.py` - appenders for the numeric instruments.
- `generate_*_quicklooks.py`, `plot_*_last24h.py` - quicklook and latest-product generators.
- `grouped_timeseries.py` - shared summary-layout, labeling, downsampling, and quicklook helpers for the 1D instruments.

## Deployed paths

Primary paths come from `/etc/aurora-dashboard.env`.

- Dashboard app: `/opt/aurora-cloud-dashboard`
- Raw data root: `/project/aurora/raw`
- Product root: `/data/aurora/products`
- Quicklook root: `/data/aurora/products/quicklooks`

## Storage layout

The deployed host keeps raw mirrored inputs and derived dashboard products in
separate trees on purpose.

### `/project/aurora`

- Function: raw mirrored source data
- What lives there: synced instrument files coming from the remote source
  machines
- Examples:
  - `/project/aurora/raw/cl61`
  - `/project/aurora/raw/rpgfmcw94`
  - `/project/aurora/raw/vaisalamet`
  - `/project/aurora/raw/asfs/loggernet`
  - `/project/aurora/raw/power/level1`
  - `/project/aurora/raw/wxcam`
- Storage type: shared Ceph network filesystem
- Current filesystem size on `2026-05-18`: `4.0T`
- Current used on `2026-05-18`: `41G`
- Current available on `2026-05-18`: `3.9T`

So `/project/aurora` is the raw landing and mirror area.

### `/data/aurora`

- Function: processed products and dashboard-serving outputs
- What lives there:
  - Zarr stores
  - quicklook PNGs
  - WXcam catalog SQLite
  - WXcam daily videos and thumbnails
  - performance logs and other dashboard products
- Examples:
  - `/data/aurora/products/cl61/...zarr`
  - `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`
  - `/data/aurora/products/quicklooks/...`
  - `/data/aurora/products/wxcam/...`
- Storage type: local disk on `/dev/vdb`
- Current filesystem size on `2026-05-18`: `983G`
- Current used on `2026-05-18`: `197G`
- Current available on `2026-05-18`: `736G`

So `/data/aurora` is the product, work, and output area.

Short version:

- `/project/aurora` = raw source data, shared/networked, large, meant for
  mirrored inputs
- `/data/aurora` = derived products, local, faster/closer to the app, meant
  for Zarrs, plots, catalogs, and media outputs

Why the split is useful:

- raw files stay separate from regenerated products
- products can be deleted and rebuilt without touching the source mirror
- the dashboard reads smaller processed artifacts from local disk instead of
  always working directly from the raw mirror

Important products:

- CL61 Zarr: `/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr`
- Radar Zarr: `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`
- Vaisala met Zarr: `/data/aurora/products/vaisalamet/vaisalamet.zarr`
- ASFS logger Zarr: `/data/aurora/products/asfs_logger/asfs_logger.zarr`
- ASFS fast-sonic Zarr: `/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr`
- Power Zarr: `/data/aurora/products/power/power.zarr`
- WXcam Zarr: `/data/aurora/products/wxcam/wxcam.zarr`
- WXcam catalog: `/data/aurora/products/wxcam/wxcam_catalog.sqlite`
- WXcam daily videos: `/data/aurora/products/wxcam/daily_videos`
- WXcam hourly thumbnails: `/data/aurora/products/wxcam/hourly_thumbnails`
- Operations raw snapshots: `/project/aurora/raw/ops_monitor`
- Operations Zarr: `/data/aurora/products/ops_monitor/ops_monitor.zarr`
- Operations quicklooks: `/data/aurora/products/quicklooks/ops_monitor`
- Operations health reports: `/data/aurora/products/ops_monitor/health`

The deployed WXcam mirror under `/project/aurora/raw/wxcam` retains only FISH
HDR JPG and MP4 files locally. PANO and AUTO/LONG/SHORT files remain on the
camera host and are not cataloged, Zarr-appended, or archived from this VM.

## Services

Systemd services are installed system-wide under `/etc/systemd/system/`.

- Dashboard:
  - `aurora-dashboard.service`
- CL61:
  - `aurora-cl61-source-sync.timer`
  - `aurora-ceilometer-append.timer`
  - `aurora-ceilometer-last24h.timer`
  - `aurora-ceilometer-quicklooks.timer`
- Radar:
  - `aurora-radar-source-sync.timer`
  - `aurora-radar-append.timer`
  - `aurora-radar-quicklooks.timer`
- Vaisala met:
  - `aurora-vaisalamet-source-sync.timer`
  - `aurora-vaisalamet-append.timer`
  - `aurora-vaisalamet-quicklooks.timer`
- ASFS logger:
  - `aurora-asfs-logger-source-sync.timer`
  - `aurora-asfs-logger-append.timer`
  - `aurora-asfs-logger-quicklooks.timer`
- ASFS fast-sonic:
  - `aurora-asfs-fast-sonic-source-sync.timer`
  - `aurora-asfs-fast-sonic-append.timer`
  - `aurora-asfs-fast-sonic-quicklooks.timer`
- Power:
  - `aurora-power-source-sync.timer`
  - `aurora-power-append.timer`
  - `aurora-power-quicklooks.timer`
- WXcam:
  - `aurora-wxcam-source-sync.timer`
  - `aurora-wxcam-catalog.timer`
  - `aurora-wxcam-daily-videos.timer` (daily MP4s plus hourly thumbnails)
  - `aurora-wxcam-append.timer`
- Operations:
  - `aurora-ops-monitor-collect.timer`
  - `aurora-ops-monitor-append.timer`
  - `aurora-ops-monitor-quicklooks.timer`
  - `aurora-mirror-verify.timer`

Useful commands:

```bash
sudo systemctl status aurora-dashboard.service
sudo systemctl list-timers --all | rg '^.*aurora-'
sudo journalctl -u aurora-dashboard.service -f
```

## Performance logging

The dashboard writes structured JSONL timing events to:

- `/data/aurora/products/dashboard/dashboard_perf.jsonl`

The log rotates automatically. You can change the path and rotation settings with:

- `AURORA_DASHBOARD_PERF_LOG`
- `AURORA_DASHBOARD_PERF_LOG_MAX_BYTES`
- `AURORA_DASHBOARD_PERF_LOG_BACKUP_COUNT`
- `AURORA_DASHBOARD_PERF_ENABLED`
- `AURORA_DASHBOARD_SESSION_HEARTBEAT_MS`

Useful commands:

```bash
tail -f /data/aurora/products/dashboard/dashboard_perf.jsonl
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 24
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 6 --event interactive_view_update
```

The main event families currently logged are:

- `base_dataset_open`
- `dataset_time_bounds`
- `dataset_time_bounds_cache_hit`
- `window_open`
- `interactive_view_update`
- `hatpro_render`
- `stacked_timeseries_render`
- `science_quicklook_render`
- `housekeeping_quicklook_render`
- `wxcam_interactive_render`
- `wxcam_calendar_day_view`
- `wxcam_calendar_sync`
- `session_loaded`
- `session_heartbeat`
- `session_destroyed`
- `ui_selection_change`
- `plot_relayout`

The interactive browser now also uses a few lightweight runtime behaviors to
keep switching responsive:

- per-instrument pane reuse, so switching back can show the last rendered view
  immediately while a refresh is queued
- short-lived cached dataset time bounds and latest timestamps
- stale-render protection, so older renders cannot overwrite a newer selection
- a loading skeleton for uncached views and a slim refresh banner for warmed
  views
- coarse-first rendering for the heavier 2D interactive plots before a full
  detail pass replaces them

Each event also carries session and concurrency context when available, including:

- `session_id`
- `live_sessions`
- `server_sessions`
- `total_sessions`
- `session_age_s`
- `busy`

The session events and UI-selection events are especially useful for understanding:

- Radar, VaisalaMET, ASFS logger, and Power browsing behavior in real use
- WXcam science-quicklook usage patterns
- multi-user overlap and concurrent browsing on the live dashboard

## Running locally

```bash
cd /opt/aurora-cloud-dashboard
source venv/bin/activate
panel serve app.py --address 127.0.0.1 --port 5006 --allow-websocket-origin=<host>
```

## Documentation publishing

- This repo carries the three repo-side pieces described in
  `https://gamb2le.pages.dev/documentation-docs/`:
  - `mkdocs.yml`
  - `docs/`
  - `.github/workflows/trigger-docs.yml`
- The repo documentation is now split into sectioned MkDocs pages so the portal
  left navigation can expose instruments, runtime/storage details, and per-Zarr
  product pages directly.
- The `trigger-docs.yml` workflow asks the central `GAMB2LE/mkdocs-portal`
  repo to rebuild the unified site at `https://gamb2le.pages.dev/`.
- The repo-local GitHub Pages workflow has been removed; the central portal is
  the only intended public documentation destination.
- Local docs checks can be run with `python3 check_docs.py`.
- That trigger workflow expects two GitHub Actions secrets in this repo:
  - `APP_ID = 2899200`
  - `APP_PRIVATE_KEY = <the GitHub App private key from the docs process>`
- The central portal repo still has to include this repository in its own
  `mkdocs.yml` nav and its docs-clone workflow, exactly as described in the
  unified docs instructions.

## Notes

- Radar data currently contains at least one bogus far-future timestamp in the Zarr store. `app.py` filters clearly invalid future times when computing bounds and plotting windows so the interactive view stays usable.
- `Meteorology`, `Radiation`, and `Aurora Power Supply` now use fixed presentation-layer summary layouts. Their ingest, local retention, and Zarr schemas stay unchanged; the dashboard just presents curated subsets of the same 1D variables on the `Interactive Data Browser` tab.
- `Meteorology` summary plots merge selected ASFS logger met variables into the Meteorology presentation layer without changing either underlying Zarr store. With the current ASFS CRD schema this includes ASFS Vaisala temperature, relative humidity, and pressure alongside the existing Metek wind and temperature context.
- `Radiation` summary plots use the current ASFS CRD fields for SR30 shortwave radiation and support data, IR20 longwave radiation and support data, flux plates, KT15 surface temperature, and SR50 distance.
- The dashboard UI now uses four top-level tabs: `Interactive Data Browser`, `Science Quicklooks`, `House Keeping Quicklooks`, and `Operations Dashboard`.
- The dashboard starts on `Aurora Power Supply` by default.
- Availability bars, freshness/status chips, and share/download controls are shown beneath the rendered content in each tab so the data view stays visually primary.
- Summary-plot legends are kept per panel in a dedicated right-side gutter beyond the right y-axis labels instead of sharing one global legend.
- `Science Quicklooks` and `House Keeping Quicklooks` use separate tab state. Science quicklooks show one product at a time, while housekeeping quicklooks only appear for instruments that actually have HK images.
- `Operations Dashboard` is the live status surface for current health, while archived `HK_Operations` quicklooks remain available from `House Keeping Quicklooks`. The live tab reads the latest Operations snapshot JSON directly so the traffic lights reflect current service and transfer state even when archived monitoring quicklooks lag behind. Archive traffic lights now treat a stream as healthy when the settled GWS mirror has zero missing or mismatched files, even if one or two brand-new files are still catching the next transfer cycle.
- `WXcam` full-archive mirroring is treated as a backfill state rather than a hard failure when the historical copy is still catching up but the live tip is current.
- `HK_Ceilometer` excludes the main science fields (`beta_att`, `linear_depol_ratio`) and focuses on real housekeeping diagnostics such as sample cadence, gain, noise, tilt, visibility, and weather flags.
- `HK_Radar` is generated from the raw RPG LV1 support variables instead of the radar science Zarr, so the housekeeping plot now reflects instrument state, thermal state, pointing, surface met, and ancillary retrieval health.
- Browser state is reflected into the URL as controls change. The deployed Panel service also uses longer session retention and websocket keepalives so mobile browsers recover better after being backgrounded; if the phone still kills the tab, reopening the URL restores the selected tab, instrument, and key controls.
- `HK_WXcam` is catalog-driven rather than pixel-driven. It summarizes hourly HDR image/video counts, median HDR JPG size, and the offset of the representative hourly image from the `:30` target used by the science grid.
- `ASFS Fast Sonic` data products still exist on disk, but the instrument is currently hidden from the dashboard selectors.
- WXcam keeps the stitched FISH HDR MP4 player on `Interactive Data Browser`. `Today (latest)` uses `latest.mp4`, which is rebuilt from the most recent 24 hourly clips. Historical days use one stitched MP4 per UTC day.
- WXcam MP4s are served from `/wxcam-media/...` with a file mtime cache key.
  This keeps large videos out of the Bokeh websocket and lets browsers request
  byte ranges for normal video seeking.
- The WXcam `Science Quicklooks` tab is image-driven. For each UTC hour it selects the HDR JPG closest to `:30` and shows a tile only when an image exists for that hour.
- `Operations` snapshots are collected every 5 minutes from source-host SSH probes, local filesystem probes, mirror-manifest summaries, systemd unit state, dashboard HTTP checks, dashboard/infra git state, and the latest Aurora Power Supply `DCInverterVolts` and `InternalTemperature` samples from the power Zarr. They stamp each snapshot with both `time_utc` and `snapshot_time_utc`, mark each stream red when the source has not produced data in the last 1.5 hours, score battery voltage as green above `52 V`, amber from `50-52 V`, and red below `50 V`, and score APS internal temperature as green below `35 C`, amber from `35-40 C`, and red at `40 C` or above. The top-level `Operations Dashboard` tab reads the latest snapshot directly, while the Phase 1 observe-only sentinel also writes `latest_health.json` and Markdown health reports under `/data/aurora/products/ops_monitor/health`. A fresh deployment can therefore show a live Operations tab and health report before the archived monitoring quicklook directory fills in.

## Zarr data products

### Ceilometer Zarr structure

Path: `/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr`

This store is a single xarray dataset with:

- dimensions: `time`, `range`, `layer`
- deployed shape checked on `2026-05-18`: `time=101532`, `range=3276`, `layer=5`
- coordinates:
  - `time` - profile timestamps
  - `range` - range gate center in meters
  - `layer` - cloud-layer index
  - `latitude`, `longitude` - site coordinates

Useful root attrs include:

- `title = "CL61D CL61 with Depolarization"`
- `source = "gamb2le_depolarisation_lidar_ceilometer"`
- `conventions = "CF-1.8"`
- `profile_interval_in_seconds = 10`
- `file_temporal_span_in_minutes = 5.0`
- `schema_version = "1.3"`
- `instrument_serial_number = "X1627532"`
- `overlap_function_provided = 1`
- `overlap_is_corrected = 1`

Variable layout:

- main `time x range` profile fields:
  - `beta_att`
  - `linear_depol_ratio`
  - `p_pol`
  - `x_pol`
- `time x layer` cloud diagnostics:
  - `cloud_base_heights`
  - `cloud_penetration_depth`
  - `cloud_thickness`
  - `sky_condition_cloud_layer_covers`
  - `sky_condition_cloud_layer_heights`
- `time`-only diagnostics:
  - `beta_att_noise_level`
  - `beta_att_sum`
  - `fog_detection`
  - `precipitation_detection`
  - `receiver_gain`
  - `sky_condition_total_cloud_cover`
  - `tilt_angle`
  - `tilt_correction`
  - `vertical_visibility`
- scalar metadata:
  - `range_resolution`
  - `elevation`
  - `azimuth_angle`
  - `airplane_filter_max_range`
  - `cloud_calibration_factor`
  - `cloud_calibration_factor_user`

Chunking:

- `time x range` fields such as `beta_att` are chunked `(30, full-range)`
- `time x layer` diagnostics are chunked `(30, 5)`
- `time`-only diagnostics are chunked `(30,)`

### Cloud Radar Zarr structure

Path: `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`

This store is a single xarray dataset with:

- dimensions: `time`, `range`
- deployed shape checked on `2026-05-18`: `time=147178`, `range=312`
- coordinates:
  - `time` - derived from `Time + Timems`
  - `range` - concatenated chirp range gates from `C1Range` and `C2Range`

The dataset currently contains 13 `float32` `time x range` fields:

- `ZE_dBZ`
- `ZE45_dBZ`
- `MeanVel`
- `ZDR`
- `SRCX`
- `SpecWidth`
- `SLDR`
- `Skew`
- `RHV`
- `PhiDP`
- `Kurt`
- `KDP`
- `DiffAtt`

Conversion notes:

- reflectivity-style fields are converted to dBZ during ingest
- fill values at or below the radar missing-data sentinel are converted to `NaN`
- the dashboard masks obviously bogus far-future timestamps when plotting or choosing the latest time window
- append runs now track the `range` layout; if new raw files arrive with a different radar geometry, the appender backs up the existing store and rebuilds a fresh store from the newest contiguous geometry run

Chunking:

- radar science variables are chunked `(400, full-range)`

### Meteorology (VaisalaMET) Zarr structure

Path: `/data/aurora/products/vaisalamet/vaisalamet.zarr`

This store is a single time-indexed xarray dataset with:

- dimension: `time`
- deployed shape checked on `2026-05-18`: `time=221243`
- coordinate:
  - `time` - parsed from the raw `timestamp` column, localized as `Europe/London`, then converted to UTC before storage

Useful root attrs include:

- `instrument = "vaisalamet"`
- `title = "Vaisala met station data"`
- `source = "vaisala_met_level0_*.dat"`

Variable layout:

- one `float32` `time` series per retained source column
- the current deployed store contains 68 variables
- examples include:
  - `baro_hPa`
  - `h1_ah`
  - `h1_e`
  - `h1_err_rh_meas_err`
  - `h1_err_temp_meas_err`
  - the various `*_err_*`, `*_dev_*`, and `*_st_*` health/status flags

Schema note:

- append runs align incoming files to the existing Zarr schema
- missing existing columns are filled with `NaN`
- newly appearing columns are dropped unless the store is rebuilt

Chunking:

- `time`-only variables are chunked `(1200,)`

### ASFS Zarr structure

Path: `/data/aurora/products/asfs_logger/asfs_logger.zarr`

This store is a single time-indexed xarray dataset with:

- dimension: `time`
- deployed shape checked on `2026-05-18`: `time=20153`
- coordinate:
  - `time` - parsed directly from the TOA5 `TIMESTAMP` column

Useful root attrs include:

- `instrument = "asfs-logger"`
- `title = "ASFS science data"`
- `source = "asfs-logger_sci_DD_MM_YYYY.dat or aurora_asfs_data_sci_YYYYMMDDHHMM.dat"`

Variable layout:

- one `float32` `time` series per retained source column
- the current deployed store contains 89 variables
- examples include:
  - `PTemp_Avg`
  - `batt_volt_Avg`
  - `amp_meter_48vdc_Avg`
  - `kt15_amb_Avg`, `kt15_tem_Avg`
  - `licor_co2_out_Avg`, `licor_h2o_out_Avg`
  - `metek_x_out_Avg`, `metek_T_out_Avg`
  - `RECORD`

Schema note:

- append runs keep the existing variable set fixed in the same way as `vaisalamet`

Chunking:

- `time`-only variables are chunked `(1200,)`

### ASFS Fast Sonic Zarr structure

Path: `/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr`

This store is a single time-indexed xarray dataset with:

- dimension: `time`
- deployed shape checked on `2026-05-18`: `time=1951976`
- coordinate:
  - `time` - parsed from `TIMESTAMP` and offset by `metek_msec_out` to preserve sub-second timing

Useful root attrs include:

- `instrument = "asfs-fast-sonic"`
- `title = "ASFS fast-sonic data"`
- `source = "asfs-logger_fast_sonic_DD_MM_YYYY.dat or aurora_asfs_data_fast_sonic_YYYYMMDDHHMM.dat"`

Variable layout:

- one `float32` `time` series per retained source column
- the deployed store currently contains these 10 variables:
  - `RECORD`
  - `metek_InclX_out`
  - `metek_InclY_out`
  - `metek_T_out`
  - `metek_msec_out`
  - `metek_quality_out`
  - `metek_senspathstate_out`
  - `metek_x_out`
  - `metek_y_out`
  - `metek_z_out`

Schema note:

- append runs keep the existing variable set fixed in the same way as `asfs-logger`

Chunking:

- `time`-only variables are chunked `(24000,)`

### Power Zarr structure

Path: `/data/aurora/products/power/power.zarr`

This store is a single time-indexed xarray dataset with:

- dimension: `time`
- deployed shape checked on `2026-05-18`: `time=842870`
- coordinate:
  - `time` - parsed from the raw `aps_time` column

Useful root attrs include:

- `instrument = "power"`
- `title = "Power level1 data"`
- `source = "power_data_YYYYMMDD.csv"`
- `wind_columns_excluded = "true"`

Variable layout:

- one `float32` `time` series per retained source column
- the current deployed store contains 43 variables
- raw column names are normalized by replacing `.` with `_`
- columns containing `wind` and columns ending in `time` are excluded at ingest
- examples include:
  - `ACOutputAmps`
  - `ACOutputHZ`
  - `ACOutputVolts`
  - `ACOutputWatts`
  - `BatteryAmps`
  - `BatteryState`
  - `BatteryWatts`
  - `DCInverterWatts`
  - `HeatsinkTemperature`
  - `InternalTemperature`
  - `TempSensor1`
  - `TempSensor2`
  - `TempSensor3`
  - `TempSensor4`
  - `MaxSolarWatts_East`

Schema note:

- append runs keep the existing variable set fixed in the same way as `vaisalamet`

Chunking:

- `time`-only variables are chunked `(1200,)`

## WXcam data products

### Local raw mirror

The deployed WXcam raw mirror retains only FISH HDR JPG and MP4 files locally:

- `/project/aurora/raw/wxcam/FISH`

PANO and AUTO/LONG/SHORT files remain on the camera host and are not cataloged,
Zarr-appended, or archived from this VM.

### Catalog

The WXcam catalog at `/data/aurora/products/wxcam/wxcam_catalog.sqlite` indexes FISH HDR JPGs and FISH HDR MP4s. Timestamps are derived from filenames and stored as UTC. Key fields include:

- `image_type` - `fish_hdr`
- `media_kind` - `image` or `video`
- `time_utc`, `time_epoch_ns`, `day_utc`
- `raw_path`, `relative_path`, `filename`
- `width`, `height`, `size_bytes`

### Daily videos and hourly thumbnails

- Daily MP4s live under `/data/aurora/products/wxcam/daily_videos/<image_type>/YYYYMMDD.mp4`
- Rolling latest MP4s live at `/data/aurora/products/wxcam/daily_videos/<image_type>/latest.mp4`
- Hourly thumbnails live under `/data/aurora/products/wxcam/hourly_thumbnails/<image_type>/YYYYMMDD/`

Daily videos are stitched from the 24 hourly MP4 clips for that UTC day. `latest.mp4` is stitched from the most recent 24 hourly clips across day boundaries.

### WXcam Zarr structure

The WXcam Zarr at `/data/aurora/products/wxcam/wxcam.zarr` contains HDR JPG image data only. MP4 products are stored separately.

Root attrs:

- `instrument = "wxcam"`
- `title = "Aurora wxcam HDR images"`
- `storage_policy = "Contains locally retained FISH HDR JPG image data with timestamps derived from filenames; MP4 products are stored separately."`

Root groups:

- `fish_hdr`

Each group stores one xarray dataset with:

- dimensions: `time`, `y`, `x`, `channel`
- coordinates:
  - `time` - UTC image timestamps
  - `y` - pixel row index
  - `x` - pixel column index
  - `channel` - RGB color channel labels: `R`, `G`, `B`
- data variables:
  - `image[time, y, x, channel]` - `uint8` RGB pixel data
  - `filename[time]`
  - `width[time]`
  - `height[time]`
  - `size_bytes[time]`

Group-specific image geometry checked on `2026-05-18`:

- `fish_hdr`: `time=13070`, `3120 x 3040` pixels, chunked as `(1, 1024, 1024, 3)`
