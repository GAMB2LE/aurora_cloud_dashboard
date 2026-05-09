# Aurora Cloud Dashboard

Panel dashboard and data-product scripts for the Aurora observing stack.

## Instruments

- `Ceilometer` - CL61 backscatter/depolarization Zarr with interactive height-time plots and calendar quicklooks.
- `Cloud Radar` - RPG FMCW 94 GHz Zarr with interactive height-time plots and calendar quicklooks.
- `vaisalamet` - stacked 1D time-series plots for all retained variables, plus daily quicklooks.
- `asfs-logger` - stacked 1D time-series plots for all retained variables, plus daily quicklooks.
- `power` - stacked 1D time-series plots for all retained non-wind variables, plus daily quicklooks.
- `WXcam` - interactive stitched HDR video browser for `FISH HDR` and `PANO HDR`, backed by a SQLite media catalog plus an HDR image Zarr. The Interactive tab shows rolling latest and per-day MP4s. The Calendar tab shows a `3 x 8` grid of hourly HDR JPG thumbnails for both today and past days, using the image nearest `:30` in each hour.

`asfs-fast-sonic` is processed into its own Zarr store for downstream analysis, but it is not exposed in the dashboard UI.

## Core files

- `app.py` - main Panel application.
- `wxcam_catalog.py` - shared helpers for the wxcam SQLite catalog.
- `index_wxcam_catalog.py` - indexes local WXcam HDR images and videos, with optional one-shot remote metadata bootstrap if explicitly requested.
- `build_wxcam_daily_videos.py` - builds daily wxcam MP4 products and hourly thumbnails from raw hourly clips.
- `append_new_wxcam_to_zarr.py` - appends local WXcam HDR JPGs into per-stream pixel Zarr groups.
- `append_new_*_to_zarr.py` - appenders for the numeric instruments.
- `generate_*_quicklooks.py`, `plot_*_last24h.py` - quicklook and latest-product generators.

## Deployed paths

Primary paths come from `/etc/aurora-dashboard.env`.

- Dashboard app: `/opt/aurora-cloud-dashboard`
- Raw data root: `/project/aurora/raw`
- Product root: `/data/aurora/products`
- Quicklook root: `/data/aurora/products/quicklooks`

Important products:

- CL61 Zarr: `/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr`
- Radar Zarr: `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`
- Vaisala met Zarr: `/data/aurora/products/vaisalamet/vaisalamet.zarr`
- ASFS logger Zarr: `/data/aurora/products/asfs_logger/asfs_logger.zarr`
- ASFS fast-sonic Zarr: `/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr`
- Power Zarr: `/data/aurora/products/power/power.zarr`
- WXcam Zarr: `/data/aurora/products/wxcam/wxcam.zarr`
- Wxcam catalog: `/data/aurora/products/wxcam/wxcam_catalog.sqlite`
- Wxcam daily videos: `/data/aurora/products/wxcam/daily_videos`
- Wxcam hourly thumbnails: `/data/aurora/products/wxcam/hourly_thumbnails`

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
- Power:
  - `aurora-power-source-sync.timer`
  - `aurora-power-append.timer`
  - `aurora-power-quicklooks.timer`
- Wxcam:
  - `aurora-wxcam-source-sync.timer`
  - `aurora-wxcam-catalog.timer`
  - `aurora-wxcam-daily-videos.timer` (daily MP4s plus hourly thumbnails)
  - `aurora-wxcam-append.timer`

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

Useful commands:

```bash
tail -f /data/aurora/products/dashboard/dashboard_perf.jsonl
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 24
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 6 --event interactive_view_update
```

The main event families currently logged are:

- `base_dataset_open`
- `dataset_time_bounds`
- `window_open`
- `interactive_view_update`
- `hatpro_render`
- `stacked_timeseries_render`
- `calendar_render`
- `wxcam_interactive_render`
- `wxcam_calendar_day_view`

## Running locally

```bash
cd /opt/aurora-cloud-dashboard
source venv/bin/activate
panel serve app.py --address 127.0.0.1 --port 5006 --allow-websocket-origin=<host>
```

## Notes

- Radar data currently contains at least one bogus far-future timestamp in the Zarr store. `app.py` filters clearly invalid future times when computing bounds and plotting windows so the interactive view stays usable.
- WXcam keeps the stitched MP4 player on the Interactive tab. `Today (latest)` uses `latest.mp4`, which is rebuilt from the most recent 24 hourly clips. Historical days use one stitched MP4 per UTC day.
- The WXcam Calendar tab is image-driven. For each UTC hour it selects the HDR JPG closest to `:30` and shows a tile only when an image exists for that hour.

## WXcam data products

### Local raw mirror

The deployed WXcam raw mirror only retains HDR assets locally:

- `FISH/HDR_*.jpg`
- `FISH/HDR_*.mp4`
- `PANO/HDR_*_PANO.jpg`
- `PANO/HDR_*_PANO.mp4`

Non-HDR WXcam files may still exist on the remote source host, but they are not mirrored into the local dashboard raw tree.

### Catalog

The WXcam catalog at `/data/aurora/products/wxcam/wxcam_catalog.sqlite` indexes both HDR JPGs and HDR MP4s. Timestamps are derived from filenames and stored as UTC. Key fields include:

- `image_type` - `fish_hdr` or `pano_hdr`
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
- `storage_policy = "Contains locally retained HDR JPG image data for fish_hdr and pano_hdr with timestamps derived from filenames; MP4 products are stored separately."`

Root groups:

- `fish_hdr`
- `pano_hdr`

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

Group-specific image geometry:

- `fish_hdr`: `3120 x 3040` pixels, chunked as `(1, 1024, 1024, 3)`
- `pano_hdr`: `2880 x 750` pixels, chunked as `(1, 750, 1024, 3)`
