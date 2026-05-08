# Aurora Cloud Dashboard

Panel dashboard and data-product scripts for the Aurora observing stack.

## Instruments

- `Ceilometer` - CL61 backscatter/depolarization Zarr with interactive height-time plots and calendar quicklooks.
- `Cloud Radar` - RPG FMCW 94 GHz Zarr with interactive height-time plots and calendar quicklooks.
- `vaisalamet` - stacked 1D time-series plots for all retained variables, plus daily quicklooks.
- `asfs-logger` - stacked 1D time-series plots for all retained variables, plus daily quicklooks.
- `power` - stacked 1D time-series plots for all retained non-wind variables, plus daily quicklooks.
- `wxcam` - interactive daily HDR video browser for `FISH HDR` and `PANO HDR`, backed by a SQLite media catalog. The Calendar tab shows the current-day daily MP4 plus a past-day `3 x 8` grid of hourly video thumbnails.

`asfs-fast-sonic` is processed into its own Zarr store for downstream analysis, but it is not exposed in the dashboard UI.

## Core files

- `app.py` - main Panel application.
- `wxcam_catalog.py` - shared helpers for the wxcam SQLite catalog.
- `index_wxcam_catalog.py` - indexes local wxcam HDR images and videos, with optional remote bootstrap metadata.
- `build_wxcam_daily_videos.py` - builds daily wxcam MP4 products and hourly thumbnails from raw hourly clips.
- `append_new_wxcam_to_zarr.py` - forward append path for wxcam pixel Zarr groups. The service exists, but the timer is currently disabled.
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
  - `aurora-wxcam-append.timer` (installed, currently disabled)

Useful commands:

```bash
sudo systemctl status aurora-dashboard.service
sudo systemctl list-timers --all | rg '^.*aurora-'
sudo journalctl -u aurora-dashboard.service -f
```

## Running locally

```bash
cd /opt/aurora-cloud-dashboard
source venv/bin/activate
panel serve app.py --address 127.0.0.1 --port 5006 --allow-websocket-origin=<host>
```

## Notes

- Radar data currently contains at least one bogus far-future timestamp in the Zarr store. `app.py` filters clearly invalid future times when computing bounds and plotting windows so the interactive view stays usable.
- Wxcam keeps the full daily player on the Interactive tab. The Calendar tab now uses the current-day daily MP4 for `Today (latest)` and a past-day hourly thumbnail grid for historical browsing.
