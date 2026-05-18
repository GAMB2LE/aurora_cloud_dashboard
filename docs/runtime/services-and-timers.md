# Services And Timers

Systemd services are installed system-wide under `/etc/systemd/system/`.

## Dashboard

- `aurora-dashboard.service`

The deployed dashboard service runs Panel with websocket keepalives and a
longer unused-session lifetime:

- `--keep-alive=15000`
- `--check-unused-sessions=600000`
- `--unused-session-lifetime=3600000`
- `--session-token-expiration=86400`

This does not stop a mobile operating system from killing a background browser
tab, but it gives short backgrounding events a better chance of reconnecting.
The app also mirrors view state into the URL so a killed tab can reload into
the same tab, instrument, and key controls.

The service also exposes `/data/aurora/products/wxcam` as the static route
`/wxcam-media`. WXcam MP4 playback uses that route so videos are fetched by the
browser over normal HTTP instead of being serialized into the Panel websocket.

## CL61

- `aurora-cl61-source-sync.timer`
- `aurora-ceilometer-append.timer`
- `aurora-ceilometer-last24h.timer`
- `aurora-ceilometer-quicklooks.timer`

## Cloud Radar

- `aurora-radar-source-sync.timer`
- `aurora-radar-append.timer`
- `aurora-radar-quicklooks.timer`

## Meteorology (VaisalaMET)

- `aurora-vaisalamet-source-sync.timer`
- `aurora-vaisalamet-append.timer`
- `aurora-vaisalamet-quicklooks.timer`

## ASFS Logger

- `aurora-asfs-logger-source-sync.timer`
- `aurora-asfs-logger-append.timer`
- `aurora-asfs-logger-quicklooks.timer`

## ASFS Fast Sonic

- `aurora-asfs-fast-sonic-source-sync.timer`
- `aurora-asfs-fast-sonic-append.timer`
- `aurora-asfs-fast-sonic-quicklooks.timer`

## Power

- `aurora-power-source-sync.timer`
- `aurora-power-append.timer`
- `aurora-power-quicklooks.timer`

## WXcam

- `aurora-wxcam-source-sync.timer`
- `aurora-wxcam-catalog.timer`
- `aurora-wxcam-daily-videos.timer`
- `aurora-wxcam-append.timer`

## Operations

- `aurora-ops-monitor-collect.timer`
- `aurora-ops-monitor-append.timer`
- `aurora-ops-monitor-quicklooks.timer`
- `aurora-mirror-verify.timer`

`aurora-ops-monitor-collect.timer` is observe-only. It writes raw JSONL
snapshots under `/project/aurora/raw/ops_monitor` and compact health outputs
under `/data/aurora/products/ops_monitor/health`; it does not restart services,
delete files, rebuild data products, or change code.

## JASMIN GWS Sync

- `aurora-gws-rsync-raw.timer`
- `aurora-gws-rsync-products.timer` for non-WXcam products
- `aurora-gws-rsync-products-wxcam.timer` for the larger WXcam product tree
- `aurora-gws-rsync-manifests.timer`

The products sync is intentionally split so the large WXcam media/Zarr tree can
run independently from the smaller Zarr and quicklook products.

## Useful commands

```bash
sudo systemctl status aurora-dashboard.service
sudo systemctl list-timers --all | rg '^.*aurora-'
sudo journalctl -u aurora-dashboard.service -f
```
