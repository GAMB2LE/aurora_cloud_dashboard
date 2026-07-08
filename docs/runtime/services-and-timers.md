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

The service also exposes camera media as static routes:

- `/wxcam-media` maps to `/data/aurora/products/wxcam`
- `/auroracam-media` maps to `/project/aurora/raw/auroracam`

WXcam MP4 playback and AURORACam JPEG display use those routes so media are
fetched by the browser over normal HTTP instead of being serialized into the
Panel websocket.

## CL61

- `aurora-cl61-source-sync.timer`
- `aurora-ceilometer-append.timer`
- `aurora-ceilometer-last24h.timer`
- `aurora-ceilometer-quicklooks.timer`

## Cloud Radar

- `aurora-radar-source-sync.timer`
- `aurora-radar-append.timer`
- `aurora-radar-quicklooks.timer`

## Scanning Microwave Radiometer

- `aurora-hatpro-source-sync.timer`
- `aurora-hatpro-append.timer`
- `aurora-hatpro-quicklooks.timer`

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

## ASFS Fast Gas

- `aurora-asfs-fast-gas-source-sync.timer`
- `aurora-asfs-fast-gas-append.timer`

Fast-gas is the high-rate LI-COR/gas file family from the ASFS logger CRD
area. It is stored in its own Zarr and is merged into `HK_ASFS` housekeeping
quicklooks for LI-COR continuity. It does not contain radiation variables.

## Power

- `aurora-power-source-sync.timer`
- `aurora-power-append.timer`
- `aurora-power-quicklooks.timer`

`aurora-power-quicklooks.service` regenerates the compact APS display summary
and prewarmed Plotly JSON after the APS append cycle.

## ASS PDU

- `aurora-pdu-source-sync.timer`
- `aurora-pdu-append.timer`

The source sync pulls ASS Linux `/home/aurora/data/pdu/pdu_DDMMYYYY.csv` files
into `/project/aurora/raw/pdu`. The appender writes
`/data/aurora/products/power/pdu.zarr`, which is folded into the APS display
summary by the Power quicklook pipeline when available.

## WXcam

- `aurora-wxcam-source-sync.timer`
- `aurora-wxcam-catalog.timer`
- `aurora-wxcam-daily-videos.timer`
- `aurora-wxcam-append.timer`

## AURORACam

- `aurora-auroracam-source-sync.timer`
- `aurora-auroracam-index.timer`

## Operations

- `aurora-ops-monitor-collect.timer`
- `aurora-ops-monitor-append.timer`
- `aurora-ops-monitor-alerts.timer`
- `aurora-ops-monitor-quicklooks.timer`
- `aurora-mirror-verify.timer`

`aurora-ops-monitor-collect.timer` is observe-only. It writes raw JSONL
snapshots under `/project/aurora/raw/ops_monitor` and compact health outputs
under `/data/aurora/products/ops_monitor/health`; it does not restart services,
delete files, rebuild data products, or change code.

`aurora-ops-monitor-alerts.timer` evaluates the latest operations snapshot
after collection and sends threshold email alerts through `mailx` backed by an
outbound relay such as `msmtp`.

## JASMIN GWS Sync

- `aurora-gws-rsync-raw.timer`
- `aurora-gws-rsync-products.timer` for non-WXcam products
- `aurora-gws-rsync-products-wxcam.timer` for the larger WXcam product tree
- `aurora-gws-rsync-manifests.timer`

The products sync is intentionally split so the large WXcam media/Zarr tree can
run independently from the smaller Zarr and quicklook products.

The mirror verifier compares HATPRO files by basename. That is deliberate:
newer HATPRO source paths are arranged under dated `Y2026/Mxx/Dxx`
directories, while older local and GWS mirrors include a legacy flat layout.
Basename comparison keeps the prune/coverage audit focused on whether the
files are present without forcing a risky raw-data reshuffle.

The HATPRO source sync also uses a three-day mtime lookback. HATPRO files can
arrive after the previous sync while retaining an older file timestamp, and the
lookback prevents those late files from being skipped permanently.

## Useful commands

```bash
sudo systemctl status aurora-dashboard.service
sudo systemctl list-timers --all | rg '^.*aurora-'
sudo journalctl -u aurora-dashboard.service -f
```

## Manual Product Regeneration

The deployed systemd services load `/etc/aurora-dashboard.env` before running
appenders and quicklook generators. That environment file points products at
the live dashboard tree, including:

- `AURORA_QUICKLOOK_ROOT=/data/aurora/products/quicklooks`
- `AURORA_INTERACTIVE_PREWARM_DIR=/data/aurora/products/dashboard/prewarm`

When running a generator manually on the deployed host, source that environment
first so the output lands where the dashboard reads it:

```bash
cd /opt/aurora-cloud-dashboard
set -a
source /etc/aurora-dashboard.env
set +a
source venv/bin/activate
./generate_asfs_logger_quicklooks.py
```

Without the environment file, several generators intentionally fall back to a
repo-local `quicklooks/` directory for development. That is useful for local
tests, but it does not update the live dashboard quicklook shown in the web
app.
