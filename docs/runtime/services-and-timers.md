# Services And Timers

Systemd services are installed system-wide under `/etc/systemd/system/`.

## Dashboard

- `aurora-dashboard.service`

The deployed dashboard service runs Panel with websocket keepalives and a
short unused-session lifetime so mobile browsers do not leave large stale
documents behind on the single-process Panel server:

- `--keep-alive=15000`
- `--check-unused-sessions=60000`
- `--unused-session-lifetime=600000`
- `--session-token-expiration=86400`

This does not stop a mobile operating system from killing a background browser
tab. The app mirrors view state into the URL so a killed tab can reload into
the same tab, instrument, and key controls without retaining old server-side
documents for an hour.

The service also exposes camera media as static routes:

- `/wxcam-media` maps to `/data/aurora/products/wxcam`
- `/auroracam-media` maps to `/project/aurora/raw/auroracam`

WXcam MP4 playback and AURORACam JPEG display use those routes so media are
fetched by the browser over normal HTTP instead of being serialized into the
Panel websocket.

## Resource Tuning

The deployed host uses systemd drop-ins to keep the interactive dashboard
responsive when appenders, quicklook generators, mirror verification, and GWS
syncs overlap.

- `aurora-dashboard.service` gets higher CPU and IO weights plus a modest
  scheduling priority increase.
- Background processing services run in `aurora-batch.slice`, which caps the
  batch pool at two CPU cores on the current four-vCPU droplet and gives it
  lower CPU/IO weights. The slice also has a soft `MemoryHigh=6G` pressure
  limit so large products do not squeeze the dashboard as aggressively on the
  current 8 GB droplet.
- Heavier jobs such as appenders, quicklook generation, WXcam daily video
  builds, mirror verification, and GWS rsync get lower priority inside that
  batch slice.
- Append, quicklook, mirror, and rsync timers have randomized delays so they
  are less likely to start in one burst.
- The guarded runner `/usr/local/bin/aurora-run-guarded` adds lightweight
  mutexes for the heaviest job classes. Quicklook-heavy and video-heavy jobs
  run one at a time; append/rsync IO jobs allow two concurrent jobs. If a slot
  is already busy the generated `ExecStart` wrapper records a clean skip
  instead of letting heavy jobs pile up.
- Guard events are written to `/run/aurora/guarded/events.jsonl` and mirrored
  to `/data/aurora/products/ops_monitor/health/guarded_jobs.jsonl` when that
  directory is writable. The Operations Dashboard shows active guarded jobs,
  lock skips in the last 24 h, and batch slice memory pressure as diagnostic
  resource telemetry.

Install or refresh these drop-ins from the deployed checkout with:

```bash
sudo /opt/aurora-cloud-dashboard/deployment/bin/aurora-install-resource-tuning
sudo systemctl restart aurora-dashboard.service
```

Useful verification commands:

```bash
systemctl show aurora-dashboard.service -p Slice -p Nice -p CPUWeight -p IOWeight
systemctl show aurora-batch.slice -p CPUQuotaPerSecUSec -p CPUWeight -p IOWeight
systemctl show aurora-power-quicklooks.service -p Slice -p Nice -p CPUWeight -p IOWeight
systemctl cat aurora-power-quicklooks.service aurora-wxcam-daily-videos.service
systemctl list-timers --all 'aurora-*'
tail -n 20 /data/aurora/products/ops_monitor/health/guarded_jobs.jsonl
```

The dashboard restart applies its service priority immediately. Existing
background services inherit the batch slice only after their current run exits
and the timer starts them again.

Optional compressed swap can be installed separately on small droplets:

```bash
sudo /opt/aurora-cloud-dashboard/deployment/bin/aurora-install-zram
systemctl status aurora-zram-swap.service
swapon --show
```

The default zram size is `4G` with `zstd` compression and priority `100`.
Override `AURORA_ZRAM_SIZE`, `AURORA_ZRAM_ALGORITHM`, or
`AURORA_ZRAM_PRIORITY` in the service environment before starting it if the VM
size changes.

## CL61

- `aurora-cl61-source-sync.timer`
- `aurora-ceilometer-append.timer`
- `aurora-ceilometer-last24h.timer`
- `aurora-ceilometer-quicklooks.timer`

## Cloud Radar

- `aurora-radar-source-sync.timer`
- `aurora-radar-append.timer`
- `aurora-radar-quicklooks.timer`
- `aurora-radar-daily-quicklooks.timer`

`aurora-radar-quicklooks.service` is overridden on the deployed host to update
only the rolling latest 24 h radar PNG. The heavier daily archive generator
runs from `aurora-radar-daily-quicklooks.timer` instead, so a frequent
quicklook refresh cannot spend several minutes backfilling daily radar products
while operators are using the interactive dashboard.

Radar PNG rendering also uses display-only thinning controlled by
`AURORA_RADAR_QUICKLOOK_MAX_TIME_SAMPLES` and
`AURORA_RADAR_QUICKLOOK_MAX_RANGE_SAMPLES`. This reduces memory use for static
quicklooks without changing the underlying radar Zarr.

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
- `aurora-power-soc-forecast.timer`
- `aurora-power-soc-forecast-learn.timer`
- `aurora-power-soc-ensemble.timer`
- `aurora-power-quicklooks.timer`

`aurora-power-quicklooks.service` regenerates the compact APS display summary
and prewarmed Plotly JSON after the APS append cycle.
`aurora-power-soc-forecast.service` refreshes the ECMWF-informed SOC forecast
and adaptive forecast-skill state from a new ECMWF download every 3 hours.
`aurora-power-soc-forecast-learn.service` runs on a 15-minute timer, reuses the
latest cached ECMWF forecast, re-anchors to current SOC, scores archived
forecast runs, and updates skill/adaptive state faster than ECMWF is refreshed.
`aurora-power-soc-ensemble.service` checks hourly for a new ECMWF 00/12 UTC
ensemble cycle. New cycles retrieve all 50 perturbed `ssrd` members, write the
compact site ensemble and probabilistic verification products, then remove the
temporary global GRIB. It uses idle I/O scheduling and a two-hour timeout so it
does not block deterministic SOC learning.

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
