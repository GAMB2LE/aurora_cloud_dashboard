# Mobile API

The native iOS app uses a small read-only API instead of scraping the Panel
dashboard or reading Zarr/SQLite files directly from the phone.

The server-side API remains on the dashboard `main` branch. The native Swift
client is maintained as a separate iOS artifact/repository so dashboard
runtime changes and app UI changes can be reviewed independently. Do not
recreate an iOS feature branch in this repository; release the dashboard and
native client independently.

## Service

Install the Python dependencies into the dashboard virtual environment:

```bash
pip install -r requirements-mobile-api.txt
```

The API is served by `mobile_api.py`:

```bash
uvicorn mobile_api:app --host 127.0.0.1 --port 8010
```

For production, the Ansible dashboard release installs the service and proxies
it under both public dashboard hostnames. The bearer token is generated once in
the root-owned `/etc/aurora-mobile-api.token` file and referenced through
`AURORA_MOBILE_API_TOKEN_FILE`; it is not committed to the dashboard or
infrastructure repositories.

```text
https://data.gamb2le.co.uk/mobile/v1
```

## Authentication

All data and media endpoints require:

```text
Authorization: Bearer <AURORA_MOBILE_API_TOKEN>
```

`GET /health` is intentionally unauthenticated so the iOS app can report whether
the service is reachable and whether a token is configured. Local development
can set `AURORA_MOBILE_API_ALLOW_PUBLIC=1` to bypass auth.

## Endpoints

- `GET /health` - service reachability and auth configuration.
- `GET /manifest` - deployment identity, tabs, instruments, WXcam streams,
  refresh defaults, and the cross-platform capability contract. The contract
  distinguishes shared live capabilities from browser-only scientific
  exploration and native-only resilience workflows. Deployment identity
  reports the current site environment, public domain, data role, dashboard
  URL, and checked-out revision.
- `GET /operations` - latest operations health, stream states, root-cause
  groups, active alerts, and compact trend cards.
- `GET /overview` - the small first-load status cards and active alerts.
- `GET /power?window=24h|96h&group=...` - bounded native-chart traces from the
  existing Power display-summary Zarr product (at most 260 points per trace).
  `group=current` returns observed conditions; `group=forecast` returns the
  24-hour forecast, 96-hour forecast/planning, and verification panels. The
  legacy `all`, `observed`, `forecast_24h`, `forecast_96h`, and `verification`
  groups remain supported.
- `GET /auroracam?day=latest|YYYY-MM-DD&time_utc=...` - the latest four
  AURORACam records with separate preview and original URLs. Historical days
  also provide a bounded list of UTC frame times for native selection.
- `GET /uas?window=24h|7d|all` - latest UAS tier and a bounded history for
  the selected server-side window. `24h` is the default. `all` means all
  available records up to the newest 2,000, protecting mobile clients from an
  unbounded response.
- `GET /instruments/{id}/summary?window=24h|7d` - mobile instrument summary and
  latest generated quicklook references.
- `GET /quicklooks?kind=science|housekeeping&instrument={id}` - available
  quicklook dates and image URLs. For an assigned PDU instrument, the response
  also includes its current power state. Clients present an intentional
  power-off as an expected collection pause instead of a missing-data fault.
- `GET /wxcam?stream=fish_hdr|pano_hdr&day=latest|YYYY-MM-DD` - stitched MP4,
  day list, poster, and hourly thumbnails.
- `GET /media/...` - authenticated image/video file responses with short cache
  headers and ETag/304 revalidation. AURORACam previews are generated only on
  demand, are capped at 960 pixels, and use a bounded 50 MB server cache.

The API reads existing deployed products only. It does not restart services,
write Zarr stores, mutate the WXcam catalog, or change Panel dashboard behavior.
The deployment fields in the manifest are derived from the existing service
environment and Git checkout; they do not create or refresh a data product.
