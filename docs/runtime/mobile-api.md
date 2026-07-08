# Mobile API

The native iOS app uses a small read-only API instead of scraping the Panel
dashboard or reading Zarr/SQLite files directly from the phone.

The server-side API remains on the dashboard `main` branch. The native Swift
client is maintained separately on the `codex/ios-app` branch so dashboard
runtime changes and app UI changes can be reviewed independently.

## Service

Install the Python dependencies into the dashboard virtual environment:

```bash
pip install -r requirements-mobile-api.txt
```

The API is served by `mobile_api.py`:

```bash
uvicorn mobile_api:app --host 127.0.0.1 --port 8010
```

For production, install `systemd/aurora-mobile-api.service`, set
`AURORA_MOBILE_API_TOKEN` in `/etc/aurora-dashboard.env`, and proxy the service
under:

```text
https://data-ocean.gamb2le.co.uk/mobile/v1
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
- `GET /manifest` - tabs, instruments, WXcam streams, and refresh defaults.
- `GET /operations` - latest operations health, stream states, root-cause
  groups, active alerts, and compact trend cards.
- `GET /instruments/{id}/summary?window=24h|7d` - mobile instrument summary and
  latest generated quicklook references.
- `GET /quicklooks?kind=science|housekeeping&instrument={id}` - available
  quicklook dates and image URLs.
- `GET /wxcam?stream=fish_hdr|pano_hdr&day=latest|YYYY-MM-DD` - stitched MP4,
  day list, poster, and hourly thumbnails.
- `GET /media/...` - authenticated image/video file responses with short cache
  headers.

The API reads existing deployed products only. It does not restart services,
write Zarr stores, mutate the WXcam catalog, or change Panel dashboard behavior.
