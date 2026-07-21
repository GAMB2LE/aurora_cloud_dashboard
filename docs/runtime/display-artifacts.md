# Display Artifact Manifest

The dashboard serves data products from local storage. It does not publish raw
files, SQLite catalogs, or Zarr stores through the display-artifact mechanism.

For the development site only, a small manifest is generated every five
minutes at:

`/data/aurora/products/dashboard/display_artifacts/latest.json`

It inventories browser-ready derived files:

- prewarmed Plotly JSON
- generated quicklook images
- WXcam hourly thumbnails
- WXcam daily videos

The manifest is written atomically. A reader sees either the previous complete
manifest or the new complete manifest, never a partial one. It is deliberately
bounded and excludes arbitrary files and Zarr chunks.

## Why this exists

The dashboard must not scan large historical media trees while opening a page.
The manifest gives a CDN or object-store publishing job a small, deterministic
input when that is introduced. It does not itself upload files or change the
current local-serving data path.

## Development service

`aurora-dashboard-display-manifest.timer` runs only when
`AURORA_SITE_ENV=development`. It is not installed or enabled for the
production role.

```bash
systemctl status aurora-dashboard-display-manifest.timer
journalctl -u aurora-dashboard-display-manifest.service --since '30 minutes ago' --no-pager
cat /data/aurora/products/dashboard/display_artifacts/latest.json
```

The mobile API exposes the same derived metadata at
`/mobile/v1/artifacts/manifest` for authenticated development tooling. The
endpoint is not an archive or data-download API.
