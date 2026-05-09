# WXcam

WXcam is the camera-oriented instrument in the dashboard. It uses a hybrid
product model rather than a purely numeric time-series Zarr.

## Interactive behavior

The interactive tab is video-driven:

- `Today (latest)` uses `latest.mp4`
- historical days use one stitched MP4 per UTC day
- the player is intentionally not auto-refreshed while you are actively
  browsing WXcam
- current-day stitched videos are refreshed ahead of the slower historical
  backfill work so the live browser stays useful during archive catch-up

## Science quicklooks

The WXcam science view is image-driven:

- each UTC hour uses the HDR JPG closest to `:30`
- the quicklook page shows a `3 x 8` hourly grid
- tiles only appear when an image exists for that hour
- long raw backfills do not pause the catalog or product refresh cycle; fresh
  in-flight files are deferred until they settle

## Housekeeping quicklooks

WXcam also has a dedicated housekeeping product:

- `HK_WXcam`

This focuses on catalog and coverage diagnostics rather than the primary image
grid.

## Backing products

- raw mirror: `/project/aurora/raw/wxcam`
- catalog: `/data/aurora/products/wxcam/wxcam_catalog.sqlite`
- daily videos: `/data/aurora/products/wxcam/daily_videos`
- hourly thumbnails:
  `/data/aurora/products/wxcam/hourly_thumbnails`
- image Zarr: `/data/aurora/products/wxcam/wxcam.zarr`

Detailed product documentation:

- [WXcam products](../data-products/wxcam-products.md)
- [WXcam Zarr](../data-products/wxcam-zarr.md)
