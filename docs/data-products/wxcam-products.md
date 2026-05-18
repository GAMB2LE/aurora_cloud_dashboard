# WXcam Products

WXcam uses a richer product model than the purely numeric instruments.

## Local raw mirror

The deployed raw mirror retains only FISH HDR JPG and MP4 files locally:

- `/project/aurora/raw/wxcam/FISH`

PANO and AUTO/LONG/SHORT files remain on the camera host and are not cataloged
or archived from this VM.

## Catalog

The WXcam catalog lives at:

- `/data/aurora/products/wxcam/wxcam_catalog.sqlite`

It indexes FISH HDR JPGs and FISH HDR MP4s. Timestamps are derived from
filenames and stored as UTC.

Key fields include:

- `image_type` - `fish_hdr`
- `media_kind` - `image` or `video`
- `time_utc`, `time_epoch_ns`, `day_utc`
- `raw_path`, `relative_path`, `filename`
- `width`, `height`, `size_bytes`

## Daily videos and hourly thumbnails

- daily MP4s:
  `/data/aurora/products/wxcam/daily_videos/<image_type>/YYYYMMDD.mp4`
- rolling latest MP4s:
  `/data/aurora/products/wxcam/daily_videos/<image_type>/latest.mp4`
- hourly thumbnails:
  `/data/aurora/products/wxcam/hourly_thumbnails/<image_type>/YYYYMMDD/`

Daily videos are stitched from the 24 hourly MP4 clips for that UTC day.
`latest.mp4` is stitched from the most recent 24 hourly clips across day
boundaries.

The dashboard serves these MP4 files through the static route
`/wxcam-media/daily_videos/...` with a file-mtime query string. This avoids
sending tens of megabytes of video data through the Panel websocket and allows
browser video controls to use HTTP range requests.

## Dashboard usage

- **Interactive Data Browser** uses the stitched MP4 products
- **Science Quicklooks** uses the HDR JPG grid, selecting the image nearest the
  `:30` mark in each UTC hour
- **House Keeping Quicklooks** uses the catalog and coverage diagnostics

## Related page

- [WXcam Zarr](wxcam-zarr.md)
