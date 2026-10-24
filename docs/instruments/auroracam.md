# AURORACam

AURORACam is the dashboard view for the active MOBOTIX M24 cameras writing to
the ASS Linux MX4 FTP ingest. Three cameras are currently displayed.

## Tab behavior

The top-level `AURORACam` tab is image-driven:

- the selected day shows the latest frame from the three active cameras
- the camera selector controls the large image viewer
- the selected camera also shows a UTC hourly still strip for quick scanning
- `Latest`, `Previous`, and `Next` move through available days
- share links preserve the selected camera and date

JPEGs are served through `/auroracam-media/...` so full-resolution images load
through normal HTTP image requests instead of being serialized into the Panel
websocket.

## Cameras

| Camera | IP | Display status |
| --- | --- | --- |
| `end-south-array-cam` | `192.168.1.27` | Active |
| `fence-post-cam` | `192.168.1.28` | Active |
| `radar-cam` | `192.168.1.29` | Active |
| `mid-south-array-cam` | `192.168.1.30` | Offline; archive retained |

Mid South Array is broken and was taken offline. It is excluded from the web
dashboard and the iOS/Android camera listings, including historical date/time
selectors. The full camera registry, JPEG files, metadata indexing and direct
historical media paths are retained; no JASMIN archive data is deleted or
excluded from archive processing. Installed native apps use the API listing and
pick up this display change on their next successful camera refresh.

## Backing products

- raw mirror: `/project/aurora/raw/auroracam`
- metadata Zarr: `/data/aurora/products/auroracam/auroracam.zarr`
- source-side FTP ingest:
  `/home/aurora/data/mx4/<camera>/YYYY-MM-DD/<camera>_YYYY-MM-DD_HH-MM.jpg`

The Zarr is a metadata index of files, camera IDs, timestamps, sizes, and
mtimes. It deliberately does not duplicate raw QXGA image pixels.

Detailed product documentation:

- [AURORACam Zarr](../data-products/auroracam-zarr.md)
