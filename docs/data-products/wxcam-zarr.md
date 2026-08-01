# WXcam Zarr

Path:

- `/data/aurora/products/wxcam/wxcam.zarr`

The active WXcam pixel Zarr is a local mutable working product. It is excluded
from both GWS and object-store product writers because it can be rebuilt from
the archived HDR JPG imagery. It is not a canonical backup and is never
accepted as raw-retention evidence.

## Store purpose

The WXcam Zarr contains HDR JPG image data only. MP4 products are stored
separately.

The active store starts at `2026-07-04T00:00:00Z`; earlier WXcam media are
left in raw/catalog products and are not decoded into this Zarr.

## Root attributes

- `instrument = "wxcam"`
- `title = "Aurora wxcam HDR images"`
- `storage_policy = "Contains locally retained FISH HDR and PANO HDR JPG image data with timestamps derived from filenames; MP4 products are stored separately."`

## Root groups

- `fish_hdr`
- `pano_hdr`

The root Zarr group is only a container. Open one of these child groups to read
image data and timestamps.

## Group dataset structure

Each group stores one xarray dataset with:

- dimensions: `time`, `y`, `x`, `channel`
- coordinates:
  - `time` - UTC image timestamps
  - `y` - pixel row index
  - `x` - pixel column index
  - `channel` - RGB labels: `R`, `G`, `B`
- data variables:
  - `image[time, y, x, channel]` - `uint8` RGB pixel data
  - `filename[time]`
  - `width[time]`
  - `height[time]`
  - `size_bytes[time]`

## Group geometry

Expected geometry is `3120 x 3040` for `fish_hdr` and `2880 x 750` for
`pano_hdr`. The time count and coverage advance as the local appender runs and
must be inspected live rather than copied from an old GWS snapshot.

## Important note

The dashboard does not render WXcam directly from this Zarr:

- interactive WXcam uses stitched MP4 products
- science WXcam uses the SQLite catalog plus hourly thumbnail products

The retained source archive is the immutable FISH/PANO HDR JPG and MP4 tree in
the cloud raw mirror, GWS, and object storage. The SQLite catalogue, daily
videos, and hourly thumbnails are separately archived products.
