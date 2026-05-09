# WXcam Zarr

Path:

- `/data/aurora/products/wxcam/wxcam.zarr`

## Store purpose

The WXcam Zarr contains HDR JPG image data only. MP4 products are stored
separately.

## Root attributes

- `instrument = "wxcam"`
- `title = "Aurora wxcam HDR images"`
- `storage_policy = "Contains locally retained HDR JPG image data for fish_hdr and pano_hdr with timestamps derived from filenames; MP4 products are stored separately."`

## Root groups

- `fish_hdr`
- `pano_hdr`

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

When checked on `2026-05-09`:

- `fish_hdr`
  - shape: `time=6606, y=3040, x=3120, channel=3`
  - image geometry: `3120 x 3040`
  - chunks: `(1, 1024, 1024, 3)`
- `pano_hdr`
  - shape: `time=6603, y=750, x=2880, channel=3`
  - image geometry: `2880 x 750`
  - chunks: `(1, 750, 1024, 3)`

## Important note

The dashboard does not render WXcam directly from this Zarr:

- interactive WXcam uses stitched MP4 products
- science WXcam uses the SQLite catalog plus hourly thumbnail products

The Zarr is the retained HDR image archive for analysis and reproducible storage.
