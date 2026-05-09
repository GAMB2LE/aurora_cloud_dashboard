# Dashboard At A Glance

The Aurora Cloud Dashboard is the public-facing data browser and product viewer
for the Aurora observing stack. It combines vertically resolved remote-sensing
instruments, curated 1D station summaries, WXcam imagery and video products,
and an operational status view into one Panel app.

## Main dashboard views

The deployed interface is organized into four top-level tabs:

- **Interactive Data Browser** for live browsing and recent windows
- **Science Quicklooks** for archived science products and WXcam day views
- **House Keeping Quicklooks** for archived diagnostics and housekeeping plots
- **Operations Dashboard** for source-host status, storage, transfers, mirror
  verification, and prune-readiness checks

## Instrument families

- **Ceilometer** and **Cloud Radar** render height-time plots from Zarr stores.
- **Meteorology**, **Radiation**, and **Aurora Power Supply** render curated
  multi-panel 1D summaries from fixed Zarr schemas.
- **WXcam** combines a SQLite catalog, stitched MP4 products, and an HDR image
  Zarr.
- **Operations Dashboard** is driven by the `ops_monitor` raw snapshots and
  monitoring Zarr.

## Key product roots

- raw mirror root: `/project/aurora/raw`
- product root: `/data/aurora/products`
- quicklooks root: `/data/aurora/products/quicklooks`

For the full storage breakdown, see [Storage layout](../runtime/storage-layout.md).
