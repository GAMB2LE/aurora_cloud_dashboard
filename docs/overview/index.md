# Dashboard At A Glance

The Aurora Cloud Dashboard is the public-facing data browser and product viewer
for the Aurora observing stack. It combines vertically resolved remote-sensing
instruments, curated 1D station summaries, WXcam imagery/video products,
AURORACam still images, and an operational status view into one Panel app.

## Main dashboard views

The deployed desktop interface has eight top-level tabs:

- **Overview** for a compact station snapshot and instrument-state grouping
- **Interactive Data Browser** for live browsing and recent windows
- **Power** for the dedicated Aurora Power Supply view and forecasts
- **Science Quicklooks** for archived science products and WXcam day views
- **House Keeping Quicklooks** for archived diagnostics and housekeeping plots
- **AURORACam** for four-camera MX4 still-image browsing
- **UAS** for the mirrored Menapia tier log and recent event history
- **Operations Dashboard** for source-host status, storage, transfers, mirror
  verification, and prune-readiness checks

The desktop default is **Overview**. The phone experience groups the same
content into **Overview**, **Power**, **Plots**, **Camera**, and **Ops** so the
most frequent field checks remain one tap away.

## Instrument families

- **Ceilometer** and **Cloud Radar** render height-time plots from Zarr stores.
- **Meteorology**, **Radiation**, and **Aurora Power Supply** render curated
  multi-panel 1D summaries from fixed Zarr schemas. Their latest interactive
  views can start from prewarmed Plotly JSON created by the quicklook
  generators.
- **WXcam** combines a SQLite catalog, stitched MP4 products, hourly
  representative thumbnails, and an HDR image Zarr. The deployed product
  streams are currently FISH HDR and PANO HDR.
- **AURORACam** reads the four MOBOTIX M24 JPEG folders from the MX4 raw mirror
  and uses a small metadata Zarr for the rebuilt file index.
- **Operations Dashboard** is driven by the `ops_monitor` raw snapshots and
  monitoring Zarr, plus the observe-only health JSON/Markdown reports written
  by the Phase 1 sentinel.

## Key product roots

- raw mirror root: `/project/aurora/raw`
- product root: `/data/aurora/products`
- quicklooks root: `/data/aurora/products/quicklooks`
- interactive prewarm root: `/data/aurora/products/dashboard/prewarm`

For the full storage breakdown, see [Storage layout](../runtime/storage-layout.md).
