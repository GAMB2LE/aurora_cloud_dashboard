# Zarr And Product Overview

The dashboard depends on a mix of numeric Zarr stores, camera media products,
and operations-monitor products.

The individual product pages describe schemas and may include dated example
sizes or time ranges. Treat those examples as historical documentation, not as
live health checks. Use the Operations Dashboard for current stream freshness
and the deployed runtime configuration for current paths.

## Numeric instrument Zarrs

- [Ceilometer Zarr](ceilometer-zarr.md)
- [Cloud Radar Zarr](cloud-radar-zarr.md)
- [HATPRO Zarr](hatpro-zarr.md)
- [Meteorology Zarr](meteorology-zarr.md)
- [ASFS Logger Zarr](asfs-logger-zarr.md)
- [ASFS Fast Sonic Zarr](asfs-fast-sonic-zarr.md)
- [ASFS Fast Gas Zarr](asfs-fast-gas-zarr.md)
- [Power Zarr](power-zarr.md)

## WXcam products

- [WXcam products](wxcam-products.md)
- [WXcam Zarr](wxcam-zarr.md)

## AURORACam products

- [AURORACam Zarr](auroracam-zarr.md)

## Operations products

- [Operations products](operations-products.md)

## Design pattern

The general rule is:

- raw mirrored inputs live under `/project/aurora/raw`
- dashboard-facing products live under `/data/aurora/products`

For most numeric instruments, that means a fixed-schema Zarr plus generated PNG
quicklooks. The fixed-summary 1D instruments can also write prewarmed latest
interactive Plotly JSON under `/data/aurora/products/dashboard/prewarm/`.
WXcam adds media/catalog products on top of that pattern. AURORACam adds a raw
JPEG browser plus a small metadata Zarr. Operations Dashboard adds raw JSONL
snapshots, a monitoring Zarr, archived PNG quicklooks, live trend cards, and
observe-only health reports.

Numeric appenders use a common safe-append policy: incoming files are sorted,
deduplicated, filtered to genuinely new timestamps, and materialized before the
Zarr append. That keeps the deployed stores monotonic and avoids partial chunk
writes that can otherwise appear as false data gaps or all-NaN stripes.
