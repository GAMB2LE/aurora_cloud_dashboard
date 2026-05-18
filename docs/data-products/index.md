# Zarr And Product Overview

The dashboard depends on a mix of numeric Zarr stores, WXcam media products,
and operations-monitor products.

## Numeric instrument Zarrs

- [Ceilometer Zarr](ceilometer-zarr.md)
- [Cloud Radar Zarr](cloud-radar-zarr.md)
- [HATPRO Zarr](hatpro-zarr.md)
- [Meteorology Zarr](meteorology-zarr.md)
- [ASFS Logger Zarr](asfs-logger-zarr.md)
- [ASFS Fast Sonic Zarr](asfs-fast-sonic-zarr.md)
- [Power Zarr](power-zarr.md)

## WXcam products

- [WXcam products](wxcam-products.md)
- [WXcam Zarr](wxcam-zarr.md)

## Operations products

- [Operations products](operations-products.md)

## Design pattern

The general rule is:

- raw mirrored inputs live under `/project/aurora/raw`
- dashboard-facing products live under `/data/aurora/products`

For most numeric instruments, that means a fixed-schema Zarr plus generated PNG
quicklooks. WXcam adds media/catalog products on top of that pattern, while
Operations Dashboard adds raw JSONL snapshots, a monitoring Zarr, archived PNG
quicklooks, and observe-only health reports.
