# Meteorology

Meteorology is the curated 1D atmospheric summary view presented in the
dashboard.

## Display model

Meteorology is not a one-to-one reflection of a single source file layout. The
presentation layer combines:

- the **VaisalaMET** Zarr as the main atmospheric source
- selected **ASFS logger** met traces for wind and temperature context

This merge happens at display time only. The underlying Zarr stores remain
separate.

## Interactive summary layout

The Meteorology instrument currently groups:

- **Air Temperature**
  - HMP1
  - T2
  - Sonic temperature
  - KT15 ambient temperature
- **Humidity / Dew Point**
- **Pressure**
- **Met**
  - Metek `x` and `y` on the left axis
  - Metek `z` on the right axis

Axes use human-readable labels with units rather than raw field names.

## Quicklooks

- science quicklooks show the curated Meteorology summary
- housekeeping quicklooks show `HK_Met`

## Backing data products

- main Zarr: `/data/aurora/products/vaisalamet/vaisalamet.zarr`
- supporting met traces come from:
  `/data/aurora/products/asfs_logger/asfs_logger.zarr`

Detailed schema:

- [Meteorology Zarr](../data-products/meteorology-zarr.md)
- [ASFS Logger Zarr](../data-products/asfs-logger-zarr.md)
