# Radiation

Radiation is the curated surface-radiation view built from the ASFS logger Zarr.

## Interactive summary layout

The current Radiation instrument intentionally stays compact:

- **Radiation**
  - Total radiation
  - Diffuse radiation
- **Surface Temperature**
  - KT15 surface temperature

This is a presentation-layer subset of the ASFS logger store rather than a
separate ingest path.

## Quicklooks

- science quicklooks show the Radiation summary
- housekeeping quicklooks show `HK_ASFS`

## Backing data product

Zarr path:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`

Detailed schema:

- [ASFS Logger Zarr](../data-products/asfs-logger-zarr.md)
