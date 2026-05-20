# Radiation

Radiation is the curated surface-radiation view built from the ASFS logger Zarr.

## Interactive summary layout

The current Radiation instrument intentionally stays compact:

- **Shortwave Radiation**
  - SPN1 total and diffuse radiation
  - SR30 downwelling and upwelling shortwave radiation
- **Longwave Radiation**
  - IR20 downwelling and upwelling longwave radiation
- **Flux Plates**
  - flux plate A and B heat flux
- **Surface Temperature**
  - KT15 surface temperature
  - SR50 distance on the right axis

This is a presentation-layer subset of the ASFS logger store rather than a
separate ingest path.

The latest interactive view can be prewarmed as Plotly JSON by
`generate_asfs_logger_quicklooks.py` under
`/data/aurora/products/dashboard/prewarm/`, so the first latest-view paint does
not have to rebuild the full figure.

## Quicklooks

- science quicklooks show the Radiation summary
- housekeeping quicklooks show `HK_ASFS`, a curated support layout for logger
  power, logger temperature/scan timing, ASFS met/LI-COR CO2/H2O output and
  signal strength, SR30 orientation/fans/heaters, IR20 support, and sensor
  variability

## Backing data product

Zarr path:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`

Detailed schema:

- [ASFS Logger Zarr](../data-products/asfs-logger-zarr.md)
