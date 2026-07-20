# Radiation

Radiation is the curated surface-radiation view built from the ASFS logger Zarr.

## Interactive summary layout

The current Radiation instrument intentionally stays compact:

- **Shortwave Radiation**
  - SPN1 total and diffuse radiation
  - SR30 downwelling and upwelling shortwave radiation
- **Longwave Radiation**
  - IR20 downwelling and upwelling longwave radiation
- **Flux Plate**
  - flux plate A heat flux
  - flux plate B is intentionally hidden because its signal is noise
- **Surface Temperature**
  - KT15 surface temperature
  - SR50 distance on the right axis

This is a presentation-layer subset of the ASFS logger store rather than a
separate ingest path.

## Freshness note

Radiation depends on the ASFS slow `sci` table. The independent ASFS
fast-sonic and fast-gas streams can keep Metek wind and LI-COR housekeeping
plots current, but they do not contain SR30, IR20, flux-plate, KT15 surface, or
SR50 radiation/support fields. If the logger stops writing new
`aurora_asfs_data_sci_*.dat` files, Radiation will correctly stop at the last
available `sci` sample even while fast-sonic and fast-gas continue.

The latest interactive view can be prewarmed as Plotly JSON by
`generate_asfs_logger_quicklooks.py` under
`/data/aurora/products/dashboard/prewarm/`, so the first latest-view paint does
not have to rebuild the full figure.

## Quicklooks

- science quicklooks show the Radiation summary
- housekeeping quicklooks show `HK_ASFS`, a curated support layout for logger
  power, logger temperature/scan timing, ASFS met/LI-COR CO2/H2O output and
  signal strength, SR30 orientation/fans/heaters, IR20 support, and sensor
  variability. LI-COR housekeeping traces may be filled from the separate
  ASFS fast-gas Zarr when the ASFS science/logger file stream has a gap.

## Backing data product

Zarr path:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`

LI-COR housekeeping support:

- `/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr`

Detailed schema:

- [ASFS Logger Zarr](../data-products/asfs-logger-zarr.md)
- [ASFS Fast Gas Zarr](../data-products/asfs-fast-gas-zarr.md)
