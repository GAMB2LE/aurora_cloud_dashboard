# ASFS Fast Sonic Zarr

Path:

- `/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr`

## Dataset shape

- dimension: `time`
- deployed shape when checked on `2026-05-09`:
  - `time=1087554`

## Time coordinate

- `time` is parsed from `TIMESTAMP`
- `metek_msec_out` is used to preserve sub-second timing

## Useful root attributes

- `instrument = "asfs-fast-sonic"`
- `title = "ASFS LoggerNet fast-sonic data"`
- `source = "asfs-logger_fast_sonic_DD_MM_YYYY.dat"`

## Variable layout

The deployed store currently contains these 10 variables:

- `RECORD`
- `metek_InclX_out`
- `metek_InclY_out`
- `metek_T_out`
- `metek_msec_out`
- `metek_quality_out`
- `metek_senspathstate_out`
- `metek_x_out`
- `metek_y_out`
- `metek_z_out`

## Schema note

- append runs keep the existing variable set fixed in the same way as
  `asfs-logger`

## Chunking

- `time`-only variables are chunked `(24000,)`

## UI note

This Zarr remains part of the deployed data products and quicklook pipeline, but
ASFS fast-sonic is currently not exposed as a normal interactive or quicklook
instrument in the main dashboard selectors.
