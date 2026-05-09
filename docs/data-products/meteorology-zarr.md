# Meteorology Zarr

Path:

- `/data/aurora/products/vaisalamet/vaisalamet.zarr`

## Dataset shape

- dimension: `time`
- deployed shape when checked on `2026-05-09`:
  - `time=114469`

## Time coordinate

- `time` is parsed from the raw `timestamp` column
- timestamps are localized as `Europe/London`
- timestamps are converted to UTC before storage

## Useful root attributes

- `instrument = "vaisalamet"`
- `title = "Vaisala met station data"`
- `source = "vaisala_met_level0_*.dat"`

## Variable layout

- one `float32` `time` series per retained source column
- the deployed store currently contains `68` variables

Examples include:

- `baro_hPa`
- `h1_ah`
- `h1_e`
- `h1_err_rh_meas_err`
- `h1_err_temp_meas_err`
- the various `*_err_*`, `*_dev_*`, and `*_st_*` health and status flags

## Schema note

- append runs align incoming files to the existing Zarr schema
- missing existing columns are filled with `NaN`
- newly appearing columns are dropped unless the store is rebuilt

## Chunking

- `time`-only variables are chunked `(1200,)`

## Display note

The dashboard's **Meteorology** instrument uses this Zarr as its main source,
but the final displayed summary also merges selected ASFS logger met traces at
presentation time.
