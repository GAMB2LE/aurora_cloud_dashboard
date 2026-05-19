# ASFS Fast Sonic Zarr

Path:

- `/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr`

## Dataset shape

- dimension: `time`
- deployed shape after the CRD parser update, checked on `2026-05-19`:
  - `time=2028871`
- time coverage when checked: `2026-05-02 00:00:30.027` to
  `2026-05-19 12:30:00`
- sorted unique `time` coordinate

## Time coordinate

- `time` is parsed from `TIMESTAMP`
- `metek_msec_out` is used to preserve sub-second timing
- both old daily LoggerNet files and newer chunked CRD files are supported

## Useful root attributes

- `instrument = "asfs-fast-sonic"`
- `title = "ASFS fast-sonic data"`
- `source = "asfs-logger_fast_sonic_DD_MM_YYYY.dat or aurora_asfs_data_fast_sonic_YYYYMMDDHHMM.dat"`

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
- the May 17, 2026 rebuild spans the old daily LoggerNet files and the newer
  CRD chunks; missing values for variables that do not exist in a given source
  format are stored as `NaN`
- append writes materialize only the already-filtered new sample block before
  writing, matching the cross-instrument Zarr append policy

## Chunking

- `time`-only variables are chunked `(24000,)`

## UI note

This Zarr remains part of the deployed data products and quicklook pipeline, but
ASFS fast-sonic is currently not exposed as a normal interactive or quicklook
instrument in the main dashboard selectors.
