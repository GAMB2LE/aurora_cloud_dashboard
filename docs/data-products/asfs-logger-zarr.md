# ASFS Logger Zarr

Path:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`

## Dataset shape

- dimension: `time`
- deployed shape when checked on `2026-05-09`:
  - `time=11234`

## Time coordinate

- `time` is parsed directly from the TOA5 `TIMESTAMP` column

## Useful root attributes

- `instrument = "asfs-logger"`
- `title = "ASFS LoggerNet science data"`
- `source = "asfs-logger_sci_DD_MM_YYYY.dat"`

## Variable layout

- one `float32` `time` series per retained source column
- the deployed store currently contains `34` variables

Examples include:

- `PTemp_Avg`
- `batt_volt_Avg`
- `amp_meter_48vdc_Avg`
- `kt15_amb_Avg`
- `kt15_tem_Avg`
- `licor_co2_out_Avg`
- `licor_h2o_out_Avg`
- `metek_x_out_Avg`
- `metek_T_out_Avg`
- `RECORD`

## Schema note

- append runs keep the existing variable set fixed in the same way as
  `vaisalamet`

## Chunking

- `time`-only variables are chunked `(1200,)`

## Display note

This Zarr underpins:

- the **Radiation** instrument directly
- parts of the **Meteorology** presentation layer
- the `HK_ASFS` housekeeping quicklooks
