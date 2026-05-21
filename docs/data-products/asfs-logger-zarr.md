# ASFS Science Zarr

Path:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`

## Dataset shape

- dimension: `time`
- deployed shape after the CRD parser update, checked on `2026-05-20`:
  - `time=21598`
- time coverage when checked: `2026-05-02 00:00:00` to
  `2026-05-20 07:30:00`
- sorted unique `time` coordinate

## Time coordinate

- `time` is parsed directly from the TOA5 `TIMESTAMP` column
- both old daily LoggerNet files and newer chunked CRD files are supported

## Useful root attributes

- `instrument = "asfs-logger"`
- `title = "ASFS science data"`
- `source = "asfs-logger_sci_DD_MM_YYYY.dat or aurora_asfs_data_sci_YYYYMMDDHHMM.dat"`

## Variable layout

- one `float32` `time` series per retained source column
- the deployed store currently contains `89` variables

Examples include:

- `PTemp_Avg`
- `batt_volt_Avg`
- `amp_meter_48vdc_Avg`
- `vaisala_T_Avg`
- `vaisala_RH_Avg`
- `vaisala_P_Avg`
- `kt15_amb_Avg`
- `kt15_tem_Avg`
- `licor_co2_out_Avg`
- `licor_h2o_out_Avg`
- `licor_co2_str_out_Avg`
- `licor_t_out_Avg`
- `metek_x_out_Avg`
- `metek_T_out_Avg`
- `sr30_swd_Irr_Avg`
- `sr30_swu_Irr_Avg`
- `sr30_swd_tilt_Avg`
- `sr30_swu_tilt_Avg`
- `sr30_swd_fantach_Avg`
- `sr30_swu_fantach_Avg`
- `sr30_swd_heatstate_Avg`
- `sr30_swu_heatstate_Avg`
- `ir20_lwd_Wm2_Avg`
- `ir20_lwu_Wm2_Avg`
- `ir20_lwd_fan_Avg`
- `ir20_lwu_fan_Avg`
- `fp_A_Wm2_Avg`
- `fp_B_Wm2_Avg`
- `sr50_dist_Avg`
- `RECORD`

## Schema note

- append runs keep the existing variable set fixed in the same way as
  `vaisalamet`
- the May 17, 2026 rebuild includes both older LoggerNet variables and newer
  CRD variables; missing values for variables that do not exist in a given
  source format are stored as `NaN`
- the appender keeps the current variable set fixed between deliberate rebuilds
- append writes materialize only the already-filtered new sample block before
  writing, matching the cross-instrument Zarr append policy

## Chunking

- `time`-only variables are chunked `(1200,)`

## Display note

This Zarr underpins:

- the **Radiation** instrument directly
- parts of the **Meteorology** presentation layer, including ASFS Vaisala
  temperature, relative humidity, pressure, Metek wind vectors, and
  display-derived Metek wind speed/direction
- the curated `HK_ASFS` housekeeping quicklooks for logger power, logger
  thermal/scan state, ASFS met/LI-COR CO2/H2O output and CO2 signal strength,
  SR30 support, IR20 support, and sensor variability

LI-COR housekeeping can also be filled from the separate ASFS fast-gas Zarr
when the slower ASFS science/logger file stream has a gap. The fast-gas store
includes high-rate `licor_diag_out` samples, so `HK_ASFS` can show LI-COR
diagnostic continuity even when `diag_out` is not present in the science Zarr.
