# ASFS Fast Gas Zarr

Path:

- `/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr`

## Purpose

The ASFS fast-gas Zarr stores the high-rate LI-COR/gas stream from the ASFS
logger CRD output. It is separate from the ASFS science/logger Zarr because the
source files are a different table and cadence.

This product is used to improve `HK_ASFS` housekeeping continuity for LI-COR
variables. It does not contain radiation sensors, so Radiation science
quicklooks still depend on the ASFS science/logger Zarr.

## Source files

Supported raw file names are:

- `asfs-logger_fast_gas_DD_MM_YYYY.dat`
- `aurora_asfs_data_fast_gas_YYYYMMDDHHMM.dat`

The deployed mirror pulls CRD fast-gas files from:

- source: `/home/aurora/data/asfs/raw/crd` on `100.124.55.22`
- local raw mirror: `/project/aurora/raw/asfs/crd`

## Dataset shape

- dimension: `time`
- sorted unique `time` coordinate
- `time` is parsed from the TOA5 `TIMESTAMP` column

CR1000X fast-gas files record bursts of samples under repeated logger
timestamps. During append, samples in each repeated-timestamp block are spread
within the interval before the next block. This keeps sample order, gives the
Zarr a useful monotonic time coordinate, and avoids duplicate-time appends.

## Useful root attributes

- `instrument = "asfs-fast-gas"`
- `title = "ASFS fast-gas data"`
- `source = "asfs-logger_fast_gas_DD_MM_YYYY.dat or aurora_asfs_data_fast_gas_YYYYMMDDHHMM.dat"`

## Variable layout

The store keeps one `float32` `time` series per retained source column.
Typical variables include:

- `RECORD`
- `licor_time_out`
- `licor_co2_out`
- `licor_h2o_out`
- `licor_pr_out`
- `licor_t_out`
- `licor_diag_out`
- `licor_co2_str_out`

## Display note

`generate_asfs_logger_quicklooks.py` resamples this store to one-minute
summary variables and merges those into ASFS housekeeping. The derived display
names use the same `_Avg` convention as the ASFS science/logger Zarr:

- `licor_co2_out_Avg`
- `licor_h2o_out_Avg`
- `licor_pr_out_Avg`
- `licor_t_out_Avg`
- `licor_diag_out_Avg`
- `licor_co2_str_out_Avg`

This merge affects housekeeping plots only. It does not change ASFS ingest,
the original ASFS science/logger Zarr, or the Radiation science quicklooks.

## Services

- `aurora-asfs-fast-gas-source-sync.timer` mirrors the raw CRD fast-gas files
  into `/project/aurora/raw/asfs/crd`
- `aurora-asfs-fast-gas-append.timer` appends new raw fast-gas samples into
  `/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr`
