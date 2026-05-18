# Power Zarr

Path:

- `/data/aurora/products/power/power.zarr`

## Dataset shape

- dimension: `time`
- deployed shape when checked on `2026-05-18`:
  - `time=842870`

## Time coordinate

- `time` is parsed from the raw `aps_time` column

## Useful root attributes

- `instrument = "power"`
- `title = "Power level1 data"`
- `source = "power_data_YYYYMMDD.csv"`
- `wind_columns_excluded = "true"`

## Variable layout

- one `float32` `time` series per retained source column
- the deployed store currently contains `43` variables

Important ingest rules:

- raw column names are normalized by replacing `.` with `_`
- columns containing `wind` are excluded
- columns ending in `time` are excluded

Examples include:

- `ACOutputAmps`
- `ACOutputHZ`
- `ACOutputVolts`
- `ACOutputWatts`
- `BatteryAmps`
- `BatteryState`
- `BatteryWatts`
- `DCInverterWatts`
- `InternalTemperature`
- `HeatsinkTemperature`
- `TempSensor1`
- `TempSensor2`
- `TempSensor3`
- `TempSensor4`
- `MaxSolarWatts_East`

## Schema note

- append runs keep the existing variable set fixed in the same way as
  `vaisalamet`

## Chunking

- `time`-only variables are chunked `(1200,)`
