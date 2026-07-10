# Power Zarr

Path:

- `/data/aurora/products/power/power.zarr`

## Dataset shape

- dimension: `time`
- deployed shape when checked on `2026-05-21`:
  - `time=1118886`
- time coverage when checked: `2026-05-05 15:15:23.598658936` to
  `2026-05-21 20:15:04`
- sorted unique `time` coordinate

## Time coordinate

- `time` is parsed from the raw `aps_time` column

## Useful root attributes

- `instrument = "power"`
- `title = "Power level1 data"`
- `source = "power_data_YYYYMMDD.csv"`
- `wind_columns_excluded = "true"`

## Variable layout

- one `float32` `time` series per retained source column
- the deployed store currently contains `62` variables

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
- `BatterySOC`
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
- append writes materialize only the already-filtered new sample block before
  writing, matching the cross-instrument Zarr append policy

## Chunking

- `time`-only variables are chunked `(1200,)`

## Dashboard performance note

The stored Zarr schema and chunking remain unchanged. The dashboard opens this
store with larger read chunks for interactive plotting, applies display-only
sanity limits for impossible APS values, uses per-trace time downsampling, and
rounds live latest windows into 5-minute cache buckets. Those choices are
presentation-layer optimizations only; they do not change ingest or storage.
The interactive APS summary prefers a compact derived display-summary Zarr
under `/data/aurora/products/power/power_display_summary.zarr`. The latest APS
interactive figure is also prewarmed as Plotly JSON by the quicklook pipeline
under `/data/aurora/products/dashboard/prewarm/`.

## ASS PDU Zarr

Path:

- `/data/aurora/products/power/pdu.zarr`

Raw source mirror:

- `/project/aurora/raw/pdu`

The ASS PDU source files are synced from ASS Linux
`/home/aurora/data/pdu/pdu_DDMMYYYY.csv`. The CSV rows contain
`datetime,outlet,name,state,amps,watts`; the appender pivots those rows into
one time-indexed variable per outlet metric:

- `PDUOutlet1Watts` through `PDUOutlet8Watts`
- `PDUOutlet1Amps` through `PDUOutlet8Amps`
- `PDUOutlet1State` through `PDUOutlet8State`

`State` is encoded as `1` for on and `0` for off. The store is derived from
the synced raw CSV files and can be rebuilt without modifying the source
mirror.

## Derived display-summary Zarr

Path:

- `/data/aurora/products/power/power_display_summary.zarr`

This is the primary dashboard-serving Power summary product. It is built from
`power.zarr` plus optional ASFS logger `watts_on_48vdc_Avg` and ASS PDU outlet
power by `generate_power_display_summary.py`. It keeps the raw Power, ASFS
logger, and PDU Zarrs authoritative, but stores only the curated one-minute
traces needed by the APS summary panels. That lets the interactive browser
switch Power windows without repeatedly opening and merging multi-day
one-second APS samples.

When checked on `2026-05-21`, this derived store had `time=21547`, 25 data
variables, sorted unique timestamps, and coverage from
`2026-05-05 15:15:00` to `2026-05-21 20:10:00`.

Important variables include:

- `SolarWatts_East`, `SolarWatts_South`, `SolarWatts_West`
- `SolarVolts_East`, `SolarVolts_South`, `SolarVolts_West`
- `BatteryAmps`, `BatteryWatts`, `BatterySOC`
- `ACOutputWatts`, `DCInverterWatts`, `watts_on_48vdc_Avg`
- `PDUOutlet1Watts` through `PDUOutlet8Watts`, when PDU data are available
- `ACOutputVolts`, `DCInverterVolts`
- `InternalTemperature`, `HeatsinkTemperature`, `TempSensor1`-`TempSensor4`
- `PowerDisplaySolarYield_East`
- `PowerDisplaySolarYield_South`
- `PowerDisplaySolarYield_West`
- `PowerDisplayCumulativePowerGeneratedTotal`
- `PowerDisplayCumulativePowerUtilised`

The APS interactive summary also renders a display-only **SOC 24 h Forecast**
panel from `BatterySOC`. It fits the latest 30 minutes and latest 2 hours of SOC
with a low-degree polynomial and extrapolates both fits 24 hours forward. These
forecast traces are not stored in any Zarr product.

When `/data/aurora/products/power/power_soc_forecast.zarr` is available, the
display summary also includes the ECMWF-informed **SOC 48 h Forecast** traces:

- `BatterySOCForecast`
- `ECMWFSolarIrradiance`
- `ForecastSolarWatts`
- `ForecastLoadWatts`
- `ForecastSOCMAERecent`
- `ForecastSolarMAERecent`

Root attributes include:

- `power_display_summary_product = "true"`
- `source = "derived from power.zarr plus optional asfs_logger.zarr ASS 48 V power, pdu.zarr outlet power, and power_soc_forecast.zarr"`
- `frequency = "1min"` unless overridden by
  `AURORA_POWER_DISPLAY_SUMMARY_FREQ`

The display-summary product is regenerated by the Power quicklook pipeline. It
is safe to delete and rebuild because it is derived from the raw Power, ASFS
logger, PDU, and Power SOC forecast Zarrs.

## Derived SOC forecast Zarr

Path:

- `/data/aurora/products/power/power_soc_forecast.zarr`

This is a derived operational forecast product generated by
`generate_power_soc_forecast.py`. It retrieves ECMWF `ssrd` surface solar
radiation forecast data, converts accumulated `J m-2` values into interval
solar power in `W m-2`, calibrates expected APS solar charging from recent
`SolarWatts_East`, `SolarWatts_South`, and `SolarWatts_West`, estimates recent
station load from `ACOutputWatts + DCInverterWatts`, and integrates SOC forward
from the latest valid `BatterySOC`.

Before replacing the forecast, the generator scores the previous forecast
against newly arrived APS observations. It stores recent SOC forecast MAE and
solar charging MAE, then updates an adaptive calibration state used by later
forecast runs. This creates an improvement loop: as more forecast/observation
pairs accumulate, the solar conversion factor and load estimate can adapt to
the measured site behavior.

Variables:

- `BatterySOCForecast` in `%`
- `ECMWFSolarIrradiance` in `W m-2`
- `ForecastSolarWatts` in `W`
- `ForecastLoadWatts` in `W`
- `ForecastSOCMAERecent` in percentage points
- `ForecastSolarMAERecent` in `W`

Root attributes include ECMWF input file, generation time, initial SOC time and
value, horizon, calibration window, solar calibration factor, forecast load, and
battery capacity. The product is separate from model-evaluation ECMWF products.

The interactive APS summary presents `ACOutputWatts` and `DCInverterWatts` on
separate left/right axes in the **Output Power** panel. The optional
`watts_on_48vdc_Avg` context trace is presented as its own **ASS 48 V DC
Power** panel when ASFS logger data are available.
The optional PDU outlet watt traces are presented as their own **ASS PDU Outlet
Power** panel when `pdu.zarr` is available.

## Derived display-energy Zarr

Path:

- `/data/aurora/products/power/power_display_energy.zarr`

This is a compact compatibility product containing only the cumulative kWh
traces used by the APS cumulative panel. It can be built directly from
`power.zarr` by `generate_power_display_energy.py`, but the normal quicklook
pipeline now refreshes it from the broader display-summary product.

`BatterySOC` is stored in the broader display-summary product, not in this
compatibility store. The generated and utilised energy traces from this compact
store are plotted on the cumulative panel's right axis when the broader summary
product is unavailable.

When checked on `2026-05-21`, this derived store had `time=21547`, 5 data
variables, sorted unique timestamps, and coverage from
`2026-05-05 15:15:00` to `2026-05-21 20:10:00`.

Variables:

- `PowerDisplaySolarYield_East`
- `PowerDisplaySolarYield_South`
- `PowerDisplaySolarYield_West`
- `PowerDisplayCumulativePowerGeneratedTotal`
- `PowerDisplayCumulativePowerUtilised`

Root attributes include:

- `power_display_energy_product = "true"`
- `source = "derived from power.zarr"`
- `frequency = "1min"` unless overridden by
  `AURORA_POWER_DISPLAY_ENERGY_FREQ`

The display-energy product is safe to delete and rebuild because it is derived
entirely from the dashboard display-summary logic.
