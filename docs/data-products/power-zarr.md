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
- `InternalHumidity` is the reserved optional APS internal relative-humidity
  field in percent, used for operations dew-point monitoring when present

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
- `InternalHumidity`, when the APS logger provides a true internal RH signal
- `HeatsinkTemperature`
- `TempSensor1`
- `TempSensor2`
- `TempSensor3`
- `TempSensor4`
- `MaxSolarWatts_East`

## Schema note

- append runs keep the existing variable set fixed in the same way as
  `vaisalamet`
- `InternalHumidity` is the one allowed optional schema expansion; if it first
  appears in new raw APS CSV files, the appender adds it to the existing store
  with `NaN` backfill for older samples instead of silently dropping it
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
display summary also includes the ECMWF-informed **SOC 96 h Forecast** traces:

- `BatterySOCForecast`
- `BatterySOCForecast_Load100W`
- `BatterySOCForecast_Load200W`
- `BatterySOCForecast_Load300W`
- `BatterySOCForecast_Load400W`
- `BatterySOCForecast_Load500W`
- `BatterySOCForecast_Load600W`
- `ECMWFSolarIrradiance`
- `ForecastSolarWatts`
- `ForecastLoadWatts`

The fixed-load fields remain in the deterministic product for backwards
compatibility, but the dashboard no longer presents them as operating plans.
The visible scenario panel is populated from the learned operating-mode product
described below.

When `/data/aurora/products/power/power_soc_forecast_skill.zarr` is available,
the display summary also includes past-facing forecast verification traces:

- `ForecastVerificationSamples`
- `ForecastSOCMAE_0_6h_Verified`
- `ForecastSOCMAE_6_24h_Verified`
- `ForecastSOCMAE_24_48h_Verified`
- `ForecastSOCMAE_48_96h_Verified`
- `ForecastSOCBias_0_6h_Verified`
- `ForecastSOCSkill_0_6h`
- `ForecastSolarMAE24h`
- `ForecastSolarBias24h`
- `ForecastSolarSkill24h`
- `ForecastLoadMAE24h`
- `ForecastLoadBias24h`
- `ForecastLoadSkill24h`

Root attributes include:

- `power_display_summary_product = "true"`
- `source = "derived from power.zarr plus optional asfs_logger.zarr ASS 48 V power, pdu.zarr outlet power, power_soc_forecast.zarr, and power_soc_forecast_skill.zarr"`
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
`SolarWatts_East`, `SolarWatts_South`, and `SolarWatts_West`, builds a
named operating-mode station-load forecast from the APS power balance, and
integrates SOC forward from the latest valid `BatterySOC`. Observed total load
is `SolarWatts_East + SolarWatts_South + SolarWatts_West - BatteryWatts`, where
positive battery power is charging and negative power is discharge. This
captures the 48 V DC load that is absent from inverter idle power.

The existing operational deterministic product uses
`kit_mode_persistence_v4`. It recognises `DC-Only` when sustained AC output is
below `25 W`. Its preferred level estimate is the median battery discharge
`-BatteryWatts` from at least four 15-minute samples in the current mode during
the latest 48 hours when summed solar production is at most `10 W`. This
zero-solar estimate avoids treating inverter idle as the DC load and avoids
daylight solar-measurement error. When dark samples are unavailable, the model
falls back to the recent full power balance. When AC kit is on, finite PDU
outlet power at or above `5 W` names the mode before the slower AC-state median
changes; relay state is only a fallback when outlet watts are unavailable.
Outlet 5 identifies `DC-Only + CL61`. A mode is recognised immediately, but its
training level is retained only after at least 30 minutes of stable state and
two aggregated samples. The robust current-mode level is persisted through the
forecast horizon. Independent hourly mode observations are retained in the
forecast state, so a mode's typical load becomes more stable each time that kit
configuration is observed. It does not invent an hour-of-day schedule.

Before replacing the forecast, the generator scores archived forecast runs
against newly arrived APS observations and updates an adaptive calibration state
used by later forecast runs. The full forecast job downloads ECMWF data every 3
hours; the learning job can run every 15 minutes with `--refresh-from-cache` to
reuse the newest cached ECMWF GRIB while re-anchoring to the latest actual SOC.

The product still contains legacy fixed-load what-if fields from `100 W` through
`600 W` for API compatibility. They are excluded from archive scoring and
adaptive learning and are not shown as operational dashboard scenarios.

Variables:

- `BatterySOCForecast` in `%`
- `BatterySOCForecast_Load100W` in `%`
- `BatterySOCForecast_Load200W` in `%`
- `BatterySOCForecast_Load300W` in `%`
- `BatterySOCForecast_Load400W` in `%`
- `BatterySOCForecast_Load500W` in `%`
- `BatterySOCForecast_Load600W` in `%`
- `ECMWFSolarIrradiance` in `W m-2`
- `ForecastSolarWatts` in `W`
- `ForecastLoadWatts` in `W`

Root attributes include ECMWF input file, generation time, initial SOC time and
value, horizon, calibration window, solar calibration factor, forecast load,
load-bias correction, SOC lead-bucket correction, adaptive alpha, and battery
capacity. Load diagnostics include `load_model`, `load_model_version`,
`load_mode`, `load_mode_source`, `load_measurement`,
`load_balance_measurement`, `load_mode_registry`, `load_mode_signature`,
`load_mode_learning_ready`, `load_mode_learning_reason`,
`load_mode_pdu_active_watts`, `load_regime_level_w`, and
`load_regime_run_hours`. `minimum_operational_soc_pct = "40"` identifies the
reference used by SOC risk plots and ensemble threshold verification. Scenario attributes include
`scenario_loads_w = "100,200,300,400,500,600"`
and `scenario_solar_mode = "ecmwf"`. The product is separate from
model-evaluation ECMWF products.

The forecast-run archive is stored at:

- `/data/aurora/products/power/power_soc_forecast_archive.zarr`

It keeps recent forecast issue times, valid times, lead hours, and selected
forecast variables for skill scoring. The latest forecast product remains the
dashboard-facing product.

The forecast-verification product is stored at:

- `/data/aurora/products/power/power_soc_forecast_skill.zarr`

It compares archived forecasts to observed `BatterySOC`, summed solar charging,
and power-balance total load at matching valid times. Metrics are computed on past
timestamps over a rolling 24-hour verification window and include SOC MAE by
lead-time bucket, 0-6 h SOC bias, recent solar/load MAE and bias, sample count,
and deterministic skill scores relative to a persistence reference. Skill is
defined as `1 - forecast_mae / persistence_mae`, so positive values indicate
the operational forecast is beating the simple persistence baseline.

Cached 15-minute learning runs are grouped by ECMWF cycle and valid time before
verification, so the independent-cycle count is not inflated by highly
overlapping forecasts. Skill is withheld when the persistence error is too
small to provide a stable denominator. Load verification is also filtered by
`LoadModelVersion`; historical errors from retired load models are not mixed
with the current model's MAE, bias, or skill.

The fixed-lead hindcast product is stored at:

- `/data/aurora/products/power/power_soc_hindcast.zarr`

It retains seven days of observed SOC together with archived forecasts made 6,
24, 48, and 72 hours earlier. The dashboard draws these against the 40% minimum
operational SOC line.

The ECMWF ensemble products are stored at:

- `/data/aurora/products/power/power_soc_ensemble_forecast.zarr`
- `/data/aurora/products/power/power_soc_ensemble_archive.zarr`
- `/data/aurora/products/power/power_soc_ensemble_skill.zarr`

The ensemble generator retrieves the 50 IFS perturbed `ssrd` members for the
latest 00/12 UTC cycle. It streams the global GRIB messages through ecCodes and
extracts only the nearest AURORA grid point before constructing the compact
member-by-time array, stores that small site-level SSRD dataset for four cycles,
then deletes the temporary global GRIB. On later hourly runs within the same
ECMWF cycle, it rebuilds the ensemble whenever the latest actual SOC, calibrated
solar factor, learned load level, model version, or named operating mode has
changed. These same-cycle updates use the site cache rather than downloading
ECMWF again. Dashboard variables
include SOC P10, P50, P90, minimum, maximum, and probability below the 40%
minimum operational SOC threshold.
Verification includes CRPS by lead bucket, P10-P90 coverage, threshold Brier
score, and verified ensemble-cycle count.

## Learned operating-state and planning Zarrs

Development paths:

- `/data/aurora/dev-products/power/power_soc_planning_forecast.zarr`
- `/data/aurora/dev-products/power/power_operating_state.zarr`
- `/data/aurora/dev-products/power/power_operating_scenarios.zarr`

The 240-hour planning forecast is refreshed from the ECMWF 00 and 12 UTC cycles.
It retains native deterministic output through 240 hours and extends shorter
native ensemble input against that deterministic solar curve rather than
holding the final irradiance value constant.

`generate_power_operating_scenarios.py` runs every five minutes. It re-anchors
all scenarios to the latest finite `BatterySOC`, derives observed total load as
summed APS solar power minus signed `BatteryWatts`, and classifies the current
kit configuration from fresh PDU outlet evidence. Stale PDU evidence is treated
as unknown rather than carried into a new mode. The state product contains:

- `OperatingModeCode` and `OperatingModeProbability`
- `OperatingModeConfidence`
- `ObservedLoadWatts` and `EstimatedModeLoadWatts`
- `LoadInnovationWatts` and `LoadObservationOutlier`

The persisted `hybrid_state_space_v5` learner combines a finite set of named
operating modes with a robust Kalman update for continuous component loads. Its
components are the DC baseline, CL61, Radar, HATPRO, UAS, and an unknown-AC
increment. Existing observations are reclassified on each run, but component
parameters are updated only from timestamps newer than the saved training
cursor. Re-running unchanged data therefore does not double count evidence.

The scenario product carries P10, P50, and P90 SOC and load for these plans:

- current recognised mode
- DC-Only
- DC + CL61 continuously on
- optimized CL61 schedule
- each additional learned kit combination

The optimized plan maximizes CL61 collection time over the first 96 hours while
requiring P10 SOC to remain at or above 40%, a minimum 12-hour run, and no more
than one start per UTC day. Hours 97-240 retain the base mode with CL61 off so
the full planning horizon still exposes later battery risk. Recommendations are
advisory only; the forecast service does not issue PDU commands. The dashboard
also evaluates a user-selected CL61 start and duration directly from the stored
solar and component ensembles, so edits react without another ECMWF download.

The interactive APS summary presents `ACOutputWatts` and `DCInverterWatts` on
separate left/right axes in the **Output Power** panel. The optional
`watts_on_48vdc_Avg` context trace is presented as its own **ASS 48 V DC
Power** panel when ASFS logger data are available.
The optional PDU outlet watt traces are presented as their own **ASS PDU Outlet
Power** panel when `pdu.zarr` is available. All eight traces remain visible when
every outlet reports `0 W`, so powered-off kit is distinguishable from missing
PDU data.

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
