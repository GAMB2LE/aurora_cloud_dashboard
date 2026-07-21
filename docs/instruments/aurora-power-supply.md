# Aurora Power Supply

Aurora Power Supply is the curated 1D electrical and thermal summary view built
from the power Zarr, with optional ASS 48 V DC power from the ASFS logger Zarr
and ASS PDU outlet power for station-load context.

## Interactive summary layout

The dashboard presents Aurora Power Supply in a deliberately scientific,
multi-panel style inspired by the reference plots used during development.
The Power view is divided consistently across desktop, phone browser, iPhone,
and iPad:

- **Current Conditions** is the default and shows observed station electrical
  and thermal behavior for the selected window.
- **Forecast & Planning** loads on demand and groups the 24-hour forecast,
  ECMWF 96-hour outlook, operating-mode scenarios, custom operating plan, and
  verification evidence.

This split changes only presentation and bounded API selection. It does not
change the forecast calculations or stored products.

## Forecast plot information

Every Forecast & Planning plot has an **Info** control in the desktop browser,
mobile browser, and native iOS app. It describes the metric and the exact
implementation used at AURORA, so a plot can be interpreted without leaving
the dashboard. The key distinctions are:

- **P10, central, and P90 SOC** are ECMWF solar-weather outcomes with the
  detected current system load held fixed. They are not different instrument
  schedules.
- **Operating-mode scenarios** are separate advisory simulations. Each includes
  the DC baseline plus the instruments named in its legend. Shaded intervals on
  the solar/load panel identify recommended CL61-on periods and do not indicate
  forecast probability or an automatic PDU command.
- **Hindcasts and verification scores** look backwards: archived forecasts are
  matched to later APS observations. MAE, CRPS, and Brier scores are lower-is-
  better; P10-P90 coverage targets 0.80.
- **Solar and load verification** is versioned with the load model, so a model
  update begins a comparable new set of evidence rather than mixing scores.

Typical panels include:

- **Renewables**
- **Battery Charging**
  - `Charging Current In`
  - `Charging Power In`
- **Output Power**
  - AC output power on the left axis
  - DC inverter power on the right axis
- **ASS 48 V DC Power**
  - ASS 48 V DC power from `watts_on_48vdc_Avg`, when available
- **ASS PDU Outlet Power**
  - assigned outlet watt traces: `PDUOutlet4Watts` = UAS, `PDUOutlet5Watts` = CL61,
    `PDUOutlet6Watts` = Radar, and `PDUOutlet8Watts` = HATPRO
  - unassigned physical outlets are retained in the data product but are not plotted
  - remains visible when every outlet reports `0 W`, so powered-off kit is an
    explicit operational state rather than a missing panel
- **Cumulative Power & State of Charge**
  - `State of Charge`, from `BatterySOC`, on the left axis in percent
  - `East Solar Generated`
  - `South Solar Generated`
  - `West Solar Generated`
  - `Total Generated`
  - `Utilised`, integrated from AC and DC output power and reset at each UTC midnight
  - cumulative generated and utilised energy traces on the right axis in kWh
- **Output Voltage**
- **Thermal State**
  - internal temperature
  - heatsink temperature
  - temperature sensors 1-4
  - left and right y-axes both use the same `Temperature [C]` range
- **SOC 24 h Forecast**
  - `State of Charge`
  - `30 min fit +24 h`, a display-only polynomial fit to the latest 30 minutes
    of `BatterySOC`
  - `2 h fit +24 h`, a display-only polynomial fit to the latest 2 hours of
    `BatterySOC`
- **SOC 96 h Forecast**
  - ECMWF-informed SOC forecast, shown when the forecast product is available
  - ECMWF solar power in `W m-2`
  - forecast solar charging and expected load
- **Suggested Instrument-Mode SOC Forecasts**
  - CL61, CL61 + Radar, CL61 + HATPRO, CL61 + HATPRO + Radar, HATPRO + Radar,
    Radar, and HATPRO
  - median SOC for each combination and a 40% minimum operational reference
- **Custom CL61 Operating Plan**
  - user-selected UTC start and run duration
  - immediate advisory safety, collection-hour, minimum-P10, and final-P10
    results without issuing PDU commands
- **SOC Forecast Skill**
  - SOC forecast mean absolute error by lead-time bucket
  - forecast skill sample count
  - recent solar forecast mean absolute error in watts
  - recent load forecast mean absolute error and bias in watts
Legends are placed in a consistent right-side gutter rather than collected into
one global legend block.

## Interactive performance behavior

The interactive view prefers
`/data/aurora/products/power/power_display_summary.zarr`, a compact
one-minute display product derived from the raw Power Zarr plus the ASFS logger
`watts_on_48vdc_Avg` context trace. The raw store remains authoritative; the
compact product only avoids multi-day one-second reads and source merges when
the browser needs the curated APS summary panels. If the broader display
summary is missing, the app falls back to the raw Power Zarr and the smaller
`/data/aurora/products/power/power_display_energy.zarr` cumulative-energy
product.

When `/data/aurora/products/power/pdu.zarr` is available, the display summary
also includes ASS PDU outlet watt traces. Those PDU samples are synced from ASS
Linux `/home/aurora/data/pdu/pdu_DDMMYYYY.csv` into `/project/aurora/raw/pdu`
and appended by `append_new_pdu_to_zarr.py`.

The app opens the Power store with larger read chunks and uses per-trace time
downsampling. Display-only sanity limits remove impossible APS values, such as
single-sample charging-current/current-power outliers, before plotting. The
live latest window is rounded into 5-minute cache buckets, and the latest Power
interactive figure is prewarmed as Plotly JSON by `generate_power_quicklooks.py`
so first paint can reuse the most recent quicklook-era render.

Observed panels in that prewarm use the measured APS display window rather than
the later timestamp of the 96-hour forecast fields. Power panels reserve a
88-pixel vertical gap for their two-line UTC tick labels and axis titles.

That prewarmed JSON lives under `/data/aurora/products/dashboard/prewarm/`.

The **Battery Charging** panel also applies a display-only 30-minute rolling
mean to `BatteryAmps` and `BatteryWatts`. This keeps isolated charging
transients from dominating the visual scale while leaving the stored Power Zarr
unchanged.

Per-trace downsampling is important for **ASS 48 V DC Power**: that line comes
from the ASFS logger at about one-minute cadence, while the APS power data are
much denser. Downsampling after each trace has dropped merged NaN timestamps
preserves the ASFS cadence instead of thinning it on the dense APS time grid.

The **ASS 48 V DC Power** panel depends on the ASFS slow `sci` table field
`watts_on_48vdc_Avg`. It can therefore go stale independently of the APS power
system. The APS AC/DC output, battery, solar, SOC, and thermal traces come from
the Power Zarr and can remain current even when the ASFS `sci` stream is not
producing new files.

The **ASS PDU Outlet Power** panel depends on the ASS PDU CSV logger. It can go
stale independently of both APS and ASFS; operations monitoring tracks its
source sync and append timers separately.

Operations monitoring scores APS `InternalTemperature` on both cold and hot
thresholds: green from `10-40 C`, amber from `5-10 C` or `40-45 C`, and red
below `5 C` or at `45 C` or above.

APS internal dew-point monitoring is optional and requires a true APS internal
relative-humidity field named `InternalHumidity` in percent. The dashboard does
not substitute ambient HATPRO, Vaisala, or ASFS humidity for internal APS
humidity. When `InternalHumidity` is present, operations snapshots calculate
dew point from same-sample `InternalTemperature` and `InternalHumidity`, then
mark the dew-point margin red when `InternalTemperature - dewpoint <= 0 C`.
When the field is absent, the Operations Dashboard shows internal humidity as
unavailable rather than estimating it from site meteorology.

`DCInverterWatts` is plotted on its own right axis because its raw value is
often much smaller than AC output power. The source CSV reports it consistently
with `DCInverterVolts * DCInverterAmps`, so a 53.75 V, 0.17 A sample appears as
about 9 W.

The cumulative panel is normalized in the display products. The
`SolarYield_*` counters are converted into positive UTC-day increments, so
delayed controller resets just after midnight do not create false drops in the
plotted generation lines. The utilised-energy line is integrated from AC+DC
output power. `BatterySOC` is plotted as `State of Charge` on the left axis.
The cumulative generated and utilised energy traces are plotted on the right
axis in kWh, so the panel shows generation, use, and battery state without
deriving a separate deficit estimate.
The daily generated and utilised traces are visually broken at UTC midnight so
their resets do not render as false vertical jumps.

The **SOC 24 h Forecast** panel is a presentation-layer forecast only. It
fits recent `BatterySOC` samples from the latest 30 minutes and latest 2 hours,
then extrapolates those fits 24 hours ahead. The default polynomial degree is
`1` for stability and can be changed with
`AURORA_POWER_SOC_PROJECTION_POLY_DEGREE`; forecast values are clipped to the
physical `0-100 %` SOC range.

The existing operational **SOC 96 h Forecast** panel is generated by
`generate_power_soc_forecast.py`. It retrieves ECMWF `ssrd` surface solar
radiation forecast data, converts the accumulated `J m-2` field to interval
`W m-2`, calibrates expected solar charging from recent APS solar production,
and derives total station load from solar generation minus signed battery
power. This includes the 48 V DC system load that is not represented by the
roughly `9 W` inverter-idle value. The minimum-power mode is named `DC-Only`.
For that mode, the preferred baseline is median battery discharge during the
latest zero-solar periods: with no solar input, `load = -BatteryWatts`. The
independent ASS 48 V trace measures the principal 48 V branch, while the
battery-side estimate also includes the remaining DC consumers and conversion
losses. A recent full power-balance median is used when there are not enough
dark samples.
When AC kit is switched on, fresh non-zero PDU outlet power names the mode from
that kit. Outlet 5 identifies the Ceilometer as `DC-Only + CL61`; relay state is
only used when outlet watts are unavailable. Recognition can update the next
forecast immediately, while durable mode learning waits for 30 minutes of
stable AC/DC state and at least two aggregated samples. The learner then stores
independent hourly load observations and persists the recognised mode's robust
level; it does not infer a daily schedule. The mode name is included in the
forecast-load legend. This version-4 model refreshes with every 15-minute
learning run.
This product is operational guidance only and is stored separately from
model-evaluation products.

The **Suggested Instrument-Mode SOC Forecasts** panel uses the same ECMWF solar
input and latest `BatterySOC` anchor for all seven stable combinations. Each
scenario includes the DC baseline and load distributions learned for its named
instruments. The separate optimized CL61 plan maximizes collection time over 96
hours while keeping P10 SOC at or above 40%, requiring a 12-hour minimum run,
and allowing no more than one start per UTC day. The custom-plan editor
evaluates a selected start and duration against the stored ensembles
immediately. All plans are advisory only.

The hybrid learner combines a finite-state/HMM-like mode classifier with robust
Kalman updates for the DC and kit load components. It saves the full component
covariance and only learns from observations newer than its persisted cursor,
so repeated five-minute runs do not count the same evidence again.

The development planning forecast retrieves eligible ECMWF 00/12 UTC output
twice daily and extends to 240 hours. A five-minute operating-state refresh
reuses that solar forecast, re-anchors every plan to the latest actual
`BatterySOC`, recognises mode changes, and incrementally updates component
loads. The existing operational deterministic forecast continues its own
three-hour/15-minute cadence until the development model is promoted.
The **Forecast Verification** panel is past-facing. It reads
`power_soc_forecast_skill.zarr`, which matches archived forecast valid times to
observed APS SOC, solar charging, and power-balance total load. It reports
rolling 24-hour SOC
MAE by `0-6 h`, `6-24 h`, `24-48 h`, and `48-96 h` lead buckets, plus recent
solar/load MAE and load bias. Skill scores are computed relative to persistence
and kept in the same verification product for diagnostics. Load metrics are
scored only against archive rows from the current load-model version so a model
change starts a new, comparable verification record.

The Power summary also includes a seven-day fixed-lead SOC hindcast, comparing
observations with forecasts issued 6, 24, 48, and 72 hours earlier. A separate
50-member ECMWF IFS ensemble supplies P10-P90 SOC uncertainty and the forecast
probability of crossing below the 40% minimum operational threshold. In this
operational panel every member holds the detected current PDU/APS system mode
and load fixed, so P10-P90 represent ECMWF solar uncertainty only. Deliberate
instrument-on schedules are evaluated solely in the separate operating-mode
plans. Ensemble CRPS,
interval coverage, and threshold Brier score remain pending until verifying
observations have arrived.

These are display-time optimizations and derived operational forecasts. Raw
APS ingest and retention are unchanged.

## Quicklooks

- science quicklooks show the curated APS summary
- housekeeping quicklooks show `HK_APS`

## Backing data product

Zarr path:

- `/data/aurora/products/power/power.zarr`

Derived display product:

- `/data/aurora/products/power/power_display_summary.zarr`
- `/data/aurora/products/power/power_display_energy.zarr`
- `/data/aurora/products/power/pdu.zarr`
- `/data/aurora/products/power/power_soc_forecast.zarr`
- `/data/aurora/dev-products/power/power_soc_planning_forecast.zarr`
- `/data/aurora/dev-products/power/power_operating_state.zarr`
- `/data/aurora/dev-products/power/power_operating_scenarios.zarr`

Presentation-layer overlay:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`
  - `watts_on_48vdc_Avg` supplies the optional **ASS 48 V DC Power** panel

Detailed schema:

- [Power Zarr](../data-products/power-zarr.md)
