# Aurora Power Supply

Aurora Power Supply is the curated 1D electrical and thermal summary view built
from the power Zarr, with optional ASS 48 V DC power from the ASFS logger Zarr
and ASS PDU outlet power for station-load context.

## Interactive summary layout

The dashboard presents Aurora Power Supply in a deliberately scientific,
multi-panel style inspired by the reference plots used during development.

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
  - outlet watt traces from `PDUOutlet1Watts` through `PDUOutlet8Watts`, when available
  - known assignments are `PDUOutlet4Watts` = UAS, `PDUOutlet5Watts` = CL61,
    `PDUOutlet6Watts` = Radar, and `PDUOutlet8Watts` = HATPRO
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
- **SOC 48 h Forecast**
  - ECMWF-informed SOC forecast, shown when the forecast product is available
  - ECMWF solar power in `W m-2`
  - forecast solar charging and expected load
- **SOC Forecast Skill**
  - recent SOC forecast mean absolute error in percentage points
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

The **SOC 48 h Forecast** panel is generated by
`generate_power_soc_forecast.py`. It retrieves ECMWF `ssrd` surface solar
radiation forecast data, converts the accumulated `J m-2` field to interval
`W m-2`, calibrates expected solar charging from recent APS solar production,
builds a historical UTC-hour station-load profile from AC and DC output power,
and integrates SOC forward from the latest valid `BatterySOC`. This product is
operational guidance only and is stored separately from model-evaluation
products.

Each forecast refresh also scores the previous forecast against APS
observations that have arrived since it was written. The generator stores recent
SOC, solar, and load MAE plus load bias, updates an adaptive solar/load
calibration state, and writes the current skill metrics into the dashboard
product. As more overlapping forecast/observation pairs accumulate, the load
model applies a bounded bias correction to the historical load profile so future
load forecasts move toward measured site behavior.

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

Presentation-layer overlay:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`
  - `watts_on_48vdc_Avg` supplies the optional **ASS 48 V DC Power** panel

Detailed schema:

- [Power Zarr](../data-products/power-zarr.md)
