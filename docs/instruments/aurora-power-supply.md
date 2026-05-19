# Aurora Power Supply

Aurora Power Supply is the curated 1D electrical and thermal summary view built
from the power Zarr, with the ASS 48 V DC power overlaid from the ASFS
logger Zarr for station-load context.

## Interactive summary layout

The dashboard presents Aurora Power Supply in a deliberately scientific,
multi-panel style inspired by the reference plots used during development.

Typical panels include:

- **Renewables**
- **Battery Charging**
  - `Charging Current In`
  - `Charging Power In`
- **Output Power**
  - AC output power
  - DC inverter power
  - ASS 48 V DC power from `watts_on_48vdc_Avg` on the right axis
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
Legends are placed in a consistent right-side gutter rather than collected into
one global legend block.

## Interactive performance behavior

The interactive view reads ordinary APS traces from the same Power Zarr used by
the append and quicklook pipelines. For browser performance, the cumulative
energy traces are read from
`/data/aurora/products/power/power_display_energy.zarr`, a compact one-minute
display product derived from the raw Power Zarr. The raw store remains
authoritative; the compact product only avoids multi-day one-second reads when
the browser needs the cumulative panel.

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

Per-trace downsampling is important for the **ASS 48 V DC Power** overlay: that
line comes from the ASFS logger at about one-minute cadence, while the APS
power data are much denser. Downsampling after each trace has dropped merged
NaN timestamps preserves the ASFS cadence instead of thinning it on the dense
APS time grid.

The cumulative panel is normalized in the display-energy product. The
`SolarYield_*` counters are converted into positive UTC-day increments, so
delayed controller resets just after midnight do not create false drops in the
plotted generation lines. The utilised-energy line is integrated from AC+DC
output power. `BatterySOC` remains a direct raw Power Zarr field and is plotted
as `State of Charge` on the left axis. The cumulative generated and utilised
energy traces are plotted on the right axis in kWh, so the panel shows
generation, use, and battery state without deriving a separate deficit
estimate.
The daily generated and utilised traces are visually broken at UTC midnight so
their resets do not render as false vertical jumps.

These are display-time optimizations only; ingest, retention, and the stored
Zarr schema are unchanged.

## Quicklooks

- science quicklooks show the curated APS summary
- housekeeping quicklooks show `HK_APS`

## Backing data product

Zarr path:

- `/data/aurora/products/power/power.zarr`

Derived display product:

- `/data/aurora/products/power/power_display_energy.zarr`

Presentation-layer overlay:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`
  - `watts_on_48vdc_Avg` supplies the **ASS 48 V DC Power** trace on the right
    axis of the **Output Power** panel

Detailed schema:

- [Power Zarr](../data-products/power-zarr.md)
