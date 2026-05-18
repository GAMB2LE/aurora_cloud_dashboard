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
- **Cumulative Power**
  - `East Solar Generated`
  - `South Solar Generated`
  - `West Solar Generated`
  - `Total Generated`
  - `Utilised`, integrated from AC and DC output power and reset at each UTC midnight
  - `Power Surplus / Deficit` on the right axis, calculated as a carried kWh
    energy balance from `Total Generated - Utilised`, with negative carried
    deficit cleared when `BatterySOC` reports full charge
- **Output Voltage**
- **Thermal State**
  - internal temperature
  - heatsink temperature
  - temperature sensors 1-4
  - left and right y-axes both use the same `Temperature [C]` range
- **State of Charge**
  - `BatterySOC` as battery state of charge in percent

Legends are placed in a consistent right-side gutter rather than collected into
one global legend block.

## Interactive performance behavior

The interactive view reads from the same Power Zarr used by the append and
quicklook pipelines. For browser performance, the app opens the store with
larger read chunks and uses the same per-trace time downsampling approach as the
quicklooks. Display-only sanity limits remove impossible APS values, such as
single-sample charging-current/current-power outliers, before plotting. The
live latest window is also rounded into 5-minute cache buckets so a small
timestamp advance does not rebuild the whole Power figure.

The **Battery Charging** panel also applies a display-only 30-minute rolling
mean to `BatteryAmps` and `BatteryWatts`. This keeps isolated charging
transients from dominating the visual scale while leaving the stored Power Zarr
unchanged.

Per-trace downsampling is important for the **ASS 48 V DC Power** overlay: that
line comes from the ASFS logger at about one-minute cadence, while the APS
power data are much denser. Downsampling after each trace has dropped merged
NaN timestamps preserves the ASFS cadence instead of thinning it on the dense
APS time grid.

The cumulative panel is normalized at display time. The `SolarYield_*` counters
are converted into positive UTC-day increments, so delayed controller resets
just after midnight do not create false drops in the plotted generation lines.
The utilised-energy line is integrated with midnight context before the view is
cropped back to the selected/latest window, so the latest 24 h view matches the
daily cumulative solar counters. The right-axis surplus/deficit trace is a
carried cumulative energy balance in kWh: each day's `Total Generated -
Utilised` balance starts from the previous day's ending surplus or deficit.
When `BatterySOC` reaches full charge, any negative carried deficit is cleared
to zero because the battery has recovered that energy shortfall.
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

Presentation-layer overlay:

- `/data/aurora/products/asfs_logger/asfs_logger.zarr`
  - `watts_on_48vdc_Avg` supplies the **ASS 48 V DC Power** trace on the right
    axis of the **Output Power** panel

Detailed schema:

- [Power Zarr](../data-products/power-zarr.md)
