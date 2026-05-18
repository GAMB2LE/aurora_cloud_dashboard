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
  - `Surplus / Deficit` on the right axis
- **Output Voltage**
- **Thermal State**
  - internal temperature
  - heatsink temperature
  - temperature sensors 1-4
- **State of Charge**
  - `BatterySOC` as battery state of charge in percent

Legends are placed in a consistent right-side gutter rather than collected into
one global legend block.

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
