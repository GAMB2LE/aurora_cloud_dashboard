# Aurora Power Supply

Aurora Power Supply is the curated 1D electrical and thermal summary view built
from the power Zarr.

## Interactive summary layout

The dashboard presents Aurora Power Supply in a deliberately scientific,
multi-panel style inspired by the reference plots used during development.

Typical panels include:

- **Renewables**
- **Cumulative Power**
  - `East`
  - `South`
  - `West`
  - `Utilised`
- **Battery Charging**
  - `Charging Current In`
  - `Charging Power In`
- **Output Power**
- **Output Voltage**

Legends are placed in a consistent right-side gutter rather than collected into
one global legend block.

## Quicklooks

- science quicklooks show the curated APS summary
- housekeeping quicklooks show `HK_APS`

## Backing data product

Zarr path:

- `/data/aurora/products/power/power.zarr`

Detailed schema:

- [Power Zarr](../data-products/power-zarr.md)
