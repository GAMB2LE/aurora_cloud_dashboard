# Scanning Microwave Radiometer

The Scanning Microwave Radiometer view is built from the RPG HATPRO G5 product
files mirrored from the ASS source host.

## Interactive layout

The interactive browser renders a fixed HATPRO layout:

- LWP and IWV on the top panel
- infrared/surface temperature context on the middle panel
- temperature profile as a `time x range` heatmap on the bottom panel

The app reads the consolidated Zarr at:

- `/data/aurora/products/hatprog5/hatpro.zarr`

## Source and processing pipeline

Raw HATPRO files are mirrored into:

- `/project/aurora/raw/hatprog5`

The source sync keeps the recursive `Y2026/M05/D18` style directory structure
from the instrument host and retains the full HATPRO file set. The Zarr builder
uses the NetCDF products for LWP, IWV, infrared temperature, meteorology, and
temperature profiles.

The active source and processing timers are:

- `aurora-hatpro-source-sync.timer`
- `aurora-hatpro-append.timer`
- `aurora-hatpro-quicklooks.timer`

The append service currently rewrites the deterministic consolidated Zarr and
then swaps it into place, so dashboard readers do not see a half-written store
during a timer run.

Science Quicklooks are written to:

- `/data/aurora/products/quicklooks/hatpro`

The static HATPRO science quicklook uses a common UTC x-axis across the LWP/IWV,
surface-temperature, and temperature-profile panels. The profile panel is built
from valid `T_PROF` profile times independently from the denser 1D variables, so
time thinning does not leave the heatmap with only a few profile columns.

Detailed schema:

- [HATPRO Zarr](../data-products/hatpro-zarr.md)
