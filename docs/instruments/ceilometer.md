# Ceilometer

The Ceilometer instrument is backed by the CL61 depolarization lidar Zarr and
is presented as a height-time plot in the interactive dashboard.

## Interactive behavior

- interactive tab shows recent windows from the CL61 Zarr
- latest-product generation also writes recent static products used elsewhere
- availability and freshness indicators reflect the selected time window

## Quicklooks

- science quicklooks show archived daily CL61 products
- housekeeping quicklooks include `HK_Ceilometer`, which focuses on
  non-science diagnostics rather than the main backscatter/depolarization
  fields

`HK_Ceilometer` currently groups:

- sample cadence
- receiver gain and backscatter signal diagnostics
- tilt, height offset, and tilt-correction state
- precipitation, fog, and total-cloud-cover flags
- vertical visibility

## Backing data product

Zarr path:

- `/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr`

Detailed schema:

- [Ceilometer Zarr](../data-products/ceilometer-zarr.md)

## Source mirror

Raw files are mirrored under:

- `/project/aurora/raw/cl61`
