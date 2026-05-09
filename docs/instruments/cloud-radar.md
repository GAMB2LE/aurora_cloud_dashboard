# Cloud Radar

The Cloud Radar instrument is backed by the RPG FMCW 94 GHz radar Zarr and is
shown as a height-time product in the interactive dashboard.

## Interactive behavior

- interactive tab shows recent radar windows from the Zarr
- the app masks obviously bogus far-future timestamps when choosing recent
  windows and plot bounds
- availability bars make true data gaps visible across the selected time range

## Quicklooks

- science quicklooks show archived daily radar products
- housekeeping quicklooks include `HK_Radar`, which focuses on radar moments
  outside the main science view

## Backing data product

Zarr path:

- `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`

Detailed schema:

- [Cloud Radar Zarr](../data-products/cloud-radar-zarr.md)

## Source mirror

Raw files are mirrored under:

- `/project/aurora/raw/rpgfmcw94`
