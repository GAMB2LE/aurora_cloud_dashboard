# Notes

This page captures dashboard-specific caveats that are important operationally.

## Cloud Radar timestamps

The radar data has previously contained obviously bogus far-future timestamps.
`app.py` filters clearly invalid future times when computing bounds and recent
plot windows so the interactive view stays usable even if the underlying store
contains bad samples.

## Curated 1D instruments

`Meteorology`, `Radiation`, and `Aurora Power Supply` use fixed summary layouts.
Their ingest, local retention, and Zarr schemas are not changed by those
presentation choices; the dashboard simply renders curated subsets of the same
stored variables.

## Power interactive performance

The raw Power Zarr schema and chunks are unchanged. The dashboard uses larger
read chunks, display-only sanity limits, per-trace time downsampling, and
5-minute latest-window cache buckets for the interactive Power view. The
cumulative-energy panel additionally reads from the compact derived store
`/data/aurora/products/power/power_display_energy.zarr`, which is regenerated
from `power.zarr` by the Power quicklook pipeline.

The Battery Charging panel additionally applies a display-only 30-minute rolling
mean to `BatteryAmps` and `BatteryWatts` so short charging transients do not
dominate the plotted scale.

The cumulative Power display product is computed with full available context.
Solar-yield counters are treated as daily counters and converted to positive
increments, so delayed controller resets after midnight do not produce false
generation drops in the latest 24 h view. The right-axis surplus/deficit trace
is a carried cumulative kWh balance: each day's
`Total Generated - Utilised` balance starts from the previous day's ending
surplus or deficit. The utilised term remains the AC+DC output power integral.
The balance is anchored to recent full-SOC history using one constant offset.
That avoids both midnight discontinuities and artificial jumps at the SOC
threshold. Daily generated and utilised traces are visually broken at UTC
midnight so their resets are shown as new segments rather than connected
vertical jumps.

## Meteorology display merge

The Meteorology summary view merges selected ASFS logger met traces into the
Meteorology presentation layer without changing either underlying Zarr store.
With the current ASFS CRD schema, those supporting traces include ASFS Vaisala
temperature, relative humidity, and pressure in addition to the existing Metek
wind and temperature context.

## Housekeeping sources

`HK_Ceilometer` uses CL61 time-only diagnostics from the ceilometer Zarr and
does not duplicate the main backscatter/depolarization fields. `HK_Radar`
reads raw RPG LV1 support variables from `/project/aurora/raw/rpgfmcw94`
because the radar science Zarr intentionally only stores height-time science
moments.

## Mobile persistence

Panel sessions still depend on an active browser websocket, so a phone can
lose the live session when the operating system backgrounds the browser. The
service is configured with longer unused-session retention and frequent
keepalives, and the app updates the URL query string as controls change. That
combination gives short backgrounding events a chance to reconnect and lets a
killed tab reload into the same selected view.

## WXcam science behavior

- WXcam interactive uses stitched MP4 products
- WXcam science quicklooks use HDR JPGs nearest the `:30` mark of each UTC hour
- WXcam operations and archive verification may track the full mirrored raw tree
  even though the dashboard-facing products only use the HDR subset
