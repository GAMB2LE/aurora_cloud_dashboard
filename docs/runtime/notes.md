# Notes

This page captures dashboard-specific caveats that are important operationally.

## Cloud Radar timestamps

The radar data has previously contained obviously bogus far-future timestamps.
The appender now drops `NaT` values and samples more than two days in the
future while building or rebuilding the radar Zarr. `app.py` also keeps a
defensive bound filter when computing recent plot windows so the interactive
view stays usable if a bad sample is ever found before the next rebuild.

## Time ordering policy

Dashboard-facing time-series Zarr stores should be strictly increasing along
their `time` coordinate, with duplicate timestamps removed by the relevant
builder. The appenders sort and deduplicate newly read raw files before
writing. The operations-monitor appender also checks the existing Zarr before
append; if it finds an older manual/timer race that left time samples out of
order, it backs up the store and rebuilds it from the raw JSONL snapshots.

Append jobs should also filter to genuinely new timestamps before chunking or
writing. This avoids partial chunk appends that can leave all-NaN profile
stripes in range-resolved products. The CL61 and cloud-radar products were
repaired from their raw mirrors after this policy was added, and the numeric
appenders now materialize only the already-filtered new block before writing to
Zarr.

## Curated 1D instruments

`Meteorology`, `Radiation`, and `Aurora Power Supply` use fixed summary layouts.
Their ingest, local retention, and Zarr schemas are not changed by those
presentation choices; the dashboard simply renders curated subsets of the same
stored variables.

The latest views for those three fixed-summary instruments can be served from
prewarmed Plotly JSON written by the quicklook generators under
`/data/aurora/products/dashboard/prewarm/`. If the selected 24 h window is not
close enough to the latest data, the browser falls back to a normal interactive
render.

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
generation drops in the latest 24 h view. The utilised term remains the AC+DC
output power integral. The Battery Deficit trace shares the cumulative kWh axis
and is computed by energy accounting rather than by transforming SOC directly:
it integrates measured `BatteryWatts` energy flow, falls back to utilised minus
generated energy if that field is absent, uses sustained `BatterySOC >= 99.5 %`
to initialize the first zero-deficit point and to clear only small integration
drift, and clips to the configured installed bank capacity of `30.8 kWh` by
default. Extra-storage values are intentionally disabled for now because
`MaxSolarWatts_*` can be nonzero at night and is not a reliable curtailed-solar
measurement. Daily generated and utilised traces are visually broken at UTC
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

The main interactive controls are collapsible on small screens and stack
vertically so the data surface remains the primary scroll target.

## WXcam science behavior

- WXcam interactive uses stitched MP4 products
- WXcam science quicklooks use HDR JPGs nearest the `:30` mark of each UTC hour
- WXcam operations and archive verification may track the full mirrored raw tree
  even though the dashboard-facing products only use the HDR subset
- the WXcam catalog stores timestamps in `time_utc`, `time_epoch_ns`, and
  `day_utc`; queries should order by `time_epoch_ns` and `raw_path`

Some code and performance events still use `wxcam_calendar_*` names because the
WXcam day grid originally lived on a tab called Calendar. The visible UI is now
Science Quicklooks.
