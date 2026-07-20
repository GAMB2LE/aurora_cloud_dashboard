# Tabs And Views

This page describes what each top-level dashboard tab is responsible for.

## Desktop And Phone Navigation

Wide screens use the same full-name top tab bar as the stable production
dashboard. The tabs are **Overview**, **Interactive Data Browser**, **Power**,
**Science Quicklooks**, **House Keeping Quicklooks**, **AURORACam**, **UAS**,
and **Operations Dashboard**. Overview is an optional compact station landing
view that shows the latest cached station snapshot and a manual refresh; it
does not replace the full scientific browser. Only the selected tab is mounted in the browser, and
the heavier quicklook, UAS, and Operations content remains lazy-loaded.
When those full labels do not fit on a narrower desktop or tablet, the tab bar
scrolls horizontally in one row instead of wrapping or abbreviating them.

Phones use the compact operational views: **Overview**, **Power**, **Plots**,
**Camera**, and **Ops**. These group the same dashboard content into smaller
plot cards and stacked controls without forcing the full desktop plotting
surface into a narrow viewport. Science and housekeeping products are grouped
under **Plots**; UAS status is available through **Ops**, with a full tier
history and event drill-down in the native app. The native Camera view supports
UTC frame-time selection for historical AURORACam days.

On every browser width and in the native app, **Power** is divided into
**Current Conditions** and **Forecast & Planning**. Current Conditions is the
default and contains observed electrical and thermal panels. Forecast &
Planning is loaded on demand and contains the 24-hour and 96-hour outlooks,
operating scenarios, custom schedules, and forecast verification. The selected
browser section is retained in share links as `power_view=current|forecast`.

The URL continues to store the selected view. A desktop share link uses the
full tab slug, while a phone link also stores its compact section in
`mobile_tab`.

## Interactive Data Browser

This is the primary live-browsing surface.

- The default instrument is **Aurora Power Supply**.
- **Ceilometer** and **Cloud Radar** show recent height-time plots.
- **Meteorology**, **Radiation**, and **Aurora Power Supply** show fixed
  summary layouts rather than a freeform variable picker.
- **WXcam** shows the stitched MP4 browser for `FISH HDR` and `PANO HDR`.

The share link, download, freshness, and availability UI are rendered beneath
the content area so the data surface stays visually primary. Freshness and
availability are populated just after the page opens so the first browser paint
does not wait for a Zarr timestamp scan.

The interactive browser keeps the last rendered pane warm per instrument,
remembers the instrument-specific control state, reuses recent rendered windows,
and shows a cached latest quicklook or loading skeleton while a refresh is in
progress. Rapid control changes are debounced so only the newest requested view
renders.

For **Aurora Power Supply**, long interactive windows use the same display-time
preparation and per-trace time downsampling approach as the quicklooks, and live
latest windows are rounded into 5-minute cache buckets. The latest **Aurora
Power Supply**, **Meteorology**, and **Radiation** interactive views can also
start from prewarmed Plotly JSON created by the quicklook generators. This keeps
the browser responsive without changing the underlying Zarr stores.

The desktop controls use two compact rows for instrument, time window, live
state, reset, and day navigation. Variable and range controls appear beneath
them only when the selected instrument needs those controls. If a selected
window has no available samples, the empty view reports the UTC window and
points to the next useful check. For CL61, Cloud Radar, and HATPRO, a fresh
PDU state of **Off** changes that message to an intentional power-off notice:
collection is expected to pause until the assigned outlet is enabled. A powered
instrument with no samples still shows the normal data or freshness guidance.

The current tab, instrument, and important control values are also kept in the
browser URL. This makes mobile recovery less painful: if the phone backgrounds
or reloads the page, the URL can restore the selected view even when the
original websocket session is gone.

## Science Quicklooks

This tab shows archived science-facing products.

- most instruments show daily PNG quicklooks
- **WXcam** is special: it uses an image-driven hourly grid built from HDR JPGs
  nearest the `:30` mark in each UTC hour

The quicklook image pane, freshness strip, and availability bar are lazy-loaded
when the tab is first opened so the initial Interactive Data Browser page can
appear sooner.
PNG quicklooks are displayed through a small derived cache that trims only
trailing blank white canvas. The original quicklook PNGs remain unchanged for
download and archival use. The dashboard wraps displayed PNGs in responsive HTML
so tall quicklooks scale to the browser width without reserving their original
pixel height as blank page space.

## House Keeping Quicklooks

This tab contains archived operational or diagnostic products such as:

- `HK_Met`
- `HK_ASFS`
- `HK_APS`
- `HK_Ceilometer`
- `HK_Radar`
- `HK_WXcam`
- `HK_Operations`

These products are intentionally separate from the science quicklooks so the
science tab can stay focused on the most interpretable instrument products.
`HK_Ceilometer` and `HK_Radar` are real diagnostic views: Ceilometer uses CL61
time-only support variables, while Radar reads RPG LV1 housekeeping variables
from the raw mirror.

The housekeeping image pane, freshness strip, and availability bar are also
lazy-loaded on first use.

## AURORACam

This tab shows the four MOBOTIX M24 cameras from the MX4 FTP ingest.

- latest frame cards show all four cameras for the selected day
- the camera selector controls the large still-image viewer
- the selected camera includes a UTC hourly still strip
- images are served under `/auroracam-media/...`

The backing metadata product is `/data/aurora/products/auroracam/auroracam.zarr`,
but the full-resolution JPEGs remain in the raw mirror under
`/project/aurora/raw/auroracam`.

## UAS

This tab shows the mirrored Menapia MQTT tier log, including the current
reported/effective tier, freshness, tier-change history, and recent parsed log
records. It starts its periodic refresh only after the tab is opened.

## Operations Dashboard

This tab is a live status dashboard rather than a static quicklook browser.
It starts its periodic refresh only after the tab is opened.

It shows:

- traffic-light indicators for source hosts, processing, transfers, and mirror
  verification
- root-cause groups that separate source, source-sync/network, local
  processing, GWS transfer, and dashboard/render symptoms
- seven-day trend cards for storage, SOC, voltage, source lag, and GWS lag
- dashboard HTTP endpoint health and dashboard/infra git state
- Aurora Power Supply battery voltage from `DCInverterVolts`, scored green
  above `52 V`, amber from `50-52 V`, and red below `50 V`
- Aurora Power Supply battery state of charge from `BatterySOC`, scored green
  at or above `50 %`, amber from `25-50 %`, and red below `25 %`
- Aurora Power Supply internal temperature from `InternalTemperature`,
  scored green below `40 C`, amber from `40-45 C`, and red at `45 C` or above
- storage cards for the source hosts, the local Aurora host, and the JASMIN GWS
- per-stream archive and prune-readiness status

For the archived operations products, see
[Operations products](../data-products/operations-products.md).
