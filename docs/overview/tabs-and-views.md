# Tabs And Views

This page describes what each top-level dashboard tab is responsible for.

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

For **Aurora Power Supply**, long interactive windows are reduced with bucketed
first/min/mean/max/last representatives per trace, and live latest windows are
rounded into 5-minute cache buckets. This keeps the Power view responsive
without changing the underlying Power Zarr.

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

## Operations Dashboard

This tab is a live status dashboard rather than a static quicklook browser.
It starts its periodic refresh only after the tab is opened.

It shows:

- traffic-light indicators for source hosts, processing, transfers, and mirror
  verification
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
