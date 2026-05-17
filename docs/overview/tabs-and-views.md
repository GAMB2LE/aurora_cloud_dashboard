# Tabs And Views

This page describes what each top-level dashboard tab is responsible for.

## Interactive Data Browser

This is the primary live-browsing surface.

- **Ceilometer** and **Cloud Radar** show recent height-time plots.
- **Meteorology**, **Radiation**, and **Aurora Power Supply** show fixed
  summary layouts rather than a freeform variable picker.
- **WXcam** shows the stitched MP4 browser for `FISH HDR`.

The share link, download, freshness, and availability UI are rendered beneath
the content area so the data surface stays visually primary.

The interactive browser keeps the last rendered pane warm per instrument,
remembers the instrument-specific control state, and shows a small loading
notice or skeleton while a refresh is in progress.

## Science Quicklooks

This tab shows archived science-facing products.

- most instruments show daily PNG quicklooks
- **WXcam** is special: it uses an image-driven hourly grid built from HDR JPGs
  nearest the `:30` mark in each UTC hour

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

## Operations Dashboard

This tab is a live status dashboard rather than a static quicklook browser.

It shows:

- traffic-light indicators for source hosts, processing, transfers, and mirror
  verification
- Aurora Power Supply battery voltage from `DCInverterVolts`, scored green
  above `52 V`, amber from `50-52 V`, and red below `50 V`
- Aurora Power Supply internal temperature from `InternalTemperature`,
  scored green below `35 C`, amber from `35-40 C`, and red at `40 C` or above
- storage cards for the source hosts, the local Aurora host, and the JASMIN GWS
- per-stream archive and prune-readiness status

For the archived operations products, see
[Operations products](../data-products/operations-products.md).
