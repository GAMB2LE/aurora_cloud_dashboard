# Instruments Overview

The dashboard presents a mix of remote-sensing, station, camera, and operations
products. The UI layer is intentionally curated rather than exposing every raw
variable directly.

## Remote-sensing instruments

- [Ceilometer](ceilometer.md)
- [Cloud Radar](cloud-radar.md)
- [Scanning Microwave Radiometer](scanning-microwave-radiometer.md)

These instruments are driven by 2D `time x range` Zarr stores and render as
height-time or profile plots in the interactive tab.

## Curated 1D summary instruments

- [Meteorology](meteorology.md)
- [Radiation](radiation.md)
- [Aurora Power Supply](aurora-power-supply.md)

These instruments use fixed multi-panel layouts built from a subset of the
stored time-series variables.

## Camera and operations views

- [WXcam](wxcam.md)
- [AURORACam](auroracam.md)
- [Operations Dashboard](operations-dashboard.md)

WXcam is a hybrid media product stack. AURORACam is the four-MOBOTIX still
image browser for the MX4 FTP ingest. Operations Dashboard is the live status
layer for the whole deployed system.
