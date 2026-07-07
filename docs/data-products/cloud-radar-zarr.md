# Cloud Radar Zarr

Path:

- `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`

## Dataset shape

- dimensions: `time`, `range`
- deployed shape when checked on `2026-05-21`:
  - `time=321488`
  - `range=312`
- time coverage when checked: `2026-05-09 08:59:59.305` to
  `2026-05-21 19:59:54`
- sorted unique `time` coordinate

## Coordinates

- `time` - derived from `Time + Timems`
- `range` - concatenated chirp range gates from `C1Range` and `C2Range`

## Variable layout

This store currently contains 13 `float32` `time x range` fields:

- `ZE_dBZ`
- `ZE45_dBZ`
- `MeanVel`
- `ZDR`
- `SRCX`
- `SpecWidth`
- `SLDR`
- `Skew`
- `RHV`
- `PhiDP`
- `Kurt`
- `KDP`
- `DiffAtt`

Radar housekeeping fields such as `Status`, `DDVolt`, `PowIF`, `TTemp`,
`RTemp`, `PCTemp`, `Elv`, `Azm`, `SurfTemp`, and `SurfRelHum` are not stored in
this science Zarr. The `HK_Radar` quicklook reads those support variables from
the mirrored raw RPG LV1 files under `/project/aurora/raw/rpgfmcw94`.

## Display colormaps

Cloud Radar display colors follow Py-ART's variable-specific radar colormap
convention. The dashboard keeps the palette definitions local in
`radar_colormaps.py` so the deployed app and quicklook generators do not require
Py-ART at runtime.

| Dashboard variable | Py-ART-style colormap |
| --- | --- |
| `ZE_dBZ` | `HomeyerRainbow` |
| `ZE45_dBZ` | `HomeyerRainbow` |
| `MeanVel` | `BuDRd18` |
| `SpecWidth` | `NWS_SPW` |
| `ZDR` | `RefDiff` |
| `RHV` | `RefDiff` |
| `DiffAtt` | `RefDiff` |
| `SLDR` | `SCook18` |
| `PhiDP` | `Wild25` |
| `KDP` | `Theodore16` |
| `SRCX` | `Carbone17` |
| `Kurt` | `Carbone17` |
| `Skew` | `BuDRd18` |

The existing dashboard value limits remain tuned for the deployed W-band cloud
radar product; only the color tables are Py-ART-style.

## Conversion notes

- reflectivity-style fields are converted to dBZ during ingest
- fill values at or below the radar missing-data sentinel are converted to `NaN`
- the builder drops `NaT` and clearly bogus future timestamps before writing;
  the dashboard also keeps a defensive future-time mask for plot bounds
- append runs track the `range` layout and automatically rebuild a fresh store
  when a new contiguous raw-file run arrives with a different radar geometry
- append writes materialize only the already-filtered new block before writing,
  avoiding partial chunk appends that can create false all-NaN vertical stripes
  in one radar moment while other moments remain valid

## Chunking

- radar science variables are chunked `(400, full-range)`

## Useful root attributes

Examples seen in the deployed store:

- `range_count = 312`
- `range_layout_key = "312:fd69e3de8501"`
