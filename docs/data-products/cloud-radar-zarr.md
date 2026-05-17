# Cloud Radar Zarr

Path:

- `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`

## Dataset shape

- dimensions: `time`, `range`
- deployed shape when checked on `2026-05-09`:
  - `time=11325`
  - `range=312`

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

## Conversion notes

- reflectivity-style fields are converted to dBZ during ingest
- fill values at or below the radar missing-data sentinel are converted to `NaN`
- the dashboard masks obviously bogus far-future timestamps when plotting or
  choosing recent windows
- append runs track the `range` layout and automatically rebuild a fresh store
  when a new contiguous raw-file run arrives with a different radar geometry

## Chunking

- radar science variables are chunked `(400, full-range)`

## Useful root attributes

Examples seen in the deployed store:

- `range_count = 312`
- `range_layout_key = "312:fd69e3de8501"`
