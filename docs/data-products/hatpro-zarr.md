# HATPRO Zarr

Path: `/data/aurora/products/hatprog5/hatpro.zarr`

This store is a consolidated xarray Zarr product for the RPG HATPRO G5 scanning
microwave radiometer.

## Source files

Raw mirrored source files live under:

- `/project/aurora/raw/hatprog5`

The source tree is recursive by UTC-ish instrument date, for example:

- `Y2026/M05/D18/HATPROG5-AURORA-ICELAND_260518_090208.LWP.NC`

The builder reads these NetCDF product families:

- `*.LWP.NC` -> `LWP`
- `*.IWV.NC` -> `IWV`
- `*.IRT.NC` -> `IRR_Map`
- non-CMP `*.TPC.NC` and `*.TPB.NC` -> `T_PROF`
- `*.CMP.TPC.NC` -> `T_PROF_CMP`
- `*.MET.NC` -> `SURF_T`

## Dimensions and coordinates

Expected dimensions:

- `time`
- `range` for the temperature profile

The temperature-profile altitude coordinate from the source files is exposed as
the `range` coordinate when present.

When checked on `2026-05-18`, the deployed store had:

- `time=694062`
- `range=94`
- 6 data variables
- time coverage: `2026-02-27 13:00:01` to `2026-05-18 21:59:56`

## Variables

- `LWP`: liquid water path, in `g / m^2`
- `IWV`: integrated water vapor, in `kg / m^2`
- `IRR_Map`: infrared radiometer surface temperature product, in degrees C
- `SURF_T`: surface meteorology temperature from the HATPRO MET product, in K
- `T_PROF`: atmospheric temperature profile from the standard TPC plus TPB
  files, in K
- `T_PROF_CMP`: separate composite TPC profile product, in K

`T_PROF` and `T_PROF_CMP` have the same `time x range` structure but are kept
as separate variables because the source files can share timestamps without
being numerically identical.

## Update behavior

The current HATPRO builder rewrites the consolidated Zarr from the mirrored raw
tree. That keeps the product deterministic while the source file volume is
small and sparse. If HATPRO becomes a dense long-running stream, this should be
converted to a true incremental append builder.

Science quicklooks are generated under:

- `/data/aurora/products/quicklooks/hatpro/latest.png`
- `/data/aurora/products/quicklooks/hatpro/hatpro_YYYYMMDD.png`
