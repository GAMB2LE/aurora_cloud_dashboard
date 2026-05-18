# Ceilometer Zarr

Path:

- `/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr`

## Dataset shape

- dimensions: `time`, `range`, `layer`
- deployed shape when checked on `2026-05-18`:
  - `time=101532`
  - `range=3276`
  - `layer=5`

## Coordinates

- `time` - profile timestamps
- `range` - range gate center in meters
- `layer` - cloud-layer index
- `latitude`, `longitude` - site coordinates

## Useful root attributes

- `title = "CL61D CL61 with Depolarization"`
- `source = "gamb2le_depolarisation_lidar_ceilometer"`
- `conventions = "CF-1.8"`
- `profile_interval_in_seconds = 10`
- `file_temporal_span_in_minutes = 5.0`
- `schema_version = "1.3"`
- `instrument_serial_number = "X1627532"`
- `overlap_function_provided = 1`
- `overlap_is_corrected = 1`

## Variable layout

### Main `time x range` profile fields

- `beta_att`
- `linear_depol_ratio`
- `p_pol`
- `x_pol`

### `time x layer` cloud diagnostics

- `cloud_base_heights`
- `cloud_penetration_depth`
- `cloud_thickness`
- `sky_condition_cloud_layer_covers`
- `sky_condition_cloud_layer_heights`

### `time`-only diagnostics

- `beta_att_noise_level`
- `beta_att_sum`
- `fog_detection`
- `precipitation_detection`
- `receiver_gain`
- `sky_condition_total_cloud_cover`
- `tilt_angle`
- `tilt_correction`
- `vertical_visibility`

### Scalar metadata

- `range_resolution`
- `elevation`
- `azimuth_angle`
- `airplane_filter_max_range`
- `cloud_calibration_factor`
- `cloud_calibration_factor_user`

## Chunking

- `time x range` fields such as `beta_att` are chunked `(30, full-range)`
- `time x layer` diagnostics are chunked `(30, 5)`
- `time`-only diagnostics are chunked `(30,)`
