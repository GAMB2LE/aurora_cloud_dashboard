# AURORACam Zarr

Path: `/data/aurora/products/auroracam/auroracam.zarr`

`auroracam.zarr` is a metadata index for the MX4 camera JPEG archive. It is
rebuilt from the raw mirror by `index_auroracam_zarr.py`.

## Source

- raw mirror: `/project/aurora/raw/auroracam`
- expected file shape:
  `<camera>/YYYY-MM-DD/<camera>_YYYY-MM-DD_HH-MM.jpg`

## Contents

The dataset has one `record` per valid JPEG and includes:

- `time`: UTC timestamp parsed from the filename
- `time_epoch_ns`: timestamp as integer nanoseconds
- `camera_index`, `camera_id`, `camera_label`, and `camera_ip`
- `day_utc`, `filename`, `relative_path`, and `raw_path`
- `size_bytes` and `mtime_ns`

The product is intentionally metadata-only. The dashboard serves the JPEG files
directly under `/auroracam-media/...` for image display.

## Rebuild

```bash
python index_auroracam_zarr.py \
  --root /project/aurora/raw/auroracam \
  --zarr /data/aurora/products/auroracam/auroracam.zarr \
  --rebuild
```

The deployed timer is `aurora-auroracam-index.timer`.
