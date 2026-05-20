# Storage Layout

The deployed host deliberately keeps raw mirrored inputs and derived products in
separate trees.

## `/project/aurora`

- **Function:** raw mirrored source data
- **What lives there:** synced instrument files coming from the remote source
  machines
- **Examples:**
  - `/project/aurora/raw/cl61`
  - `/project/aurora/raw/rpgfmcw94`
  - `/project/aurora/raw/vaisalamet`
  - `/project/aurora/raw/asfs/loggernet`
  - `/project/aurora/raw/power/level1`
  - `/project/aurora/raw/wxcam`
- **Storage type:** shared Ceph network filesystem
- **Current filesystem size on `2026-05-20`:** `4.0T`
- **Current used on `2026-05-20`:** `41G`
- **Current available on `2026-05-20`:** `3.9T`

So `/project/aurora` is the raw landing and mirror area.

## `/data/aurora`

- **Function:** processed products and dashboard-serving outputs
- **What lives there:**
  - Zarr stores
  - quicklook PNGs
  - WXcam catalog SQLite
  - WXcam daily videos and thumbnails
  - performance logs and other dashboard products
- **Examples:**
  - `/data/aurora/products/cl61/...zarr`
  - `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`
  - `/data/aurora/products/quicklooks/...`
  - `/data/aurora/products/wxcam/...`
- **Storage type:** local disk on `/dev/vdb`
- **Current filesystem size on `2026-05-20`:** `983G`
- **Current used on `2026-05-20`:** `238G`
- **Current available on `2026-05-20`:** `695G`

So `/data/aurora` is the product, work, and output area.

## Why the split matters

- raw files stay separate from regenerated products
- products can be deleted and rebuilt without touching the source mirror
- the dashboard reads smaller processed artifacts from local disk instead of
  always working directly from the raw mirror
