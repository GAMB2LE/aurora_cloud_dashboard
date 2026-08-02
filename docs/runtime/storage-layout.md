# Storage Layout

The deployed system deliberately keeps raw mirrored inputs and derived products
in separate trees. The paths are stable across sites; the underlying storage
device is host-specific and should be checked live in the Operations Dashboard.

## `/project/aurora`

- **Function:** raw mirrored source data
- **What lives there:** synced instrument files coming from the remote source
  machines
- **Examples:**
  - `/project/aurora/raw/cl61`
  - `/project/aurora/raw/rpgfmcw94`
  - `/project/aurora/raw/vaisalamet`
  - `/project/aurora/raw/asfs/crd`
  - `/project/aurora/raw/power/level1`
  - `/project/aurora/raw/pdu`
  - `/project/aurora/raw/wxcam`
  - `/project/aurora/raw/auroracam`
- **Production storage:** shared Ceph network filesystem
- **Development storage:** replicated live data on the development host

So `/project/aurora` is the raw landing and mirror area.

## `/data/aurora`

- **Function:** processed products and dashboard-serving outputs
- **What lives there:**
  - Zarr stores
  - quicklook PNGs
  - WXcam catalog SQLite
  - WXcam daily videos and thumbnails
  - AURORACam metadata Zarr
  - performance logs and other dashboard products
- **Examples:**
  - `/data/aurora/products/cl61/...zarr`
  - `/data/aurora/products/rpgfmcw94/cloud_radar.zarr`
  - `/data/aurora/products/power/pdu.zarr`
  - `/data/aurora/products/quicklooks/...`
  - `/data/aurora/products/wxcam/...`
  - `/data/aurora/products/auroracam/auroracam.zarr`
- **Storage:** local product disk on each host

So `/data/aurora` is the product, work, and output area.

Development experiments use `/project/aurora/dev-raw` and
`/data/aurora/dev-products`. They must not overwrite the mirrored production
trees above.

Long-term backups are not defined by either local filesystem alone. Production
infrastructure copies verified raw and selected product paths additively to the
JASMIN GWS and object storage. The dashboard only consumes their health
contract; see [Archive health](archive-health.md).

## Why the split matters

- raw files stay separate from regenerated products
- products can be deleted and rebuilt without touching the source mirror
- the dashboard reads smaller processed artifacts from local disk instead of
  always working directly from the raw mirror
