# Model Evaluation Cleanup

The `model-evaluation` page is operational-first. Its active example is
`2026-08-01` UTC at the AURORA Iceland deployment. Historical Leeds cases and
sensitivity tests are provenance, not the default science-review surface.

## Default Active Surface

- Four top-level sections: Overview, Instrument Comparisons, Case Library, and
  Details / Provenance
- Direct model-variable evaluation for IFS, GFS, and ICON through MODF/MMDF
  support matching
- A GFS-forced CM1 full-LES virtual observatory; no LES-bridge product is an
  active comparison
- Official Cloudnet products plus radar, LWC/IWC, HATPRO/LWP, surface met,
  radiation/SEB, sonic, and gas records
- Hogan et al. (2009) cloud-fraction metrics, with headline-ranking exclusions
  shown whenever required method support is incomplete
- W-band radar rows should expose the active PAMTRA descriptor family when the
  product metadata are available
- Surface met should use
  `scorecards/surface_met_cm1_gfs_full_day_v1.json` and its rendered plot for
  the CM1 virtual-instrument comparison. Keep model/observation support-height
  differences visible; do not present a lowest-model-level comparison as a
  validated 2 m score.
- All seven instrument streams are physically collocated in Iceland from
  `2026-08-01`. CL61 backscatter, cloud occurrence, and cloud base are
  production-eligible. Linear depolarisation is observation-only because ALCF
  does not simulate it.
- Cloudnet/process evaluation uses the documented common radar/lidar/MWR
  overlap. For 1 August this is 00:02:08 to 23:56:10 UTC, or 23.900556 hours.
- Daily review queue rows with bundle/QA status, missing inputs, diagnostic and
  blocked stream counts, runner status, QA actions, and per-day archive class
  counts
- Current campaign artifacts under
  `/data/aurora/model-evaluation/campaigns/aurora_iceland_model_evaluation_v1`

## Removed Legacy Surface

The earlier hard-coded run explorer and candidate leaderboard have been removed
from the served page. The dashboard is no longer an entry point for old CM1
smoke tests, moisture-forcing experiments, ERA5/CARRA experiments, or proxy
W-band comparisons. Forensic review should use campaign bundle provenance and
archived files directly, not dashboard modes.

- `AURORA_MODEL_EVALUATION_SHOW_OPERATIONAL_DETAILS=1`

Operational details remain opt-in because they expose paths and scheduler state
that are useful for developers but too noisy for external science review.

## Data Retention Policy

Do not advertise old model directories during dashboard cleanup. Treat them as
archived evidence and classify them through `archive_manifest.json`. Active
review should use only campaign-root products and daily AURORA-LASSO bundles.

This is a dashboard visibility and scientific-provenance classification, not a
storage-backup or deletion policy. The independent GWS and object-store copies,
their verification, and any storage retention decision are owned by
`aurora-cloud-infra`; changing `archive_manifest.json` does not move, back up,
or delete a file.

Recommended archive classes:

- `active_campaign`: current daily AURORA-LASSO products and scorecards
- `reference`: fixed recipes, contracts, and community-method references
- `archived_experiment`: old CM1 sensitivity tests and exploratory PAMTRA sweeps
- `runtime_proof`: CM1 smoke tests and build/runtime checks
- `retired_dead_end`: intentionally inactive bridge-era or proxy outputs

The machine-readable archive manifest is the cleanup contract. Move or hide old
products by manifest class rather than by ad hoc path deletion.
