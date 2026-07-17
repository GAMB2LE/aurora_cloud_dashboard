# Model Evaluation Cleanup

The `model-evaluation` page is now operational-first. Its default view should
show the AURORA-LASSO case library and daily operational campaign status, not
every historic sensitivity test.

## Default Active Surface

- AURORA-LASSO daily bundles
- Operational campaign readiness
- Daily ERA5, CM1 LES, Cloudnet, radar, LWC/IWC, LWP, SEB, ASFS/met summary
  records through the campaign products
- CM1 full-LES virtual-observatory products only; bridge-era products are
  archive/provenance material, not active review products
- W-band radar rows should expose the active PAMTRA descriptor family when the
  product metadata are available
- Surface met should use
  `scorecards/surface_met_cm1_era5_full_day.json` and its rendered plot for the
  CM1 virtual-instrument comparison. Keep the 50 m model support and missing
  Vaisala sensor-height metadata visible as a diagnostic caveat; do not present
  it as a validated 2 m score.
- CL61 is diagnostic for the Leeds replay, but can become production-valid for
  Iceland when the site metadata audit and scorecard both report a coincident
  comparison policy
- Daily review queue rows with bundle/QA status, missing inputs, diagnostic and
  blocked stream counts, runner status, QA actions, and per-day archive class
  counts
- Current campaign artifacts under
  `/data/aurora/les/campaigns/aurora_multistream_pilot_20260520_20260602`

## Removed Legacy Surface

The earlier hard-coded run explorer and candidate leaderboard have been removed
from the served page. The dashboard is no longer an entry point for old CM1
smoke tests, moisture-forcing experiments, IFS/HRES attempts, or proxy W-band
comparisons. Forensic review should use campaign bundle provenance and archived
files directly, not dashboard modes.

- `AURORA_MODEL_EVALUATION_SHOW_OPERATIONAL_DETAILS=1`

Operational details remain opt-in because they expose paths and scheduler state
that are useful for developers but too noisy for external science review.

## Data Retention Policy

Do not advertise old model directories during dashboard cleanup. Treat them as
archived evidence and classify them through `archive_manifest.json`. Active
review should use only campaign-root products and daily AURORA-LASSO bundles.

Recommended archive classes:

- `active_campaign`: current daily AURORA-LASSO products and scorecards
- `reference`: ERA5 reference and current CM1 daily recipe
- `archived_experiment`: old CM1 sensitivity tests and exploratory PAMTRA sweeps
- `runtime_proof`: CM1 smoke tests and build/runtime checks
- `retired_dead_end`: scripted but intentionally inactive paths such as IFS/HRES
  attempts or bridge-era W-band proxy outputs

The machine-readable archive manifest is the cleanup contract. Move or hide old
products by manifest class rather than by ad hoc path deletion.
