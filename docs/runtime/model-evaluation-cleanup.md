# Model Evaluation Cleanup

The `model-evaluation` page is now operational-first. Its default view should
show the AURORA-LASSO case library and daily operational campaign status, not
every historic sensitivity test.

## Default Active Surface

- AURORA-LASSO daily bundles
- Operational campaign readiness
- Daily ERA5, CM1 LES, Cloudnet, radar, LWC/IWC, LWP, SEB, ASFS/met summary
  records through the campaign products
- Current campaign artifacts under
  `/data/aurora/les/campaigns/aurora_multistream_pilot_20260520_20260602`

## Hidden Legacy Surface

The earlier hard-coded run explorer and candidate leaderboard are hidden by
default. They can be re-enabled only for forensic review with:

- `AURORA_MODEL_EVALUATION_SHOW_LEGACY_EXPLORER=1`
- `AURORA_MODEL_EVALUATION_SHOW_CANDIDATE_LEADERBOARD=1`
- `AURORA_MODEL_EVALUATION_SHOW_OPERATIONAL_DETAILS=1`

This keeps old CM1 smoke tests, moisture-forcing experiments, PAMTRA sweeps and
blocked IFS/HRES attempts from looking like active operational candidates.

## Data Retention Policy

Do not delete old model directories during dashboard cleanup. Treat them as
archived evidence until a separate archive manifest exists.

Recommended archive classes:

- `active_campaign`: current daily AURORA-LASSO products and scorecards
- `reference`: ERA5 reference and current CM1 daily recipe
- `archived_experiment`: old CM1 sensitivity tests and exploratory PAMTRA sweeps
- `runtime_proof`: CM1 smoke tests and build/runtime checks
- `blocked_dead_end`: scripted but intentionally inactive paths such as IFS/HRES
  without the required ECMWF access

The next cleanup step should write a machine-readable archive manifest on
aurora-cloud, then move or hide old products by manifest class rather than by
ad hoc path deletion.
