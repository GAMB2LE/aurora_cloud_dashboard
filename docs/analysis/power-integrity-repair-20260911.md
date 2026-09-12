# Evaluation-first power forecast repair

## Scope and release boundary

This change repairs evidence integrity, solar calibration, operating-state
verification and candidate uncertainty. It does **not** enable CL61/PDU control,
change the production deployment, or claim that a physical candidate has passed
its promotion campaign. Original v10 code/products and immutable issues remain
baseline evidence. Corrected legacy generation is a new semantic version, not
new training observations for v10.

The public development product must not be relabelled as a validated physical
forecast. Stage replay and recovery in an isolated candidate tree; promote only
after the cumulative gates below pass.

## Implemented contracts

- Archive writes clear inherited string widths/chunks, preserve nanosecond
  times, and verify a staged store before replacement. A failed validation
  leaves the previous archive intact. The recovery command creates a new
  archive solely from complete checksum-verified issue snapshots. Conflicting
  and unverifiable history is reported, never guessed or relabelled.
- Scalar PV calibration uses contemporaneous completed daylight intervals,
  not thousands of near-anchor records matched to one future value or a ratio
  of historical and future percentiles. Cached re-anchors cannot repeatedly
  learn the same errors. The ensemble uses the deterministic anchor and bound
  calibration snapshot, with its own source-cycle eligibility.
- Forecast and observation power share a total battery-side energy convention.
  AC plus inverter output is explicitly labelled an incomplete fallback, not
  training truth. SOC is an endpoint observation; solar/load are matched to
  their actual interval support, with coverage and gap checks. Available-PV
  verification requires all three MPPT-active flags; curtailed delivery is a
  separate quantity.
- Exact observed operating-state changes and unknown telemetry are separated
  from unchanged-state evidence. Residual fitting does not learn operator
  switches as a same-state bias. Retrospective observed-load/observed-solar/
  both-input integrations are explicitly ineligible for predictive scoring.
- Candidate member grids can be aligned only with an identical SOC anchor and
  source cycle, complete source coverage and conserved interval energy. SOC
  is reintegrated, never interpolated. Missing tails/cycles still fail closed.
- Residual quantiles create reproducible, temporally coherent member errors.
  This is not a claim of calibrated coverage: held-out CRPS and interval
  coverage remain required. Solar hardware uncertainty is not invented from
  unverified specifications.
- Campaign semantic keys exclude per-issue member arrays, learned battery
  values and publication signatures. New candidates retain the complete code
  revision but also carry a scoped implementation-content digest. UI-only
  changes therefore need not erase a compatible campaign. Legacy evidence
  still uses its original full code identity.
- Reserve-event sufficiency counts distinct observed downcrossings, not many
  forecast rows during one low-battery period. Initial low states and crossings
  across long observation gaps are not counted as established events.

## Safe recovery and replay

Run on the development host using its existing scientific Python runtime and a
tested, checksum-recorded isolated checkout. Examples deliberately require new
output paths and do not overwrite the live archives:

```sh
python repair_power_forecast_archive.py \
  --issues-root /data/aurora/dev-products/power/forecast-issues \
  --output-zarr /data/aurora/dev-products/power/candidates/integrity-repair-20260911/recovered.zarr \
  --report /data/aurora/dev-products/power/candidates/integrity-repair-20260911/recovery-report.json

python evaluate_power_diagnostic_replays.py \
  --issue-directory /path/to/verified/issue-directory \
  --power-zarr /data/aurora/products/power/power.zarr \
  --output-zarr /path/to/new/retrospective-diagnostic.zarr
```

The bounded replay queue accepts an explicit recovered archive and immutable
issue root. Its service example uses `MemoryMax=1.5G`, CPU limits and no public
model retrieval. It defers when the wider model evaluator is active; unknown
service state is not treated as permission to run. Deferral/failure never
marks an issue processed. Installing or activating the example timer is a
separate deployment action, not a side effect of running tests.

Before replacing any development artifact, preserve checksum-verified source,
configuration, state, archives and score products. Verify the new archive and
score outputs, stop only the named competing writer if a live swap is later
approved, and retain a one-command path rollback. No production replacement is
part of this repair.

## Data-dependent work still gated

1. Obtain surveyed per-array rated power, azimuth, tilt and controller limits.
   The existing configuration is explicitly provisional; do not convert
   curtailed observed peaks into claimed hardware ratings.
2. Enable the already implemented upstream register-791 acquisition through a
   controlled logger release/restart. Existing daily CSV headers must remain
   unchanged; new fields start in the next daily file. Do not restart the
   acquisition system as an incidental forecast-library deployment.
3. Carry optional exact-cycle direct radiation, temperature and wind through
   issue snapshots where actually delivered. Missing historical fields remain
   missing. Test ASFS nowcast/cloud corrections as distinct latency-safe
   ablations before enrolling them as predictors.
4. Fit PV response on simultaneous irradiance and MPPT-active observations;
   assess UAS recharge/heating phases only where actual state/energy telemetry
   isolates them. Do not invent measured component values or widen controls.
5. Run blocked rolling-origin replay and a short unpublished live campaign.
   Missing solar truth may allow SOC diagnostics but cannot pass the solar gate.

## Promotion acceptance

- At least 30 paired independent cycles per lead bucket across 10 UTC days,
  including clear/cloudy conditions and multiple important operating states.
- Combined 0–24 h SOC MAE improves by at least 10%; neither short bucket worsens
  by more than 2%; both have positive persistence skill.
- Each longer bucket has non-negative persistence skill or at least 25% lower
  MAE than the frozen baseline, with no worse absolute bias.
- Solar MAE and absolute load bias improve by at least 10% on consistent truth.
- CRPS beats baseline and persistence; P10–P90 coverage is 75–90%.
- Reserve Brier skill passes when sufficient independent episodes exist;
  otherwise report `insufficient_events`, not success.
- Provenance, reproducibility, missing-source, stale-input, bounds, quantile,
  API compatibility and resource-isolation checks pass.

Daily rolling diagnostics remain separate from cumulative promotion evidence.
Neither an empty score nor a full battery establishes forecast skill.
