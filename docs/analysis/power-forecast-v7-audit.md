# APS forecast version 7 audit

Audit date: 2026-08-01.

## Why the model changed

Recent operating periods included different combinations of CL61, Radar,
HATPRO, and UAS. Comparing forecast archives, APS observations, PDU telemetry,
and UAS tier logs exposed four structural problems in the previous model:

- the published deterministic load was about `265 W` while observed
  all-instrument operation was commonly `881-990 W`;
- the scenario learner shifted the DC component to force its components to a
  deterministic target, which could reduce the inferred DC baseline to `0 W`;
- recent load bias was about `-446 W` on development and `-485 W` on
  production;
- nominal P10-P90 ensemble coverage was about `63%`, below the `80%` target,
  and 24-48 hour SOC forecasts were worse than persistence.

Different forecast paths also used solar conversion factors between roughly
`1.8` and `3.2`, SOC integration assumed a lossless 26 kWh battery, and repeated
cached forecast runs created thousands of effectively duplicate publications.
These are model defects, not evidence that the dashboard hosts need more CPU or
memory.

## Version 7 corrections

1. The system-as-is load is anchored to the latest measured whole-station
   energy balance. Learned components are diagnostics and scenario inputs.
2. UAS effective tier is joined to the 15-minute operating state and learned
   separately by tier.
3. Battery integration uses fitted usable capacity, directional efficiencies,
   and observed charge/discharge limits.
4. All deterministic, ensemble, and scenario calculations use one solar
   calibration contract.
5. Ensemble members include weather, recent load-residual, and battery-model
   uncertainty.
6. Semantic publication signatures prevent unchanged runs from rewriting the
   forecast or inflating archive counts.
7. Runtime invariants reject unordered quantiles, a non-positive DC baseline,
   mismatched solar contracts, and an all-instruments tier-3 load below DC.

## UAS tier-3 scenario

`All instruments + UAS tier 3` keeps CL61, Radar, HATPRO, and UAS active for
the complete horizon and sets the UAS effective tier to 3. Tier-3 evidence is
considered mature only after at least three independent episodes and six
observed hours. Before that gate, the scenario is visibly provisional and uses
the historical fallback distribution P10/P50/P90 `55/108/302 W` for UAS.

## Promotion checks

The model may be published as advisory after its unit, API-contract, compile,
and documentation checks pass. It must not be treated as validated automation
until live evidence shows:

- current-load bias is materially closer to zero than the version-6 baseline;
- 0-6 h and 6-24 h SOC MAE are no worse than persistence;
- P10-P90 coverage trends toward `0.80` across independent ECMWF cycles;
- tier-3 maturity satisfies the episode/hour gate;
- deterministic, ensemble, and scenario solar-contract identifiers match;
- no duplicate forecast issues are added when only generation time changes.

The forecast remains advisory and never controls a PDU outlet.
