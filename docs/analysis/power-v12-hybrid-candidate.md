# Evaluation-first v12 hybrid power candidate

Status: development-only candidate.  It is not an operational forecast and it
does not replace the v11 development product or any production product.

## Purpose

v12 evaluates two bounded, issue-time-safe changes independently and together:

| Lane | Solar forcing | Load forcing |
| --- | --- | --- |
| B | Three-array physical PV model | Existing finite-state load profile |
| C | Existing solar model | Bounded load-residual correction |
| D | Three-array physical PV model | Bounded load-residual correction |

The physical PV lane resolves East, South and West arrays at ten-minute
substeps, conserves each ECMWF source-interval irradiance energy, and integrates
battery acceptance on that substep trace.  Its array configuration is explicitly
provisional until a surveyed bill of materials and uncurtailed MPPT evidence are
available.

The load lane fits a regularised residual only from forecast issues and observed
load available before the candidate issue time.  Candidate execution reads only
the issue-time-safe 21-day APS/PDU history needed by the seven/fourteen-day
calibration and verification windows; it never materialises the whole mirrored
power store.  It needs at least 48 valid
samples, three independent cycles and three UTC days; otherwise it emits a zero
correction with an `insufficient_issue_time_evidence` status.  Corrections are
shrunk and clipped to 500 W, and cannot reduce the forecast below a measured
or state-registry DC-only core-load floor.  This keeps a residual from erasing
the station's always-on electrical demand while preserving the finite-state
load model.

## Isolation and provenance

Each v12 lane writes only below:

```text
/data/aurora/dev-products/power/candidates/v12/lanes/<lane>/
```

The baseline latest forecast, archive, adaptive state, skill product and
hindcast remain read-only inputs.  A run accepts only an archive-eligible full
ECMWF-cycle baseline, then records the exact SOC anchor, cycle, forcing checksum,
physical configuration digest, source manifest, code revision and forecast
identity.  It fails closed if the baseline pairing, configured input checksum,
or protected path relationship is invalid.

Completed baseline/candidate pairs are immutable directories at:

```text
pairs/<pair-id>/<candidate-publication-signature>/
```

Only a manifest with `pair_status: complete` is valid evaluation input.  The
runner additionally writes an append-only hash-chained evaluation history,
`campaign_evidence.zarr`, the rolling `daily_diagnostic_skill.zarr`, a
`review_summary.json`, and a `not_accepted` acceptance record.

## Evaluation rule

Rolling diagnostics are health signals, not promotion evidence.  The required
promotion surface is cumulative paired campaign evidence, stratified by lead,
load state, source availability, degraded mode and an issue-time-safe cloud
proxy.  A candidate is retained until it has at least 30 independent paired
cycles in every lead bucket across at least ten UTC days and has passed the
specified SOC, solar, load, calibration, reproducibility, resource and API
gates.  There is no automatic promotion path in v12.

Direct solar skill remains withheld until Victron MPP mode (register 791) is
available to exclude charger-limited observations.  Ensemble and reserve-event
metrics are explicitly marked unavailable in this initial bounded candidate;
their absence prevents acceptance rather than being hidden in an aggregate.

## Development and iOS exposure

`aurora-power-v12-candidate.timer` is development-only, installed disabled, and
resource-limited to 25% CPU, 1 GiB soft memory and 1.5 GiB hard memory.  It
defers when the wider AURORA
model-evaluation service is active.  The mobile API exposes an additive,
read-only `/power/solar-evaluation` endpoint only when the development
candidate feature is enabled.  The iOS Development scheme is pinned to
`data-ocean` and renders it as “Development candidate — not operational”; the
operational `/power` schema and production app remain unchanged.
