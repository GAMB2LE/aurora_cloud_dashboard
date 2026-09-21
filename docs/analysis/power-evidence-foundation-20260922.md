# Forecast improvement: evidence foundation, 22 September 2026

## Confirmed defect

Live development archive metadata retained old `<U3` storage encodings after
concatenation. Complete strings became `for`, `pow`, `5e7`, and `ecm`. The
forecast model contract itself retained its full value. Empty deterministic
verification was therefore a data-integrity failure, not evidence of zero skill.

## Implemented

- Deterministic and ensemble archive writers use a shared staging writer that
  discards inherited encodings, preserves nanosecond datetimes, verifies all
  coordinates and textual provenance after reopening, and restores the prior
  archive if activation fails.
- Recovered 128 verified immutable forecast issues and 64 distinct source-cycle
  IDs without guessing missing metadata. These are archive inventory counts,
  not promotion-eligible counts; campaign identity, maturity and diversity gates
  still apply.
- Original damaged deterministic archive retained at
  `/data/aurora/dev-products/power/candidates/archive-repair-20260922/original-damaged-archive.zarr`.
- Separate live collector copies checksum-verified issues, admits only matching
  ensemble anchors/cycles and preserves comparator bytes. It runs in an isolated
  1500 MiB, 25%-CPU, network-disabled service, deferring during evaluator/replay
  activity. It writes only a private candidate tree, never products or controls.
- Capture and ensemble matching are not candidate generation, scientific
  acceptance, or grid-alignment acceptance. Collector manifests say so explicitly.

## Validation

100 tests passed in the server runtime under a 1500 MiB cgroup, including memory,
forecast, hybrid and new archive round-trip regression tests. Eight collector
tests cover idempotency, tampering, overlap, symlinks, concurrent locking, exact
anchors/cycles and read-only staging cleanup. Local bundled Python lacked xarray
and Dask, so no local-runtime pass is claimed.

## Operational boundaries

Only development timers were paused; the in-flight refresh was allowed to finish.
Both timers were restored after deployment. Production, frozen v10 evidence,
the old replay tree, wider evaluator and instrument controls are unchanged.

## Still required before promotion

1. Feed captured issues into a candidate worker with explicit source-latency
   evidence, frozen issue-time predictors, and separate matured-truth rescoring.
   This collector deliberately does not assert that it captures those predictors.
2. Confirm installed panel geometry and MPPT/charge-acceptance evidence before
   calibrating available PV. Do not learn uncurtailed PV from battery-limited
   delivered power.
3. Diagnose residual-training exclusions using recovered identities; fit and
   validate component/phase loads without relaxing exact-state safeguards.
4. Preserve matching baseline ensembles, propagate memberwise physical PV/load/
   battery uncertainty and calibrate with blocked held-out evidence.
5. Retain all original promotion thresholds: at least 30 paired independent
   cycles per bucket over 10 UTC days plus diversity and scientific gates.

## Recovery / rollback boundary

Pre-change forecast source: `5e702ed95c3023b9bd042fb23fb1f05d22ba700f`.
Initial archive fix: `28538138c456fda075e278e2fc9e686efc9cbadc`.
The root-only `dashboard.env.before` is retained in the recovery directory.
Do not print or commit that file; it contains unrelated operational secrets.

Rollback requires idle forecast writers and their exclusive generator lock.
Restore code and its matching provenance together; preserve the current archive
before replacing it. Returning to the old writer reintroduces the truncation
defect, so rollback is an operational escape hatch, not a preferred state.
