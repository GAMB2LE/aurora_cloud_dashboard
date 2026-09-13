# Development forecast service memory repair — 13 September 2026

## Scope and incident

Repair the development `aurora-power-soc-forecast.service` memory failure. Do
not change physical-model parameters, evidence gates, instrument controls,
production, the frozen v10 baseline, or the wider model evaluator.

The job started at 21:09:45 UTC and the kernel killed generator PID 836058 at
21:12:56 UTC. This was host-wide OOM, not the service's 3500 MiB limit. The
victim had 2,466,848 KiB anonymous RSS. The cached 96-hour ECMWF GRIB download
had completed; deterministic output, ensemble generation and publication had
not. Candidate replay had finished at 21:00:57; the wider evaluator's next run
started at 21:20:30. Neither was running at the incident time.

The power store had approximately 7.9 million observations. Freshness checking
loaded all history and columns just to obtain the latest SOC. Forecast building
then loaded it again, copied normal sorted data during normalization, and made
additional full-frame copies for a short battery calibration window and
row-wise solar/load calculations. Reclaimable host slab also constrained
available memory; raising the service cap would not fix this host-wide failure.

## Changes

- Read only the latest finite SOC, backwards in bounded chunks; preserve the
  old normalization policy for duplicate, unsorted or invalid timestamps.
- Avoid unnecessary pandas copies for already normalized time series.
- Keep large complete telemetry frames in temporary file-backed float64 arrays,
  filled synchronously in 65,536-row chunks. Pages can be reclaimed by the OS.
  Temporary files are unlinked automatically; no retained telemetry copy is
  added to a public product or cache.
- Select the existing battery calibration window before making its table copy.
  Preserve the full-history observed-load fallback decision and all observations
  used for verification. No new history truncation or precision reduction.
- Sum solar channels in bounded batches with identical NaN/minimum-coverage
  behavior; construct DC-only diagnostics only within their existing window.
- A full-path replay uncovered stale Zarr chunk encodings after archive
  expansion. Rewritten archives now derive storage chunks from their new Dask
  layout, retaining safe-chunk validation rather than disabling it.
- Verification shares per-issue provenance strings across forecast steps instead
  of repeating large fixed-width Unicode buffers and then making string objects
  again during pandas construction. Values and filtering contracts are unchanged.

Source commits: `38b93aa`, `c287de9`, `49b0e28`, `5e702ed`, based on the clean actually
deployed revision `266ebe22d89a3660177394bcbef340cab48bfa4a`.

## Verification and boundaries

The first copy-reduction patch alone exceeded a 1500 MiB diagnostic cgroup cap.
The file-backed implementation completed deterministic generation under that
same enforced cap, using every row through a fixed 21:09 UTC cutoff and the
failed job's exact cached GRIB. It wrote 34 forecast samples in 3m20s at 50% CPU.
Do not equate process maximum RSS (which includes mapped pages) with cgroup
anonymous memory, or claim this deterministic-only test proves publication.

The 49b0e28 full-path replay passed forecast building and archive writing, then
hit the diagnostic cap during skill generation (5m27s). This was a contained
cgroup OOM, not a new host-wide OOM. The subsequent provenance-string change
addresses a further verification allocation; final replay results follow below.

Two initial probes using periodic asynchronous traceback dumps terminated with
SIGSEGV, not OOM. Subsequent probes omitted periodic dumps. No definitive
attribution of those diagnostic crashes was established.

The final local power suite passed 228 tests plus five subtests (118 seconds).
The focused forecast/memory suite passed 77 tests. All eight memory/archive
tests also passed in the server's existing runtime.

Final full-path replay `aurora-forecast-memory-full-v4-20260913.service` passed
with exit 0 in 6m48s, under the enforced 1500 MiB cap and 50% CPU quota. With the
effective environment loaded, it produced 82 forecast samples, a 771-issue /
82-step archive, 170 skill timestamps across three load modes and four lead
buckets, and 673 hindcast timestamps. Forecast SOC was finite and within
66.76–100%; the archive was ordered and had no duplicate issue times. Exact
forecast code provenance was verified after reopening all four products. The
hindcast contained observations only: existing identity filters were preserved,
not relaxed to manufacture matched forecast evidence.

The replay used fixed archived inputs, not live predictions. It demonstrates
execution and resource safety, not improved forecast skill or model promotion.

Live forecast and learning timers were paused only while their writers were
idle. Candidate replay and the wider evaluator were left unchanged. All replay
writes are under the isolated evidence root, with network disabled, read-only
source trees, 1500 MiB memory limit and 50% CPU quota.

## Recovery evidence

Host: `data-ocean.gamb2le.co.uk`.

Evidence root:
`/data/aurora/dev-products/power/candidates/forecast-memory-repair-20260913-pfvNCT`.

This contains the original service and runner, source archives, isolated
forecast/state/archive/skill/hindcast products and diagnostic scripts.
Original source archive SHA-256:
`d9e63f2059cde424757d0add888971ca7adb40fd1c9767f1cc985d9978cd9e99`.
The c287de9 source archive SHA-256:
`6e0c7c4d492f3c18511beea3d04fc1f58fb4093a5ff9ccc6d76033f1ba9db2b9`.
Intermediate 49b0e28 source archive SHA-256:
`01401fd37ef67811608f9c33ef71bcbce7c917e7b54e3d5d65f891df527205f7`.
Final 5e702ed source archive SHA-256:
`82fa36ee634c12d389552fdb7e9dae72b1883b4e91df9c119cade7a931859bed`.

## Deployment and recovery

At 22:09:41 UTC, the clean development checkout was advanced from 266ebe22 to
the exact tested/pushed `5e702ed95c3023b9bd042fb23fb1f05d22ba700f` using a
verified Git bundle. Only `AURORA_FORECAST_CODE_REVISION` changed in the effective
environment. Hashes excluding that line match before/after:
`764d921e58fbfebeccdca446984dd8ced38090d57c9ed6c3195e4e5a30ae9101`.
The original environment was backed up root-only; its SHA-256 is
`5e57ca93a7ebd77db5b042f404e2ce32e7866fb140ff049e4c6723b20eb405a7`.

Both forecast timers were restored and verified active. The normal full-cycle
service was started at 22:09:41 UTC. Existing live service limits were retained
(3500 MiB hard limit), because the same service also runs ensemble generation
and publication. Dashboard/mobile processes were not restarted. A subsequent
infrastructure deployment must retain this exact app revision, rather than
silently restoring an older inventory pin.

Cached refreshes had failed after the OOM because their forecast did not share
the latest immutable cycle. Recovery uses the regular full-cycle workflow to
restore cycle consistency; no publication guard is bypassed.

One-command source/provenance rollback, when the forecast writer is idle:

```sh
ssh root@data-ocean.gamb2le.co.uk 'bash /data/aurora/dev-products/power/candidates/forecast-memory-repair-20260913-pfvNCT/rollback.sh'
```

The rollback checks the exact expected revision, clean checkout, environment
line and exclusive writer lock. It restores code/provenance only, leaving
forecast evidence, products and timers intact. Reverse environment patch was
dry-run verified; rollback itself was not performed.

## Verified live outcome

The regular full-cycle service exited successfully at **22:20:47 UTC**, after
starting at 22:09:41 UTC. It generated the 34-point deterministic forecast,
50-member ensemble, 66-point planning forecast, operating scenarios, display
products and dashboard quicklooks. Whole-service memory peak was
**2,040,389,632 bytes (1.90 GiB)**. No kernel OOM was recorded during the live run.

The active development bundle is `20260913T221728Z-fd81426f`, completed at
**22:20:26.046566 UTC**. Reopened `generation.json` confirms:

- `status=complete`, exact forecast code revision 5e702ed;
- `controlAuthority=advisory_only`, `cl61ActuationEnabled=false`;
- `independentCycle=true` and a checksum-bound source manifest;
- manifest permissions 0444 before activation.

Source manifest digest:
`sha256:5139a2f4f3fb570b3e264f693f0206228148dd25c7bc225d05440df73bd744e5`.

Both timers remain active. The queued normal cached-refresh run started at
22:20:47 UTC. Consequently the status overlay returned to `assembling` for the
next attempt at 22:20:48 while retaining the newly completed active bundle;
this is not failure of the verified full-cycle publication.

Development source is clean at the tested revision. Production, frozen v10
evidence, instrument controls and the separate evaluator/replay services were
not changed. This repair establishes resource-safe execution on the current
full-history workload; it does not claim unbounded future memory scaling or
improved scientific forecast skill.
