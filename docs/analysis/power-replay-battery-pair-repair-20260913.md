# Immutable-issue battery pairing repair, 13 September 2026

## Failure and cause

The isolated campaign replay repeatedly failed on issue
`2026-09-06T00:51:05.620263184` after successfully processing
`2026-09-06T00:25:45.116355713`. Both use the same ECMWF weather cycle.
The non-solar pairing validator correctly rejected differences in usable
capacity, both efficiencies, and both battery power limits.

The candidate seed copied load state and SOC bias, but did not carry the
archived battery model. A first/new cycle could fit the battery again, whereas
a subsequent same-cycle issue disabled calibration and fell back to defaults.
Neither is an acceptable solar/load ablation: the battery must be the model
recorded by that individual immutable baseline issue, not a refit, an earlier
candidate's state, or today's defaults.

## Repair

- Read the complete battery physics from each verified baseline issue before
  candidate generation. Missing, non-finite, unsupported, conflicting, or
  out-of-range parameters fail closed; no default or clipped substitution is
  accepted.
- Pass that battery model explicitly through the generator to its integrator
  for the physical-solar, load-residual and hybrid lanes, and the standalone
  physical-solar runner. It takes precedence over adaptive state and newly
  matured verification.
- Retain the exact non-solar pairing validators unchanged. Battery calibration
  and cached-cycle behavior of unpaired operational forecasts are unchanged.
- Use a new implementation-keyed candidate tree. Keep prior candidate results,
  original issues, recovered archive, frozen v10 evidence and public products
  unchanged. No CL61/PDU control is enabled or altered.

## Regression evidence

The new end-to-end test first reproduced the exact five-field live exception
on its second same-cycle issue. After repair, both issues complete all three
lanes and retain their own battery physics and calibration provenance.
The zero-residual load lane reproduces baseline SOC to `1e-6` points, neither
candidate refits the battery, and baseline tree checksums remain unchanged.

Additional tests cover new matured verification, disabled calibration, and
missing/invalid battery inputs. The targeted run passed all 61 tests; the full
`test_power_*.py` run passed 293 tests and five subtests in 132.92 seconds.
One existing NumPy extension-size runtime warning was emitted, with no failed
tests. The tests ran in the isolated local power-test environment.

This is an execution/pair-integrity repair, not evidence of improved forecast
skill or permission to promote. The independent-cycle/day, solar-truth,
geometry, ensemble and load-training evidence gates still apply.

## Deployment boundary and recovery evidence

Before switching the isolated service, its previous unit and append-only
history were copied into:
`/data/aurora/dev-products/power/candidates/battery-pair-repair-20260913-tLQfkG`.

- `replay-service-before.conf` SHA-256:
  `a94f226d22f28b605790e0d8587acc6ebff0ce09c3959668a59fc16d51ce3196`.
- `replay-history-before.jsonl` SHA-256:
  `b0ef08e6c59dba86f0c6022e030b23b68ed6a3269a86a956bbaeea1cc0e80838`.

The previous runtime remains `/opt/aurora-power-integrity-cecffb8`; the public
dashboard checkout remains `266ebe22d89a3660177394bcbef340cab48bfa4a` and is not
part of this deployment. The evaluator's running job is left undisturbed.

To roll back the isolated replay service, first wait for that service to be
idle, then run this single command as root on data-ocean:

```sh
cp /data/aurora/dev-products/power/candidates/battery-pair-repair-20260913-tLQfkG/replay-service-before.conf /etc/systemd/system/aurora-power-integrity-campaign-replay.service && systemctl daemon-reload
```

This restores the prior code pin without deleting either implementation's
candidate evidence or changing the timer, resource cap, or scientific gates.
