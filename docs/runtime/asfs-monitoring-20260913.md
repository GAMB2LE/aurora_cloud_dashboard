# ASFS collection monitoring deployment

Base: production `db6bc869cef05254569f8e3599883062f2fddce0`.
The change reads product timestamps independently from service success and
archive parity, propagates missing/stale evidence to browser/mobile status,
and restores product-age alerts without claiming direct logger telemetry.

Before deployment, account for the existing Operations Zarr schema-expansion
path. New numeric metrics make `append_new_ops_monitor_to_zarr.py` move the
old product to `ops_monitor.backup_schema_YYYYMMDDTHHMMSSZ.zarr` and rebuild
from retained raw monitoring JSONL. This implementation does not change or
manually trigger that path. Coordinate its ordinary append timer and verify
the backup, rebuild resource use, final time coordinate, and newest metrics.
Do not remove the backup until the rebuilt product has been checked. Live
collector JSON, Operations status, and alerts work independently of this
historical trend rebuild.

Validation on 13 September 2026:

- 153 passed and 15 subtests passed across `test_asfs_storage_health.py`, `test_collection_freshness.py`,
  `test_mobile_catalog.py`, `test_ops_alerts.py`,
  `test_operations_storage_paths.py`, and `test_dashboard_shell.py`.
- One unrelated date-dependent power fixture was excluded:
  `test_power_bundle_status_validates_digests_and_observes_failed_retry`.
  Its fixed forecast ends 9 September; the identical failure was reproduced
  with unchanged production `mobile_catalog.py` and its original test.
- Python compilation and `git diff --check` passed. Ruff passes for the
  collection/collector/mobile/alert modules and new tests. `app.py` has four
  existing Ruff findings; unchanged production reproduces the same findings.

After rollout, inspect real ASFS science/sonic/gas sample timestamps and
cloud-product evidence fields in collector JSON. Confirm that sync/append
success alongside stale products yields red collection/Operations state,
that absent evidence degrades status, and that current PDU-off evidence
preserves intentional-off status. Verify actual alert state after its normal
evaluation; testing the evaluator does not prove an external message was sent.

The edge producer also publishes
`/home/aurora/data/asfs/logger_storage_health.json` during its ASFS sync.
The collector reads at most 64 KiB over the existing ASS SSH identity with a
15-second timeout, using `ASFS_LOGGER_SOURCE_HOST/USER` (the existing radar
host is a compatibility fallback). Deploy the producer from `realtime-scripts`
commit `588d267` or its descendant before validating this evidence. No new
infrastructure role, credentials, or logger privileges are required.

Schema version 1 records a filename-based FAT directory-slot estimate;
55,000 slots warns and 60,000 slots is critical against a 65,536-slot limit.
It is explicitly independent of authenticated CardStatus availability.
Missing, malformed, future, or older-than-two-hours evidence degrades status.
Unknown evidence has a 180-minute alert persistence rule; current pressure
thresholds alert immediately on evaluation. Browser/mobile Operations expose
this independently from current collection and archive delivery.
