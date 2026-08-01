# Archive Health

The dashboard is a read-only consumer of archive state. Transfer, verification,
object-store, and pruning services are owned by `aurora-cloud-infra` and
`aurora-edge-infra`.

The cloud archive monitor publishes:

```text
/data/aurora/internal/archive_status/health-v1.json
```

`ARCHIVE_HEALTH_PATH` may override that path. The dashboard merges the stable
`metrics` object into its operations snapshot and exposes the source path in
the mobile catalogue. It must not infer pruning safety from display artefacts
or enable, stop, or repair archive writers.

## What the status means

The production backup path is edge source -> cloud raw mirror -> two
independent additive archives: JASMIN GWS and object storage. The development
live mirror and Proxmox/PBS guest backups do not count as raw archive parity.
ASS retention requires exact raw evidence in the cloud, GWS, and object store;
APS Power is non-prunable.

The contract deliberately separates current delivery from archive failure:

| Contract state | Dashboard meaning |
| --- | --- |
| `*_pending_upload_count` | Product files are inside their 30-hour settle window and are not yet required for parity. Treat as pending/in progress, not red parity. |
| `*_missing_count` or `*_mismatch_count` | Settled archive data is absent or differs. This is a real archive problem. |
| inventory running with a recent progress heartbeat | The sharded verifier is making progress; report age alone is not a stall. |
| running with a heartbeat older than five minutes | Inventory is stalled. |
| clean streak `1` | One clean complete report; stable parity still requires another distinct report. |
| `object_store_stable_parity_state=1` | Two complete clean reports established the global stability gate. |
| `*_prune_ready_state=1` | One raw stream has age-bounded candidates; this is not permission by itself. |

Raw object parity remains strict for files older than six hours. Products and
quicklooks younger than 30 hours are excluded from stable parity and reported
as pending. Once a product is older than 30 hours, a missing or mismatched copy
is a genuine failure. Product families are verified in bounded source-derived
shards rather than through one unbounded object-store root listing.

If the contract is absent or invalid, the dashboard records the read failure
and continues serving other observability data. The infrastructure monitor is
the authoritative source for archive health.

The complete writer, schedule, evidence, repair, and retention contract lives
in `aurora-cloud-infra/docs/ARCHIVE_SERVICES.md`; the edge permit validator is
documented in `aurora-edge-infra/docs/retention.md`.
