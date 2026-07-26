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
the mobile catalogue. Source-sync timer and service states also come from this
contract; the dashboard does not maintain or probe a parallel unit catalogue.
It must not infer pruning safety from display artefacts or enable, stop, or
repair archive writers.

If the contract is absent or invalid, the dashboard records the read failure
and continues serving other observability data. The infrastructure monitor is
the authoritative source for archive health.
