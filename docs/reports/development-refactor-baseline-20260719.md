# Development dashboard refactor baseline

Recorded on 19 July 2026 before the staged internal simplification.

## Release

- Repository: `GAMB2LE/aurora_cloud_dashboard`
- Baseline commit: `6725454`
- Development tag: `dev-20260719.7`
- Development endpoint: `https://data-ocean.gamb2le.co.uk/app`
- Production was not changed.

## Behavioural contract

The desktop dashboard exposes Overview, Interactive Data Browser, Power,
Science Quicklooks, House Keeping Quicklooks, AURORACam, UAS, and Operations
Dashboard. The phone experience exposes the corresponding Overview, Power,
Plots, Camera, and Ops navigation while preserving direct desktop query links.

The public interfaces that must remain stable are documented in
`docs/architecture.md` and are enforced by dashboard-shell, mobile-API,
instrument-registry, and presentation-model tests.

## Measured baseline

- Initial HTTPS document: approximately 385 kB.
- Observed direct request: HTTP 200, about 1.42 s to first byte and 1.46 s total.
- Rolling 24 h development median reported by the health probe: about 2.1 s.
- Rolling 24 h development p95: about 3.9 s.
- Dashboard service after the preceding release: about 393 MB peak memory.
- Test baseline: 100 tests passed and 8 optional-data tests skipped.

The refactor must not regress these measurements. Live latency and memory vary
with active sessions and mirror I/O, so release comparisons use rolling
measurements rather than a single request.

## Known operational constraint

The former combined development mirror serially scanned roughly 209 GB of raw
data, 528 GB of products, internal metadata, and service state. A run observed
during the audit exceeded 18 minutes, preventing the intended five-minute
freshness cadence. The infrastructure release therefore separates these
stages, records stage metrics, and publishes dashboard freshness only after the
product stage succeeds.
