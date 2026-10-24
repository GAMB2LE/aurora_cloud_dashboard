# Production, development and app fallback

I use `https://data.gamb2le.co.uk/app` for the operational dashboard. JASMIN
owns normal ingestion, operational products, archives and alerts. The public
dashboard at `https://data-ocean.gamb2le.co.uk/app` is development; it can show
experiments alongside mirrored production observations.

## Mobile endpoints

| Endpoint | Purpose |
| --- | --- |
| `https://data.gamb2le.co.uk/mobile/v1` | Normal operational API |
| `https://data-ocean.gamb2le.co.uk/mobile/v1` | Development API and candidate evidence |
| `https://data-ocean.gamb2le.co.uk/mobile/operational/v1` | Separately enabled read-only operational mirror |

The operational mirror uses a separate process, production replica paths and
its own cache. It must not read development forecasts or continuity products.
Its manifest retains schema version 3 and adds
`deployment.operationalMirror: true`, alongside `dataRole: live-mirror`.
The ordinary development API does not advertise this capability. Configuration
and activation belong to the infrastructure repository; this page is not proof
that the mirror has been activated or that its observations are fresh.

The production apps try JASMIN first. They may use the operational mirror after
a connection failure or server error, once its manifest passes their checks.
They show the mirror role and observation age, keep related data and media on
the same endpoint, and retry JASMIN on a later normal refresh. Authentication,
certificate and schema failures are not reasons to switch hosts. Custom
servers are not silently redirected to public infrastructure.

Media paths resolve beneath the endpoint that supplied the response. Browser
links from operational fallback stay on the canonical JASMIN dashboard because
the development browser may show experimental products.

## Releases and recovery

Completed changes, tests and documentation go to `main`. Development and
production each pin an exact tested revision. A code rollback restores the
application and configuration, not observations or archives.

The presentation release uses its own Python 3.12 environment. Pillow 13 fixes
the image-decoder advisories detected by the release audit and requires Python
3.11 or newer. Keep the existing processing environment separate: changing a
shared checkout or environment would also change the next scheduled science
job. The infrastructure presentation-release guide owns the service overrides.

I check source, product and API observation times before calling a stream
current. A responsive website, successful copy or recent file modification is
not enough. The infrastructure
[operator guide](https://github.com/GAMB2LE/aurora-cloud-infra/blob/main/docs/OPERATOR_QUICKSTART.md)
and [production/development contract](https://github.com/GAMB2LE/aurora-cloud-infra/blob/main/docs/PRODUCTION_DEVELOPMENT.md)
own service commands, mirror safeguards and promotion gates. App fallback does
not transfer ingestion or archive ownership.
