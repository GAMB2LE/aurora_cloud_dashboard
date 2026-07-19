# Architecture

The dashboard keeps source data, derived products, and user interfaces as
separate responsibilities. This lets products be rebuilt without modifying raw
mirrors and lets the browser and iOS app consume the same bounded data views.

## Data flow

```text
source hosts
    -> raw mirrors
    -> appenders and catalog builders
    -> derived Zarr, media, quicklooks, and forecast products
    -> Panel dashboard and mobile API
    -> browser and native iOS clients
```

Operations snapshots follow the same pattern. They collect health evidence,
write a raw snapshot and derived monitoring product, then present status in the
dashboard and mobile API. They report state; they do not control instruments.

## Repository boundaries

`aurora_cloud_dashboard` owns read paths, product builders, UI, and the mobile
API contract. `aurora-cloud-infra` owns host configuration, systemd units,
reverse-proxy configuration, and deployment roles. `aurora-dashboard-ios` owns
the native client and never reads dashboard files directly.

This separation is deliberate. Do not copy deployment overrides into the
dashboard source tree, and do not add source-writer logic to the native app.

## Production and development

Production is the authoritative live writer. Development mirrors the products
for public testing and can host explicitly isolated experimental outputs. A
release is tested on development before it is promoted as an immutable
production tag. The detailed role and rollback policy lives in the
infrastructure repository.

## Performance boundaries

The dashboard should open lightweight containers first and fetch expensive
products only when a user selects the relevant tab or time window. Prewarmed
Plotly JSON and generated quicklooks are preferred for common latest views.
The mobile API returns bounded summaries so the native app does not download
full Zarr datasets or media unless the user asks for them.
