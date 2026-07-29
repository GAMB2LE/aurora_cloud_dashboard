# AURORA Dashboard Containerisation Plan

Status: Deferred until after 24 August 2026.

## Goal

Package the AURORA dashboard and mobile API into reproducible containers while
leaving live data, authoritative writers, host networking, and production
operations under their existing controls.

Containerisation should improve deployment consistency, dependency isolation,
rollback, and security. It is not, by itself, a performance fix.

## Proposed Architecture

- Keep Nginx, TLS certificates, SSH access, Tailscale, and GWS mounts on the
  host.
- Build one dashboard image and run it with two commands:
  - Panel dashboard service.
  - Uvicorn mobile API service.
- Keep raw data, Zarr products, catalogs, media, caches, and operational state
  outside the image on host-mounted storage.
- Keep existing source-sync, writer, append, forecast, quicklook, retention,
  alert, and housekeeping timers on the host during the first rollout.
- Manage container configuration and releases through `aurora-cloud-infra`.

## Image Contents

Include:

- `app.py`, `dart.py`, `model-evaluation.py`, `mobile_api.py`, and supporting
  dashboard modules.
- Pinned Python dependencies.
- Static assets and stylesheets.
- Health endpoints and build metadata.

Do not include:

- Raw data, Zarr stores, SQLite catalogs, camera media, tokens, credentials, or
  host keys.
- Nginx, Certbot, SSH, Tailscale, GWS mounts, or system-level writer timers.

Build requirements:

- Use a pinned Python base-image digest.
- Install from the repository's pinned runtime requirements.
- Copy dependency manifests before application source to improve build caching.
- Run as a dedicated non-root UID and GID.
- Add OCI labels for commit, version, source repository, and build date.
- Use a strict `.dockerignore`.
- Use a read-only root filesystem and a temporary filesystem for `/tmp`.

## Runtime Configuration

- Use Docker Compose, deployed and configured by Ansible.
- Bind the dashboard and API to loopback-only host ports.
- Keep public routing and WebSocket handling in host Nginx.
- Mount raw and product data read-only.
- Mount only named performance-log and API-cache paths as writable.
- Mount API tokens read-only at runtime; never bake them into an image.
- Drop all Linux capabilities and enable `no-new-privileges`.
- Define CPU and memory limits that protect the host without causing avoidable
  throttling.
- Add readiness and liveness checks for both services.
- Stop and replace containers gracefully so active browser sessions receive a
  controlled restart.

## Build And Release Pipeline

For every candidate:

1. Run formatting, compilation, unit tests, import smoke tests, and strict
   documentation builds.
2. Scan source and image layers for secrets.
3. Generate an SBOM and run dependency and container vulnerability scans.
4. Build the image once in CI.
5. Run the dashboard tests inside that exact image.
6. Publish to a private registry using immutable commit and release tags.
7. Sign the image and record its digest.
8. Deploy by digest rather than a mutable tag such as `latest`.

## Staged Rollout

### Phase 1: Image Creation

- Add the Dockerfile, Compose definition, health checks, and local test
  commands.
- Confirm the image can render all routes and serve the existing mobile API.
- Record image size, startup time, memory use, and request timings.

### Phase 2: Development Shadow Deployment

- Run containers on unused ports on data-ocean.
- Keep the current development services live.
- Compare desktop, phone-browser, and mobile-API responses.
- Verify direct links, WebSockets, camera media, forecasts, powered-off states,
  and performance logging.

### Phase 3: Development Cutover

- Point data-ocean Nginx at the containers.
- Retain the existing host services as the immediate rollback target.
- Observe errors, latency, memory, restarts, and data freshness for several
  days.

### Phase 4: Production Approval

- Produce a parity and performance report.
- Request explicit approval before touching JASMIN production.
- Build an annotated production release and deploy the exact signed image
  digest.

### Phase 5: Optional Worker Containers

Only after the web tier is stable, assess whether selected stateless batch jobs
benefit from containers. Do not move data writers merely for architectural
uniformity.

## Rollback

- Keep the previous host service definitions and Nginx upstream configuration.
- Roll back by restoring the previous upstream and restarting the original
  Panel and API services.
- Never roll back, delete, or rewrite raw data or product stores as part of an
  application rollback.
- Target a complete application rollback in under five minutes.

## Acceptance Criteria

- Desktop, mobile-browser, iOS API, direct-link, and WebSocket behaviour matches
  the current development deployment.
- Containers run as non-root with no added capabilities.
- The application cannot write to raw or production product paths.
- Images contain no credentials or live data.
- Builds are reproducible and identified by an immutable digest.
- Startup time, page latency, memory use, and API latency are no worse than the
  recorded baseline.
- Development operates successfully for several days before production is
  considered.
- Production remains untouched until an explicit approval gate.

## Questions To Recheck When Work Resumes

- Are Docker Engine and Compose supported and patched on both hosts?
- Which private registry and authentication method should be authoritative?
- What UID/GID should match the existing host data ownership?
- Which current cache and performance paths genuinely require write access?
- How much disk space is available for images and rollback versions?
- What are the latest baseline load, tab-switch, API, memory, and mirror-lag
  measurements?
- Should deployment remain Compose-based, or has an existing platform become a
  better operational fit?
