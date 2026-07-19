# Start Here

Aurora Cloud Dashboard is a read-only viewer for live and archived Aurora
station data. It is not the source system for instrument control. The dashboard
shows what the processing system has already received and made available.

## Choose the right site

- **Production**: `data.gamb2le.co.uk` is the stable public service.
- **Development**: `data-ocean.gamb2le.co.uk` is the public development
  service. Its banner identifies it as a live mirror, so it can be used to
  check recent data without becoming the operational source of truth.

Both sites are intended to show live data. Check the Operations Dashboard for
the deployment identity and mirror freshness before acting on a discrepancy.

## Use the dashboard

1. Start at **Overview** for the station snapshot and instrument state.
2. Open **Power** to check battery state, active loads, and SOC forecasts.
3. Use **Interactive Data Browser** for a chosen instrument and time window.
4. Use **Science Quicklooks** and **House Keeping Quicklooks** for generated
   products rather than waiting for an expensive interactive render.
5. Use **Operations Dashboard** to understand data age, stream state, and
   active alerts before treating a missing trace as an instrument fault.

## Contribute safely

The project has three separate repositories:

- `aurora_cloud_dashboard`: the Panel dashboard, product scripts, mobile API,
  tests, and this documentation.
- `aurora-dashboard-ios`: the native iPhone and iPad client of the mobile API.
- `aurora-cloud-infra`: host roles, services, deployment, and production or
  development policy.

Work on a feature in the dashboard repository first. Keep raw mirrors and
derived products separate: raw inputs are never rewritten by a UI change.
Run the local tests and documentation build before proposing a deployment.

For a local setup, see [Running locally](runtime/running-locally.md). For the
main modules, see [Core files](overview/core-files.md). For deployment policy,
use the infrastructure repository's
[production/development guide](https://github.com/GAMB2LE/aurora-cloud-infra/blob/main/docs/PRODUCTION_DEVELOPMENT.md).
