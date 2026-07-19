# Aurora Cloud Dashboard

The Aurora Cloud Dashboard is the read-only browser, mobile API, and derived
product tooling for the Aurora observing stack. It presents live and archived
instrument data without changing raw mirrored inputs.

## Start here

- New to the project: [documentation start page](docs/getting-started.md)
- How the pieces fit together: [architecture](docs/architecture.md)
- Dashboard views and instruments: [dashboard overview](docs/overview/index.md)
- Local development: [running locally](docs/runtime/running-locally.md)
- Product schemas and storage: [data products](docs/data-products/index.md)

The published documentation is built from `docs/` with MkDocs.

## Repository responsibilities

This repository owns:

- the Panel browser dashboard in `app.py`
- dashboard-facing product builders and quicklook generators
- the read-only mobile API in `mobile_api.py` and `mobile_catalog.py`
- unit tests and dashboard documentation

It does not own server configuration, timer ownership, deployment policy, or
the native iOS application. Those live in `aurora-cloud-infra` and
`aurora-dashboard-ios` respectively.

## First local run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dashboard.txt -r requirements-mobile-api.txt
python -m unittest discover -p 'test_*.py'
python check_docs.py
panel serve app.py --address 127.0.0.1 --port 5006 --allow-websocket-origin=localhost:5006
```

Without a configured read-only mirror of the Aurora products, the app can
validate its UI and empty-data handling but cannot render real observations.

## Safety and releases

Raw mirrors and derived products are separate by design. A browser or UI change
must not rewrite raw data. Development is deployed and checked before an
immutable production release tag is promoted. See the infrastructure
repository's
[production/development guide](https://github.com/GAMB2LE/aurora-cloud-infra/blob/main/docs/PRODUCTION_DEVELOPMENT.md)
for role ownership, deployment, and rollback.

## Checks

```bash
python -m py_compile app.py mobile_api.py mobile_catalog.py
python -m unittest discover -p 'test_*.py'
python check_docs.py
```

The GitHub workflow runs compilation, the unit suite, and a strict MkDocs
build. Keep user-facing behavior documented alongside any changed API or
product contract.
