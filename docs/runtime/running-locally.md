# Running Locally

## First local run

From a fresh clone, create an isolated environment and install the dashboard
and mobile API dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dashboard.txt -r requirements-mobile-api.txt
```

Run the fast checks before serving the app:

```bash
python -m unittest discover -p 'test_*.py'
python check_docs.py
```

## Basic local serve

```bash
cd aurora_cloud_dashboard
source .venv/bin/activate
panel serve app.py --address 127.0.0.1 --port 5006 --allow-websocket-origin=<host>
```

For local AURORACam image display, add a matching static route when you have a
local raw mirror:

```bash
panel serve app.py \
  --address 127.0.0.1 \
  --port 5006 \
  --allow-websocket-origin=<host> \
  --static-dirs=auroracam-media=/project/aurora/raw/auroracam
```

## What you need locally

- the Python environment created above
- access to the configured raw and product roots if you want realistic data
- a websocket origin that matches your browser host

## Notes

- the deployed application expects `/etc/aurora-dashboard.env`; do not copy
  production credentials into a local checkout
- without real Zarr and quicklook trees, the app should still import and show
  its empty-data states, but it cannot validate live instruments
- local development is easiest with a read-only mirror of the deployed data
- quicklook generators fall back to repo-local `quicklooks/` when
  `AURORA_QUICKLOOK_ROOT` is not set; source `/etc/aurora-dashboard.env` before
  manual deployed runs if the goal is to update the live dashboard products
