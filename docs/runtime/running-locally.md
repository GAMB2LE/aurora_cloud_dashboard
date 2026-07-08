# Running Locally

## Basic local serve

```bash
cd /opt/aurora-cloud-dashboard
source venv/bin/activate
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

- the Python environment from `venv`
- access to the configured raw and product roots if you want realistic data
- a websocket origin that matches your browser host

## Notes

- the deployed application expects `/etc/aurora-dashboard.env`
- many instrument views assume the real Zarr and quicklook trees already exist
- local development is easiest on the deployed host or a close mirror of it
- quicklook generators fall back to repo-local `quicklooks/` when
  `AURORA_QUICKLOOK_ROOT` is not set; source `/etc/aurora-dashboard.env` before
  manual deployed runs if the goal is to update the live dashboard products
