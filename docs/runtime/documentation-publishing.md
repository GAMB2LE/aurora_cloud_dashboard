# Documentation Publishing

This repo follows the GAMB2LE central documentation model described at:

- <https://gamb2le.pages.dev/documentation-docs/>

## Repo-side files

This repo carries the required repo-side pieces:

- `mkdocs.yml`
- `docs/`
- `.github/workflows/trigger-docs.yml`

## How publishing works

- `trigger-docs.yml` asks the central `GAMB2LE/mkdocs-portal` repo to rebuild
  the unified site at `https://gamb2le.pages.dev/`
- this repo no longer carries a repo-local GitHub Pages deployment workflow;
  the central portal is the only intended public documentation destination
- local checks can be run with `python3 check_docs.py`, which builds the MkDocs
  site in an isolated `.venv-docs` environment

## Required GitHub Actions secrets

- `APP_ID = 2899200`
- `APP_PRIVATE_KEY = <GitHub App private key from the portal workflow setup>`

## Current portal model

The central portal repo must:

- include this repository in its own `mkdocs.yml` navigation
- clone this repository inside its docs workflow before building the portal

## Practical note

The Aurora docs home page is intentionally self-contained rather than relying
on Markdown include plugins, so the central portal can render it without any
repo-specific include behavior.
