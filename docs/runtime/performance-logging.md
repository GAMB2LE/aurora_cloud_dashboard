# Performance Logging

The dashboard writes structured JSONL timing events to:

- `/data/aurora/products/dashboard/dashboard_perf.jsonl`

The log rotates automatically.

Operations monitoring records whether this log exists, how large it is, and
when it was last written. The Operations Dashboard shows that as the
`Dashboard perf log` card so stale browsing telemetry is visible alongside the
source-transfer checks. It also reads recent timed events directly and reports
the current render-performance summary, including p50, p95, the slowest event,
and the maximum live browser-session count seen in the sampled log window.
These performance signals are diagnostic only; they do not drive the
Operations Dashboard **Overall** action state or the health report's
`overall_level`.

## Environment controls

- `AURORA_DASHBOARD_PERF_LOG`
- `AURORA_DASHBOARD_PERF_LOG_MAX_BYTES`
- `AURORA_DASHBOARD_PERF_LOG_BACKUP_COUNT`
- `AURORA_DASHBOARD_PERF_ENABLED`
- `AURORA_DASHBOARD_SESSION_HEARTBEAT_MS`
- `AURORA_RENDER_DEBOUNCE_MS`
- `AURORA_INTERACTIVE_RENDER_CACHE_SIZE`
- `AURORA_INTERACTIVE_MAX_TIME_SAMPLES`
- `AURORA_POWER_INTERACTIVE_MAX_TIME_SAMPLES`
- `AURORA_MET_INTERACTIVE_MAX_TIME_SAMPLES`
- `AURORA_RADIATION_INTERACTIVE_MAX_TIME_SAMPLES`
- `AURORA_OPS_INTERACTIVE_MAX_TIME_SAMPLES`
- `AURORA_SUMMARY_COARSE_TIME_SAMPLES`
- `AURORA_POWER_LATEST_CACHE_ROUND_MINUTES`
- `AURORA_POWER_LATEST_CACHE_TOLERANCE_MINUTES`
- `AURORA_PREWARM_LATEST_CACHE_TOLERANCE_MINUTES`
- `AURORA_POWER_GENERAL_CACHE_ROUND_MINUTES`
- `AURORA_POWER_DISPLAY_ENERGY_FREQ`
- `POWER_DISPLAY_ENERGY_ZARR_PATH`
- `AURORA_INTERACTIVE_PREWARM_DIR`
- `AURORA_OPS_TREND_CACHE_TTL_MINUTES`
- `AURORA_OPS_TREND_DAYS`
- `AURORA_QUICKLOOK_MAX_TIME_SAMPLES`

## Useful commands

```bash
tail -f /data/aurora/products/dashboard/dashboard_perf.jsonl
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 24
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 6 --event interactive_view_update
```

## Logged event families

- `base_dataset_open`
- `dataset_time_bounds`
- `dataset_time_bounds_cache_hit`
- `interactive_render_deferred`
- `interactive_render_cache_hit`
- `interactive_render_debounced`
- `interactive_prewarm_load`
- `window_open`
- `power_display_energy_open`
- `power_display_energy_window`
- `interactive_view_update`
- `hatpro_render`
- `stacked_timeseries_render`
- `science_quicklook_render`
- `housekeeping_quicklook_render`
- `wxcam_interactive_render`
- `wxcam_calendar_options`
- `wxcam_calendar_day_view`
- `wxcam_calendar_sync`
- `operations_dashboard_render`
- `session_loaded`
- `session_heartbeat`
- `session_destroyed`
- `ui_selection_change`
- `plot_relayout`

The interactive browser also uses a few runtime behaviors that affect how these
events should be interpreted:

- per-instrument pane reuse keeps the last rendered view warm while a refresh is
  queued
- exact interactive views are cached for the most recent instrument/window
  combinations, so returning to a recent view can repaint immediately
- rapid widget-change bursts are debounced before rendering to avoid duplicate
  back-to-back Plotly builds
- dataset time bounds and latest timestamps are cached briefly
- the periodic metadata timer invalidates timestamp bounds only; open Zarr
  handles are reopened by the live-refresh path when they age past the configured
  data-refresh interval
- stale-render protection drops older queued renders before they can repaint the
  page
- the matching cached Science Quicklook is shown first when available,
  otherwise a loading skeleton is shown for uncached views
- the initial interactive render is deferred until the browser session is
  loaded, which keeps application startup from blocking on a full Plotly build
- the heavier 2D interactive plots use a coarse-first pass before a full detail
  pass replaces it
- Power interactive plots use the same display-time preparation and per-trace time
  downsampling approach as the quicklooks, with display-only sanity limits for
  impossible APS values
- Power cumulative-energy traces are read from a compact one-minute display
  Zarr when available. Latest Power, Meteorology, and Radiation interactive
  figures can be loaded from prewarmed Plotly JSON created by their quicklook
  generators
- the live Power 24 h window is rounded into 5-minute cache buckets, so a small
  latest-timestamp change does not force an immediate full rebuild
- fixed-summary instruments use instrument-specific display-time sample caps;
  this changes only the browser/PNG rendering density, not the ingested Zarr
  data
- inactive quicklook and operations tabs are lazy-loaded the first time the user
  opens them
- interactive freshness and availability bars are also delayed until after the
  page opens, so first paint does not wait for a full timestamp scan
- empty interactive views report the selected UTC window and suggest the next
  useful check instead of showing a bare blank plot

## Session context

When available, events also carry:

- `session_id`
- `live_sessions`
- `server_sessions`
- `total_sessions`
- `session_age_s`
- `busy`

`stacked_timeseries_render` events include phase timings for the main render
path:

- `source_open_ms`
- `combine_ms`
- `figure_build_ms`

These fields make it easier to distinguish slow Zarr reads from slow Plotly
figure construction.

These fields are useful for understanding real browser behavior, including
multi-user overlap and concurrent browsing.
