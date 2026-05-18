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
- `window_open`
- `interactive_view_update`
- `hatpro_render`
- `stacked_timeseries_render`
- `science_quicklook_render`
- `housekeeping_quicklook_render`
- `wxcam_interactive_render`
- `wxcam_calendar_day_view`
- `wxcam_calendar_sync`
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
- a cached latest quicklook is shown first when available, otherwise a loading
  skeleton is shown for uncached views
- the initial interactive render is deferred until the browser session is
  loaded, which keeps application startup from blocking on a full Plotly build
- the heavier 2D interactive plots use a coarse-first pass before a full detail
  pass replaces it
- inactive quicklook and operations tabs are lazy-loaded the first time the user
  opens them
- interactive freshness and availability bars are also delayed until after the
  page opens, so first paint does not wait for a full timestamp scan

## Session context

When available, events also carry:

- `session_id`
- `live_sessions`
- `server_sessions`
- `total_sessions`
- `session_age_s`
- `busy`

These fields are useful for understanding real browser behavior, including
multi-user overlap and concurrent browsing.
