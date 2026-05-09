# Performance Logging

The dashboard writes structured JSONL timing events to:

- `/data/aurora/products/dashboard/dashboard_perf.jsonl`

The log rotates automatically.

## Environment controls

- `AURORA_DASHBOARD_PERF_LOG`
- `AURORA_DASHBOARD_PERF_LOG_MAX_BYTES`
- `AURORA_DASHBOARD_PERF_LOG_BACKUP_COUNT`
- `AURORA_DASHBOARD_PERF_ENABLED`
- `AURORA_DASHBOARD_SESSION_HEARTBEAT_MS`

## Useful commands

```bash
tail -f /data/aurora/products/dashboard/dashboard_perf.jsonl
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 24
/opt/aurora-cloud-dashboard/venv/bin/python summarize_dashboard_perf.py --hours 6 --event interactive_view_update
```

## Logged event families

- `base_dataset_open`
- `dataset_time_bounds`
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
