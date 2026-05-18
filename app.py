"""Aurora dashboard application.

This module hosts the multi-instrument Panel and Plotly browser for atmospheric
curtains, station summaries, WXcam media, quicklooks, and operations monitoring.
It keeps per-instrument state warm and uses cached bounds, stale-render
protection, coarse-first rendering on heavier 2D plots, and Power-specific
trace bucketing to keep the UI responsive during normal browsing.
"""

import asyncio
from base64 import b64encode
from collections import OrderedDict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
from html import escape
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import panel as pn
import param
from panel.io import hold
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from grouped_timeseries import (
    build_summary_plotly,
    calendar_date_tokens,
    combine_summary_datasets,
    default_calendar_label,
    default_interactive_label,
    display_name,
    housekeeping_label,
    housekeeping_daily_png,
    housekeeping_latest_png,
    is_summary_instrument,
    summary_daily_png,
    summary_latest_png,
    summary_source_instruments,
    SUMMARY_DISPLAY_END_ATTR,
    SUMMARY_DISPLAY_START_ATTR,
    POWER_BALANCE_LOOKBACK_DAYS,
    widget_group_options,
)
from extra_housekeeping import (
    extra_housekeeping_daily_png,
    extra_housekeeping_label,
    extra_housekeeping_latest_png,
    extra_housekeeping_tokens,
)
from wxcam_catalog import (
    WXCAM_IMAGE_TYPES,
    available_days,
    catalog_time_bounds,
    latest_record,
    representative_hourly_records,
)

pn.extension("plotly", notifications=True, sizing_mode="stretch_width")


def _static_asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }.get(suffix, "application/octet-stream")
    encoded = b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


DASHBOARD_LOGO = _static_asset_data_uri(Path(__file__).resolve().parent / "assets" / "logo.png")
DASHBOARD_FAVICON = "https://gamb2le.pages.dev/assets/logo.png"
THEME_TEXT = "#22313f"
THEME_MUTED = "#5f6c7b"
THEME_BORDER = "#d8e1e8"
THEME_LINE = "#c5d0da"
THEME_GRID = "#e5eaef"
THEME_PANEL = "#fbfcfd"
THEME_ACCENT = "#0b7285"


class WxcamVideoPlayer(pn.reactive.ReactiveHTML):
    src = param.String(default="")
    poster = param.String(default="")
    title = param.String(default="")
    subtitle = param.String(default="")
    mode_class = param.String(default="wxcam-player--wide")

    _template = """
    <div id="player_shell" class="wxcam-player ${mode_class}">
      <div id="meta_row" class="wxcam-player__meta">
        <div id="meta_text" class="wxcam-player__meta-text">
          <div id="title_text" class="wxcam-player__title">{{ title }}</div>
          <div id="subtitle_text" class="wxcam-player__subtitle">{{ subtitle }}</div>
        </div>
      </div>
      <div id="control_row" class="wxcam-player__controls">
        <button id="play_btn" type="button" onclick="${script('toggle_play')}">Play</button>
        <button id="back_btn" type="button" onclick="${script('jump_back')}">-0.5s</button>
        <button id="forward_btn" type="button" onclick="${script('jump_forward')}">+0.5s</button>
        <label id="speed_wrap" class="wxcam-player__inline-label">
          <span>Speed</span>
          <select id="speed_select" onchange="${script('change_speed')}">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </label>
        <label id="loop_wrap" class="wxcam-player__inline-label wxcam-player__checkbox">
          <input id="loop_toggle" type="checkbox" onchange="${script('toggle_loop')}"></input>
          <span>Loop</span>
        </label>
      </div>
      <div id="seek_row" class="wxcam-player__seek">
        <span id="current_time" class="wxcam-player__time">00:00</span>
        <input id="seek_slider" type="range" min="0" max="1000" value="0" step="1" oninput="${script('seek')}"></input>
        <span id="duration" class="wxcam-player__time">00:00</span>
      </div>
      <div id="video_frame" class="wxcam-player__frame">
        <video
          id="video_el"
          src="${src}"
          poster="${poster}"
          controls
          preload="metadata"
          playsinline
          onloadedmetadata="${script('sync_metadata')}"
          ontimeupdate="${script('sync_time')}"
          onplay="${script('sync_play_state')}"
          onpause="${script('sync_play_state')}"
          onended="${script('sync_play_state')}"
          onratechange="${script('sync_speed_state')}"
        ></video>
      </div>
    </div>
    """

    _scripts = {
        "src": """
          video_el.pause();
          video_el.load();
          seek_slider.value = 0;
          current_time.textContent = "00:00";
          duration.textContent = "00:00";
          play_btn.textContent = "Play";
          speed_select.value = "1";
        """,
        "toggle_play": """
          if (video_el.paused) {
            video_el.play();
          } else {
            video_el.pause();
          }
        """,
        "jump_back": """
          const nextTime = Math.max(0, (video_el.currentTime || 0) - 0.5);
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "jump_forward": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const candidateTime = (video_el.currentTime || 0) + 0.5;
          const nextTime = durationSeconds > 0 ? Math.min(durationSeconds, candidateTime) : candidateTime;
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "change_speed": """
          video_el.playbackRate = Number(speed_select.value || 1);
          view.run_script('sync_speed_state');
        """,
        "toggle_loop": """
          video_el.loop = loop_toggle.checked;
        """,
        "seek": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          if (!durationSeconds) {
            return;
          }
          video_el.currentTime = (Number(seek_slider.value || 0) / 1000) * durationSeconds;
          view.run_script('sync_time');
        """,
        "sync_metadata": """
          const formatTime = (seconds) => {
            if (!Number.isFinite(seconds) || seconds < 0) {
              return "00:00";
            }
            const total = Math.floor(seconds);
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const secs = total % 60;
            if (hours > 0) {
              return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
          };
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          current_time.textContent = formatTime(video_el.currentTime || 0);
          duration.textContent = formatTime(durationSeconds);
          seek_slider.value = 0;
          video_el.loop = loop_toggle.checked;
          video_el.playbackRate = Number(speed_select.value || 1);
          view.run_script('sync_play_state');
          view.run_script('sync_speed_state');
        """,
        "sync_time": """
          const formatTime = (seconds) => {
            if (!Number.isFinite(seconds) || seconds < 0) {
              return "00:00";
            }
            const total = Math.floor(seconds);
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const secs = total % 60;
            if (hours > 0) {
              return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
          };
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const currentSeconds = Number(video_el.currentTime || 0);
          current_time.textContent = formatTime(currentSeconds);
          duration.textContent = formatTime(durationSeconds);
          seek_slider.value = durationSeconds > 0 ? Math.round((currentSeconds / durationSeconds) * 1000) : 0;
        """,
        "sync_play_state": """
          play_btn.textContent = video_el.paused ? "Play" : "Pause";
        """,
        "sync_speed_state": """
          speed_select.value = String(video_el.playbackRate || 1);
        """,
    }

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
OPS_SNAPSHOT_PATH = Path(os.environ.get("OPS_MONITOR_SNAPSHOT_PATH", "/project/aurora/raw/ops_monitor/latest.json"))
PERF_LOG_ENABLED = os.environ.get("AURORA_DASHBOARD_PERF_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PERF_LOG_PATH = Path(os.environ.get("AURORA_DASHBOARD_PERF_LOG", "/data/aurora/products/dashboard/dashboard_perf.jsonl"))
PERF_LOG_MAX_BYTES = int(os.environ.get("AURORA_DASHBOARD_PERF_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
PERF_LOG_BACKUP_COUNT = int(os.environ.get("AURORA_DASHBOARD_PERF_LOG_BACKUP_COUNT", "5"))
SESSION_HEARTBEAT_MS = int(os.environ.get("AURORA_DASHBOARD_SESSION_HEARTBEAT_MS", "60000"))
_SESSION_BOOT_TS = datetime.now(timezone.utc)


def _session_id() -> str | None:
    try:
        doc = pn.state.curdoc
        if doc is None:
            return None
        session_context = doc.session_context
        if session_context is None:
            return None
        return session_context.id
    except Exception:
        return None


def _request_header(name: str) -> str | None:
    try:
        headers = pn.state.headers or {}
    except Exception:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() != wanted:
            continue
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)
    return None


def _request_path() -> str | None:
    try:
        doc = pn.state.curdoc
        if doc and doc.session_context and doc.session_context.request:
            return str(doc.session_context.request.path)
    except Exception:
        return None
    return None


def _request_query_args() -> dict[str, str]:
    try:
        doc = pn.state.curdoc
        if not doc or not doc.session_context or not doc.session_context.request:
            return {}
        query_args = getattr(doc.session_context.request, "query_arguments", {}) or {}
    except Exception:
        return {}
    parsed: dict[str, str] = {}
    for key, values in query_args.items():
        if not values:
            continue
        raw = values[0]
        if isinstance(raw, bytes):
            parsed[str(key)] = raw.decode("utf-8", errors="ignore")
        else:
            parsed[str(key)] = str(raw)
    return parsed


def _request_base_url() -> str:
    proto = _request_header("X-Forwarded-Proto") or "http"
    host = _request_header("Host") or "127.0.0.1:5006"
    path = _request_path() or "/app"
    return f"{proto}://{host}{path}"


def _client_ip() -> str | None:
    forwarded = _request_header("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = _request_header("X-Real-Ip")
    if real_ip:
        return real_ip.strip()
    try:
        doc = pn.state.curdoc
        if doc and doc.session_context and doc.session_context.request:
            remote_ip = getattr(doc.session_context.request, "remote_ip", None)
            if remote_ip:
                return str(remote_ip)
    except Exception:
        return None
    return None


def _server_session_count() -> int | None:
    try:
        doc = pn.state.curdoc
        if doc and doc.session_context and doc.session_context.server_context:
            return int(len(doc.session_context.server_context.sessions))
    except Exception:
        return None
    return None


def _live_session_count() -> int | None:
    try:
        return int((pn.state.session_info or {}).get("live", 0))
    except Exception:
        return None


def _total_session_count() -> int | None:
    try:
        return int((pn.state.session_info or {}).get("total", 0))
    except Exception:
        return None


def _session_age_seconds() -> float:
    return round((datetime.now(timezone.utc) - _SESSION_BOOT_TS).total_seconds(), 3)


def _normalize_perf_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value
        return dt.isoformat()
    if isinstance(value, timedelta):
        return round(value.total_seconds(), 6)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalize_perf_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_perf_value(item) for key, item in value.items()}
    return str(value)


_PERF_LOGGER = logging.getLogger("aurora.dashboard.perf")
_PERF_LOGGER.setLevel(logging.INFO)
_PERF_LOGGER.propagate = False
_PERF_LOG_READY = False

if PERF_LOG_ENABLED and not _PERF_LOGGER.handlers:
    try:
        PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _perf_handler = RotatingFileHandler(
            PERF_LOG_PATH,
            maxBytes=PERF_LOG_MAX_BYTES,
            backupCount=PERF_LOG_BACKUP_COUNT,
        )
        _perf_handler.setFormatter(logging.Formatter("%(message)s"))
        _PERF_LOGGER.addHandler(_perf_handler)
        _PERF_LOG_READY = True
    except Exception as exc:
        print(f"[perf] disabled: could not initialize {PERF_LOG_PATH}: {exc}")


def _perf_log(event: str, duration_ms: float | None = None, **fields) -> None:
    if not _PERF_LOG_READY:
        return
    instrument = fields.get("instrument", globals().get("CURRENT_INSTRUMENT"))
    session_id = fields.get("session_id", _session_id())
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "duration_ms": None if duration_ms is None else round(float(duration_ms), 3),
        "session_id": session_id,
        "instrument": instrument,
        "live_sessions": fields.get("live_sessions", _live_session_count()),
        "server_sessions": fields.get("server_sessions", _server_session_count()),
        "total_sessions": fields.get("total_sessions", _total_session_count()),
        "session_age_s": fields.get("session_age_s", _session_age_seconds()),
        "busy": fields.get("busy", bool(getattr(pn.state, "busy", False))),
    }
    for key, value in fields.items():
        record[key] = _normalize_perf_value(value)
    try:
        _PERF_LOGGER.info(json.dumps(record, sort_keys=True))
    except Exception as exc:
        print(f"[perf] write failed for {event}: {exc}")


@contextmanager
def _timed_perf(event: str, **fields):
    details = dict(fields)
    start = perf_counter()
    try:
        yield details
    except Exception as exc:
        details.setdefault("status", "error")
        details.setdefault("error_type", type(exc).__name__)
        details.setdefault("error", str(exc))
        raise
    finally:
        _perf_log(event, duration_ms=(perf_counter() - start) * 1000.0, **details)


def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, default))


# --- Configuration ---
INSTRUMENTS = {
    "Ceilometer": {
        "zarr_env": "CEILOMETER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr",
        "chunk_spec": {"time": 600},
        "consolidated": True,
        "height_load_max": 10_000,
        "top_range_default": 8000,
        "vars": {
            "beta_att": {"label": "Attenuated Backscatter", "clim": (1e-7, 1e-4), "log": True, "colorscale": "Cividis"},
            "linear_depol_ratio": {"label": "Linear Depolarization Ratio", "clim": (0.0, 0.5), "log": False, "colorscale": "Viridis"},
        },
        "default_top": "beta_att",
        "default_bottom": "linear_depol_ratio",
        "quicklook_dir": _path_from_env("CEILOMETER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "ceilometer"),
        "latest_image": _path_from_env("CEILOMETER_LATEST_IMAGE", QUICKLOOK_ROOT / "ceilometer" / "latest.png"),
    },
    "Cloud Radar": {
        "zarr_env": "CLOUD_RADAR_ZARR_PATH",
        "zarr_default": "/data/aurora/products/rpgfmcw94/cloud_radar.zarr",
        "chunk_spec": {"time": 400},
        "consolidated": True,
        "height_load_max": 9_000,
        "top_range_default": 9_000,
        "vars": {
            "ZE_dBZ": {"label": "ZE (dBZ)", "clim": (-30.0, 10.0), "log": False, "colorscale": "Cividis"},
            "ZE45_dBZ": {"label": "ZE45 (dBZ)", "clim": (-30.0, 10.0), "log": False, "colorscale": "Cividis"},
            "MeanVel": {"label": "Mean Velocity (m/s)", "clim": (-5.0, 5.0), "log": False, "colorscale": "RdBu_r"},
            "ZDR": {"label": "ZDR (dB)", "clim": (-10.0, 6.0), "log": False, "colorscale": "RdBu_r"},
            "SRCX": {"label": "SRCX", "clim": (0.0, 1.0), "log": False, "colorscale": "Viridis"},
            "SpecWidth": {"label": "Spectrum Width (m/s)", "clim": (0.0, 3.0), "log": False, "colorscale": "Plasma"},
            "SLDR": {"label": "SLDR (dB)", "clim": (-100.0, -10.0), "log": False, "colorscale": "RdBu_r"},
            "Skew": {"label": "Skew", "clim": (-2.0, 2.0), "log": False, "colorscale": "RdBu_r"},
            "RHV": {"label": "RHV", "clim": (0.8, 1.0), "log": False, "colorscale": "Viridis"},
            "PhiDP": {"label": "PhiDP (rad)", "clim": (-2.0, 2.0), "log": False, "colorscale": "RdBu_r"},
            "Kurt": {"label": "Kurtosis", "clim": (0.0, 8.0), "log": False, "colorscale": "Magma"},
            "KDP": {"label": "KDP (rad/km)", "clim": (-4.0, 4.0), "log": False, "colorscale": "RdBu_r"},
            "DiffAtt": {"label": "Differential Attenuation (dB/km)", "clim": (-5.0, 5.0), "log": False, "colorscale": "RdBu_r"},
        },
        "default_top": "ZE_dBZ",
        "default_bottom": "MeanVel",
        "quicklook_dir": _path_from_env("CLOUD_RADAR_QUICKLOOK_DIR", QUICKLOOK_ROOT / "cloud_radar"),
        "latest_image": _path_from_env("CLOUD_RADAR_LATEST_IMAGE", QUICKLOOK_ROOT / "cloud_radar" / "latest.png"),
    },
    "vaisalamet": {
        "zarr_env": "VAISALAMET_ZARR_PATH",
        "zarr_default": "/data/aurora/products/vaisalamet/vaisalamet.zarr",
        "chunk_spec": {"time": 1200},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("vaisalamet"),
        "default_top": default_interactive_label("vaisalamet"),
        "default_bottom": default_interactive_label("vaisalamet"),
        "default_calendar": default_calendar_label("vaisalamet"),
        "quicklook_dir": _path_from_env("VAISALAMET_QUICKLOOK_DIR", QUICKLOOK_ROOT / "vaisalamet"),
        "latest_image": _path_from_env("VAISALAMET_LATEST_IMAGE", QUICKLOOK_ROOT / "vaisalamet" / "latest.png"),
    },
    "asfs-logger": {
        "zarr_env": "ASFS_LOGGER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/asfs_logger/asfs_logger.zarr",
        "chunk_spec": {"time": 1200},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("asfs-logger"),
        "default_top": default_interactive_label("asfs-logger"),
        "default_bottom": default_interactive_label("asfs-logger"),
        "default_calendar": default_calendar_label("asfs-logger"),
        "quicklook_dir": _path_from_env("ASFS_LOGGER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_logger"),
        "latest_image": _path_from_env("ASFS_LOGGER_LATEST_IMAGE", QUICKLOOK_ROOT / "asfs_logger" / "latest.png"),
    },
    "asfs-fast-sonic": {
        "zarr_env": "ASFS_FAST_SONIC_ZARR_PATH",
        "zarr_default": "/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr",
        "chunk_spec": {"time": 24000},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("asfs-fast-sonic"),
        "default_top": default_interactive_label("asfs-fast-sonic"),
        "default_bottom": default_interactive_label("asfs-fast-sonic"),
        "default_calendar": default_calendar_label("asfs-fast-sonic"),
        "quicklook_dir": _path_from_env("ASFS_FAST_SONIC_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_fast_sonic"),
        "latest_image": _path_from_env("ASFS_FAST_SONIC_LATEST_IMAGE", QUICKLOOK_ROOT / "asfs_fast_sonic" / "latest.png"),
    },
    "power": {
        "zarr_env": "POWER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/power/power.zarr",
        # The power source is high-frequency. Larger dashboard read chunks avoid
        # thousands of tiny Dask tasks when building the latest 24 h browser view.
        "chunk_spec": {"time": 24000},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("power"),
        "default_top": default_interactive_label("power"),
        "default_bottom": default_interactive_label("power"),
        "default_calendar": default_calendar_label("power"),
        "quicklook_dir": _path_from_env("POWER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "power"),
        "latest_image": _path_from_env("POWER_LATEST_IMAGE", QUICKLOOK_ROOT / "power" / "latest.png"),
    },
    "ops-monitor": {
        "zarr_env": "OPS_MONITOR_ZARR_PATH",
        "zarr_default": "/data/aurora/products/ops_monitor/ops_monitor.zarr",
        "chunk_spec": {"time": 720},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("ops-monitor"),
        "default_top": default_interactive_label("ops-monitor"),
        "default_bottom": default_interactive_label("ops-monitor"),
        "default_calendar": default_calendar_label("ops-monitor"),
        "quicklook_dir": _path_from_env("OPS_MONITOR_QUICKLOOK_DIR", QUICKLOOK_ROOT / "ops_monitor"),
        "latest_image": _path_from_env("OPS_MONITOR_LATEST_IMAGE", QUICKLOOK_ROOT / "ops_monitor" / "latest.png"),
    },
    "wxcam": {
        "zarr_env": "WXCAM_ZARR_PATH",
        "zarr_default": "/data/aurora/products/wxcam/wxcam.zarr",
        "catalog_env": "WXCAM_CATALOG_PATH",
        "catalog_default": "/data/aurora/products/wxcam/wxcam_catalog.sqlite",
        "chunk_spec": {"time": 1},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": {
            spec["label"]: {
                "label": spec["label"],
                "image_type": image_type,
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            }
            for image_type, spec in WXCAM_IMAGE_TYPES.items()
        },
        "default_top": next((spec["label"] for spec in WXCAM_IMAGE_TYPES.values()), "FISH HDR"),
        "default_bottom": next((spec["label"] for spec in WXCAM_IMAGE_TYPES.values()), "FISH HDR"),
        "quicklook_dir": _path_from_env("WXCAM_QUICKLOOK_DIR", QUICKLOOK_ROOT / "wxcam"),
        "latest_image": _path_from_env("WXCAM_LATEST_IMAGE", QUICKLOOK_ROOT / "wxcam" / "latest.jpg"),
    },
    "Scanning Microwave Radiometer": {
        "zarr_env": "HATPRO_ZARR_PATH",
        "zarr_default": "/data/aurora/products/hatprog5/hatpro.zarr",
        "chunk_spec": {"time": 600},
        "consolidated": True,
        "height_load_max": 10_000,
        "top_range_default": 10_000,
        "vars": {
            "T_PROF": {"label": "Temperature Profile (K)", "clim": (210.0, 310.0), "log": False, "colorscale": "Inferno"},
        },
        "default_top": "T_PROF",
        "default_bottom": "T_PROF",
        "quicklook_dir": _path_from_env("HATPRO_QUICKLOOK_DIR", QUICKLOOK_ROOT / "hatpro"),
        "latest_image": _path_from_env("HATPRO_LATEST_IMAGE", QUICKLOOK_ROOT / "hatpro" / "latest.png"),
    },
}

INSTRUMENT_OPTIONS = {
    ("WXcam" if name == "wxcam" else display_name(name)): name
    for name in INSTRUMENTS.keys()
    if name not in {"asfs-fast-sonic", "ops-monitor"}
}
HK_INSTRUMENT_OPTIONS = {
    ("WXcam" if name == "wxcam" else display_name(name)): name
    for name in INSTRUMENTS.keys()
    if name != "asfs-fast-sonic"
}

DEFAULT_WINDOW = timedelta(hours=24)
LIVE_REFRESH_MS = 60_000  # how often to snap to latest when live is on (ms)
TIME_SUBSAMPLE = 2  # slice time to lighten payloads
TIME_TARGET = 300  # target max time samples for plotting
HEIGHT_TARGET = 200  # target max height samples for plotting
DATA_REFRESH_MS = 300_000  # reload base dataset every 5 minutes
RENDER_DEBOUNCE_MS = int(os.environ.get("AURORA_RENDER_DEBOUNCE_MS", "150"))
INTERACTIVE_RENDER_CACHE_SIZE = int(os.environ.get("AURORA_INTERACTIVE_RENDER_CACHE_SIZE", "12"))
POWER_INTERACTIVE_MAX_TIME_SAMPLES = int(os.environ.get("AURORA_POWER_INTERACTIVE_MAX_TIME_SAMPLES", "700"))
POWER_LATEST_CACHE_ROUND_MINUTES = int(os.environ.get("AURORA_POWER_LATEST_CACHE_ROUND_MINUTES", "5"))
POWER_GENERAL_CACHE_ROUND_MINUTES = int(os.environ.get("AURORA_POWER_GENERAL_CACHE_ROUND_MINUTES", "1"))
POWER_LATEST_CACHE_TOLERANCE = timedelta(minutes=int(os.environ.get("AURORA_POWER_LATEST_CACHE_TOLERANCE_MINUTES", "10")))
# A small future tolerance keeps normal clock skew harmless while protecting
# the dashboard from bogus outlier timestamps that can blank the latest window.
FUTURE_TIME_TOLERANCE = timedelta(days=2)
TIME_BOUNDS_CACHE_TTL = timedelta(seconds=45)
INTERACTIVE_PLACEHOLDER_HEIGHT = 540

_BASE_DS: dict[str, xr.Dataset | None] = {}
_TIME_BOUNDS_CACHE: dict[str, dict[str, object]] = {}
_INTERACTIVE_FIGURE_CACHE: dict[str, go.Figure] = {}
_INTERACTIVE_RENDER_CACHE: OrderedDict[tuple[object, ...], go.Figure] = OrderedDict()
_INSTRUMENT_VIEW_STATE: dict[str, dict[str, object]] = {}
_DATASET_VERSION: dict[str, int] = {}
_DATASET_REFRESHED_AT: dict[str, datetime] = {}
CURRENT_INSTRUMENT = "power"
_RENDER_REQUEST_COUNTER = 0
_ACTIVE_RENDER_REQUEST_ID = 0
_DISPLAYED_INTERACTIVE_INSTRUMENT: str | None = None
_PENDING_INTERACTIVE_RENDER_ARGS: tuple[object, ...] | None = None
_PENDING_INTERACTIVE_RENDER_CB = None
_APP_BOOTSTRAPPING = True
_INTERACTIVE_RENDER_ENABLED = False
_INTERACTIVE_FOOTER_LOADED = False
_POWER_DISPLAY_ENERGY_DS: xr.Dataset | None = None
_POWER_DISPLAY_ENERGY_REFRESHED_AT: datetime | None = None


def _safe_periodic_callback(callback, period: int, start: bool = True):
    """Register a Panel timer, but allow plain Python imports for smoke tests."""
    timer = pn.state.add_periodic_callback(callback, period=period, start=False)
    if start:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return timer
        try:
            timer.start()
        except RuntimeError as exc:
            # A normal `python -c "import app"` has no running asyncio loop.
            # `panel serve` provides one, so this only affects local smoke tests.
            if "no running event loop" not in str(exc):
                raise
    return timer


def _cfg(inst: str | None = None):
    return INSTRUMENTS[inst or CURRENT_INSTRUMENT]


def _zarr_path(inst: str | None = None):
    cfg = _cfg(inst)
    return os.environ.get(cfg["zarr_env"], cfg["zarr_default"])


def _power_display_energy_path() -> Path:
    return Path(os.environ.get("POWER_DISPLAY_ENERGY_ZARR_PATH", "/data/aurora/products/power/power_display_energy.zarr"))


def _prewarmed_interactive_dir() -> Path:
    return Path(os.environ.get("AURORA_INTERACTIVE_PREWARM_DIR", "/data/aurora/products/dashboard/prewarm"))


def _prewarmed_interactive_path(inst: str) -> Path:
    safe = inst.replace(" ", "_").replace("-", "_").lower()
    return _prewarmed_interactive_dir() / f"{safe}_latest_interactive.json"


def _wxcam_catalog_path(inst: str | None = None) -> Path:
    cfg = _cfg(inst)
    return Path(os.environ.get(cfg["catalog_env"], cfg["catalog_default"]))


def _wxcam_daily_video_root() -> Path:
    return Path(os.environ.get("WXCAM_DAILY_VIDEO_DIR", "/data/aurora/products/wxcam/daily_videos"))


def _wxcam_hourly_thumbnail_root() -> Path:
    return Path(os.environ.get("WXCAM_HOURLY_THUMB_DIR", "/data/aurora/products/wxcam/hourly_thumbnails"))


def _wxcam_media_root() -> Path:
    return Path(os.environ.get("WXCAM_MEDIA_ROOT", "/data/aurora/products/wxcam"))


def _wxcam_media_url(path: Path) -> str:
    """Return a browser URL for WXcam media served by Panel's static route."""
    root = _wxcam_media_root().resolve()
    resolved = path.resolve()
    rel = resolved.relative_to(root)
    prefix = os.environ.get("WXCAM_MEDIA_URL_PREFIX", "/wxcam-media").strip() or "/wxcam-media"
    prefix = "/" + prefix.strip("/")
    version = path.stat().st_mtime_ns
    return f"{prefix}/{quote(rel.as_posix())}?{urlencode({'v': version})}"


def _ensure_utc(dt):
    """Return a naive UTC datetime (or None) for consistent comparisons."""
    if dt is None:
        return None
    tz = getattr(dt, "tzinfo", None)
    if tz is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _get_base_dataset(inst: str | None = None):
    """Open the Zarr store (memoized per instrument) with configured chunks and consolidation."""
    inst = inst or CURRENT_INSTRUMENT
    if inst in _BASE_DS and _BASE_DS[inst] is not None:
        return _BASE_DS[inst]
    cfg = _cfg(inst)
    zarr_path = _zarr_path(inst)
    print(f"[base-ds] open {inst} -> {zarr_path}")
    with _timed_perf(
        "base_dataset_open",
        instrument=inst,
        zarr_path=zarr_path,
        requested_chunks=cfg["chunk_spec"],
        requested_consolidated=cfg["consolidated"],
    ) as perf:
        try:
            ds = xr.open_zarr(zarr_path, chunks=cfg["chunk_spec"], consolidated=cfg["consolidated"])
            perf["status"] = "configured"
        except Exception as first_exc:
            perf["status"] = "fallback"
            perf["configured_error"] = str(first_exc)
            try:
                ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False)
                perf["fallback_chunks"] = "auto"
                perf["fallback_consolidated"] = False
            except Exception as second_exc:
                perf["status"] = "unavailable"
                perf["fallback_error"] = str(second_exc)
                print(f"[base-ds] unavailable for {inst}: {first_exc}; fallback failed: {second_exc}")
                _BASE_DS[inst] = None
                return None
        perf["dims"] = dict(ds.sizes)
    _BASE_DS[inst] = ds
    _DATASET_REFRESHED_AT[inst] = datetime.now(timezone.utc)
    return ds


def _refresh_base_dataset(inst: str | None = None):
    """Drop the cached dataset so the next access reopens the Zarr (captures new data)."""
    inst = inst or CURRENT_INSTRUMENT
    ds = _BASE_DS.get(inst)
    if ds is not None:
        try:
            ds.close()
        except Exception:
            pass
    _BASE_DS[inst] = None
    _TIME_BOUNDS_CACHE.pop(inst, None)
    _DATASET_VERSION[inst] = _DATASET_VERSION.get(inst, 0) + 1
    _DATASET_REFRESHED_AT[inst] = datetime.now(timezone.utc)
    if inst == "power":
        _refresh_power_display_energy_dataset()


def _refresh_time_bounds_cache(inst: str | None = None):
    """Invalidate lightweight timestamp metadata without closing open Zarr datasets."""
    _TIME_BOUNDS_CACHE.pop(inst or CURRENT_INSTRUMENT, None)


def _dataset_cache_age(inst: str | None = None) -> timedelta:
    """Return age of the opened dataset handle; old handles are reopened during live refresh."""
    inst = inst or CURRENT_INSTRUMENT
    refreshed_at = _DATASET_REFRESHED_AT.get(inst)
    if refreshed_at is None or _BASE_DS.get(inst) is None:
        return timedelta.max
    return datetime.now(timezone.utc) - refreshed_at


def _get_power_display_energy_dataset() -> xr.Dataset | None:
    """Open the compact Power display-energy store when it is available."""
    global _POWER_DISPLAY_ENERGY_DS, _POWER_DISPLAY_ENERGY_REFRESHED_AT
    if _POWER_DISPLAY_ENERGY_DS is not None:
        return _POWER_DISPLAY_ENERGY_DS
    path = _power_display_energy_path()
    if not path.exists():
        return None
    with _timed_perf("power_display_energy_open", instrument="power", zarr_path=str(path)) as perf:
        try:
            ds = xr.open_zarr(path, chunks={"time": 1440}, consolidated=True)
        except Exception as exc:
            perf["status"] = "unavailable"
            perf["error"] = str(exc)
            return None
        perf["status"] = "ok"
        perf["dims"] = dict(ds.sizes)
    _POWER_DISPLAY_ENERGY_DS = ds
    _POWER_DISPLAY_ENERGY_REFRESHED_AT = datetime.now(timezone.utc)
    return ds


def _refresh_power_display_energy_dataset() -> None:
    """Drop the compact Power display-energy handle so latest products reopen."""
    global _POWER_DISPLAY_ENERGY_DS, _POWER_DISPLAY_ENERGY_REFRESHED_AT
    if _POWER_DISPLAY_ENERGY_DS is not None:
        try:
            _POWER_DISPLAY_ENERGY_DS.close()
        except Exception:
            pass
    _POWER_DISPLAY_ENERGY_DS = None
    _POWER_DISPLAY_ENERGY_REFRESHED_AT = None


def _open_power_display_energy_window(start, end) -> xr.Dataset | None:
    ds = _get_power_display_energy_dataset()
    if ds is None or "time" not in ds:
        return None
    start_dt = _as_naive_utc_datetime(start)
    end_dt = _as_naive_utc_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    with _timed_perf("power_display_energy_window", instrument="power", start=start_dt, end=end_dt) as perf:
        times = pd.DatetimeIndex(ds["time"].values)
        mask = (times >= start_dt) & (times <= end_dt)
        perf["matched_time_count"] = int(np.count_nonzero(mask))
        if not mask.any():
            perf["status"] = "empty"
            return None
        window = ds.isel(time=mask).sortby("time")
        perf["status"] = "ok"
        perf["output_time_count"] = int(window.sizes.get("time", 0))
        return window


def _remember_time_bounds(inst: str, lower: datetime | None, upper: datetime | None) -> tuple[datetime | None, datetime | None]:
    _TIME_BOUNDS_CACHE[inst] = {
        "captured_at": datetime.now(timezone.utc),
        "bounds": (lower, upper),
    }
    return lower, upper


def _valid_time_mask(times: np.ndarray) -> np.ndarray:
    """Mask out NaT and clearly bogus future timestamps while preserving original indices."""
    if times.size == 0:
        return np.zeros(times.shape, dtype=bool)
    valid = ~np.isnat(times)
    cutoff = np.datetime64(_ensure_utc(datetime.now(timezone.utc) + FUTURE_TIME_TOLERANCE))
    valid &= times <= cutoff
    return valid if np.any(valid) else ~np.isnat(times)


def _median_filter_nan(arr, k=3):
    """Simple nan-aware median filter with square window k x k."""
    if arr.ndim != 2 or k < 2:
        return arr
    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return np.nanmedian(windows, axis=(-2, -1))


def _dataset_time_bounds(inst: str | None = None):
    """Compute earliest and latest timestamps in the dataset (or None/None)."""
    inst = inst or CURRENT_INSTRUMENT
    cached = _TIME_BOUNDS_CACHE.get(inst)
    if cached:
        captured_at = cached.get("captured_at")
        bounds = cached.get("bounds")
        if isinstance(captured_at, datetime) and isinstance(bounds, tuple):
            if datetime.now(timezone.utc) - captured_at <= TIME_BOUNDS_CACHE_TTL:
                _perf_log("dataset_time_bounds_cache_hit", instrument=inst)
                return bounds
    with _timed_perf("dataset_time_bounds", instrument=inst) as perf:
        if inst == "wxcam":
            lower, upper = catalog_time_bounds(_wxcam_catalog_path(inst))
            perf["source"] = "wxcam_catalog"
            perf["time_start"] = lower
            perf["time_end"] = upper
            return _remember_time_bounds(inst, lower, upper)
        ds = _get_base_dataset(inst)
        if ds is None or "time" not in ds:
            perf["status"] = "no_dataset"
            return _remember_time_bounds(inst, None, None)
        times = np.asarray(ds["time"].values)
        perf["raw_time_count"] = int(times.size)
        if times.size == 0:
            perf["status"] = "empty"
            return _remember_time_bounds(inst, None, None)
        valid = _valid_time_mask(times)
        times = times[valid]
        perf["valid_time_count"] = int(times.size)
        lower = pd.Timestamp(times.min()).to_pydatetime(warn=False)
        upper = pd.Timestamp(times.max()).to_pydatetime(warn=False)
        perf["time_start"] = lower
        perf["time_end"] = upper
        return _remember_time_bounds(inst, lower, upper)


def _instrument_time_index(inst: str) -> pd.DatetimeIndex:
    if inst == "wxcam":
        return pd.DatetimeIndex([])
    ds = _get_base_dataset(inst)
    if ds is None or "time" not in ds:
        return pd.DatetimeIndex([])
    times = np.asarray(ds["time"].values)
    if times.size == 0:
        return pd.DatetimeIndex([])
    valid = _valid_time_mask(times)
    return pd.DatetimeIndex(times[valid])


def _format_status_time(dt: datetime | None) -> str:
    if dt is None:
        return "No data"
    stamp = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return stamp.strftime("%H:%M UTC")


def _format_duration(delta: timedelta | None) -> str:
    if delta is None:
        return "n/a"
    total_seconds = max(int(delta.total_seconds()), 0)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _hourly_coverage_summary(times: pd.DatetimeIndex, start: datetime | None, end: datetime | None) -> tuple[list[bool], int, int]:
    if start is None or end is None or end < start:
        return [], 0, 0
    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).floor("h")
    expected = pd.date_range(start=start_ts, end=end_ts, freq="1h")
    if len(expected) == 0:
        return [], 0, 0
    covered = set(pd.DatetimeIndex(times).floor("h"))
    bits = [stamp in covered for stamp in expected]
    missing = sum(1 for bit in bits if not bit)
    return bits, missing, len(expected)


def _binned_time_coverage(times: pd.DatetimeIndex, start: datetime | None, end: datetime | None, segments: int = 64) -> list[bool]:
    if start is None or end is None or end <= start or len(times) == 0:
        return []
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    if end_ns <= start_ns:
        return []
    time_ns = pd.DatetimeIndex(times).asi8
    mask = (time_ns >= start_ns) & (time_ns <= end_ns)
    if not np.any(mask):
        return [False] * segments
    window = time_ns[mask]
    edges = np.linspace(start_ns, end_ns, segments + 1)
    bits: list[bool] = []
    for idx in range(segments):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == segments - 1:
            bits.append(bool(np.any((window >= lo) & (window <= hi))))
        else:
            bits.append(bool(np.any((window >= lo) & (window < hi))))
    return bits


def _availability_hour_titles(start: datetime | None, end: datetime | None) -> list[str]:
    if start is None or end is None or end < start:
        return []
    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).floor("h")
    expected = pd.date_range(start=start_ts, end=end_ts, freq="1h")
    return [f"{stamp.strftime('%H:00')} UTC" for stamp in expected]


def _availability_binned_titles(start: datetime | None, end: datetime | None, segments: int) -> list[str]:
    if start is None or end is None or end <= start or segments <= 0:
        return []
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    if end_ns <= start_ns:
        return []
    edges = np.linspace(start_ns, end_ns, segments + 1)
    labels: list[str] = []
    for idx in range(segments):
        lo = pd.to_datetime(int(edges[idx]), utc=True)
        hi = pd.to_datetime(int(edges[idx + 1]), utc=True)
        labels.append(f"{lo.strftime('%m-%d %H:%M')} to {hi.strftime('%m-%d %H:%M')} UTC")
    return labels


def _availability_bucket_text(start: datetime | None, end: datetime | None, count: int) -> str:
    if start is None or end is None or count <= 0 or end <= start:
        return "Each block shows one slice of the selected range."
    seconds = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / count
    if seconds >= 3600:
        hours = seconds / 3600
        if abs(hours - round(hours)) < 0.15:
            value = int(round(hours))
            unit = "hour" if value == 1 else "hours"
            return f"Each block represents {value} {unit} of the selected range."
        return f"Each block represents about {hours:.1f} hours of the selected range."
    minutes = max(int(round(seconds / 60)), 1)
    unit = "minute" if minutes == 1 else "minutes"
    return f"Each block represents about {minutes} {unit} of the selected range."


def _wxcam_hour_bits(selection: str, day_token: str) -> list[bool]:
    rows_by_hour = _wxcam_hourly_image_rows(selection, day_token)
    return [hour in rows_by_hour for hour in range(24)]


def _wxcam_combined_hour_states(day_token: str) -> list[int]:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return [0] * 24
    records_by_type = {
        image_type: representative_hourly_records(
            _wxcam_catalog_path("wxcam"), image_type, day_utc, media_kind="image"
        )
        for image_type in WXCAM_IMAGE_TYPES
    }
    required_types = max(len(records_by_type), 1)
    states: list[int] = []
    for hour in range(24):
        present_types = sum(1 for rows in records_by_type.values() if hour in rows)
        states.append(2 if present_types == required_types else 1 if present_types else 0)
    return states


def _availability_bar_markup(
    states: list[int | bool],
    start_label: str,
    end_label: str,
    caption: str,
    explainer: str,
    full_label: str = "Data present",
    partial_label: str = "Partial coverage",
    empty_label: str = "No data",
    segment_titles: list[str] | None = None,
) -> str:
    if not states:
        return "<div class='availability-shell'><div class='availability-empty'>No availability information</div></div>"
    parts = []
    for idx, state in enumerate(states):
        if state in (True, 2):
            cls = "availability-segment availability-segment--full"
            state_label = full_label
        elif state == 1:
            cls = "availability-segment availability-segment--partial"
            state_label = partial_label
        else:
            cls = "availability-segment availability-segment--empty"
            state_label = empty_label
        title = state_label
        if segment_titles and idx < len(segment_titles):
            title = f"{segment_titles[idx]}: {state_label}"
        parts.append(f"<span class='{cls}' title='{escape(title)}'></span>")

    legend_parts = [
        "<span class='availability-legend-item'>"
        "<span class='availability-legend-swatch availability-segment--full'></span>"
        f"{escape(full_label)}"
        "</span>"
    ]
    if any(state == 1 for state in states):
        legend_parts.append(
            "<span class='availability-legend-item'>"
            "<span class='availability-legend-swatch availability-segment--partial'></span>"
            f"{escape(partial_label)}"
            "</span>"
        )
    legend_parts.append(
        "<span class='availability-legend-item'>"
        "<span class='availability-legend-swatch availability-segment--empty'></span>"
        f"{escape(empty_label)}"
        "</span>"
    )
    return (
        "<div class='availability-shell'>"
        f"<div class='availability-caption'>{escape(caption)}</div>"
        f"<div class='availability-explainer'>{escape(explainer)}</div>"
        f"<div class='availability-bar'>{''.join(parts)}</div>"
        f"<div class='availability-scale'><span>{escape(start_label)}</span><span>{escape(end_label)}</span></div>"
        f"<div class='availability-legend'>{''.join(legend_parts)}</div>"
        "</div>"
    )


def _status_strip_markup(items: list[tuple[str, str, str]]) -> str:
    pills = []
    for label, value, tone in items:
        pills.append(
            f"<span class='status-pill status-pill--{escape(tone)}'><strong>{escape(label)}</strong> {escape(value)}</span>"
        )
    return f"<div class='status-strip'>{''.join(pills)}</div>" if pills else ""


OPS_STREAM_SPECS = (
    {
        "label": "Ceilometer",
        "stream_prefix": "cl61",
        "source_key": "cl61_source_sync_service_healthy_state",
        "processing_keys": (
            "ceilometer_append_service_healthy_state",
            "ceilometer_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Cloud Radar",
        "stream_prefix": "radar",
        "source_key": "radar_source_sync_service_healthy_state",
        "processing_keys": (
            "radar_append_service_healthy_state",
            "radar_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "HATPRO",
        "stream_prefix": "hatpro",
        "source_key": "hatpro_source_sync_service_healthy_state",
        "processing_keys": (
            "hatpro_append_service_healthy_state",
            "hatpro_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Meteorology",
        "stream_prefix": "vaisalamet",
        "source_key": "vaisalamet_source_sync_service_healthy_state",
        "processing_keys": (
            "vaisalamet_append_service_healthy_state",
            "vaisalamet_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Radiation",
        "stream_prefix": "asfs_logger",
        "source_key": "asfs_logger_source_sync_service_healthy_state",
        "processing_keys": (
            "asfs_logger_append_service_healthy_state",
            "asfs_logger_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "ASFS Fast Sonic",
        "stream_prefix": "asfs_fast_sonic",
        "source_key": "asfs_fast_sonic_source_sync_service_healthy_state",
        "processing_keys": (
            "asfs_fast_sonic_append_service_healthy_state",
            "asfs_fast_sonic_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Aurora Power Supply",
        "stream_prefix": "power",
        "source_key": "power_source_sync_service_healthy_state",
        "processing_keys": (
            "power_append_service_healthy_state",
            "power_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "WXcam",
        "stream_prefix": "wxcam",
        "source_key": "wxcam_source_sync_service_healthy_state",
        "processing_keys": (
            "wxcam_append_service_healthy_state",
            "wxcam_catalog_service_healthy_state",
            "wxcam_daily_videos_service_healthy_state",
        ),
    },
)


def _ops_snapshot_path() -> Path:
    return OPS_SNAPSHOT_PATH


def _ops_read_snapshot() -> dict:
    path = _ops_snapshot_path()
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        snapshot = json.loads(path.read_text())
    except Exception as exc:
        return {"_error": str(exc), "_path": str(path)}
    snapshot["_path"] = str(path)
    return snapshot


def _ops_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _ops_bool(value) -> bool | None:
    number = _ops_float(value)
    if number is None:
        return None
    return bool(number)


def _ops_timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime(warn=False)


def _ops_level_value(level: str) -> int:
    order = {"green": 0, "amber": 1, "red": 2, "gray": -1}
    return order.get(level, -1)


def _ops_worst_level(levels: list[str]) -> str:
    meaningful = [level for level in levels if level != "gray"]
    if not meaningful:
        return "gray"
    return max(meaningful, key=_ops_level_value)


def _ops_level_from_bool(value) -> str:
    state = _ops_bool(value)
    if state is None:
        return "gray"
    return "green" if state else "red"


def _ops_level_from_count(value, amber_at: float = 1.0) -> str:
    count = _ops_float(value)
    if count is None:
        return "gray"
    if count <= 0:
        return "green"
    if count <= amber_at:
        return "amber"
    return "red"


def _ops_level_from_used_pct(value) -> str:
    used_pct = _ops_float(value)
    if used_pct is None:
        return "gray"
    if used_pct < 75.0:
        return "green"
    if used_pct < 90.0:
        return "amber"
    return "red"


def _ops_level_from_age_minutes(value) -> str:
    age_min = _ops_float(value)
    if age_min is None:
        return "gray"
    if age_min <= 10.0:
        return "green"
    if age_min <= 30.0:
        return "amber"
    return "red"


def _ops_level_from_source_probes(fail_count_value, total_hosts: int = 3) -> str:
    fail_count = _ops_float(fail_count_value)
    if fail_count is None:
        return "gray"
    if fail_count <= 0:
        return "green"
    if fail_count < total_hosts:
        return "amber"
    return "red"


def _ops_level_from_battery_voltage(value) -> str:
    voltage = _ops_float(value)
    if voltage is None:
        return "gray"
    if voltage > 52.0:
        return "green"
    if voltage >= 50.0:
        return "amber"
    return "red"


def _ops_level_from_battery_soc(value) -> str:
    soc = _ops_float(value)
    if soc is None:
        return "gray"
    if soc >= 50.0:
        return "green"
    if soc >= 25.0:
        return "amber"
    return "red"


def _ops_level_from_internal_temp(value) -> str:
    temperature = _ops_float(value)
    if temperature is None:
        return "gray"
    if temperature < 40.0:
        return "green"
    if temperature < 45.0:
        return "amber"
    return "red"


def _ops_level_from_perf_log(snapshot: dict) -> str:
    exists = _ops_bool(snapshot.get("dashboard_perf_log_exists_state"))
    age_min = _ops_float(snapshot.get("dashboard_perf_log_age_min"))
    if exists is False:
        return "red"
    if age_min is None:
        return "gray"
    if age_min <= 30.0:
        return "green"
    if age_min <= 360.0:
        return "amber"
    return "red"


def _ops_source_freshness_text(snapshot: dict, prefix: str) -> str:
    recent = _ops_bool(snapshot.get(f"{prefix}_source_recent_state"))
    age_min = _ops_float(snapshot.get(f"{prefix}_source_age_min"))
    if recent is None:
        return "No source timestamp"
    if age_min is None:
        return "Recent" if recent else "Stale"
    age_text = _format_duration(timedelta(minutes=age_min))
    return f"Recent ({age_text})" if recent else f"Stale ({age_text})"


def _ops_battery_text(snapshot: dict) -> tuple[str, str]:
    voltage = _ops_float(snapshot.get("aps_battery_voltage_v"))
    age_min = _ops_float(snapshot.get("aps_battery_voltage_age_min"))
    if voltage is None:
        return "No data", "Aurora Power Supply DC inverter voltage unavailable"
    value = f"{voltage:.2f} V"
    if age_min is None:
        return value, "Aurora Power Supply DC inverter voltage"
    age_text = _format_duration(timedelta(minutes=age_min))
    return value, f"Aurora Power Supply DC inverter voltage, {age_text} old"


def _ops_battery_soc_text(snapshot: dict) -> tuple[str, str]:
    soc = _ops_float(snapshot.get("aps_battery_soc_pct"))
    age_min = _ops_float(snapshot.get("aps_battery_soc_age_min"))
    if soc is None:
        return "No data", "Aurora Power Supply battery state of charge unavailable"
    value = f"{soc:.0f} %"
    if age_min is None:
        return value, "Aurora Power Supply battery state of charge"
    age_text = _format_duration(timedelta(minutes=age_min))
    return value, f"Aurora Power Supply battery state of charge, {age_text} old"


def _ops_internal_temp_text(snapshot: dict) -> tuple[str, str]:
    temperature = _ops_float(snapshot.get("aps_internal_temp_c"))
    age_min = _ops_float(snapshot.get("aps_internal_temp_age_min"))
    if temperature is None:
        return "No data", "Aurora Power Supply internal temperature unavailable"
    value = f"{temperature:.1f} C"
    if age_min is None:
        return value, "Aurora Power Supply internal temperature"
    age_text = _format_duration(timedelta(minutes=age_min))
    return value, f"Aurora Power Supply internal temperature, {age_text} old"


def _ops_perf_log_text(snapshot: dict) -> tuple[str, str]:
    exists = _ops_bool(snapshot.get("dashboard_perf_log_exists_state"))
    age_min = _ops_float(snapshot.get("dashboard_perf_log_age_min"))
    size_mb = _ops_float(snapshot.get("dashboard_perf_log_size_mb"))
    path = snapshot.get("dashboard_perf_log_path") or "/data/aurora/products/dashboard/dashboard_perf.jsonl"
    if exists is False:
        return "Missing", f"Expected {path}"
    size_text = "" if size_mb is None else f", {size_mb:.1f} MB"
    if age_min is None:
        return "Unknown", f"{path}{size_text}"
    age_text = _format_duration(timedelta(minutes=age_min))
    return f"{age_text} old", f"{path}{size_text}"


def _ops_perf_summary(path: Path, hours: float = 24.0, max_rows: int = 5000) -> dict:
    if not path.exists():
        return {"level": "red", "value": "Missing", "meta": f"Expected {path}"}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            tail = deque(handle, maxlen=max_rows)
        for line in tail:
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = _ops_timestamp(row.get("ts_utc"))
            if ts is None or ts < cutoff:
                continue
            rows.append(row)
    except Exception as exc:
        return {"level": "red", "value": "Unreadable", "meta": str(exc)}

    durations = sorted(float(row["duration_ms"]) for row in rows if isinstance(row.get("duration_ms"), (int, float)))
    if not durations:
        return {"level": "gray", "value": "No timing samples", "meta": f"No timed events in the last {hours:g} h"}

    def quantile(q: float) -> float:
        if len(durations) == 1:
            return durations[0]
        position = (len(durations) - 1) * q
        low = int(np.floor(position))
        high = int(np.ceil(position))
        if low == high:
            return durations[low]
        return durations[low] * (high - position) + durations[high] * (position - low)

    p50 = quantile(0.50)
    p95 = quantile(0.95)
    max_duration = durations[-1]
    max_live = max((int(row["live_sessions"]) for row in rows if row.get("live_sessions") is not None), default=0)
    slowest = max((row for row in rows if isinstance(row.get("duration_ms"), (int, float))), key=lambda row: float(row["duration_ms"]))
    slow_label = str(slowest.get("event") or "event")
    slow_instrument = str(slowest.get("instrument") or "").strip()
    if slow_instrument:
        slow_label = f"{slow_label} / {slow_instrument}"

    if p95 <= 1000.0:
        level = "green"
    elif p95 <= 3000.0:
        level = "amber"
    else:
        level = "red"
    return {
        "level": level,
        "value": f"p95 {p95 / 1000.0:.1f}s",
        "meta": (
            f"{len(durations)} timed events in {hours:g} h; "
            f"p50 {p50 / 1000.0:.1f}s; max {max_duration / 1000.0:.1f}s ({slow_label}); "
            f"max live sessions {max_live}"
        ),
    }


def _ops_storage_text(snapshot: dict, key_prefix: str) -> str:
    used = _ops_float(snapshot.get(f"{key_prefix}_used_gb"))
    total = _ops_float(snapshot.get(f"{key_prefix}_total_gb"))
    used_pct = _ops_float(snapshot.get(f"{key_prefix}_used_pct"))
    if used is None or total is None:
        return "No data"
    if used_pct is None:
        return f"{used:.1f} / {total:.1f} GB"
    return f"{used:.1f} / {total:.1f} GB ({used_pct:.0f}%)"


def _ops_storage_location(snapshot: dict, key_prefix: str, host_label: str, fallback_path: str) -> str:
    resolved = snapshot.get(f"{key_prefix}_resolved_path")
    if resolved:
        return f"{host_label} {resolved}"
    return f"{host_label} {fallback_path}"


def _ops_manifest_ready(snapshot: dict) -> bool:
    for spec in OPS_STREAM_SPECS:
        prefix = spec["stream_prefix"]
        if _ops_float(snapshot.get(f"{prefix}_source_count")) not in (None, 0.0):
            return True
        if _ops_float(snapshot.get(f"{prefix}_local_count")) not in (None, 0.0):
            return True
        if _ops_float(snapshot.get(f"{prefix}_gws_count")) not in (None, 0.0):
            return True
        if _ops_float(snapshot.get(f"{prefix}_local_coverage_pct")) is not None:
            return True
        if _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct")) is not None:
            return True
    return False


def _ops_archive_level(snapshot: dict, prefix: str) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        return "amber"
    gws_missing = int(_ops_float(snapshot.get(f"{prefix}_gws_missing_count")) or 0)
    gws_mismatch = int(_ops_float(snapshot.get(f"{prefix}_gws_mismatch_count")) or 0)
    gws_coverage = _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct"))
    if gws_coverage is None:
        if _ops_bool(snapshot.get("gws_probe_ok_state")):
            return "amber"
        return "gray"
    if gws_missing == 0 and gws_mismatch == 0:
        return "green"
    if gws_coverage >= 99.9:
        return "green"
    if gws_coverage >= 95.0:
        return "amber"
    return "red"


def _ops_archive_text(snapshot: dict, prefix: str) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        local_coverage = _ops_float(snapshot.get(f"{prefix}_local_coverage_pct"))
        gws_coverage = _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct"))
        local_text = "?" if local_coverage is None else f"{local_coverage:.0f}%"
        gws_text = "?" if gws_coverage is None else f"{gws_coverage:.0f}%"
        return f"Backfill {local_text} local / {gws_text} GWS"
    gws_missing = int(_ops_float(snapshot.get(f"{prefix}_gws_missing_count")) or 0)
    gws_mismatch = int(_ops_float(snapshot.get(f"{prefix}_gws_mismatch_count")) or 0)
    gws_coverage = _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct"))
    if gws_coverage is None:
        return "Pending manifest sync"
    if gws_missing == 0 and gws_mismatch == 0 and gws_coverage < 99.9:
        return f"{gws_coverage:.0f}% mirrored, settled OK"
    return f"{gws_coverage:.0f}% mirrored"


def _ops_prune_level(snapshot: dict, prefix: str, manifest_ready: bool) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        return "amber"
    if not manifest_ready:
        return "amber"
    prune_ready = _ops_bool(snapshot.get(f"{prefix}_prune_ready_state"))
    if prune_ready is None:
        return "gray"
    return "green" if prune_ready else "red"


def _ops_prune_text(snapshot: dict, prefix: str, manifest_ready: bool) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        return "Backfill"
    if not manifest_ready:
        return "Pending verification"
    prune_ready = _ops_bool(snapshot.get(f"{prefix}_prune_ready_state"))
    if prune_ready is None:
        return "Unknown"
    return "Ready" if prune_ready else "Hold"


def _ops_failed_service_names(snapshot: dict) -> list[str]:
    names: list[str] = []
    for key, value in sorted(snapshot.items()):
        if not key.endswith("_service_healthy_state"):
            continue
        if _ops_bool(value) is not False:
            continue
        label = key.removesuffix("_service_healthy_state").replace("_", " ")
        names.append(label)
    return names


def _ops_light_markup(level: str, text: str) -> str:
    return (
        f"<span class='ops-light ops-light--{escape(level)}' aria-hidden='true'></span>"
        f"<span class='ops-light-text'>{escape(text)}</span>"
    )


def _ops_card_markup(title: str, level: str, value: str, meta: str = "") -> str:
    meta_markup = f"<div class='ops-card__meta'>{escape(meta)}</div>" if meta else ""
    return (
        "<div class='ops-card'>"
        f"<div class='ops-card__head'>{_ops_light_markup(level, title)}</div>"
        f"<div class='ops-card__value'>{escape(value)}</div>"
        f"{meta_markup}"
        "</div>"
    )


def _ops_table_cell(level: str, label: str, detail: str = "") -> str:
    detail_markup = f"<div class='ops-table__detail'>{escape(detail)}</div>" if detail else ""
    return (
        "<td class='ops-table__cell'>"
        f"<div class='ops-table__state'>{_ops_light_markup(level, label)}</div>"
        f"{detail_markup}"
        "</td>"
    )


def _ops_operations_markup() -> str:
    snapshot = _ops_read_snapshot()
    with _timed_perf("operations_dashboard_render", instrument="ops-monitor", snapshot_path=snapshot.get("_path")) as perf:
        if snapshot.get("_missing"):
            perf["status"] = "missing_snapshot"
            missing_meta = f"Expected {snapshot.get('_path', 'unknown path')}"
            return (
                "<div class='ops-shell'>"
                "<div class='ops-section-title'>Operations Dashboard</div>"
                f"{_ops_card_markup('Operations snapshot', 'red', 'Missing', missing_meta)}"
                "</div>"
            )
        if snapshot.get("_error"):
            perf["status"] = "snapshot_error"
            return (
                "<div class='ops-shell'>"
                "<div class='ops-section-title'>Operations Dashboard</div>"
                f"{_ops_card_markup('Operations snapshot', 'red', 'Unreadable', snapshot.get('_error', 'Unknown error'))}"
                "</div>"
            )

        updated_at = _ops_timestamp(snapshot.get("time_utc"))
        snapshot_age_min = None
        if updated_at is not None:
            snapshot_age_min = max((datetime.now(timezone.utc) - updated_at).total_seconds() / 60.0, 0.0)
        manifest_ready = _ops_manifest_ready(snapshot)
        failed_services = _ops_failed_service_names(snapshot)
        target_stream_count = int(_ops_float(snapshot.get("streams_target_count")) or len(OPS_STREAM_SPECS))
        backfill_pending_count = int(_ops_float(snapshot.get("streams_backfill_pending_count")) or 0)

        snapshot_level = _ops_level_from_age_minutes(snapshot_age_min)
        source_level = _ops_level_from_source_probes(snapshot.get("source_host_probe_fail_count"))
        source_freshness_level = _ops_level_from_count(snapshot.get("streams_source_stale_count"), amber_at=0.0)
        battery_level = _ops_level_from_battery_voltage(snapshot.get("aps_battery_voltage_v"))
        battery_soc_level = _ops_level_from_battery_soc(snapshot.get("aps_battery_soc_pct"))
        internal_temp_level = _ops_level_from_internal_temp(snapshot.get("aps_internal_temp_c"))
        perf_log_level = _ops_level_from_perf_log(snapshot)
        processing_level = _ops_level_from_count(snapshot.get("failed_processing_unit_count"), amber_at=1.0)
        transfer_level = _ops_worst_level(
            [
                _ops_level_from_count(snapshot.get("failed_transfer_unit_count"), amber_at=1.0),
                _ops_level_from_bool(snapshot.get("gws_probe_ok_state")),
            ]
        )
        if not manifest_ready:
            mirror_level = "amber"
        else:
            mirror_level = _ops_worst_level(
                [
                    _ops_level_from_bool(snapshot.get("mirror_verify_service_healthy_state")),
                    _ops_level_from_count(snapshot.get("streams_local_issue_count"), amber_at=1.0),
                    _ops_level_from_count(snapshot.get("streams_gws_issue_count"), amber_at=1.0),
                ]
            )
            if mirror_level == "green" and backfill_pending_count > 0:
                mirror_level = "amber"
        # Render/performance telemetry is diagnostic: it should stay visible,
        # but it should not turn the overall operations state into "Action
        # needed" when the data/transfer/power systems are otherwise healthy.
        overall_level = _ops_worst_level([snapshot_level, source_level, source_freshness_level, battery_level, battery_soc_level, internal_temp_level, processing_level, transfer_level, mirror_level])

        overall_value = "Healthy"
        if overall_level == "amber":
            overall_value = "Attention needed"
        elif overall_level == "red":
            overall_value = "Action needed"
        elif overall_level == "gray":
            overall_value = "Waiting for data"

        updated_label = updated_at.strftime("%Y-%m-%d %H:%M UTC") if updated_at else "Unknown"
        age_label = f"{snapshot_age_min:.0f} min old" if snapshot_age_min is not None else "Age unknown"
        battery_value, battery_meta = _ops_battery_text(snapshot)
        battery_soc_value, battery_soc_meta = _ops_battery_soc_text(snapshot)
        internal_temp_value, internal_temp_meta = _ops_internal_temp_text(snapshot)
        perf_log_value, perf_log_meta = _ops_perf_log_text(snapshot)
        perf_summary = _ops_perf_summary(Path(snapshot.get("dashboard_perf_log_path") or PERF_LOG_PATH))

        summary_cards = [
            _ops_card_markup(
                "Overall",
                overall_level,
                overall_value,
                f"Snapshot {age_label}; {len(failed_services)} unhealthy services",
            ),
            _ops_card_markup(
                "Snapshot freshness",
                snapshot_level,
                age_label,
                f"Last operations sample {updated_label}",
            ),
            _ops_card_markup(
                "Source hosts",
                source_level,
                f"{max(0, 3 - int(_ops_float(snapshot.get('source_host_probe_fail_count')) or 0))}/3 reachable",
                f"{int(_ops_float(snapshot.get('source_host_probe_fail_count')) or 0)} probe failures",
            ),
            _ops_card_markup(
                "Source freshness",
                source_freshness_level,
                (
                    f"{int(_ops_float(snapshot.get('streams_source_recent_count')) or 0)}/{len(OPS_STREAM_SPECS)} recent"
                    if int(_ops_float(snapshot.get('streams_source_stale_count')) or 0) == 0
                    else f"{int(_ops_float(snapshot.get('streams_source_stale_count')) or 0)} stale streams"
                ),
                "Source data seen within the last 1.5 hours",
            ),
            _ops_card_markup(
                "Battery voltage",
                battery_level,
                battery_value,
                f"{battery_meta}; green >52 V, amber 50-52 V, red <50 V",
            ),
            _ops_card_markup(
                "Battery SOC",
                battery_soc_level,
                battery_soc_value,
                f"{battery_soc_meta}; green >=50 %, amber 25-50 %, red <25 %",
            ),
            _ops_card_markup(
                "APS internal temp",
                internal_temp_level,
                internal_temp_value,
                f"{internal_temp_meta}; green <40 C, amber 40-45 C, red >=45 C",
            ),
            _ops_card_markup(
                "Dashboard perf log",
                perf_log_level,
                perf_log_value,
                f"{perf_log_meta}; diagnostic only, not part of Overall",
            ),
            _ops_card_markup(
                "Render performance",
                perf_summary["level"],
                perf_summary["value"],
                f"{perf_summary['meta']}; diagnostic only, not part of Overall",
            ),
            _ops_card_markup(
                "Processing pipeline",
                processing_level,
                f"{int(_ops_float(snapshot.get('failed_processing_unit_count')) or 0)} failed services",
                ", ".join(failed_services[:3]) if failed_services else "Append and quicklook services healthy",
            ),
            _ops_card_markup(
                "Transfers and GWS",
                transfer_level,
                "Reachable" if _ops_bool(snapshot.get("gws_probe_ok_state")) else "Unreachable",
                f"{int(_ops_float(snapshot.get('failed_transfer_unit_count')) or 0)} transfer failures",
            ),
            _ops_card_markup(
                "Mirror verification",
                mirror_level,
                "Pending manifest seed" if not manifest_ready else "Active",
                (
                    "Waiting for raw/GWS manifests to populate"
                    if not manifest_ready
                    else (
                        f"Local issues: {int(_ops_float(snapshot.get('streams_local_issue_count')) or 0)}, "
                        f"GWS issues: {int(_ops_float(snapshot.get('streams_gws_issue_count')) or 0)}, "
                        f"Backfills: {backfill_pending_count}"
                    )
                ),
            ),
        ]

        storage_cards = [
            _ops_card_markup(
                "CL61 root disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_celine_source_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_celine_source_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_celine_source"),
                _ops_storage_location(snapshot, "host_celine_source", "100.117.101.84", "/"),
            ),
            _ops_card_markup(
                "CL61 data disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_celine_data_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_celine_data_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_celine_data"),
                _ops_storage_location(snapshot, "host_celine_data", "100.117.101.84", "/home/aurora/data"),
            ),
            _ops_card_markup(
                "ASS data disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_ass_data_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_ass_data_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_ass_data"),
                _ops_storage_location(snapshot, "host_ass_data", "100.124.55.22", "/home/aurora/data"),
            ),
            _ops_card_markup(
                "ASS root disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_ass_root_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_ass_root_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_ass_root"),
                _ops_storage_location(snapshot, "host_ass_root", "100.124.55.22", "/"),
            ),
            _ops_card_markup(
                "APS data disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_aps_data_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_aps_data_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_aps_data"),
                _ops_storage_location(snapshot, "host_aps_data", "100.81.226.30", "/data"),
            ),
            _ops_card_markup(
                "APS root disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_aps_root_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_aps_root_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_aps_root"),
                _ops_storage_location(snapshot, "host_aps_root", "100.81.226.30", "/"),
            ),
            _ops_card_markup(
                "AURORA Cloud product disk",
                _ops_level_from_used_pct(snapshot.get("aurora_data_used_pct")),
                _ops_storage_text(snapshot, "aurora_data"),
                _ops_storage_location(snapshot, "aurora_data", "AURORA Cloud", "/data/aurora"),
            ),
            _ops_card_markup(
                "AURORA Cloud root disk",
                _ops_level_from_used_pct(snapshot.get("aurora_root_used_pct")),
                _ops_storage_text(snapshot, "aurora_root"),
                _ops_storage_location(snapshot, "aurora_root", "AURORA Cloud", "/"),
            ),
            _ops_card_markup(
                "JASMIN GWS",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("gws_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("gws_storage_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "gws_storage"),
                "/gws/ssde/j25b/gamb2le",
            ),
        ]

        manifest_cards = [
            _ops_card_markup(
                "Local raw mirror",
                (
                    "amber"
                    if not manifest_ready
                    else (
                        "amber"
                        if backfill_pending_count > 0 and int(_ops_float(snapshot.get("streams_local_issue_count")) or 0) == 0
                        else _ops_level_from_count(snapshot.get("streams_local_issue_count"), amber_at=1.0)
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_local_issue_count')) or 0)} issues"
                        if backfill_pending_count == 0
                        else f"{int(_ops_float(snapshot.get('streams_local_issue_count')) or 0)} issues, {backfill_pending_count} backfill"
                    )
                ),
                "Mirror against source-host manifests",
            ),
            _ops_card_markup(
                "GWS archive mirror",
                (
                    "amber"
                    if not manifest_ready
                    else (
                        "amber"
                        if backfill_pending_count > 0 and int(_ops_float(snapshot.get("streams_gws_issue_count")) or 0) == 0
                        else _ops_level_from_count(snapshot.get("streams_gws_issue_count"), amber_at=1.0)
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_gws_issue_count')) or 0)} issues"
                        if backfill_pending_count == 0
                        else f"{int(_ops_float(snapshot.get('streams_gws_issue_count')) or 0)} issues, {backfill_pending_count} backfill"
                    )
                ),
                "Mirror against JASMIN manifests",
            ),
            _ops_card_markup(
                "Product gates",
                "amber"
                if not manifest_ready
                else (
                    "green"
                    if int(_ops_float(snapshot.get("streams_product_gate_ok_count")) or 0) == target_stream_count and backfill_pending_count == 0
                    else (
                        "amber"
                        if int(_ops_float(snapshot.get("streams_product_gate_ok_count")) or 0) == target_stream_count
                        else "red"
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_product_gate_ok_count')) or 0)}/{target_stream_count} streams ready"
                        + (f", {backfill_pending_count} backfill" if backfill_pending_count else "")
                    )
                ),
                "Processing success through candidate prune windows",
            ),
            _ops_card_markup(
                "Prune readiness",
                "amber"
                if not manifest_ready
                else (
                    "green"
                    if int(_ops_float(snapshot.get("streams_prune_ready_count")) or 0) == target_stream_count and backfill_pending_count == 0
                    else (
                        "amber"
                        if int(_ops_float(snapshot.get("streams_prune_ready_count")) or 0) == target_stream_count
                        else "red"
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_prune_ready_count')) or 0)}/{target_stream_count} streams deletable"
                        + (f", {backfill_pending_count} backfill" if backfill_pending_count else "")
                    )
                ),
                "Only prune upstream when source, local, and GWS agree",
            ),
        ]

        table_rows = []
        for spec in OPS_STREAM_SPECS:
            source_level_stream = _ops_worst_level(
                [
                    _ops_level_from_bool(snapshot.get(spec["source_key"])),
                    _ops_level_from_bool(snapshot.get(f"{spec['stream_prefix']}_source_recent_state")),
                ]
            )
            processing_level_stream = _ops_worst_level([_ops_level_from_bool(snapshot.get(key)) for key in spec["processing_keys"]])
            processing_ok = sum(1 for key in spec["processing_keys"] if _ops_bool(snapshot.get(key)) is True)
            archive_level_stream = _ops_archive_level(snapshot, spec["stream_prefix"])
            prune_level_stream = _ops_prune_level(snapshot, spec["stream_prefix"], manifest_ready)
            processing_detail = f"{processing_ok}/{len(spec['processing_keys'])} healthy"
            table_rows.append(
                "<tr>"
                f"<th class='ops-table__rowlabel'>{escape(spec['label'])}</th>"
                f"{_ops_table_cell(source_level_stream, 'Source', _ops_source_freshness_text(snapshot, spec['stream_prefix']))}"
                f"{_ops_table_cell(processing_level_stream, 'Processing', processing_detail)}"
                f"{_ops_table_cell(archive_level_stream, _ops_archive_text(snapshot, spec['stream_prefix']), 'Raw mirror to GWS')}"
                f"{_ops_table_cell(prune_level_stream, _ops_prune_text(snapshot, spec['stream_prefix'], manifest_ready), 'Deletion gate')}"
                "</tr>"
            )

        failed_markup = ""
        if failed_services:
            failed_items = "".join(f"<li>{escape(name)}</li>" for name in failed_services[:8])
            failed_markup = (
                "<div class='ops-section'>"
                "<div class='ops-section-title'>Current service issues</div>"
                f"<div class='ops-callout ops-callout--red'><ul>{failed_items}</ul></div>"
                "</div>"
            )

        perf["status"] = "ok"
        perf["failed_services"] = len(failed_services)
        perf["manifest_ready"] = manifest_ready
        perf["snapshot_age_min"] = snapshot_age_min

        return (
            "<div class='ops-shell'>"
            "<div class='ops-headline'>"
            "<div class='ops-headline__main'>"
            f"<div class='ops-section-title ops-section-title--headline'>{_ops_light_markup(overall_level, 'Operations Dashboard')}</div>"
            "<div class='ops-headline__text'>Traffic lights summarize service health, storage pressure, archive transfer status, and prune readiness from the latest operations snapshot.</div>"
            "</div>"
            "<div class='ops-legend'>"
            f"{_ops_light_markup('green', 'Healthy')}"
            f"{_ops_light_markup('amber', 'Attention or pending')}"
            f"{_ops_light_markup('red', 'Failing or blocked')}"
            f"{_ops_light_markup('gray', 'Unknown')}"
            "</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>System summary</div>"
            f"<div class='ops-grid ops-grid--summary'>{''.join(summary_cards)}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Storage</div>"
            f"<div class='ops-grid ops-grid--storage'>{''.join(storage_cards)}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Archive and pruning</div>"
            f"<div class='ops-grid ops-grid--summary'>{''.join(manifest_cards)}</div>"
            "</div>"
            f"{failed_markup}"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Per-stream health</div>"
            "<div class='ops-table-wrap'>"
            "<table class='ops-table'>"
            "<thead><tr><th>Stream</th><th>Source sync</th><th>Processing</th><th>Archive</th><th>Prune gate</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
            "</div>"
            "<div class='ops-footnote'>Amber can mean an expected wait state, such as a manifest seed or verification window that has not completed yet. Red indicates a failing service, blocked gate, or critical storage pressure.</div>"
            "</div>"
            "</div>"
        )


def _selected_token_window(selected: str | None) -> tuple[datetime | None, datetime | None, str | None]:
    if not selected:
        return None, None, None
    if selected == "Today (latest)":
        day_token = datetime.now(timezone.utc).strftime("%Y%m%d")
    elif selected == "latest":
        day_token = datetime.now(timezone.utc).strftime("%Y%m%d")
    elif len(selected) == 8 and selected.isdigit():
        day_token = selected
    else:
        return None, None, None
    start = datetime.strptime(day_token, "%Y%m%d")
    if day_token == datetime.now(timezone.utc).strftime("%Y%m%d"):
        end = datetime.now(timezone.utc).replace(tzinfo=None)
    else:
        end = start + timedelta(days=1) - timedelta(seconds=1)
    return start, end, day_token


def _coarsen_targets(duration: timedelta | None, height_span: float | None):
    """Return subsample/target counts that scale up for zoomed-in windows."""
    time_subsample = TIME_SUBSAMPLE
    time_target = TIME_TARGET
    height_target = HEIGHT_TARGET
    if duration is not None:
        hours = duration.total_seconds() / 3600.0
        if hours <= 2:
            time_subsample = 1
            time_target = 1200
        elif hours <= 6:
            time_subsample = 1
            time_target = 800
        elif hours <= 24:
            time_subsample = 1
            time_target = 400
    if height_span is not None:
        if height_span <= 1000:
            height_target = 400
        elif height_span <= 3000:
            height_target = 300
    return time_subsample, time_target, height_target


def open_window(t0, t1, bottom_m=None, top_m=None, instrument: str | None = None, render_quality: str = "full"):
    """Slice the base dataset, adapt coarsening to window span, and filter height."""
    instrument = instrument or CURRENT_INSTRUMENT
    cfg = _cfg(instrument)
    t0 = _ensure_utc(t0)
    t1 = _ensure_utc(t1)
    with _timed_perf(
        "window_open",
        instrument=instrument,
        start=t0,
        end=t1,
        bottom_m=bottom_m,
        top_m=top_m,
        render_quality=render_quality,
    ) as perf:
        if t0 is None or t1 is None or t0 >= t1:
            perf["status"] = "invalid_window"
            return xr.Dataset()
        duration = t1 - t0
        perf["window_hours"] = round(duration.total_seconds() / 3600.0, 3)
        height_span = None
        if bottom_m is not None or top_m is not None:
            b = max(bottom_m or 0.0, 0.0)
            t = top_m if top_m is not None else cfg["height_load_max"]
            height_span = max(t - b, 0.0)
        time_subsample, time_target, height_target = _coarsen_targets(duration, height_span)
        if render_quality == "coarse":
            time_subsample = max(time_subsample, 2)
            time_target = max(96, time_target // 3)
            height_target = max(72, height_target // 2)
        preserve_time_detail = render_quality == "summary_full_time"
        perf["time_subsample"] = int(time_subsample)
        perf["time_target"] = int(time_target)
        perf["height_target"] = int(height_target)
        perf["preserve_time_detail"] = bool(preserve_time_detail)
        base = _get_base_dataset(instrument)
        if base is None:
            perf["status"] = "no_dataset"
            return xr.Dataset()
        perf["base_time_count"] = int(base.sizes.get("time", 0))
        perf["base_range_count"] = int(base.sizes.get("range", 0))

        phase_start = perf_counter()
        try:
            tvals = base["time"].values
            mask = _valid_time_mask(tvals) & (tvals >= np.datetime64(t0)) & (tvals <= np.datetime64(t1))
            perf["matched_time_count"] = int(np.count_nonzero(mask))
            if not np.any(mask):
                perf["status"] = "no_match"
                perf["select_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)
                return xr.Dataset()
            idx = np.nonzero(mask)[0]
            ds = base.isel(time=idx)
        except Exception as exc:
            ds = base
            perf["time_select_error"] = str(exc)
        perf["select_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)

        has_range = "range" in ds.coords or "range" in ds.dims
        phase_start = perf_counter()
        if has_range:
            try:
                ds = ds.sel({"range": slice(0, cfg["height_load_max"])})
            except Exception:
                ds = ds.where(ds["range"] <= cfg["height_load_max"], drop=True)
        if has_range and (bottom_m is not None or top_m is not None):
            low = max(bottom_m or 0.0, 0.0)
            high = min(top_m or cfg["height_load_max"], cfg["height_load_max"])
            try:
                ds = ds.sel({"range": slice(low, high)})
            except Exception:
                ds = ds.where((ds["range"] >= low) & (ds["range"] <= high), drop=True)
        perf["range_filter_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)

        phase_start = perf_counter()
        if not preserve_time_detail and time_subsample > 1:
            ds = ds.isel(time=slice(None, None, time_subsample))
        try:
            if ds.sizes.get("range", 0) > height_target:
                fh = max(int(np.ceil(ds.sizes["range"] / height_target)), 1)
                perf["range_coarsen_factor"] = int(fh)
                ds = ds.coarsen({"range": fh}, boundary="trim").mean()
            if not preserve_time_detail and ds.sizes.get("time", 0) > time_target:
                ft = max(int(np.ceil(ds.sizes["time"] / time_target)), 1)
                perf["time_coarsen_factor"] = int(ft)
                ds = ds.coarsen({"time": ft}, boundary="trim").mean()
        except Exception as exc:
            perf["coarsen_error"] = str(exc)
        perf["coarsen_ms"] = round((perf_counter() - phase_start) * 1000.0, 3)
        perf["output_time_count"] = int(ds.sizes.get("time", 0))
        perf["output_range_count"] = int(ds.sizes.get("range", 0))
        perf["status"] = "ok"
        return ds


def _make_plot(ds, var, clim, logz, coloraxis):
    """Build a Plotly heatmap trace for a variable with optional log10 scaling."""
    times = pd.to_datetime(ds["time"].values)
    heights = ds["range"].values
    data = np.array(ds[var].transpose("range", "time"))
    zmin, zmax = clim
    if logz:
        data = np.where(data > 0, data, np.nan)
        with np.errstate(divide="ignore"):
            data = np.log10(data)
        zmin, zmax = np.log10(clim[0]), np.log10(clim[1])
    trace = go.Heatmap(
        x=times,
        y=heights,
        z=data,
        zmin=zmin,
        zmax=zmax,
        coloraxis=coloraxis,
        showscale=False,
    )
    return trace


def _numeric_time_vars(ds: xr.Dataset) -> list[str]:
    """Return numeric one-dimensional data variables aligned to time."""
    names: list[str] = []
    for name, da in ds.data_vars.items():
        if da.dims != ("time",):
            continue
        if np.issubdtype(da.dtype, np.number):
            names.append(name)
    return names


def _is_stacked_timeseries_instrument(inst: str) -> bool:
    return is_summary_instrument(inst)


def _is_wxcam_instrument(inst: str) -> bool:
    return inst == "wxcam"


# Widgets / controls (Panel wires these into the view updater)
default_end = datetime.utcnow()
default_start = default_end - DEFAULT_WINDOW
range_start = pn.widgets.DatetimePicker(name="Start (UTC)", value=default_start)
range_end = pn.widgets.DatetimePicker(name="End (UTC)", value=default_end)
top_range_m = pn.widgets.IntInput(name="Top range (m)", value=_cfg()["top_range_default"], step=100, start=500)
bottom_range_m = pn.widgets.IntInput(name="Bottom range (m)", value=0, step=100, start=0)
var1_select = pn.widgets.Select(name="Top var", options=list(_cfg()["vars"].keys()), value=_cfg()["default_top"])
var2_select = pn.widgets.Select(name="Bottom var", options=list(_cfg()["vars"].keys()), value=_cfg()["default_bottom"])
beta_vmin = pn.widgets.FloatInput(name="Var1 min", value=_cfg()["vars"][var1_select.value]["clim"][0], step=0.1)
beta_vmax = pn.widgets.FloatInput(name="Var1 max", value=_cfg()["vars"][var1_select.value]["clim"][1], step=0.1)
ldr_vmin = pn.widgets.FloatInput(name="Var2 min", value=_cfg()["vars"][var2_select.value]["clim"][0], step=0.1)
ldr_vmax = pn.widgets.FloatInput(name="Var2 max", value=_cfg()["vars"][var2_select.value]["clim"][1], step=0.1)
lwp_ymin = pn.widgets.FloatInput(name="LWP min (g/m²)", value=0.0, step=10.0, visible=False)
lwp_ymax = pn.widgets.FloatInput(name="LWP max (g/m²)", value=400.0, step=10.0, visible=False)
iwv_ymin = pn.widgets.FloatInput(name="IWV min (kg/m²)", value=0.0, step=1.0, visible=False)
iwv_ymax = pn.widgets.FloatInput(name="IWV max (kg/m²)", value=40.0, step=1.0, visible=False)
irr_ymin = pn.widgets.FloatInput(name="IRR / SURF_T min (°C)", value=-20.0, step=1.0, visible=False)
irr_ymax = pn.widgets.FloatInput(name="IRR / SURF_T max (°C)", value=10.0, step=1.0, visible=False)
prev_btn = pn.widgets.Button(name="Previous Day", button_type="default")
next_btn = pn.widgets.Button(name="Next Day/Current Day", button_type="default")
live_toggle = pn.widgets.Toggle(name="Live Update (Last 24h)", button_type="primary", value=True)
instrument_select = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=INSTRUMENT_OPTIONS)
science_instrument = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=INSTRUMENT_OPTIONS)
science_image_type = pn.widgets.Select(name="Image type", options=[], visible=False)
hk_instrument = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=HK_INSTRUMENT_OPTIONS)

_live_guard = False
_instrument_guard = False
_live_cb = None  # handle for periodic callback (used for live refresh)
_relayout_guard = False  # prevents loops when syncing zoom back to widgets
_base_dataset_timer = _safe_periodic_callback(_refresh_time_bounds_cache, period=DATA_REFRESH_MS, start=True)


def _capture_current_instrument_state(inst: str | None = None) -> None:
    inst = inst or CURRENT_INSTRUMENT
    _INSTRUMENT_VIEW_STATE[inst] = {
        "range_start": _ensure_utc(range_start.value),
        "range_end": _ensure_utc(range_end.value),
        "top_range_m": top_range_m.value,
        "bottom_range_m": bottom_range_m.value,
        "var1_select": var1_select.value,
        "var2_select": var2_select.value,
        "beta_vmin": beta_vmin.value,
        "beta_vmax": beta_vmax.value,
        "ldr_vmin": ldr_vmin.value,
        "ldr_vmax": ldr_vmax.value,
        "lwp_ymin": lwp_ymin.value,
        "lwp_ymax": lwp_ymax.value,
        "iwv_ymin": iwv_ymin.value,
        "iwv_ymax": iwv_ymax.value,
        "irr_ymin": irr_ymin.value,
        "irr_ymax": irr_ymax.value,
        "live_toggle": live_toggle.value,
        "science_image_type": science_image_type.value,
        "wxcam_image_type": globals().get("wxcam_image_type").value if "wxcam_image_type" in globals() else None,
        "wxcam_date": globals().get("wxcam_date").value if "wxcam_date" in globals() else None,
    }


def _apply_instrument_defaults(inst: str, reset_time: bool = True):
    """Switch instrument: refresh dataset cache, reset controls, and relabel color widgets."""
    global CURRENT_INSTRUMENT, _instrument_guard
    _instrument_guard = True
    previous_instrument = CURRENT_INSTRUMENT
    CURRENT_INSTRUMENT = inst
    if previous_instrument != inst or _BASE_DS.get(inst) is None:
        _refresh_base_dataset(inst)
    else:
        _refresh_time_bounds_cache(inst)
    cfg = _cfg(inst)
    saved_state = _INSTRUMENT_VIEW_STATE.get(inst, {})
    with hold():
        vars_cfg = cfg["vars"]
        is_hatpro = inst == "Scanning Microwave Radiometer"
        is_stacked_timeseries = _is_stacked_timeseries_instrument(inst)
        is_wxcam = _is_wxcam_instrument(inst)
        var1_name = cfg["default_top"]
        var2_name = cfg["default_bottom"]
        var1_select.options = list(vars_cfg.keys())
        var2_select.options = list(vars_cfg.keys())
        var1_select.value = var1_name
        var2_select.value = var2_name
        var1_select.name = "Overview" if is_stacked_timeseries else "Top var"
        var2_select.name = "Bottom var"
        var1 = vars_cfg[var1_name]
        var2 = vars_cfg[var2_name]
        beta_vmin.name = f"{var1['label']} min"
        beta_vmax.name = f"{var1['label']} max"
        ldr_vmin.name = f"{var2['label']} min"
        ldr_vmax.name = f"{var2['label']} max"

        def _set_input(inp, vmin, vmax):
            inp.value = vmin
            # heuristic step
            span = abs(vmax - vmin) or 1.0
            inp.step = span / 100.0
            inp.format = "0.000000e+00" if max(abs(vmin), abs(vmax)) < 1e-3 else None

        _set_input(beta_vmin, var1["clim"][0], var1["clim"][1])
        _set_input(beta_vmax, var1["clim"][1], var1["clim"][0])
        _set_input(ldr_vmin, var2["clim"][0], var2["clim"][1])
        _set_input(ldr_vmax, var2["clim"][1], var2["clim"][0])

        bottom_range_m.value = 0
        top_range_m.value = cfg["top_range_default"]

        science_instrument.value = inst
        hk_instrument.value = inst

        if reset_time:
            saved_live = bool(saved_state.get("live_toggle")) if (saved_state and not is_wxcam) else False
            saved_start = saved_state.get("range_start")
            saved_end = saved_state.get("range_end")
            if (
                not saved_live
                and isinstance(saved_start, datetime)
                and isinstance(saved_end, datetime)
                and saved_start < saved_end
            ):
                range_start.value = saved_start
                range_end.value = saved_end
            elif _APP_BOOTSTRAPPING:
                end = _ensure_utc(range_end.value) or datetime.utcnow()
                range_start.value = end - DEFAULT_WINDOW
                range_end.value = end
            else:
                tmin, tmax = _dataset_time_bounds(inst)
                end = tmax or datetime.utcnow()
                start = end - DEFAULT_WINDOW
                range_start.value = start
                range_end.value = end
            # WXcam is a manual browser: refresh when switching back into it,
            # but do not keep a hidden live timer running while it is selected.
            _set_live(saved_live if not is_wxcam else False)

        # Instrument-specific UI trimming
        range_start.visible = not is_wxcam
        range_end.visible = not is_wxcam
        live_toggle.visible = not is_wxcam
        var1_select.visible = not is_hatpro and not is_stacked_timeseries and not is_wxcam
        var2_select.visible = not is_hatpro and not is_stacked_timeseries and not is_wxcam
        bottom_range_m.visible = not is_stacked_timeseries and not is_wxcam
        top_range_m.visible = not is_stacked_timeseries and not is_wxcam
        ldr_vmin.visible = not (is_hatpro or is_stacked_timeseries or is_wxcam)
        ldr_vmax.visible = not (is_hatpro or is_stacked_timeseries or is_wxcam)
        beta_vmin.visible = not (is_stacked_timeseries or is_wxcam)
        beta_vmax.visible = not (is_stacked_timeseries or is_wxcam)
        prev_btn.visible = not is_wxcam
        next_btn.visible = not is_wxcam
        science_image_type.visible = is_wxcam
        if is_wxcam:
            science_image_type.name = "Image type"
            science_image_type.options = list(vars_cfg.keys())
            saved_wxcam_type = saved_state.get("wxcam_image_type") or saved_state.get("science_image_type")
            science_image_type.value = saved_wxcam_type if saved_wxcam_type in vars_cfg else var1_name
            wxcam_image_type.options = list(vars_cfg.keys())
            wxcam_image_type.value = saved_wxcam_type if saved_wxcam_type in vars_cfg else var1_name
            _refresh_wxcam_ql_options(preserve_current=False)
            saved_wxcam_date = saved_state.get("wxcam_date")
            if saved_wxcam_date in list(wxcam_date.options):
                wxcam_date.value = saved_wxcam_date
        else:
            science_image_type.name = "Image type"
            science_image_type.options = []
        if saved_state and not is_wxcam:
            if saved_state.get("var1_select") in vars_cfg:
                var1_select.value = saved_state["var1_select"]
            if saved_state.get("var2_select") in vars_cfg:
                var2_select.value = saved_state["var2_select"]
            if not is_stacked_timeseries:
                bottom_range_m.value = int(saved_state.get("bottom_range_m", bottom_range_m.value) or 0)
                top_range_m.value = int(saved_state.get("top_range_m", top_range_m.value) or top_range_m.value)
                beta_vmin.value = saved_state.get("beta_vmin", beta_vmin.value)
                beta_vmax.value = saved_state.get("beta_vmax", beta_vmax.value)
                ldr_vmin.value = saved_state.get("ldr_vmin", ldr_vmin.value)
                ldr_vmax.value = saved_state.get("ldr_vmax", ldr_vmax.value)
        if is_hatpro:
            beta_vmin.name = "T_PROF min (K)"
            beta_vmax.name = "T_PROF max (K)"
            _set_input(beta_vmin, var1["clim"][0], var1["clim"][1])
            _set_input(beta_vmax, var1["clim"][1], var1["clim"][0])
            # Show and reset timeseries y-range controls
            lwp_ymin.visible = lwp_ymax.visible = True
            iwv_ymin.visible = iwv_ymax.visible = True
            irr_ymin.visible = irr_ymax.visible = True
            lwp_ymin.value, lwp_ymax.value = 0.0, 400.0
            iwv_ymin.value, iwv_ymax.value = 0.0, 60.0
            irr_ymin.value, irr_ymax.value = -20.0, 60.0
            if saved_state:
                lwp_ymin.value = saved_state.get("lwp_ymin", lwp_ymin.value)
                lwp_ymax.value = saved_state.get("lwp_ymax", lwp_ymax.value)
                iwv_ymin.value = saved_state.get("iwv_ymin", iwv_ymin.value)
                iwv_ymax.value = saved_state.get("iwv_ymax", iwv_ymax.value)
                irr_ymin.value = saved_state.get("irr_ymin", irr_ymin.value)
                irr_ymax.value = saved_state.get("irr_ymax", irr_ymax.value)
        else:
            lwp_ymin.visible = lwp_ymax.visible = False
            iwv_ymin.visible = iwv_ymax.visible = False
            irr_ymin.visible = irr_ymax.visible = False

        _refresh_ql_options(preserve_current=False)
        _refresh_hk_options(preserve_current=False)
        # Force quicklook panes to refresh even if the selection string did not change.
        ql_date.param.trigger("value")
        hk_date.param.trigger("value")
    _instrument_guard = False
    # Run a single consolidated refresh after batching widget changes
    _update_view(
        range_start.value,
        range_end.value,
        bottom_range_m.value,
        top_range_m.value,
        var1_select.value,
        var2_select.value,
        beta_vmin.value,
        beta_vmax.value,
        ldr_vmin.value,
        ldr_vmax.value,
        lwp_ymin.value,
        lwp_ymax.value,
        iwv_ymin.value,
        iwv_ymax.value,
        irr_ymin.value,
        irr_ymax.value,
        instrument_select.value,
    )


def _refresh_to_latest(_event=None):
    """Jump the interactive time controls to the latest 24 h window."""
    global _live_guard
    _live_guard = True
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT) and _dataset_cache_age(CURRENT_INSTRUMENT) >= timedelta(milliseconds=DATA_REFRESH_MS):
        _refresh_base_dataset(CURRENT_INSTRUMENT)
    tmin, tmax = _dataset_time_bounds()
    end = tmax or datetime.utcnow()
    start = end - DEFAULT_WINDOW
    range_start.value = start
    range_end.value = end
    bottom_range_m.value = 0
    top_range_m.value = _cfg()["top_range_default"]
    _live_guard = False


def _set_live(state: bool):
    """Set live toggle state without re-triggering handlers."""
    global _live_guard
    _live_guard = True
    live_toggle.value = state
    live_toggle.name = "Live Update (Last 24h)" if state else "Live Off"
    live_toggle.button_type = "primary" if state else "default"
    _live_guard = False


def _on_live_toggle(event):
    """Handle live toggle clicks."""
    if _live_guard:
        return
    if event.new:
        _set_live(True)
        _refresh_to_latest()
    else:
        _set_live(False)


live_toggle.param.watch(_on_live_toggle, "value")


def _on_instrument_change(event):
    """Switch datasets and reset controls when instrument dropdown changes."""
    if _instrument_guard:
        return
    if event.old:
        _capture_current_instrument_state(event.old)
    _apply_instrument_defaults(event.new, reset_time=True)


instrument_select.param.watch(_on_instrument_change, "value")


def _on_science_instrument_change(event):
    """Sync science quicklook instrument dropdown back to the main instrument selector."""
    if _instrument_guard:
        return
    instrument_select.value = event.new


science_instrument.param.watch(_on_science_instrument_change, "value")


def _on_hk_instrument_change(event):
    """Sync housekeeping quicklook instrument dropdown back to the main selector."""
    if _instrument_guard:
        return
    instrument_select.value = event.new


hk_instrument.param.watch(_on_hk_instrument_change, "value")


def _on_var_change(event):
    """Update limit widgets when variable selection changes."""
    cfg = _cfg()
    vars_cfg = cfg["vars"]
    if _is_wxcam_instrument(CURRENT_INSTRUMENT):
        if science_image_type.value != var1_select.value:
            science_image_type.value = var1_select.value
        return
    if _is_stacked_timeseries_instrument(CURRENT_INSTRUMENT):
        return
    var1 = vars_cfg.get(var1_select.value, None)
    var2 = vars_cfg.get(var2_select.value, None)
    if var1:
        beta_vmin.name = f"{var1['label']} min"
        beta_vmax.name = f"{var1['label']} max"
        beta_vmin.value, beta_vmax.value = var1["clim"]
    if var2:
        ldr_vmin.name = f"{var2['label']} min"
        ldr_vmax.name = f"{var2['label']} max"
        ldr_vmin.value, ldr_vmax.value = var2["clim"]


var1_select.param.watch(_on_var_change, "value")
var2_select.param.watch(_on_var_change, "value")


def _on_science_image_type_change(event):
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        return
    _refresh_ql_options(preserve_current=False)
    ql_date.param.trigger("value")


science_image_type.param.watch(_on_science_image_type_change, "value")


def _on_wxcam_image_type_change(event):
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        return
    _refresh_wxcam_ql_options(preserve_current=False)
    wxcam_date.param.trigger("value")

def _auto_refresh():
    """Periodic refresh when live mode is on."""
    if live_toggle.value:
        _refresh_to_latest()


# Kick off periodic live refresh.
_live_cb = _safe_periodic_callback(_auto_refresh, period=LIVE_REFRESH_MS, start=True)


def _shift_previous(_event=None):
    """Jump to the previous full UTC day (00:00–24:00), clamping to data start."""
    _set_live(False)
    tmin, tmax = _dataset_time_bounds()
    anchor_end = _ensure_utc(range_end.value) or _ensure_utc(range_start.value) or (tmax or datetime.utcnow())
    prev_day = (anchor_end - timedelta(days=1)).date()
    prev_start = datetime.combine(prev_day, datetime.min.time())
    prev_end = datetime.combine(prev_day, time(hour=23, minute=59))
    if tmin and prev_end < tmin:
        # no data that far back; keep current window
        return
    if tmin and prev_start < tmin:
        # Clamp to first day available, ending at that day's 23:59 (or tmax if earlier)
        day_start = datetime.combine(tmin.date(), time.min)
        prev_start = tmin
        prev_end = datetime.combine(tmin.date(), time(hour=23, minute=59))
        if tmax:
            prev_end = min(prev_end, tmax)
    range_start.value = prev_start
    range_end.value = prev_end


def _shift_next(_event=None):
    """Jump to the next full UTC day; if past data end, snap to latest 24h."""
    _set_live(False)
    tmin, tmax = _dataset_time_bounds()
    if range_end.value is None:
        _refresh_to_latest()
        return
    anchor = _ensure_utc(range_start.value) or _ensure_utc(range_end.value)
    if anchor is None:
        _refresh_to_latest()
        return
    next_start = datetime.combine(anchor.date() + timedelta(days=1), datetime.min.time())
    next_end = next_start + timedelta(days=1) - timedelta(minutes=1)
    if tmax and next_end > tmax:
        # Not enough data ahead; show latest 24h
        latest_end = tmax
        latest_start = tmax - DEFAULT_WINDOW
        if tmin:
            latest_start = max(latest_start, tmin)
        range_start.value = latest_start
        range_end.value = latest_end
    else:
        range_start.value = next_start
        range_end.value = next_end


prev_btn.on_click(_shift_previous)
next_btn.on_click(_shift_next)


def _on_manual_time_change(event):
    """Disable live mode if user edits time pickers manually."""
    if _live_guard or _relayout_guard:
        return
    if live_toggle.value:
        _set_live(False)


range_start.param.watch(_on_manual_time_change, "value")
range_end.param.watch(_on_manual_time_change, "value")

# Persistent plot pane so we can listen for zoom/pan events (relayout).
# Keep one stable shell for the interactive area so switching instruments
# reuses the same pane tree instead of rebuilding the whole section.
plot_pane = pn.pane.Plotly(config={"responsive": True}, sizing_mode="stretch_width")
interactive_loading = pn.pane.HTML("", visible=False, sizing_mode="stretch_width", margin=(0, 0, 8, 0))
interactive_placeholder = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
interactive_body = pn.Column(plot_pane, sizing_mode="stretch_width", margin=0)
interactive_content = pn.Column(interactive_loading, interactive_body, sizing_mode="stretch_width", margin=0)


def _interactive_placeholder_height(inst: str) -> int:
    if inst == "Scanning Microwave Radiometer":
        return 760
    if _is_stacked_timeseries_instrument(inst):
        return 620
    if _is_wxcam_instrument(inst):
        return 520
    return 760


def _interactive_loading_notice_markup(inst: str, message: str, phase: str) -> str:
    label = "WXcam" if inst == "wxcam" else display_name(inst)
    phase_label = "Refining detail" if phase == "refining" else "Loading"
    return (
        "<div class='interactive-loading-notice'>"
        f"<span class='interactive-loading-notice__badge'>{escape(phase_label)}</span>"
        f"<div class='interactive-loading-notice__text'><strong>{escape(label)}</strong> {escape(message)}</div>"
        "</div>"
    )


def _interactive_placeholder_markup(inst: str, message: str) -> str:
    label = "WXcam" if inst == "wxcam" else display_name(inst)
    return (
        "<div class='interactive-skeleton'>"
        f"<div class='interactive-skeleton__title'>{escape(label)}</div>"
        f"<div class='interactive-skeleton__subtitle'>{escape(message)}</div>"
        "<div class='interactive-skeleton__plot'></div>"
        "<div class='interactive-skeleton__plot interactive-skeleton__plot--secondary'></div>"
        "</div>"
    )


def _set_interactive_body(target) -> None:
    if len(interactive_body.objects) != 1 or interactive_body.objects[0] is not target:
        interactive_body[:] = [target]


def _show_interactive_placeholder(inst: str, message: str) -> None:
    placeholder_height = _interactive_placeholder_height(inst)
    interactive_placeholder.height = placeholder_height
    interactive_placeholder.min_height = placeholder_height
    interactive_placeholder.object = _interactive_placeholder_markup(inst, message)
    interactive_body.height = placeholder_height
    interactive_body.min_height = placeholder_height
    _set_interactive_body(interactive_placeholder)


def _set_interactive_loading(inst: str, message: str, phase: str = "loading", visible: bool = True) -> None:
    interactive_loading.object = _interactive_loading_notice_markup(inst, message, phase) if visible else ""
    interactive_loading.visible = visible


def _clear_interactive_loading() -> None:
    interactive_loading.object = ""
    interactive_loading.visible = False


def _show_plot(fig: go.Figure, instrument: str | None = None, cache_figure: bool = True) -> None:
    global _DISPLAYED_INTERACTIVE_INSTRUMENT
    instrument = instrument or CURRENT_INSTRUMENT
    plot_height = int(getattr(fig.layout, "height", 900) or 900)
    plot_pane.height = plot_height
    plot_pane.min_height = plot_height
    interactive_body.height = plot_height
    interactive_body.min_height = plot_height
    _set_interactive_body(plot_pane)
    plot_pane.object = fig
    _DISPLAYED_INTERACTIVE_INSTRUMENT = instrument
    if cache_figure:
        _INTERACTIVE_FIGURE_CACHE[instrument] = go.Figure(fig)


def _show_interactive_panel(panel_obj, instrument: str | None = None) -> None:
    global _DISPLAYED_INTERACTIVE_INSTRUMENT
    plot_pane.height = None
    plot_pane.min_height = None
    interactive_body.height = None
    interactive_body.min_height = None
    _set_interactive_body(panel_obj)
    _DISPLAYED_INTERACTIVE_INSTRUMENT = instrument or CURRENT_INSTRUMENT


def _begin_render_request() -> int:
    global _RENDER_REQUEST_COUNTER, _ACTIVE_RENDER_REQUEST_ID
    _RENDER_REQUEST_COUNTER += 1
    _ACTIVE_RENDER_REQUEST_ID = _RENDER_REQUEST_COUNTER
    return _ACTIVE_RENDER_REQUEST_ID


def _render_request_active(request_id: int | None) -> bool:
    return request_id is None or request_id == _ACTIVE_RENDER_REQUEST_ID


def _normalize_cache_value(value):
    if isinstance(value, datetime):
        return _ensure_utc(value).replace(microsecond=0).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime(warn=False).replace(microsecond=0).isoformat()
    if isinstance(value, float):
        return round(value, 4)
    return value


def _as_naive_utc_datetime(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime(warn=False)
    return _ensure_utc(value)


def _floor_datetime(value, step: timedelta):
    value = _as_naive_utc_datetime(value)
    if value is None:
        return None
    step_seconds = max(int(step.total_seconds()), 1)
    timestamp = int(value.replace(tzinfo=timezone.utc).timestamp())
    floored = timestamp - (timestamp % step_seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc).replace(tzinfo=None)


def _is_power_latest_window(start, end, instrument: str) -> bool:
    if instrument != "power":
        return False
    start_dt = _as_naive_utc_datetime(start)
    end_dt = _as_naive_utc_datetime(end)
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        return False
    if abs((end_dt - start_dt) - DEFAULT_WINDOW) > timedelta(minutes=2):
        return False
    _lower, latest = _dataset_time_bounds(instrument)
    latest_dt = _as_naive_utc_datetime(latest)
    if latest_dt is None:
        return False
    return abs(end_dt - latest_dt) <= POWER_LATEST_CACHE_TOLERANCE


def _canonical_interactive_window(start, end, instrument: str):
    """Round Power's live latest window so small timestamp nudges can reuse cache."""
    if not _is_power_latest_window(start, end, instrument):
        return start, end, "exact"
    step = timedelta(minutes=max(1, POWER_LATEST_CACHE_ROUND_MINUTES))
    rounded_end = _floor_datetime(end, step)
    if rounded_end is None:
        return start, end, "exact"
    return rounded_end - DEFAULT_WINDOW, rounded_end, f"power_latest_{POWER_LATEST_CACHE_ROUND_MINUTES}min"


def _summary_context_start(start, instrument: str):
    """Include Power lookback context for cumulative balance anchoring."""
    if instrument != "power":
        return start
    if _power_display_energy_path().exists():
        return start
    start_dt = _as_naive_utc_datetime(start)
    if start_dt is None:
        return start
    context_date = (start_dt - timedelta(days=max(0, POWER_BALANCE_LOOKBACK_DAYS))).date()
    return datetime.combine(context_date, time.min)


def _interactive_render_cache_key(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
    render_quality: str = "full",
) -> tuple[object, ...]:
    start, end, window_cache_mode = _canonical_interactive_window(start, end, instrument)
    if instrument == "power" and window_cache_mode == "exact":
        step = timedelta(minutes=max(1, POWER_GENERAL_CACHE_ROUND_MINUTES))
        rounded_start = _floor_datetime(start, step)
        rounded_end = _floor_datetime(end, step)
        if rounded_start is not None and rounded_end is not None:
            start = rounded_start
            end = rounded_end
            window_cache_mode = f"power_{POWER_GENERAL_CACHE_ROUND_MINUTES}min"
    dataset_version: object = _DATASET_VERSION.get(instrument, 0)
    if window_cache_mode.startswith("power_"):
        dataset_version = window_cache_mode
    return (
        instrument,
        render_quality,
        dataset_version,
        window_cache_mode,
        _normalize_cache_value(start),
        _normalize_cache_value(end),
        _normalize_cache_value(bottom_val),
        _normalize_cache_value(top_val),
        var1_name,
        var2_name,
        _normalize_cache_value(bmin),
        _normalize_cache_value(bmax),
        _normalize_cache_value(lmin),
        _normalize_cache_value(lmax),
        _normalize_cache_value(lymin),
        _normalize_cache_value(lymax),
        _normalize_cache_value(iymin),
        _normalize_cache_value(iymax),
        _normalize_cache_value(rymin),
        _normalize_cache_value(rymax),
    )


def _store_interactive_render_cache(cache_key: tuple[object, ...] | None, fig: go.Figure) -> None:
    if cache_key is None or _is_wxcam_instrument(str(cache_key[0])):
        return
    _INTERACTIVE_RENDER_CACHE[cache_key] = go.Figure(fig)
    _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
    while len(_INTERACTIVE_RENDER_CACHE) > INTERACTIVE_RENDER_CACHE_SIZE:
        _INTERACTIVE_RENDER_CACHE.popitem(last=False)


def _restore_exact_interactive_cache(cache_key: tuple[object, ...] | None, inst: str) -> bool:
    if cache_key is None:
        return False
    cached = _INTERACTIVE_RENDER_CACHE.get(cache_key)
    if cached is None:
        cached = _load_prewarmed_interactive_figure(cache_key, inst)
        if cached is None:
            return False
        _INTERACTIVE_RENDER_CACHE[cache_key] = go.Figure(cached)
        _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
    _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
    _show_plot(go.Figure(cached), instrument=inst, cache_figure=False)
    _perf_log("interactive_render_cache_hit", instrument=inst)
    return True


def _load_prewarmed_interactive_figure(cache_key: tuple[object, ...], inst: str) -> go.Figure | None:
    """Load a latest-view Plotly JSON produced by the quicklook pipeline."""
    if len(cache_key) < 4 or cache_key[0] != inst:
        return None
    window_mode = str(cache_key[3])
    if not window_mode.startswith("power_latest_"):
        return None
    path = _prewarmed_interactive_path(inst)
    if not path.exists():
        return None
    with _timed_perf("interactive_prewarm_load", instrument=inst, path=str(path)) as perf:
        try:
            fig = go.Figure(json.loads(path.read_text()))
        except Exception as exc:
            perf["status"] = "error"
            perf["error"] = str(exc)
            return None
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        return fig


def _publish_plot_if_current(
    fig: go.Figure,
    instrument: str,
    request_id: int | None,
    cache_figure: bool = True,
    cache_key: tuple[object, ...] | None = None,
) -> bool:
    if not _render_request_active(request_id):
        return False
    _show_plot(fig, instrument=instrument, cache_figure=cache_figure)
    if cache_figure:
        _store_interactive_render_cache(cache_key, fig)
    return True


def _publish_panel_if_current(panel_obj, request_id: int | None) -> bool:
    if not _render_request_active(request_id):
        return False
    _show_interactive_panel(panel_obj)
    return True


def _restore_cached_interactive_view(inst: str) -> bool:
    if _is_wxcam_instrument(inst):
        if _DISPLAYED_INTERACTIVE_INSTRUMENT == inst and len(interactive_body.objects) == 1 and interactive_body.objects[0] is wxcam_interactive_browser:
            return True
        _show_interactive_panel(wxcam_interactive_browser, instrument=inst)
        return True
    cached = _INTERACTIVE_FIGURE_CACHE.get(inst)
    if cached is None:
        return False
    if _DISPLAYED_INTERACTIVE_INSTRUMENT == inst and len(interactive_body.objects) == 1 and interactive_body.objects[0] is plot_pane and plot_pane.object is not None:
        return True
    _show_plot(go.Figure(cached), instrument=inst, cache_figure=False)
    return True


def _science_quicklook_path_for_interactive(inst: str, start=None, end=None) -> Path | None:
    cfg = _cfg(inst)
    candidates: list[Path] = []
    if _is_stacked_timeseries_instrument(inst):
        quick_dir = Path(cfg["quicklook_dir"])
        start_dt = _as_naive_utc_datetime(start)
        end_dt = _as_naive_utc_datetime(end)
        if start_dt is not None and end_dt is not None:
            latest_path = summary_latest_png(quick_dir, inst)
            _lower, latest = _dataset_time_bounds(inst)
            latest_dt = _as_naive_utc_datetime(latest)
            if latest_dt is not None and abs(end_dt - latest_dt) <= POWER_LATEST_CACHE_TOLERANCE and latest_path.exists():
                candidates.append(latest_path)
            elif start_dt.date() == end_dt.date() and start_dt.hour == 0:
                candidates.append(summary_daily_png(quick_dir, inst, pd.Timestamp(start_dt)))
        candidates.append(summary_latest_png(quick_dir, inst))
    latest_image = cfg.get("latest_image")
    if latest_image:
        candidates.append(Path(latest_image))
    for path in candidates:
        if path.exists():
            return path
    return None


def _show_cached_quicklook_placeholder(inst: str, start=None, end=None) -> bool:
    path = _science_quicklook_path_for_interactive(inst, start=start, end=end)
    if path is None:
        return False
    panel_obj = pn.Column(
        pn.pane.HTML(
            "<div class='interactive-loading-note'>Showing a cached Science Quicklook while the interactive view loads.</div>",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        ),
        pn.pane.Image(str(path), sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
    _show_interactive_panel(panel_obj, instrument=inst)
    return True


def _interactive_supports_refine(inst: str) -> bool:
    return inst in {"Ceilometer", "Cloud Radar", "Scanning Microwave Radiometer"}


def _interactive_initial_quality(inst: str) -> str:
    """Choose the first render pass used to make an instrument feel responsive."""
    if inst == "power":
        return "downsampled"
    return "coarse" if _interactive_supports_refine(inst) else "full"


def _interactive_final_quality(inst: str) -> str:
    """Return the cache/render quality that represents the settled view."""
    return "downsampled" if inst == "power" else "full"


def _stacked_interactive_max_time_samples(instrument: str, render_quality: str) -> int:
    """Cap 1D interactive trace density by instrument and render pass."""
    if instrument == "power":
        return POWER_INTERACTIVE_MAX_TIME_SAMPLES
    return 900 if render_quality == "coarse" else 1600


def _schedule_next_tick(callback) -> None:
    doc = pn.state.curdoc
    if doc is not None:
        doc.add_next_tick_callback(callback)
    else:
        callback()


def _schedule_timeout(callback, timeout_ms: int):
    doc = pn.state.curdoc
    if doc is not None:
        return doc.add_timeout_callback(callback, timeout_ms)
    callback()
    return None


def _image_type_from_selection(selection: str) -> str:
    cfg = _cfg("wxcam")
    return cfg["vars"][selection]["image_type"]


def _wxcam_today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _wxcam_daily_video_dir(image_type: str) -> Path:
    return _wxcam_daily_video_root() / image_type


def _wxcam_row_time(row) -> datetime | None:
    if not row:
        return None
    try:
        return datetime.fromisoformat(str(row["time_utc"])).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _humanize_age(moment: datetime | None) -> str:
    if moment is None:
        return "Updated time unavailable"
    delta = max(datetime.now(timezone.utc) - moment, timedelta(0))
    seconds = int(delta.total_seconds())
    if seconds < 90:
        return "Updated just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"Updated {minutes} min ago"
    hours = minutes // 60
    if hours < 48:
        rem_minutes = minutes % 60
        return f"Updated {hours} h {rem_minutes:02d} min ago" if rem_minutes else f"Updated {hours} h ago"
    days = hours // 24
    return f"Updated {days} d ago"

def _wxcam_video_context(selection: str, selected_label: str | None) -> dict[str, object]:
    image_type = _image_type_from_selection(selection)
    path_str = _wxcam_interactive_video_options(selection).get(selected_label)
    if not path_str:
        return {}
    video_path = Path(path_str)
    latest_mode = selected_label == "Today (latest)"
    day_token = _wxcam_today_token() if latest_mode else (selected_label or "")
    headline = "Rolling latest 24 hours" if latest_mode else (
        pd.Timestamp(_wxcam_day_token_to_utc(day_token)).strftime("%Y-%m-%d UTC")
        if _wxcam_day_token_to_utc(day_token)
        else (selected_label or "Selected day")
    )
    last_row = latest_record(_wxcam_catalog_path("wxcam"), image_type, media_kind="video")
    last_time = _wxcam_row_time(last_row)
    last_clip_text = last_time.strftime("%H:%M UTC") if last_time else "n/a"
    subtitle_bits = [
        _humanize_age(last_time),
        f"Last clip {last_clip_text}",
    ]
    return {
        "path": video_path,
        "image_type": image_type,
        "selected_label": selected_label or "",
        "headline": headline,
        "day_token": day_token,
        "title": f"{selection} | {headline}",
        "subtitle": " | ".join(bit for bit in subtitle_bits if bit),
    }


def _wxcam_day_strip_markup(image_type: str, day_token: str) -> str:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return ""
    hourly_rows = representative_hourly_records(
        _wxcam_catalog_path("wxcam"),
        image_type,
        day_utc,
        media_kind="image",
    )
    cells: list[str] = []
    for hour in range(24):
        row = hourly_rows.get(hour)
        hour_label = f"{hour:02d}"
        if row is None:
            cells.append(
                "<div class='wxcam-hour-strip__tile wxcam-hour-strip__tile--empty'>"
                f"<div class='wxcam-hour-strip__placeholder'>{hour_label}</div>"
                "</div>"
            )
            continue
        thumb_path = _wxcam_hourly_thumbnail_path(image_type, day_token, str(row["filename"]))
        if thumb_path.exists():
            image_markup = f"<img class='wxcam-hour-strip__thumb' src='{_image_data_uri(thumb_path)}' alt='Hour {hour_label}'>"
        else:
            image_markup = f"<div class='wxcam-hour-strip__placeholder'>{hour_label}</div>"
        cells.append(
            "<div class='wxcam-hour-strip__tile wxcam-hour-strip__tile--day'>"
            f"{image_markup}"
            f"<div class='wxcam-hour-strip__hour'>{hour_label}</div>"
            "</div>"
        )
    return "".join(cells)


def _wxcam_hour_strip_markup(selection: str, selected_label: str | None) -> str:
    context = _wxcam_video_context(selection, selected_label)
    if not context:
        return ""
    if selected_label == "Today (latest)":
        return ""
    image_type = str(context["image_type"])
    body = _wxcam_day_strip_markup(image_type, str(context["day_token"]))
    if not body:
        return ""
    return (
        "<div class='wxcam-hour-strip'>"
        "<div class='wxcam-hour-strip__header'>"
        "<div class='wxcam-hour-strip__title'>Representative hourly stills</div>"
        "<div class='wxcam-hour-strip__hint'>Historical days keep a small hourly image strip for quick visual scanning.</div>"
        "</div>"
        f"<div class='wxcam-hour-strip__scroller'>{body}</div>"
        "</div>"
    )


def _wxcam_interactive_video_options(selection: str) -> dict[str, str | None]:
    image_type = _image_type_from_selection(selection)
    day_dir = _wxcam_daily_video_dir(image_type)
    with _timed_perf("wxcam_interactive_options", instrument="wxcam", image_type=image_type, day_dir=day_dir) as perf:
        if not day_dir.exists():
            perf["status"] = "missing_dir"
            return {"No videos available": None}
        opts: dict[str, str | None] = {}
        video_count = 0
        for video_path in sorted(day_dir.glob("*.mp4")):
            video_count += 1
            if video_path.name == "latest.mp4":
                continue
            opts[video_path.stem] = str(video_path)
        latest_path = day_dir / "latest.mp4"
        if latest_path.exists():
            opts["Today (latest)"] = str(latest_path)
        perf["video_count"] = video_count
        perf["option_count"] = len(opts)
        perf["status"] = "ok" if opts else "empty"
        return opts or {"No videos available": None}


def _wxcam_calendar_options(selection: str) -> dict[str, str | None]:
    """Return WXcam science-quicklook day choices for the HDR thumbnail grid."""
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_calendar_options", instrument="wxcam", image_type=image_type) as perf:
        day_values = available_days(_wxcam_catalog_path("wxcam"), image_type, media_kind="image")
        if not day_values:
            perf["status"] = "empty"
            return {"No images available": None}
        today_token = _wxcam_today_token()
        opts: dict[str, str | None] = {}
        for day_utc in day_values:
            day_token = day_utc.replace("-", "")
            if day_token == today_token:
                continue
            opts[day_token] = day_token
        if any(day_utc.replace("-", "") == today_token for day_utc in day_values):
            opts["Today (latest)"] = today_token
        perf["day_count"] = len(day_values)
        perf["option_count"] = len(opts)
        perf["status"] = "ok"
        return opts or {"No images available": None}


def _wxcam_day_token_to_utc(day_token: str) -> str | None:
    if len(day_token) != 8 or not day_token.isdigit():
        return None
    return f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:8]}"


def _wxcam_calendar_day_token(selected_day: str | None) -> str | None:
    """Normalize the quicklook day selector's latest label to today's token."""
    if selected_day == "Today (latest)":
        return _wxcam_today_token()
    return selected_day


def _wxcam_hourly_image_rows(selection: str, day_token: str | None) -> dict[int, object]:
    day_utc = _wxcam_day_token_to_utc(day_token or "")
    if not day_utc:
        return {}
    image_type = _image_type_from_selection(selection)
    return representative_hourly_records(
        _wxcam_catalog_path("wxcam"),
        image_type,
        day_utc,
        media_kind="image",
    )


def _wxcam_hourly_thumbnail_path(image_type: str, day_token: str, source_name: str) -> Path:
    return _wxcam_hourly_thumbnail_root() / image_type / day_token / f"{Path(source_name).stem}.jpg"


def _wxcam_video_poster_data_uri(image_type: str, day_token: str, selected_label: str) -> str:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return ""
    rows = representative_hourly_records(
        _wxcam_catalog_path("wxcam"),
        image_type,
        day_utc,
        media_kind="image",
    )
    if not rows:
        return ""
    if selected_label == "Today (latest)":
        row = rows[max(rows)]
    else:
        row = rows.get(12) or rows[min(rows)]
    thumb_path = _wxcam_hourly_thumbnail_path(image_type, day_token, str(row["filename"]))
    if not thumb_path.exists():
        return ""
    try:
        return _image_data_uri(thumb_path)
    except Exception:
        return ""


def _media_pane(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".mp4":
        return pn.pane.Video(path, sizing_mode="stretch_width", autoplay=False, loop=False, muted=True)
    return pn.pane.Image(path, sizing_mode="stretch_width")


@lru_cache(maxsize=256)
def _cached_image_data_uri(path_str: str, size_bytes: int, mtime_ns: int) -> str:
    image_bytes = Path(path_str).read_bytes()
    encoded = b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _image_data_uri(path: Path) -> str:
    stat_result = path.stat()
    return _cached_image_data_uri(str(path), stat_result.st_size, stat_result.st_mtime_ns)


wxcam_player = WxcamVideoPlayer(sizing_mode="stretch_width")
wxcam_player_shell = pn.Column(wxcam_player, sizing_mode="stretch_width")


def _build_wxcam_video_view(path: Path, selection: str, selected_label: str, context: dict[str, object]):
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_video_view_build", instrument="wxcam", image_type=image_type, path=path, selected_label=selected_label) as perf:
        mode_class = "wxcam-player--vertical" if image_type == "fish_hdr" else "wxcam-player--wide"
        title = str(context.get("title", f"{selection} | {selected_label}"))
        subtitle = str(context.get("subtitle", ""))
        stat_result = path.stat()
        perf["size_bytes"] = int(stat_result.st_size)
        wxcam_player.src = _wxcam_media_url(path)
        perf["src_mode"] = "static_url"
        wxcam_player.title = title
        wxcam_player.subtitle = subtitle
        wxcam_player.mode_class = mode_class
        wxcam_player.poster = _wxcam_video_poster_data_uri(
            image_type,
            str(context.get("day_token", "")),
            selected_label,
        )
        return wxcam_player_shell


def _build_wxcam_image_view(path: Path, selection: str, selected_label: str):
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_image_view_build", instrument="wxcam", image_type=image_type, path=path, selected_label=selected_label) as perf:
        mode_class = "wxcam-still--vertical" if image_type == "fish_hdr" else "wxcam-still--wide"
        title = escape(f"{selection} | {selected_label} | {path.name}")
        alt = escape(f"{selection} {selected_label}")
        stat_result = path.stat()
        perf["size_bytes"] = int(stat_result.st_size)
        return pn.Column(
            pn.pane.HTML(
                (
                    f"<div class='wxcam-still {mode_class}'>"
                    f"<div class='wxcam-player__meta'><div class='wxcam-player__title'>{title}</div></div>"
                    f"<div class='wxcam-still__frame'><img src='{_image_data_uri(path)}' alt='{alt}'></div>"
                    f"</div>"
                ),
                sizing_mode="stretch_width",
                margin=0,
            ),
            sizing_mode="stretch_width",
        )


wxcam_image_type = pn.widgets.Select(
    name="Image type",
    options=list(_cfg("wxcam")["vars"].keys()),
    value=_cfg("wxcam")["default_top"],
)
_wxcam_ql_options = _wxcam_interactive_video_options(wxcam_image_type.value)
wxcam_date = pn.widgets.Select(name="Date", options=list(_wxcam_ql_options.keys()))
if _wxcam_ql_options:
    wxcam_date.value = list(_wxcam_ql_options.keys())[-1]
wxcam_latest = pn.widgets.Button(name="Latest", button_type="primary")
wxcam_prev = pn.widgets.Button(name="Previous", button_type="default")
wxcam_next = pn.widgets.Button(name="Next", button_type="default")


def _refresh_wxcam_ql_options(preserve_current: bool = True):
    global _wxcam_ql_options
    current = wxcam_date.value if preserve_current else None
    _wxcam_ql_options = _wxcam_interactive_video_options(wxcam_image_type.value or _cfg("wxcam")["default_top"])
    opts = list(_wxcam_ql_options.keys())
    wxcam_date.options = opts
    if not opts:
        wxcam_date.value = None
        return
    if preserve_current and current in opts:
        wxcam_date.value = current
    else:
        wxcam_date.value = opts[-1]


def _shift_wxcam_ql(delta: int):
    _refresh_wxcam_ql_options(preserve_current=True)
    opts = list(wxcam_date.options)
    if not opts or wxcam_date.value not in opts:
        return
    idx = opts.index(wxcam_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        wxcam_date.value = opts[new_idx]
    else:
        wxcam_date.param.trigger("value")


def _go_wxcam_latest(_event=None):
    _refresh_wxcam_ql_options(preserve_current=True)
    opts = list(wxcam_date.options)
    if "Today (latest)" in opts:
        if wxcam_date.value == "Today (latest)":
            wxcam_date.param.trigger("value")
        else:
            wxcam_date.value = "Today (latest)"
    elif opts:
        if wxcam_date.value == opts[-1]:
            wxcam_date.param.trigger("value")
        else:
            wxcam_date.value = opts[-1]


wxcam_latest.on_click(_go_wxcam_latest)
wxcam_prev.on_click(lambda _e: _shift_wxcam_ql(-1))
wxcam_next.on_click(lambda _e: _shift_wxcam_ql(1))


class WxcamSelectionState(param.Parameterized):
    selected_hour_path = param.String(default="")


wxcam_calendar_state = WxcamSelectionState()


def _refresh_wxcam_latest_if_needed():
    # WXcam should not auto-refresh while selected. We refresh its options when
    # the user switches away and comes back, or when they explicitly change the
    # WXcam controls.
    return


_wxcam_ql_timer = _safe_periodic_callback(_refresh_wxcam_latest_if_needed, period=300_000, start=True)


@pn.depends(wxcam_date.param.value, wxcam_image_type.param.value)
def _wxcam_interactive_media(selected, selection):
    selection = selection or _cfg("wxcam")["default_top"]
    with _timed_perf("wxcam_interactive_render", instrument="wxcam", selection=selection, selected=selected) as perf:
        context = _wxcam_video_context(selection, selected)
        path = context.get("path")
        if not path:
            perf["status"] = "missing_option"
            return pn.pane.Markdown("No media available for this selection.")
        video_path = Path(str(path))
        perf["path"] = video_path
        if not video_path.exists():
            perf["status"] = "missing_file"
            return pn.pane.Markdown("No media available for this selection.")
        perf["status"] = "ok"
        return _build_wxcam_video_view(video_path, selection, selected or "", context)


@pn.depends(wxcam_date.param.value, wxcam_image_type.param.value)
def _wxcam_interactive_hour_strip(selected, selection):
    selection = selection or _cfg("wxcam")["default_top"]
    return pn.pane.HTML(
        _wxcam_hour_strip_markup(selection, selected),
        sizing_mode="stretch_width",
        margin=0,
    )


# WXcam now uses the Interactive Data Browser as its primary browser/player so
# the controls and playback state live in one place instead of competing with
# the science-quicklook flow used by the other instruments.
wxcam_interactive_browser = pn.Column(
    pn.Card(
        pn.Row(wxcam_image_type, wxcam_latest, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(wxcam_prev, wxcam_date, wxcam_next, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card", "wxcam-browser__toolbar"],
    ),
    _wxcam_interactive_hour_strip,
    _wxcam_interactive_media,
    sizing_mode="stretch_width",
    css_classes=["wxcam-browser"],
)

wxcam_image_type.param.watch(_on_wxcam_image_type_change, "value")


def _update_wxcam_view(start, end, top_name: str, bottom_name: str, request_id: int | None = None) -> bool:
    return _publish_panel_if_current(wxcam_interactive_browser, request_id)


def _update_hatpro_view(
    start,
    end,
    bottom_val,
    top_val,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    request_id: int | None = None,
    render_quality: str = "full",
    cache_key: tuple[object, ...] | None = None,
):
    """Custom renderer for HATPRO radiometer: split LWP/IWV and IRR; T_PROF heatmap."""
    print(f"[hatpro] render window {start} -> {end}")
    with _timed_perf(
        "hatpro_render",
        instrument="Scanning Microwave Radiometer",
        start=start,
        end=end,
        render_quality=render_quality,
    ) as perf:
        if not _render_request_active(request_id):
            perf["status"] = "stale_before_start"
            return False
        bottom = max(float(bottom_val), 0.0)
        top = max(float(top_val), bottom + 100.0)
        ds = open_window(
            start,
            end,
            bottom_m=bottom,
            top_m=top,
            instrument="Scanning Microwave Radiometer",
            render_quality=render_quality,
        )
        cfg = _cfg("Scanning Microwave Radiometer")
        times = pd.to_datetime(ds["time"].values) if "time" in ds else None
        perf["time_count"] = 0 if times is None else int(len(times))
        if times is None or len(times) == 0:
            perf["status"] = "no_data"
            fig = go.Figure()
            fig.add_annotation(
                text="No data",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=THEME_TEXT, size=16),
            )
            fig.update_layout(height=600, margin=dict(l=40, r=40, t=40, b=40))
            return _publish_plot_if_current(fig, "Scanning Microwave Radiometer", request_id, cache_key=cache_key)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]],
            row_heights=[0.25, 0.25, 0.5],
            subplot_titles=("LWP / IWV", "IRR / SURF_T", cfg["vars"]["T_PROF"]["label"]),
        )
        # Color the subplot titles to match their traces.
        if len(fig.layout.annotations) >= 1:
            fig.layout.annotations[0].update(text='<span style="color:#1f77b4">LWP</span> / <span style="color:#2ca02c">IWV</span>', font=dict(size=14))
        if len(fig.layout.annotations) >= 2:
            fig.layout.annotations[1].update(
                text='<span style="color:#d62728">IRR</span> / <span style="color:#9467bd">SURF_T</span>',
                font=dict(size=14),
            )
        if len(fig.layout.annotations) >= 3:
            fig.layout.annotations[2].update(font=dict(size=14))

        # Row 1: LWP (left), IWV (right, kg/m²)
        if "LWP" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["LWP"]),
                    mode="lines",
                    name="LWP (g/m²)",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        if "IWV" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["IWV"]),
                    mode="lines",
                    name="IWV (kg/m²)",
                    line=dict(color="#2ca02c", width=2, dash="dot"),
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # Row 2: IRR only
        if "IRR_Map" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["IRR_Map"]),
                    mode="lines",
                    name="IRR (°C)",
                    line=dict(color="#d62728", width=2),
                ),
                row=2,
                col=1,
            )
        if "SURF_T" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["SURF_T"]) - 273.15,  # convert K to °C for display
                    mode="lines",
                    name="SURF_T (°C)",
                    line=dict(color="#9467bd", width=2, dash="dot"),
                ),
                row=2,
                col=1,
            )

        # Row 3: Temperature profile heatmap + contours
        if "T_PROF" in ds:
            heights = ds["range"].values if "range" in ds else np.arange(ds["T_PROF"].shape[1])
            temps = np.array(ds["T_PROF"].transpose("range", "time"))
            vmin, vmax = cfg["vars"]["T_PROF"]["clim"]
            fig.add_trace(
                go.Heatmap(
                    x=times,
                    y=heights,
                    z=temps,
                    zmin=vmin,
                    zmax=vmax,
                    coloraxis="coloraxis",
                    showscale=True,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Contour(
                    x=times,
                    y=heights,
                    z=temps,
                    showscale=False,
                    contours=dict(coloring="none", showlabels=True, labelfont=dict(color="white", size=10)),
                    line=dict(color="white", width=1),
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

        tickvals = []
        ticktext = []
        noon_annots = []
        if start and end:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            duration = end_ts - start_ts
            freq = "2h" if duration > pd.Timedelta(hours=24) else "h"
            hours = pd.date_range(start=start_ts.floor("h"), end=end_ts.ceil("h"), freq=freq)
            for t in hours:
                tickvals.append(t.to_pydatetime())
                ticktext.append(t.strftime("%H:%M"))
                if t.hour == 12:
                    noon_annots.append(
                        dict(
                            x=t.to_pydatetime(),
                            y=-0.06,
                            xref="x",
                            yref="paper",
                            text=t.strftime("%Y-%m-%d"),
                            showarrow=False,
                            xanchor="center",
                            yanchor="top",
                            font=dict(size=14, color=THEME_TEXT),
                        )
                    )

        # x-axes
        for row in (1, 2, 3):
            fig.update_xaxes(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=-45,
                showgrid=True,
                gridcolor=THEME_GRID,
                linecolor=THEME_LINE,
                tickfont=dict(color=THEME_TEXT, size=12),
                title_font=dict(color=THEME_TEXT, size=12),
                row=row,
                col=1,
            )
        fig.update_xaxes(title_text="Date and Time (UTC)", title_standoff=40, row=3, col=1)

        # y-axes
        fig.update_yaxes(title_text="LWP (g/m²)", range=[float(lymin), float(lymax)], row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="IWV (kg/m²)", range=[float(iymin), float(iymax)], row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="IRR / SURF_T (°C)", range=[float(rymin), float(rymax)], row=2, col=1)
        fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=3, col=1, showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE)

        fig.update_layout(
            showlegend=False,
            height=max(700, int(pn.state.viewport_height * 0.8)) if hasattr(pn.state, "viewport_height") else 900,
            margin=dict(l=60, r=80, t=30, b=110),
            coloraxis=dict(
                colorscale=cfg["vars"]["T_PROF"]["colorscale"],
                cmin=cfg["vars"]["T_PROF"]["clim"][0],
                cmax=cfg["vars"]["T_PROF"]["clim"][1],
                colorbar=dict(title=dict(text="T (K)", side="right"), x=1.02, y=0.18, len=0.3, tickfont=dict(color=THEME_TEXT, size=9)),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=THEME_TEXT, size=13),
            annotations=tuple(list(fig.layout.annotations) + noon_annots),
        )
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        return _publish_plot_if_current(fig, "Scanning Microwave Radiometer", request_id, cache_key=cache_key)


def _update_stacked_timeseries_view(
    instrument: str,
    start,
    end,
    request_id: int | None = None,
    render_quality: str = "full",
    cache_key: tuple[object, ...] | None = None,
):
    """Render a 1D summary instrument with fixed multi-panel layouts."""
    print(f"[{instrument}] render window {start} -> {end}")
    with _timed_perf(
        "stacked_timeseries_render",
        instrument=instrument,
        start=start,
        end=end,
        render_quality=render_quality,
    ) as perf:
        if not _render_request_active(request_id):
            perf["status"] = "stale_before_start"
            return False
        source_instruments = summary_source_instruments(instrument)
        perf["source_instruments"] = list(source_instruments)
        power_display_available = instrument == "power" and _power_display_energy_path().exists()
        # Power cumulative-energy traces now come from a compact display product.
        # Only keep full raw time detail as a fallback when that product is absent.
        if instrument == "power" and not power_display_available:
            source_render_quality = "summary_full_time"
        else:
            source_render_quality = render_quality
        perf["source_render_quality"] = source_render_quality
        perf["power_display_energy_available"] = bool(power_display_available)
        context_start = _summary_context_start(start, instrument)
        perf["context_start"] = context_start
        source_open_started = perf_counter()
        source_windows = [
            open_window(context_start, end, instrument=source_inst, render_quality=source_render_quality)
            for source_inst in source_instruments
        ]
        if instrument == "power":
            display_window = _open_power_display_energy_window(start, end)
            if display_window is not None:
                source_windows.append(display_window)
                perf["power_display_energy"] = "used"
            else:
                perf["power_display_energy"] = "missing"
        perf["source_open_ms"] = round((perf_counter() - source_open_started) * 1000, 3)
        combine_started = perf_counter()
        ds = combine_summary_datasets(instrument, *source_windows)
        perf["combine_ms"] = round((perf_counter() - combine_started) * 1000, 3)
        if instrument == "power" and ds is not None:
            display_start = _as_naive_utc_datetime(start)
            display_end = _as_naive_utc_datetime(end)
            if display_start is not None:
                ds.attrs[SUMMARY_DISPLAY_START_ATTR] = display_start.isoformat()
            if display_end is not None:
                ds.attrs[SUMMARY_DISPLAY_END_ATTR] = display_end.isoformat()
        times = pd.to_datetime(ds["time"].values) if ds is not None and "time" in ds else None
        perf["time_count"] = 0 if times is None else int(len(times))
        if times is None or len(times) == 0:
            perf["status"] = "no_data"
            empty = go.Figure()
            empty.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            empty.update_layout(height=600, paper_bgcolor="white", plot_bgcolor="white")
            return _publish_plot_if_current(empty, instrument, request_id, cache_key=cache_key)
        try:
            max_time_samples = _stacked_interactive_max_time_samples(instrument, render_quality)
            perf["max_time_samples"] = max_time_samples
            if instrument == "power":
                perf["plot_density_mode"] = "time_downsampled"
            fig_started = perf_counter()
            fig = build_summary_plotly(ds, instrument, title=display_name(instrument), max_time_samples=max_time_samples)
            perf["figure_build_ms"] = round((perf_counter() - fig_started) * 1000, 3)
        except ValueError:
            perf["status"] = "no_data"
            empty = go.Figure()
            empty.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            empty.update_layout(height=600, paper_bgcolor="white", plot_bgcolor="white")
            return _publish_plot_if_current(empty, instrument, request_id, cache_key=cache_key)
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        return _publish_plot_if_current(fig, instrument, request_id, cache_key=cache_key)


def _render_interactive_view(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
    request_id: int | None = None,
    render_quality: str = "full",
):
    """Render the current interactive view, dropping stale work if a newer request arrives."""
    if _instrument_guard:
        return
    global CURRENT_INSTRUMENT
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    start, end, cache_window_mode = _canonical_interactive_window(start, end, instrument)
    print(f"[render-view] instrument={instrument} quality={render_quality} request={request_id}")
    with _timed_perf(
        "interactive_view_update",
        instrument=instrument,
        start=start,
        end=end,
        render_quality=render_quality,
        request_id=request_id,
    ) as perf:
        if not _render_request_active(request_id):
            perf["status"] = "stale_before_start"
            return
        cache_key = _interactive_render_cache_key(
            start,
            end,
            bottom_val,
            top_val,
            var1_name,
            var2_name,
            bmin,
            bmax,
            lmin,
            lmax,
            lymin,
            lymax,
            iymin,
            iymax,
            rymin,
            rymax,
            instrument,
            render_quality=render_quality,
        )
        perf["cache_window_mode"] = cache_window_mode
        perf["top_var"] = var1_name
        perf["bottom_var"] = var2_name
        perf["bottom_m"] = bottom_val
        perf["top_m"] = top_val
        if instrument == "Scanning Microwave Radiometer":
            perf["view_type"] = "hatpro"
            published = _update_hatpro_view(
                start,
                end,
                bottom_val,
                top_val,
                lymin,
                lymax,
                iymin,
                iymax,
                rymin,
                rymax,
                request_id=request_id,
                render_quality=render_quality,
                cache_key=cache_key,
            )
            if not published:
                perf["status"] = "stale"
                return
            perf["status"] = "ok"
        elif _is_wxcam_instrument(instrument):
            perf["view_type"] = "wxcam"
            published = _update_wxcam_view(start, end, var1_name, var2_name, request_id=request_id)
            if not published:
                perf["status"] = "stale"
                return
            perf["status"] = "ok"
        elif _is_stacked_timeseries_instrument(instrument):
            perf["view_type"] = "stacked_timeseries"
            published = _update_stacked_timeseries_view(
                instrument,
                start,
                end,
                request_id=request_id,
                render_quality=render_quality,
                cache_key=cache_key,
            )
            if not published:
                perf["status"] = "stale"
                return
            perf["status"] = "ok"
        else:
            perf["view_type"] = "heatmap"
            bottom = max(float(bottom_val), 0.0)
            top = max(float(top_val), bottom + 100.0)
            ds = open_window(start, end, bottom_m=bottom, top_m=top, instrument=instrument, render_quality=render_quality)
            cfg = _cfg()
            vars_cfg = cfg["vars"]
            var1 = vars_cfg.get(var1_name)
            var2 = vars_cfg.get(var2_name)
            bg = "white"
            fg = THEME_TEXT
            grid = THEME_GRID
            perf["time_count"] = int(ds.sizes.get("time", 0)) if ds is not None else 0
            perf["range_count"] = int(ds.sizes.get("range", 0)) if ds is not None else 0
            if ds is None or not ds.data_vars:
                perf["status"] = "no_data"
                fig = go.Figure()
                fig.add_annotation(
                    text="No data",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color=fg, size=16),
                )
                fig.update_layout(height=600, paper_bgcolor=bg, plot_bgcolor=bg, margin=dict(l=40, r=40, t=40, b=40))
                if not _publish_plot_if_current(fig, instrument, request_id, cache_key=cache_key):
                    perf["status"] = "stale"
                    return
            else:
                if var1 and var1.get("log"):
                    b_cmin = np.log10(bmin)
                    b_cmax = np.log10(bmax)
                    b_tickvals = list(range(int(np.floor(b_cmin)), int(np.ceil(b_cmax)) + 1))
                    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
                    b_ticktext = [f"10{str(v).translate(sup)}" for v in b_tickvals]
                else:
                    b_cmin, b_cmax = bmin, bmax
                    b_tickvals = None
                    b_ticktext = None
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    shared_yaxes=False,
                    vertical_spacing=0.08,
                    subplot_titles=(var1["label"] if var1 else "", var2["label"] if var2 else ""),
                )
                if var1 and var1_name in ds:
                    fig.add_trace(_make_plot(ds, var1_name, (bmin, bmax), var1.get("log", False), coloraxis="coloraxis"), row=1, col=1)
                if var2 and var2_name in ds:
                    if var1_name == "beta_att" and var2_name == "linear_depol_ratio" and "beta_att" in ds:
                        times = pd.to_datetime(ds["time"].values)
                        heights = ds["range"].values
                        ldr = np.array(ds[var2_name].transpose("range", "time"))
                        beta_vals = np.array(ds["beta_att"].transpose("range", "time"))
                        ldr = np.where((ldr >= 0.0) & (ldr <= 1.0), ldr, np.nan)
                        mask_threshold = 10 ** -6.5
                        ldr = np.where(beta_vals >= mask_threshold, ldr, np.nan)
                        fig.add_trace(
                            go.Heatmap(
                                x=times,
                                y=heights,
                                z=ldr,
                                zmin=lmin,
                                zmax=lmax,
                                coloraxis="coloraxis2",
                                showscale=False,
                            ),
                            row=2,
                            col=1,
                        )
                    else:
                        times = pd.to_datetime(ds["time"].values)
                        heights = ds["range"].values
                        ldr = np.array(ds[var2_name].transpose("range", "time"))
                        fig.add_trace(
                            go.Heatmap(
                                x=times,
                                y=heights,
                                z=ldr,
                                zmin=lmin,
                                zmax=lmax,
                                coloraxis="coloraxis2",
                                showscale=False,
                            ),
                            row=2,
                            col=1,
                        )
                fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=1, col=1)
                fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=2, col=1)
                tickvals = []
                ticktext = []
                noon_annots = []
                if start and end:
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    duration = end_ts - start_ts
                    freq = "2h" if duration > pd.Timedelta(hours=24) else "h"
                    hours = pd.date_range(start=start_ts.floor("h"), end=end_ts.ceil("h"), freq=freq)
                    for t in hours:
                        tickvals.append(t.to_pydatetime())
                        ticktext.append(t.strftime("%H:%M"))
                        if t.hour == 12:
                            noon_annots.append(
                                dict(
                                    x=t.to_pydatetime(),
                                    y=-0.06,
                                    xref="x",
                                    yref="paper",
                                    text=t.strftime("%Y-%m-%d"),
                                    showarrow=False,
                                    xanchor="center",
                                    yanchor="top",
                                    font=dict(size=14, color=fg),
                                )
                            )
                fig.update_xaxes(
                    title_text="Date and Time (UTC)",
                    title_standoff=50,
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=-45,
                    showgrid=True,
                    gridcolor=grid,
                    linecolor=THEME_LINE,
                    tickfont=dict(color=fg, size=12),
                    title_font=dict(color=fg, size=12),
                    row=2,
                    col=1,
                )
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=-45,
                    showgrid=True,
                    gridcolor=grid,
                    linecolor=THEME_LINE,
                    tickfont=dict(color=fg, size=12),
                    title_font=dict(color=fg, size=12),
                    row=1,
                    col=1,
                )
                fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=THEME_LINE, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=1, col=1)
                fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=THEME_LINE, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=2, col=1)
                fig.update_yaxes(matches="y", row=2, col=1)
                fig.update_layout(
                    height=max(630, int(pn.state.viewport_height * 0.675)) if hasattr(pn.state, "viewport_height") else 810,
                    margin=dict(l=50, r=70, t=30, b=90),
                    coloraxis=dict(
                        colorscale=var1["colorscale"] if var1 else "Cividis",
                        cmin=b_cmin,
                        cmax=b_cmax,
                        colorbar=dict(
                            title=dict(text=var1["label"] if var1 else "", side="right"),
                            x=1.04,
                            y=0.77,
                            len=0.35,
                            tickvals=b_tickvals,
                            ticktext=b_ticktext,
                            tickfont=dict(color=fg, size=9),
                        ),
                    ),
                    coloraxis2=dict(
                        colorscale=var2["colorscale"] if var2 else "Viridis",
                        cmin=lmin,
                        cmax=lmax,
                        colorbar=dict(title=dict(text=var2["label"] if var2 else "", side="right"), x=1.04, y=0.27, len=0.35, tickfont=dict(color=fg, size=9)),
                    ),
                    paper_bgcolor=bg,
                    plot_bgcolor=bg,
                    font=dict(color=fg, size=13),
                    annotations=tuple(list(fig.layout.annotations) + noon_annots),
                )
                perf["status"] = "ok"
                perf["trace_count"] = len(fig.data)
                if not _publish_plot_if_current(fig, instrument, request_id, cache_key=cache_key):
                    perf["status"] = "stale"
                    return
        if (
            render_quality == "coarse"
            and _interactive_supports_refine(instrument)
            and _render_request_active(request_id)
            and perf.get("status") == "ok"
        ):
            _set_interactive_loading(instrument, "Refining higher-detail data for this view.", phase="refining", visible=True)
            _schedule_next_tick(
                lambda: _render_interactive_view(
                    start,
                    end,
                    bottom_val,
                    top_val,
                    var1_name,
                    var2_name,
                    bmin,
                    bmax,
                    lmin,
                    lmax,
                    lymin,
                    lymax,
                    iymin,
                    iymax,
                    rymin,
                    rymax,
                    instrument,
                    request_id=request_id,
                    render_quality="full",
                )
            )
        elif _render_request_active(request_id):
            _clear_interactive_loading()


def _start_interactive_render(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
):
    if _instrument_guard:
        return
    global CURRENT_INSTRUMENT
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    _capture_current_instrument_state(instrument)
    start, end, _cache_window_mode = _canonical_interactive_window(start, end, instrument)
    request_id = _begin_render_request()
    final_quality = _interactive_final_quality(instrument)
    first_quality = _interactive_initial_quality(instrument)
    full_cache_key = _interactive_render_cache_key(
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
        render_quality=final_quality,
    )
    if _restore_exact_interactive_cache(full_cache_key, instrument):
        _clear_interactive_loading()
        return
    first_cache_key = _interactive_render_cache_key(
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
        render_quality=first_quality,
    )
    has_cached_view = _restore_exact_interactive_cache(first_cache_key, instrument)
    if has_cached_view and first_quality == "coarse":
        first_quality = "full"
    if not has_cached_view:
        has_cached_view = _restore_cached_interactive_view(instrument)
    loading_message = "Refreshing the current selection." if has_cached_view else "Preparing the latest view for this instrument."
    _set_interactive_loading(instrument, loading_message, phase="loading", visible=True)
    if not has_cached_view:
        if not _show_cached_quicklook_placeholder(instrument, start=start, end=end):
            _show_interactive_placeholder(instrument, "Loading the selected window and preparing the first pass of the plot.")

    _schedule_next_tick(
        lambda: _render_interactive_view(
            start,
            end,
            bottom_val,
            top_val,
            var1_name,
            var2_name,
            bmin,
            bmax,
            lmin,
            lmax,
            lymin,
            lymax,
            iymin,
            iymax,
            rymin,
            rymax,
            instrument,
            request_id=request_id,
            render_quality=first_quality,
        )
    )


def _schedule_interactive_render(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
):
    if _instrument_guard:
        return
    global CURRENT_INSTRUMENT, _PENDING_INTERACTIVE_RENDER_ARGS, _PENDING_INTERACTIVE_RENDER_CB
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    _PENDING_INTERACTIVE_RENDER_ARGS = (
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
    )
    if _APP_BOOTSTRAPPING or not _INTERACTIVE_RENDER_ENABLED:
        if _DISPLAYED_INTERACTIVE_INSTRUMENT is None:
            if not _show_cached_quicklook_placeholder(instrument, start=start, end=end):
                _show_interactive_placeholder(instrument, "Preparing the initial dashboard view.")
        _perf_log("interactive_render_deferred", instrument=instrument)
        return
    if _PENDING_INTERACTIVE_RENDER_CB is not None:
        _perf_log("interactive_render_debounced", instrument=instrument)
        return

    def _flush_pending_interactive_render() -> None:
        global _PENDING_INTERACTIVE_RENDER_ARGS, _PENDING_INTERACTIVE_RENDER_CB
        args = _PENDING_INTERACTIVE_RENDER_ARGS
        _PENDING_INTERACTIVE_RENDER_ARGS = None
        _PENDING_INTERACTIVE_RENDER_CB = None
        if args is not None:
            _start_interactive_render(*args)

    _PENDING_INTERACTIVE_RENDER_CB = _schedule_timeout(_flush_pending_interactive_render, RENDER_DEBOUNCE_MS)


@pn.depends(
    range_start.param.value,
    range_end.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
    var1_select.param.value,
    var2_select.param.value,
    beta_vmin.param.value,
    beta_vmax.param.value,
    ldr_vmin.param.value,
    ldr_vmax.param.value,
    lwp_ymin.param.value,
    lwp_ymax.param.value,
    iwv_ymin.param.value,
    iwv_ymax.param.value,
    irr_ymin.param.value,
    irr_ymax.param.value,
    instrument_select.param.value,
    watch=True,
)
def _update_view(start, end, bottom_val, top_val, var1_name, var2_name, bmin, bmax, lmin, lmax, lymin, lymax, iymin, iymax, rymin, rymax, instrument):
    """Schedule an interactive render while keeping the current pane warm."""
    _schedule_interactive_render(
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
    )


def _schedule_current_interactive_render() -> None:
    _schedule_interactive_render(
        range_start.value,
        range_end.value,
        bottom_range_m.value,
        top_range_m.value,
        var1_select.value,
        var2_select.value,
        beta_vmin.value,
        beta_vmax.value,
        ldr_vmin.value,
        ldr_vmax.value,
        lwp_ymin.value,
        lwp_ymax.value,
        iwv_ymin.value,
        iwv_ymax.value,
        irr_ymin.value,
        irr_ymax.value,
        instrument_select.value,
    )


def _enable_browser_interactive_render() -> None:
    global _INTERACTIVE_RENDER_ENABLED
    if pn.state.curdoc is None:
        return
    _INTERACTIVE_RENDER_ENABLED = True
    _schedule_timeout(_activate_interactive_footer_metrics, 750)
    _schedule_timeout(_schedule_current_interactive_render, 250)


def _parse_relayout_time(val):
    """Parse plotly relayout timestamps safely."""
    try:
        return pd.Timestamp(pd.to_datetime(val)).to_pydatetime(warn=False)
    except Exception:
        return None


def _on_relayout(event):
    """When the user zooms/pans, sync controls (and disable live) so we reload data at higher detail."""
    global _relayout_guard
    if _relayout_guard:
        return
    data = event.new or {}
    x0 = data.get("xaxis.range[0]")
    x1 = data.get("xaxis.range[1]")
    start = _parse_relayout_time(x0) if x0 is not None else None
    end = _parse_relayout_time(x1) if x1 is not None else None
    if start is not None and end is not None and start > end:
        start, end = end, start
    y0 = None
    y1 = None
    if not _is_stacked_timeseries_instrument(CURRENT_INSTRUMENT) and not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        y0 = data.get("yaxis.range[0]") or data.get("yaxis2.range[0]")
        y1 = data.get("yaxis.range[1]") or data.get("yaxis2.range[1]")
        if y0 is not None and y1 is not None and y0 > y1:
            y0, y1 = y1, y0
    if start is None and end is None and y0 is None and y1 is None:
        return
    _perf_log(
        "plot_relayout",
        instrument=CURRENT_INSTRUMENT,
        start=start,
        end=end,
        y0=y0,
        y1=y1,
        request_path=_request_path(),
        **_selection_snapshot(),
    )
    _relayout_guard = True
    try:
        _set_live(False)
        if start is not None and end is not None:
            range_start.value = start
            range_end.value = end
        if y0 is not None and y1 is not None:
            low = max(float(y0), 0.0)
            high = max(float(y1), low + 100.0)
            bottom_range_m.value = int(low)
            top_range_m.value = int(high)
    finally:
        _relayout_guard = False


plot_pane.param.watch(_on_relayout, "relayout_data")
# Initial render
_update_view(
    range_start.value,
    range_end.value,
    bottom_range_m.value,
    top_range_m.value,
    var1_select.value,
    var2_select.value,
    beta_vmin.value,
    beta_vmax.value,
    ldr_vmin.value,
    ldr_vmax.value,
    lwp_ymin.value,
    lwp_ymax.value,
    iwv_ymin.value,
    iwv_ymax.value,
    irr_ymin.value,
    irr_ymax.value,
    instrument_select.value,
)




# -------- Science / housekeeping quicklooks --------

def _empty_quicklook_options(mode: str) -> dict[str, None]:
    if mode == "housekeeping":
        return {"No housekeeping quicklooks available": None}
    return {"No images available": None}


def _housekeeping_quicklook_label(inst: str) -> str | None:
    return housekeeping_label(inst) if _is_stacked_timeseries_instrument(inst) else extra_housekeeping_label(inst)


def _housekeeping_latest_path(inst: str, quick_dir: Path) -> Path | None:
    return housekeeping_latest_png(quick_dir, inst) if _is_stacked_timeseries_instrument(inst) else extra_housekeeping_latest_png(quick_dir, inst)


def _housekeeping_daily_path(inst: str, quick_dir: Path, token: str) -> Path | None:
    return housekeeping_daily_png(quick_dir, inst, token) if _is_stacked_timeseries_instrument(inst) else extra_housekeeping_daily_png(quick_dir, inst, token)


def _housekeeping_tokens(inst: str, quick_dir: Path) -> list[str]:
    if _is_stacked_timeseries_instrument(inst):
        label = _housekeeping_quicklook_label(inst)
        if label is None:
            return []
        return [token for token in calendar_date_tokens(quick_dir, inst) if (_housekeeping_daily_path(inst, quick_dir, token) or Path()).exists()]
    return extra_housekeeping_tokens(quick_dir, inst)


def _quicklook_options(inst: str | None = None, wxcam_selection: str | None = None, mode: str = "science"):
    """Build a mapping of label -> quicklook asset token/path for a quicklook mode."""
    inst = inst or CURRENT_INSTRUMENT
    cfg = _cfg(inst)
    opts = {}
    if _is_wxcam_instrument(inst):
        if mode == "housekeeping":
            return _empty_quicklook_options(mode)
        return _wxcam_calendar_options(wxcam_selection or _cfg("wxcam")["default_top"])
    if _is_stacked_timeseries_instrument(inst):
        quick_dir = cfg["quicklook_dir"]
        if mode == "science" and quick_dir.exists():
            for token in calendar_date_tokens(quick_dir, inst):
                if summary_daily_png(quick_dir, inst, token).exists():
                    opts[token] = token
            if summary_latest_png(quick_dir, inst).exists():
                opts["Today (latest)"] = "latest"
        elif mode == "housekeeping" and quick_dir.exists():
            hk_latest = _housekeeping_latest_path(inst, quick_dir)
            for token in _housekeeping_tokens(inst, quick_dir):
                hk_path = _housekeeping_daily_path(inst, quick_dir, token)
                if hk_path and hk_path.exists():
                    opts[token] = token
            if hk_latest and hk_latest.exists():
                opts["Today (latest)"] = "latest"
        return opts or _empty_quicklook_options(mode)
    if mode == "housekeeping":
        quick_dir = cfg["quicklook_dir"]
        hk_latest = _housekeeping_latest_path(inst, quick_dir)
        if quick_dir.exists():
            for token in _housekeeping_tokens(inst, quick_dir):
                hk_path = _housekeeping_daily_path(inst, quick_dir, token)
                if hk_path and hk_path.exists():
                    opts[token] = token
            if hk_latest and hk_latest.exists():
                opts["Today (latest)"] = "latest"
        return opts or _empty_quicklook_options(mode)
    quick_dir = cfg["quicklook_dir"]
    latest = cfg["latest_image"]
    # Collect dated quicklooks (sorted ascending), then append "Today" last.
    date_labels = []
    if quick_dir.exists():
        for png in sorted(quick_dir.glob("*.png")):
            if png.name == latest.name:
                continue
            stem = png.stem
            if stem.startswith("cloud_radar_"):
                label = stem.removeprefix("cloud_radar_")
            elif stem.startswith("ceilometer_"):
                label = stem.removeprefix("ceilometer_")
            elif stem.startswith("vaisalamet_"):
                label = stem.removeprefix("vaisalamet_")
            elif stem.startswith("asfs_logger_"):
                label = stem.removeprefix("asfs_logger_")
            elif stem.startswith("power_"):
                label = stem.removeprefix("power_")
            elif stem.startswith("hatpro_"):
                label = stem.removeprefix("hatpro_")
            else:
                label = stem
            date_labels.append((label, str(png)))
    for label, path in date_labels:
        opts[label] = path
    if latest.exists():
        opts["Today (latest)"] = str(latest)
    return opts or _empty_quicklook_options(mode)


_ql_options = _quicklook_options(mode="science")
ql_date = pn.widgets.Select(name="Date", options=list(_ql_options.keys()))
if _ql_options:
    ql_date.value = list(_ql_options.keys())[-1]

_hk_options = _quicklook_options(mode="housekeeping")
hk_date = pn.widgets.Select(name="Date", options=list(_hk_options.keys()))
if _hk_options:
    hk_date.value = list(_hk_options.keys())[-1]


def _refresh_ql_options(preserve_current: bool = True):
    """Refresh available science quicklook options, optionally preserving current selection."""
    global _ql_options
    current = ql_date.value if preserve_current else None
    _ql_options = _quicklook_options(science_instrument.value, science_image_type.value, mode="science")
    opts = list(_ql_options.keys())
    ql_date.options = opts
    if not opts:
        ql_date.value = None
        return
    if preserve_current and current in opts:
        ql_date.value = current
    elif _is_wxcam_instrument(science_instrument.value) or _is_stacked_timeseries_instrument(science_instrument.value):
        ql_date.value = "Today (latest)" if "Today (latest)" in opts else opts[-1]
    else:
        ql_date.value = opts[-1]


def _refresh_hk_options(preserve_current: bool = True):
    """Refresh available housekeeping quicklook options, optionally preserving current selection."""
    global _hk_options
    current = hk_date.value if preserve_current else None
    _hk_options = _quicklook_options(hk_instrument.value, mode="housekeeping")
    opts = list(_hk_options.keys())
    hk_date.options = opts
    if not opts:
        hk_date.value = None
        return
    if preserve_current and current in opts:
        hk_date.value = current
    elif _is_stacked_timeseries_instrument(hk_instrument.value):
        hk_date.value = "Today (latest)" if "Today (latest)" in opts else opts[-1]
    else:
        hk_date.value = opts[-1]

# Quicklook navigation buttons
ql_prev = pn.widgets.Button(name="<<", button_type="default")
ql_next = pn.widgets.Button(name=">>", button_type="default")
hk_prev = pn.widgets.Button(name="<<", button_type="default")
hk_next = pn.widgets.Button(name=">>", button_type="default")


def _shift_ql(delta: int):
    """Move science quicklook day selection by delta steps."""
    _refresh_ql_options(preserve_current=True)
    opts = list(ql_date.options)
    if not opts or ql_date.value not in opts:
        return
    idx = opts.index(ql_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        ql_date.value = opts[new_idx]
    else:
        # Force a refresh even if the value is unchanged (e.g., only one option)
        ql_date.param.trigger("value")


ql_prev.on_click(lambda _e: _shift_ql(-1))
ql_next.on_click(lambda _e: _shift_ql(1))


def _shift_hk(delta: int):
    """Move housekeeping selection by delta steps in the refreshed options list."""
    _refresh_hk_options(preserve_current=True)
    opts = list(hk_date.options)
    if not opts or hk_date.value not in opts:
        return
    idx = opts.index(hk_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        hk_date.value = opts[new_idx]
    else:
        hk_date.param.trigger("value")


hk_prev.on_click(lambda _e: _shift_hk(-1))
hk_next.on_click(lambda _e: _shift_hk(1))

# Periodically refresh the "Today (latest)" selection to pick up new PNGs.
def _refresh_latest_if_needed():
    """If viewing the latest science quicklook, reload the mapping and redraw."""
    if _is_wxcam_instrument(science_instrument.value):
        return
    if ql_date.value == "Today (latest)":
        global _ql_options
        _ql_options = _quicklook_options(science_instrument.value, science_image_type.value, mode="science")
        ql_date.param.trigger("value")


_ql_timer = _safe_periodic_callback(_refresh_latest_if_needed, period=300_000, start=True)


def _refresh_hk_latest_if_needed():
    """If viewing the latest housekeeping quicklook, reload the mapping and redraw."""
    if hk_date.value == "Today (latest)":
        global _hk_options
        _hk_options = _quicklook_options(hk_instrument.value, mode="housekeeping")
        hk_date.param.trigger("value")


_hk_timer = _safe_periodic_callback(_refresh_hk_latest_if_needed, period=300_000, start=True)

# Ensure initial map is fresh
_refresh_ql_options(preserve_current=True)
_refresh_hk_options(preserve_current=True)
_apply_instrument_defaults(CURRENT_INSTRUMENT, reset_time=True)


def _safe_widget_value(widget_name: str):
    widget = globals().get(widget_name)
    return getattr(widget, "value", None) if widget is not None else None


def _selection_snapshot() -> dict[str, object]:
    return {
        "current_instrument": _safe_widget_value("instrument_select"),
        "science_instrument": _safe_widget_value("science_instrument"),
        "science_date": _safe_widget_value("ql_date"),
        "science_image_type": _safe_widget_value("science_image_type"),
        "hk_instrument": _safe_widget_value("hk_instrument"),
        "hk_date": _safe_widget_value("hk_date"),
        "wxcam_image_type": _safe_widget_value("wxcam_image_type"),
        "wxcam_date": _safe_widget_value("wxcam_date"),
        "wxcam_selected_hour_path": getattr(wxcam_calendar_state, "selected_hour_path", ""),
        "live_mode": _safe_widget_value("live_toggle"),
    }


def _selection_snapshot_safe() -> dict[str, object]:
    try:
        snapshot_fn = globals().get("_selection_snapshot")
        if callable(snapshot_fn):
            return snapshot_fn()
    except Exception:
        pass
    return {}


def _log_control_change(control: str, event, context: str, instrument: str | None = None) -> None:
    old = getattr(event, "old", None)
    new = getattr(event, "new", None)
    if old == new:
        return
    fields = _selection_snapshot()
    fields.update(
        {
            "control": control,
            "context": context,
            "old": old,
            "new": new,
            "request_path": _request_path(),
        }
    )
    _perf_log("ui_selection_change", instrument=instrument or CURRENT_INSTRUMENT, **fields)


def _log_session_loaded() -> None:
    fields = _selection_snapshot_safe()
    fields.update(
        {
            "request_path": _request_path(),
            "client_ip": _client_ip(),
            "user_agent": _request_header("User-Agent"),
            "status": "loaded",
        }
    )
    _perf_log("session_loaded", instrument=_safe_widget_value("instrument_select") or CURRENT_INSTRUMENT, **fields)


def _log_session_heartbeat() -> None:
    fields = _selection_snapshot_safe()
    fields.update(
        {
            "request_path": _request_path(),
            "status": "alive",
        }
    )
    _perf_log("session_heartbeat", instrument=_safe_widget_value("instrument_select") or CURRENT_INSTRUMENT, **fields)


def _log_session_destroyed(session_context) -> None:
    perf_logger = globals().get("_perf_log")
    if not callable(perf_logger):
        return
    request = getattr(session_context, "request", None)
    path = str(getattr(request, "path", "")) or None
    server_context = getattr(session_context, "server_context", None)
    server_sessions = None
    if server_context is not None:
        try:
            server_sessions = int(len(server_context.sessions))
        except Exception:
            server_sessions = None
    fields = {}
    try:
        instrument_widget = globals().get("instrument_select")
        science_widget = globals().get("science_instrument")
        ql_widget = globals().get("ql_date")
        science_type_widget = globals().get("science_image_type")
        hk_widget = globals().get("hk_instrument")
        hk_date_widget = globals().get("hk_date")
        wxcam_type_widget = globals().get("wxcam_image_type")
        wxcam_date_widget = globals().get("wxcam_date")
        wxcam_state = globals().get("wxcam_calendar_state")
        live_widget = globals().get("live_toggle")
        fields.update(
            {
                "current_instrument": getattr(instrument_widget, "value", None),
                "science_instrument": getattr(science_widget, "value", None),
                "science_date": getattr(ql_widget, "value", None),
                "science_image_type": getattr(science_type_widget, "value", None),
                "hk_instrument": getattr(hk_widget, "value", None),
                "hk_date": getattr(hk_date_widget, "value", None),
                "wxcam_image_type": getattr(wxcam_type_widget, "value", None),
                "wxcam_date": getattr(wxcam_date_widget, "value", None),
                "wxcam_selected_hour_path": getattr(wxcam_state, "selected_hour_path", ""),
                "live_mode": getattr(live_widget, "value", None),
            }
        )
    except Exception:
        fields = {}
    fields.update(
        {
            "request_path": path,
            "status": "destroyed",
            "session_id": getattr(session_context, "id", None),
            "server_sessions": server_sessions,
        }
    )
    current_instrument = fields.get("current_instrument") or globals().get("CURRENT_INSTRUMENT")
    perf_logger("session_destroyed", instrument=current_instrument, **fields)


instrument_select.param.watch(
    lambda event: _log_control_change("instrument_select", event, context="interactive", instrument=event.new),
    "value",
)
science_instrument.param.watch(
    lambda event: _log_control_change("science_instrument", event, context="science_quicklooks", instrument=event.new),
    "value",
)
ql_date.param.watch(
    lambda event: _log_control_change("ql_date", event, context="science_quicklooks", instrument=science_instrument.value),
    "value",
)
science_image_type.param.watch(
    lambda event: _log_control_change("science_image_type", event, context="science_quicklooks", instrument="wxcam"),
    "value",
)
hk_instrument.param.watch(
    lambda event: _log_control_change("hk_instrument", event, context="housekeeping_quicklooks", instrument=event.new),
    "value",
)
hk_date.param.watch(
    lambda event: _log_control_change("hk_date", event, context="housekeeping_quicklooks", instrument=hk_instrument.value),
    "value",
)
wxcam_image_type.param.watch(
    lambda event: _log_control_change("wxcam_image_type", event, context="wxcam_interactive", instrument="wxcam"),
    "value",
)
wxcam_date.param.watch(
    lambda event: _log_control_change("wxcam_date", event, context="wxcam_interactive", instrument="wxcam"),
    "value",
)
live_toggle.param.watch(
    lambda event: _log_control_change("live_toggle", event, context="interactive", instrument=instrument_select.value),
    "value",
)
wxcam_calendar_state.param.watch(
    lambda event: _log_control_change("wxcam_selected_hour_path", event, context="wxcam_calendar", instrument="wxcam"),
    "selected_hour_path",
)
pn.state.onload(_log_session_loaded)
pn.state.on_session_destroyed(_log_session_destroyed)
_session_heartbeat_cb = _safe_periodic_callback(_log_session_heartbeat, period=SESSION_HEARTBEAT_MS, start=True)


def _sync_wxcam_calendar_hour(*_events):
    with _timed_perf(
        "wxcam_calendar_sync",
        instrument="wxcam",
        selected_day=ql_date.value,
        selection=science_image_type.value,
    ) as perf:
        if not _is_wxcam_instrument(science_instrument.value):
            perf["status"] = "non_wxcam"
            wxcam_calendar_state.selected_hour_path = ""
            return
        selected_day = ql_date.value
        selection = science_image_type.value or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(selected_day)
        day_utc = _wxcam_day_token_to_utc(day_token or "")
        if not day_utc:
            perf["status"] = "invalid_day"
            wxcam_calendar_state.selected_hour_path = ""
            return
        image_type = _image_type_from_selection(selection)
        rows_by_hour = representative_hourly_records(
            _wxcam_catalog_path("wxcam"),
            image_type,
            day_utc,
            media_kind="image",
        )
        available_paths = [str(row["raw_path"]) for row in rows_by_hour.values()]
        perf["hour_count"] = len(rows_by_hour)
        if not available_paths:
            perf["status"] = "empty"
            wxcam_calendar_state.selected_hour_path = ""
            return
        if wxcam_calendar_state.selected_hour_path not in available_paths:
            wxcam_calendar_state.selected_hour_path = ""
        perf["status"] = "ok"


science_instrument.param.watch(_sync_wxcam_calendar_hour, "value")
science_image_type.param.watch(_sync_wxcam_calendar_hour, "value")
ql_date.param.watch(_sync_wxcam_calendar_hour, "value")


def _build_wxcam_hour_tile(
    image_type: str,
    day_token: str,
    hour_index: int,
    row,
    selected_hour_path: str,
):
    hour_label = f"{hour_index:02d}:00"
    tile_classes = ["wxcam-hour-tile"]
    if image_type == "fish_hdr":
        tile_classes.append("wxcam-hour-tile--fish")
    if row is None:
        preview = pn.pane.HTML(
            f"<div class='wxcam-hour-tile__placeholder'>{hour_label}</div>",
            sizing_mode="stretch_width",
            margin=0,
        )
        button = pn.widgets.Button(name=f"{hour_label}", disabled=True, sizing_mode="stretch_width", margin=0)
        return pn.Column(
            preview,
            button,
            css_classes=tile_classes,
            sizing_mode="stretch_width",
            margin=0,
        )

    thumb_path = _wxcam_hourly_thumbnail_path(image_type, day_token, str(row["filename"]))
    if thumb_path.exists():
        preview = pn.pane.HTML(
            f"<img class='wxcam-hour-tile__img' src='{_image_data_uri(thumb_path)}' alt='{hour_label}'>",
            sizing_mode="stretch_width",
            margin=0,
        )
    else:
        preview = pn.pane.HTML(
            f"<div class='wxcam-hour-tile__placeholder'>{hour_label}</div>",
            sizing_mode="stretch_width",
            margin=0,
        )
    button = pn.widgets.Button(
        name=hour_label,
        button_type="primary" if str(row["raw_path"]) == selected_hour_path else "default",
        sizing_mode="stretch_width",
        margin=0,
    )
    button.on_click(lambda _event, path=str(row["raw_path"]): setattr(wxcam_calendar_state, "selected_hour_path", path))
    return pn.Column(
        preview,
        button,
        css_classes=tile_classes,
        sizing_mode="stretch_width",
        margin=0,
    )


def _build_wxcam_calendar_day_view(selection: str, day_token: str, selected_hour_path: str):
    with _timed_perf(
        "wxcam_calendar_day_view",
        instrument="wxcam",
        selection=selection,
        day_token=day_token,
        selected_hour_path=selected_hour_path,
    ) as perf:
        day_utc = _wxcam_day_token_to_utc(day_token)
        if not day_utc:
            perf["status"] = "invalid_day"
            return pn.pane.Markdown("No hourly images available for this selection.")
        image_type = _image_type_from_selection(selection)
        rows_by_hour = representative_hourly_records(
            _wxcam_catalog_path("wxcam"),
            image_type,
            day_utc,
            media_kind="image",
        )
        perf["hour_count"] = len(rows_by_hour)
        if not rows_by_hour:
            perf["status"] = "empty"
            return pn.pane.Markdown("No hourly images available for this selection.")
        tiles = [
            _build_wxcam_hour_tile(image_type, day_token, hour_index, rows_by_hour.get(hour_index), selected_hour_path)
            for hour_index in range(24)
        ]
        grid = pn.GridBox(*tiles, ncols=8, sizing_mode="stretch_width")
        selected_row = next((row for row in rows_by_hour.values() if str(row["raw_path"]) == selected_hour_path), None)
        if selected_row is None:
            perf["status"] = "grid_only"
            return pn.Column(grid, sizing_mode="stretch_width")
        selected_hour_label = str(selected_row["time_utc"])[11:16] + " UTC"
        viewer = _build_wxcam_image_view(Path(str(selected_row["raw_path"])), selection, f"{day_token} | {selected_hour_label}")
        perf["status"] = "with_viewer"
        return pn.Column(grid, viewer, sizing_mode="stretch_width")


@pn.depends(
    ql_date.param.value,
    science_instrument.param.value,
    science_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
def _science_quicklook_image(selected, science_inst, wxcam_selection, selected_hour_path):
    """Show the selected science quicklook asset (or a message if missing)."""
    instrument = science_inst or CURRENT_INSTRUMENT
    with _timed_perf("science_quicklook_render", instrument=instrument, selected=selected) as perf:
        if _is_wxcam_instrument(instrument):
            selection = wxcam_selection or _cfg("wxcam")["default_top"]
            perf["selection"] = selection
            day_token = _wxcam_calendar_day_token(selected)
            return _build_wxcam_calendar_day_view(selection, day_token or "", selected_hour_path)
        if _is_stacked_timeseries_instrument(instrument):
            token = _quicklook_options(instrument, mode="science").get(selected)
            quick_dir = _cfg(instrument)["quicklook_dir"]
            if token is None:
                perf["status"] = "missing_file"
                return pn.pane.Markdown("No image available for this selection.")
            path = summary_latest_png(quick_dir, instrument) if token == "latest" else summary_daily_png(quick_dir, instrument, token)
            perf["path"] = path
            if not path.exists():
                perf["status"] = "missing_file"
                return pn.pane.Markdown("No image available for this selection.")
            perf["status"] = "ok"
            return pn.pane.PNG(path, sizing_mode="stretch_width")
        path = _quicklook_options(instrument).get(selected)
        perf["path"] = path
        if path and Path(path).exists():
            perf["status"] = "ok"
            return _media_pane(path)
        perf["status"] = "missing_file"
        return pn.pane.Markdown("No image available for this selection.")


@pn.depends(hk_date.param.value, hk_instrument.param.value)
def _housekeeping_quicklook_image(selected, hk_inst):
    """Show the selected housekeeping quicklook asset (or a message if missing)."""
    instrument = hk_inst or CURRENT_INSTRUMENT
    with _timed_perf("housekeeping_quicklook_render", instrument=instrument, selected=selected) as perf:
        quick_dir = _cfg(instrument)["quicklook_dir"]
        hk_label = _housekeeping_quicklook_label(instrument)
        if hk_label is None:
            perf["status"] = "unsupported"
            return pn.pane.Markdown("No housekeeping quicklooks available for this instrument.")
        token = _quicklook_options(instrument, mode="housekeeping").get(selected)
        if token is None:
            perf["status"] = "missing_file"
            return pn.pane.Markdown("No housekeeping quicklooks available for this instrument.")
        path = _housekeeping_latest_path(instrument, quick_dir) if token == "latest" else _housekeeping_daily_path(instrument, quick_dir, token)
        perf["path"] = path
        if path and path.exists():
            perf["status"] = "ok"
            return pn.pane.PNG(path, sizing_mode="stretch_width")
        perf["status"] = "missing_file"
        return pn.pane.Markdown("No housekeeping quicklooks available for this instrument.")


def _current_interactive_status_markup() -> str:
    inst = instrument_select.value
    if _is_wxcam_instrument(inst):
        selection = wxcam_image_type.value or _cfg("wxcam")["default_top"]
        selected = wxcam_date.value
        day_token = _wxcam_calendar_day_token(selected)
        bits = _wxcam_hour_bits(selection, day_token or "")
        missing = sum(1 for bit in bits if not bit)
        latest_row = latest_record(_wxcam_catalog_path("wxcam"), _image_type_from_selection(selection), media_kind="image")
        latest_dt = pd.Timestamp(str(latest_row["time_utc"])).to_pydatetime(warn=False) if latest_row else None
        lag = datetime.now() - latest_dt if latest_dt is not None else None
        items = [
            ("Last sample", _format_status_time(latest_dt), "info"),
            ("Hourly gaps", str(missing), "warn" if missing else "ok"),
            ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
        ]
        return _status_strip_markup(items)
    latest = _dataset_time_bounds(inst)[1]
    lag = datetime.now() - latest if latest is not None else None
    times = _instrument_time_index(inst)
    bits, missing, total = _hourly_coverage_summary(times, _ensure_utc(range_start.value), _ensure_utc(range_end.value))
    items = [
        ("Last sample", _format_status_time(latest), "info"),
        ("Hourly gaps", str(missing), "warn" if missing else "ok"),
        ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
    ]
    if total:
        items.insert(2, ("Coverage", f"{total - missing}/{total} h", "info"))
    return _status_strip_markup(items)


def _current_interactive_availability_markup() -> str:
    inst = instrument_select.value
    if _is_wxcam_instrument(inst):
        selection = wxcam_image_type.value or _cfg("wxcam")["default_top"]
        selected = wxcam_date.value
        day_token = _wxcam_calendar_day_token(selected)
        bits = _wxcam_hour_bits(selection, day_token or "")
        start_label = "00:00"
        end_label = "23:00"
        if day_token == datetime.now(timezone.utc).strftime("%Y%m%d"):
            end_label = datetime.now(timezone.utc).strftime("%H:00")
        return _availability_bar_markup(
            bits,
            start_label,
            end_label,
            "HDR image availability by hour",
            "Each block represents one UTC hour. Teal means at least one HDR image exists for that hour.",
            full_label="HDR image present",
            empty_label="No HDR image",
            segment_titles=[f"{hour:02d}:00 UTC" for hour in range(24)],
        )
    start = _ensure_utc(range_start.value)
    end = _ensure_utc(range_end.value)
    bits = _binned_time_coverage(_instrument_time_index(inst), start, end, segments=72)
    start_label = start.strftime("%m-%d %H:%M") if start else "--"
    end_label = end.strftime("%m-%d %H:%M") if end else "--"
    return _availability_bar_markup(
        bits,
        start_label,
        end_label,
        "Data availability across the selected time window",
        _availability_bucket_text(start, end, len(bits)) + " Teal means at least one sample exists in that slice.",
        full_label="Samples present",
        empty_label="No samples",
        segment_titles=_availability_binned_titles(start, end, len(bits)),
    )


def _current_science_status_markup() -> str:
    inst = science_instrument.value
    if _is_wxcam_instrument(inst):
        selection = science_image_type.value or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(ql_date.value)
        bits = _wxcam_hour_bits(selection, day_token or "")
        missing = sum(1 for bit in bits if not bit)
        latest_row = latest_record(_wxcam_catalog_path("wxcam"), _image_type_from_selection(selection), media_kind="image")
        latest_dt = pd.Timestamp(str(latest_row["time_utc"])).to_pydatetime(warn=False) if latest_row else None
        lag = datetime.now() - latest_dt if latest_dt is not None else None
        items = [
            ("Last sample", _format_status_time(latest_dt), "info"),
            ("Hourly gaps", str(missing), "warn" if missing else "ok"),
            ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
        ]
        return _status_strip_markup(items)
    start, end, _day_token = _selected_token_window(ql_date.value)
    times = _instrument_time_index(inst)
    if start is not None and end is not None:
        mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        window_times = times[mask]
    else:
        window_times = times
    latest_dt = window_times.max().to_pydatetime(warn=False) if len(window_times) else _dataset_time_bounds(inst)[1]
    bits, missing, total = _hourly_coverage_summary(window_times, start, end)
    lag = datetime.now() - latest_dt if latest_dt is not None and start and start.date() == datetime.utcnow().date() else None
    items = [("Last sample", _format_status_time(latest_dt), "info")]
    if total:
        items.append(("Hourly gaps", str(missing), "warn" if missing else "ok"))
        items.append(("Coverage", f"{total - missing}/{total} h", "info"))
    if lag is not None:
        items.append(("Lag", _format_duration(lag), "warn" if lag > timedelta(hours=1) else "ok"))
    return _status_strip_markup(items)


def _current_science_availability_markup() -> str:
    inst = science_instrument.value
    if _is_wxcam_instrument(inst):
        selection = science_image_type.value or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(ql_date.value)
        bits = _wxcam_hour_bits(selection, day_token or "")
        end_label = datetime.now(timezone.utc).strftime("%H:00") if day_token == datetime.now(timezone.utc).strftime("%Y%m%d") else "23:00"
        return _availability_bar_markup(
            bits,
            "00:00",
            end_label,
            "HDR image availability by hour",
            "Each block represents one UTC hour. Teal means the science quicklook has an HDR image source for that hour.",
            full_label="Representative HDR image present",
            empty_label="No representative HDR image",
            segment_titles=[f"{hour:02d}:00 UTC" for hour in range(24)],
        )
    start, end, _day_token = _selected_token_window(ql_date.value)
    times = _instrument_time_index(inst)
    if start is None or end is None:
        return _availability_bar_markup([], "--", "--", "Data availability by hour", "Each block would represent one UTC hour in the selected day.")
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    bits, _missing, _total = _hourly_coverage_summary(times[mask], start, end)
    end_label = end.strftime("%H:%M")
    return _availability_bar_markup(
        bits,
        "00:00",
        end_label,
        "Data availability by hour",
        "Each block represents one UTC hour in the selected day. Teal means at least one sample exists in that hour.",
        full_label="Samples present",
        empty_label="No samples",
        segment_titles=_availability_hour_titles(start, end),
    )


def _current_hk_status_markup() -> str:
    inst = hk_instrument.value
    start, end, day_token = _selected_token_window(hk_date.value)
    if inst == "wxcam":
        states = _wxcam_combined_hour_states(day_token or "")
        missing = sum(1 for state in states if state != 2)
        latest_dt = _dataset_time_bounds("wxcam")[1]
        lag = datetime.now() - latest_dt if latest_dt is not None else None
        items = [
            ("Last sample", _format_status_time(latest_dt), "info"),
            ("Partial/Missing hours", str(missing), "warn" if missing else "ok"),
            ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
        ]
        return _status_strip_markup(items)
    times = _instrument_time_index(inst)
    if start is not None and end is not None:
        mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        window_times = times[mask]
    else:
        window_times = times
    latest_dt = window_times.max().to_pydatetime(warn=False) if len(window_times) else _dataset_time_bounds(inst)[1]
    bits, missing, total = _hourly_coverage_summary(window_times, start, end)
    lag = datetime.now() - latest_dt if latest_dt is not None and start and start.date() == datetime.utcnow().date() else None
    items = [("Last sample", _format_status_time(latest_dt), "info")]
    if total:
        items.append(("Hourly gaps", str(missing), "warn" if missing else "ok"))
        items.append(("Coverage", f"{total - missing}/{total} h", "info"))
    if lag is not None:
        items.append(("Lag", _format_duration(lag), "warn" if lag > timedelta(hours=1) else "ok"))
    return _status_strip_markup(items)


def _current_hk_availability_markup() -> str:
    inst = hk_instrument.value
    start, end, day_token = _selected_token_window(hk_date.value)
    if inst == "wxcam":
        states = _wxcam_combined_hour_states(day_token or "")
        end_label = datetime.now(timezone.utc).strftime("%H:00") if day_token == datetime.now(timezone.utc).strftime("%Y%m%d") else "23:00"
        return _availability_bar_markup(
            states,
            "00:00",
            end_label,
            "WXcam HDR availability by hour",
            (
                "Each block represents one UTC hour. Teal means the retained "
                "WXcam HDR streams have an image for that hour; gold means "
                "only some retained streams are present; gray means no "
                "retained HDR image is available."
            ),
            full_label="All retained HDR streams present",
            partial_label="Partial retained HDR coverage",
            empty_label="No retained HDR image",
            segment_titles=[f"{hour:02d}:00 UTC" for hour in range(24)],
        )
    if start is None or end is None:
        return _availability_bar_markup(
            [],
            "--",
            "--",
            "Housekeeping data availability by hour",
            "Each block would represent one UTC hour in the selected day.",
        )
    times = _instrument_time_index(inst)
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    bits, _missing, _total = _hourly_coverage_summary(times[mask], start, end)
    return _availability_bar_markup(
        bits,
        "00:00",
        end.strftime("%H:%M"),
        "Housekeeping data availability by hour",
        "Each block represents one UTC hour in the selected day. Teal means at least one housekeeping sample exists in that hour.",
        full_label="Samples present",
        empty_label="No samples",
        segment_titles=_availability_hour_titles(start, end),
    )


interactive_status = pn.bind(
    lambda *_deps: _current_interactive_status_markup(),
    instrument_select.param.value,
    range_start.param.value,
    range_end.param.value,
    live_toggle.param.value,
    wxcam_image_type.param.value,
    wxcam_date.param.value,
    var1_select.param.value,
    var2_select.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
)
interactive_availability = pn.bind(
    lambda *_deps: _current_interactive_availability_markup(),
    instrument_select.param.value,
    range_start.param.value,
    range_end.param.value,
    live_toggle.param.value,
    wxcam_image_type.param.value,
    wxcam_date.param.value,
    var1_select.param.value,
    var2_select.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
)
science_status = pn.bind(
    lambda *_deps: _current_science_status_markup(),
    science_instrument.param.value,
    ql_date.param.value,
    science_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
science_availability = pn.bind(
    lambda *_deps: _current_science_availability_markup(),
    science_instrument.param.value,
    ql_date.param.value,
    science_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
hk_status = pn.bind(
    lambda *_deps: _current_hk_status_markup(),
    hk_instrument.param.value,
    hk_date.param.value,
)
hk_availability = pn.bind(
    lambda *_deps: _current_hk_availability_markup(),
    hk_instrument.param.value,
    hk_date.param.value,
)


interactive_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
science_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
hk_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
interactive_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
science_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
hk_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
interactive_download = pn.widgets.Button(name="Download PNG", button_type="default", width=130)
science_download = pn.widgets.FileDownload(name="", label="Download PNG", button_type="default", auto=False, embed=False, width=130)
hk_download = pn.widgets.FileDownload(name="", label="Download PNG", button_type="default", auto=False, embed=False, width=130)


for button, widget in (
    (interactive_copy, interactive_share_url),
    (science_copy, science_share_url),
    (hk_copy, hk_share_url),
):
    button.js_on_click(
        args={"share": widget},
        code="""
        const text = share.value || '';
        if (!text) { return; }
        navigator.clipboard.writeText(text);
        """,
    )


interactive_download.js_on_click(
    args={"inst": instrument_select},
    code="""
    const plot = Array.from(document.querySelectorAll('.js-plotly-plot')).find((el) => el.offsetParent !== null);
    if (!plot || !window.Plotly) { return; }
    const base = (inst.value || 'aurora').toLowerCase().replace(/[^a-z0-9]+/g, '_');
    window.Plotly.downloadImage(plot, {format: 'png', filename: `${base}_interactive`});
    """,
)


def _science_download_path() -> Path | None:
    inst = science_instrument.value
    if _is_wxcam_instrument(inst):
        if wxcam_calendar_state.selected_hour_path:
            path = Path(wxcam_calendar_state.selected_hour_path)
            return path if path.exists() else None
        return None
    if _is_stacked_timeseries_instrument(inst):
        token = _quicklook_options(inst, mode="science").get(ql_date.value)
        if token is None:
            return None
        quick_dir = _cfg(inst)["quicklook_dir"]
        return summary_latest_png(quick_dir, inst) if token == "latest" else summary_daily_png(quick_dir, inst, token)
    raw_path = _quicklook_options(inst, mode="science").get(ql_date.value)
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _hk_download_path() -> Path | None:
    inst = hk_instrument.value
    token = _quicklook_options(inst, mode="housekeeping").get(hk_date.value)
    if token is None:
        return None
    quick_dir = _cfg(inst)["quicklook_dir"]
    path = _housekeeping_latest_path(inst, quick_dir) if token == "latest" else _housekeeping_daily_path(inst, quick_dir, token)
    return path if path and path.exists() else None


def _view_query_params(tab_slug: str) -> dict[str, str]:
    params: dict[str, str] = {
        "tab": tab_slug,
        "instrument": instrument_select.value,
    }
    if tab_slug == "interactive":
        params["start"] = range_start.value.isoformat() if range_start.value else ""
        params["end"] = range_end.value.isoformat() if range_end.value else ""
        params["live"] = "1" if live_toggle.value else "0"
        if _is_wxcam_instrument(instrument_select.value):
            params["wxcam_image_type"] = wxcam_image_type.value or ""
            params["wxcam_date"] = wxcam_date.value or ""
        else:
            params["top_var"] = var1_select.value or ""
            params["bottom_var"] = var2_select.value or ""
            params["bottom_m"] = str(bottom_range_m.value)
            params["top_m"] = str(top_range_m.value)
    elif tab_slug == "science":
        params["science_instrument"] = science_instrument.value
        params["science_date"] = ql_date.value or ""
        params["science_image_type"] = science_image_type.value or ""
        if wxcam_calendar_state.selected_hour_path:
            params["science_selected_hour"] = wxcam_calendar_state.selected_hour_path
    elif tab_slug == "housekeeping":
        params["hk_instrument"] = hk_instrument.value
        params["hk_date"] = hk_date.value or ""
    return {k: v for k, v in params.items() if v not in ("", None)}


def _build_share_url(tab_slug: str) -> str:
    query = urlencode(_view_query_params(tab_slug))
    return f"{_request_base_url()}?{query}" if query else _request_base_url()


def _active_tab_slug() -> str:
    if "tabs" not in globals():
        return "interactive"
    return {0: "interactive", 1: "science", 2: "housekeeping", 3: "operations"}.get(getattr(tabs, "active", 0), "interactive")


def _update_browser_location() -> None:
    """Keep the address bar aligned with the current view for mobile reconnects."""
    try:
        location = pn.state.location
    except Exception:
        location = None
    if location is None:
        return
    search = "?" + urlencode(_view_query_params(_active_tab_slug()))
    if getattr(location, "search", None) != search:
        location.search = search


def _refresh_share_and_download_state(*_events) -> None:
    interactive_share_url.value = _build_share_url("interactive")
    science_share_url.value = _build_share_url("science")
    hk_share_url.value = _build_share_url("housekeeping")
    _update_browser_location()

    interactive_download.visible = not _is_wxcam_instrument(instrument_select.value)

    science_path = _science_download_path()
    science_download.file = science_path
    science_download.filename = science_path.name if science_path else "science_quicklook.png"
    science_download.label = "Download PNG"
    science_download.disabled = science_path is None

    hk_path = _hk_download_path()
    hk_download.file = hk_path
    hk_download.filename = hk_path.name if hk_path else "housekeeping_quicklook.png"
    hk_download.label = "Download PNG"
    hk_download.disabled = hk_path is None


def _apply_query_state() -> None:
    args = _request_query_args()
    if not args:
        return
    visible_instruments = set(INSTRUMENT_OPTIONS.values())
    hk_visible_instruments = set(HK_INSTRUMENT_OPTIONS.values())
    instrument = args.get("instrument")
    if instrument in visible_instruments:
        instrument_select.value = instrument
    start_raw = args.get("start")
    end_raw = args.get("end")
    if start_raw and end_raw and not _is_wxcam_instrument(instrument_select.value):
        try:
            range_start.value = pd.Timestamp(start_raw).to_pydatetime(warn=False)
            range_end.value = pd.Timestamp(end_raw).to_pydatetime(warn=False)
            _set_live(args.get("live", "0") == "1")
        except Exception:
            pass
    if args.get("bottom_m"):
        try:
            bottom_range_m.value = int(args["bottom_m"])
        except Exception:
            pass
    if args.get("top_m"):
        try:
            top_range_m.value = int(args["top_m"])
        except Exception:
            pass
    if args.get("top_var") in _cfg(instrument_select.value)["vars"]:
        var1_select.value = args["top_var"]
    if args.get("bottom_var") in _cfg(instrument_select.value)["vars"]:
        var2_select.value = args["bottom_var"]
    if args.get("wxcam_image_type") in list(wxcam_image_type.options):
        wxcam_image_type.value = args["wxcam_image_type"]
    if args.get("wxcam_date") in list(wxcam_date.options):
        wxcam_date.value = args["wxcam_date"]
    if args.get("science_instrument") in visible_instruments:
        science_instrument.value = args["science_instrument"]
    if args.get("science_image_type") in list(science_image_type.options):
        science_image_type.value = args["science_image_type"]
    if args.get("science_date") in list(ql_date.options):
        ql_date.value = args["science_date"]
    if args.get("science_selected_hour"):
        wxcam_calendar_state.selected_hour_path = args["science_selected_hour"]
    if args.get("hk_instrument") in hk_visible_instruments:
        hk_instrument.value = args["hk_instrument"]
    if args.get("hk_date") in list(hk_date.options):
        hk_date.value = args["hk_date"]


for widget, parameter in (
    (instrument_select, "value"),
    (range_start, "value"),
    (range_end, "value"),
    (live_toggle, "value"),
    (var1_select, "value"),
    (var2_select, "value"),
    (bottom_range_m, "value"),
    (top_range_m, "value"),
    (wxcam_image_type, "value"),
    (wxcam_date, "value"),
    (science_instrument, "value"),
    (ql_date, "value"),
    (science_image_type, "value"),
    (hk_instrument, "value"),
    (hk_date, "value"),
):
    widget.param.watch(_refresh_share_and_download_state, parameter)

wxcam_calendar_state.param.watch(_refresh_share_and_download_state, "selected_hour_path")


ACCENT = THEME_ACCENT  # header/accent color
css = """
# Global font override for a clean, consistent look.
body, .bk {
    font-family: "SF Pro Display","SF Pro","-apple-system","BlinkMacSystemFont","Segoe UI",sans-serif;
    font-size: 15px;
    background: #ffffff;
    color: #22313f;
}
.bk.card, .bk-panel-models-card {
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    box-shadow: none;
    background: #ffffff;
}
.bk-btn, button.bk-btn {
    border-radius: 6px;
    border: 1px solid #c5d0da;
    box-shadow: none;
    background: #ffffff;
    color: #22313f;
}
.bk-btn-primary, button.bk-btn-primary {
    background: #0b7285;
    border-color: #0b7285;
    color: #ffffff;
}
.bk-input {
    border-radius: 6px;
    border-color: #c5d0da;
    background: #ffffff;
    color: #22313f;
}
.mobile-stack {
    flex-wrap: wrap !important;
    gap: 8px;
}
.mobile-stack > .bk {
    flex: 1 1 220px;
    min-width: 160px;
}
.small-card .bk-card-header {
    padding: 4px 8px;
}
.small-card .bk-card-body {
    padding: 6px 8px;
}
.status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid #d8e1e8;
    background: #fbfcfd;
    color: #334155;
    font-size: 12px;
    line-height: 1.2;
}
.status-pill--ok {
    border-color: #b7e4dc;
    background: #f1fbf8;
    color: #0b6b5d;
}
.status-pill--warn {
    border-color: #f1d4b5;
    background: #fff8ef;
    color: #9a5b16;
}
.status-pill--info {
    border-color: #d8dee4;
    background: #f8fafb;
    color: #334155;
}
.interactive-loading-notice {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #fbfcfd;
}
.interactive-loading-notice__badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 78px;
    padding: 3px 8px;
    border-radius: 999px;
    background: #edf6f8;
    color: #0b7285;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0;
}
.interactive-loading-notice__text {
    font-size: 12px;
    color: #536171;
    line-height: 1.35;
}
.interactive-loading-note,
.lazy-tab-placeholder {
    padding: 10px 12px;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #fbfcfd;
    color: #536171;
    font-size: 12px;
    line-height: 1.35;
}
.interactive-skeleton {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 100%;
    padding: 16px 18px;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #ffffff;
}
.interactive-skeleton__title {
    font-size: 15px;
    font-weight: 600;
    color: #22313f;
}
.interactive-skeleton__subtitle {
    font-size: 12px;
    color: #647283;
}
.interactive-skeleton__plot {
    height: 260px;
    border-radius: 8px;
    border: 1px solid #e5eaef;
    background:
        linear-gradient(90deg, rgba(248, 250, 252, 0.95), rgba(237, 242, 247, 0.75), rgba(248, 250, 252, 0.95));
    background-size: 220% 100%;
    animation: interactive-skeleton-shimmer 1.6s ease-in-out infinite;
}
.interactive-skeleton__plot--secondary {
    height: 160px;
}
@keyframes interactive-skeleton-shimmer {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}
.availability-shell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
}
.availability-caption {
    font-size: 12px;
    color: #4c5c6b;
}
.availability-explainer {
    font-size: 11px;
    color: #647283;
    line-height: 1.35;
}
.availability-bar {
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(4px, 1fr);
    gap: 2px;
    align-items: center;
}
.availability-segment {
    height: 8px;
    border-radius: 3px;
    border: 1px solid #d8e1e8;
    background: #ffffff;
}
.availability-segment--full {
    background: #0b7285;
    border-color: #0b7285;
}
.availability-segment--partial {
    background: #e0b15c;
    border-color: #e0b15c;
}
.availability-segment--empty {
    background: #eef2f5;
    border-color: #d8e1e8;
}
.availability-scale {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #6b7280;
}
.availability-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    font-size: 11px;
    color: #566370;
}
.availability-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.availability-legend-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 3px;
    border: 1px solid #d8dee4;
}
.availability-empty {
    font-size: 12px;
    color: #6b7280;
}
.action-row {
    align-items: flex-end;
    gap: 10px;
}
.action-row > .bk {
    margin: 0 !important;
}
.action-row .bk-input-group {
    min-width: 320px;
}
.action-row .bk-btn,
.action-row button.bk-btn {
    min-height: 38px;
}
.action-row .bk-panel-models-widgets-FileDownload,
.action-row .bk-panel-models-widgets-Button {
    flex: 0 0 auto;
}
.site-footer {
    margin-top: 8px;
    padding: 12px 14px;
    border-top: 1px solid #d8dee4;
    background: #ffffff;
}
.site-footer__title {
    font-size: 12px;
    font-weight: 600;
    color: #243b53;
    margin-bottom: 8px;
}
.site-footer__links {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}
.site-footer__link {
    min-width: 240px;
    flex: 1 1 280px;
}
.site-footer__link a {
    color: #0b7285;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
}
.site-footer__link a:hover {
    text-decoration: underline;
}
.site-footer__desc {
    margin-top: 4px;
    font-size: 12px;
    color: #5b6673;
    line-height: 1.4;
}
.ops-shell {
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.ops-headline {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
}
.ops-headline__main {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.ops-headline__text,
.ops-footnote {
    font-size: 12px;
    color: #5b6673;
    line-height: 1.45;
}
.ops-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.ops-section-title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #243b53;
}
.ops-section-title--headline {
    font-size: 16px;
}
.ops-grid {
    display: grid;
    gap: 12px;
}
.ops-grid--summary {
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.ops-grid--storage {
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
}
.ops-card {
    border: 1px solid #d8dee4;
    border-radius: 8px;
    background: #ffffff;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.ops-card__head {
    display: flex;
    align-items: center;
    gap: 8px;
}
.ops-card__value {
    font-size: 14px;
    font-weight: 600;
    color: #1f2933;
}
.ops-card__meta {
    font-size: 11px;
    color: #6b7280;
    line-height: 1.4;
}
.ops-light {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 999px;
    border: 1px solid transparent;
    flex: 0 0 auto;
}
.ops-light--green {
    background: #2a9d8f;
    border-color: #2a9d8f;
}
.ops-light--amber {
    background: #e9c46a;
    border-color: #e9c46a;
}
.ops-light--red {
    background: #e76f51;
    border-color: #e76f51;
}
.ops-light--gray {
    background: #cbd5e1;
    border-color: #cbd5e1;
}
.ops-light-text {
    font-size: 12px;
    color: #243b53;
}
.ops-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
}
.ops-table-wrap {
    overflow-x: auto;
    border: 1px solid #d8dee4;
    border-radius: 8px;
    background: #ffffff;
}
.ops-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 760px;
}
.ops-table th,
.ops-table td {
    border-bottom: 1px solid #e6ebf1;
    padding: 10px 12px;
    vertical-align: top;
    text-align: left;
}
.ops-table thead th {
    background: #f8fafb;
    color: #3b4a5a;
    font-size: 12px;
    font-weight: 600;
}
.ops-table__rowlabel {
    font-size: 12px;
    font-weight: 600;
    color: #243b53;
    white-space: nowrap;
}
.ops-table__state {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.ops-table__detail {
    margin-top: 4px;
    font-size: 11px;
    color: #6b7280;
}
.ops-callout {
    border-radius: 8px;
    padding: 10px 12px;
    background: #fff8ef;
    border: 1px solid #f1d4b5;
    color: #7c4a12;
}
.ops-callout--red {
    background: #fff5f4;
    border-color: #f3c1bb;
    color: #8a2f24;
}
.ops-callout ul {
    margin: 0;
    padding-left: 18px;
}
.wxcam-player {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
}
.wxcam-player__meta {
    display: flex;
    justify-content: center;
    text-align: center;
}
.wxcam-player__meta-text {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-width: 820px;
}
.wxcam-player__title {
    font-size: 16px;
    font-weight: 600;
    color: #22313f;
    text-align: center;
    word-break: break-word;
}
.wxcam-player__subtitle {
    font-size: 12px;
    color: #5f6c7b;
    line-height: 1.4;
}
.wxcam-player__controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: center;
}
.wxcam-player__controls button,
.wxcam-player__controls select {
    border: 1px solid #c5d0da;
    background: #ffffff;
    color: #22313f;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}
.wxcam-player__controls button {
    cursor: pointer;
}
.wxcam-player__inline-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    color: #475569;
}
.wxcam-player__checkbox input {
    margin: 0;
}
.wxcam-player__seek {
    display: grid;
    grid-template-columns: 60px minmax(0, 1fr) 60px;
    gap: 10px;
    align-items: center;
    width: 100%;
}
.wxcam-player__seek input[type="range"] {
    width: 100%;
}
.wxcam-player__time {
    font-size: 13px;
    color: #5f6c7b;
    text-align: center;
}
.wxcam-player__frame {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #edf2f7;
    border-radius: 8px;
    overflow: hidden;
    padding: 10px;
    border: 1px solid #d8e1e8;
}
.wxcam-player__frame video {
    display: block;
    width: 100%;
    height: auto;
    max-height: 68vh;
}
.wxcam-still {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 100%;
}
.wxcam-still__frame {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #edf2f7;
    border-radius: 8px;
    overflow: hidden;
    padding: 10px;
    border: 1px solid #d8e1e8;
}
.wxcam-still__frame img {
    display: block;
    width: auto;
    height: auto;
    max-width: 100%;
    max-height: min(68vh, 900px);
    object-fit: contain;
}
.wxcam-still--vertical .wxcam-still__frame img {
    max-height: min(56vh, 680px);
}
.wxcam-browser {
    gap: 10px;
}
.wxcam-browser__toolbar .bk-card-body {
    padding: 10px 12px;
}
.wxcam-hour-strip {
    display: flex;
    flex-direction: column;
    gap: 8px;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    padding: 10px 12px;
    background: #fbfcfd;
}
.wxcam-hour-strip__header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px 14px;
    align-items: baseline;
}
.wxcam-hour-strip__title {
    font-size: 12px;
    font-weight: 600;
    color: #22313f;
}
.wxcam-hour-strip__hint {
    font-size: 11px;
    color: #5f6c7b;
}
.wxcam-hour-strip__scroller {
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(62px, 62px);
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 2px;
}
.wxcam-hour-strip__tile {
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: center;
    justify-content: flex-start;
    padding: 4px;
    border: 1px solid #d8e1e8;
    border-radius: 7px;
    background: #ffffff;
    min-height: 76px;
}
.wxcam-hour-strip__tile--day {
    min-height: 96px;
}
.wxcam-hour-strip__tile--recent {
    justify-content: center;
    min-height: 56px;
}
.wxcam-hour-strip__tile--active {
    border-color: #0b7285;
    box-shadow: inset 0 0 0 1px #0b7285;
}
.wxcam-hour-strip__tile--empty {
    background: #f6f8fb;
    border-style: dashed;
}
.wxcam-hour-strip__thumb {
    display: block;
    width: 100%;
    height: 54px;
    object-fit: cover;
    border-radius: 5px;
    background: #edf2f7;
}
.wxcam-hour-strip__placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 54px;
    border-radius: 5px;
    background: #edf2f7;
    color: #7b8794;
    font-size: 11px;
}
.wxcam-hour-strip__chip,
.wxcam-hour-strip__hour {
    font-size: 11px;
    color: #4c5c6b;
    line-height: 1.2;
}
.wxcam-hour-strip__chip {
    padding: 2px 6px;
    border-radius: 999px;
    background: #eff6f8;
    color: #0b7285;
}
.wxcam-hour-strip__empty {
    font-size: 12px;
    color: #647283;
}
.wxcam-hour-tile {
    gap: 1px;
    padding: 1px;
    border: 1px solid #d8e1e8;
    border-radius: 2px;
    background: #ffffff;
}
.wxcam-hour-tile > .bk {
    margin: 0 !important;
}
.wxcam-hour-tile__img {
    display: block;
    width: auto;
    max-width: 100%;
    max-height: 88px;
    margin: 0 auto;
    border-radius: 2px;
    background: #edf2f7;
}
.wxcam-hour-tile button {
    padding: 1px 4px !important;
    min-height: 18px;
    font-size: 10px !important;
    line-height: 1.1 !important;
}
.wxcam-hour-tile__placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 56px;
    border-radius: 2px;
    background: #edf2f7;
    color: #647283;
    font-size: 10px;
}
.wxcam-hour-tile--fish .wxcam-hour-tile__img {
    max-height: 198px;
}
.wxcam-hour-tile--fish .wxcam-hour-tile__placeholder {
    min-height: 126px;
}
.wxcam-player--wide .wxcam-player__frame video {
    max-width: min(100%, 1400px);
    object-fit: contain;
}
.wxcam-player--vertical .wxcam-player__frame {
    min-height: min(60vh, 720px);
}
.wxcam-player--vertical .wxcam-player__frame video {
    width: auto;
    max-width: 100%;
    max-height: calc(100vh - 320px);
    object-fit: contain;
}
@media (max-width: 768px) {
    body, .bk { font-size: 14px; }
    .bk.card { padding: 8px; }
    .bk-panel-card { padding: 8px; }
    .bk.pn-row { gap: 8px; }
    .wxcam-player__seek {
        grid-template-columns: 52px minmax(0, 1fr) 52px;
        gap: 8px;
    }
    .wxcam-player--vertical .wxcam-player__frame {
        min-height: 50vh;
    }
    .wxcam-player--vertical .wxcam-player__frame video {
        max-height: calc(100vh - 260px);
    }
    .wxcam-still__frame {
        padding: 6px;
    }
    .wxcam-still__frame img,
    .wxcam-still--vertical .wxcam-still__frame img {
        max-height: 52vh;
    }
}
"""

# Controls card: group all widgets in a tidy stack.
controls = pn.Card(
    pn.Column(
        pn.Row(instrument_select, range_start, range_end, live_toggle, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(var1_select, var2_select, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(bottom_range_m, top_range_m, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(beta_vmin, beta_vmax, ldr_vmin, ldr_vmax, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(lwp_ymin, lwp_ymax, iwv_ymin, iwv_ymax, irr_ymin, irr_ymax, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(prev_btn, next_btn, sizing_mode="stretch_width", margin=(5, 0, 0, 0), css_classes=["mobile-stack"]),
        sizing_mode="stretch_width",
    ),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

pn.extension(raw_css=[css])

# Template layout: header + tabs
template = pn.template.MaterialTemplate(
    title="AURORA Data Viewer",
    logo=DASHBOARD_LOGO,
    favicon=DASHBOARD_FAVICON,
    header_background=ACCENT,
    header_color="white",
    main_max_width="1800px",  # wide but keeps a valid string
)


def _lightweight_placeholder(label: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div class='lazy-tab-placeholder'>{escape(label)} will load after the page opens.</div>",
        sizing_mode="stretch_width",
        margin=0,
    )


interactive_status_container = pn.Column(_lightweight_placeholder("Freshness and status"), sizing_mode="stretch_width")
interactive_availability_container = pn.Column(_lightweight_placeholder("Data availability"), sizing_mode="stretch_width")
science_status_container = pn.Column(_lightweight_placeholder("Science freshness and status"), sizing_mode="stretch_width")
science_availability_container = pn.Column(_lightweight_placeholder("Science data availability"), sizing_mode="stretch_width")
hk_status_container = pn.Column(_lightweight_placeholder("Housekeeping freshness and status"), sizing_mode="stretch_width")
hk_availability_container = pn.Column(_lightweight_placeholder("Housekeeping data availability"), sizing_mode="stretch_width")


def _activate_interactive_footer_metrics() -> None:
    global _INTERACTIVE_FOOTER_LOADED
    if _INTERACTIVE_FOOTER_LOADED:
        return
    interactive_status_container[:] = [interactive_status]
    interactive_availability_container[:] = [interactive_availability]
    _INTERACTIVE_FOOTER_LOADED = True


interactive_footer = pn.Card(
    pn.Row(
        interactive_copy,
        interactive_download,
        interactive_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    interactive_status_container,
    interactive_availability_container,
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

science_footer = pn.Card(
    pn.Row(
        science_copy,
        science_download,
        science_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    science_status_container,
    science_availability_container,
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

hk_footer = pn.Card(
    pn.Row(
        hk_copy,
        hk_download,
        hk_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    hk_status_container,
    hk_availability_container,
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

operations_dashboard = pn.pane.HTML(_ops_operations_markup(), sizing_mode="stretch_width", margin=0)


def _refresh_operations_dashboard() -> None:
    operations_dashboard.object = _ops_operations_markup()


_operations_timer = _safe_periodic_callback(_refresh_operations_dashboard, period=60_000, start=False)


def _lazy_tab_placeholder(label: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div class='lazy-tab-placeholder'>{escape(label)} will load when this tab is opened.</div>",
        sizing_mode="stretch_width",
        margin=0,
    )


science_quicklook_container = pn.Column(_lazy_tab_placeholder("Science quicklooks"), sizing_mode="stretch_width")
housekeeping_quicklook_container = pn.Column(_lazy_tab_placeholder("House keeping quicklooks"), sizing_mode="stretch_width")
operations_container = pn.Column(_lazy_tab_placeholder("Operations dashboard"), sizing_mode="stretch_width")
_LOADED_TABS: set[str] = set()

interactive_tab = pn.Column(controls, interactive_content, interactive_footer, sizing_mode="stretch_width")
science_quicklooks_tab = pn.Column(
    pn.Card(
        pn.Row(science_instrument, science_image_type, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(ql_prev, ql_date, ql_next, sizing_mode="stretch_width"),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card"],
    ),
    science_quicklook_container,
    science_footer,
    sizing_mode="stretch_width",
)
housekeeping_quicklooks_tab = pn.Column(
    pn.Card(
        pn.Row(hk_instrument, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(hk_prev, hk_date, hk_next, sizing_mode="stretch_width"),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card"],
    ),
    housekeeping_quicklook_container,
    hk_footer,
    sizing_mode="stretch_width",
)
operations_tab = pn.Column(operations_container, sizing_mode="stretch_width")
tabs = pn.Tabs(
    ("Interactive Data Browser", interactive_tab),
    ("Science Quicklooks", science_quicklooks_tab),
    ("House Keeping Quicklooks", housekeeping_quicklooks_tab),
    ("Operations Dashboard", operations_tab),
    sizing_mode="stretch_both",
)


def _ensure_active_tab_loaded() -> None:
    active = tabs.active
    if active == 1 and "science" not in _LOADED_TABS:
        science_quicklook_container[:] = [_science_quicklook_image]
        science_status_container[:] = [science_status]
        science_availability_container[:] = [science_availability]
        _LOADED_TABS.add("science")
    elif active == 2 and "housekeeping" not in _LOADED_TABS:
        housekeeping_quicklook_container[:] = [_housekeeping_quicklook_image]
        hk_status_container[:] = [hk_status]
        hk_availability_container[:] = [hk_availability]
        _LOADED_TABS.add("housekeeping")
    elif active == 3 and "operations" not in _LOADED_TABS:
        operations_container[:] = [operations_dashboard]
        _LOADED_TABS.add("operations")
        try:
            _operations_timer.start()
        except RuntimeError:
            pass


def _on_tabs_active_change(_event=None) -> None:
    _ensure_active_tab_loaded()
    _refresh_share_and_download_state()


tabs.param.watch(_on_tabs_active_change, "active")

SITE_FOOTER_HTML = """
<div class="site-footer">
  <div class="site-footer__title">More Aurora project information</div>
  <div class="site-footer__links">
    <div class="site-footer__link">
      <a href="https://gamb2le.pages.dev/" target="_blank" rel="noopener noreferrer">gamb2le.pages.dev</a>
      <div class="site-footer__desc">Project documentation portal with dashboard, infrastructure, and operational notes.</div>
    </div>
    <div class="site-footer__link">
      <a href="https://www.gamb2le.co.uk/" target="_blank" rel="noopener noreferrer">gamb2le.co.uk</a>
      <div class="site-footer__desc">Main Gamb2le website with broader project context and public-facing information.</div>
    </div>
  </div>
</div>
"""


def _site_footer_pane() -> pn.pane.HTML:
    return pn.pane.HTML(SITE_FOOTER_HTML, sizing_mode="stretch_width", margin=0)


interactive_tab.append(_site_footer_pane())
science_quicklooks_tab.append(_site_footer_pane())
housekeeping_quicklooks_tab.append(_site_footer_pane())
operations_tab.append(_site_footer_pane())

main_layout = pn.Column(tabs, sizing_mode="stretch_width", margin=0)

_QUERY_TAB_INDEX = {"interactive": 0, "science": 1, "housekeeping": 2, "operations": 3}

_apply_query_state()
requested_tab = _request_query_args().get("tab")
if requested_tab in _QUERY_TAB_INDEX:
    tabs.active = _QUERY_TAB_INDEX[requested_tab]
_ensure_active_tab_loaded()
_refresh_share_and_download_state()
_APP_BOOTSTRAPPING = False
pn.state.onload(_enable_browser_interactive_render)

tabs.sizing_mode = "stretch_width"
template.main[:] = [main_layout]

def _apply_theme(dark: bool):
    """No-op placeholder (dark mode removed)."""
    return

# Serve the app. `location=True` installs Panel's Location model so the app can
# keep the browser URL aligned with the selected view.
template.servable(location=True)
