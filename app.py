# app.py
# Minimal Panel + Plotly viewer for ceilometer Zarr data.
# - Loads a Zarr dataset once and slices out a time window for plotting.
# - Two heatmaps: attenuated backscatter (log-scaled) and linear depol ratio.
# - Controls: instrument (placeholder), time window, range limits, color limits,
#   a “live” toggle to jump to the latest 24h, and previous/next day navigation.
# - Lightweight coarsening and subsampling to keep plots responsive.

import os
from base64 import b64encode
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
from html import escape
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode

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
    widget_group_options,
)
from extra_housekeeping import (
    extra_housekeeping_daily_png,
    extra_housekeeping_label,
    extra_housekeeping_latest_png,
    extra_housekeeping_tokens,
)
from wxcam_catalog import (
    available_days,
    catalog_time_bounds,
    latest_record,
    records_for_day,
    representative_hourly_records,
)

pn.extension("plotly", notifications=True, sizing_mode="stretch_width")


class WxcamVideoPlayer(pn.reactive.ReactiveHTML):
    src = param.String(default="")
    title = param.String(default="")
    mode_class = param.String(default="wxcam-player--wide")

    _template = """
    <div id="player_shell" class="wxcam-player ${mode_class}">
      <div id="meta_row" class="wxcam-player__meta">
        <div id="title_text" class="wxcam-player__title">{{ title }}</div>
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
        "zarr_default": "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr",
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
        "latest_image": _path_from_env("CEILOMETER_LATEST_IMAGE", APP_DIR / "last24h.png"),
    },
    "Cloud Radar": {
        "zarr_env": "CLOUD_RADAR_ZARR_PATH",
        "zarr_default": "/mnt/data/ass/rpgfmcw94/cloud_radar.zarr",
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
        "latest_image": _path_from_env("CLOUD_RADAR_LATEST_IMAGE", APP_DIR / "last24h_cloudradar.png"),
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
        "chunk_spec": {"time": 1200},
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
            "FISH HDR": {
                "label": "FISH HDR",
                "image_type": "fish_hdr",
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            },
            "PANO HDR": {
                "label": "PANO HDR",
                "image_type": "pano_hdr",
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            },
        },
        "default_top": "FISH HDR",
        "default_bottom": "PANO HDR",
        "quicklook_dir": _path_from_env("WXCAM_QUICKLOOK_DIR", QUICKLOOK_ROOT / "wxcam"),
        "latest_image": _path_from_env("WXCAM_LATEST_IMAGE", QUICKLOOK_ROOT / "wxcam" / "latest.jpg"),
    },
    "Scanning Microwave Radiometer": {
        "zarr_env": "HATPRO_ZARR_PATH",
        "zarr_default": "/mnt/data/ass/hatprog5/hatpro.zarr",
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
    if name != "asfs-fast-sonic"
}

DEFAULT_WINDOW = timedelta(hours=24)
LIVE_REFRESH_MS = 60_000  # how often to snap to latest when live is on (ms)
TIME_SUBSAMPLE = 2  # slice time to lighten payloads
TIME_TARGET = 300  # target max time samples for plotting
HEIGHT_TARGET = 200  # target max height samples for plotting
DATA_REFRESH_MS = 300_000  # reload base dataset every 5 minutes
# A small future tolerance keeps normal clock skew harmless while protecting
# the dashboard from bogus outlier timestamps that can blank the latest window.
FUTURE_TIME_TOLERANCE = timedelta(days=2)

_BASE_DS: dict[str, xr.Dataset | None] = {}
CURRENT_INSTRUMENT = "Ceilometer"


def _cfg(inst: str | None = None):
    return INSTRUMENTS[inst or CURRENT_INSTRUMENT]


def _zarr_path(inst: str | None = None):
    cfg = _cfg(inst)
    return os.environ.get(cfg["zarr_env"], cfg["zarr_default"])


def _wxcam_catalog_path(inst: str | None = None) -> Path:
    cfg = _cfg(inst)
    return Path(os.environ.get(cfg["catalog_env"], cfg["catalog_default"]))


def _wxcam_daily_video_root() -> Path:
    return Path(os.environ.get("WXCAM_DAILY_VIDEO_DIR", "/data/aurora/products/wxcam/daily_videos"))


def _wxcam_hourly_thumbnail_root() -> Path:
    return Path(os.environ.get("WXCAM_HOURLY_THUMB_DIR", "/data/aurora/products/wxcam/hourly_thumbnails"))


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
    return ds


def _refresh_base_dataset(inst: str | None = None):
    """Drop the cached dataset so the next access reopens the Zarr (captures new data)."""
    inst = inst or CURRENT_INSTRUMENT
    _BASE_DS[inst] = None


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
    with _timed_perf("dataset_time_bounds", instrument=inst) as perf:
        if inst == "wxcam":
            lower, upper = catalog_time_bounds(_wxcam_catalog_path(inst))
            perf["source"] = "wxcam_catalog"
            perf["time_start"] = lower
            perf["time_end"] = upper
            return lower, upper
        ds = _get_base_dataset(inst)
        if ds is None or "time" not in ds:
            perf["status"] = "no_dataset"
            return None, None
        times = np.asarray(ds["time"].values)
        perf["raw_time_count"] = int(times.size)
        if times.size == 0:
            perf["status"] = "empty"
            return None, None
        valid = _valid_time_mask(times)
        times = times[valid]
        perf["valid_time_count"] = int(times.size)
        lower = pd.Timestamp(times.min()).to_pydatetime(warn=False)
        upper = pd.Timestamp(times.max()).to_pydatetime(warn=False)
        perf["time_start"] = lower
        perf["time_end"] = upper
        return lower, upper


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


def _wxcam_hour_bits(selection: str, day_token: str) -> list[bool]:
    rows_by_hour = _wxcam_hourly_image_rows(selection, day_token)
    return [hour in rows_by_hour for hour in range(24)]


def _wxcam_combined_hour_states(day_token: str) -> list[int]:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return [0] * 24
    fish = representative_hourly_records(_wxcam_catalog_path("wxcam"), "fish_hdr", day_utc, media_kind="image")
    pano = representative_hourly_records(_wxcam_catalog_path("wxcam"), "pano_hdr", day_utc, media_kind="image")
    states: list[int] = []
    for hour in range(24):
        have_fish = hour in fish
        have_pano = hour in pano
        states.append(2 if (have_fish and have_pano) else 1 if (have_fish or have_pano) else 0)
    return states


def _availability_bar_markup(
    states: list[int | bool],
    start_label: str,
    end_label: str,
    caption: str,
) -> str:
    if not states:
        return "<div class='availability-shell'><div class='availability-empty'>No availability information</div></div>"
    parts = []
    for state in states:
        if state in (True, 2):
            cls = "availability-segment availability-segment--full"
        elif state == 1:
            cls = "availability-segment availability-segment--partial"
        else:
            cls = "availability-segment availability-segment--empty"
        parts.append(f"<span class='{cls}'></span>")
    return (
        "<div class='availability-shell'>"
        f"<div class='availability-caption'>{escape(caption)}</div>"
        f"<div class='availability-bar'>{''.join(parts)}</div>"
        f"<div class='availability-scale'><span>{escape(start_label)}</span><span>{escape(end_label)}</span></div>"
        "</div>"
    )


def _status_strip_markup(items: list[tuple[str, str, str]]) -> str:
    pills = []
    for label, value, tone in items:
        pills.append(
            f"<span class='status-pill status-pill--{escape(tone)}'><strong>{escape(label)}</strong> {escape(value)}</span>"
        )
    return f"<div class='status-strip'>{''.join(pills)}</div>" if pills else ""


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


def open_window(t0, t1, bottom_m=None, top_m=None, instrument: str | None = None):
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
        perf["time_subsample"] = int(time_subsample)
        perf["time_target"] = int(time_target)
        perf["height_target"] = int(height_target)
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
        if time_subsample > 1:
            ds = ds.isel(time=slice(None, None, time_subsample))
        try:
            if ds.sizes.get("range", 0) > height_target:
                fh = max(int(np.ceil(ds.sizes["range"] / height_target)), 1)
                perf["range_coarsen_factor"] = int(fh)
                ds = ds.coarsen({"range": fh}, boundary="trim").mean()
            if ds.sizes.get("time", 0) > time_target:
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
tmin, tmax = _dataset_time_bounds()
default_end = tmax or datetime.utcnow()
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
hk_instrument = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=INSTRUMENT_OPTIONS)

_live_guard = False
_instrument_guard = False
_live_cb = None  # handle for periodic callback (used for live refresh)
_relayout_guard = False  # prevents loops when syncing zoom back to widgets
pn.state.add_periodic_callback(_refresh_base_dataset, period=DATA_REFRESH_MS, start=True)


def _apply_instrument_defaults(inst: str, reset_time: bool = True):
    """Switch instrument: refresh dataset cache, reset controls, and relabel color widgets."""
    global CURRENT_INSTRUMENT, _instrument_guard
    _instrument_guard = True
    CURRENT_INSTRUMENT = inst
    _refresh_base_dataset(inst)
    cfg = _cfg(inst)
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
            tmin, tmax = _dataset_time_bounds(inst)
            end = tmax or datetime.utcnow()
            start = end - DEFAULT_WINDOW
            range_start.value = start
            range_end.value = end
            # WXcam is a manual browser: refresh when switching back into it,
            # but do not keep a hidden live timer running while it is selected.
            _set_live(not is_wxcam)

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
            science_image_type.value = var1_name
            wxcam_image_type.options = list(vars_cfg.keys())
            wxcam_image_type.value = var1_name
            _refresh_wxcam_ql_options(preserve_current=False)
        else:
            science_image_type.name = "Image type"
            science_image_type.options = []
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
    """Jump window to latest 24h and update date pickers."""
    global _live_guard
    _live_guard = True
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
_live_cb = pn.state.add_periodic_callback(_auto_refresh, period=LIVE_REFRESH_MS, start=True)


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
# Use stretch_width so height stays predictable on mobile.
plot_pane = pn.pane.Plotly(config={"responsive": True}, sizing_mode="stretch_width")
interactive_content = pn.Column(plot_pane, sizing_mode="stretch_both")


def _show_plot(fig: go.Figure) -> None:
    plot_pane.object = fig
    interactive_content[:] = [plot_pane]


def _image_type_from_selection(selection: str) -> str:
    cfg = _cfg("wxcam")
    return cfg["vars"][selection]["image_type"]


def _wxcam_today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _wxcam_daily_video_dir(image_type: str) -> Path:
    return _wxcam_daily_video_root() / image_type


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


def _media_pane(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".mp4":
        return pn.pane.Video(path, sizing_mode="stretch_width", autoplay=False, loop=False, muted=True)
    return pn.pane.Image(path, sizing_mode="stretch_width")


@lru_cache(maxsize=8)
def _cached_video_data_uri(path_str: str, size_bytes: int, mtime_ns: int) -> str:
    video_bytes = Path(path_str).read_bytes()
    encoded = b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def _video_data_uri(path: Path) -> str:
    stat_result = path.stat()
    return _cached_video_data_uri(str(path), stat_result.st_size, stat_result.st_mtime_ns)


@lru_cache(maxsize=256)
def _cached_image_data_uri(path_str: str, size_bytes: int, mtime_ns: int) -> str:
    image_bytes = Path(path_str).read_bytes()
    encoded = b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _image_data_uri(path: Path) -> str:
    stat_result = path.stat()
    return _cached_image_data_uri(str(path), stat_result.st_size, stat_result.st_mtime_ns)


def _build_wxcam_video_view(path: Path, selection: str, selected_label: str):
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_video_view_build", instrument="wxcam", image_type=image_type, path=path, selected_label=selected_label) as perf:
        mode_class = "wxcam-player--vertical" if image_type == "fish_hdr" else "wxcam-player--wide"
        title = f"{selection} | {selected_label} | {path.name}"
        stat_result = path.stat()
        perf["size_bytes"] = int(stat_result.st_size)
        return pn.Column(
            WxcamVideoPlayer(src=_video_data_uri(path), title=title, mode_class=mode_class, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )


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
wxcam_prev = pn.widgets.Button(name="<<", button_type="default")
wxcam_next = pn.widgets.Button(name=">>", button_type="default")


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


_wxcam_ql_timer = pn.state.add_periodic_callback(_refresh_wxcam_latest_if_needed, period=300_000, start=True)


@pn.depends(wxcam_date.param.value, wxcam_image_type.param.value)
def _wxcam_interactive_media(selected, selection):
    selection = selection or _cfg("wxcam")["default_top"]
    with _timed_perf("wxcam_interactive_render", instrument="wxcam", selection=selection, selected=selected) as perf:
        path = _wxcam_interactive_video_options(selection).get(selected)
        if not path:
            perf["status"] = "missing_option"
            return pn.pane.Markdown("No media available for this selection.")
        video_path = Path(path)
        perf["path"] = video_path
        if not video_path.exists():
            perf["status"] = "missing_file"
            return pn.pane.Markdown("No media available for this selection.")
        perf["status"] = "ok"
        return _build_wxcam_video_view(video_path, selection, selected)


# wxcam now uses the Interactive tab as its primary browser/player so the
# controls and playback state live in one place instead of competing with the
# calendar quicklook flow used by the other instruments.
wxcam_interactive_browser = pn.Column(
    pn.Card(
        pn.Row(wxcam_image_type, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(wxcam_prev, wxcam_date, wxcam_next, sizing_mode="stretch_width"),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
    ),
    _wxcam_interactive_media,
    sizing_mode="stretch_width",
)


wxcam_image_type.param.watch(_on_wxcam_image_type_change, "value")


def _update_wxcam_view(start, end, top_name: str, bottom_name: str) -> None:
    interactive_content[:] = [wxcam_interactive_browser]


def _update_hatpro_view(start, end, bottom_val, top_val, lymin, lymax, iymin, iymax, rymin, rymax):
    """Custom renderer for HATPRO radiometer: split LWP/IWV and IRR; T_PROF heatmap."""
    print(f"[hatpro] render window {start} -> {end}")
    with _timed_perf("hatpro_render", instrument="Scanning Microwave Radiometer", start=start, end=end) as perf:
        bottom = max(float(bottom_val), 0.0)
        top = max(float(top_val), bottom + 100.0)
        ds = open_window(start, end, bottom_m=bottom, top_m=top, instrument="Scanning Microwave Radiometer")
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
                font=dict(color="#222", size=16),
            )
            fig.update_layout(height=600, margin=dict(l=40, r=40, t=40, b=40))
            _show_plot(fig)
            return

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
                            font=dict(size=14, color="#222"),
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
                gridcolor="#dddddd",
                linecolor="#222222",
                tickfont=dict(color="#222222", size=12),
                title_font=dict(color="#222222", size=12),
                row=row,
                col=1,
            )
        fig.update_xaxes(title_text="Date and Time (UTC)", title_standoff=40, row=3, col=1)

        # y-axes
        fig.update_yaxes(title_text="LWP (g/m²)", range=[float(lymin), float(lymax)], row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="IWV (kg/m²)", range=[float(iymin), float(iymax)], row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="IRR / SURF_T (°C)", range=[float(rymin), float(rymax)], row=2, col=1)
        fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=3, col=1, showgrid=True, gridcolor="#dddddd", linecolor="#222222")

        fig.update_layout(
            showlegend=False,
            height=max(700, int(pn.state.viewport_height * 0.8)) if hasattr(pn.state, "viewport_height") else 900,
            margin=dict(l=60, r=80, t=30, b=110),
            coloraxis=dict(
                colorscale=cfg["vars"]["T_PROF"]["colorscale"],
                cmin=cfg["vars"]["T_PROF"]["clim"][0],
                cmax=cfg["vars"]["T_PROF"]["clim"][1],
                colorbar=dict(title=dict(text="T (K)", side="right"), x=1.02, y=0.18, len=0.3, tickfont=dict(color="#222222", size=9)),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#222222", size=13),
            annotations=tuple(list(fig.layout.annotations) + noon_annots),
        )
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        _show_plot(fig)


def _update_stacked_timeseries_view(instrument: str, start, end):
    """Render a 1D summary instrument with fixed multi-panel layouts."""
    print(f"[{instrument}] render window {start} -> {end}")
    with _timed_perf("stacked_timeseries_render", instrument=instrument, start=start, end=end) as perf:
        source_instruments = summary_source_instruments(instrument)
        perf["source_instruments"] = list(source_instruments)
        ds = combine_summary_datasets(
            instrument,
            *(open_window(start, end, instrument=source_inst) for source_inst in source_instruments),
        )
        times = pd.to_datetime(ds["time"].values) if ds is not None and "time" in ds else None
        perf["time_count"] = 0 if times is None else int(len(times))
        if times is None or len(times) == 0:
            perf["status"] = "no_data"
            empty = go.Figure()
            empty.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            empty.update_layout(height=600, paper_bgcolor="white", plot_bgcolor="white")
            _show_plot(empty)
            return
        try:
            fig = build_summary_plotly(ds, instrument, title=display_name(instrument))
        except ValueError:
            perf["status"] = "no_data"
            empty = go.Figure()
            empty.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            empty.update_layout(height=600, paper_bgcolor="white", plot_bgcolor="white")
            _show_plot(empty)
            return
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        _show_plot(fig)


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
    """Render both heatmaps for the current window and control values."""
    if _instrument_guard:
        # Skip expensive re-renders while we batch instrument switch updates.
        return
    # Ensure global instrument matches the dropdown to avoid stale cache use.
    global CURRENT_INSTRUMENT
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    print(f"[update-view] instrument param={instrument} current={CURRENT_INSTRUMENT}")
    with _timed_perf("interactive_view_update", instrument=instrument, start=start, end=end) as perf:
        perf["top_var"] = var1_name
        perf["bottom_var"] = var2_name
        perf["bottom_m"] = bottom_val
        perf["top_m"] = top_val
        if instrument == "Scanning Microwave Radiometer":
            perf["view_type"] = "hatpro"
            _update_hatpro_view(start, end, bottom_val, top_val, lymin, lymax, iymin, iymax, rymin, rymax)
            return
        if _is_wxcam_instrument(instrument):
            perf["view_type"] = "wxcam"
            _update_wxcam_view(start, end, var1_name, var2_name)
            return
        if _is_stacked_timeseries_instrument(instrument):
            perf["view_type"] = "stacked_timeseries"
            _update_stacked_timeseries_view(instrument, start, end)
            return
        perf["view_type"] = "heatmap"
        bottom = max(float(bottom_val), 0.0)
        top = max(float(top_val), bottom + 100.0)
        ds = open_window(start, end, bottom_m=bottom, top_m=top, instrument=instrument)
        cfg = _cfg()
        vars_cfg = cfg["vars"]
        var1 = vars_cfg.get(var1_name)
        var2 = vars_cfg.get(var2_name)
        bg = "white"
        fg = "#222222"
        grid = "#dddddd"
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
            _show_plot(fig)
            return
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
            linecolor=fg,
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
            linecolor=fg,
            tickfont=dict(color=fg, size=12),
            title_font=dict(color=fg, size=12),
            row=1,
            col=1,
        )
        fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=fg, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=fg, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=2, col=1)
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
        _show_plot(fig)


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
    """Move calendar selection by delta steps in the refreshed options list."""
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


_ql_timer = pn.state.add_periodic_callback(_refresh_latest_if_needed, period=300_000, start=True)


def _refresh_hk_latest_if_needed():
    """If viewing the latest housekeeping quicklook, reload the mapping and redraw."""
    if hk_date.value == "Today (latest)":
        global _hk_options
        _hk_options = _quicklook_options(hk_instrument.value, mode="housekeeping")
        hk_date.param.trigger("value")


_hk_timer = pn.state.add_periodic_callback(_refresh_hk_latest_if_needed, period=300_000, start=True)

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
_session_heartbeat_cb = pn.state.add_periodic_callback(_log_session_heartbeat, period=SESSION_HEARTBEAT_MS, start=True)


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
        return _availability_bar_markup(bits, start_label, end_label, "HDR image availability by hour")
    start = _ensure_utc(range_start.value)
    end = _ensure_utc(range_end.value)
    bits = _binned_time_coverage(_instrument_time_index(inst), start, end, segments=72)
    start_label = start.strftime("%m-%d %H:%M") if start else "--"
    end_label = end.strftime("%m-%d %H:%M") if end else "--"
    return _availability_bar_markup(bits, start_label, end_label, "Availability across the selected time window")


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
        return _availability_bar_markup(bits, "00:00", end_label, "HDR image availability by hour")
    start, end, _day_token = _selected_token_window(ql_date.value)
    times = _instrument_time_index(inst)
    if start is None or end is None:
        return _availability_bar_markup([], "--", "--", "Availability by hour")
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    bits, _missing, _total = _hourly_coverage_summary(times[mask], start, end)
    end_label = end.strftime("%H:%M")
    return _availability_bar_markup(bits, "00:00", end_label, "Availability by hour")


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
        return _availability_bar_markup(states, "00:00", end_label, "HDR availability by hour (full or partial)")
    if start is None or end is None:
        return _availability_bar_markup([], "--", "--", "Availability by hour")
    times = _instrument_time_index(inst)
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    bits, _missing, _total = _hourly_coverage_summary(times[mask], start, end)
    return _availability_bar_markup(bits, "00:00", end.strftime("%H:%M"), "Availability by hour")


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
science_download = pn.widgets.FileDownload(name="Download PNG", button_type="default", auto=False, embed=False, width=130)
hk_download = pn.widgets.FileDownload(name="Download PNG", button_type="default", auto=False, embed=False, width=130)


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


def _build_share_url(tab_slug: str) -> str:
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
    query = urlencode({k: v for k, v in params.items() if v not in ("", None)})
    return f"{_request_base_url()}?{query}" if query else _request_base_url()


def _refresh_share_and_download_state(*_events) -> None:
    interactive_share_url.value = _build_share_url("interactive")
    science_share_url.value = _build_share_url("science")
    hk_share_url.value = _build_share_url("housekeeping")

    interactive_download.visible = not _is_wxcam_instrument(instrument_select.value)

    science_path = _science_download_path()
    science_download.file = science_path
    science_download.filename = science_path.name if science_path else "science_quicklook.png"
    science_download.disabled = science_path is None

    hk_path = _hk_download_path()
    hk_download.file = hk_path
    hk_download.filename = hk_path.name if hk_path else "housekeeping_quicklook.png"
    hk_download.disabled = hk_path is None


def _apply_query_state() -> None:
    args = _request_query_args()
    if not args:
        return
    instrument = args.get("instrument")
    if instrument in INSTRUMENTS:
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
    if args.get("science_instrument") in INSTRUMENTS:
        science_instrument.value = args["science_instrument"]
    if args.get("science_image_type") in list(science_image_type.options):
        science_image_type.value = args["science_image_type"]
    if args.get("science_date") in list(ql_date.options):
        ql_date.value = args["science_date"]
    if args.get("science_selected_hour"):
        wxcam_calendar_state.selected_hour_path = args["science_selected_hour"]
    if args.get("hk_instrument") in INSTRUMENTS:
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


ACCENT = "#0b7285"  # header/accent color
css = """
# Global font override for a clean, consistent look.
body, .bk {
    font-family: "SF Pro Display","SF Pro","-apple-system","BlinkMacSystemFont","Segoe UI",sans-serif;
    font-size: 15px;
    background: #ffffff;
    color: #1f2933;
}
.bk.card, .bk-panel-models-card {
    border: 1px solid #d8dee4;
    border-radius: 8px;
    box-shadow: none;
    background: #ffffff;
}
.bk-btn, button.bk-btn {
    border-radius: 6px;
    border: 1px solid #cfd8df;
    box-shadow: none;
}
.bk-btn-primary, button.bk-btn-primary {
    background: #0b7285;
    border-color: #0b7285;
    color: #ffffff;
}
.bk-input {
    border-radius: 6px;
    border-color: #cfd8df;
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
    border: 1px solid #d8dee4;
    background: #f8fafb;
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
.availability-shell {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
}
.availability-caption {
    font-size: 12px;
    color: #566370;
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
    border: 1px solid #d8dee4;
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
    border-color: #d8dee4;
}
.availability-scale {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #6b7280;
}
.availability-empty {
    font-size: 12px;
    color: #6b7280;
}
.action-row {
    align-items: center;
    gap: 8px;
}
.wxcam-player {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 100%;
}
.wxcam-player__meta {
    display: flex;
    justify-content: center;
}
.wxcam-player__title {
    font-size: 14px;
    color: #1f2933;
    text-align: center;
    word-break: break-word;
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
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #0f172a;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 14px;
}
.wxcam-player__controls button {
    cursor: pointer;
}
.wxcam-player__inline-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    color: #334155;
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
    color: #475569;
    text-align: center;
}
.wxcam-player__frame {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f172a;
    border-radius: 8px;
    overflow: hidden;
    padding: 8px;
}
.wxcam-player__frame video {
    display: block;
    width: 100%;
    height: auto;
    max-height: 68vh;
}
.wxcam-hour-tile {
    gap: 1px;
    padding: 1px;
    border: 1px solid #cbd5e1;
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
    background: #0f172a;
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
    background: #e2e8f0;
    color: #475569;
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
}
"""

# Controls card: group all widgets in a tidy stack.
controls = pn.Card(
    pn.Column(
        pn.Row(instrument_select, range_start, range_end, live_toggle, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(
            interactive_copy,
            interactive_download,
            interactive_share_url,
            sizing_mode="stretch_width",
            css_classes=["mobile-stack", "action-row"],
        ),
        interactive_status,
        interactive_availability,
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
    header_background=ACCENT,
    header_color="white",
    main_max_width="1800px",  # wide but keeps a valid string
)

interactive_tab = pn.Column(controls, interactive_content, sizing_mode="stretch_both")
science_quicklooks_tab = pn.Column(
    pn.Card(
        pn.Row(science_instrument, science_image_type, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(ql_prev, ql_date, ql_next, sizing_mode="stretch_width"),
        pn.Row(
            science_copy,
            science_download,
            science_share_url,
            sizing_mode="stretch_width",
            css_classes=["mobile-stack", "action-row"],
        ),
        science_status,
        science_availability,
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
    ),
    _science_quicklook_image,
    sizing_mode="stretch_both",
)
housekeeping_quicklooks_tab = pn.Column(
    pn.Card(
        pn.Row(hk_instrument, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(hk_prev, hk_date, hk_next, sizing_mode="stretch_width"),
        pn.Row(
            hk_copy,
            hk_download,
            hk_share_url,
            sizing_mode="stretch_width",
            css_classes=["mobile-stack", "action-row"],
        ),
        hk_status,
        hk_availability,
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
    ),
    _housekeeping_quicklook_image,
    sizing_mode="stretch_both",
)
tabs = pn.Tabs(
    ("Interactive Data Browser", interactive_tab),
    ("Science Quicklooks", science_quicklooks_tab),
    ("House Keeping Quicklooks", housekeeping_quicklooks_tab),
    sizing_mode="stretch_both",
)

_QUERY_TAB_INDEX = {"interactive": 0, "science": 1, "housekeeping": 2}

_apply_query_state()
requested_tab = _request_query_args().get("tab")
if requested_tab in _QUERY_TAB_INDEX:
    tabs.active = _QUERY_TAB_INDEX[requested_tab]
_refresh_share_and_download_state()

template.main[:] = [tabs]

def _apply_theme(dark: bool):
    """No-op placeholder (dark mode removed)."""
    return

# Serve the app
template.servable()
