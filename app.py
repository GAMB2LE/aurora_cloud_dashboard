# app.py
# Minimal Panel + Plotly viewer for ceilometer Zarr data.
# - Loads a Zarr dataset once and slices out a time window for plotting.
# - Two heatmaps: attenuated backscatter (log-scaled) and linear depol ratio.
# - Controls: instrument (placeholder), time window, range limits, color limits,
#   a “live” toggle to jump to the latest 24h, and previous/next day navigation.
# - Lightweight coarsening and subsampling to keep plots responsive.

import os
from base64 import b64encode
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import param
from panel.io import hold
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from wxcam_catalog import (
    catalog_time_bounds,
    records_for_day,
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
        <button id="back_btn" type="button" onclick="${script('jump_back')}">-0.1s</button>
        <button id="forward_btn" type="button" onclick="${script('jump_forward')}">+0.1s</button>
        <label id="speed_wrap" class="wxcam-player__inline-label">
          <span>Speed</span>
          <select id="speed_select" onchange="${script('change_speed')}">
            <option value="0.1">0.1x</option>
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1" selected>1x</option>
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
          const nextTime = Math.max(0, (video_el.currentTime || 0) - 0.1);
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "jump_forward": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const candidateTime = (video_el.currentTime || 0) + 0.1;
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
        "vars": {
            "all": {
                "label": "All Variables",
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            },
        },
        "default_top": "all",
        "default_bottom": "all",
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
        "vars": {
            "all": {
                "label": "All Variables",
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            },
        },
        "default_top": "all",
        "default_bottom": "all",
        "quicklook_dir": _path_from_env("ASFS_LOGGER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_logger"),
        "latest_image": _path_from_env("ASFS_LOGGER_LATEST_IMAGE", QUICKLOOK_ROOT / "asfs_logger" / "latest.png"),
    },
    "power": {
        "zarr_env": "POWER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/power/power.zarr",
        "chunk_spec": {"time": 1200},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": {
            "all": {
                "label": "All Variables",
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            },
        },
        "default_top": "all",
        "default_bottom": "all",
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
    try:
        ds = xr.open_zarr(zarr_path, chunks=cfg["chunk_spec"], consolidated=cfg["consolidated"])
    except Exception as first_exc:
        try:
            ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False)
        except Exception as second_exc:
            print(f"[base-ds] unavailable for {inst}: {first_exc}; fallback failed: {second_exc}")
            _BASE_DS[inst] = None
            return None
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
    if inst == "wxcam":
        return catalog_time_bounds(_wxcam_catalog_path(inst))
    ds = _get_base_dataset(inst)
    if ds is None or "time" not in ds:
        return None, None
    times = np.asarray(ds["time"].values)
    if times.size == 0:
        return None, None
    valid = _valid_time_mask(times)
    times = times[valid]
    return (
        pd.Timestamp(times.min()).to_pydatetime(warn=False),
        pd.Timestamp(times.max()).to_pydatetime(warn=False),
    )


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
    cfg = _cfg(instrument)
    t0 = _ensure_utc(t0)
    t1 = _ensure_utc(t1)
    if t0 is None or t1 is None or t0 >= t1:
        return xr.Dataset()
    duration = t1 - t0
    height_span = None
    if bottom_m is not None or top_m is not None:
        b = max(bottom_m or 0.0, 0.0)
        t = top_m if top_m is not None else cfg["height_load_max"]
        height_span = max(t - b, 0.0)
    time_subsample, time_target, height_target = _coarsen_targets(duration, height_span)
    base = _get_base_dataset(instrument)
    if base is None:
        return xr.Dataset()
    try:
        tvals = base["time"].values
        mask = _valid_time_mask(tvals) & (tvals >= np.datetime64(t0)) & (tvals <= np.datetime64(t1))
        if not np.any(mask):
            return xr.Dataset()
        idx = np.nonzero(mask)[0]
        ds = base.isel(time=idx)
    except Exception:
        ds = base
    has_range = "range" in ds.coords or "range" in ds.dims
    if has_range:
        try:
            ds = ds.sel({"range": slice(0, cfg["height_load_max"])})
        except Exception:
            ds = ds.where(ds["range"] <= cfg["height_load_max"], drop=True)
    # If the user narrowed the plotted range, trim the data before coarsening so
    # we keep more vertical detail within the zoomed band.
    if has_range and (bottom_m is not None or top_m is not None):
        low = max(bottom_m or 0.0, 0.0)
        high = min(top_m or cfg["height_load_max"], cfg["height_load_max"])
        try:
            ds = ds.sel({"range": slice(low, high)})
        except Exception:
            ds = ds.where((ds["range"] >= low) & (ds["range"] <= high), drop=True)
    if time_subsample > 1:
        ds = ds.isel(time=slice(None, None, time_subsample))
    # Coarsen to target sample counts to keep payloads small.
    try:
        if ds.sizes.get("range", 0) > height_target:
            fh = max(int(np.ceil(ds.sizes["range"] / height_target)), 1)
            ds = ds.coarsen({"range": fh}, boundary="trim").mean()
        if ds.sizes.get("time", 0) > time_target:
            ft = max(int(np.ceil(ds.sizes["time"] / time_target)), 1)
            ds = ds.coarsen({"time": ft}, boundary="trim").mean()
    except Exception:
        pass
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
    return inst in {"vaisalamet", "asfs-logger", "power"}


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
instrument_select = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=list(INSTRUMENTS.keys()))
calendar_instrument = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=list(INSTRUMENTS.keys()))
calendar_image_type = pn.widgets.Select(name="Image type", options=[], visible=False)

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
        var1_name = cfg["default_top"]
        var2_name = cfg["default_bottom"]
        var1_select.options = list(vars_cfg.keys())
        var2_select.options = list(vars_cfg.keys())
        var1_select.value = var1_name
        var2_select.value = var2_name
        var1_select.name = "Top var"
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

        calendar_instrument.value = inst

        if reset_time:
            tmin, tmax = _dataset_time_bounds(inst)
            end = tmax or datetime.utcnow()
            start = end - DEFAULT_WINDOW
            range_start.value = start
            range_end.value = end
            _set_live(True)

        # Instrument-specific UI trimming
        is_hatpro = inst == "Scanning Microwave Radiometer"
        is_stacked_timeseries = _is_stacked_timeseries_instrument(inst)
        is_wxcam = _is_wxcam_instrument(inst)
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
        calendar_image_type.visible = is_wxcam
        if is_wxcam:
            calendar_image_type.options = list(vars_cfg.keys())
            calendar_image_type.value = var1_name
            wxcam_image_type.options = list(vars_cfg.keys())
            wxcam_image_type.value = var1_name
            _refresh_wxcam_ql_options(preserve_current=False)
        else:
            calendar_image_type.options = []
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
        # Force quicklook pane refresh even if selection string didn't change
        ql_date.param.trigger("value")
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


def _on_calendar_instrument_change(event):
    """Sync calendar instrument dropdown back to the main instrument selector."""
    if _instrument_guard:
        return
    instrument_select.value = event.new


calendar_instrument.param.watch(_on_calendar_instrument_change, "value")


def _on_var_change(event):
    """Update limit widgets when variable selection changes."""
    cfg = _cfg()
    vars_cfg = cfg["vars"]
    if _is_wxcam_instrument(CURRENT_INSTRUMENT):
        if calendar_image_type.value != var1_select.value:
            calendar_image_type.value = var1_select.value
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


def _on_calendar_image_type_change(event):
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        return
    _refresh_ql_options(preserve_current=False)
    ql_date.param.trigger("value")


calendar_image_type.param.watch(_on_calendar_image_type_change, "value")


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


def _wxcam_daily_video_dir(image_type: str) -> Path:
    return _wxcam_daily_video_root() / image_type


def _wxcam_today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _wxcam_daily_video_options(selection: str) -> dict[str, str | None]:
    image_type = _image_type_from_selection(selection)
    day_dir = _wxcam_daily_video_dir(image_type)
    if not day_dir.exists():
        return {"No videos available": None}
    today_token = _wxcam_today_token()
    opts: dict[str, str | None] = {}
    for video_path in sorted(day_dir.glob("*.mp4")):
        if video_path.stem == today_token:
            continue
        opts[video_path.stem] = str(video_path)
    today_path = day_dir / f"{today_token}.mp4"
    if today_path.exists():
        opts["Today (latest)"] = str(today_path)
    return opts or {"No videos available": None}


def _wxcam_day_token_to_utc(day_token: str) -> str | None:
    if len(day_token) != 8 or not day_token.isdigit():
        return None
    return f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:8]}"


def _wxcam_calendar_day_token(selected_day: str | None) -> str | None:
    if selected_day == "Today (latest)":
        return _wxcam_today_token()
    return selected_day


def _wxcam_hourly_thumbnail_path(image_type: str, day_token: str, video_name: str) -> Path:
    return _wxcam_hourly_thumbnail_root() / image_type / day_token / f"{Path(video_name).stem}.jpg"


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
    mode_class = "wxcam-player--vertical" if image_type == "fish_hdr" else "wxcam-player--wide"
    title = f"{selection} | {selected_label} | {path.name}"
    return pn.Column(
        WxcamVideoPlayer(src=_video_data_uri(path), title=title, mode_class=mode_class, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


wxcam_image_type = pn.widgets.Select(
    name="Image type",
    options=list(_cfg("wxcam")["vars"].keys()),
    value=_cfg("wxcam")["default_top"],
)
_wxcam_ql_options = _wxcam_daily_video_options(wxcam_image_type.value)
wxcam_date = pn.widgets.Select(name="Date", options=list(_wxcam_ql_options.keys()))
if _wxcam_ql_options:
    wxcam_date.value = list(_wxcam_ql_options.keys())[-1]
wxcam_prev = pn.widgets.Button(name="<<", button_type="default")
wxcam_next = pn.widgets.Button(name=">>", button_type="default")


def _refresh_wxcam_ql_options(preserve_current: bool = True):
    global _wxcam_ql_options
    current = wxcam_date.value if preserve_current else None
    _wxcam_ql_options = _wxcam_daily_video_options(wxcam_image_type.value or _cfg("wxcam")["default_top"])
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


class WxcamCalendarState(param.Parameterized):
    selected_hour_path = param.String(default="")


wxcam_calendar_state = WxcamCalendarState()


def _refresh_wxcam_latest_if_needed():
    if CURRENT_INSTRUMENT != "wxcam":
        return
    if wxcam_date.value == "Today (latest)":
        # Refresh the current-day video option map slowly enough that playback
        # is stable while still letting new daily products appear automatically.
        global _wxcam_ql_options
        _wxcam_ql_options = _wxcam_daily_video_options(wxcam_image_type.value or _cfg("wxcam")["default_top"])
        wxcam_date.param.trigger("value")


_wxcam_ql_timer = pn.state.add_periodic_callback(_refresh_wxcam_latest_if_needed, period=300_000, start=True)


@pn.depends(wxcam_date.param.value, wxcam_image_type.param.value)
def _wxcam_interactive_media(selected, selection):
    path = _wxcam_daily_video_options(selection or _cfg("wxcam")["default_top"]).get(selected)
    if not path:
        return pn.pane.Markdown("No media available for this selection.")
    video_path = Path(path)
    if not video_path.exists():
        return pn.pane.Markdown("No media available for this selection.")
    return _build_wxcam_video_view(video_path, selection or _cfg("wxcam")["default_top"], selected)


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
    bottom = max(float(bottom_val), 0.0)
    top = max(float(top_val), bottom + 100.0)
    ds = open_window(start, end, bottom_m=bottom, top_m=top, instrument="Scanning Microwave Radiometer")
    cfg = _cfg("Scanning Microwave Radiometer")
    times = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if times is None or len(times) == 0:
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
    _show_plot(fig)


def _update_stacked_timeseries_view(instrument: str, start, end):
    """Render a 1D logger dataset as stacked time series, one row per variable."""
    print(f"[{instrument}] render window {start} -> {end}")
    ds = open_window(start, end, instrument=instrument)
    bg = "white"
    fg = "#222222"
    grid = "#dddddd"
    times = pd.to_datetime(ds["time"].values) if "time" in ds else None
    names = _numeric_time_vars(ds) if ds is not None else []
    if times is None or len(times) == 0 or not names:
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

    max_rows = len(names)
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
                        y=-0.08,
                        xref="x",
                        yref="paper",
                        text=t.strftime("%Y-%m-%d"),
                        showarrow=False,
                        xanchor="center",
                        yanchor="top",
                        font=dict(size=14, color=fg),
                    )
                )

    fig = make_subplots(
        rows=max_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.02, 0.6 / max(max_rows - 1, 1)),
    )
    colors = ["#0b7285", "#c92a2a", "#2b8a3e", "#5f3dc4", "#e67700", "#087f5b", "#364fc7", "#a61e4d"]
    for idx, name in enumerate(names, start=1):
        values = np.asarray(ds[name].values, dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=colors[(idx - 1) % len(colors)], width=1.4),
                hovertemplate=f"Time=%{{x}}<br>{name}=%{{y:.6g}}<extra></extra>",
                connectgaps=False,
            ),
            row=idx,
            col=1,
        )
        fig.update_yaxes(
            title_text=name,
            showgrid=True,
            gridcolor=grid,
            linecolor=fg,
            tickfont=dict(color=fg, size=9),
            title_font=dict(color=fg, size=9),
            row=idx,
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
    )
    fig.update_xaxes(
        title_text="Date and Time (UTC)",
        title_standoff=50,
        row=max_rows,
        col=1,
    )
    fig.update_layout(
        showlegend=False,
        height=max(650, min(4200, 76 * len(names) + 120)),
        margin=dict(l=115, r=35, t=30, b=95),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=fg, size=13),
        annotations=tuple(noon_annots),
    )
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
    if instrument == "Scanning Microwave Radiometer":
        _update_hatpro_view(start, end, bottom_val, top_val, lymin, lymax, iymin, iymax, rymin, rymax)
        return
    if _is_wxcam_instrument(instrument):
        _update_wxcam_view(start, end, var1_name, var2_name)
        return
    if _is_stacked_timeseries_instrument(instrument):
        _update_stacked_timeseries_view(instrument, start, end)
        return
    bottom = max(float(bottom_val), 0.0)
    top = max(float(top_val), bottom + 100.0)
    ds = open_window(start, end, bottom_m=bottom, top_m=top, instrument=instrument)
    cfg = _cfg()
    vars_cfg = cfg["vars"]
    var1 = vars_cfg.get(var1_name)
    var2 = vars_cfg.get(var2_name)
    # Simple light theme for plots
    bg = "white"
    fg = "#222222"
    grid = "#dddddd"
    if ds is None or not ds.data_vars:
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
    # Colorbar configs
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
            # Only plot depol where beta is above threshold.
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
    # Hourly ticks; add horizontal date annotations at 12:00.
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
            if t.hour == 12:  # add a date label at local noon
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
    # Keep both panels locked together when the user pans/zooms vertically.
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




# -------- Calendar quicklooks --------

def _quicklook_options(inst: str | None = None, wxcam_selection: str | None = None):
    """Build a mapping of label -> quicklook asset token/path."""
    inst = inst or CURRENT_INSTRUMENT
    cfg = _cfg(inst)
    if _is_wxcam_instrument(inst):
        return _wxcam_daily_video_options(wxcam_selection or _cfg("wxcam")["default_top"])
    quick_dir = cfg["quicklook_dir"]
    latest = cfg["latest_image"]
    opts = {}
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
    return opts or {"No images available": None}


_ql_options = _quicklook_options()
ql_date = pn.widgets.Select(name="Date", options=list(_ql_options.keys()))
if _ql_options:
    ql_date.value = list(_ql_options.keys())[-1]


def _refresh_ql_options(preserve_current: bool = True):
    """Refresh available quicklook options, optionally preserving current selection."""
    global _ql_options
    current = ql_date.value if preserve_current else None
    _ql_options = _quicklook_options(calendar_instrument.value, calendar_image_type.value)
    opts = list(_ql_options.keys())
    ql_date.options = opts
    if not opts:
        ql_date.value = None
        return
    if preserve_current and current in opts:
        ql_date.value = current
    elif _is_wxcam_instrument(calendar_instrument.value):
        historical = [label for label in opts if label != "Today (latest)"]
        ql_date.value = historical[-1] if historical else opts[-1]
    else:
        ql_date.value = opts[-1]

# Calendar navigation buttons
ql_prev = pn.widgets.Button(name="<<", button_type="default")
ql_next = pn.widgets.Button(name=">>", button_type="default")


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

# Periodically refresh the "Today (latest)" selection to pick up new PNGs.
def _refresh_latest_if_needed():
    """If viewing the latest image, reload the mapping and redraw without changing selection."""
    if ql_date.value == "Today (latest)":
        # Update the cached map so _quicklook_image sees fresh file paths,
        # but do not touch the selector options to avoid snapping UI.
        global _ql_options
        _ql_options = _quicklook_options(calendar_instrument.value, calendar_image_type.value)
        ql_date.param.trigger("value")


_ql_timer = pn.state.add_periodic_callback(_refresh_latest_if_needed, period=300_000, start=True)

# Ensure initial map is fresh
_refresh_ql_options(preserve_current=True)
_apply_instrument_defaults(CURRENT_INSTRUMENT, reset_time=True)


def _sync_wxcam_calendar_hour(*_events):
    if not _is_wxcam_instrument(calendar_instrument.value):
        wxcam_calendar_state.selected_hour_path = ""
        return
    selected_day = ql_date.value
    selection = calendar_image_type.value or _cfg("wxcam")["default_top"]
    day_token = _wxcam_calendar_day_token(selected_day)
    day_utc = _wxcam_day_token_to_utc(day_token or "")
    if not day_utc:
        wxcam_calendar_state.selected_hour_path = ""
        return
    image_type = _image_type_from_selection(selection)
    rows = records_for_day(_wxcam_catalog_path("wxcam"), image_type, day_utc, media_kind="video")
    available_paths = [str(row["raw_path"]) for row in rows]
    if not available_paths:
        wxcam_calendar_state.selected_hour_path = ""
        return
    if wxcam_calendar_state.selected_hour_path not in available_paths:
        wxcam_calendar_state.selected_hour_path = ""


calendar_instrument.param.watch(_sync_wxcam_calendar_hour, "value")
calendar_image_type.param.watch(_sync_wxcam_calendar_hour, "value")
ql_date.param.watch(_sync_wxcam_calendar_hour, "value")


def _build_wxcam_hour_tile(
    image_type: str,
    day_token: str,
    hour_index: int,
    row,
    selected_hour_path: str,
):
    hour_label = f"{hour_index:02d}:00"
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
            css_classes=["wxcam-hour-tile"],
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
        css_classes=["wxcam-hour-tile"],
        sizing_mode="stretch_width",
        margin=0,
    )


def _build_wxcam_calendar_day_view(selection: str, day_token: str, selected_hour_path: str):
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return pn.pane.Markdown("No hourly clips available for this selection.")
    image_type = _image_type_from_selection(selection)
    rows = records_for_day(_wxcam_catalog_path("wxcam"), image_type, day_utc, media_kind="video")
    if not rows:
        return pn.pane.Markdown("No hourly clips available for this selection.")

    rows_by_hour = {
        int(str(row["time_utc"])[11:13]): row
        for row in rows
    }
    tiles = [
        _build_wxcam_hour_tile(image_type, day_token, hour_index, rows_by_hour.get(hour_index), selected_hour_path)
        for hour_index in range(24)
    ]
    grid = pn.GridBox(*tiles, ncols=8, sizing_mode="stretch_width")
    selected_row = next((row for row in rows if str(row["raw_path"]) == selected_hour_path), None)
    if selected_row is None:
        return pn.Column(grid, sizing_mode="stretch_width")
    selected_hour_label = str(selected_row["time_utc"])[11:16] + " UTC"
    viewer = _build_wxcam_video_view(Path(str(selected_row["raw_path"])), selection, f"{day_token} | {selected_hour_label}")
    return pn.Column(grid, viewer, sizing_mode="stretch_width")


@pn.depends(
    ql_date.param.value,
    calendar_instrument.param.value,
    calendar_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
def _quicklook_image(selected, calendar_inst, wxcam_selection, selected_hour_path):
    """Show the selected quicklook asset (or a message if missing)."""
    instrument = calendar_inst or CURRENT_INSTRUMENT
    if _is_wxcam_instrument(instrument):
        selection = wxcam_selection or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(selected)
        return _build_wxcam_calendar_day_view(selection, day_token or "", selected_hour_path)
    # Use the latest map in case files changed since last refresh.
    path = _quicklook_options(instrument).get(selected)
    if path and Path(path).exists():
        return _media_pane(path)
    return pn.pane.Markdown("No image available for this selection.")


ACCENT = "#0b7285"  # header/accent color
css = """
# Global font override for a clean, consistent look.
body, .bk {
    font-family: "SF Pro Display","SF Pro","-apple-system","BlinkMacSystemFont","Segoe UI",sans-serif;
    font-size: 15px;
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
    max-height: 44px;
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
    min-height: 28px;
    border-radius: 2px;
    background: #e2e8f0;
    color: #475569;
    font-size: 10px;
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

# Template layout: header + tabs (Interactive, Calendar placeholder)
template = pn.template.MaterialTemplate(
    title="AURORA Data Viewer",
    header_background=ACCENT,
    header_color="white",
    main_max_width="1800px",  # wide but keeps a valid string
)

interactive_tab = pn.Column(controls, interactive_content, sizing_mode="stretch_both")
tabs = pn.Tabs(
    ("Interactive", interactive_tab),
    (
        "Calendar",
        pn.Column(
            pn.Card(
                pn.Row(calendar_instrument, calendar_image_type, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
                pn.Row(ql_prev, ql_date, ql_next, sizing_mode="stretch_width"),
                title="",
                collapsible=False,
                sizing_mode="stretch_width",
            ),
            _quicklook_image,
            sizing_mode="stretch_both",
        ),
    ),
    sizing_mode="stretch_both",
)

template.main[:] = [tabs]

def _apply_theme(dark: bool):
    """No-op placeholder (dark mode removed)."""
    return

# Serve the app
template.servable()
