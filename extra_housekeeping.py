#!/usr/bin/env python3
"""Housekeeping quicklook helpers for non-summary Aurora instruments."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wxcam_catalog import ensure_schema, open_catalog

EXTRA_HK_SPECS = {
    "Ceilometer": {"label": "HK_Ceilometer", "prefix": "ceilometer"},
    "Cloud Radar": {"label": "HK_Radar", "prefix": "cloud_radar"},
    "wxcam": {"label": "HK_WXcam", "prefix": "wxcam"},
}

RANGE_MAX = 9000


def extra_housekeeping_label(instrument: str) -> str | None:
    spec = EXTRA_HK_SPECS.get(instrument)
    return None if spec is None else str(spec["label"])


def extra_housekeeping_latest_png(quicklook_dir: Path, instrument: str) -> Path | None:
    spec = EXTRA_HK_SPECS.get(instrument)
    if spec is None:
        return None
    key = str(spec["label"]).lower()
    return quicklook_dir / f"{spec['prefix']}__{key}__latest.png"


def extra_housekeeping_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path | None:
    spec = EXTRA_HK_SPECS.get(instrument)
    if spec is None:
        return None
    key = str(spec["label"]).lower()
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{spec['prefix']}__{key}__{stamp}.png"


def extra_housekeeping_tokens(quicklook_dir: Path, instrument: str) -> list[str]:
    spec = EXTRA_HK_SPECS.get(instrument)
    if spec is None:
        return []
    key = str(spec["label"]).lower()
    tokens: list[str] = []
    for png in sorted(quicklook_dir.glob(f"{spec['prefix']}__{key}__*.png")):
        suffix = png.stem.rsplit("__", 1)[-1]
        if suffix == "latest":
            continue
        tokens.append(suffix)
    return tokens


def _apply_time_axis(ax, times: pd.DatetimeIndex) -> None:
    if len(times) == 0:
        return
    span_hours = max((times.max() - times.min()) / np.timedelta64(1, "h"), 1.0)
    interval = 1 if span_hours <= 18 else 2 if span_hours <= 36 else 6
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)


def _pick_range_coord(ds: xr.Dataset) -> str:
    for cand in ("range", "height", "altitude", "distance"):
        if cand in ds.coords:
            return cand
    raise KeyError("No range/height-like coordinate found")


def _has_finite(data: np.ndarray) -> bool:
    try:
        return bool(np.isfinite(np.asarray(data, dtype=np.float64)).any())
    except Exception:
        return False


def plot_ceilometer_housekeeping(ds: xr.Dataset, title: str, output: Path) -> None:
    time_index = pd.DatetimeIndex(ds["time"].values)
    panels: list[tuple[str, str | None, list[tuple[str, np.ndarray, bool]]]] = []

    series_specs = [
        ("receiver_gain", "Receiver Gain", None, False),
        ("beta_att_noise_level", "Backscatter Noise Level", None, False),
        ("beta_att_sum", "Backscatter Sum", None, False),
        ("vertical_visibility", "Vertical Visibility [m]", "m", False),
        ("height_offset", "Height Offset [m]", "m", False),
        ("tilt_angle", "Tilt Angle [deg]", "deg", False),
        ("tilt_correction", "Tilt Correction", None, True),
        ("sky_condition_total_cloud_cover", "Total Cloud Cover", None, True),
        ("fog_detection", "Fog Detection", None, True),
        ("precipitation_detection", "Precipitation Detection", None, True),
    ]
    layer_specs = [
        ("cloud_base_heights", "Cloud Base Heights [m]", "m"),
        ("cloud_thickness", "Cloud Thickness [m]", "m"),
        ("cloud_penetration_depth", "Cloud Penetration Depth [m]", "m"),
        ("sky_condition_cloud_layer_heights", "Sky Condition Layer Heights [m]", "m"),
        ("sky_condition_cloud_layer_covers", "Sky Condition Layer Covers", None),
    ]

    for name, label, unit, step in series_specs:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values, dtype=np.float64)
        if not _has_finite(values):
            continue
        panels.append((label, unit, [(label, values, step)]))

    for name, label, unit in layer_specs:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values, dtype=np.float64)
        if values.ndim != 2 or not _has_finite(values):
            continue
        traces = []
        for idx in range(values.shape[1]):
            layer_values = values[:, idx]
            if not _has_finite(layer_values):
                continue
            traces.append((f"Layer {idx + 1}", layer_values, False))
        if traces:
            panels.append((label, unit, traces))

    if len(time_index) == 0 or not panels:
        raise ValueError("No ceilometer housekeeping variables available")

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, max(8.5, 1.9 * len(panels))), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = ["#2bb3b1", "#b52020", "#4b66c4", "#7a52c7", "#aa5a2a", "#c43aa7"]
    for ax, (label, unit, traces) in zip(axes, panels):
        for idx, (trace_label, values, step) in enumerate(traces):
            finite = np.isfinite(values)
            if not finite.any():
                continue
            drawstyle = "steps-post" if step else "default"
            ax.plot(
                time_index[finite],
                values[finite],
                linewidth=1.0,
                color=colors[idx % len(colors)],
                drawstyle=drawstyle,
                label=trace_label,
            )
        ax.set_ylabel(unit or "", fontsize=8)
        ax.set_title(label, loc="left", fontsize=10, pad=2)
        ax.grid(True, color="#dddddd", linewidth=0.45)
        ax.tick_params(axis="y", labelsize=8)
        if len(traces) > 1:
            ax.legend(loc="upper right", fontsize=7, ncol=min(3, len(traces)))
    for ax in axes:
        _apply_time_axis(ax, time_index)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.06, hspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_cloud_radar_housekeeping(ds: xr.Dataset, title: str, output: Path) -> None:
    vars_titles = [
        ("ZE45_dBZ", "ZE45 (dBZ)", -30.0, 10.0, "ZE45 (dBZ)", "cividis"),
        ("ZDR", "ZDR (dB)", -10.0, 6.0, "ZDR (dB)", "RdBu_r"),
        ("PhiDP", "PhiDP (rad)", -2.0, 2.0, "PhiDP (rad)", "RdBu_r"),
        ("KDP", "KDP (rad/km)", -4.0, 4.0, "KDP (rad/km)", "RdBu_r"),
        ("DiffAtt", "Differential Attenuation (dB/km)", -5.0, 5.0, "DiffAtt (dB/km)", "RdBu_r"),
    ]
    available = [(var, text, vmin, vmax, cbar, cmap) for var, text, vmin, vmax, cbar, cmap in vars_titles if var in ds]
    if not available or ds.sizes.get("time", 0) == 0:
        raise ValueError("No cloud radar housekeeping variables available")
    ds = ds.sel({"range": slice(0, RANGE_MAX)})
    fig, axes = plt.subplots(len(available), 1, figsize=(12, max(9.0, 3.1 * len(available))), sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (var, label, vmin, vmax, cbar_label, cmap) in zip(axes, available):
        da = ds[var].transpose("time", "range")
        mesh = ax.pcolormesh(
            da["time"].values,
            da["range"].values,
            da.values.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label(cbar_label)
        ax.set_ylabel("Range (m)")
        ax.set_title(label, loc="left", fontsize=10, pad=2)
        ax.set_ylim(0, RANGE_MAX)
    times = pd.DatetimeIndex(ds["time"].values)
    for ax in axes:
        _apply_time_axis(ax, times)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97, bottom=0.08, hspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def _wxcam_query_dataframe(catalog_path: Path, where_sql: str, params: tuple[object, ...]) -> pd.DataFrame:
    if not catalog_path.exists():
        return pd.DataFrame()
    with open_catalog(catalog_path) as conn:
        ensure_schema(conn)
        query = f"""
            SELECT time_utc, time_epoch_ns, day_utc, image_type, media_kind, size_bytes
            FROM images
            WHERE {where_sql}
            ORDER BY time_epoch_ns ASC
        """
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time_utc"], utc=False)
    df["hour"] = df["time"].dt.floor("h")
    df["hour_of_day"] = df["time"].dt.hour
    df["minute_offset"] = df["time"].dt.minute + df["time"].dt.second / 60.0 - 30.0
    df["size_mb"] = df["size_bytes"] / (1024.0 * 1024.0)
    return df


def _wxcam_hourly_series(frame: pd.DataFrame, value_col: str, agg: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["hour", "image_type"])[value_col]
    if agg == "count":
        out = grouped.count().unstack(fill_value=0)
    elif agg == "median":
        out = grouped.median().unstack()
    else:
        raise ValueError(f"Unsupported aggregate {agg}")
    return out.sort_index()


def _wxcam_representative_offset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    chosen_rows = []
    for (_hour, image_type), group in frame.groupby(["hour", "image_type"]):
        chosen_rows.append(group.iloc[(group["minute_offset"].abs()).argmin()])
    rep = pd.DataFrame(chosen_rows)
    if rep.empty:
        return pd.DataFrame()
    return rep.pivot(index="hour", columns="image_type", values="minute_offset").sort_index()


def _plot_wxcam_housekeeping_frame(df: pd.DataFrame, title: str, output: Path) -> None:
    if df.empty:
        raise ValueError("No WXcam housekeeping rows available")
    image_counts = _wxcam_hourly_series(df[df["media_kind"] == "image"], "time_epoch_ns", "count")
    video_counts = _wxcam_hourly_series(df[df["media_kind"] == "video"], "time_epoch_ns", "count")
    image_sizes = _wxcam_hourly_series(df[df["media_kind"] == "image"], "size_mb", "median")
    image_offsets = _wxcam_representative_offset(df[df["media_kind"] == "image"])

    panels = [
        ("HDR Image Count per Hour", "Count", image_counts),
        ("HDR Video Count per Hour", "Count", video_counts),
        ("Median HDR Image Size [MB]", "MB", image_sizes),
        ("Representative Image Offset from :30 [min]", "Minutes", image_offsets),
    ]
    colors = {"fish_hdr": "#2bb3b1", "pano_hdr": "#7a52c7"}
    labels = {"fish_hdr": "FISH HDR", "pano_hdr": "PANO HDR"}
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 10.5), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (panel_title, y_label, frame) in zip(axes, panels):
        if frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#555555")
        else:
            for image_type in ("fish_hdr", "pano_hdr"):
                if image_type not in frame.columns:
                    continue
                series = frame[image_type]
                finite = np.isfinite(series.values.astype(float))
                if not finite.any():
                    continue
                ax.plot(
                    frame.index[finite],
                    series.values[finite],
                    marker="o",
                    linewidth=1.2,
                    markersize=3.0,
                    color=colors[image_type],
                    label=labels[image_type],
                )
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_title(panel_title, loc="left", fontsize=10, pad=2)
        ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True, color="#dddddd", linewidth=0.45)
        ax.tick_params(axis="y", labelsize=8)
    times = pd.DatetimeIndex(df["hour"].drop_duplicates().sort_values())
    for ax in axes:
        _apply_time_axis(ax, times)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.08, hspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_wxcam_housekeeping_latest(catalog_path: Path, title: str, output: Path, lookback_hours: int = 24) -> None:
    end = pd.Timestamp.utcnow().to_pydatetime()
    start = end - timedelta(hours=lookback_hours)
    start_ns = int(pd.Timestamp(start).value)
    end_ns = int(pd.Timestamp(end).value)
    df = _wxcam_query_dataframe(
        catalog_path,
        "time_epoch_ns >= ? AND time_epoch_ns <= ? AND media_kind IN ('image', 'video')",
        (start_ns, end_ns),
    )
    _plot_wxcam_housekeeping_frame(df, title, output)


def plot_wxcam_housekeeping_day(catalog_path: Path, day_utc: str, title: str, output: Path) -> None:
    df = _wxcam_query_dataframe(
        catalog_path,
        "day_utc = ? AND media_kind IN ('image', 'video')",
        (day_utc,),
    )
    _plot_wxcam_housekeeping_frame(df, title, output)
