#!/usr/bin/env python3
"""Housekeeping quicklook helpers for non-summary Aurora instruments."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wxcam_catalog import WXCAM_IMAGE_TYPES, ensure_schema, open_catalog

EXTRA_HK_SPECS = {
    "Ceilometer": {"label": "HK_Ceilometer", "prefix": "ceilometer"},
    "Cloud Radar": {"label": "HK_Radar", "prefix": "cloud_radar"},
    "wxcam": {"label": "HK_WXcam", "prefix": "wxcam"},
}

RANGE_MAX = 9000
RADAR_TIME_ZERO = np.datetime64("2001-01-01T00:00:00")
RADAR_NC_REGEX = re.compile(r"_(\d{6})_(\d{6})")


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


def pick_range_coord(ds: xr.Dataset) -> str:
    for cand in ("range", "height", "altitude", "distance"):
        if cand in ds.coords:
            return cand
    raise KeyError("No range/height-like coordinate found")


def _has_finite(data: np.ndarray) -> bool:
    try:
        return bool(np.isfinite(np.asarray(data, dtype=np.float64)).any())
    except Exception:
        return False


def _sample_interval_seconds(time_index: pd.DatetimeIndex) -> np.ndarray:
    intervals = np.full(len(time_index), np.nan, dtype=np.float64)
    if len(time_index) > 1:
        intervals[1:] = np.diff(time_index.asi8.astype(np.float64)) / 1.0e9
    return intervals


def _downsample_frame(times: pd.DatetimeIndex, arrays: dict[str, np.ndarray], max_samples: int = 2500) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    if len(times) <= max_samples:
        return times, arrays
    step = int(np.ceil(len(times) / max_samples))
    return times[::step], {name: values[::step] for name, values in arrays.items()}


def _plot_grouped_housekeeping(
    time_index: pd.DatetimeIndex,
    panels: list[dict[str, object]],
    title: str,
    output: Path,
    max_samples: int = 2500,
) -> None:
    if len(time_index) == 0 or not panels:
        raise ValueError("No housekeeping variables available")

    arrays: dict[str, np.ndarray] = {}
    active_panels: list[dict[str, object]] = []
    for panel in panels:
        traces = []
        for trace in panel["traces"]:
            name = trace["name"]
            values = np.asarray(trace["values"], dtype=np.float64)
            if not _has_finite(values):
                continue
            arrays[name] = values
            traces.append(trace)
        if traces:
            active = dict(panel)
            active["traces"] = traces
            active_panels.append(active)

    if not active_panels:
        raise ValueError("No finite housekeeping variables available")

    time_index, arrays = _downsample_frame(time_index, arrays, max_samples=max_samples)
    fig, axes = plt.subplots(
        len(active_panels),
        1,
        figsize=(14, max(8.0, 2.35 * len(active_panels))),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, panel in zip(axes, active_panels, strict=False):
        right_ax = ax.twinx() if panel.get("right_label") else None
        left_color = "#22313f"
        right_color = "#22313f"
        for trace in panel["traces"]:
            values = arrays[trace["name"]]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            target = right_ax if trace.get("axis") == "right" and right_ax is not None else ax
            color = trace.get("color", "#0b7285")
            drawstyle = "steps-post" if trace.get("step") else "default"
            target.plot(
                time_index[finite],
                values[finite],
                color=color,
                linewidth=1.05,
                drawstyle=drawstyle,
                label=trace.get("label", trace["name"]),
            )
            if target is right_ax:
                right_color = color
            else:
                left_color = color
        ax.set_ylabel(str(panel.get("left_label") or ""), color=left_color, fontsize=9)
        ax.tick_params(axis="y", colors=left_color, labelsize=8)
        if right_ax is not None:
            right_ax.set_ylabel(str(panel.get("right_label") or ""), color=right_color, fontsize=9)
            right_ax.tick_params(axis="y", colors=right_color, labelsize=8)
        ax.grid(True, color="#e5eaef", linewidth=0.5)
        ax.text(
            0.01,
            0.94,
            str(panel["title"]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#22313f", linewidth=0.9, boxstyle="square,pad=0.25"),
        )
        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = right_ax.get_legend_handles_labels() if right_ax is not None else ([], [])
        handles = handles_left + handles_right
        labels = labels_left + labels_right
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=1)

    for ax in axes:
        _apply_time_axis(ax, time_index)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.91, top=0.95, bottom=0.07, hspace=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_ceilometer_housekeeping(ds: xr.Dataset, title: str, output: Path) -> None:
    time_index = pd.DatetimeIndex(ds["time"].values)
    panels = [
        {
            "title": "Sample Cadence",
            "left_label": "Interval [s]",
            "traces": [
                {"name": "sample_interval_s", "label": "Sample Interval", "values": _sample_interval_seconds(time_index), "color": "#0b7285"},
            ],
        },
        {
            "title": "Receiver / Signal",
            "left_label": "Receiver Gain",
            "right_label": "Signal Diagnostic",
            "traces": [
                {"name": "receiver_gain", "label": "Receiver Gain", "values": ds["receiver_gain"].values, "color": "#4f8c63", "step": True}
                if "receiver_gain" in ds
                else None,
                {"name": "beta_att_noise_level", "label": "Noise Level", "values": ds["beta_att_noise_level"].values, "color": "#c05647", "axis": "right"}
                if "beta_att_noise_level" in ds
                else None,
                {"name": "beta_att_sum", "label": "Backscatter Sum", "values": ds["beta_att_sum"].values, "color": "#4d6fb3", "axis": "right"}
                if "beta_att_sum" in ds
                else None,
            ],
        },
        {
            "title": "Alignment / Reference",
            "left_label": "Tilt [deg]",
            "right_label": "Offset [m]",
            "traces": [
                {"name": "tilt_angle", "label": "Tilt Angle", "values": ds["tilt_angle"].values, "color": "#7768b8"}
                if "tilt_angle" in ds
                else None,
                {"name": "height_offset", "label": "Height Offset", "values": ds["height_offset"].values, "color": "#4f7d8d", "axis": "right"}
                if "height_offset" in ds
                else None,
                {"name": "tilt_correction", "label": "Tilt Correction", "values": ds["tilt_correction"].values, "color": "#718195", "axis": "right", "step": True}
                if "tilt_correction" in ds
                else None,
            ],
        },
        {
            "title": "Weather Flags",
            "left_label": "State",
            "right_label": "Cloud Cover [oktas]",
            "traces": [
                {"name": "precipitation_detection", "label": "Precipitation", "values": ds["precipitation_detection"].values, "color": "#c05647", "step": True}
                if "precipitation_detection" in ds
                else None,
                {"name": "fog_detection", "label": "Fog", "values": ds["fog_detection"].values, "color": "#7768b8", "step": True}
                if "fog_detection" in ds
                else None,
                {"name": "sky_condition_total_cloud_cover", "label": "Cloud Cover", "values": ds["sky_condition_total_cloud_cover"].values, "color": "#0b7285", "axis": "right", "step": True}
                if "sky_condition_total_cloud_cover" in ds
                else None,
            ],
        },
        {
            "title": "Vertical Visibility",
            "left_label": "Visibility [m]",
            "traces": [
                {"name": "vertical_visibility", "label": "Vertical Visibility", "values": ds["vertical_visibility"].values, "color": "#0b7285"}
                if "vertical_visibility" in ds
                else None,
            ],
        },
    ]
    for panel in panels:
        panel["traces"] = [trace for trace in panel["traces"] if trace is not None]
    _plot_grouped_housekeeping(time_index, panels, title, output)


def _parse_radar_file_time(path: Path) -> pd.Timestamp | None:
    match = RADAR_NC_REGEX.search(path.name)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return pd.to_datetime(date_part + time_part, format="%y%m%d%H%M%S")
    except ValueError:
        return None


def _radar_raw_files(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    files: list[tuple[pd.Timestamp, Path]] = []
    scan_start = start - pd.Timedelta(hours=2)
    for path in root.rglob("*.NC"):
        if not path.name.upper().endswith("LV1.NC"):
            continue
        stamp = _parse_radar_file_time(path)
        if stamp is None:
            continue
        if scan_start <= stamp <= end:
            files.append((stamp, path))
    files.sort(key=lambda item: item[0])
    return [path for _stamp, path in files]


def _load_radar_hk_file(path: Path) -> xr.Dataset:
    with xr.open_dataset(path, decode_times=False) as raw:
        time = RADAR_TIME_ZERO + raw["Time"].astype("timedelta64[s]") + raw["Timems"].astype("timedelta64[ms]")
        time_vals = np.asarray(time.values)
        names = [
            "Rain",
            "SurfRelHum",
            "SurfTemp",
            "SurfPres",
            "SurfWS",
            "SurfWD",
            "DDVolt",
            "DDTb",
            "LWP",
            "PowIF",
            "Elv",
            "Azm",
            "Status",
            "TPow",
            "TTemp",
            "RTemp",
            "PCTemp",
            "CBH",
            "Inc_El",
            "Inc_ElA",
        ]
        data_vars = {}
        for name in names:
            if name not in raw:
                continue
            data_vars[name] = (("time",), np.asarray(raw[name].values, dtype=np.float32))
    if not data_vars:
        return xr.Dataset()
    return xr.Dataset(data_vars, coords={"time": time_vals}).sortby("time")


def load_cloud_radar_housekeeping_from_raw(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    datasets = []
    for path in _radar_raw_files(root, start, end):
        try:
            ds = _load_radar_hk_file(path)
        except Exception as exc:
            print(f"Skipping unreadable radar housekeeping file {path}: {exc}")
            continue
        if ds.sizes.get("time", 0):
            datasets.append(ds)
    if not datasets:
        return xr.Dataset()
    combined = xr.concat(datasets, dim="time", join="outer").sortby("time")
    times = np.asarray(combined["time"].values)
    _, unique_idx = np.unique(times, return_index=True)
    if len(unique_idx) != len(times):
        combined = combined.isel(time=np.sort(unique_idx))
    mask = (pd.DatetimeIndex(combined["time"].values) >= start) & (pd.DatetimeIndex(combined["time"].values) <= end)
    return combined.isel(time=mask)


def plot_cloud_radar_housekeeping(ds: xr.Dataset, title: str, output: Path) -> None:
    if ds.sizes.get("time", 0) == 0:
        raise ValueError("No cloud radar housekeeping variables available")
    time_index = pd.DatetimeIndex(ds["time"].values)

    def trace(name: str, label: str, color: str, axis: str = "left", step: bool = False) -> dict[str, object] | None:
        if name not in ds:
            return None
        return {"name": name, "label": label, "values": ds[name].values, "color": color, "axis": axis, "step": step}

    panels = [
        {
            "title": "Sample Cadence / Status",
            "left_label": "Interval [s]",
            "right_label": "State",
            "traces": [
                {"name": "sample_interval_s", "label": "Sample Interval", "values": _sample_interval_seconds(time_index), "color": "#0b7285"},
                trace("Status", "Radar Status", "#c05647", axis="right", step=True),
            ],
        },
        {
            "title": "Power / Receiver Chain",
            "left_label": "Voltage [V]",
            "right_label": "IF / Brightness",
            "traces": [
                trace("DDVolt", "DD Voltage", "#4f8c63"),
                trace("PowIF", "IF Power", "#7768b8", axis="right"),
                trace("DDTb", "DD Brightness Temp", "#4d6fb3", axis="right"),
            ],
        },
        {
            "title": "Thermal State",
            "left_label": "Temperature [C]",
            "right_label": "Transmitter Power",
            "traces": [
                trace("TTemp", "Transmitter Temp", "#c05647"),
                trace("RTemp", "Receiver Temp", "#0b7285"),
                trace("PCTemp", "PC Temp", "#7768b8"),
                trace("TPow", "Transmitter Power", "#4f8c63", axis="right"),
            ],
        },
        {
            "title": "Antenna Pointing",
            "left_label": "Elevation [deg]",
            "right_label": "Azimuth / Inclination [deg]",
            "traces": [
                trace("Elv", "Elevation", "#0b7285"),
                trace("Azm", "Azimuth", "#4f7d8d", axis="right"),
                trace("Inc_El", "Inclination Elevation", "#7768b8", axis="right"),
                trace("Inc_ElA", "Inclination Elevation A", "#718195", axis="right"),
            ],
        },
        {
            "title": "Surface Met At Radar",
            "left_label": "Temperature [C] / Wind [m s^-1]",
            "right_label": "RH [%] / Pressure [hPa]",
            "traces": [
                trace("SurfTemp", "Surface Temp", "#c05647"),
                trace("SurfWS", "Surface Wind Speed", "#0b7285"),
                trace("SurfRelHum", "Surface RH", "#7768b8", axis="right"),
                trace("SurfPres", "Surface Pressure", "#4f8c63", axis="right"),
            ],
        },
        {
            "title": "Ancillary Retrievals",
            "left_label": "Rain / LWP",
            "right_label": "Cloud Base [m]",
            "traces": [
                trace("Rain", "Rain", "#0b7285", step=True),
                trace("LWP", "Liquid Water Path", "#7768b8"),
                trace("CBH", "Cloud Base Height", "#4f8c63", axis="right"),
            ],
        },
    ]
    for panel in panels:
        panel["traces"] = [item for item in panel["traces"] if item is not None]
    _plot_grouped_housekeeping(time_index, panels, title, output)


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
    for (_hour, _image_type), group in frame.groupby(["hour", "image_type"]):
        chosen_rows.append(group.iloc[(group["minute_offset"].abs()).argmin()])
    rep = pd.DataFrame(chosen_rows)
    if rep.empty:
        return pd.DataFrame()
    return rep.pivot(index="hour", columns="image_type", values="minute_offset").sort_index()


def _plot_wxcam_housekeeping_frame(df: pd.DataFrame, title: str, output: Path) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(13, 3.0))
        ax.text(
            0.5,
            0.5,
            "No WXcam housekeeping rows available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#555555",
        )
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"Wrote {output} with no WXcam rows")
        return
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
    palette = ["#2bb3b1", "#7a52c7", "#b5651d", "#4968b3"]
    colors = {image_type: palette[idx % len(palette)] for idx, image_type in enumerate(WXCAM_IMAGE_TYPES)}
    labels = {image_type: spec["label"] for image_type, spec in WXCAM_IMAGE_TYPES.items()}
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 10.5), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (panel_title, y_label, frame) in zip(axes, panels, strict=False):
        if frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#555555")
        else:
            for image_type in WXCAM_IMAGE_TYPES:
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
