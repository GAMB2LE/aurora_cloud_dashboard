#!/usr/bin/env python3
"""Shared grouping and plotting helpers for 1D Aurora time-series instruments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

MAX_TIME_SAMPLES = 2200


@dataclass(frozen=True)
class TimeseriesGroup:
    key: str
    label: str
    variables: tuple[str, ...] | None = None


GROUPS: dict[str, tuple[TimeseriesGroup, ...]] = {
    "vaisalamet": (
        TimeseriesGroup("hk_met", "HK_Met"),
        TimeseriesGroup(
            "core_met",
            "Core Met",
            ("baro_hPa", "h1_t", "h1_rh", "h1_td", "h1_ah", "h1_mr", "h1_e", "t2_t"),
        ),
        TimeseriesGroup("probe_compare", "Probe Comparison", ("h1_t", "t2_t")),
        TimeseriesGroup(
            "sensor_status",
            "Sensor Status",
            (
                "h1_online",
                "h1_error_status",
                "h1_dev_critical_error",
                "h1_dev_warning",
                "h1_dev_notification",
                "t2_online",
                "t2_error_status",
                "t2_dev_critical_error",
                "t2_dev_warning",
                "t2_dev_notification",
                "baro_st_sensor_failure",
                "baro_st_value_locked",
                "baro_err_pressure_meas_err",
                "baro_err_pressure_oor",
            ),
        ),
    ),
    "asfs-logger": (
        TimeseriesGroup("hk_asfs", "HK_ASFS"),
        TimeseriesGroup(
            "power_hk",
            "Power/HK",
            ("batt_volt_Avg", "amp_meter_48vdc_Avg", "watts_on_48vdc_Avg", "PTemp_Avg", "scantime"),
        ),
        TimeseriesGroup(
            "wind_t",
            "Wind/T",
            ("metek_x_out_Avg", "metek_y_out_Avg", "metek_z_out_Avg", "metek_T_out_Avg"),
        ),
        TimeseriesGroup(
            "radiation_surface",
            "Radiation/Surface",
            ("spn1_tot_Avg", "spn1_dif_Avg", "sr50_dist_Avg", "sr50_qc_Avg", "kt15_amb_Avg", "kt15_tem_Avg"),
        ),
        TimeseriesGroup(
            "licor",
            "LICOR",
            ("licor_co2_out_Avg", "licor_h2o_out_Avg", "licor_t_out_Avg", "licor_co2_str_out_Avg"),
        ),
    ),
    "asfs-fast-sonic": (
        TimeseriesGroup("wind_components", "Wind Components", ("metek_x_out", "metek_y_out", "metek_z_out")),
        TimeseriesGroup(
            "tilt_temperature",
            "Tilt/Temperature",
            ("metek_InclX_out", "metek_InclY_out", "metek_T_out"),
        ),
        TimeseriesGroup(
            "quality",
            "Quality",
            ("metek_quality_out", "metek_senspathstate_out"),
        ),
    ),
    "power": (
        TimeseriesGroup("hk_aps", "HK_APS"),
        TimeseriesGroup(
            "ac_output",
            "AC Output",
            ("ACOutputVolts", "ACOutputAmps", "ACOutputWatts", "ACOutputHZ", "ACkWh", "ACnHours"),
        ),
        TimeseriesGroup(
            "battery_dc",
            "Battery/DC",
            (
                "BatteryAmps",
                "BatteryWatts",
                "BatteryState",
                "BattsOnline",
                "DCInverterAmps",
                "DCInverterVolts",
                "DCInverterWatts",
                "TotCapacity",
            ),
        ),
        TimeseriesGroup(
            "solar",
            "Solar",
            (
                "SolarWatts_East",
                "SolarWatts_South",
                "SolarWatts_West",
                "SolarVolts_East",
                "SolarVolts_South",
                "SolarVolts_West",
                "SolarAmps_East",
                "SolarAmps_South",
                "SolarAmps_West",
                "SolarYield_East",
                "SolarYield_South",
                "SolarYield_West",
                "MaxSolarWatts_East",
                "MaxSolarWatts_South",
                "MaxSolarWatts_West",
            ),
        ),
        TimeseriesGroup(
            "thermal_status",
            "Thermal/Status",
            (
                "InternalTemperature",
                "HeatsinkTemperature",
                "TempSensor1",
                "TempSensor2",
                "TempSensor3",
                "TempSensor4",
                "AlarmBits",
                "FaultBits",
                "HeatsinkTempAlarm",
                "InternalTempAlarm",
                "time_discrepancy",
            ),
        ),
    ),
}

DEFAULT_INTERACTIVE_GROUP_KEY = {
    "vaisalamet": "core_met",
    "asfs-logger": "wind_t",
    "asfs-fast-sonic": "wind_components",
    "power": "ac_output",
}

DEFAULT_CALENDAR_GROUP_KEY = {
    "vaisalamet": "hk_met",
    "asfs-logger": "hk_asfs",
    "asfs-fast-sonic": "wind_components",
    "power": "hk_aps",
}

QUICKLOOK_PREFIX = {
    "vaisalamet": "vaisalamet",
    "asfs-logger": "asfs_logger",
    "asfs-fast-sonic": "asfs_fast_sonic",
    "power": "power",
}

LEGACY_ALIAS_PREFIX = {
    "vaisalamet": "vaisalamet",
    "asfs-logger": "asfs_logger",
    "power": "power",
}

STATUS_TOKENS = (
    "alarm",
    "bits",
    "critical_error",
    "dev_",
    "discrepancy",
    "err_",
    "error",
    "failure",
    "locked",
    "not_available",
    "not_ready",
    "not_reliable",
    "online",
    "over_range",
    "qc",
    "quality",
    "senspathstate",
    "sensor_fail",
    "sensor_failure",
    "state",
    "status",
    "under_range",
    "warning",
)


def group_specs(instrument: str) -> tuple[TimeseriesGroup, ...]:
    return GROUPS.get(instrument, ())


def group_spec_for_selection(instrument: str, selection: str | None) -> TimeseriesGroup:
    specs = group_specs(instrument)
    if not specs:
        raise KeyError(f"No grouped time-series config for instrument {instrument}")
    if selection:
        for spec in specs:
            if selection in {spec.key, spec.label}:
                return spec
    default_key = DEFAULT_INTERACTIVE_GROUP_KEY.get(instrument, specs[0].key)
    for spec in specs:
        if spec.key == default_key:
            return spec
    return specs[0]


def default_interactive_label(instrument: str) -> str:
    return group_spec_for_selection(instrument, DEFAULT_INTERACTIVE_GROUP_KEY.get(instrument)).label


def default_calendar_label(instrument: str) -> str:
    key = DEFAULT_CALENDAR_GROUP_KEY.get(instrument, DEFAULT_INTERACTIVE_GROUP_KEY.get(instrument))
    return group_spec_for_selection(instrument, key).label


def widget_group_options(instrument: str) -> OrderedDict[str, dict[str, object]]:
    return OrderedDict(
        (
            spec.label,
            {
                "label": spec.label,
                "group_key": spec.key,
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            },
        )
        for spec in group_specs(instrument)
    )


def numeric_time_vars(ds: xr.Dataset) -> list[str]:
    names: list[str] = []
    for name, da in ds.data_vars.items():
        if da.dims != ("time",):
            continue
        if name == "RECORD":
            continue
        if np.issubdtype(da.dtype, np.number):
            names.append(name)
    return names


def grouped_numeric_time_vars(ds: xr.Dataset, instrument: str, selection: str | None) -> list[str]:
    names = numeric_time_vars(ds)
    spec = group_spec_for_selection(instrument, selection)
    if spec.variables is None:
        return names
    available = set(names)
    return [name for name in spec.variables if name in available]


def downsample_time(ds: xr.Dataset, max_time_samples: int = MAX_TIME_SAMPLES) -> xr.Dataset:
    if "time" not in ds:
        return ds
    count = ds.sizes.get("time", 0)
    if count > max_time_samples:
        step = int(np.ceil(count / max_time_samples))
        ds = ds.isel(time=slice(None, None, step))
    return ds


def is_status_like_var(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in STATUS_TOKENS)


def _plot_title(instrument: str, selection: str | None, suffix: str) -> str:
    label = group_spec_for_selection(instrument, selection).label
    return f"{label} - {suffix}"


def plot_grouped_timeseries(
    ds: xr.Dataset,
    instrument: str,
    selection: str | None,
    title: str,
    output: Path,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> list[str]:
    ds = downsample_time(ds, max_time_samples=max_time_samples)
    times = pd.to_datetime(ds["time"].values) if "time" in ds else pd.DatetimeIndex([])
    names = grouped_numeric_time_vars(ds, instrument, selection)
    if len(times) == 0 or not names:
        raise ValueError(f"No numeric {instrument} time-series variables available for {selection}")

    max_height = 42.0 if instrument == "power" else 34.0
    per_var = 1.0 if instrument == "power" else 1.15
    height = max(8.0, min(max_height, per_var * len(names)))
    fig, axes = plt.subplots(len(names), 1, figsize=(13, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = ["#0b7285", "#c92a2a", "#2b8a3e", "#5f3dc4", "#e67700", "#087f5b", "#364fc7", "#a61e4d"]
    for idx, (ax, name) in enumerate(zip(axes, names)):
        values = np.asarray(ds[name].values, dtype=np.float64)
        drawstyle = "steps-post" if is_status_like_var(name) else "default"
        ax.plot(times, values, color=colors[idx % len(colors)], linewidth=0.8, drawstyle=drawstyle)
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        ax.grid(True, color="#d9d9d9", linewidth=0.4)
        ax.tick_params(axis="y", labelsize=7)

    span_hours = max((times.max() - times.min()) / np.timedelta64(1, "h"), 1.0)
    interval = 1 if span_hours <= 12 else 2 if span_hours <= 36 else 6
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(left=0.23, bottom=0.08, top=0.96, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return names


def plot_last_24h(
    zarr_path: Path,
    output: Path,
    instrument: str,
    selection: str | None,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> list[str]:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = pd.DatetimeIndex(ds["time"].values)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    return plot_grouped_timeseries(
        window,
        instrument=instrument,
        selection=selection,
        title=_plot_title(instrument, selection, "Latest 24 hours"),
        output=output,
        max_time_samples=max_time_samples,
    )


def quicklook_prefix(instrument: str) -> str:
    return QUICKLOOK_PREFIX[instrument]


def group_latest_png(quicklook_dir: Path, instrument: str, selection: str | None) -> Path:
    spec = group_spec_for_selection(instrument, selection)
    return quicklook_dir / f"{quicklook_prefix(instrument)}__{spec.key}__latest.png"


def group_daily_png(quicklook_dir: Path, instrument: str, selection: str | None, day: pd.Timestamp | str) -> Path:
    spec = group_spec_for_selection(instrument, selection)
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{quicklook_prefix(instrument)}__{spec.key}__{stamp}.png"


def legacy_latest_png(quicklook_dir: Path, instrument: str) -> Path | None:
    prefix = LEGACY_ALIAS_PREFIX.get(instrument)
    if prefix is None:
        return None
    return quicklook_dir / "latest.png"


def legacy_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path | None:
    prefix = LEGACY_ALIAS_PREFIX.get(instrument)
    if prefix is None:
        return None
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{prefix}_{stamp}.png"


def clear_grouped_quicklooks(quicklook_dir: Path, instrument: str) -> None:
    prefix = quicklook_prefix(instrument)
    for png in quicklook_dir.glob(f"{prefix}*.png"):
        png.unlink()
    legacy_prefix = LEGACY_ALIAS_PREFIX.get(instrument)
    if legacy_prefix:
        for png in quicklook_dir.glob(f"{legacy_prefix}_*.png"):
            png.unlink()
        legacy_latest = legacy_latest_png(quicklook_dir, instrument)
        if legacy_latest and legacy_latest.exists():
            legacy_latest.unlink()


def refresh_legacy_aliases(
    quicklook_dir: Path,
    instrument: str,
    day_png: Path | None = None,
    latest_png: Path | None = None,
) -> None:
    if day_png is not None:
        legacy_day = legacy_daily_png(quicklook_dir, instrument, day_png.stem.rsplit("__", 1)[-1])
        if legacy_day:
            shutil.copyfile(day_png, legacy_day)
    if latest_png is not None:
        legacy_latest = legacy_latest_png(quicklook_dir, instrument)
        if legacy_latest:
            shutil.copyfile(latest_png, legacy_latest)
