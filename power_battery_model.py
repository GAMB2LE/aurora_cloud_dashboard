#!/usr/bin/env python3
"""Shared APS battery-energy model and conservative history calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_CHARGE_EFFICIENCY = 0.92
DEFAULT_DISCHARGE_EFFICIENCY = 0.92
DEFAULT_MAX_CHARGE_W = 3_000.0
DEFAULT_MAX_DISCHARGE_W = 3_000.0


@dataclass(frozen=True)
class BatteryModel:
    usable_capacity_kwh: float = 26.0
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    discharge_efficiency: float = DEFAULT_DISCHARGE_EFFICIENCY
    parasitic_load_w: float = 0.0
    max_charge_w: float = DEFAULT_MAX_CHARGE_W
    max_discharge_w: float = DEFAULT_MAX_DISCHARGE_W
    calibration_sample_count: int = 0
    calibration_confidence: str = "default"

    @classmethod
    def from_attrs(
        cls,
        attrs: Mapping[str, object],
        *,
        default_capacity_kwh: float = 26.0,
    ) -> "BatteryModel":
        def number(name: str, default: float) -> float:
            try:
                value = float(attrs.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if np.isfinite(value) else default

        try:
            sample_count = int(float(attrs.get("battery_calibration_sample_count", 0)))
        except (TypeError, ValueError):
            sample_count = 0
        return cls(
            usable_capacity_kwh=number("battery_usable_capacity_kwh", number("battery_capacity_kwh", default_capacity_kwh)),
            charge_efficiency=number("battery_charge_efficiency", DEFAULT_CHARGE_EFFICIENCY),
            discharge_efficiency=number("battery_discharge_efficiency", DEFAULT_DISCHARGE_EFFICIENCY),
            parasitic_load_w=number("battery_parasitic_load_w", 0.0),
            max_charge_w=number("battery_max_charge_w", DEFAULT_MAX_CHARGE_W),
            max_discharge_w=number("battery_max_discharge_w", DEFAULT_MAX_DISCHARGE_W),
            calibration_sample_count=max(sample_count, 0),
            calibration_confidence=str(attrs.get("battery_calibration_confidence", "default")),
        ).validated()

    def validated(self) -> "BatteryModel":
        return BatteryModel(
            usable_capacity_kwh=float(np.clip(self.usable_capacity_kwh, 10.0, 40.0)),
            charge_efficiency=float(np.clip(self.charge_efficiency, 0.65, 1.0)),
            discharge_efficiency=float(np.clip(self.discharge_efficiency, 0.65, 1.0)),
            parasitic_load_w=float(np.clip(self.parasitic_load_w, 0.0, 500.0)),
            max_charge_w=float(np.clip(self.max_charge_w, 100.0, 20_000.0)),
            max_discharge_w=float(np.clip(self.max_discharge_w, 100.0, 20_000.0)),
            calibration_sample_count=max(int(self.calibration_sample_count), 0),
            calibration_confidence=str(self.calibration_confidence),
        )

    def attrs(self) -> dict[str, str]:
        return {
            "battery_capacity_kwh": f"{self.usable_capacity_kwh:.6g}",
            "battery_usable_capacity_kwh": f"{self.usable_capacity_kwh:.6g}",
            "battery_charge_efficiency": f"{self.charge_efficiency:.6g}",
            "battery_discharge_efficiency": f"{self.discharge_efficiency:.6g}",
            "battery_parasitic_load_w": f"{self.parasitic_load_w:.6g}",
            "battery_max_charge_w": f"{self.max_charge_w:.6g}",
            "battery_max_discharge_w": f"{self.max_discharge_w:.6g}",
            "battery_calibration_sample_count": str(self.calibration_sample_count),
            "battery_calibration_confidence": self.calibration_confidence,
            "battery_energy_model": "bounded_bidirectional_efficiency_v1",
        }


def battery_energy_delta_kwh(
    net_station_power_w: np.ndarray | float,
    hours: float,
    model: BatteryModel,
) -> np.ndarray:
    """Convert station solar-minus-load power into stored battery energy."""
    values = np.asarray(net_station_power_w, dtype=np.float64) - model.parasitic_load_w
    charging = np.minimum(np.clip(values, 0.0, None), model.max_charge_w)
    discharging = np.minimum(np.clip(-values, 0.0, None), model.max_discharge_w)
    stored_w = charging * model.charge_efficiency - discharging / model.discharge_efficiency
    return stored_w * max(float(hours), 0.0) / 1000.0


def soc_delta_percent(
    net_station_power_w: np.ndarray | float,
    hours: float,
    model: BatteryModel,
) -> np.ndarray:
    return 100.0 * battery_energy_delta_kwh(net_station_power_w, hours, model) / model.usable_capacity_kwh


def fit_battery_model(
    frame: pd.DataFrame,
    *,
    nominal_capacity_kwh: float = 26.0,
    lookback_days: float = 30.0,
) -> BatteryModel:
    """Estimate bounded APS battery parameters from stable non-saturated intervals."""
    required = {"BatterySOC", "BatteryWatts"}
    if frame.empty or not required.issubset(frame.columns):
        return BatteryModel(usable_capacity_kwh=nominal_capacity_kwh).validated()

    end = pd.Timestamp(frame.index.max())
    source = frame.loc[frame.index >= end - pd.Timedelta(days=float(lookback_days))].copy()
    sample = source[["BatterySOC", "BatteryWatts"]].resample("30min").median().dropna()
    if len(sample) < 3:
        return BatteryModel(usable_capacity_kwh=nominal_capacity_kwh).validated()

    if "ObservedLoadWatts" in source:
        load = source["ObservedLoadWatts"].resample("30min").median().reindex(sample.index)
    else:
        load = pd.Series(np.nan, index=sample.index)
    previous_soc = sample["BatterySOC"].shift(1)
    delta_soc = sample["BatterySOC"] - previous_soc
    elapsed_h = sample.index.to_series().diff() / pd.Timedelta(hours=1)
    battery_w = (sample["BatteryWatts"] + sample["BatteryWatts"].shift(1)) / 2.0
    stable_load = load.diff().abs().fillna(0.0) <= 100.0
    usable = (
        previous_soc.between(5.0, 95.0)
        & sample["BatterySOC"].between(5.0, 95.0)
        & elapsed_h.between(0.25, 1.5)
        & (delta_soc.abs() >= 0.20)
        & stable_load
        & (((delta_soc > 0.0) & (battery_w > 25.0)) | ((delta_soc < 0.0) & (battery_w < -25.0)))
    )
    delta_soc = delta_soc[usable]
    elapsed_h = elapsed_h[usable]
    battery_w = battery_w[usable]
    if len(delta_soc) < 4:
        observed = source["BatteryWatts"].dropna().to_numpy(dtype=np.float64)
        return BatteryModel(
            usable_capacity_kwh=nominal_capacity_kwh,
            max_charge_w=_observed_limit(observed, positive=True),
            max_discharge_w=_observed_limit(observed, positive=False),
        ).validated()

    electrical_kwh = battery_w.abs() * elapsed_h / 1000.0
    capacity_candidates = np.where(
        delta_soc > 0.0,
        electrical_kwh * DEFAULT_CHARGE_EFFICIENCY * 100.0 / delta_soc.abs(),
        electrical_kwh * 100.0 / (DEFAULT_DISCHARGE_EFFICIENCY * delta_soc.abs()),
    )
    capacity_candidates = capacity_candidates[
        np.isfinite(capacity_candidates) & (capacity_candidates >= 15.0) & (capacity_candidates <= 35.0)
    ]
    capacity = float(np.nanmedian(capacity_candidates)) if len(capacity_candidates) >= 6 else float(nominal_capacity_kwh)

    stored_change = delta_soc.abs() * capacity / 100.0
    charge_eff = (stored_change[delta_soc > 0.0] / electrical_kwh[delta_soc > 0.0]).to_numpy(dtype=np.float64)
    discharge_eff = (electrical_kwh[delta_soc < 0.0] / stored_change[delta_soc < 0.0]).to_numpy(dtype=np.float64)
    charge_eff = charge_eff[np.isfinite(charge_eff) & (charge_eff >= 0.65) & (charge_eff <= 1.05)]
    discharge_eff = discharge_eff[np.isfinite(discharge_eff) & (discharge_eff >= 0.65) & (discharge_eff <= 1.05)]
    charge = float(np.nanmedian(charge_eff)) if len(charge_eff) >= 3 else DEFAULT_CHARGE_EFFICIENCY
    discharge = float(np.nanmedian(discharge_eff)) if len(discharge_eff) >= 3 else DEFAULT_DISCHARGE_EFFICIENCY
    observed = source["BatteryWatts"].dropna().to_numpy(dtype=np.float64)
    count = int(len(delta_soc))
    return BatteryModel(
        usable_capacity_kwh=capacity,
        charge_efficiency=charge,
        discharge_efficiency=discharge,
        max_charge_w=_observed_limit(observed, positive=True),
        max_discharge_w=_observed_limit(observed, positive=False),
        calibration_sample_count=count,
        calibration_confidence="calibrated" if count >= 12 else "provisional",
    ).validated()


def _observed_limit(values: np.ndarray, *, positive: bool) -> float:
    selected = values[values > 0.0] if positive else -values[values < 0.0]
    default = DEFAULT_MAX_CHARGE_W if positive else DEFAULT_MAX_DISCHARGE_W
    if len(selected) < 4:
        return default
    return max(float(np.nanquantile(selected, 0.995) * 1.10), 100.0)
