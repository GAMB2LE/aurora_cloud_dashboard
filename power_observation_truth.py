"""Shared interval-aligned electrical truth and retrospective attribution.

Forecast power is an interval mean; SOC is an endpoint observation.  Missing
channels, telemetry gaps and limited MPPT output remain explicit exclusions.
No function in this module writes products or controls equipment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


TRUTH_CONTRACT_VERSION = "interval_electrical_truth_v1"
SOLAR_FIELDS = tuple(f"SolarWatts_{direction}" for direction in ("East", "South", "West"))
MPP_FIELDS = tuple(f"SolarMPPMode_{direction}" for direction in ("East", "South", "West"))
KIT_OUTLETS = {"CL61": 5, "Radar": 6, "HATPRO": 8, "UAS": 4}
STATE_FIELDS = (
    "ObservedLoadMode", "RealizedLoadMode", "OperatingModeCode", "OperatingLoadState",
    "DirectStateConfirmed", "OperatingStateConfirmed",
    *(f"PDUOutlet{outlet}{metric}" for outlet in KIT_OUTLETS.values() for metric in ("State", "Watts")),
)
POWER_OBSERVATION_FIELDS = ("BatterySOC", "BatteryWatts", *SOLAR_FIELDS, *MPP_FIELDS,
                            "ACOutputWatts", "DCInverterWatts", *STATE_FIELDS)


@dataclass(frozen=True)
class ObservationPolicy:
    minimum_coverage: float = 0.90
    max_gap: pd.Timedelta = pd.Timedelta(minutes=10)
    point_tolerance: pd.Timedelta = pd.Timedelta(minutes=10)

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("Observation coverage must be in (0, 1]")
        if self.max_gap <= pd.Timedelta(0) or self.point_tolerance < pd.Timedelta(0):
            raise ValueError("Observation gap must be positive and tolerance nonnegative")


def _times(values: Iterable[object]) -> pd.DatetimeIndex:
    if isinstance(values, np.ndarray) and values.dtype.kind == "M":
        values = values.astype("datetime64[ns]")
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_localize(None)


def forecast_interval_starts(valid_times: Iterable[object], issue_time: object) -> pd.DatetimeIndex:
    """Recover right-labelled forecast intervals, leaving the SOC anchor empty.

    Uses the issue time for the first forecast interval, not an invented
    cadence. NaT padding stays NaT; duplicate or decreasing endpoints fail.
    """
    times = _times(valid_times)
    issue = _times([issue_time])[0]
    result = np.full(len(times), np.datetime64("NaT"), dtype="datetime64[ns]")
    previous = issue
    for index, end in enumerate(times):
        if pd.isna(end):
            continue
        if end < previous or (end == previous and end != issue):
            raise ValueError("Forecast endpoints must increase after the SOC anchor")
        if end > issue:
            result[index] = previous.to_datetime64()
        previous = end
    return pd.DatetimeIndex(result)


def bounded_observation_view(
    power: xr.Dataset, interval_starts: Iterable[object], valid_times: Iterable[object],
    *, tolerance: pd.Timedelta = pd.Timedelta(minutes=10),
) -> xr.Dataset:
    """Lazily keep only requested interval unions and endpoint neighbours.

    Load time coordinates first, never unrelated telemetry variables. Retain
    every sample within requested intervals: reducing these to endpoint points
    would silently change energy truth. A single boundary neighbour supports
    exact partial-segment integration; it cannot bridge a disallowed gap.
    """
    if "time" not in power.coords:
        raise ValueError("Power observations need a time coordinate")
    times = _times(power.time.values)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Observation timestamps must be unique and monotonic")
    if pd.Timedelta(tolerance) < pd.Timedelta(0):
        raise ValueError("Observation tolerance cannot be negative")
    source = power[[name for name in POWER_OBSERVATION_FIELDS if name in power and power[name].dims == ("time",)]]
    starts, ends = _times(interval_starts), _times(valid_times)
    if len(starts) != len(ends):
        raise ValueError("Interval starts and endpoints must have equal length")
    ranges: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if pd.isna(end):
            continue
        if pd.isna(start) or start >= end:
            position = times.get_indexer([end], method="nearest", tolerance=tolerance)[0]
            if position >= 0:
                ranges.append((position, position + 1))
            continue
        left = max(0, int(times.searchsorted(start - tolerance, side="left")) - 1)
        right = min(len(times), int(times.searchsorted(end + tolerance, side="right")) + 1)
        if left < right:
            ranges.append((left, right))
    merged: list[list[int]] = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    positions = np.concatenate([np.arange(left, right) for left, right in merged]) if merged else np.array([], dtype=int)
    selected = source.isel(time=positions)
    selected.attrs = {**selected.attrs,
        "paired_observation_selection": "bounded_interval_union_with_endpoint_neighbours",
        "paired_observation_contract": TRUTH_CONTRACT_VERSION,
        "paired_observation_match_tolerance_seconds": float(tolerance.total_seconds()),
        "paired_observation_target_count": int(len(ends.dropna().unique())),
        "paired_observation_sample_count": int(len(positions)),
        "paired_observation_source_latest_utc": times.max().isoformat() if len(times) else "",
    }
    return selected


def _frame(power: xr.Dataset) -> pd.DataFrame:
    if "time" not in power.coords:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    times = _times(power.time.values)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Observation timestamps must be unique and monotonic")
    return pd.DataFrame({name: np.asarray(power[name].values) for name in POWER_OBSERVATION_FIELDS
                         if name in power and power[name].dims == ("time",)}, index=times)


def electrical_observation_series(power: xr.Dataset) -> pd.DataFrame:
    """Return complete-channel station truth plus a labelled partial fallback."""
    source = _frame(power)
    result = pd.DataFrame(index=source.index)
    solar = source.reindex(columns=SOLAR_FIELDS).apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=3)
    battery = pd.to_numeric(source.get("BatteryWatts", pd.Series(np.nan, index=source.index)), errors="coerce")
    balance = solar - battery
    # Negative station consumption is bad telemetry, not a zero-watt truth.
    result["load_w"] = balance.where(balance >= 0.0)
    result["solar_delivered_w"] = solar.where(solar >= 0.0)
    result["load_fallback_w"] = source.reindex(columns=("ACOutputWatts", "DCInverterWatts")).apply(
        pd.to_numeric, errors="coerce").sum(axis=1, min_count=2).clip(lower=0.0)
    modes = source.reindex(columns=MPP_FIELDS).apply(pd.to_numeric, errors="coerce")
    result["mpp_active"] = modes.eq(2).all(axis=1)
    result["solar_available_w"] = result["solar_delivered_w"].where(result["mpp_active"])
    result["soc"] = pd.to_numeric(source.get("BatterySOC", pd.Series(np.nan, index=source.index)), errors="coerce")
    result.attrs["source_latest_time"] = power.attrs.get("paired_observation_source_latest_utc", source.index.max())
    return result


def _canonical_mode(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "unknown", "unknown_ac", "ac load (unlabelled)"}:
        return ""
    if text == "Ceilometer-on-AC":
        return "DC-Only + CL61"
    kits = [kit for kit in KIT_OUTLETS if kit.lower() in text.lower()]
    if kits:
        return "DC-Only + " + " + ".join(kits)
    return "DC-Only" if text.lower() in {"dc-only", "dc only", "dc_only", "dc"} else ""


def _operating_modes(state: xr.Dataset | None) -> tuple[pd.Series, pd.Series, bool]:
    if state is None or "time" not in state.coords:
        return pd.Series(dtype=object), pd.Series(dtype=object), False
    frame = _frame(state)
    metadata = any(name in frame for name in STATE_FIELDS)
    modes = pd.Series("", index=frame.index, dtype=object)
    exact = pd.Series("", index=frame.index, dtype=object)
    for name in ("ObservedLoadMode", "RealizedLoadMode"):
        if name in frame:
            modes = frame[name].map(_canonical_mode)
            exact = frame[name].fillna("").astype(str)
            break
    else:
        if "OperatingModeCode" in frame:
            try:
                mapping = json.loads(str(state.OperatingModeCode.attrs.get("mode_mapping", "{}")))
            except (TypeError, ValueError):
                mapping = {}
            modes = frame.OperatingModeCode.map(lambda value: _canonical_mode(mapping.get(str(int(value)), "")) if pd.notna(value) else "")
        else:
            # Require the complete direct four-outlet vector. APS AC watts
            # alone cannot establish DC-only or identify an instrument.
            vector = pd.DataFrame(index=frame.index)
            for kit, outlet in KIT_OUTLETS.items():
                state_name, watts_name = f"PDUOutlet{outlet}State", f"PDUOutlet{outlet}Watts"
                value = pd.to_numeric(frame.get(state_name, pd.Series(np.nan, index=frame.index)), errors="coerce")
                watts = pd.to_numeric(frame.get(watts_name, pd.Series(np.nan, index=frame.index)), errors="coerce")
                vector[kit] = value.where(value.isin([0, 1]), (watts >= 5.0).where(watts.notna())).astype(float)
            complete = vector.notna().all(axis=1)
            codes = vector.fillna(0).to_numpy(dtype=int) @ (1 << np.arange(len(KIT_OUTLETS)))
            labels = {}
            for code in range(1 << len(KIT_OUTLETS)):
                kits = [kit for bit, kit in enumerate(KIT_OUTLETS) if code & (1 << bit)]
                labels[code] = "DC-Only" + (" + " + " + ".join(kits) if kits else "")
            modes = pd.Series(codes, index=frame.index).map(labels).where(complete, "")
    for name in ("DirectStateConfirmed", "OperatingStateConfirmed"):
        if name in frame:
            modes = modes.where(pd.to_numeric(frame[name], errors="coerce").eq(1), "")
    if "OperatingLoadState" in frame:
        exact = frame.OperatingLoadState.fillna("").astype(str)
    exact = exact.where(exact.ne(""), modes).where(modes.ne(""), "")
    modes.attrs["state_scope"] = "composed_operating_load_state" if "OperatingLoadState" in frame else "direct_pdu_mode_only"
    return modes, exact, metadata


def _interval_stats(series: pd.Series, starts: pd.DatetimeIndex, ends: pd.DatetimeIndex,
                    policy: ObservationPolicy) -> pd.DataFrame:
    """Integrate linear segments without extrapolating or bridging gaps."""
    count = len(starts)
    result = pd.DataFrame({"mean_w": np.full(count, np.nan), "energy_wh": np.full(count, np.nan),
                           "coverage": np.zeros(count), "max_gap_seconds": np.full(count, np.nan),
                           "status": np.full(count, "no_interval", dtype=object)})
    if len(series) < 2:
        result.loc[(~starts.isna()) & (ends > starts), "status"] = "insufficient_samples"
        return result
    times = series.index.asi8 / 1e9
    values = series.to_numpy(dtype=float)
    delta = np.diff(times)
    eligible = np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (delta <= policy.max_gap.total_seconds())
    for index, (start, end) in enumerate(zip(starts, ends)):
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        a, b = start.value / 1e9, end.value / 1e9
        left = max(0, int(np.searchsorted(times, a, side="right")) - 1)
        right = min(len(delta), int(np.searchsorted(times, b, side="left")))
        segment = np.arange(left, right)
        lo, hi = np.maximum(times[segment], a), np.minimum(times[segment + 1], b)
        duration = np.maximum(hi - lo, 0.0)
        use = eligible[segment] & (duration > 0)
        covered = float(duration[use].sum())
        coverage = covered / (b - a)
        gaps = delta[segment][duration > 0]
        edge_gap = max(min(times[0], b) - a, b - max(times[-1], a), 0.0)
        largest_gap = max(float(gaps.max()) if len(gaps) else b - a, edge_gap)
        result.loc[index, ["coverage", "max_gap_seconds"]] = [coverage, largest_gap]
        if largest_gap > policy.max_gap.total_seconds():
            result.at[index, "status"] = "gap_exceeds_limit"
            continue
        if coverage + 1e-9 < policy.minimum_coverage:
            result.at[index, "status"] = "insufficient_coverage"
            continue
        selected = segment[use]
        slope = (values[selected + 1] - values[selected]) / delta[selected]
        v_left = values[selected] + slope * (lo[use] - times[selected])
        v_right = values[selected] + slope * (hi[use] - times[selected])
        energy = float(np.sum(0.5 * (v_left + v_right) * duration[use]))
        # Wh is measured energy over covered support, never gap-filled energy.
        result.loc[index, ["mean_w", "energy_wh", "status"]] = [energy / covered, energy / 3600.0, "complete" if coverage >= 1 - 1e-9 else "partial_coverage"]
    return result


def prepare_observation_truth(power: xr.Dataset, operating_state: xr.Dataset | None = None) -> tuple:
    """Prepare repeated campaign matching once, not once per archived issue."""
    return (electrical_observation_series(power), *_operating_modes(
        operating_state if operating_state is not None else power))


def evaluate_power_observations(
    power: xr.Dataset, interval_starts: Iterable[object], valid_times: Iterable[object], *,
    expected_modes: Iterable[object] | None = None, operating_state: xr.Dataset | None = None,
    policy: ObservationPolicy | None = None, prepared: tuple | None = None,
) -> pd.DataFrame:
    """Endpoint SOC and covered interval power, with truth and state provenance."""
    policy = policy or ObservationPolicy()
    starts, ends = _times(interval_starts), _times(valid_times)
    if len(starts) != len(ends):
        raise ValueError("Interval bounds must have equal lengths")
    electrical, modes, exact, metadata = prepared if prepared is not None else prepare_observation_truth(power, operating_state)
    result = pd.DataFrame({"interval_start": starts, "valid_time": ends})
    soc = electrical.soc.reindex(ends, method="nearest", tolerance=policy.point_tolerance)
    result["soc"] = soc.to_numpy(dtype=float)
    if not electrical.empty:
        latest = pd.to_datetime(electrical.attrs.get("source_latest_time", electrical.index.max()), utc=True)
        result.loc[ends > latest.tz_localize(None), "soc"] = np.nan
    result["soc_status"] = np.where(np.isfinite(result.soc), "endpoint_matched", "endpoint_unavailable")
    for name in ("load", "load_fallback", "solar_delivered", "solar_available"):
        stats = _interval_stats(electrical[f"{name}_w"], starts, ends, policy)
        for field in ("mean_w", "energy_wh", "coverage", "max_gap_seconds", "status"):
            result[f"{name}_{'w' if field == 'mean_w' else field}"] = stats[field].values
    result["load_measurement"] = np.where(np.isfinite(result.load_w), "total_solar_minus_battery",
        np.where(np.isfinite(result.load_fallback_w), "fallback_ac_inverter_not_total", "unavailable"))
    result["solar_available_status"] = np.where(
        np.isfinite(result.solar_delivered_w) & ~np.isfinite(result.solar_available_w),
        "censored_or_missing_mpp", result.solar_available_status)
    expected = [_canonical_mode(value) for value in expected_modes] if expected_modes is not None else [""] * len(ends)
    if len(expected) != len(ends):
        raise ValueError("Expected operating modes must match interval bounds")
    result["realized_mode"] = ""
    result["realized_exact_state"] = ""
    result["operating_state_scope"] = modes.attrs.get("state_scope", "metadata_unavailable")
    result["operating_state_status"] = "metadata_unavailable" if not metadata else "unknown"
    for index, (start, end) in enumerate(zip(starts, ends)):
        if not metadata or modes.empty or pd.isna(end):
            continue
        start = end if pd.isna(start) else start
        # Categorical states are piecewise constant and may never be filled
        # backwards from a future PDU observation.
        prior = modes.index.get_indexer([start], method="pad", tolerance=policy.max_gap)[0]
        if prior < 0:
            continue
        selected = (modes.index >= start) & (modes.index < end)
        indices = np.unique(np.r_[prior, np.flatnonzero(selected)])
        interval_modes, interval_exact = modes.iloc[indices], exact.iloc[indices]
        if (interval_modes == "").any() or (interval_exact == "").any():
            continue
        support = pd.DatetimeIndex([start]).append(modes.index[indices][modes.index[indices] > start]).append(pd.DatetimeIndex([end]))
        if len(support) > 1 and max(np.diff(support.asi8), default=0) > policy.max_gap.value:
            continue
        if interval_modes.nunique() != 1 or interval_exact.nunique() != 1:
            result.at[index, "operating_state_status"] = "transition"
            continue
        mode, state_name = str(interval_modes.iloc[0]), str(interval_exact.iloc[0])
        result.loc[index, ["realized_mode", "realized_exact_state"]] = [mode, state_name]
        result.at[index, "operating_state_status"] = (
            "unchanged" if expected[index] and mode == expected[index]
            else "mismatch" if expected[index] else "known_no_issue_state")
    result.attrs.update({"truth_contract": TRUTH_CONTRACT_VERSION,
                         "minimum_coverage": policy.minimum_coverage,
                         "maximum_gap_seconds": policy.max_gap.total_seconds(),
                         "energy_semantics": "observed_Wh_over_covered_support_no_gap_imputation"})
    return result


def diagnostic_soc_replays(forecast: xr.Dataset, power: xr.Dataset, *,
                           operating_state: xr.Dataset | None = None,
                           policy: ObservationPolicy | None = None) -> xr.Dataset:
    """Retrospective energy-error attribution, never an issue-time predictor.

    Delivered solar is not available PV. Observed-solar lanes are therefore
    eligible only on MPPT-active intervals. After a missing interval the entire
    subsequent replay remains unavailable; there is no zero-fill/re-anchoring.
    """
    from power_battery_model import BatteryModel, soc_delta_percent

    times = _times(forecast.time.values)
    issue = pd.Timestamp(forecast.attrs.get("initial_soc_time", times[0]))
    starts = forecast_interval_starts(times, issue)
    truth = evaluate_power_observations(power, starts, times, operating_state=operating_state, policy=policy or ObservationPolicy(minimum_coverage=1.0))
    model = BatteryModel.from_attrs(forecast.attrs)
    anchor = float(forecast.attrs.get("initial_soc_pct", np.nan))
    if not np.isfinite(anchor) and "BatterySOCForecast" in forecast:
        anchor = float(forecast.BatterySOCForecast.values[0])
    forecast_solar = np.asarray(forecast.get("ForecastPVAvailableWatts", forecast.get("ForecastSolarWatts", xr.DataArray(np.full(len(times), np.nan)))).values, dtype=float)
    forecast_load = np.asarray(forecast.get("ForecastLoadWatts", xr.DataArray(np.full(len(times), np.nan))).values, dtype=float)
    output: dict[str, tuple[str, np.ndarray]] = {}
    for label, solar, load in (
        ("ObservedSolar", truth.solar_available_w.to_numpy(), forecast_load),
        ("ObservedLoad", forecast_solar, truth.load_w.to_numpy()),
        ("ObservedSolarAndLoad", truth.solar_available_w.to_numpy(), truth.load_w.to_numpy()),
    ):
        values = np.full(len(times), np.nan)
        prior_soc = anchor
        for index, (start, end) in enumerate(zip(starts, times)):
            if pd.isna(start) and end == issue:
                values[index] = prior_soc
                continue
            if pd.isna(start) or not np.isfinite(prior_soc) or not np.isfinite(solar[index] - load[index]):
                prior_soc = np.nan
                continue
            hours = (end - start) / pd.Timedelta(hours=1)
            prior_soc = float(np.clip(prior_soc + soc_delta_percent(solar[index] - load[index], hours, model), 0, 100))
            values[index] = prior_soc
        output[f"DiagnosticSOC{label}"] = ("time", values)
    output["ObservedSOC"] = ("time", truth.soc.to_numpy())
    return xr.Dataset(output, coords={"time": times}, attrs={
        "authority": "retrospective_diagnostic_only", "predictive_skill_eligible": "false",
        "truth_contract": TRUTH_CONTRACT_VERSION,
        "solar_truth": "available_PV_only_when_all_three_MPPT_active",
        "integration": "interval_mean_energy_with_forecast_battery_model;not_substep_counterfactual",
        "missing_interval_policy": "stop_remaining_replay_without_reanchoring",
    })
