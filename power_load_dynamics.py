"""Learn transient load phases within exact APS operating states.

Instrument state remains the primary forecast control.  This module only
models repeatable load phases observed while that exact state is unchanged:
startup, steady operation, and low/high fan regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from power_load_contract import ControlledLoadEstimate, controlled_load_member_levels


LOAD_PHASE_SCHEMA_VERSION = 1
PHASE_STEADY = "steady"
PHASE_STARTUP = "startup"
PHASE_FAN_LOW = "fan_low"
PHASE_FAN_HIGH = "fan_high"
PHASE_CODES = {
    PHASE_STEADY: 0,
    PHASE_STARTUP: 1,
    PHASE_FAN_LOW: 2,
    PHASE_FAN_HIGH: 3,
}
DEFAULT_CHANGE_THRESHOLD_W = 20.0
DEFAULT_RELATIVE_CHANGE = 0.08
DEFAULT_MIN_PHASE_SAMPLES = 4
DEFAULT_MAX_STARTUP_MINUTES = 12.0 * 60.0


@dataclass(frozen=True)
class LoadDistribution:
    """Robust load distribution for one phase of one operating state."""

    p10_w: float
    p50_w: float
    p90_w: float
    sample_count: int

    @classmethod
    def from_values(cls, values: Sequence[float] | np.ndarray) -> "LoadDistribution":
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite) & (finite >= 0.0)]
        if not finite.size:
            raise ValueError("A load distribution needs at least one finite observation")
        p10, p50, p90 = np.nanquantile(finite, (0.10, 0.50, 0.90))
        return cls(float(max(p10, 0.0)), float(max(p50, 0.0)), float(max(p90, p50)), int(finite.size))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LoadDistribution":
        return cls(
            p10_w=float(value["p10_w"]),
            p50_w=float(value["p50_w"]),
            p90_w=float(value["p90_w"]),
            sample_count=int(value.get("sample_count", 0)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "p10_w": self.p10_w,
            "p50_w": self.p50_w,
            "p90_w": self.p90_w,
            "sample_count": self.sample_count,
        }

    def as_estimate(self, *, source: str) -> ControlledLoadEstimate:
        return ControlledLoadEstimate(
            self.p10_w,
            self.p50_w,
            self.p90_w,
            source,
            self.sample_count,
        ).validated()


@dataclass(frozen=True)
class StateLoadDynamics:
    """Learned phases and timing for one exact controlled state."""

    state: str
    current_phase: str
    state_started_at: str
    phase_started_at: str
    startup_duration_p10_minutes: float
    startup_duration_p50_minutes: float
    startup_duration_p90_minutes: float
    phase_profiles: dict[str, LoadDistribution]
    phase_weights: dict[str, float]
    phase_dwell_minutes: dict[str, float]
    sample_count: int
    episode_count: int
    change_count: int

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StateLoadDynamics":
        profiles = {
            str(name): LoadDistribution.from_dict(profile)
            for name, profile in dict(value.get("phase_profiles", {})).items()
            if isinstance(profile, Mapping)
        }
        if not profiles:
            raise ValueError("State load dynamics do not contain phase profiles")
        return cls(
            state=str(value["state"]),
            current_phase=str(value.get("current_phase", PHASE_STEADY)),
            state_started_at=str(value.get("state_started_at", "")),
            phase_started_at=str(value.get("phase_started_at", "")),
            startup_duration_p10_minutes=float(value.get("startup_duration_p10_minutes", 0.0)),
            startup_duration_p50_minutes=float(value.get("startup_duration_p50_minutes", 0.0)),
            startup_duration_p90_minutes=float(value.get("startup_duration_p90_minutes", 0.0)),
            phase_profiles=profiles,
            phase_weights={str(name): float(weight) for name, weight in dict(value.get("phase_weights", {})).items()},
            phase_dwell_minutes={str(name): float(minutes) for name, minutes in dict(value.get("phase_dwell_minutes", {})).items()},
            sample_count=int(value.get("sample_count", 0)),
            episode_count=int(value.get("episode_count", 0)),
            change_count=int(value.get("change_count", 0)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": LOAD_PHASE_SCHEMA_VERSION,
            "state": self.state,
            "current_phase": self.current_phase,
            "state_started_at": self.state_started_at,
            "phase_started_at": self.phase_started_at,
            "startup_duration_p10_minutes": self.startup_duration_p10_minutes,
            "startup_duration_p50_minutes": self.startup_duration_p50_minutes,
            "startup_duration_p90_minutes": self.startup_duration_p90_minutes,
            "phase_profiles": {name: profile.to_dict() for name, profile in self.phase_profiles.items()},
            "phase_weights": dict(self.phase_weights),
            "phase_dwell_minutes": dict(self.phase_dwell_minutes),
            "sample_count": self.sample_count,
            "episode_count": self.episode_count,
            "change_count": self.change_count,
        }


@dataclass(frozen=True)
class ControlledLoadProfile:
    """Lead-dependent quantiles and phase codes for deterministic integration."""

    p10_w: np.ndarray
    p50_w: np.ndarray
    p90_w: np.ndarray
    phase_codes: np.ndarray
    source: str


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 3.0
    median = float(np.nanmedian(finite))
    return max(1.4826 * float(np.nanmedian(np.abs(finite - median))), 3.0)


def _material_difference(left: np.ndarray, right: np.ndarray) -> bool:
    if not len(left) or not len(right):
        return False
    left_median = float(np.nanmedian(left))
    right_median = float(np.nanmedian(right))
    threshold = max(
        DEFAULT_CHANGE_THRESHOLD_W,
        DEFAULT_RELATIVE_CHANGE * max(min(left_median, right_median), 1.0),
        3.0 * max(_robust_scale(left), _robust_scale(right)),
    )
    return abs(left_median - right_median) >= threshold


def _run_groups(frame: pd.DataFrame, *, cadence: pd.Timedelta) -> list[pd.DataFrame]:
    gaps = frame.index.to_series().diff().gt(max(cadence * 2.5, pd.Timedelta(minutes=30)))
    changed = frame["mode"].ne(frame["mode"].shift()) | gaps
    return [group.drop(columns="_run") for _, group in frame.assign(_run=changed.cumsum()).groupby("_run")]


def _startup_boundary(values: np.ndarray, *, cadence_minutes: float) -> int:
    """Return the first settled sample, or zero when no startup is evident."""
    count = int(len(values))
    minimum = DEFAULT_MIN_PHASE_SAMPLES
    if count < minimum * 3:
        return 0
    smoothed = pd.Series(values).rolling(3, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
    maximum_index = min(count - minimum, int(DEFAULT_MAX_STARTUP_MINUTES / max(cadence_minutes, 1.0)))
    candidates: list[tuple[float, int]] = []
    for index in range(minimum, maximum_index + 1):
        before = smoothed[index - minimum : index]
        after = smoothed[index : index + minimum]
        if _material_difference(before, after):
            difference = abs(float(np.nanmedian(before)) - float(np.nanmedian(after)))
            candidates.append((difference, index))
    if not candidates:
        return 0
    # Prefer the strongest early transition. Equal startup/fan steps resolve to
    # the earlier boundary because only the leading segment can be startup.
    return max(candidates, key=lambda value: (value[0], -value[1]))[1]


def _split_steady_phases(values: np.ndarray) -> tuple[dict[str, LoadDistribution], np.ndarray]:
    finite = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(finite) & (finite >= 0.0)
    if np.count_nonzero(valid) < DEFAULT_MIN_PHASE_SAMPLES * 2:
        profile = LoadDistribution.from_values(finite[valid])
        return {PHASE_STEADY: profile}, np.full(len(finite), PHASE_STEADY, dtype=object)
    ordered = np.sort(finite[valid])
    differences = np.diff(ordered)
    split = int(np.argmax(differences)) + 1
    low = ordered[:split]
    high = ordered[split:]
    if (
        len(low) < DEFAULT_MIN_PHASE_SAMPLES
        or len(high) < DEFAULT_MIN_PHASE_SAMPLES
        or not _material_difference(low, high)
    ):
        profile = LoadDistribution.from_values(finite[valid])
        return {PHASE_STEADY: profile}, np.full(len(finite), PHASE_STEADY, dtype=object)
    profiles = {
        PHASE_FAN_LOW: LoadDistribution.from_values(low),
        PHASE_FAN_HIGH: LoadDistribution.from_values(high),
    }
    midpoint = (profiles[PHASE_FAN_LOW].p50_w + profiles[PHASE_FAN_HIGH].p50_w) / 2.0
    labels = np.where(finite <= midpoint, PHASE_FAN_LOW, PHASE_FAN_HIGH).astype(object)
    return profiles, labels


def _debounce_labels(labels: np.ndarray, *, minimum_samples: int = 2) -> np.ndarray:
    result = np.asarray(labels, dtype=object).copy()
    if len(result) < 3:
        return result
    start = 0
    while start < len(result):
        stop = start + 1
        while stop < len(result) and result[stop] == result[start]:
            stop += 1
        if stop - start < minimum_samples:
            replacement = result[start - 1] if start else (result[stop] if stop < len(result) else result[start])
            result[start:stop] = replacement
        start = stop
    return result


def learn_state_load_dynamics(
    observations: pd.DataFrame,
    state: str,
    *,
    mode_column: str = "direct_mode",
    load_column: str = "load_w",
) -> StateLoadDynamics | None:
    """Learn startup and steady/fan phases for one exact state."""
    if observations.empty or mode_column not in observations or load_column not in observations:
        return None
    frame = pd.DataFrame(
        {
            "mode": observations[mode_column].astype(str),
            "load_w": pd.to_numeric(observations[load_column], errors="coerce"),
        },
        index=pd.DatetimeIndex(observations.index),
    ).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[np.isfinite(frame["load_w"]) & (frame["load_w"] >= 0.0)]
    if frame.empty or state not in set(frame["mode"]):
        return None
    differences = frame.index.to_series().diff().dropna()
    cadence = differences.median() if not differences.empty else pd.Timedelta(minutes=15)
    if pd.isna(cadence) or cadence <= pd.Timedelta(0):
        cadence = pd.Timedelta(minutes=15)
    cadence_minutes = max(float(cadence / pd.Timedelta(minutes=1)), 1.0)

    runs = [group for group in _run_groups(frame, cadence=cadence) if str(group["mode"].iloc[0]) == state]
    startup_values: list[float] = []
    startup_durations: list[float] = []
    steady_values: list[float] = []
    run_details: list[tuple[pd.DataFrame, int]] = []
    for run in runs:
        values = run["load_w"].to_numpy(dtype=np.float64)
        boundary = _startup_boundary(values, cadence_minutes=cadence_minutes)
        run_details.append((run, boundary))
        if boundary:
            startup_values.extend(values[:boundary])
            startup_durations.append(boundary * cadence_minutes)
            steady_values.extend(values[boundary:])
        else:
            steady_values.extend(values)
    if not steady_values:
        steady_values = [float(value) for value in frame.loc[frame["mode"] == state, "load_w"]]
    steady_profiles, _ = _split_steady_phases(np.asarray(steady_values, dtype=np.float64))
    phase_profiles = dict(steady_profiles)
    if startup_values:
        phase_profiles[PHASE_STARTUP] = LoadDistribution.from_values(startup_values)

    steady_names = [name for name in (PHASE_FAN_LOW, PHASE_FAN_HIGH, PHASE_STEADY) if name in steady_profiles]
    steady_centres = np.asarray([steady_profiles[name].p50_w for name in steady_names], dtype=np.float64)
    phase_counts = {name: 0 for name in steady_names}
    phase_durations: dict[str, list[float]] = {name: [] for name in steady_names}
    current_phase = steady_names[0]
    state_started_at = ""
    phase_started_at = ""
    change_count = 0
    latest_time = frame.index[-1]
    latest_run: pd.DataFrame | None = None
    latest_boundary = 0
    for run, boundary in run_details:
        values = run["load_w"].to_numpy(dtype=np.float64)
        labels = np.full(len(run), PHASE_STARTUP if boundary else steady_names[0], dtype=object)
        if boundary < len(run):
            distances = np.abs(values[boundary:, None] - steady_centres[None, :])
            steady_labels = np.asarray([steady_names[index] for index in np.argmin(distances, axis=1)], dtype=object)
            labels[boundary:] = _debounce_labels(steady_labels)
        segment_start = boundary
        while segment_start < len(run):
            segment_stop = segment_start + 1
            while segment_stop < len(run) and labels[segment_stop] == labels[segment_start]:
                segment_stop += 1
            name = str(labels[segment_start])
            if name in phase_counts:
                phase_counts[name] += segment_stop - segment_start
                phase_durations[name].append((segment_stop - segment_start) * cadence_minutes)
            if segment_start > 0:
                change_count += 1
            segment_start = segment_stop
        if run.index[-1] == latest_time:
            latest_run = run
            latest_boundary = boundary
            current_phase = str(labels[-1])
            state_started_at = run.index[0].isoformat()
            changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
            phase_started_at = run.index[int(changes[-1])].isoformat() if changes.size else run.index[0].isoformat()

    if startup_durations:
        startup_q10, startup_q50, startup_q90 = np.nanquantile(startup_durations, (0.10, 0.50, 0.90))
    else:
        startup_q10 = startup_q50 = startup_q90 = 0.0
    if latest_run is not None and PHASE_STARTUP in phase_profiles:
        elapsed = float((latest_time - latest_run.index[0]) / pd.Timedelta(minutes=1)) + cadence_minutes
        if latest_boundary == 0 and elapsed <= max(startup_q90, startup_q50, cadence_minutes):
            current_phase = PHASE_STARTUP
            phase_started_at = latest_run.index[0].isoformat()
    total_steady = max(sum(phase_counts.values()), 1)
    phase_weights = {name: count / total_steady for name, count in phase_counts.items()}
    if not any(phase_weights.values()):
        phase_weights = {steady_names[0]: 1.0}
    dwell = {
        name: float(np.nanmedian(values)) if values else cadence_minutes
        for name, values in phase_durations.items()
    }
    selected = frame.loc[frame["mode"] == state]
    if not state_started_at:
        state_started_at = selected.index[-1].isoformat()
        phase_started_at = state_started_at
    return StateLoadDynamics(
        state=state,
        current_phase=current_phase,
        state_started_at=state_started_at,
        phase_started_at=phase_started_at,
        startup_duration_p10_minutes=float(startup_q10),
        startup_duration_p50_minutes=float(startup_q50),
        startup_duration_p90_minutes=float(startup_q90),
        phase_profiles=phase_profiles,
        phase_weights=phase_weights,
        phase_dwell_minutes=dwell,
        sample_count=int(len(selected)),
        episode_count=int(len(runs)),
        change_count=int(change_count),
    )


def _dominant_steady_phase(dynamics: StateLoadDynamics) -> str:
    names = [name for name in (PHASE_FAN_LOW, PHASE_FAN_HIGH, PHASE_STEADY) if name in dynamics.phase_profiles]
    if not names:
        return PHASE_STARTUP
    return max(names, key=lambda name: dynamics.phase_weights.get(name, 0.0))


def _phase_estimate(
    dynamics: StateLoadDynamics,
    phase: str,
    fallback: ControlledLoadEstimate,
) -> ControlledLoadEstimate:
    profile = dynamics.phase_profiles.get(phase)
    if profile is None:
        return fallback.validated()
    return profile.as_estimate(source=f"learned_exact_state_{phase}_distribution")


def build_controlled_load_profile(
    dynamics: StateLoadDynamics | None,
    forecast_times: pd.DatetimeIndex,
    fallback: ControlledLoadEstimate,
) -> ControlledLoadProfile:
    """Build central and interval load paths for the currently detected phase."""
    times = pd.DatetimeIndex(forecast_times)
    checked = fallback.validated()
    if dynamics is None or not len(times):
        return ControlledLoadProfile(
            np.full(len(times), checked.p10_w),
            np.full(len(times), checked.p50_w),
            np.full(len(times), checked.p90_w),
            np.full(len(times), PHASE_CODES[PHASE_STEADY], dtype=np.int8),
            checked.source,
        )
    current_phase = dynamics.current_phase if dynamics.current_phase in dynamics.phase_profiles else _dominant_steady_phase(dynamics)
    phases = np.full(len(times), current_phase, dtype=object)
    if current_phase == PHASE_STARTUP and len(times):
        state_start = pd.to_datetime(dynamics.state_started_at, errors="coerce")
        if not pd.isna(state_start):
            startup_end = pd.Timestamp(state_start) + pd.Timedelta(minutes=dynamics.startup_duration_p50_minutes)
            phases[times > startup_end] = _dominant_steady_phase(dynamics)
    p10 = np.empty(len(times), dtype=np.float64)
    p50 = np.empty(len(times), dtype=np.float64)
    p90 = np.empty(len(times), dtype=np.float64)
    steady_profiles = [
        profile
        for name, profile in dynamics.phase_profiles.items()
        if name != PHASE_STARTUP
    ]
    steady_floor = min((profile.p10_w for profile in steady_profiles), default=checked.p10_w)
    steady_ceiling = max((profile.p90_w for profile in steady_profiles), default=checked.p90_w)
    for index, phase in enumerate(phases):
        estimate = _phase_estimate(dynamics, str(phase), checked)
        p10[index] = estimate.p10_w
        p50[index] = estimate.p50_w
        p90[index] = estimate.p90_w
        if phase != PHASE_STARTUP:
            p10[index] = min(p10[index], steady_floor)
            p90[index] = max(p90[index], steady_ceiling)
    return ControlledLoadProfile(
        p10,
        p50,
        p90,
        np.asarray([PHASE_CODES.get(str(name), PHASE_CODES[PHASE_STEADY]) for name in phases], dtype=np.int8),
        f"learned_exact_state_phase:{current_phase}",
    )


def controlled_load_member_profiles(
    dynamics: StateLoadDynamics | None,
    forecast_times: pd.DatetimeIndex,
    fallback: ControlledLoadEstimate,
    member_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible startup/fan paths inside one exact state."""
    times = pd.DatetimeIndex(forecast_times)
    count = max(int(member_count), 1)
    if dynamics is None or not len(times):
        levels = controlled_load_member_levels(fallback, count)
        return (
            np.repeat(levels[:, None], len(times), axis=1),
            np.full((count, len(times)), PHASE_CODES[PHASE_STEADY], dtype=np.int8),
        )
    rng = np.random.default_rng(int(seed))
    cadence_minutes = (
        max(float(np.nanmedian(np.diff(times.asi8))) / 60e9, 1.0)
        if len(times) > 1
        else 60.0
    )
    steady_names = [name for name in (PHASE_FAN_LOW, PHASE_FAN_HIGH, PHASE_STEADY) if name in dynamics.phase_profiles]
    if not steady_names:
        steady_names = [PHASE_STARTUP]
    current = dynamics.current_phase if dynamics.current_phase in dynamics.phase_profiles else _dominant_steady_phase(dynamics)
    loads = np.empty((count, len(times)), dtype=np.float64)
    codes = np.empty((count, len(times)), dtype=np.int8)
    phase_levels = {
        name: controlled_load_member_levels(_phase_estimate(dynamics, name, fallback), count)
        for name in dynamics.phase_profiles
    }
    fallback_levels = controlled_load_member_levels(fallback, count)
    weights = np.asarray([max(dynamics.phase_weights.get(name, 0.0), 0.0) for name in steady_names], dtype=np.float64)
    weights = weights / weights.sum() if weights.sum() > 0.0 else np.full(len(steady_names), 1.0 / len(steady_names))
    for member in range(count):
        phase = current
        transition_index = len(times)
        if phase == PHASE_STARTUP:
            durations = np.asarray(
                [
                    dynamics.startup_duration_p10_minutes,
                    dynamics.startup_duration_p50_minutes,
                    dynamics.startup_duration_p90_minutes,
                ],
                dtype=np.float64,
            )
            duration_rank = (member + 0.5) / count
            duration = float(np.interp(duration_rank, (0.10, 0.50, 0.90), durations))
            state_start = pd.to_datetime(dynamics.state_started_at, errors="coerce")
            elapsed = 0.0 if pd.isna(state_start) or not len(times) else max(float((times[0] - state_start) / pd.Timedelta(minutes=1)), 0.0)
            transition_index = int(np.ceil(max(duration - elapsed, 0.0) / cadence_minutes))
        next_phase = str(rng.choice(steady_names, p=weights))
        phase_elapsed = 0
        dwell_steps = max(int(round(dynamics.phase_dwell_minutes.get(next_phase, 6.0 * 60.0) / cadence_minutes)), 1)
        for time_index in range(len(times)):
            if phase == PHASE_STARTUP and time_index >= transition_index:
                phase = next_phase
                phase_elapsed = 0
                dwell_steps = max(int(round(dynamics.phase_dwell_minutes.get(phase, 6.0 * 60.0) / cadence_minutes)), 1)
            elif phase != PHASE_STARTUP and len(steady_names) > 1 and phase_elapsed >= dwell_steps:
                alternatives = [name for name in steady_names if name != phase]
                phase = str(rng.choice(alternatives))
                phase_elapsed = 0
                mean_steps = max(dynamics.phase_dwell_minutes.get(phase, 6.0 * 60.0) / cadence_minutes, 1.0)
                dwell_steps = max(int(round(rng.exponential(mean_steps))), 1)
            levels = phase_levels.get(phase, fallback_levels)
            loads[member, time_index] = levels[member]
            codes[member, time_index] = PHASE_CODES.get(phase, PHASE_CODES[PHASE_STEADY])
            phase_elapsed += 1
    return loads, codes


def force_startup(dynamics: StateLoadDynamics, start: pd.Timestamp) -> StateLoadDynamics:
    """Return a scenario copy that starts in the learned startup phase."""
    if PHASE_STARTUP not in dynamics.phase_profiles:
        return replace(
            dynamics,
            current_phase=_dominant_steady_phase(dynamics),
            state_started_at=pd.Timestamp(start).isoformat(),
            phase_started_at=pd.Timestamp(start).isoformat(),
        )
    return replace(
        dynamics,
        current_phase=PHASE_STARTUP,
        state_started_at=pd.Timestamp(start).isoformat(),
        phase_started_at=pd.Timestamp(start).isoformat(),
    )
