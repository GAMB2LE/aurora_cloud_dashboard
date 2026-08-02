"""Finite controlled-state load contract shared by APS forecast products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CONTROLLED_LOAD_CONTRACT = "finite_operating_state_v1"
STATE_HOLD_POLICY = "hold_latest_confirmed_state_until_explicit_schedule_transition"


@dataclass(frozen=True)
class ControlledLoadEstimate:
    """A stationary load distribution for one confirmed operating state."""

    p10_w: float
    p50_w: float
    p90_w: float
    source: str
    sample_count: int

    def validated(self) -> "ControlledLoadEstimate":
        values = np.asarray((self.p10_w, self.p50_w, self.p90_w), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Controlled-state load estimate must be finite")
        p50 = max(float(values[1]), 0.0)
        return ControlledLoadEstimate(
            p10_w=min(max(float(values[0]), 0.0), p50),
            p50_w=p50,
            p90_w=max(float(values[2]), p50),
            source=str(self.source),
            sample_count=max(int(self.sample_count), 0),
        )


def estimate_controlled_load(
    *,
    mode: str,
    measured_current_w: float,
    learned_observations_w: Iterable[float] = (),
    learned_level_w: float | None = None,
    component_estimate_w: float | None = None,
    dc_only_estimate_w: float | None = None,
    minimum_observations: int = 3,
) -> ControlledLoadEstimate:
    """Estimate one finite state's load without borrowing another state's history.

    Fresh PDU component measurements are preferred for an active state. A clean
    dark-period DC estimate is preferred for DC-only. A mature exact-state
    median is the next choice. The current measured balance is only a bootstrap
    for a state that has not yet accumulated enough observations.
    """

    observations = np.asarray(tuple(learned_observations_w), dtype=np.float64)
    observations = observations[np.isfinite(observations) & (observations >= 0.0)]
    sample_count = int(observations.size)
    learned = float(learned_level_w) if learned_level_w is not None else np.nan
    component = float(component_estimate_w) if component_estimate_w is not None else np.nan
    dc_only = float(dc_only_estimate_w) if dc_only_estimate_w is not None else np.nan
    measured = float(measured_current_w)

    if np.isfinite(component):
        centre = component
        source = "fresh_pdu_components_for_current_state"
    elif str(mode).strip().lower() in {"dc-only", "dc_only"} and np.isfinite(dc_only):
        centre = dc_only
        source = "clean_dc_only_state_observation"
    elif sample_count >= int(minimum_observations) and np.isfinite(learned):
        centre = learned
        source = "learned_exact_state_distribution"
    elif np.isfinite(measured):
        centre = measured
        source = "current_state_bootstrap_observation"
    else:
        raise ValueError(f"No finite load estimate is available for operating state {mode!r}")

    centre = max(float(centre), 0.0)
    if sample_count >= int(minimum_observations):
        residuals = observations - float(np.nanmedian(observations))
        p10_residual, p90_residual = np.nanquantile(residuals, (0.10, 0.90))
        p10 = max(centre + float(p10_residual), 0.0)
        p90 = max(centre + float(p90_residual), centre)
    else:
        p10 = centre
        p90 = centre
    return ControlledLoadEstimate(p10, centre, p90, source, sample_count).validated()


def controlled_load_member_levels(
    estimate: ControlledLoadEstimate,
    member_count: int,
) -> np.ndarray:
    """Return stationary member levels spanning one state's learned uncertainty."""

    checked = estimate.validated()
    count = max(int(member_count), 1)
    if count == 1:
        return np.asarray([checked.p50_w], dtype=np.float64)
    ranks = (np.arange(count, dtype=np.float64) + 0.5) / count
    return np.interp(
        ranks,
        (0.0, 0.10, 0.50, 0.90, 1.0),
        (checked.p10_w, checked.p10_w, checked.p50_w, checked.p90_w, checked.p90_w),
    )


def validate_state_held_load(
    mode_codes: np.ndarray,
    load_values: np.ndarray,
    *,
    tolerance_w: float = 1e-4,
) -> None:
    """Require load to remain fixed between explicit operating-state changes."""

    modes = np.asarray(mode_codes)
    loads = np.asarray(load_values, dtype=np.float64)
    if modes.ndim != 1:
        raise ValueError("Operating-state codes must be one-dimensional")
    if loads.shape[-1] != modes.size:
        raise ValueError("Load time dimension must match operating-state codes")
    if not np.isfinite(loads).all():
        raise ValueError("Controlled-state load values must be finite")
    unchanged = modes[1:] == modes[:-1]
    if unchanged.any():
        differences = np.abs(np.diff(loads, axis=-1)[..., unchanged])
        if np.any(differences > float(tolerance_w)):
            raise ValueError("Forecast load changed without an operating-state transition")
