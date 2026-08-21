#!/usr/bin/env python3
"""Canonical operator-defined power states used by the APS load learner.

The UAS tier and CL61 heater are independent subsystem states.  They are
composed with the measured PDU instrument configuration when the learner builds
an exact whole-station load state; they are not additional PDU commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


UAS_CHARGE_ESTIMATE_W = 300.0
UAS_CHARGE_DURATION_HOURS = 3.0
UAS_CHARGE_EVENT_KIT = "UASCharge"

# Tier 11 is the safe field proxy for Tier 1.  Tier 12 cycles the heaters and
# is the field proxy for Tier 2.  Raw 1/2 values are still recognised for live
# presentation, but they do not train the canonical profiles.
UAS_TIER_LEARNING_SOURCES: dict[int, tuple[int, ...]] = {
    1: (11,),
    2: (12,),
    3: (3,),
    4: (4,),
    5: (5,),
}
UAS_TIER_ALIASES: dict[int, int] = {
    1: 1,
    11: 1,
    2: 2,
    12: 2,
    3: 3,
    4: 4,
    5: 5,
}
UAS_CHARGE_TIERS = (1, 2, 3)


@dataclass(frozen=True)
class LearnedPowerStateDefinition:
    state_id: str
    label: str
    subsystem: str
    source_effective_tiers: tuple[int, ...] = ()
    base_state_id: str | None = None
    estimated_increment_w: float | None = None
    estimated_duration_hours: float | None = None
    cl61_phase: str | None = None


def uas_state_id(tier: int, *, charging: bool = False) -> str:
    suffix = "_charging" if charging else ""
    return f"uas_tier_{int(tier)}{suffix}"


def uas_state_label(tier: int, *, charging: bool = False) -> str:
    tier = int(tier)
    if tier == 4:
        base = "Tier 4 (12 V only)"
    elif tier == 5:
        base = "Tier 5 (all off)"
    else:
        base = f"Tier {tier}"
    return f"{base} + UAS Charging" if charging else base


def canonical_uas_tier(value: object) -> int | None:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return None
    return UAS_TIER_ALIASES.get(numeric)


def tier_is_learning_source(raw_tier: object, canonical_tier: int) -> bool:
    try:
        numeric = int(float(raw_tier))
    except (TypeError, ValueError):
        return False
    return numeric in UAS_TIER_LEARNING_SOURCES.get(int(canonical_tier), ())


def operating_load_state_id(
    mode: str,
    *,
    uas_tier: int | None = None,
    uas_charging: bool = False,
    cl61_heater_on: bool = False,
) -> str:
    """Compose orthogonal subsystem states with one measured PDU mode."""
    parts = [str(mode)]
    if uas_tier is not None:
        parts.append(uas_state_id(int(uas_tier), charging=uas_charging))
    if cl61_heater_on:
        parts.append("cl61_heater_on")
    return "__".join(parts)


LEARNED_POWER_STATES = (
    LearnedPowerStateDefinition(
        uas_state_id(1),
        uas_state_label(1),
        "uas",
        source_effective_tiers=UAS_TIER_LEARNING_SOURCES[1],
    ),
    LearnedPowerStateDefinition(
        uas_state_id(1, charging=True),
        uas_state_label(1, charging=True),
        "uas",
        base_state_id=uas_state_id(1),
        estimated_increment_w=UAS_CHARGE_ESTIMATE_W,
        estimated_duration_hours=UAS_CHARGE_DURATION_HOURS,
    ),
    LearnedPowerStateDefinition(
        uas_state_id(2),
        uas_state_label(2),
        "uas",
        source_effective_tiers=UAS_TIER_LEARNING_SOURCES[2],
    ),
    LearnedPowerStateDefinition(
        uas_state_id(2, charging=True),
        uas_state_label(2, charging=True),
        "uas",
        base_state_id=uas_state_id(2),
        estimated_increment_w=UAS_CHARGE_ESTIMATE_W,
        estimated_duration_hours=UAS_CHARGE_DURATION_HOURS,
    ),
    LearnedPowerStateDefinition(
        uas_state_id(3),
        uas_state_label(3),
        "uas",
        source_effective_tiers=UAS_TIER_LEARNING_SOURCES[3],
    ),
    LearnedPowerStateDefinition(
        uas_state_id(3, charging=True),
        uas_state_label(3, charging=True),
        "uas",
        base_state_id=uas_state_id(3),
        estimated_increment_w=UAS_CHARGE_ESTIMATE_W,
        estimated_duration_hours=UAS_CHARGE_DURATION_HOURS,
    ),
    LearnedPowerStateDefinition(
        uas_state_id(4),
        uas_state_label(4),
        "uas",
        source_effective_tiers=UAS_TIER_LEARNING_SOURCES[4],
    ),
    LearnedPowerStateDefinition(
        uas_state_id(5),
        uas_state_label(5),
        "uas",
        source_effective_tiers=UAS_TIER_LEARNING_SOURCES[5],
    ),
    LearnedPowerStateDefinition("cl61", "CL61", "cl61", cl61_phase="fan_low"),
    LearnedPowerStateDefinition(
        "cl61_heater_on",
        "CL61 (heater on)",
        "cl61",
        cl61_phase="fan_high",
    ),
)
LEARNED_POWER_STATE_IDS = tuple(value.state_id for value in LEARNED_POWER_STATES)


@dataclass(frozen=True)
class PowerStateScenarioDefinition:
    scenario_id: str
    state_id: str
    label: str
    instruments: tuple[str, ...]
    uas_tier: int | None = None
    uas_charging: bool = False
    cl61_phase: str | None = None


POWER_STATE_SCENARIOS = tuple(
    PowerStateScenarioDefinition(
        scenario_id=f"state_{definition.state_id}",
        state_id=definition.state_id,
        label=definition.label,
        instruments=("UAS",),
        uas_tier=tier,
        uas_charging=charging,
    )
    for tier in range(1, 6)
    for charging in ((False, True) if tier in UAS_CHARGE_TIERS else (False,))
    for definition in (
        next(
            value
            for value in LEARNED_POWER_STATES
            if value.state_id == uas_state_id(tier, charging=charging)
        ),
    )
) + (
    PowerStateScenarioDefinition(
        "state_cl61",
        "cl61",
        "CL61",
        ("CL61",),
        cl61_phase="fan_low",
    ),
    PowerStateScenarioDefinition(
        "state_cl61_heater_on",
        "cl61_heater_on",
        "CL61 (heater on)",
        ("CL61",),
        cl61_phase="fan_high",
    ),
)
POWER_STATE_SCENARIO_IDS = tuple(value.scenario_id for value in POWER_STATE_SCENARIOS)


def state_catalog_records() -> list[dict[str, object]]:
    return [
        {
            "id": value.state_id,
            "label": value.label,
            "subsystem": value.subsystem,
            "source_effective_tiers": list(value.source_effective_tiers),
            "base_state_id": value.base_state_id,
            "estimated_increment_w": value.estimated_increment_w,
            "estimated_duration_hours": value.estimated_duration_hours,
            "cl61_phase": value.cl61_phase,
        }
        for value in LEARNED_POWER_STATES
    ]


def state_ids(values: Iterable[LearnedPowerStateDefinition] = LEARNED_POWER_STATES) -> tuple[str, ...]:
    return tuple(value.state_id for value in values)
