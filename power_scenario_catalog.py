#!/usr/bin/env python3
"""Stable operating scenarios shared by forecast generation and presentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingScenarioDefinition:
    scenario_id: str
    instruments: tuple[str, ...]
    label: str
    uas_effective_tier: int | None = None


@dataclass(frozen=True)
class UASTierScenarioDefinition:
    """One standard Menapia tier used in the UAS-only comparison panel."""

    scenario_id: str
    tier: int
    label: str
    fallback_p10_w: float
    fallback_p50_w: float
    fallback_p90_w: float
    station_powered: bool = True


SUGGESTED_OPERATING_SCENARIOS = (
    OperatingScenarioDefinition("cl61_continuous", ("CL61",), "CL61"),
    OperatingScenarioDefinition("suggested_cl61_radar", ("CL61", "Radar"), "CL61 + Radar"),
    OperatingScenarioDefinition("suggested_cl61_hatpro", ("CL61", "HATPRO"), "CL61 + HATPRO"),
    OperatingScenarioDefinition(
        "suggested_cl61_hatpro_radar",
        ("CL61", "HATPRO", "Radar"),
        "CL61 + HATPRO + Radar",
    ),
    OperatingScenarioDefinition("suggested_hatpro_radar", ("HATPRO", "Radar"), "HATPRO + Radar"),
    OperatingScenarioDefinition("suggested_radar", ("Radar",), "Radar"),
    OperatingScenarioDefinition("suggested_hatpro", ("HATPRO",), "HATPRO"),
    OperatingScenarioDefinition(
        "suggested_all_uas_tier3",
        ("CL61", "Radar", "HATPRO", "UAS"),
        "All instruments + UAS tier 3",
        uas_effective_tier=3,
    ),
)

SUGGESTED_OPERATING_SCENARIO_IDS = tuple(
    definition.scenario_id for definition in SUGGESTED_OPERATING_SCENARIOS
)


# The documented standard operating tiers are 1-5. Tiers 11 and 12 are
# diagnostic modes which mimic tiers 1 and 2, so they are intentionally not
# presented as independent operational forecasts. The fallback distributions
# keep the comparison available before each tier passes the empirical evidence
# gate; reliable learned quantiles replace them automatically.
UAS_TIER_SCENARIOS = (
    UASTierScenarioDefinition(
        "uas_tier_1",
        1,
        "Tier 1 - Unrestricted",
        300.0,
        375.0,
        800.0,
    ),
    UASTierScenarioDefinition(
        "uas_tier_2",
        2,
        "Tier 2 - Flight operations",
        120.0,
        160.0,
        590.0,
    ),
    UASTierScenarioDefinition(
        "uas_tier_3",
        3,
        "Tier 3 - Heating disabled",
        55.0,
        108.0,
        302.0,
    ),
    UASTierScenarioDefinition(
        "uas_tier_4",
        4,
        "Tier 4 - 12 V standby",
        24.0,
        32.0,
        40.0,
    ),
    UASTierScenarioDefinition(
        "uas_tier_5",
        5,
        "Tier 5 - Internal battery only",
        0.0,
        0.0,
        0.0,
        station_powered=False,
    ),
)

UAS_TIER_SCENARIO_IDS = tuple(definition.scenario_id for definition in UAS_TIER_SCENARIOS)
