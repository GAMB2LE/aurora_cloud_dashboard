#!/usr/bin/env python3
"""Stable operating scenarios shared by forecast generation and presentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingScenarioDefinition:
    scenario_id: str
    instruments: tuple[str, ...]
    label: str


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
)

SUGGESTED_OPERATING_SCENARIO_IDS = tuple(
    definition.scenario_id for definition in SUGGESTED_OPERATING_SCENARIOS
)
