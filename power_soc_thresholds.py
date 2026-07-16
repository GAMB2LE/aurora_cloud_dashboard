"""Shared operational battery SOC thresholds and product field names."""

MINIMUM_OPERATIONAL_SOC_PCT = 40.0
MINIMUM_OPERATIONAL_SOC_LABEL = f"{MINIMUM_OPERATIONAL_SOC_PCT:g}%"
MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL = f"{MINIMUM_OPERATIONAL_SOC_LABEL} Minimum Operational SOC"

SOC_BELOW_THRESHOLD_PROBABILITY_FIELD = (
    f"BatterySOCBelow{int(MINIMUM_OPERATIONAL_SOC_PCT)}Probability"
)
SOC_BELOW_THRESHOLD_BRIER_FIELD = (
    f"ForecastSOCBelow{int(MINIMUM_OPERATIONAL_SOC_PCT)}Brier"
)

SOC_REFERENCE_PANEL_KEYS = frozenset(
    {
        "soc_24h_forecast",
        "soc_ecmwf_forecast",
        "soc_hindcast",
        "soc_load_scenarios",
    }
)
