"""Regression for the captured 7 September 06--09 UTC replay interval."""
import numpy as np
import pandas as pd
import pytest

from generate_power_soc_forecast import (
    _apply_soc_bias_corrections, validate_soc_physical_consistency,
)


EFFICIENCIES = {"charge_efficiency": 0.884466, "discharge_efficiency": 0.947819}


def captured_interval():
    return pd.DataFrame({
        "BatterySOCForecast": [50., 50. - 0.13188438439999572],
        "ForecastBatteryChargeInputWatts": [np.nan, 111.8898056923],
        "ForecastBatteryDischargeOutputWatts": [np.nan, 104.2834681598],
    }, index=pd.to_datetime(["2026-09-07T06:00", "2026-09-07T09:00"]))


def test_captured_interval_checks_stored_energy_not_terminal_net_power():
    frame = captured_interval()
    # The former unity-efficiency check falsely rejected this correct fall.
    with pytest.raises(ValueError, match="falls without"):
        validate_soc_physical_consistency(frame)
    validate_soc_physical_consistency(frame, **EFFICIENCIES)
    energy_w = 111.8898056923 * EFFICIENCIES["charge_efficiency"] - 104.2834681598 / EFFICIENCIES["discharge_efficiency"]
    assert energy_w == pytest.approx(-11.061936333959238)
    assert 100 * energy_w * 3 / (1000 * 25.163) == pytest.approx(-0.1318843844, abs=2e-6)


def test_efficiency_aware_guard_still_rejects_unphysical_rise():
    frame = captured_interval()
    frame.iloc[1, 0] = 50.1
    with pytest.raises(ValueError, match="rises without"):
        validate_soc_physical_consistency(frame, **EFFICIENCIES)


def test_bias_correction_can_only_attenuate_efficiency_adjusted_discharge():
    frame = captured_interval()
    corrected = _apply_soc_bias_corrections(frame, {"0_6h": 0.1},
        issue_time=frame.index[0], **EFFICIENCIES)
    assert frame.iloc[1, 0] < corrected.iloc[1, 0] <= frame.iloc[0, 0]
    validate_soc_physical_consistency(corrected, **EFFICIENCIES)


@pytest.mark.parametrize("efficiency", [0, -0.1, 1.01, np.nan, np.inf])
def test_invalid_efficiency_fails_closed(efficiency):
    with pytest.raises(ValueError, match="efficiencies"):
        validate_soc_physical_consistency(captured_interval(), charge_efficiency=efficiency)
