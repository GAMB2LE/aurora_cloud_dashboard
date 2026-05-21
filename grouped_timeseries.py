#!/usr/bin/env python3
"""Summary and housekeeping plotting helpers for 1D Aurora instruments.

The helpers in this module define the curated panel layouts, human-readable
labels, static quicklook PNG generation, and interactive Plotly summaries used
by the Meteorology, Radiation, Aurora Power Supply, and Operations views.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from quicklook_time_axis import apply_quicklook_time_axis
from time_gap_breaks import insert_time_gap_breaks

MAX_TIME_SAMPLES = int(os.environ.get("AURORA_QUICKLOOK_MAX_TIME_SAMPLES", "2200"))
INTERACTIVE_MAX_TIME_SAMPLES = int(os.environ.get("AURORA_INTERACTIVE_MAX_TIME_SAMPLES", "1600"))
OVERVIEW_LABEL = "Overview"
# Reserve a fixed right-side gutter for per-panel legends so they sit beyond the
# secondary-axis labels in both the interactive Plotly view and saved PNGs.
MATPLOTLIB_PANEL_RIGHT = 0.72
MATPLOTLIB_LEGEND_X = 1.12
PLOTLY_SUMMARY_PANEL_DOMAIN_END = 0.78
PLOTLY_SUMMARY_LEGEND_X = 0.91
PLOTLY_SUMMARY_RIGHT_MARGIN = 110
PLOTLY_SUMMARY_PANEL_HEIGHT = 225
PLOTLY_SUMMARY_POWER_PANEL_HEIGHT = 250
PLOTLY_SUMMARY_MAX_HEIGHT = 1650
PLOTLY_SUMMARY_POWER_MAX_HEIGHT = 1850
MATPLOTLIB_Y_HEADROOM_FRACTION = 0.28
MATPLOTLIB_Y_FOOTROOM_FRACTION = 0.04
SUMMARY_DISPLAY_START_ATTR = "summary_display_start"
SUMMARY_DISPLAY_END_ATTR = "summary_display_end"
POWER_CUMULATIVE_CONTEXT_DAYS = int(os.environ.get("AURORA_POWER_CUMULATIVE_CONTEXT_DAYS", "7"))
POWER_DISPLAY_ENERGY_FREQ = os.environ.get("AURORA_POWER_DISPLAY_ENERGY_FREQ", "1min")
POWER_DISPLAY_SUMMARY_FREQ = os.environ.get("AURORA_POWER_DISPLAY_SUMMARY_FREQ", POWER_DISPLAY_ENERGY_FREQ)
POWER_DISPLAY_ENERGY_ATTR = "power_display_energy_product"
POWER_DISPLAY_SUMMARY_ATTR = "power_display_summary_product"
POWER_DISPLAY_ENERGY_MAP = {
    "SolarYield_East": "PowerDisplaySolarYield_East",
    "SolarYield_South": "PowerDisplaySolarYield_South",
    "SolarYield_West": "PowerDisplaySolarYield_West",
    "CumulativePowerGeneratedTotal": "PowerDisplayCumulativePowerGeneratedTotal",
    "CumulativePowerUtilised": "PowerDisplayCumulativePowerUtilised",
}
POWER_DISPLAY_SUMMARY_FIELDS = (
    "SolarWatts_East",
    "SolarWatts_South",
    "SolarWatts_West",
    "SolarVolts_East",
    "SolarVolts_South",
    "SolarVolts_West",
    "BatteryAmps",
    "BatteryWatts",
    "ACOutputWatts",
    "DCInverterWatts",
    "BatterySOC",
    "ACOutputVolts",
    "DCInverterVolts",
    "InternalTemperature",
    "HeatsinkTemperature",
    "TempSensor1",
    "TempSensor2",
    "TempSensor3",
    "TempSensor4",
)
POWER_DISPLAY_SUMMARY_CONTEXT_FIELDS = ("watts_on_48vdc_Avg",)
FAST_SONIC_TO_LOGGER_AVG = {
    "metek_x_out": "metek_x_out_Avg",
    "metek_y_out": "metek_y_out_Avg",
    "metek_z_out": "metek_z_out_Avg",
    "metek_T_out": "metek_T_out_Avg",
    "metek_InclX_out": "metek_InclX_out_Avg",
    "metek_InclY_out": "metek_InclY_out_Avg",
}
FAST_GAS_TO_LOGGER_AVG = {
    "licor_co2_out": "licor_co2_out_Avg",
    "licor_h2o_out": "licor_h2o_out_Avg",
    "licor_pr_out": "licor_pr_out_Avg",
    "licor_t_out": "licor_t_out_Avg",
    "licor_diag_out": "licor_diag_out_Avg",
    "licor_co2_str_out": "licor_co2_str_out_Avg",
}


@dataclass(frozen=True)
class TraceSpec:
    var: str
    label: str
    color: str
    axis: str = "left"
    scale: float = 1.0
    dash: str | None = None
    step: bool = False
    valid_min: float | None = None
    valid_max: float | None = None
    skip_if_all_zero: bool = False
    smooth_minutes: float | None = None
    break_on_day_change: bool = False


@dataclass(frozen=True)
class PanelSpec:
    key: str
    label: str
    left_axis_label: str
    right_axis_label: str | None
    traces: tuple[TraceSpec, ...]


COLOR = {
    "teal": "#0b7285",
    "light_blue": "#7fb6d6",
    "blue": "#4d6fb3",
    "purple": "#7768b8",
    "brown": "#4f7d8d",
    "olive": "#7a9964",
    "red": "#c05647",
    "magenta": "#9f6b9f",
    "green": "#4f8c63",
    "slate": "#718195",
    "black": "#22313f",
}

PLOT_TEXT = "#22313f"
PLOT_LINE = "#c5d0da"
PLOT_GRID = "#e5eaef"
PLOT_BORDER = "#d8e1e8"


SUMMARY_INSTRUMENTS = ("vaisalamet", "asfs-logger", "asfs-fast-sonic", "power", "ops-monitor")

DISPLAY_NAMES = {
    "vaisalamet": "Meteorology",
    "asfs-logger": "Radiation",
    "asfs-fast-sonic": "ASFS Fast Sonic",
    "power": "Aurora Power Supply",
    "ops-monitor": "Operations",
}

HOUSEKEEPING_LABELS = {
    "vaisalamet": "HK_Met",
    "asfs-logger": "HK_ASFS",
    "power": "HK_APS",
    "ops-monitor": "HK_Operations",
}

QUICKLOOK_PREFIX = {
    "vaisalamet": "vaisalamet",
    "asfs-logger": "asfs_logger",
    "asfs-fast-sonic": "asfs_fast_sonic",
    "power": "power",
    "ops-monitor": "ops_monitor",
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

HUMAN_LABELS = {
    "baro_hPa": "Pressure",
    "h1_t": "HMP1 Air Temperature",
    "t2_t": "T2 Air Temperature",
    "h1_td": "Dew Point",
    "h1_rh": "Relative Humidity",
    "h1_e": "Vapor Pressure",
    "h1_ah": "Absolute Humidity",
    "h1_mr": "Mixing Ratio",
    "h1_online": "HMP1 Online",
    "t2_online": "T2 Online",
    "h1_error_status": "HMP1 Error Status",
    "t2_error_status": "T2 Error Status",
    "baro_err_pressure_meas_err": "Pressure Measurement Error",
    "baro_err_pressure_oor": "Pressure Out of Range",
    "baro_st_sensor_failure": "Pressure Sensor Failure",
    "baro_st_value_locked": "Pressure Value Locked",
    "batt_volt_Avg": "Battery Voltage",
    "amp_meter_48vdc_Avg": "48 V Current",
    "watts_on_48vdc_Avg": "48 V Power",
    "PTemp_Avg": "Panel Temperature",
    "metek_x_out_Avg": "Metek U Wind",
    "metek_y_out_Avg": "Metek V Wind",
    "metek_z_out_Avg": "Metek W Wind",
    "metek_T_out_Avg": "Sonic Temperature",
    "metek_InclX_out_Avg": "Metek Tilt X",
    "metek_InclY_out_Avg": "Metek Tilt Y",
    "MetekWindSpeed": "Metek Wind Speed",
    "MetekWindDirection": "Metek Wind Direction",
    "spn1_tot_Avg": "Total Radiation",
    "spn1_dif_Avg": "Diffuse Radiation",
    "sr30_swd_Irr_Avg": "Downwelling Shortwave",
    "sr30_swu_Irr_Avg": "Upwelling Shortwave",
    "sr30_swd_DegC_Avg": "Downwelling SR30 Body Temperature",
    "sr30_swu_DegC_Avg": "Upwelling SR30 Body Temperature",
    "sr30_swd_tilt_Avg": "Downwelling SR30 Tilt",
    "sr30_swu_tilt_Avg": "Upwelling SR30 Tilt",
    "sr30_swd_rot_Avg": "Downwelling SR30 Rotation",
    "sr30_swu_rot_Avg": "Upwelling SR30 Rotation",
    "sr30_swd_fantach_Avg": "Downwelling SR30 Fan Tach",
    "sr30_swu_fantach_Avg": "Upwelling SR30 Fan Tach",
    "sr30_swd_fanstate_Avg": "Downwelling SR30 Fan State",
    "sr30_swu_fanstate_Avg": "Upwelling SR30 Fan State",
    "sr30_swd_heatstate_Avg": "Downwelling SR30 Heater State",
    "sr30_swu_heatstate_Avg": "Upwelling SR30 Heater State",
    "ir20_lwd_Wm2_Avg": "Downwelling Longwave",
    "ir20_lwu_Wm2_Avg": "Upwelling Longwave",
    "ir20_lwd_DegC_Avg": "Downwelling IR20 Body Temperature",
    "ir20_lwu_DegC_Avg": "Upwelling IR20 Body Temperature",
    "ir20_lwd_fan_Avg": "Downwelling IR20 Fan",
    "ir20_lwu_fan_Avg": "Upwelling IR20 Fan",
    "fp_A_Wm2_Avg": "Flux Plate A",
    "fp_B_Wm2_Avg": "Flux Plate B",
    "sr50_dist_Avg": "SR50 Distance",
    "sr50_qc_Avg": "SR50 Quality",
    "kt15_amb_Avg": "KT15 Ambient Temperature",
    "kt15_tem_Avg": "KT15 Surface Temperature",
    "licor_co2_out_Avg": "LI-COR CO2",
    "licor_h2o_out_Avg": "LI-COR H2O",
    "licor_pr_out_Avg": "LI-COR Pressure",
    "licor_t_out_Avg": "LI-COR Temperature",
    "licor_diag_out_Avg": "LI-COR Diagnostic",
    "licor_co2_str_out_Avg": "LI-COR CO2 Strength",
    "vaisala_T_Avg": "ASFS Vaisala Temperature",
    "vaisala_RH_Avg": "ASFS Vaisala Relative Humidity",
    "vaisala_P_Avg": "ASFS Vaisala Pressure",
    "metek_x_out": "Metek U Wind",
    "metek_y_out": "Metek V Wind",
    "metek_z_out": "Metek W Wind",
    "metek_T_out": "Sonic Temperature",
    "metek_InclX_out": "Tilt X",
    "metek_InclY_out": "Tilt Y",
    "metek_quality_out": "Metek Quality",
    "metek_senspathstate_out": "Sensor Path State",
    "ACOutputWatts": "AC Output Power",
    "DCInverterWatts": "DC Inverter Power",
    "ACOutputVolts": "AC Output Voltage",
    "DCInverterVolts": "DC Inverter Voltage",
    "BatteryWatts": "Battery Power",
    "BatteryAmps": "Battery Current",
    "BatterySOC": "State of Charge",
    "BatteryState": "Battery State",
    "BattsOnline": "Batteries Online",
    "InternalTemperature": "Internal Temperature",
    "HeatsinkTemperature": "Heatsink Temperature",
    "TempSensor1": "Temperature Sensor 1",
    "TempSensor2": "Temperature Sensor 2",
    "TempSensor3": "Temperature Sensor 3",
    "TempSensor4": "Temperature Sensor 4",
    "SolarWatts_East": "Solar East Power",
    "SolarWatts_South": "Solar South Power",
    "SolarWatts_West": "Solar West Power",
    "SolarVolts_East": "Solar East Voltage",
    "SolarVolts_South": "Solar South Voltage",
    "SolarVolts_West": "Solar West Voltage",
    "SolarAmps_East": "Solar East Current",
    "SolarAmps_South": "Solar South Current",
    "SolarAmps_West": "Solar West Current",
    "SolarYield_East": "East Solar Generated",
    "SolarYield_South": "South Solar Generated",
    "SolarYield_West": "West Solar Generated",
    "PowerDisplaySolarYield_East": "East Solar Generated",
    "PowerDisplaySolarYield_South": "South Solar Generated",
    "PowerDisplaySolarYield_West": "West Solar Generated",
    "CumulativePowerGeneratedTotal": "Total Generated",
    "CumulativePowerUtilised": "Power Utilised",
    "PowerDisplayCumulativePowerGeneratedTotal": "Total Generated",
    "PowerDisplayCumulativePowerUtilised": "Power Utilised",
    "SolarState_East": "Solar East State",
    "SolarState_South": "Solar South State",
    "SolarState_West": "Solar West State",
    "AlarmBits": "Alarm Bits",
    "FaultBits": "Fault Bits",
    "HeatsinkTempAlarm": "Heatsink Alarm",
    "InternalTempAlarm": "Internal Alarm",
    "time_discrepancy": "Clock Discrepancy",
}

HUMAN_UNITS = {
    "baro_hPa": "hPa",
    "h1_t": "C",
    "t2_t": "C",
    "h1_td": "C",
    "h1_rh": "%",
    "h1_e": "hPa",
    "h1_ah": "g m^-3",
    "h1_mr": "g kg^-1",
    "h1_online": "state",
    "t2_online": "state",
    "h1_error_status": "state",
    "t2_error_status": "state",
    "baro_err_pressure_meas_err": "state",
    "baro_err_pressure_oor": "state",
    "baro_st_sensor_failure": "state",
    "baro_st_value_locked": "state",
    "batt_volt_Avg": "V",
    "amp_meter_48vdc_Avg": "A",
    "watts_on_48vdc_Avg": "W",
    "PTemp_Avg": "C",
    "metek_x_out_Avg": "m s^-1",
    "metek_y_out_Avg": "m s^-1",
    "metek_z_out_Avg": "m s^-1",
    "metek_T_out_Avg": "C",
    "metek_InclX_out_Avg": "deg",
    "metek_InclY_out_Avg": "deg",
    "MetekWindSpeed": "m s^-1",
    "MetekWindDirection": "deg",
    "spn1_tot_Avg": "W m^-2",
    "spn1_dif_Avg": "W m^-2",
    "sr30_swd_Irr_Avg": "W m^-2",
    "sr30_swu_Irr_Avg": "W m^-2",
    "sr30_swd_DegC_Avg": "C",
    "sr30_swu_DegC_Avg": "C",
    "sr30_swd_tilt_Avg": "deg",
    "sr30_swu_tilt_Avg": "deg",
    "sr30_swd_rot_Avg": "deg",
    "sr30_swu_rot_Avg": "deg",
    "sr30_swd_fantach_Avg": "Hz",
    "sr30_swu_fantach_Avg": "Hz",
    "sr30_swd_fanstate_Avg": "state",
    "sr30_swu_fanstate_Avg": "state",
    "sr30_swd_heatstate_Avg": "state",
    "sr30_swu_heatstate_Avg": "state",
    "ir20_lwd_Wm2_Avg": "W m^-2",
    "ir20_lwu_Wm2_Avg": "W m^-2",
    "ir20_lwd_DegC_Avg": "C",
    "ir20_lwu_DegC_Avg": "C",
    "ir20_lwd_fan_Avg": "Hz",
    "ir20_lwu_fan_Avg": "Hz",
    "fp_A_Wm2_Avg": "W m^-2",
    "fp_B_Wm2_Avg": "W m^-2",
    "sr50_dist_Avg": "m",
    "sr50_qc_Avg": "state",
    "kt15_amb_Avg": "C",
    "kt15_tem_Avg": "C",
    "licor_co2_out_Avg": "mmol m^-3",
    "licor_h2o_out_Avg": "mmol m^-3",
    "licor_pr_out_Avg": "kPa",
    "licor_t_out_Avg": "C",
    "licor_diag_out_Avg": "code",
    "licor_co2_str_out_Avg": "%",
    "vaisala_T_Avg": "C",
    "vaisala_RH_Avg": "%",
    "vaisala_P_Avg": "hPa",
    "metek_x_out": "m s^-1",
    "metek_y_out": "m s^-1",
    "metek_z_out": "m s^-1",
    "metek_T_out": "C",
    "metek_InclX_out": "deg",
    "metek_InclY_out": "deg",
    "metek_msec_out": "ms",
    "metek_quality_out": "state",
    "metek_senspathstate_out": "state",
    "ACOutputAmps": "A",
    "ACOutputHZ": "Hz",
    "ACOutputVolts": "V",
    "ACOutputWatts": "W",
    "ACkWh": "kWh",
    "ACnHours": "h",
    "BatteryAmps": "A",
    "BatterySOC": "%",
    "BatteryState": "state",
    "BatteryWatts": "W",
    "BattsOnline": "state",
    "DCInverterAmps": "A",
    "DCInverterVolts": "V",
    "DCInverterWatts": "W",
    "FaultBits": "bits",
    "AlarmBits": "bits",
    "HeatsinkTempAlarm": "state",
    "HeatsinkTemperature": "C",
    "InternalTempAlarm": "state",
    "InternalTemperature": "C",
    "MaxSolarWatts_East": "W",
    "MaxSolarWatts_South": "W",
    "MaxSolarWatts_West": "W",
    "SolarAmps_East": "A",
    "SolarAmps_South": "A",
    "SolarAmps_West": "A",
    "SolarState_East": "state",
    "SolarState_South": "state",
    "SolarState_West": "state",
    "SolarVolts_East": "V",
    "SolarVolts_South": "V",
    "SolarVolts_West": "V",
    "SolarWatts_East": "W",
    "SolarWatts_South": "W",
    "SolarWatts_West": "W",
    "SolarYield_East": "kWh",
    "SolarYield_South": "kWh",
    "SolarYield_West": "kWh",
    "PowerDisplaySolarYield_East": "kWh",
    "PowerDisplaySolarYield_South": "kWh",
    "PowerDisplaySolarYield_West": "kWh",
    "CumulativePowerGeneratedTotal": "kWh",
    "CumulativePowerUtilised": "kWh",
    "PowerDisplayCumulativePowerGeneratedTotal": "kWh",
    "PowerDisplayCumulativePowerUtilised": "kWh",
    "TempSensor1": "C",
    "TempSensor2": "C",
    "TempSensor3": "C",
    "TempSensor4": "C",
    "TotCapacity": "capacity units",
    "time_discrepancy": "s",
    "scantime": "s",
}

DISPLAY_SCALE = {}

SUMMARY_SOURCE_INSTRUMENTS = {
    "vaisalamet": ("vaisalamet", "asfs-logger", "asfs-fast-sonic"),
    "asfs-logger": ("asfs-logger",),
    "asfs-fast-sonic": ("asfs-fast-sonic",),
    "power": ("power", "asfs-logger"),
    "ops-monitor": ("ops-monitor",),
}

SUMMARY_LAYOUTS: dict[str, tuple[PanelSpec, ...]] = {
    "vaisalamet": (
        PanelSpec(
            "air_temperature",
            "Air Temperature",
            "Air Temperature [C]",
            None,
            (
                TraceSpec("h1_t", "HMP1 Air Temperature", COLOR["teal"]),
                TraceSpec("t2_t", "T2 Air Temperature", COLOR["light_blue"]),
                TraceSpec("vaisala_T_Avg", "ASFS Vaisala Temperature", COLOR["green"]),
                TraceSpec("metek_T_out_Avg", "Sonic Temperature", COLOR["brown"]),
            ),
        ),
        PanelSpec(
            "humidity",
            "Humidity / Dew Point",
            "Dew Point [C]",
            "Relative Humidity [%]",
            (
                TraceSpec("h1_td", "Dew Point", COLOR["purple"]),
                TraceSpec("h1_rh", "Relative Humidity", COLOR["brown"], axis="right"),
                TraceSpec("vaisala_RH_Avg", "ASFS Vaisala Relative Humidity", COLOR["green"], axis="right"),
            ),
        ),
        PanelSpec(
            "pressure",
            "Pressure",
            "Pressure [hPa]",
            None,
            (
                TraceSpec("baro_hPa", "Pressure", COLOR["green"]),
                TraceSpec("vaisala_P_Avg", "ASFS Vaisala Pressure", COLOR["teal"]),
            ),
        ),
        PanelSpec(
            "met",
            "Met",
            "Metek U / V Wind [m/s]",
            "Metek W Wind [m/s]",
            (
                TraceSpec("metek_x_out_Avg", "Metek U Wind", COLOR["teal"]),
                TraceSpec("metek_y_out_Avg", "Metek V Wind", COLOR["light_blue"]),
                TraceSpec("metek_z_out_Avg", "Metek W Wind", COLOR["purple"], axis="right"),
            ),
        ),
        PanelSpec(
            "metek_wind_speed_direction",
            "Metek Wind Speed / Direction",
            "Wind Speed [m/s]",
            "Wind Direction [deg]",
            (
                TraceSpec("MetekWindSpeed", "Wind Speed", COLOR["teal"], valid_min=0.0, valid_max=100.0),
                TraceSpec("MetekWindDirection", "Wind Direction", COLOR["purple"], axis="right", valid_min=0.0, valid_max=360.0),
            ),
        ),
    ),
    "asfs-logger": (
        PanelSpec(
            "shortwave_radiation",
            "Shortwave Radiation",
            "Radiation [W m^-2]",
            None,
            (
                TraceSpec("spn1_tot_Avg", "Total Radiation", COLOR["brown"]),
                TraceSpec("spn1_dif_Avg", "Diffuse Radiation", COLOR["purple"]),
                TraceSpec("sr30_swd_Irr_Avg", "Downwelling Shortwave", COLOR["brown"]),
                TraceSpec("sr30_swu_Irr_Avg", "Upwelling Shortwave", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "longwave_radiation",
            "Longwave Radiation",
            "Radiation [W m^-2]",
            None,
            (
                TraceSpec("ir20_lwd_Wm2_Avg", "Downwelling Longwave", COLOR["teal"]),
                TraceSpec("ir20_lwu_Wm2_Avg", "Upwelling Longwave", COLOR["light_blue"]),
            ),
        ),
        PanelSpec(
            "flux_plates",
            "Flux Plates",
            "Flux [W m^-2]",
            None,
            (
                TraceSpec("fp_A_Wm2_Avg", "Flux Plate A", COLOR["teal"]),
                TraceSpec("fp_B_Wm2_Avg", "Flux Plate B", COLOR["light_blue"]),
            ),
        ),
        PanelSpec(
            "surface_temperature",
            "Surface / Snow",
            "KT15 Surface Temperature [C]",
            "SR50 Distance [m]",
            (
                TraceSpec("kt15_tem_Avg", "KT15 Surface Temperature", COLOR["magenta"]),
                TraceSpec("sr50_dist_Avg", "SR50 Distance", COLOR["olive"], axis="right"),
            ),
        ),
    ),
    "asfs-logger-hk": (
        PanelSpec(
            "logger_power",
            "Logger Power",
            "Voltage [V]",
            "Current [A] / Power [W]",
            (
                TraceSpec("batt_volt_Avg", "Battery Voltage", COLOR["teal"]),
                TraceSpec("amp_meter_48vdc_Avg", "48 V Current", COLOR["purple"], axis="right"),
                TraceSpec("watts_on_48vdc_Avg", "48 V Power", COLOR["brown"], axis="right"),
            ),
        ),
        PanelSpec(
            "logger_thermal_scan",
            "Logger Thermal / Scan",
            "Panel Temperature [C]",
            "Scan Time [s]",
            (
                TraceSpec("PTemp_Avg", "Panel Temperature", COLOR["brown"]),
                TraceSpec("scantime", "Scan Time", COLOR["slate"], axis="right"),
            ),
        ),
        PanelSpec(
            "asfs_met_licor",
            "ASFS Met / LI-COR",
            "CO2 / H2O [mmol m^-3]",
            "Signal Strength [%]",
            (
                TraceSpec("licor_co2_out_Avg", "CO2 Output", COLOR["teal"]),
                TraceSpec("licor_h2o_out_Avg", "H2O Output", COLOR["light_blue"]),
                TraceSpec("licor_co2_str_out_Avg", "CO2 Signal Strength", COLOR["green"], axis="right", valid_min=0.0, valid_max=100.0),
            ),
        ),
        PanelSpec(
            "licor_diagnostics",
            "LI-COR Diagnostics",
            "Diagnostic Code",
            None,
            (
                TraceSpec("licor_diag_out_Avg", "Diagnostic Code", COLOR["red"], step=True),
            ),
        ),
        PanelSpec(
            "shortwave_mechanics",
            "SR30 Orientation",
            "Tilt [deg]",
            "Rotation [deg]",
            (
                TraceSpec("sr30_swd_tilt_Avg", "Downwelling Tilt", COLOR["teal"], valid_min=-5.0, valid_max=20.0),
                TraceSpec("sr30_swu_tilt_Avg", "Upwelling Tilt", COLOR["light_blue"], valid_min=150.0, valid_max=200.0),
                TraceSpec("sr30_swd_rot_Avg", "Downwelling Rotation", COLOR["purple"], axis="right", valid_min=-20.0, valid_max=360.0),
                TraceSpec("sr30_swu_rot_Avg", "Upwelling Rotation", COLOR["olive"], axis="right", valid_min=-20.0, valid_max=360.0),
            ),
        ),
        PanelSpec(
            "shortwave_support",
            "SR30 Fans / Heaters",
            "Fan Tach [Hz]",
            "State",
            (
                TraceSpec("sr30_swd_fantach_Avg", "Downwelling Fan Tach", COLOR["teal"]),
                TraceSpec("sr30_swu_fantach_Avg", "Upwelling Fan Tach", COLOR["light_blue"]),
                TraceSpec("sr30_swd_fanstate_Avg", "Downwelling Fan State", COLOR["green"], axis="right", step=True),
                TraceSpec("sr30_swu_fanstate_Avg", "Upwelling Fan State", COLOR["olive"], axis="right", step=True),
                TraceSpec("sr30_swd_heatstate_Avg", "Downwelling Heater State", COLOR["purple"], axis="right", step=True),
                TraceSpec("sr30_swu_heatstate_Avg", "Upwelling Heater State", COLOR["brown"], axis="right", step=True),
            ),
        ),
        PanelSpec(
            "longwave_support",
            "IR20 Support",
            "Fan Tach [Hz]",
            "Body Temperature [C]",
            (
                TraceSpec("ir20_lwd_fan_Avg", "Downwelling IR20 Fan", COLOR["teal"]),
                TraceSpec("ir20_lwu_fan_Avg", "Upwelling IR20 Fan", COLOR["light_blue"]),
                TraceSpec("ir20_lwd_DegC_Avg", "Downwelling IR20 Body Temp", COLOR["purple"], axis="right"),
                TraceSpec("ir20_lwu_DegC_Avg", "Upwelling IR20 Body Temp", COLOR["brown"], axis="right"),
            ),
        ),
        PanelSpec(
            "sensor_variability",
            "Sensor Variability",
            "Standard Deviation",
            "SR50 QC",
            (
                TraceSpec("kt15_tem_Std", "KT15 Surface Std", COLOR["magenta"]),
                TraceSpec("spn1_tot_Std", "SPN1 Total Std", COLOR["brown"]),
                TraceSpec("sr50_dist_Std", "SR50 Distance Std", COLOR["teal"]),
                TraceSpec("sr50_qc_Avg", "SR50 Quality", COLOR["olive"], axis="right", step=True),
            ),
        ),
    ),
    "asfs-fast-sonic": (
        PanelSpec(
            "met",
            "Met",
            "Metek U / V Wind [m/s]",
            "Metek W Wind [m/s]",
            (
                TraceSpec("metek_x_out", "Metek U Wind", COLOR["teal"], valid_min=-100.0, valid_max=100.0),
                TraceSpec("metek_y_out", "Metek V Wind", COLOR["light_blue"], valid_min=-100.0, valid_max=100.0),
                TraceSpec("metek_z_out", "Metek W Wind", COLOR["purple"], axis="right", valid_min=-30.0, valid_max=30.0),
            ),
        ),
        PanelSpec(
            "tilt_temperature",
            "Tilt / Temperature",
            "Tilt [deg]",
            "Sonic Temperature [C]",
            (
                TraceSpec("metek_InclX_out", "Tilt X", COLOR["brown"], valid_min=-10.0, valid_max=360.0),
                TraceSpec("metek_InclY_out", "Tilt Y", COLOR["olive"], valid_min=-10.0, valid_max=360.0),
                TraceSpec("metek_T_out", "Sonic Temperature", COLOR["magenta"], axis="right", valid_min=-50.0, valid_max=50.0),
            ),
        ),
        PanelSpec(
            "quality",
            "Quality",
            "Quality",
            "State",
            (
                TraceSpec("metek_quality_out", "Metek Quality", COLOR["red"], step=True),
                TraceSpec("metek_senspathstate_out", "Sensor Path State", COLOR["slate"], axis="right", step=True),
            ),
        ),
    ),
    "power": (
        PanelSpec(
            "renewables",
            "Renewables",
            "Solar Power [W]",
            "Solar Voltage [V]",
            (
                TraceSpec("SolarWatts_East", "Solar East Power", COLOR["brown"]),
                TraceSpec("SolarWatts_South", "Solar South Power", COLOR["purple"]),
                TraceSpec("SolarWatts_West", "Solar West Power", COLOR["magenta"]),
                TraceSpec("SolarVolts_East", "Solar East Voltage", COLOR["olive"], axis="right", valid_min=0.0, valid_max=200.0),
                TraceSpec("SolarVolts_South", "Solar South Voltage", COLOR["green"], axis="right", valid_min=0.0, valid_max=200.0),
                TraceSpec("SolarVolts_West", "Solar West Voltage", COLOR["blue"], axis="right", valid_min=0.0, valid_max=200.0),
            ),
        ),
        PanelSpec(
            "battery_charging",
            "Battery Charging",
            "Charging Current In [A]",
            "Charging Power In [W]",
            (
                TraceSpec("BatteryAmps", "Charging Current In", COLOR["teal"], valid_min=-250.0, valid_max=250.0, smooth_minutes=30.0),
                TraceSpec("BatteryWatts", "Charging Power In", COLOR["light_blue"], axis="right", valid_min=-10000.0, valid_max=10000.0, smooth_minutes=30.0),
            ),
        ),
        PanelSpec(
            "output_power",
            "Output Power",
            "Output Power [W]",
            "ASS 48 V DC Power [W]",
            (
                TraceSpec("ACOutputWatts", "AC Output Power", COLOR["red"], valid_min=0.0, valid_max=10000.0),
                TraceSpec("DCInverterWatts", "DC Inverter Power", COLOR["teal"], valid_min=0.0, valid_max=10000.0),
                TraceSpec("watts_on_48vdc_Avg", "ASS 48 V DC Power", COLOR["purple"], axis="right"),
            ),
        ),
        PanelSpec(
            "cumulative_power",
            "Cumulative Power & State of Charge",
            "SOC [%]",
            "Cumulative Energy [kWh]",
            (
                TraceSpec("BatterySOC", "State of Charge", COLOR["green"], valid_min=0.0, valid_max=100.0),
                TraceSpec("SolarYield_East", "East Solar Generated", COLOR["brown"], axis="right", break_on_day_change=True),
                TraceSpec("SolarYield_South", "South Solar Generated", COLOR["purple"], axis="right", break_on_day_change=True),
                TraceSpec("SolarYield_West", "West Solar Generated", COLOR["magenta"], axis="right", break_on_day_change=True),
                TraceSpec("CumulativePowerGeneratedTotal", "Total Generated", COLOR["green"], axis="right", break_on_day_change=True),
                TraceSpec("CumulativePowerUtilised", "Utilised", COLOR["teal"], axis="right", break_on_day_change=True),
            ),
        ),
        PanelSpec(
            "output_voltage",
            "Output Voltage",
            "AC Output Voltage [V]",
            "DC Inverter Voltage [V]",
            (
                TraceSpec("ACOutputVolts", "AC Output Voltage", COLOR["brown"], valid_min=180.0, valid_max=260.0),
                TraceSpec("DCInverterVolts", "DC Inverter Voltage", COLOR["magenta"], axis="right", valid_min=40.0, valid_max=70.0),
            ),
        ),
        PanelSpec(
            "thermal_state",
            "Thermal State",
            "Temperature [C]",
            "Temperature [C]",
            (
                TraceSpec("InternalTemperature", "Internal Temperature", COLOR["red"], valid_min=-40.0, valid_max=100.0),
                TraceSpec("HeatsinkTemperature", "Heatsink Temperature", COLOR["brown"], valid_min=-40.0, valid_max=120.0),
                TraceSpec("TempSensor1", "Temperature Sensor 1", COLOR["teal"], axis="right", valid_min=-40.0, valid_max=100.0),
                TraceSpec("TempSensor2", "Temperature Sensor 2", COLOR["light_blue"], axis="right", valid_min=-40.0, valid_max=100.0),
                TraceSpec("TempSensor3", "Temperature Sensor 3", COLOR["purple"], axis="right", valid_min=-40.0, valid_max=100.0),
                TraceSpec("TempSensor4", "Temperature Sensor 4", COLOR["olive"], axis="right", valid_min=-40.0, valid_max=100.0),
            ),
        ),
    ),
    "ops-monitor": (
        PanelSpec(
            "source_disk_use",
            "Host Disk Use",
            "Used [%]",
            None,
            (
                TraceSpec("host_celine_source_used_pct", "CL61 Root", COLOR["teal"]),
                TraceSpec("host_celine_data_used_pct", "CL61 Data", COLOR["blue"]),
                TraceSpec("host_ass_data_used_pct", "ASS Data", COLOR["slate"]),
                TraceSpec("host_ass_root_used_pct", "ASS Root", COLOR["purple"]),
                TraceSpec("host_aps_data_used_pct", "APS Data", COLOR["brown"]),
                TraceSpec("host_aps_root_used_pct", "APS Root", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "aurora_storage",
            "Aurora / Archive Storage",
            "Used [%]",
            None,
            (
                TraceSpec("aurora_project_used_pct", "Aurora Raw (/project)", COLOR["teal"]),
                TraceSpec("aurora_data_used_pct", "AURORA Cloud Products (/data/aurora)", COLOR["purple"]),
                TraceSpec("aurora_root_used_pct", "AURORA Cloud Root (/)", COLOR["olive"]),
                TraceSpec("gws_storage_used_pct", "JASMIN GWS", COLOR["green"]),
            ),
        ),
        PanelSpec(
            "aps_battery_voltage",
            "APS Electrical / Thermal",
            "Voltage [V] / SOC [%]",
            "Temperature [C]",
            (
                TraceSpec("aps_battery_voltage_v", "DC Inverter Voltage", COLOR["brown"]),
                TraceSpec("aps_battery_soc_pct", "State of Charge", COLOR["green"]),
                TraceSpec("aps_internal_temp_c", "Internal Temperature", COLOR["red"], axis="right"),
            ),
        ),
        PanelSpec(
            "local_coverage",
            "Local Mirror Coverage",
            "Coverage [%]",
            None,
            (
                TraceSpec("cl61_local_coverage_pct", "CL61", COLOR["teal"]),
                TraceSpec("radar_local_coverage_pct", "Radar", COLOR["blue"]),
                TraceSpec("hatpro_local_coverage_pct", "HATPRO", COLOR["black"]),
                TraceSpec("vaisalamet_local_coverage_pct", "Meteorology", COLOR["green"]),
                TraceSpec("asfs_logger_local_coverage_pct", "Radiation", COLOR["purple"]),
                TraceSpec("asfs_fast_sonic_local_coverage_pct", "ASFS Fast Sonic", COLOR["magenta"]),
                TraceSpec("power_local_coverage_pct", "Aurora Power Supply", COLOR["brown"]),
                TraceSpec("wxcam_local_coverage_pct", "WXcam", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "local_lag",
            "Local Mirror Lag",
            "Lag [min]",
            None,
            (
                TraceSpec("cl61_local_lag_min", "CL61", COLOR["teal"]),
                TraceSpec("radar_local_lag_min", "Radar", COLOR["blue"]),
                TraceSpec("hatpro_local_lag_min", "HATPRO", COLOR["black"]),
                TraceSpec("vaisalamet_local_lag_min", "Meteorology", COLOR["green"]),
                TraceSpec("asfs_logger_local_lag_min", "Radiation", COLOR["purple"]),
                TraceSpec("asfs_fast_sonic_local_lag_min", "ASFS Fast Sonic", COLOR["magenta"]),
                TraceSpec("power_local_lag_min", "Aurora Power Supply", COLOR["brown"]),
                TraceSpec("wxcam_local_lag_min", "WXcam", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "gws_coverage",
            "JASMIN Mirror Coverage",
            "Coverage [%]",
            None,
            (
                TraceSpec("cl61_gws_coverage_pct", "CL61", COLOR["teal"]),
                TraceSpec("radar_gws_coverage_pct", "Radar", COLOR["blue"]),
                TraceSpec("hatpro_gws_coverage_pct", "HATPRO", COLOR["black"]),
                TraceSpec("vaisalamet_gws_coverage_pct", "Meteorology", COLOR["green"]),
                TraceSpec("asfs_logger_gws_coverage_pct", "Radiation", COLOR["purple"]),
                TraceSpec("asfs_fast_sonic_gws_coverage_pct", "ASFS Fast Sonic", COLOR["magenta"]),
                TraceSpec("power_gws_coverage_pct", "Aurora Power Supply", COLOR["brown"]),
                TraceSpec("wxcam_gws_coverage_pct", "WXcam", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "prune_gates",
            "Prune / Product Gates",
            "Stream Count",
            "State",
            (
                TraceSpec("streams_product_gate_ok_count", "Product Gates OK", COLOR["teal"]),
                TraceSpec("streams_prune_ready_count", "Prune Ready", COLOR["blue"]),
                TraceSpec("streams_local_issue_count", "Local Mirror Issues", COLOR["brown"]),
                TraceSpec("streams_gws_issue_count", "JASMIN Mirror Issues", COLOR["red"]),
                TraceSpec("gws_available_state", "JASMIN Available", COLOR["green"], axis="right", step=True),
            ),
        ),
        PanelSpec(
            "service_health",
            "Service Health",
            "Failure Count",
            "State",
            (
                TraceSpec("source_host_probe_fail_count", "Source Host Probe Failures", COLOR["red"]),
                TraceSpec("failed_source_sync_unit_count", "Source Sync Failures", COLOR["brown"]),
                TraceSpec("failed_processing_unit_count", "Processing Failures", COLOR["purple"]),
                TraceSpec("failed_transfer_unit_count", "Transfer Failures", COLOR["magenta"]),
                TraceSpec("mirror_verify_timer_active_state", "Mirror Verify Timer", COLOR["teal"], axis="right", step=True),
            ),
        ),
    ),
    # Archived Operations housekeeping should focus on exceptions and drift,
    # not on every raw metric collected in the live snapshot stream.
    "ops-monitor-hk": (
        PanelSpec(
            "storage_use",
            "Storage Use",
            "Used [%]",
            None,
            (
                TraceSpec("host_celine_source_used_pct", "CL61 Root Disk", COLOR["teal"]),
                TraceSpec("host_celine_data_used_pct", "CL61 Data Disk", COLOR["blue"]),
                TraceSpec("host_ass_data_used_pct", "ASS Data Disk", COLOR["slate"]),
                TraceSpec("host_ass_root_used_pct", "ASS Root Disk", COLOR["purple"]),
                TraceSpec("host_aps_data_used_pct", "APS Data Disk", COLOR["brown"]),
                TraceSpec("host_aps_root_used_pct", "APS Root Disk", COLOR["olive"]),
                TraceSpec("aurora_project_used_pct", "Aurora Raw (/project)", COLOR["olive"]),
                TraceSpec("aurora_data_used_pct", "AURORA Cloud Products (/data/aurora)", COLOR["magenta"]),
                TraceSpec("gws_storage_used_pct", "JASMIN GWS", COLOR["green"]),
            ),
        ),
        PanelSpec(
            "mirror_issue_counts",
            "Mirror Issue Counts",
            "Local Issues [count]",
            "GWS Issues [count]",
            (
                TraceSpec("cl61_local_issue_count", "CL61 Local", COLOR["teal"], step=True, skip_if_all_zero=True),
                TraceSpec("radar_local_issue_count", "Radar Local", COLOR["blue"], step=True, skip_if_all_zero=True),
                TraceSpec("hatpro_local_issue_count", "HATPRO Local", COLOR["black"], step=True, skip_if_all_zero=True),
                TraceSpec("vaisalamet_local_issue_count", "Meteorology Local", COLOR["green"], step=True, skip_if_all_zero=True),
                TraceSpec("asfs_logger_local_issue_count", "Radiation Local", COLOR["purple"], step=True, skip_if_all_zero=True),
                TraceSpec("asfs_fast_sonic_local_issue_count", "ASFS Fast Sonic Local", COLOR["magenta"], step=True, skip_if_all_zero=True),
                TraceSpec("power_local_issue_count", "APS Local", COLOR["brown"], step=True, skip_if_all_zero=True),
                TraceSpec("cl61_gws_issue_count", "CL61 GWS", COLOR["teal"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("radar_gws_issue_count", "Radar GWS", COLOR["blue"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("hatpro_gws_issue_count", "HATPRO GWS", COLOR["black"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("vaisalamet_gws_issue_count", "Meteorology GWS", COLOR["green"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_logger_gws_issue_count", "Radiation GWS", COLOR["purple"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_fast_sonic_gws_issue_count", "ASFS Fast Sonic GWS", COLOR["magenta"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("power_gws_issue_count", "APS GWS", COLOR["brown"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "mirror_lag",
            "Mirror Lag Outliers",
            "Local Lag [min]",
            "GWS Lag [min]",
            (
                TraceSpec("cl61_local_lag_min", "CL61 Local", COLOR["teal"], valid_min=2.0),
                TraceSpec("radar_local_lag_min", "Radar Local", COLOR["blue"], valid_min=2.0),
                TraceSpec("hatpro_local_lag_min", "HATPRO Local", COLOR["black"], valid_min=2.0),
                TraceSpec("vaisalamet_local_lag_min", "Meteorology Local", COLOR["green"], valid_min=2.0),
                TraceSpec("asfs_logger_local_lag_min", "Radiation Local", COLOR["purple"], valid_min=2.0),
                TraceSpec("asfs_fast_sonic_local_lag_min", "ASFS Fast Sonic Local", COLOR["magenta"], valid_min=2.0),
                TraceSpec("power_local_lag_min", "APS Local", COLOR["brown"], valid_min=2.0),
                TraceSpec("wxcam_local_lag_min", "WXcam Local", COLOR["olive"], valid_min=2.0),
                TraceSpec("cl61_gws_lag_min", "CL61 GWS", COLOR["teal"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("radar_gws_lag_min", "Radar GWS", COLOR["blue"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("hatpro_gws_lag_min", "HATPRO GWS", COLOR["black"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("vaisalamet_gws_lag_min", "Meteorology GWS", COLOR["green"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("asfs_logger_gws_lag_min", "Radiation GWS", COLOR["purple"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("asfs_fast_sonic_gws_lag_min", "ASFS Fast Sonic GWS", COLOR["magenta"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("power_gws_lag_min", "APS GWS", COLOR["brown"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("wxcam_gws_lag_min", "WXcam GWS", COLOR["olive"], axis="right", dash="dot", valid_min=10.0),
            ),
        ),
        PanelSpec(
            "wxcam_backfill",
            "WXcam Backfill",
            "Coverage [%]",
            "Issue Count",
            (
                TraceSpec("wxcam_local_coverage_pct", "WXcam Local Coverage", COLOR["olive"]),
                TraceSpec("wxcam_gws_coverage_pct", "WXcam GWS Coverage", COLOR["green"]),
                TraceSpec("wxcam_local_issue_count", "WXcam Local Issues", COLOR["brown"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("wxcam_gws_issue_count", "WXcam GWS Issues", COLOR["red"], axis="right", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "gate_blocks",
            "Blocked Streams",
            "Blocked Streams [count]",
            None,
            (
                TraceSpec("streams_product_gate_block_count", "Product Gate Blocked", COLOR["teal"], step=True, skip_if_all_zero=True),
                TraceSpec("streams_prune_block_count", "Prune Blocked", COLOR["blue"], step=True, skip_if_all_zero=True),
                TraceSpec("streams_local_issue_count", "Streams with Local Issues", COLOR["brown"], step=True, skip_if_all_zero=True),
                TraceSpec("streams_gws_issue_count", "Streams with GWS Issues", COLOR["red"], step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "source_freshness",
            "Source Freshness",
            "Stale Streams [count]",
            "Recent State",
            (
                TraceSpec("streams_source_stale_count", "Stale Streams", COLOR["red"], step=True, skip_if_all_zero=True),
                TraceSpec("cl61_source_recent_state", "CL61 Recent", COLOR["teal"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("radar_source_recent_state", "Radar Recent", COLOR["blue"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("hatpro_source_recent_state", "HATPRO Recent", COLOR["black"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("vaisalamet_source_recent_state", "Meteorology Recent", COLOR["green"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_logger_source_recent_state", "Radiation Recent", COLOR["purple"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_fast_sonic_source_recent_state", "ASFS Fast Sonic Recent", COLOR["magenta"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("power_source_recent_state", "APS Recent", COLOR["brown"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("wxcam_source_recent_state", "WXcam Recent", COLOR["olive"], axis="right", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "service_problems",
            "Service / Transfer Problems",
            "Failure Count",
            "Problem State",
            (
                TraceSpec("source_host_probe_fail_count", "Source Host Probe Failures", COLOR["red"], step=True, skip_if_all_zero=True),
                TraceSpec("failed_source_sync_unit_count", "Source Sync Failures", COLOR["brown"], step=True, skip_if_all_zero=True),
                TraceSpec("failed_processing_unit_count", "Processing Failures", COLOR["purple"], step=True, skip_if_all_zero=True),
                TraceSpec("failed_transfer_unit_count", "Transfer Failures", COLOR["magenta"], step=True, skip_if_all_zero=True),
                TraceSpec("gws_unavailable_state", "GWS Unavailable", COLOR["green"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("mirror_verify_problem_state", "Mirror Verify Problem", COLOR["teal"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("ops_monitor_append_problem_state", "Ops Append Problem", COLOR["blue"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("ops_monitor_alerts_problem_state", "Ops Alert Problem", COLOR["black"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("ops_monitor_quicklooks_problem_state", "Ops Quicklook Problem", COLOR["slate"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("dashboard_perf_log_stale_state", "Dashboard Perf Log Stale", COLOR["olive"], axis="right", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "dashboard_perf_log",
            "Dashboard Performance Log",
            "Age [min]",
            None,
            (
                TraceSpec("dashboard_perf_log_age_min", "Perf Log Age", COLOR["olive"], valid_min=30.0),
            ),
        ),
    ),
}

CURATED_HOUSEKEEPING_LAYOUTS = {
    "asfs-logger": "asfs-logger-hk",
}


def is_summary_instrument(instrument: str) -> bool:
    return instrument in SUMMARY_INSTRUMENTS


def display_name(instrument: str) -> str:
    return DISPLAY_NAMES.get(instrument, instrument)


def housekeeping_label(instrument: str) -> str | None:
    return HOUSEKEEPING_LABELS.get(instrument)


def summary_source_instruments(instrument: str) -> tuple[str, ...]:
    return SUMMARY_SOURCE_INSTRUMENTS.get(instrument, (instrument,))


def default_interactive_label(instrument: str) -> str:
    return OVERVIEW_LABEL


def default_calendar_label(instrument: str) -> str:
    """Return the default dated quicklook group label.

    The function name is kept for compatibility with older app code that called
    the dated PNG browser a calendar; the visible UI is now Science Quicklooks.
    """
    return OVERVIEW_LABEL


def widget_group_options(instrument: str) -> OrderedDict[str, dict[str, object]]:
    return OrderedDict(
        [
            (
                OVERVIEW_LABEL,
                {
                    "label": OVERVIEW_LABEL,
                    "clim": (0.0, 1.0),
                    "log": False,
                    "colorscale": "Viridis",
                },
            )
        ]
    )


def quicklook_prefix(instrument: str) -> str:
    return QUICKLOOK_PREFIX[instrument]


def summary_latest_png(quicklook_dir: Path, instrument: str) -> Path:
    return quicklook_dir / f"{quicklook_prefix(instrument)}__summary__latest.png"


def summary_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path:
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{quicklook_prefix(instrument)}__summary__{stamp}.png"


def housekeeping_latest_png(quicklook_dir: Path, instrument: str) -> Path | None:
    label = housekeeping_label(instrument)
    if label is None:
        return None
    key = label.lower()
    return quicklook_dir / f"{quicklook_prefix(instrument)}__{key}__latest.png"


def housekeeping_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path | None:
    label = housekeeping_label(instrument)
    if label is None:
        return None
    key = label.lower()
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{quicklook_prefix(instrument)}__{key}__{stamp}.png"


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


def clear_generated_quicklooks(quicklook_dir: Path, instrument: str) -> None:
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
        token = day_png.stem.rsplit("__", 1)[-1]
        legacy_day = legacy_daily_png(quicklook_dir, instrument, token)
        if legacy_day:
            shutil.copyfile(day_png, legacy_day)
    if latest_png is not None:
        legacy_latest = legacy_latest_png(quicklook_dir, instrument)
        if legacy_latest:
            shutil.copyfile(latest_png, legacy_latest)


def calendar_date_tokens(quicklook_dir: Path, instrument: str) -> list[str]:
    """List daily Science Quicklook tokens.

    The public UI no longer uses the word calendar, but these helper names are
    retained because existing quicklook and performance code still imports them.
    """
    prefix = quicklook_prefix(instrument)
    tokens: list[str] = []
    for png in sorted(quicklook_dir.glob(f"{prefix}__summary__*.png")):
        suffix = png.stem.split("__")[-1]
        if suffix == "latest":
            continue
        tokens.append(suffix)
    return tokens


def calendar_product_paths(quicklook_dir: Path, instrument: str, token: str) -> list[tuple[str, Path]]:
    """Return science and housekeeping PNGs associated with a dated token."""
    paths: list[tuple[str, Path]] = []
    if token == "latest":
        summary = summary_latest_png(quicklook_dir, instrument)
        if summary.exists():
            paths.append((display_name(instrument), summary))
        hk = housekeeping_latest_png(quicklook_dir, instrument)
        if hk and hk.exists():
            paths.append((housekeeping_label(instrument) or "Housekeeping", hk))
        return paths

    summary = summary_daily_png(quicklook_dir, instrument, token)
    if summary.exists():
        paths.append((display_name(instrument), summary))
    hk = housekeeping_daily_png(quicklook_dir, instrument, token)
    if hk and hk.exists():
        paths.append((housekeeping_label(instrument) or "Housekeeping", hk))
    return paths


def human_label(name: str) -> str:
    if name in HUMAN_LABELS:
        return HUMAN_LABELS[name]
    tokens = [token for token in name.split("_") if token]
    if tokens and tokens[-1] in {"pct", "gb", "count", "min", "state", "bytes"}:
        tokens = tokens[:-1]
    token_map = {
        "ac": "AC",
        "aps": "APS",
        "asfs": "ASFS",
        "cl61": "CL61",
        "gws": "GWS",
        "hk": "HK",
        "radar": "Radar",
        "utc": "UTC",
        "wxcam": "WXcam",
    }
    parts = [token_map.get(token.lower(), token.replace("-", " ").title()) for token in tokens]
    return " ".join(parts) if parts else name.replace("_", " ")


def human_unit(name: str) -> str | None:
    if name in HUMAN_UNITS:
        return HUMAN_UNITS[name]
    lower = name.lower()
    if lower.endswith("_pct"):
        return "%"
    if lower.endswith("_gb"):
        return "GB"
    if lower.endswith("_min"):
        return "min"
    if lower.endswith("_count"):
        return "count"
    if lower.endswith("_state"):
        return "state"
    if lower.endswith("_bytes"):
        return "B"
    if "volt" in lower:
        return "V"
    if "watt" in lower:
        return "W"
    if "amp" in lower:
        return "A"
    if lower.endswith("hz") or "_hz" in lower:
        return "Hz"
    if "yield" in lower or lower.endswith("kwh"):
        return "kWh"
    if "temp" in lower or lower.endswith("_t") or lower.endswith("_td") or "_amb_" in lower or "_tem_" in lower:
        return "C"
    if lower.endswith("_rh") or "_rh_" in lower or lower == "batterystate":
        return "%"
    if lower.startswith("baro") or lower.endswith("_hpa") or lower == "h1_e":
        return "hPa"
    if "dist" in lower:
        return "m"
    if "incl" in lower:
        return "deg"
    if "msec" in lower:
        return "ms"
    if lower == "scantime" or lower == "time_discrepancy":
        return "s"
    if "co2" in lower and "str" not in lower:
        return "ppm"
    if "h2o" in lower:
        return "mmol mol^-1"
    if "co2_str" in lower:
        return "%"
    if any(token in lower for token in STATUS_TOKENS):
        return "state"
    return None


def human_axis_label(name: str) -> str:
    label = human_label(name)
    unit = human_unit(name)
    return f"{label} [{unit}]" if unit else label


def display_scale(name: str) -> float:
    return DISPLAY_SCALE.get(name, 1.0)


def summary_trace_vars(instrument: str) -> set[str]:
    return {trace.var for panel in SUMMARY_LAYOUTS.get(instrument, ()) for trace in panel.traces}


def combine_summary_datasets(instrument: str, *datasets: xr.Dataset | None) -> xr.Dataset:
    """Merge 1D time-series sources, using later sources to fill gaps.

    This is used for summary instruments whose displayed variables may come
    from more than one Zarr store. Existing values keep priority, while NaNs
    or missing times can be filled by an independent source such as ASFS fast
    sonic.
    """
    merged_inputs: list[xr.Dataset] = []
    for ds in datasets:
        if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
            continue
        keep_names = [name for name, da in ds.data_vars.items() if da.dims == ("time",)]
        if not keep_names:
            continue
        subset = ds[keep_names].sortby("time")
        merged_inputs.append(subset)
    if not merged_inputs:
        return xr.Dataset()
    merged = merged_inputs[0]
    for subset in merged_inputs[1:]:
        merged, aligned = xr.align(merged, subset, join="outer")
        assignments = {}
        for name, da in aligned.data_vars.items():
            if name in merged.data_vars:
                filled = merged[name].combine_first(da)
                filled.attrs = dict(merged[name].attrs)
                assignments[name] = filled
            else:
                assignments[name] = da
        if assignments:
            merged = merged.assign(**assignments)
    merged = merged.sortby("time")
    merged.attrs["summary_instrument"] = instrument
    return merged


def fast_sonic_metek_summary_dataset(ds: xr.Dataset, freq: str = "1min") -> xr.Dataset:
    """Resample high-rate ASFS fast-sonic Metek fields onto summary names."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return xr.Dataset()
    if any(name in ds.data_vars for name in FAST_SONIC_TO_LOGGER_AVG.values()):
        keep = [
            name
            for name in FAST_SONIC_TO_LOGGER_AVG.values()
            if name in ds.data_vars and ds[name].dims == ("time",)
        ]
        return ds[keep].sortby("time") if keep else xr.Dataset()
    keep = [name for name in FAST_SONIC_TO_LOGGER_AVG if name in ds and ds[name].dims == ("time",)]
    if not keep:
        return xr.Dataset()
    frame = pd.DataFrame(
        {FAST_SONIC_TO_LOGGER_AVG[name]: np.asarray(ds[name].values, dtype=np.float64) for name in keep},
        index=pd.DatetimeIndex(ds["time"].values),
    )
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame.resample(freq).mean().dropna(how="all")
    if frame.empty:
        return xr.Dataset()
    return xr.Dataset(
        {name: (("time",), frame[name].to_numpy(dtype=np.float32)) for name in frame.columns},
        coords={"time": frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={"source": "derived from ASFS fast-sonic high-rate Metek fields", "frequency": freq},
    )


def fast_gas_licor_summary_dataset(ds: xr.Dataset, freq: str = "1min") -> xr.Dataset:
    """Resample high-rate ASFS fast-gas LI-COR fields onto summary names."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return xr.Dataset()
    if any(name in ds.data_vars for name in FAST_GAS_TO_LOGGER_AVG.values()):
        keep = [
            name
            for name in FAST_GAS_TO_LOGGER_AVG.values()
            if name in ds.data_vars and ds[name].dims == ("time",)
        ]
        return ds[keep].sortby("time") if keep else xr.Dataset()
    keep = [name for name in FAST_GAS_TO_LOGGER_AVG if name in ds and ds[name].dims == ("time",)]
    if not keep:
        return xr.Dataset()
    frame = pd.DataFrame(
        {FAST_GAS_TO_LOGGER_AVG[name]: np.asarray(ds[name].values, dtype=np.float64) for name in keep},
        index=pd.DatetimeIndex(ds["time"].values),
    )
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame.resample(freq).mean().dropna(how="all")
    if frame.empty:
        return xr.Dataset()
    return xr.Dataset(
        {name: (("time",), frame[name].to_numpy(dtype=np.float32)) for name in frame.columns},
        coords={"time": frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={"source": "derived from ASFS fast-gas LI-COR fields", "frequency": freq},
    )


def augment_meteorology_from_fast_sonic(ds: xr.Dataset) -> xr.Dataset:
    """Fill Meteorology Metek summary fields from the high-rate sonic stream.

    The ASFS science/logger stream carries one-minute Metek averages alongside
    radiation data. When that slow table has a source gap, the independent
    fast-sonic files can still provide the same Metek components. This helper
    maps those raw fast-sonic variables onto the one-minute summary names and
    only fills places where the slow-table values are missing.
    """
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds
    assignments: dict[str, xr.DataArray] = {}
    for source_name, target_name in FAST_SONIC_TO_LOGGER_AVG.items():
        if source_name not in ds or ds[source_name].dims != ("time",):
            continue
        source = ds[source_name].copy(deep=False)
        source.attrs = dict(source.attrs)
        source.attrs["derived_from"] = source_name
        if target_name in ds and ds[target_name].dims == ("time",):
            filled = ds[target_name].combine_first(source)
            filled.attrs = dict(ds[target_name].attrs)
            filled.attrs["gap_fill_source"] = source_name
            assignments[target_name] = filled
        else:
            source.name = target_name
            assignments[target_name] = source
    return ds.assign(**assignments) if assignments else ds


def augment_asfs_from_fast_gas(ds: xr.Dataset) -> xr.Dataset:
    """Fill ASFS LI-COR summary fields from the independent fast-gas stream."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds
    assignments: dict[str, xr.DataArray] = {}
    for source_name, target_name in FAST_GAS_TO_LOGGER_AVG.items():
        if source_name not in ds or ds[source_name].dims != ("time",):
            continue
        source = ds[source_name].copy(deep=False)
        source.attrs = dict(source.attrs)
        source.attrs["derived_from"] = source_name
        if target_name in ds and ds[target_name].dims == ("time",):
            filled = ds[target_name].combine_first(source)
            filled.attrs = dict(ds[target_name].attrs)
            filled.attrs["gap_fill_source"] = source_name
            assignments[target_name] = filled
        else:
            source.name = target_name
            assignments[target_name] = source
    return ds.assign(**assignments) if assignments else ds


def _daily_cumulative_energy_kwh(times: pd.DatetimeIndex, power_w: np.ndarray) -> np.ndarray:
    """Integrate power to kWh, resetting the displayed total at each UTC day."""
    cumulative_kwh = np.zeros(len(times), dtype=np.float64)
    if len(times) <= 1:
        return cumulative_kwh

    day_starts = times.normalize()
    time_ns = times.asi8.astype(np.float64)
    for idx in range(1, len(times)):
        if day_starts[idx] != day_starts[idx - 1]:
            # Start each day visibly from zero rather than carrying yesterday's
            # utilised energy into the new UTC day.
            cumulative_kwh[idx] = 0.0
            continue
        dt_hours = max((time_ns[idx] - time_ns[idx - 1]) / 3.6e12, 0.0)
        incremental_kwh = 0.5 * (power_w[idx] + power_w[idx - 1]) * dt_hours / 1000.0
        cumulative_kwh[idx] = cumulative_kwh[idx - 1] + incremental_kwh
    return cumulative_kwh


def _daily_cumulative_counter_delta(times: pd.DatetimeIndex, counter_kwh: np.ndarray) -> np.ndarray:
    """Convert daily-ish energy counters into UTC-day cumulative increments.

    The APS solar-yield counters can reset tens of minutes after midnight. For
    display, accumulate only positive counter changes within each UTC day and
    ignore reset drops, so the plotted generation starts cleanly at midnight.
    """
    cumulative_kwh = np.full(len(times), np.nan, dtype=np.float64)
    if len(times) == 0:
        return cumulative_kwh

    day_starts = times.normalize()
    current_day = None
    running_total = 0.0
    last_value = np.nan
    for idx, (day_start, raw_value) in enumerate(zip(day_starts, counter_kwh)):
        if current_day is None or day_start != current_day:
            current_day = day_start
            running_total = 0.0
            last_value = np.nan
        if not np.isfinite(raw_value):
            continue
        if np.isfinite(last_value):
            delta = float(raw_value - last_value)
            if delta > 0.0:
                running_total += delta
        cumulative_kwh[idx] = running_total
        last_value = float(raw_value)
    return cumulative_kwh


def _display_energy_assignments(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Map compact Power display-energy variables onto the standard plot names."""
    assignments: dict[str, xr.DataArray] = {}
    for target_name, source_name in POWER_DISPLAY_ENERGY_MAP.items():
        if source_name not in ds:
            continue
        da = ds[source_name].copy(deep=False)
        da.attrs = dict(da.attrs)
        da.attrs["units"] = "kWh"
        assignments[target_name] = da
    return assignments


def build_power_display_energy_dataset(
    ds: xr.Dataset,
    freq: str = POWER_DISPLAY_ENERGY_FREQ,
) -> xr.Dataset:
    """Build a compact Power display product for cumulative energy traces.

    The raw APS Zarr stays authoritative. This derived product stores only the
    one-minute cumulative kWh traces needed by the dashboard so interactive
    plotting does not need to read many days of one-second samples to compute
    solar generation and utilised energy.
    """
    if "time" not in ds or ds.sizes.get("time", 0) == 0:
        return xr.Dataset()

    ds = ds.sortby("time")
    times = pd.DatetimeIndex(ds["time"].values)
    frame: dict[str, np.ndarray] = {}
    generated_arrays: list[np.ndarray] = []
    for field_name in ("SolarYield_East", "SolarYield_South", "SolarYield_West"):
        if field_name not in ds:
            continue
        generated = _daily_cumulative_counter_delta(times, np.asarray(ds[field_name].values, dtype=np.float64))
        frame[POWER_DISPLAY_ENERGY_MAP[field_name]] = generated
        generated_arrays.append(generated)

    if generated_arrays:
        valid_generated = np.zeros(len(times), dtype=bool)
        for values in generated_arrays:
            valid_generated |= np.isfinite(values)
        total_generated = np.full(len(times), np.nan, dtype=np.float64)
        if np.any(valid_generated):
            summed = np.zeros(int(np.count_nonzero(valid_generated)), dtype=np.float64)
            for values in generated_arrays:
                summed += np.nan_to_num(values[valid_generated], nan=0.0)
            total_generated[valid_generated] = summed
        frame[POWER_DISPLAY_ENERGY_MAP["CumulativePowerGeneratedTotal"]] = total_generated
    else:
        total_generated = np.full(len(times), np.nan, dtype=np.float64)

    if "ACOutputWatts" in ds or "DCInverterWatts" in ds:
        ac_power = np.asarray(
            ds["ACOutputWatts"].values if "ACOutputWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        dc_power = np.asarray(
            ds["DCInverterWatts"].values if "DCInverterWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        valid_power = np.isfinite(ac_power) | np.isfinite(dc_power)
        utilised = np.full(len(times), np.nan, dtype=np.float64)
        if np.any(valid_power):
            utilised_power_w = np.nan_to_num(ac_power[valid_power], nan=0.0) + np.nan_to_num(dc_power[valid_power], nan=0.0)
            utilised_power_w = np.clip(utilised_power_w, a_min=0.0, a_max=None)
            utilised[valid_power] = _daily_cumulative_energy_kwh(times[valid_power], utilised_power_w)
        frame[POWER_DISPLAY_ENERGY_MAP["CumulativePowerUtilised"]] = utilised
    else:
        utilised = np.full(len(times), np.nan, dtype=np.float64)

    if not frame:
        return xr.Dataset()

    display_frame = pd.DataFrame(frame, index=times).resample(freq).last().dropna(how="all")
    if display_frame.empty:
        return xr.Dataset()
    out = xr.Dataset(
        {name: (("time",), display_frame[name].to_numpy(dtype=np.float32)) for name in display_frame.columns},
        coords={"time": display_frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            POWER_DISPLAY_ENERGY_ATTR: "true",
            "source": "derived from power.zarr",
            "frequency": freq,
            "description": "Display-only one-minute cumulative APS energy traces for dashboard plotting.",
        },
    )
    for name in out.data_vars:
        out[name].attrs["units"] = "kWh"
    return out


def _time_frame_from_dataset(ds: xr.Dataset, fields: tuple[str, ...]) -> pd.DataFrame:
    """Load selected 1D time-series fields into a sorted pandas frame."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return pd.DataFrame()
    names = [name for name in fields if name in ds and ds[name].dims == ("time",)]
    if not names:
        return pd.DataFrame()
    times = pd.DatetimeIndex(ds["time"].values)
    frame = pd.DataFrame(
        {name: np.asarray(ds[name].values, dtype=np.float64) for name in names},
        index=times,
    )
    frame = frame[~frame.index.isna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _resample_display_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.resample(freq).mean().dropna(how="all")


def build_power_display_summary_dataset(
    power_ds: xr.Dataset,
    ass_power_ds: xr.Dataset | None = None,
    freq: str = POWER_DISPLAY_SUMMARY_FREQ,
) -> xr.Dataset:
    """Build one-minute APS traces for fast dashboard plotting.

    The raw APS and ASFS logger Zarrs remain authoritative. This derived store
    keeps only the fields used by the curated Power summary panels, resampled
    to the dashboard display cadence, plus the cumulative-energy variables
    already produced for the APS cumulative panel.
    """
    if "time" not in power_ds or power_ds.sizes.get("time", 0) == 0:
        return xr.Dataset()

    frames: list[pd.DataFrame] = []
    sorted_power = power_ds.sortby("time")
    power_times = pd.DatetimeIndex(sorted_power["time"].values)
    power_start = power_times.min()
    power_end = power_times.max()

    power_frame = _time_frame_from_dataset(sorted_power, POWER_DISPLAY_SUMMARY_FIELDS)
    power_frame = _resample_display_frame(power_frame, freq)
    if not power_frame.empty:
        frames.append(power_frame)

    energy = build_power_display_energy_dataset(power_ds, freq=freq)
    if energy.sizes.get("time", 0):
        frames.append(energy.to_dataframe())

    if ass_power_ds is not None:
        ass_frame = _time_frame_from_dataset(ass_power_ds.sortby("time"), POWER_DISPLAY_SUMMARY_CONTEXT_FIELDS)
        if not ass_frame.empty:
            ass_frame = ass_frame[(ass_frame.index >= power_start) & (ass_frame.index <= power_end)]
        ass_frame = _resample_display_frame(ass_frame, freq)
        if not ass_frame.empty:
            frames.append(ass_frame)

    if not frames:
        return xr.Dataset()

    display_frame = pd.concat(frames, axis=1).sort_index()
    display_frame = display_frame.loc[:, ~display_frame.columns.duplicated(keep="last")]
    display_frame = display_frame.dropna(how="all")
    if display_frame.empty:
        return xr.Dataset()

    start = pd.Timestamp(display_frame.index.min()).isoformat()
    end = pd.Timestamp(display_frame.index.max()).isoformat()
    out = xr.Dataset(
        {name: (("time",), display_frame[name].to_numpy(dtype=np.float32)) for name in display_frame.columns},
        coords={"time": display_frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            POWER_DISPLAY_SUMMARY_ATTR: "true",
            "source": "derived from power.zarr and optional asfs_logger.zarr ASS 48 V power",
            "frequency": freq,
            "time_coverage_start": start,
            "time_coverage_end": end,
            "description": "Display-only one-minute APS summary traces for fast dashboard plotting.",
        },
    )
    for name in out.data_vars:
        unit = human_unit(name)
        if unit:
            out[name].attrs["units"] = unit
    for name in POWER_DISPLAY_ENERGY_MAP.values():
        if name in out:
            out[name].attrs["units"] = "kWh"
    return out


def _summary_display_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _crop_to_summary_display_window(ds: xr.Dataset, times: pd.DatetimeIndex) -> xr.Dataset:
    start = _summary_display_timestamp(ds.attrs.get(SUMMARY_DISPLAY_START_ATTR))
    end = _summary_display_timestamp(ds.attrs.get(SUMMARY_DISPLAY_END_ATTR))
    if start is None and end is None:
        return ds
    mask = np.ones(len(times), dtype=bool)
    if start is not None:
        mask &= times >= start
    if end is not None:
        mask &= times <= end
    return ds.isel(time=mask)


def _metek_wind_assignments(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Derive horizontal wind speed and meteorological direction from Metek U/V."""
    if "metek_x_out_Avg" not in ds or "metek_y_out_Avg" not in ds or "time" not in ds:
        return {}
    u = np.asarray(ds["metek_x_out_Avg"].values, dtype=np.float64)
    v = np.asarray(ds["metek_y_out_Avg"].values, dtype=np.float64)
    valid = np.isfinite(u) & np.isfinite(v)
    speed = np.full(len(u), np.nan, dtype=np.float64)
    direction = np.full(len(u), np.nan, dtype=np.float64)
    speed[valid] = np.hypot(u[valid], v[valid])
    # Meteorological convention: direction wind is coming from, clockwise from north.
    direction[valid] = (270.0 - np.degrees(np.arctan2(v[valid], u[valid]))) % 360.0
    return {
        "MetekWindSpeed": xr.DataArray(
            speed,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "m s^-1", "description": "Horizontal wind speed derived from metek_x_out_Avg and metek_y_out_Avg."},
        ),
        "MetekWindDirection": xr.DataArray(
            direction,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={
                "units": "degree",
                "description": "Meteorological wind direction derived from metek_x_out_Avg and metek_y_out_Avg.",
            },
        ),
    }


def _prepare_summary_dataset(ds: xr.Dataset, instrument: str) -> xr.Dataset:
    if instrument not in {"power", "vaisalamet"} or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds

    times = pd.DatetimeIndex(ds["time"].values)
    if len(times) == 0:
        return ds

    assignments: dict[str, xr.DataArray] = {}
    if instrument == "vaisalamet":
        ds = augment_meteorology_from_fast_sonic(ds)
        assignments.update(_metek_wind_assignments(ds))
        prepared = ds.assign(**assignments) if assignments else ds
        prepared_times = pd.DatetimeIndex(prepared["time"].values)
        return _crop_to_summary_display_window(prepared, prepared_times)

    display_assignments = _display_energy_assignments(ds)
    if display_assignments:
        prepared = ds.assign(**display_assignments)
        prepared_times = pd.DatetimeIndex(prepared["time"].values)
        return _crop_to_summary_display_window(prepared, prepared_times)

    generated_fields = [name for name in ("SolarYield_East", "SolarYield_South", "SolarYield_West") if name in ds]
    for field_name in generated_fields:
        generated = _daily_cumulative_counter_delta(times, np.asarray(ds[field_name].values, dtype=np.float64))
        assignments[field_name] = xr.DataArray(
            generated,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    if "ACOutputWatts" in ds or "DCInverterWatts" in ds:
        ac_power = np.asarray(
            ds["ACOutputWatts"].values if "ACOutputWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        dc_power = np.asarray(
            ds["DCInverterWatts"].values if "DCInverterWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        # The power summary can include ASFS logger overlay traces on a merged
        # time grid. Only integrate rows where APS output power was actually
        # sampled, otherwise ASFS-only timestamps would look like zero APS load.
        valid_power = np.isfinite(ac_power) | np.isfinite(dc_power)
        utilised_full = np.full(len(times), np.nan, dtype=np.float64)
        if np.any(valid_power):
            utilised_power_w = np.nan_to_num(ac_power[valid_power], nan=0.0) + np.nan_to_num(dc_power[valid_power], nan=0.0)
            utilised_power_w = np.clip(utilised_power_w, a_min=0.0, a_max=None)
            utilised_full[valid_power] = _daily_cumulative_energy_kwh(times[valid_power], utilised_power_w)

        assignments["CumulativePowerUtilised"] = xr.DataArray(
            utilised_full,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    if generated_fields:
        total_generated = np.full(len(times), np.nan, dtype=np.float64)
        valid_generated = np.zeros(len(times), dtype=bool)
        field_values: list[np.ndarray] = []
        for field_name in generated_fields:
            values = np.asarray(assignments[field_name].values, dtype=np.float64)
            field_values.append(values)
            valid_generated |= np.isfinite(values)
        if np.any(valid_generated):
            summed = np.zeros(int(np.count_nonzero(valid_generated)), dtype=np.float64)
            for values in field_values:
                summed += np.nan_to_num(values[valid_generated], nan=0.0)
            total_generated[valid_generated] = summed
        assignments["CumulativePowerGeneratedTotal"] = xr.DataArray(
            total_generated,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    prepared = ds.assign(**assignments) if assignments else ds
    prepared_times = pd.DatetimeIndex(prepared["time"].values)
    return _crop_to_summary_display_window(prepared, prepared_times)


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


def _clean_values(values: np.ndarray, trace: TraceSpec) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if trace.valid_min is not None:
        arr[arr < trace.valid_min] = np.nan
    if trace.valid_max is not None:
        arr[arr > trace.valid_max] = np.nan
    if trace.scale != 1.0:
        arr *= trace.scale
    return arr


def _has_signal(values: np.ndarray, trace: TraceSpec) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    if trace.skip_if_all_zero and np.allclose(finite, 0.0):
        return False
    return True


def _panel_series(ds: xr.Dataset, panel: PanelSpec) -> list[tuple[TraceSpec, np.ndarray]]:
    rows: list[tuple[TraceSpec, np.ndarray]] = []
    for trace in panel.traces:
        if trace.var not in ds:
            continue
        values = _clean_values(ds[trace.var].values, trace)
        if not _has_signal(values, trace):
            continue
        rows.append((trace, values))
    return rows


def _active_panels(ds: xr.Dataset, instrument: str) -> list[tuple[PanelSpec, list[tuple[TraceSpec, np.ndarray]]]]:
    panels: list[tuple[PanelSpec, list[tuple[TraceSpec, np.ndarray]]]] = []
    for panel in SUMMARY_LAYOUTS[instrument]:
        rows = _panel_series(ds, panel)
        if rows:
            panels.append((panel, rows))
    return panels


def _time_index(ds: xr.Dataset) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ds["time"].values) if "time" in ds else pd.DatetimeIndex([])


def _trace_time_values(times: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Drop merged-grid NaNs so each trace renders on its own real sampling cadence."""
    if len(times) == 0:
        return times, values
    finite = np.isfinite(values)
    if not np.any(finite):
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    return times[finite], values[finite]


def _downsample_trace(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    max_time_samples: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Downsample one trace after dropping merged-grid NaNs.

    Summary datasets can merge one-second APS data with one-minute ASFS context.
    Downsampling each rendered trace separately preserves the lower-cadence
    context lines instead of skipping them on the dense merged time grid.
    """
    count = min(len(times), values.size)
    if count == 0:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    times = times[:count]
    values = np.asarray(values[:count], dtype=np.float64)
    if count <= max_time_samples:
        return times, values
    keep_count = max(2, int(max_time_samples))
    keep = np.unique(np.linspace(0, count - 1, keep_count, dtype=int))
    return times[keep], values[keep]


def _smooth_trace_values(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    trace: TraceSpec,
) -> np.ndarray:
    """Apply optional display-only smoothing before browser/PNG downsampling."""
    if trace.smooth_minutes is None or trace.smooth_minutes <= 0 or len(times) < 3:
        return values
    series = pd.Series(np.asarray(values, dtype=np.float64), index=times)
    window = f"{float(trace.smooth_minutes):g}min"
    return series.rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)


def _insert_day_breaks(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    trace: TraceSpec,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Break display lines where UTC-day cumulative counters reset."""
    if not trace.break_on_day_change or len(times) < 2:
        return times, values
    day_starts = times.normalize()
    if not np.any(day_starts[1:] != day_starts[:-1]):
        return times, values

    out_times = []
    out_values: list[float] = []
    for idx, (timestamp, value) in enumerate(zip(times, values)):
        if idx > 0 and day_starts[idx] != day_starts[idx - 1]:
            out_times.append(timestamp)
            out_values.append(np.nan)
        out_times.append(timestamp)
        out_values.append(float(value))
    return pd.DatetimeIndex(out_times), np.asarray(out_values, dtype=np.float64)


def _insert_line_gap_breaks(times: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Insert NaNs into line traces so outages render as visible white breaks."""
    if len(times) < 2 or values.size < 2:
        return times, values
    expanded_times, expanded_values = insert_time_gap_breaks(times, np.asarray(values, dtype=np.float64)[None, :], time_axis=1)
    return pd.DatetimeIndex(expanded_times), expanded_values[0]


def _trace_plot_values(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    max_time_samples: int,
    trace: TraceSpec,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    trace_times, trace_values = _trace_time_values(times, values)
    trace_values = _smooth_trace_values(trace_times, trace_values, trace)
    trace_times, trace_values = _downsample_trace(trace_times, trace_values, max_time_samples)
    trace_times, trace_values = _insert_line_gap_breaks(trace_times, trace_values)
    return _insert_day_breaks(trace_times, trace_values, trace)


def _include_zero_in_limits(limits: tuple[float, float] | None) -> tuple[float, float] | None:
    """Expand an axis range just enough to keep the zero reference visible."""
    if limits is None:
        return None
    lower, upper = limits
    if lower <= 0.0 <= upper:
        return limits
    span = max(upper - lower, max(abs(lower), abs(upper), 1.0) * 0.1)
    if lower > 0.0:
        return -0.04 * span, upper
    return lower, 0.08 * span


def _axis_tick_values(
    limits: tuple[float, float] | None,
    step: float = 2.0,
) -> tuple[list[float], list[str]]:
    """Build fixed-step numeric tick labels for secondary energy axes."""
    if limits is None:
        return [], []
    lower, upper = limits
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return [], []
    if not np.isfinite(step) or step <= 0:
        return [], []
    start = np.floor(lower / step) * step
    stop = np.ceil(upper / step) * step
    values = np.arange(start, stop + step * 0.5, step, dtype=float)
    values = values[(values >= lower - step * 0.05) & (values <= upper + step * 0.05)]
    if values.size == 0:
        return [], []
    decimals = 0 if step >= 1 and np.isclose(step, round(step)) else int(max(0, np.ceil(-np.log10(step)))) + 1
    labels = []
    for value in values:
        if abs(value) < step * 1.0e-6:
            labels.append("0")
        elif decimals == 0:
            labels.append(f"{value:.0f}")
        else:
            labels.append(f"{value:.{decimals}f}".rstrip("0").rstrip("."))
    return values.tolist(), labels


def _axis_tick_step(limits: tuple[float, float] | None, target_ticks: int = 6) -> float:
    """Choose a quiet 1/2/5-style tick step for a numeric axis range."""
    if limits is None:
        return 1.0
    lower, upper = limits
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return 1.0
    raw_step = (upper - lower) / max(target_ticks - 1, 1)
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        step = multiplier * magnitude
        if step >= raw_step:
            return float(step)
    return float(10.0 * magnitude)


def _padded_axis_limits(
    series: list[np.ndarray],
    headroom: float = MATPLOTLIB_Y_HEADROOM_FRACTION,
    footroom: float = MATPLOTLIB_Y_FOOTROOM_FRACTION,
) -> tuple[float, float] | None:
    finite_parts = [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in series]
    finite_parts = [values for values in finite_parts if values.size]
    if not finite_parts:
        return None
    values = np.concatenate(finite_parts)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    span = upper - lower
    if span <= 0:
        scale = max(abs(upper), 1.0)
        span = scale * 0.1
        lower -= span * 0.5
        upper += span * 0.5
    return (
        lower - span * footroom,
        upper + span * headroom,
    )


def _apply_matplotlib_axis_padding(ax, series: list[np.ndarray]) -> tuple[float, float] | None:
    """Add y-range headroom so boxed panel labels do not sit on top of traces."""
    limits = _padded_axis_limits(series)
    if limits is not None:
        ax.set_ylim(*limits)
    return limits


def _window_title(suffix: str, instrument: str) -> str:
    return f"{display_name(instrument)} - {suffix}"


def plot_housekeeping_timeseries(
    ds: xr.Dataset,
    instrument: str,
    title: str,
    output: Path,
    max_time_samples: int = MAX_TIME_SAMPLES,
    exclude_vars: set[str] | None = None,
) -> list[str]:
    curated_layout = CURATED_HOUSEKEEPING_LAYOUTS.get(instrument)
    if curated_layout:
        save_summary_png(ds, curated_layout, title, output, max_time_samples=max_time_samples)
        return sorted(summary_trace_vars(curated_layout))

    ds = downsample_time(ds, max_time_samples=max_time_samples)
    times = _time_index(ds)
    names = [name for name in numeric_time_vars(ds) if not exclude_vars or name not in exclude_vars]
    if len(times) == 0 or not names:
        raise ValueError(f"No numeric {instrument} time-series variables available")

    max_height = 42.0 if instrument == "power" else 34.0
    per_var = 1.0 if instrument == "power" else 1.1
    height = max(8.0, min(max_height, per_var * len(names)))
    fig, axes = plt.subplots(len(names), 1, figsize=(13, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = [COLOR["teal"], COLOR["red"], COLOR["green"], COLOR["purple"], COLOR["brown"], COLOR["magenta"], COLOR["olive"], COLOR["blue"]]
    for idx, (ax, name) in enumerate(zip(axes, names)):
        values = np.asarray(ds[name].values, dtype=np.float64) * display_scale(name)
        trace_times, trace_values = _trace_time_values(times, values)
        trace_times, trace_values = _insert_line_gap_breaks(trace_times, trace_values)
        drawstyle = "steps-post" if is_status_like_var(name) else "default"
        if len(trace_times):
            ax.plot(trace_times, trace_values, color=colors[idx % len(colors)], linewidth=0.8, drawstyle=drawstyle)
        ax.set_ylabel(human_axis_label(name), fontsize=7, rotation=0, ha="right", va="center")
        ax.grid(True, color=PLOT_GRID, linewidth=0.4)
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
    fig.subplots_adjust(left=0.27, bottom=0.08, top=0.96, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return names


def plot_housekeeping_last_24h(
    zarr_path: Path,
    output: Path,
    instrument: str,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> list[str]:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = _time_index(ds)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    title = _window_title(f"{housekeeping_label(instrument) or 'Housekeeping'} - Latest 24 hours", instrument)
    return plot_housekeeping_timeseries(window, instrument=instrument, title=title, output=output, max_time_samples=max_time_samples)


def _apply_time_axis_matplotlib(ax, times: pd.DatetimeIndex) -> None:
    apply_quicklook_time_axis(ax, times, label_rotation=0, label_size=9)


def save_summary_png(
    ds: xr.Dataset,
    instrument: str,
    title: str,
    output: Path,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> int:
    ds = _prepare_summary_dataset(ds, instrument)
    times = _time_index(ds)
    panels = _active_panels(ds, instrument)
    if len(times) == 0 or not panels:
        raise ValueError(f"No summary time-series panels available for {instrument}")

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, max(7.5, 2.6 * len(panels))), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (panel, rows) in zip(axes, panels):
        right_ax = ax.twinx() if panel.right_axis_label else None
        left_color = None
        right_color = None
        left_axis_values: list[np.ndarray] = []
        right_axis_values: list[np.ndarray] = []
        for trace, values in rows:
            target = right_ax if trace.axis == "right" and right_ax is not None else ax
            drawstyle = "steps-post" if trace.step else "default"
            trace_times, trace_values = _trace_plot_values(times, values, max_time_samples, trace)
            if len(trace_times) == 0:
                continue
            target.plot(
                trace_times,
                trace_values,
                color=trace.color,
                linewidth=1.25,
                drawstyle=drawstyle,
                label=trace.label,
            )
            if target is right_ax:
                right_axis_values.append(trace_values)
            else:
                left_axis_values.append(trace_values)
            if trace.axis == "right" and right_color is None:
                right_color = trace.color
            if trace.axis == "left" and left_color is None:
                left_color = trace.color

        left_limits = _apply_matplotlib_axis_padding(ax, left_axis_values)
        if right_ax is not None:
            right_limits = _apply_matplotlib_axis_padding(right_ax, right_axis_values)
            if panel.right_axis_label == panel.left_axis_label:
                common_limits = _padded_axis_limits(left_axis_values + right_axis_values)
                if common_limits is not None:
                    ax.set_ylim(*common_limits)
                    right_ax.set_ylim(*common_limits)
            elif left_limits is not None and right_limits is None:
                right_ax.set_ylim(*left_limits)
            if panel.key == "cumulative_power":
                right_limits = _include_zero_in_limits(right_limits)
                if right_limits is not None:
                    right_ax.set_ylim(*right_limits)
                    tick_values, tick_labels = _axis_tick_values(right_limits, step=_axis_tick_step(right_limits))
                    if tick_values:
                        right_ax.set_yticks(tick_values)
                        right_ax.set_yticklabels(tick_labels)

        ax.set_facecolor("white")
        ax.grid(True, color=PLOT_GRID, linewidth=0.5)
        ax.tick_params(axis="y", colors=left_color or COLOR["black"], labelsize=9)
        ax.set_ylabel(panel.left_axis_label, color=left_color or COLOR["black"], fontsize=11)
        if right_ax is not None:
            right_ax.tick_params(axis="y", colors=right_color or COLOR["black"], labelsize=9)
            right_ax.set_ylabel(panel.right_axis_label or "", color=right_color or COLOR["black"], fontsize=11)

        ax.text(
            0.01,
            0.94,
            panel.label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor=PLOT_TEXT, linewidth=1.0, boxstyle="square,pad=0.3"),
        )

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = right_ax.get_legend_handles_labels() if right_ax is not None else ([], [])
        handles = handles_left + handles_right
        labels = labels_left + labels_right
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(MATPLOTLIB_LEGEND_X, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                frameon=False,
                ncol=1,
            )

    _apply_time_axis_matplotlib(axes[-1], times)
    axes[-1].set_xlabel("Time (UTC)", fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=MATPLOTLIB_PANEL_RIGHT, bottom=0.08, top=0.95, hspace=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return len(panels)


def plot_summary_last_24h(
    zarr_path: Path,
    output: Path,
    instrument: str,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> int:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = _time_index(ds)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    if instrument == "power":
        context_start = (pd.Timestamp(start_time) - pd.Timedelta(days=max(0, POWER_CUMULATIVE_CONTEXT_DAYS))).normalize()
    else:
        context_start = start_time
    mask = (time_index >= context_start) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    if instrument == "power":
        window = window.copy(deep=False)
        window.attrs[SUMMARY_DISPLAY_START_ATTR] = pd.Timestamp(start_time).isoformat()
        window.attrs[SUMMARY_DISPLAY_END_ATTR] = pd.Timestamp(end_time).isoformat()
    title = _window_title("Latest 24 hours", instrument)
    return save_summary_png(window, instrument=instrument, title=title, output=output, max_time_samples=max_time_samples)


def build_summary_plotly(
    ds: xr.Dataset,
    instrument: str,
    title: str | None = None,
    max_time_samples: int = INTERACTIVE_MAX_TIME_SAMPLES,
) -> go.Figure:
    ds = _prepare_summary_dataset(ds, instrument)
    times = _time_index(ds)
    panels = _active_panels(ds, instrument)
    if len(times) == 0 or not panels:
        raise ValueError(f"No summary time-series panels available for {instrument}")

    vertical_spacing = 0.028 if len(panels) >= 6 else 0.04
    panel_domain_end = PLOTLY_SUMMARY_PANEL_DOMAIN_END
    legend_x = PLOTLY_SUMMARY_LEGEND_X
    right_margin = PLOTLY_SUMMARY_RIGHT_MARGIN
    if instrument == "power":
        per_panel_height = PLOTLY_SUMMARY_POWER_PANEL_HEIGHT
        max_height = PLOTLY_SUMMARY_POWER_MAX_HEIGHT
    else:
        per_panel_height = PLOTLY_SUMMARY_PANEL_HEIGHT
        max_height = PLOTLY_SUMMARY_MAX_HEIGHT

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        specs=[[{"secondary_y": panel.right_axis_label is not None}] for panel, _rows in panels],
        subplot_titles=[panel.label for panel, _rows in panels],
    )

    panel_height = (1.0 - vertical_spacing * (len(panels) - 1)) / len(panels)
    legend_layouts: dict[str, dict[str, object]] = {}
    for row_index, (panel, rows) in enumerate(panels, start=1):
        legend_name = "legend" if row_index == 1 else f"legend{row_index}"
        panel_top = 1.0 - (row_index - 1) * (panel_height + vertical_spacing)
        legend_layouts[legend_name] = dict(
            x=legend_x,
            xanchor="left",
            y=max(0.02, panel_top - 0.02),
            yanchor="top",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=PLOT_BORDER,
            borderwidth=1,
            font=dict(size=9, color=PLOT_TEXT),
            itemsizing="constant",
            tracegroupgap=2,
        )
        left_color = None
        right_color = None
        left_axis_values: list[np.ndarray] = []
        right_axis_values: list[np.ndarray] = []
        for trace, values in rows:
            secondary = trace.axis == "right" and panel.right_axis_label is not None
            if secondary and right_color is None:
                right_color = trace.color
            if not secondary and left_color is None:
                left_color = trace.color
            trace_times, trace_values = _trace_plot_values(times, values, max_time_samples, trace)
            if len(trace_times) == 0:
                continue
            if secondary:
                right_axis_values.append(trace_values)
            else:
                left_axis_values.append(trace_values)
            fig.add_trace(
                go.Scatter(
                    x=trace_times,
                    y=trace_values,
                    mode="lines",
                    name=trace.label,
                    legend=legend_name,
                    line=dict(color=trace.color, width=2.0, dash=trace.dash or "solid", shape="hv" if trace.step else "linear"),
                    hovertemplate=f"Time=%{{x}}<br>{trace.label}=%{{y:.6g}}<extra></extra>",
                    connectgaps=False,
                    showlegend=True,
                ),
                row=row_index,
                col=1,
                secondary_y=secondary,
            )
        left_range = _padded_axis_limits(left_axis_values, headroom=0.08, footroom=0.04)
        right_range = _padded_axis_limits(right_axis_values, headroom=0.08, footroom=0.04)
        if panel.key == "cumulative_power":
            right_range = _include_zero_in_limits(right_range)
        if panel.right_axis_label == panel.left_axis_label:
            common_range = _padded_axis_limits(left_axis_values + right_axis_values, headroom=0.08, footroom=0.04)
            if common_range is not None:
                left_range = common_range
                right_range = common_range
        fig.update_yaxes(
            title_text=panel.left_axis_label,
            automargin=True,
            showgrid=True,
            gridcolor=PLOT_GRID,
            linecolor=PLOT_LINE,
            tickfont=dict(color=left_color or COLOR["black"], size=10),
            title_font=dict(color=left_color or COLOR["black"], size=11),
            range=list(left_range) if left_range is not None else None,
            row=row_index,
            col=1,
            secondary_y=False,
        )
        if panel.right_axis_label is not None:
            right_tick_values: list[float] = []
            right_tick_labels: list[str] = []
            if panel.key == "cumulative_power":
                right_tick_values, right_tick_labels = _axis_tick_values(right_range, step=_axis_tick_step(right_range))
            fig.update_yaxes(
                title_text=panel.right_axis_label,
                automargin=True,
                showgrid=False,
                gridcolor=PLOT_GRID,
                zeroline=False,
                zerolinecolor=PLOT_GRID,
                zerolinewidth=1,
                linecolor=PLOT_LINE,
                tickfont=dict(color=right_color or COLOR["black"], size=10),
                title_font=dict(color=right_color or COLOR["black"], size=11),
                range=list(right_range) if right_range is not None else None,
                tickmode="array" if right_tick_values else "auto",
                tickvals=right_tick_values or None,
                ticktext=right_tick_labels or None,
                ticks="outside" if panel.key == "cumulative_power" else "",
                ticklen=5 if panel.key == "cumulative_power" else None,
                row=row_index,
                col=1,
                secondary_y=True,
            )

    tickvals = []
    ticktext = []
    if len(times):
        start = times.min()
        end = times.max()
        duration = end - start
        freq = "1h" if duration <= pd.Timedelta(hours=18) else "2h" if duration <= pd.Timedelta(hours=36) else "6h"
        for stamp in pd.date_range(start=start.floor("h"), end=end.ceil("h"), freq=freq):
            tickvals.append(stamp.to_pydatetime())
            ticktext.append(stamp.strftime("%H:%M"))
    fig.update_xaxes(
        domain=[0.0, panel_domain_end],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridcolor=PLOT_GRID,
        linecolor=PLOT_LINE,
        tickfont=dict(color=PLOT_TEXT, size=11),
    )
    fig.update_xaxes(title_text="Time (UTC)", row=len(panels), col=1)
    fig.update_layout(
        showlegend=True,
        height=max(520, min(max_height, per_panel_height * len(panels) + 90)),
        margin=dict(l=80, r=right_margin, t=60, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=PLOT_TEXT, size=12),
        title=dict(text=title or display_name(instrument), x=0.01, xanchor="left", font=dict(size=17, color=PLOT_TEXT)),
        **legend_layouts,
    )
    for ann in fig.layout.annotations:
        ann.update(
            x=0.01,
            xref="paper",
            xanchor="left",
            bgcolor="white",
            bordercolor=PLOT_TEXT,
            borderwidth=1,
            font=dict(size=12, color=PLOT_TEXT),
        )
    return fig
