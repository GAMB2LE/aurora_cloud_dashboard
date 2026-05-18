#!/usr/bin/env python3
"""Summary and housekeeping plotting helpers for 1D Aurora instruments."""

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

MAX_TIME_SAMPLES = int(os.environ.get("AURORA_QUICKLOOK_MAX_TIME_SAMPLES", "2200"))
INTERACTIVE_MAX_TIME_SAMPLES = int(os.environ.get("AURORA_INTERACTIVE_MAX_TIME_SAMPLES", "1600"))
OVERVIEW_LABEL = "Overview"
# Reserve a fixed right-side gutter for per-panel legends so they sit beyond the
# secondary-axis labels in both the interactive Plotly view and saved PNGs.
MATPLOTLIB_PANEL_RIGHT = 0.72
MATPLOTLIB_LEGEND_X = 1.12
PLOTLY_PANEL_DOMAIN_END = 0.73
PLOTLY_LEGEND_X = 0.84


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
    "licor_t_out_Avg": "LI-COR Temperature",
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
    "CumulativePowerGeneratedTotal": "Total Generated",
    "CumulativePowerBalance": "Surplus / Deficit",
    "CumulativePowerUtilised": "Power Utilised",
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
    "licor_co2_out_Avg": "ppm",
    "licor_h2o_out_Avg": "mmol mol^-1",
    "licor_t_out_Avg": "C",
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
    "CumulativePowerGeneratedTotal": "kWh",
    "CumulativePowerBalance": "kWh",
    "CumulativePowerUtilised": "kWh",
    "TempSensor1": "C",
    "TempSensor2": "C",
    "TempSensor3": "C",
    "TempSensor4": "C",
    "TotCapacity": "Ah",
    "time_discrepancy": "s",
    "scantime": "s",
}

DISPLAY_SCALE = {}

SUMMARY_SOURCE_INSTRUMENTS = {
    "vaisalamet": ("vaisalamet", "asfs-logger"),
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
                TraceSpec("kt15_amb_Avg", "KT15 Ambient Temperature", COLOR["olive"]),
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
                TraceSpec("SolarVolts_East", "Solar East Voltage", COLOR["olive"], axis="right"),
                TraceSpec("SolarVolts_South", "Solar South Voltage", COLOR["green"], axis="right"),
                TraceSpec("SolarVolts_West", "Solar West Voltage", COLOR["blue"], axis="right"),
            ),
        ),
        PanelSpec(
            "battery_charging",
            "Battery Charging",
            "Charging Current In [A]",
            "Charging Power In [W]",
            (
                TraceSpec("BatteryAmps", "Charging Current In", COLOR["teal"]),
                TraceSpec("BatteryWatts", "Charging Power In", COLOR["light_blue"], axis="right"),
            ),
        ),
        PanelSpec(
            "output_power",
            "Output Power",
            "Output Power [W]",
            "ASS 48 V DC Power [W]",
            (
                TraceSpec("ACOutputWatts", "AC Output Power", COLOR["red"]),
                TraceSpec("DCInverterWatts", "DC Inverter Power", COLOR["teal"]),
                TraceSpec("watts_on_48vdc_Avg", "ASS 48 V DC Power", COLOR["purple"], axis="right"),
            ),
        ),
        PanelSpec(
            "cumulative_power",
            "Cumulative Power",
            "Cumulative Energy [kWh]",
            "Surplus / Deficit [kWh]",
            (
                TraceSpec("SolarYield_East", "East Solar Generated", COLOR["brown"]),
                TraceSpec("SolarYield_South", "South Solar Generated", COLOR["purple"]),
                TraceSpec("SolarYield_West", "West Solar Generated", COLOR["magenta"]),
                TraceSpec("CumulativePowerGeneratedTotal", "Total Generated", COLOR["green"]),
                TraceSpec("CumulativePowerUtilised", "Utilised", COLOR["teal"]),
                TraceSpec("CumulativePowerBalance", "Surplus / Deficit", COLOR["red"], axis="right"),
            ),
        ),
        PanelSpec(
            "output_voltage",
            "Output Voltage",
            "AC Output Voltage [V]",
            "DC Inverter Voltage [V]",
            (
                TraceSpec("ACOutputVolts", "AC Output Voltage", COLOR["brown"]),
                TraceSpec("DCInverterVolts", "DC Inverter Voltage", COLOR["slate"], axis="right"),
            ),
        ),
        PanelSpec(
            "thermal_state",
            "Thermal State",
            "Temperature [C]",
            None,
            (
                TraceSpec("InternalTemperature", "Internal Temperature", COLOR["red"]),
                TraceSpec("HeatsinkTemperature", "Heatsink Temperature", COLOR["brown"]),
                TraceSpec("TempSensor1", "Temperature Sensor 1", COLOR["teal"]),
                TraceSpec("TempSensor2", "Temperature Sensor 2", COLOR["light_blue"]),
                TraceSpec("TempSensor3", "Temperature Sensor 3", COLOR["purple"]),
                TraceSpec("TempSensor4", "Temperature Sensor 4", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "state_of_charge",
            "State of Charge",
            "SOC [%]",
            None,
            (
                TraceSpec("BatterySOC", "State of Charge", COLOR["green"]),
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
    prefix = quicklook_prefix(instrument)
    tokens: list[str] = []
    for png in sorted(quicklook_dir.glob(f"{prefix}__summary__*.png")):
        suffix = png.stem.split("__")[-1]
        if suffix == "latest":
            continue
        tokens.append(suffix)
    return tokens


def calendar_product_paths(quicklook_dir: Path, instrument: str, token: str) -> list[tuple[str, Path]]:
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
    merged = xr.merge(merged_inputs, join="outer", compat="override", combine_attrs="drop_conflicts")
    merged = merged.sortby("time")
    merged.attrs["summary_instrument"] = instrument
    return merged


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


def _prepare_summary_dataset(ds: xr.Dataset, instrument: str) -> xr.Dataset:
    if instrument != "power" or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds

    times = pd.DatetimeIndex(ds["time"].values)
    if len(times) == 0:
        return ds

    assignments: dict[str, xr.DataArray] = {}

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

    if "CumulativePowerGeneratedTotal" not in ds:
        generated_fields = [name for name in ("SolarYield_East", "SolarYield_South", "SolarYield_West") if name in ds]
        if generated_fields:
            total_generated = np.full(len(times), np.nan, dtype=np.float64)
            valid_generated = np.zeros(len(times), dtype=bool)
            field_values: list[np.ndarray] = []
            for field_name in generated_fields:
                values = np.asarray(ds[field_name].values, dtype=np.float64)
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

    generated_total_da = assignments.get("CumulativePowerGeneratedTotal", ds.get("CumulativePowerGeneratedTotal"))
    utilised_da = assignments.get("CumulativePowerUtilised", ds.get("CumulativePowerUtilised"))
    if generated_total_da is not None and utilised_da is not None:
        generated_total = np.asarray(generated_total_da.values, dtype=np.float64)
        utilised = np.asarray(utilised_da.values, dtype=np.float64)
        balance = np.full(len(times), np.nan, dtype=np.float64)
        valid_balance = np.isfinite(generated_total) & np.isfinite(utilised)
        balance[valid_balance] = generated_total[valid_balance] - utilised[valid_balance]
        assignments["CumulativePowerBalance"] = xr.DataArray(
            balance,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    if not assignments:
        return ds
    return ds.assign(**assignments)


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
        drawstyle = "steps-post" if is_status_like_var(name) else "default"
        ax.plot(times, values, color=colors[idx % len(colors)], linewidth=0.8, drawstyle=drawstyle)
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
    span_hours = max((times.max() - times.min()) / np.timedelta64(1, "h"), 1.0)
    interval = 1 if span_hours <= 18 else 2 if span_hours <= 36 else 6
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)


def save_summary_png(
    ds: xr.Dataset,
    instrument: str,
    title: str,
    output: Path,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> int:
    ds = _prepare_summary_dataset(ds, instrument)
    ds = downsample_time(ds, max_time_samples=max_time_samples)
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
        for trace, values in rows:
            target = right_ax if trace.axis == "right" and right_ax is not None else ax
            drawstyle = "steps-post" if trace.step else "default"
            trace_times, trace_values = _trace_time_values(times, values)
            if len(trace_times) == 0:
                continue
            target.plot(trace_times, trace_values, color=trace.color, linewidth=1.25, drawstyle=drawstyle, label=trace.label)
            if trace.axis == "right" and right_color is None:
                right_color = trace.color
            if trace.axis == "left" and left_color is None:
                left_color = trace.color

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
    start_stamp = times.min().strftime("%b %d, %Y")
    axes[-1].set_xlabel(f"Hours after 00:00 UTC on {start_stamp}", fontsize=12)
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
    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    title = _window_title("Latest 24 hours", instrument)
    return save_summary_png(window, instrument=instrument, title=title, output=output, max_time_samples=max_time_samples)


def build_summary_plotly(
    ds: xr.Dataset,
    instrument: str,
    title: str | None = None,
    max_time_samples: int = INTERACTIVE_MAX_TIME_SAMPLES,
) -> go.Figure:
    vertical_spacing = 0.04
    ds = _prepare_summary_dataset(ds, instrument)
    ds = downsample_time(ds, max_time_samples=max_time_samples)
    times = _time_index(ds)
    panels = _active_panels(ds, instrument)
    if len(times) == 0 or not panels:
        raise ValueError(f"No summary time-series panels available for {instrument}")

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
            x=PLOTLY_LEGEND_X,
            xanchor="left",
            y=max(0.02, panel_top - 0.02),
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=PLOT_BORDER,
            borderwidth=1,
            font=dict(size=10, color=PLOT_TEXT),
            tracegroupgap=2,
        )
        left_color = None
        right_color = None
        for trace, values in rows:
            secondary = trace.axis == "right" and panel.right_axis_label is not None
            if secondary and right_color is None:
                right_color = trace.color
            if not secondary and left_color is None:
                left_color = trace.color
            trace_times, trace_values = _trace_time_values(times, values)
            if len(trace_times) == 0:
                continue
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
        fig.update_yaxes(
            title_text=panel.left_axis_label,
            showgrid=True,
            gridcolor=PLOT_GRID,
            linecolor=PLOT_LINE,
            tickfont=dict(color=left_color or COLOR["black"], size=10),
            title_font=dict(color=left_color or COLOR["black"], size=11),
            row=row_index,
            col=1,
            secondary_y=False,
        )
        if panel.right_axis_label is not None:
            fig.update_yaxes(
                title_text=panel.right_axis_label,
                showgrid=False,
                linecolor=PLOT_LINE,
                tickfont=dict(color=right_color or COLOR["black"], size=10),
                title_font=dict(color=right_color or COLOR["black"], size=11),
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
        domain=[0.0, PLOTLY_PANEL_DOMAIN_END],
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
        height=max(620, min(1800, 280 * len(panels) + 90)),
        margin=dict(l=80, r=80, t=60, b=70),
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
