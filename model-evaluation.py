"""Standalone AURORA model-evaluation viewer.

This Panel app is intentionally separate from the operational dashboard tabs
while the LES and Cloudnet evaluation workflow is still changing.
"""

from __future__ import annotations

from base64 import b64encode
from collections import OrderedDict
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import xarray as xr

pn.extension("plotly", sizing_mode="stretch_width")

APP_DIR = Path(__file__).resolve().parent
CASE_ROOT = Path(os.environ.get("AURORA_MODEL_EVALUATION_CASE_ROOT", "/data/aurora/les/cases"))
CASE_ID = os.environ.get("AURORA_MODEL_EVALUATION_CASE_ID", "aurora_multistream_pilot_20260520_20260602")
OUTPUT_ROOT = CASE_ROOT / CASE_ID
SCORECARD_CF_V0_STEM = "scorecard_cf_model_cf_vs_cloudnet_cf_v_cf_a_20260621"
OBSERVATION_AUDIT_STEM = "observation_audit_cloudnet_cf_sources_20260621"
IWC_SCORECARD_STEM = "scorecard_iwc_model_iwc_vs_cloudnet_iwc_iwc_adv_20260621"
LWP_SCORECARD_STEM = "scorecard_lwp_model_lwp_vs_hatpro_lwp_20260622"
CL61_SCORECARD_STEM = "scorecard_cl61_beta_att_v0_20260621"
WBAND_RADAR_SCORECARD_STEM = "scorecard_wband_radar_z_vs_cloudnet_z_20260622"
PAMTRA_WBAND_RADAR_SCORECARD_STEM = (
    "scorecard_pamtra_wband_cosmo_1mom_sensitivity_z_vs_cloudnet_z_20260622"
)
PAMTRA_WBAND_RADAR_SENSITIVITY_SWEEP_STEM = (
    "scorecard_pamtra_wband_cosmo_1mom_sensitivity_margin_sweep_20260622"
)
PAMTRA_WBAND_HYDROMETEOR_SWEEP_STEM = "pamtra_wband_hydrometeor_contribution_sweep_20260622"
PAMTRA_WBAND_DESCRIPTOR_GROUP_SWEEP_STEM = "pamtra_wband_descriptor_group_sweep_20260622"
PAMTRA_WBAND_AMPLITUDE_SWEEP_STEM = "pamtra_wband_amplitude_sweep_20260622"
PAMTRA_WBAND_TARGETED_CALIBRATION_STEM = "pamtra_wband_targeted_calibration_20260622"
PAMTRA_WBAND_CALIBRATION_GATE_STEM = "pamtra_wband_constrained_calibration_gate_20260622"
PAMTRA_WBAND_DESCRIPTOR_PHYSICS_SWEEP_STEM = "pamtra_wband_descriptor_physics_sweep_20260622"
PAMTRA_WBAND_DESCRIPTOR_PSD_SWEEP_STEM = "pamtra_wband_descriptor_psd_sweep_20260622"
ARTIFACT_STEMS = {
    "scorecard": SCORECARD_CF_V0_STEM,
    "observation_audit": OBSERVATION_AUDIT_STEM,
    "iwc_scorecard": IWC_SCORECARD_STEM,
    "lwp_scorecard": LWP_SCORECARD_STEM,
    "cl61_scorecard": CL61_SCORECARD_STEM,
    "wband_radar_scorecard": WBAND_RADAR_SCORECARD_STEM,
    "pamtra_wband_radar_scorecard": PAMTRA_WBAND_RADAR_SCORECARD_STEM,
    "pamtra_wband_radar_sensitivity_sweep": PAMTRA_WBAND_RADAR_SENSITIVITY_SWEEP_STEM,
    "pamtra_wband_hydrometeor_sweep": PAMTRA_WBAND_HYDROMETEOR_SWEEP_STEM,
    "pamtra_wband_descriptor_group_sweep": PAMTRA_WBAND_DESCRIPTOR_GROUP_SWEEP_STEM,
    "pamtra_wband_amplitude_sweep": PAMTRA_WBAND_AMPLITUDE_SWEEP_STEM,
    "pamtra_wband_targeted_calibration": PAMTRA_WBAND_TARGETED_CALIBRATION_STEM,
    "pamtra_wband_calibration_gate": PAMTRA_WBAND_CALIBRATION_GATE_STEM,
    "pamtra_wband_descriptor_physics_sweep": PAMTRA_WBAND_DESCRIPTOR_PHYSICS_SWEEP_STEM,
    "pamtra_wband_descriptor_psd_sweep": PAMTRA_WBAND_DESCRIPTOR_PSD_SWEEP_STEM,
}
ARTIFACT_TITLES = {
    "scorecard": "CF scorecard",
    "observation_audit": "Observation audit",
    "iwc_scorecard": "IWC scorecard",
    "lwp_scorecard": "HATPRO LWP diagnostic",
    "cl61_scorecard": "CL61 diagnostic",
    "wband_radar_scorecard": "W-band radar scorecard",
    "pamtra_wband_radar_scorecard": "PAMTRA W-band sensitivity scorecard",
    "pamtra_wband_radar_sensitivity_sweep": "PAMTRA W-band sensitivity-margin sweep",
    "pamtra_wband_hydrometeor_sweep": "PAMTRA W-band hydrometeor sweep",
    "pamtra_wband_descriptor_group_sweep": "PAMTRA W-band descriptor-group sweep",
    "pamtra_wband_amplitude_sweep": "PAMTRA W-band amplitude sweep",
    "pamtra_wband_targeted_calibration": "PAMTRA W-band targeted calibration",
    "pamtra_wband_calibration_gate": "PAMTRA W-band calibration gate",
    "pamtra_wband_descriptor_physics_sweep": "PAMTRA W-band descriptor-physics sweep",
    "pamtra_wband_descriptor_psd_sweep": "PAMTRA W-band descriptor PSD sweep",
}

THEME_TEXT = "#22313f"
THEME_MUTED = "#5f6c7b"
THEME_BORDER = "#d8e1e8"
THEME_GRID = "#e5eaef"
THEME_PANEL = "#fbfcfd"
THEME_ACCENT = "#0b7285"


def _asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    return f"data:{mime};base64,{b64encode(path.read_bytes()).decode('utf-8')}"


DASHBOARD_LOGO = _asset_data_uri(APP_DIR / "assets" / "logo.png")


def _path(*parts: str) -> Path:
    return OUTPUT_ROOT.joinpath(*parts)


RUNS: OrderedDict[str, dict[str, object]] = OrderedDict(
    [
        (
            "cm1_reference_dry_run",
            {
                "label": "CM1 reference spin-up",
                "model": "CM1",
                "status": "spin-up: no observation overlap",
                "summary": "64 x 64 x 120, Thompson, native CM1 output converted to Cloudnet model contract",
                "cloudnet_model": _path("model", "cm1_reference", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_reference_dry_run",
                    "aurora_multistream_pilot_20260520_20260602_cm1_l3-cf.nc",
                ),
                "l3_iwc": _path(
                    "cloudnet_l3",
                    "cm1_reference_dry_run",
                    "aurora_multistream_pilot_20260520_20260602_cm1_l3-iwc.nc",
                ),
                "run_dir": _path("model", "cm1_reference", "run_20260521"),
                "uuid": "l3-cf 98fa587f-c092-4a29-8586-bb16df847ea9; l3-iwc 0b3d2345-a983-4f68-881e-49c71a50cb49",
                "runtime": "2 native CM1 outputs, 00:00-00:05 UTC; Cloudnet L3 CF/IWC handoff complete",
                "scorecard_png": _path(
                    "model",
                    "cm1_reference",
                    "scorecard_cf_model_cf_vs_cloudnet_cf_v_cf_a_20260621.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_reference",
                    "scorecard_cf_model_cf_vs_cloudnet_cf_v_cf_a_20260621.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_reference",
                    "scorecard_cf_model_cf_vs_cloudnet_cf_v_cf_a_20260621.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_rh105_25_60",
            {
                "label": "CM1 04:00 RH105 2.5-6.0 km",
                "model": "CM1",
                "status": "best 04:00 ensemble",
                "summary": "64 x 64 x 200, 1800 s, Thompson, 105% RH from 2.5-6.0 km",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "cloudnet_model.nc",
                ),
                "wband_radar": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "wband_radar_proxy_20260622.nc",
                ),
                "pamtra_wband_radar": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "pamtra_wband_radar_cosmo_1mom_20260622.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_rh105_25_60_official_l3-cf_2026-05-21.nc",
                ),
                "l3_iwc": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_rh105_25_60_official_l3-iwc_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_0400_thompson_tall_rh105_25_60", "run_20260521"),
                "uuid": "l3-cf e63672c1-26d5-4bda-8839-84b199ce6823; l3-iwc f18143be-c574-4dc7-9482-f9493fbbd4a7",
                "runtime": "completed normally at 1800 s; 16 native outputs; CF/IWC handoff complete",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
            {
                "label": "CM1 04:00 ERA5 qv+theta tau10",
                "model": "CM1",
                "status": "best ERA5 relaxation",
                "summary": "64 x 64 x 200, 1800 s, Thompson, ERA5 qv+theta relaxation to 05:00 UTC",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
                    "run_20260521",
                ),
                "uuid": "39349240-389b-46cb-9e07-588c271e952c",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau10",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
            {
                "label": "CM1 04:00 ERA5 lsnudge qv+theta",
                "model": "CM1",
                "status": "ERA5 runtime nudging",
                "summary": "64 x 64 x 200, 1800 s, Thompson, CM1-native ERA5 qv+theta lsnudge",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
                    "run_20260521",
                ),
                "uuid": "360bc46f-2c60-48c0-9f72-c64b95c36bd7",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsnudge_qvtheta_tau10",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
            {
                "label": "CM1 04:00 ERA5 qv tau10",
                "model": "CM1",
                "status": "ERA5 relaxation",
                "summary": "64 x 64 x 200, 1800 s, Thompson, ERA5 qv relaxation to 05:00 UTC",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_qv_nudge_05_tau10_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
                    "run_20260521",
                ),
                "uuid": "52f55697-84c5-4729-a917-c5e1cd343691",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau10",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
            {
                "label": "CM1 04:00 ERA5 LSF qv",
                "model": "CM1",
                "status": "ERA5 time-series projection",
                "summary": "64 x 64 x 200, 1800 s, Thompson, ERA5 qv time-series target projected to 04:30 UTC",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_lsf_qv_tau10_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
                    "run_20260521",
                ),
                "uuid": "432f2f45-5653-4b8a-9d0e-97c768358805",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qv_tau10",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
            {
                "label": "CM1 04:00 ERA5 LSF qv+theta",
                "model": "CM1",
                "status": "ERA5 time-series projection",
                "summary": "64 x 64 x 200, 1800 s, Thompson, ERA5 qv+theta time-series target projected to 04:30 UTC",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
                    "run_20260521",
                ),
                "uuid": "392ec0e5-d89c-4ffa-bd34-b3bebe45428a",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvtheta_tau10",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
            {
                "label": "CM1 04:00 ERA5 LSF qv+theta+wind",
                "model": "CM1",
                "status": "ERA5 time-series projection",
                "summary": "64 x 64 x 200, 1800 s, Thompson, ERA5 qv+theta+wind time-series target projected to 04:30 UTC",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
                    "run_20260521",
                ),
                "uuid": "012b861d-2c52-499d-9a86-7771d3dfc7cb",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_lsf_qvthetauv_tau10",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
            {
                "label": "CM1 04:00 ERA5 qv+theta tau30",
                "model": "CM1",
                "status": "ERA5 relaxation",
                "summary": "64 x 64 x 200, 1800 s, Thompson, weaker ERA5 qv+theta relaxation",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
                    "run_20260521",
                ),
                "uuid": "72ce4056-6a2b-4441-b004-17927b6bbdd1",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qvtheta_nudge_05_tau30",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
            {
                "label": "CM1 04:00 ERA5 qv tau30",
                "model": "CM1",
                "status": "ERA5 relaxation",
                "summary": "64 x 64 x 200, 1800 s, Thompson, weaker ERA5 qv relaxation",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_era5_qv_nudge_05_tau30_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
                    "run_20260521",
                ),
                "uuid": "5515c9a8-a612-4ab5-a30f-700ffa2c10f8",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_era5_qv_nudge_05_tau30",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_rh105_cool_25_45",
            {
                "label": "CM1 04:00 RH105 cool 2.5-4.5 km",
                "model": "CM1",
                "status": "04:00 ensemble",
                "summary": "64 x 64 x 200, 1800 s, Thompson, 105% RH and -0.25 K from 2.5-4.5 km",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_cool_25_45",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_rh105_cool_25_45",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_rh105_cool_25_45_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_0400_thompson_tall_rh105_cool_25_45", "run_20260521"),
                "uuid": "5ce7606f-74df-47e0-ac20-03a9d3914dfc",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_cool_25_45",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_cool_25_45",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_cool_25_45",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall_rh105_25_45",
            {
                "label": "CM1 04:00 RH105 2.5-4.5 km",
                "model": "CM1",
                "status": "04:00 ensemble",
                "summary": "64 x 64 x 200, 1800 s, Thompson, 105% RH from 2.5-4.5 km",
                "cloudnet_model": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_45",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_rh105_25_45",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_rh105_25_45_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_0400_thompson_tall_rh105_25_45", "run_20260521"),
                "uuid": "70221d67-c1d0-45cc-a238-339e0c9c5fd1",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_45",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_45",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall_rh105_25_45",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_0400_thompson_tall",
            {
                "label": "CM1 04:00 Thompson tall",
                "model": "CM1",
                "status": "science candidate",
                "summary": "64 x 64 x 200, 1800 s, Thompson, 10 km top, ERA5 04:00 init",
                "cloudnet_model": _path("model", "cm1_0400_thompson_tall", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_0400_thompson_tall", "run_20260521"),
                "uuid": "8551ba28-f6fc-4ab2-b5e0-051e387cb783",
                "runtime": "completed normally at 1800 s; 16 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_0400_thompson_tall",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_0400_thompson_tall",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_0400_thompson_tall",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_forced_moist_thompson_aligned",
            {
                "label": "CM1 forced-moist Thompson aligned",
                "model": "CM1",
                "status": "domain mismatch",
                "summary": "64 x 64 x 80, 4500 s, Thompson, official categorize overlap",
                "cloudnet_model": _path(
                    "model",
                    "cm1_forced_moist_thompson_aligned",
                    "cloudnet_model.nc",
                ),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_forced_moist_thompson_aligned",
                    "aurora_multistream_pilot_20260520_20260602_cm1_forced_moist_thompson_aligned_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_forced_moist_thompson_aligned", "run_20260521"),
                "uuid": "f6504c20-e953-4c7d-94c7-0024961c1774",
                "runtime": "completed normally; 38 native outputs",
                "scorecard_png": _path(
                    "model",
                    "cm1_forced_moist_thompson_aligned",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_forced_moist_thompson_aligned",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_forced_moist_thompson_aligned",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "cm1_forced_moist_thompson",
            {
                "label": "CM1 forced-moist Thompson",
                "model": "CM1",
                "status": "native cloud",
                "summary": "64 x 64 x 80, 900 s, Thompson, 105% RH layer, 4 MPI ranks",
                "cloudnet_model": _path("model", "cm1_forced_moist_thompson", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_forced_moist_thompson",
                    "aurora_multistream_pilot_20260520_20260602_cm1_forced_moist_thompson_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_forced_moist_thompson", "run_20260521"),
                "uuid": "54c1d650-6627-4b5d-bfc4-b887b0671b2c",
                "runtime": "1:51.25 wall, 398% CPU, 520292 kB RSS",
                "scorecard_png": _path(
                    "model",
                    "cm1_forced_moist_thompson",
                    "scorecard_cf_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "cm1_forced_moist_thompson",
                    "scorecard_cf_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "cm1_forced_moist_thompson",
                    "scorecard_cf_20260620.json",
                ),
            },
        ),
        (
            "cm1_1h_thompson",
            {
                "label": "CM1 Thompson 1 h",
                "model": "CM1",
                "status": "dry baseline",
                "summary": "64 x 64 x 80, 3600 s, Thompson, 4 MPI ranks",
                "cloudnet_model": _path("model", "cm1_1h_thompson", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_1h_thompson",
                    "aurora_multistream_pilot_20260520_20260602_cm1_1h_thompson_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_1h_thompson", "run_20260521"),
                "uuid": "2f5841c7-04b6-4b20-a5ad-580a47224a2a",
                "runtime": "6:11.06 wall, 399% CPU, 519472 kB RSS",
            },
        ),
        (
            "cm1_short_thompson",
            {
                "label": "CM1 Thompson 15 min",
                "model": "CM1",
                "status": "scale-up check",
                "summary": "64 x 64 x 80, 900 s, Thompson, 4 MPI ranks",
                "cloudnet_model": _path("model", "cm1_short_thompson", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_short_thompson",
                    "aurora_multistream_pilot_20260520_20260602_cm1_short_thompson_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_short_thompson", "run_20260521"),
                "uuid": "f2994f48-ce5e-4136-a4f2-6bc0c0a76aa7",
                "runtime": "1:32.02 wall, 387% CPU, 518944 kB RSS",
            },
        ),
        (
            "cm1_smoke_thompson",
            {
                "label": "CM1 Thompson smoke",
                "model": "CM1",
                "status": "runtime proof",
                "summary": "32 x 32 x 60, 60 s, Thompson, cached lookup tables",
                "cloudnet_model": _path("model", "cm1_smoke_thompson", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_smoke_thompson",
                    "aurora_multistream_pilot_20260520_20260602_cm1_thompson_smoke_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_smoke_thompson", "run_20260521"),
                "uuid": "07f2a46b-850d-49ca-bb71-51415b80b9cc",
                "runtime": "2.27 s wall after cached Thompson table staging",
            },
        ),
        (
            "cm1_smoke_kessler",
            {
                "label": "CM1 Kessler smoke",
                "model": "CM1",
                "status": "runtime proof",
                "summary": "32 x 32 x 60, 60 s, Kessler warm rain",
                "cloudnet_model": _path("model", "cm1_smoke_kessler", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_smoke_kessler",
                    "aurora_multistream_pilot_20260520_20260602_cm1_l3-cf.nc",
                ),
                "l3_cf_candidates": [
                    _path(
                        "cloudnet_l3",
                        "cm1_smoke_kessler",
                        "aurora_multistream_pilot_20260520_20260602_cm1_smoke_l3-cf_2026-05-21.nc",
                    ),
                ],
                "run_dir": _path("model", "cm1_smoke_kessler", "run_20260521"),
                "uuid": "f0fdcad7-2ce0-459d-987e-9970955dd6e3",
            },
        ),
        (
            "era5_reference",
            {
                "label": "ERA5 reference",
                "model": "ERA5",
                "status": "reference",
                "summary": "Leeds ERA5 pressure-level reference",
                "cloudnet_model": _path("model", "era5_reference", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "era5_reference",
                    "aurora_multistream_pilot_20260520_20260602_era5_l3-cf.nc",
                ),
                "l3_cf_candidates": [
                    _path(
                        "cloudnet_l3",
                        "era5_reference",
                        "aurora_multistream_pilot_20260520_20260602_era5_l3-cf_2026-05-21.nc",
                    ),
                ],
                "uuid": "594d87e1-3bf2-428a-a22d-4beffd9ad344",
                "scorecard_png": _path(
                    "model",
                    "era5_reference",
                    "scorecard_cf_official_categorize_20260620.png",
                ),
                "scorecard_markdown": _path(
                    "model",
                    "era5_reference",
                    "scorecard_cf_official_categorize_20260620.md",
                ),
                "scorecard_json": _path(
                    "model",
                    "era5_reference",
                    "scorecard_cf_official_categorize_20260620.json",
                ),
            },
        ),
        (
            "ifs_hres_reference",
            {
                "label": "IFS/HRES reference (pending)",
                "model": "IFS",
                "status": "blocked: MARS entitlement",
                "summary": "Historical 2026-05-21 IFS/HRES request is scripted and authenticates, but the ECMWF account lacks services/mars access.",
            },
        ),
        (
            "les_bridge_reference",
            {
                "label": "ERA5 LES bridge",
                "model": "LES bridge",
                "status": "diagnostic",
                "summary": "ERA5-seeded subcolumn representativeness diagnostic",
                "cloudnet_model": _path("model", "les_bridge_reference", "cloudnet_model.nc"),
                "l3_cf": _path(
                    "cloudnet_l3",
                    "les_bridge_reference",
                    "aurora_multistream_pilot_20260520_20260602_les_bridge_l3-cf.nc",
                ),
                "l3_cf_candidates": [
                    _path(
                        "cloudnet_l3",
                        "les_bridge_reference",
                        "aurora_multistream_pilot_20260520_20260602_les_bridge_l3-cf_2026-05-21.nc",
                    ),
                ],
                "uuid": "0ed9db33-2101-4b32-999e-b3fe61315dc5",
            },
        ),
    ]
)

DATASETS = OrderedDict(
    [
        ("Cloudnet L3 CF", "l3_cf"),
        ("Cloudnet L3 IWC", "l3_iwc"),
        ("Cloudnet model", "cloudnet_model"),
        ("Synthetic W-band radar", "wband_radar"),
        ("PAMTRA W-band radar", "pamtra_wband_radar"),
        ("CF scorecard", "scorecard"),
        ("IWC scorecard", "iwc_scorecard"),
        ("HATPRO LWP diagnostic", "lwp_scorecard"),
        ("Observation audit", "observation_audit"),
        ("CL61 diagnostic", "cl61_scorecard"),
        ("W-band radar scorecard", "wband_radar_scorecard"),
        ("PAMTRA W-band scorecard", "pamtra_wband_radar_scorecard"),
        ("PAMTRA W-band sensitivity sweep", "pamtra_wband_radar_sensitivity_sweep"),
        ("PAMTRA W-band hydrometeor sweep", "pamtra_wband_hydrometeor_sweep"),
        ("PAMTRA W-band descriptor group sweep", "pamtra_wband_descriptor_group_sweep"),
        ("PAMTRA W-band amplitude sweep", "pamtra_wband_amplitude_sweep"),
        ("PAMTRA W-band targeted calibration", "pamtra_wband_targeted_calibration"),
        ("PAMTRA W-band calibration gate", "pamtra_wband_calibration_gate"),
        ("PAMTRA W-band descriptor physics sweep", "pamtra_wband_descriptor_physics_sweep"),
        ("PAMTRA W-band descriptor PSD sweep", "pamtra_wband_descriptor_psd_sweep"),
    ]
)


def _request_header(name: str) -> str | None:
    try:
        headers = pn.state.headers or {}
    except Exception:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() != wanted:
            continue
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)
    return None


def _request_args() -> dict[str, str]:
    try:
        doc = pn.state.curdoc
        if not doc or not doc.session_context or not doc.session_context.request:
            return {}
        query_args = getattr(doc.session_context.request, "query_arguments", {}) or {}
    except Exception:
        return {}
    parsed: dict[str, str] = {}
    for key, values in query_args.items():
        if not values:
            continue
        raw = values[0]
        parsed[str(key)] = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    return parsed


def _dataset_path(run_id: str, dataset_id: str) -> Path | None:
    spec = RUNS.get(run_id)
    if not spec:
        return None
    if dataset_id in ARTIFACT_STEMS:
        return _artifact_path(run_id, spec, dataset_id, "png")
    candidates: list[Path] = []
    path = spec.get(dataset_id)
    if isinstance(path, Path):
        candidates.append(path)
    extra = spec.get(f"{dataset_id}_candidates")
    if isinstance(extra, list):
        candidates.extend(candidate for candidate in extra if isinstance(candidate, Path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _scorecard_path(run_id: str, spec: dict[str, object], kind: str) -> Path | None:
    return _artifact_path(run_id, spec, "scorecard", kind)


def _artifact_path(
    run_id: str,
    spec: dict[str, object],
    dataset_id: str,
    kind: str,
) -> Path | None:
    suffix_by_kind = {"png": "png", "markdown": "md", "json": "json"}
    suffix = suffix_by_kind[kind]
    stem = ARTIFACT_STEMS[dataset_id]
    candidates: list[Path] = [
        _path("model", run_id, f"{stem}.{suffix}"),
    ]
    extra = spec.get(f"{dataset_id}_{kind}_candidates")
    if isinstance(extra, list):
        candidates.extend(path for path in extra if isinstance(path, Path))
    configured = spec.get(f"{dataset_id}_{kind}")
    if isinstance(configured, Path):
        candidates.append(configured)
    if dataset_id == "scorecard":
        extra = spec.get(f"scorecard_{kind}_candidates")
        if isinstance(extra, list):
            candidates.extend(path for path in extra if isinstance(path, Path))
        configured = spec.get(f"scorecard_{kind}")
        if isinstance(configured, Path):
            candidates.append(configured)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _dataset_label(dataset_id: str) -> str:
    for label, value in DATASETS.items():
        if value == dataset_id:
            return label
    return dataset_id


def _format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "missing"
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def _format_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        return "missing"


def _axis_values(ds: xr.Dataset, dim: str) -> tuple[list[object], str]:
    if dim in ds.coords:
        values = ds.coords[dim].values
    elif dim in ds:
        values = ds[dim].values
    else:
        values = np.arange(int(ds.sizes.get(dim, 0)))
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr).to_pydatetime().tolist(), dim
    if np.issubdtype(arr.dtype, np.timedelta64):
        seconds = arr.astype("timedelta64[s]").astype(float)
        return (seconds / 60.0).tolist(), "forecast time (min)"
    try:
        return arr.astype(float).tolist(), dim
    except Exception:
        return [str(item) for item in arr.tolist()], dim


def _height_values(ds: xr.Dataset, level_dim: str) -> tuple[list[object], str]:
    for candidate in ("height", "model_height"):
        if candidate not in ds:
            continue
        da = ds[candidate]
        if level_dim not in da.dims:
            continue
        values = np.asarray(da.values)
        if values.ndim > 1:
            axis = da.dims.index(level_dim)
            values = np.nanmean(values, axis=tuple(i for i in range(values.ndim) if i != axis))
        try:
            return values.astype(float).tolist(), f"{candidate} ({da.attrs.get('units', 'm')})"
        except Exception:
            return values.tolist(), candidate
    return list(range(int(ds.sizes.get(level_dim, 0)))), level_dim


def _open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_timedelta=True)


def _finite_numeric_variables(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names: set[str] = set()
    try:
        with _open_dataset(path) as ds:
            for name, da in ds.data_vars.items():
                if not np.issubdtype(da.dtype, np.number):
                    continue
                values = np.asarray(da.values, dtype=float)
                if np.isfinite(values).any():
                    names.add(name)
    except Exception:
        return set()
    return names


def _variable_options(run_id: str, dataset_id: str) -> OrderedDict[str, str]:
    if dataset_id in ARTIFACT_STEMS:
        path = _dataset_path(run_id, dataset_id)
        if path is not None and path.exists():
            return OrderedDict([(f"{ARTIFACT_TITLES[dataset_id]} image", dataset_id)])
        return OrderedDict()
    path = _dataset_path(run_id, dataset_id)
    if path is None or not path.exists():
        return OrderedDict()
    options: OrderedDict[str, str] = OrderedDict()
    try:
        with _open_dataset(path) as ds:
            for name, da in ds.data_vars.items():
                if not np.issubdtype(da.dtype, np.number):
                    continue
                values = np.asarray(da.values, dtype=float)
                finite = values[np.isfinite(values)]
                dims = " x ".join(str(dim) for dim in da.dims) or "scalar"
                if finite.size:
                    suffix = f"{dims}; finite {finite.size}"
                else:
                    suffix = f"{dims}; no finite values"
                options[f"{name} ({suffix})"] = name
    except Exception:
        return OrderedDict()
    return options


def _run_options() -> OrderedDict[str, str]:
    return OrderedDict((str(spec["label"]), run_id) for run_id, spec in RUNS.items())


def _card(label: str, value: object) -> str:
    return (
        "<div class='model-card'>"
        f"<div class='model-card__label'>{escape(str(label))}</div>"
        f"<div class='model-card__value'>{escape(str(value))}</div>"
        "</div>"
    )


def _variable_stats(ds: xr.Dataset, variable: str | None) -> list[tuple[str, str]]:
    if not variable or variable not in ds:
        return []
    values = np.asarray(ds[variable].values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [("selected var", variable), ("finite values", 0)]
    return [
        ("selected var", variable),
        ("finite values", finite.size),
        ("min", f"{np.nanmin(finite):.4g}"),
        ("mean", f"{np.nanmean(finite):.4g}"),
        ("max", f"{np.nanmax(finite):.4g}"),
    ]


def _scorecard_json(run_id: str, spec: dict[str, object]) -> dict[str, object] | None:
    return _artifact_json(run_id, spec, "scorecard")


def _artifact_json(
    run_id: str,
    spec: dict[str, object],
    dataset_id: str,
) -> dict[str, object] | None:
    path = _artifact_path(run_id, spec, dataset_id, "json")
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_cards(run_id: str, spec: dict[str, object], dataset_id: str) -> list[str]:
    if dataset_id == "scorecard":
        return _scorecard_cards(run_id, spec)
    if dataset_id == "observation_audit":
        return _observation_audit_cards(run_id, spec)
    if dataset_id == "iwc_scorecard":
        return _iwc_scorecard_cards(run_id, spec)
    if dataset_id == "lwp_scorecard":
        return _lwp_scorecard_cards(run_id, spec)
    if dataset_id == "cl61_scorecard":
        return _cl61_scorecard_cards(run_id, spec)
    if dataset_id in {"wband_radar_scorecard", "pamtra_wband_radar_scorecard"}:
        return _wband_radar_scorecard_cards(run_id, spec, dataset_id)
    if dataset_id == "pamtra_wband_radar_sensitivity_sweep":
        return _wband_radar_sensitivity_sweep_cards(run_id, spec)
    if dataset_id == "pamtra_wband_hydrometeor_sweep":
        return _pamtra_wband_hydrometeor_sweep_cards(run_id, spec, dataset_id)
    if dataset_id == "pamtra_wband_descriptor_group_sweep":
        return _pamtra_wband_hydrometeor_sweep_cards(run_id, spec, dataset_id)
    if dataset_id in {"pamtra_wband_amplitude_sweep", "pamtra_wband_targeted_calibration"}:
        return _pamtra_wband_amplitude_sweep_cards(run_id, spec)
    if dataset_id == "pamtra_wband_calibration_gate":
        return _pamtra_wband_calibration_gate_cards(run_id, spec)
    if dataset_id in {
        "pamtra_wband_descriptor_physics_sweep",
        "pamtra_wband_descriptor_psd_sweep",
    }:
        return _pamtra_wband_descriptor_physics_sweep_cards(run_id, spec, dataset_id)
    return []


def _scorecard_cards(run_id: str, spec: dict[str, object]) -> list[str]:
    scorecard = _scorecard_json(run_id, spec)
    if not scorecard:
        return []
    comparisons = scorecard.get("comparisons")
    if not isinstance(comparisons, dict):
        return []
    comparison = comparisons.get("cf_V") or comparisons.get("cf_A")
    if not isinstance(comparison, dict):
        return []
    contingency = comparison.get("contingency")
    if not isinstance(contingency, dict):
        return []
    radar = scorecard.get("radar_reflectivity")
    radar = radar if isinstance(radar, dict) else {}
    cards = [
        _card("hits", contingency.get("hits", "n/a")),
        _card("misses", contingency.get("misses", "n/a")),
        _card("false alarms", contingency.get("false_alarms", "n/a")),
        _card("POD", _compact_float(contingency.get("probability_of_detection"))),
        _card("FAR", _compact_float(contingency.get("false_alarm_ratio"))),
        _card("CSI", _compact_float(contingency.get("critical_success_index"))),
    ]
    if radar.get("available"):
        cards.append(_card("max ZE_dBZ", _compact_float(radar.get("max_dbz"))))
    return cards


def _observation_audit_cards(run_id: str, spec: dict[str, object]) -> list[str]:
    audit = _artifact_json(run_id, spec, "observation_audit")
    if not audit:
        return []
    summaries = audit.get("summaries")
    if not isinstance(summaries, dict):
        return []
    summary = summaries.get("cf_V") or summaries.get("cf_A")
    if not isinstance(summary, dict):
        return []
    contingency = summary.get("contingency")
    contingency = contingency if isinstance(contingency, dict) else {}
    alignment = audit.get("source_alignment")
    alignment = alignment if isinstance(alignment, dict) else {}
    return [
        _card("hits", contingency.get("hits", "n/a")),
        _card("misses", contingency.get("misses", "n/a")),
        _card("false alarms", contingency.get("false_alarms", "n/a")),
        _card("raw lidar FA", summary.get("false_alarm_with_raw_lidar_echo_gates", "n/a")),
        _card("raw radar FA", summary.get("false_alarm_with_raw_radar_echo_gates", "n/a")),
        _card("CL61 finite", _compact_float(alignment.get("raw_cl61_finite_fraction"))),
    ]


def _iwc_scorecard_cards(run_id: str, spec: dict[str, object]) -> list[str]:
    scorecard = _artifact_json(run_id, spec, "iwc_scorecard")
    if not scorecard:
        return []
    comparisons = scorecard.get("comparisons")
    if not isinstance(comparisons, dict):
        return []
    comparison = comparisons.get("iwc") or next(iter(comparisons.values()), None)
    if not isinstance(comparison, dict):
        return []
    point = comparison.get("point_metrics")
    point = point if isinstance(point, dict) else {}
    iwp = comparison.get("iwp_metrics")
    iwp = iwp if isinstance(iwp, dict) else {}
    occurrence = comparison.get("ice_occurrence")
    occurrence = occurrence if isinstance(occurrence, dict) else {}
    return [
        _card("valid gates", point.get("valid_points", "n/a")),
        _card("bias kg m-3", _compact_float(point.get("bias_mean"))),
        _card("RMSE kg m-3", _compact_float(point.get("root_mean_square_error"))),
        _card("corr", _compact_float(point.get("pearson_correlation"))),
        _card("IWP bias", _compact_float(iwp.get("iwp_bias_mean_kg_m2"))),
        _card("IWP RMSE", _compact_float(iwp.get("iwp_root_mean_square_error_kg_m2"))),
        _card("hits", occurrence.get("hits", "n/a")),
        _card("CSI", _compact_float(occurrence.get("critical_success_index"))),
    ]


def _lwp_scorecard_cards(run_id: str, spec: dict[str, object]) -> list[str]:
    scorecard = _artifact_json(run_id, spec, "lwp_scorecard")
    if not scorecard:
        return []
    comparisons = scorecard.get("comparisons")
    if not isinstance(comparisons, dict):
        return []
    comparison = comparisons.get("hatpro_lwp") or next(iter(comparisons.values()), None)
    if not isinstance(comparison, dict):
        return []
    point = comparison.get("point_metrics")
    point = point if isinstance(point, dict) else {}
    lwp = comparison.get("lwp_metrics")
    lwp = lwp if isinstance(lwp, dict) else {}
    occurrence = comparison.get("liquid_occurrence")
    occurrence = occurrence if isinstance(occurrence, dict) else {}
    model_mean = lwp.get("model_lwp_mean_kg_m2")
    observed_mean = lwp.get("observed_lwp_mean_kg_m2")
    target = (
        float(observed_mean) / float(model_mean)
        if isinstance(model_mean, (int, float))
        and isinstance(observed_mean, (int, float))
        and float(model_mean) > 0.0
        else None
    )
    return [
        _card("status", scorecard.get("scoring_status", "n/a")),
        _card("valid times", point.get("valid_times", "n/a")),
        _card("model LWP", _compact_float(model_mean)),
        _card("HATPRO LWP", _compact_float(observed_mean)),
        _card("bias kg m-2", _compact_float(lwp.get("lwp_bias_mean_kg_m2"))),
        _card("RMSE kg m-2", _compact_float(lwp.get("lwp_root_mean_square_error_kg_m2"))),
        _card("target ql", _compact_float(target)),
        _card("CSI", _compact_float(occurrence.get("critical_success_index"))),
    ]


def _cl61_scorecard_cards(run_id: str, spec: dict[str, object]) -> list[str]:
    scorecard = _artifact_json(run_id, spec, "cl61_scorecard")
    if not scorecard:
        return []
    if scorecard.get("excluded_from_scoring"):
        diagnostic = scorecard.get("diagnostic_summary")
        diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
        return [
            _card("status", scorecard.get("scoring_status", "excluded")),
            _card("site km", _compact_float(scorecard.get("site_distance_km"))),
            _card("valid gates", diagnostic.get("valid_points", "n/a")),
            _card("sim signal", diagnostic.get("simulated_signal_gates", "n/a")),
            _card("obs signal", diagnostic.get("observed_signal_gates", "n/a")),
            _card("common signal", diagnostic.get("common_signal_gates", "n/a")),
        ]
    contingency = scorecard.get("contingency")
    contingency = contingency if isinstance(contingency, dict) else {}
    base_top = scorecard.get("cloud_base_top")
    base_top = base_top if isinstance(base_top, dict) else {}
    return [
        _card("hits", contingency.get("hits", "n/a")),
        _card("misses", contingency.get("misses", "n/a")),
        _card("false alarms", contingency.get("false_alarms", "n/a")),
        _card("POD", _compact_float(contingency.get("probability_of_detection"))),
        _card("CSI", _compact_float(contingency.get("critical_success_index"))),
        _card("top bias m", _compact_float(base_top.get("cloud_top_bias_mean_m"))),
    ]


def _wband_radar_scorecard_cards(
    run_id: str,
    spec: dict[str, object],
    dataset_id: str = "wband_radar_scorecard",
) -> list[str]:
    scorecard = _artifact_json(run_id, spec, dataset_id)
    if not scorecard:
        return []
    contingency = scorecard.get("contingency")
    contingency = contingency if isinstance(contingency, dict) else {}
    reflectivity = scorecard.get("reflectivity_metrics")
    reflectivity = reflectivity if isinstance(reflectivity, dict) else {}
    base_top = scorecard.get("cloud_base_top")
    base_top = base_top if isinstance(base_top, dict) else {}
    return [
        _card("hits", contingency.get("hits", "n/a")),
        _card("misses", contingency.get("misses", "n/a")),
        _card("false alarms", contingency.get("false_alarms", "n/a")),
        _card("POD", _compact_float(contingency.get("probability_of_detection"))),
        _card("CSI", _compact_float(contingency.get("critical_success_index"))),
        _card("bias dB", _compact_float(reflectivity.get("mean_bias_db"))),
        _card("RMSE dB", _compact_float(reflectivity.get("root_mean_square_error_db"))),
        _card("base bias m", _compact_float(base_top.get("cloud_base_bias_mean_m"))),
    ]


def _wband_radar_sensitivity_sweep_cards(
    run_id: str,
    spec: dict[str, object],
) -> list[str]:
    sweep = _artifact_json(run_id, spec, "pamtra_wband_radar_sensitivity_sweep")
    if not sweep:
        return []
    best = sweep.get("best_margin_by_csi")
    best = best if isinstance(best, dict) else {}
    return [
        _card("best margin dB", _compact_float(best.get("sensitivity_margin_db"))),
        _card("best CSI", _compact_float(best.get("critical_success_index"))),
        _card("POD", _compact_float(best.get("probability_of_detection"))),
        _card("FAR", _compact_float(best.get("false_alarm_ratio"))),
        _card("false alarms", best.get("false_alarms", "n/a")),
        _card("misses", best.get("misses", "n/a")),
    ]


def _pamtra_wband_hydrometeor_sweep_cards(
    run_id: str,
    spec: dict[str, object],
    dataset_id: str,
) -> list[str]:
    sweep = _artifact_json(run_id, spec, dataset_id)
    if not sweep:
        return []
    best = sweep.get("best_descriptor_by_csi")
    best = best if isinstance(best, dict) else {}
    products = sweep.get("products")
    products = products if isinstance(products, list) else []
    by_model = {
        item.get("model_variable"): item
        for item in products
        if isinstance(item, dict)
    }
    by_descriptor = {
        item.get("descriptor_name"): item
        for item in products
        if isinstance(item, dict)
    }
    ql = by_model.get("ql") or {}
    qs = by_model.get("qs") or {}
    liquid_snow = by_descriptor.get("liquid_snow") or {}
    full_cosmo = by_descriptor.get("full_cosmo") or {}
    return [
        _card("best descriptor", best.get("descriptor_name", "n/a")),
        _card("best model var", best.get("model_variable", "n/a")),
        _card("best CSI", _compact_float(best.get("critical_success_index"))),
        _card("best bias dB", _compact_float(best.get("reflectivity_mean_bias_db"))),
        _card("ql CSI", _compact_float(ql.get("critical_success_index"))),
        _card("qs CSI", _compact_float(qs.get("critical_success_index"))),
        _card("liq+snow CSI", _compact_float(liquid_snow.get("critical_success_index"))),
        _card("full CSI", _compact_float(full_cosmo.get("critical_success_index"))),
    ]


def _pamtra_wband_amplitude_sweep_cards(
    run_id: str,
    spec: dict[str, object],
) -> list[str]:
    sweep = _artifact_json(run_id, spec, "pamtra_wband_amplitude_sweep")
    if not sweep:
        return []
    best_bias = sweep.get("best_scale_by_abs_bias")
    best_bias = best_bias if isinstance(best_bias, dict) else {}
    best_csi = sweep.get("best_scale_by_csi")
    best_csi = best_csi if isinstance(best_csi, dict) else {}
    products = sweep.get("products")
    products = products if isinstance(products, list) else []
    baseline = next(
        (
            item
            for item in products
            if isinstance(item, dict) and item.get("scale_label") == "baseline"
        ),
        {},
    )
    return [
        _card("best bias scale", best_bias.get("scale_label", "n/a")),
        _card("bias dB", _compact_float(best_bias.get("reflectivity_mean_bias_db"))),
        _card("best CSI scale", best_csi.get("scale_label", "n/a")),
        _card("best CSI", _compact_float(best_csi.get("critical_success_index"))),
        _card("baseline CSI", _compact_float(baseline.get("critical_success_index"))),
        _card("POD", _compact_float(best_csi.get("probability_of_detection"))),
        _card("FAR", _compact_float(best_csi.get("false_alarm_ratio"))),
        _card("misses", best_csi.get("misses", "n/a")),
        _card("false alarms", best_csi.get("false_alarms", "n/a")),
    ]


def _pamtra_wband_calibration_gate_cards(
    run_id: str,
    spec: dict[str, object],
) -> list[str]:
    gate = _artifact_json(run_id, spec, "pamtra_wband_calibration_gate")
    if not gate:
        return []
    selected = gate.get("selected_candidate")
    selected = selected if isinstance(selected, dict) else {}
    radar_best = gate.get("radar_best_candidate")
    radar_best = radar_best if isinstance(radar_best, dict) else {}
    constraints = gate.get("constraints")
    constraints = constraints if isinstance(constraints, dict) else {}
    iwc = constraints.get("iwc")
    iwc = iwc if isinstance(iwc, dict) else {}
    lwc = constraints.get("lwc")
    lwc = lwc if isinstance(lwc, dict) else {}
    next_spec = str(gate.get("suggested_next_scale_spec") or "n/a")
    if ":" in next_spec:
        next_spec = next_spec.split(":", 1)[1]
    return [
        _card("selected", selected.get("scale_label", "none")),
        _card("selected CSI", _compact_float(selected.get("critical_success_index"))),
        _card("selected bias dB", _compact_float(selected.get("reflectivity_mean_bias_db"))),
        _card("radar best", radar_best.get("scale_label", "n/a")),
        _card("radar best CSI", _compact_float(radar_best.get("critical_success_index"))),
        _card("IWC target", _compact_float(iwc.get("target_scale_factor"))),
        _card("LWC status", lwc.get("status", "n/a")),
        _card("next scale", next_spec),
    ]


def _pamtra_wband_descriptor_physics_sweep_cards(
    run_id: str,
    spec: dict[str, object],
    dataset_id: str = "pamtra_wband_descriptor_physics_sweep",
) -> list[str]:
    sweep = _artifact_json(run_id, spec, dataset_id)
    if not sweep:
        return []
    best_csi = sweep.get("best_variant_by_csi")
    best_csi = best_csi if isinstance(best_csi, dict) else {}
    best_bias = sweep.get("best_variant_by_abs_bias")
    best_bias = best_bias if isinstance(best_bias, dict) else {}
    products = sweep.get("products")
    products = products if isinstance(products, list) else []
    baseline = next(
        (
            item
            for item in products
            if isinstance(item, dict) and item.get("variant_label") == "baseline"
        ),
        {},
    )
    return [
        _card("best variant", best_csi.get("variant_label", "n/a")),
        _card("best CSI", _compact_float(best_csi.get("critical_success_index"))),
        _card("best bias dB", _compact_float(best_bias.get("reflectivity_mean_bias_db"))),
        _card("baseline CSI", _compact_float(baseline.get("critical_success_index"))),
        _card("hits", best_csi.get("hits", "n/a")),
        _card("misses", best_csi.get("misses", "n/a")),
        _card("false alarms", best_csi.get("false_alarms", "n/a")),
    ]


def _compact_float(value: object) -> object:
    return f"{value:.3g}" if isinstance(value, float) else value


def _summary_markup(run_id: str, dataset_id: str, variable: str | None, _clicks: int = 0) -> str:
    spec = RUNS.get(run_id, {})
    path = _dataset_path(run_id, dataset_id)
    label = str(spec.get("label", run_id))
    status = str(spec.get("status", "unknown"))
    summary = str(spec.get("summary", ""))
    uuid = str(spec.get("uuid", ""))
    cards: list[str] = []
    variables = "none"
    file_state = "not configured"
    if dataset_id in ARTIFACT_STEMS:
        cards.extend(_artifact_cards(run_id, spec, dataset_id))
        variables = f"{ARTIFACT_TITLES[dataset_id]} PNG, Markdown and JSON"
        file_state = f"{_format_size(path)}; {_format_mtime(path)}" if path is not None and path.exists() else "missing"
        if not cards:
            cards.append(_card(ARTIFACT_TITLES[dataset_id], file_state))
    elif path is not None and path.exists():
        file_state = f"{_format_size(path)}; {_format_mtime(path)}"
        try:
            with _open_dataset(path) as ds:
                cards.extend(_card(name, size) for name, size in ds.sizes.items())
                cards.extend(_card(name, value) for name, value in _variable_stats(ds, variable))
                variables = ", ".join(escape(str(name)) for name in list(ds.data_vars)[:20])
        except Exception as exc:
            cards.append(_card("dataset", f"unreadable: {exc}"))
    elif path is not None:
        file_state = "missing"
        cards.append(_card("dataset", "missing"))
    else:
        cards.append(_card("dataset", "not configured"))
    run_dir = spec.get("run_dir")
    run_dir_row = ""
    if isinstance(run_dir, Path):
        run_dir_row = f"<tr><th>native run</th><td><code>{escape(str(run_dir))}</code></td></tr>"
    runtime_row = ""
    if spec.get("runtime"):
        runtime_row = f"<tr><th>runtime</th><td>{escape(str(spec['runtime']))}</td></tr>"
    artifact_rows = ""
    if dataset_id in ARTIFACT_STEMS:
        markdown_path = _artifact_path(run_id, spec, dataset_id, "markdown")
        json_path = _artifact_path(run_id, spec, dataset_id, "json")
        if isinstance(markdown_path, Path):
            artifact_rows += f"<tr><th>artifact markdown</th><td><code>{escape(str(markdown_path))}</code></td></tr>"
        if isinstance(json_path, Path):
            artifact_rows += f"<tr><th>artifact json</th><td><code>{escape(str(json_path))}</code></td></tr>"
    path_markup = f"<code>{escape(str(path))}</code>" if path is not None else "not configured"
    return f"""
    <div class="model-shell">
      <div class="model-headline">
        <div>
          <div class="model-title">{escape(label)}</div>
          <div class="model-subtitle">{escape(summary)}</div>
        </div>
        <div class="model-pill">{escape(status)}</div>
      </div>
      <div class="model-grid">{''.join(cards)}</div>
      <div class="model-table-wrap">
        <table class="model-table">
          <tbody>
            <tr><th>dataset</th><td>{escape(_dataset_label(dataset_id))}</td></tr>
            <tr><th>file</th><td>{path_markup}</td></tr>
            {run_dir_row}
            {runtime_row}
            {artifact_rows}
            <tr><th>file state</th><td>{escape(file_state)}</td></tr>
            <tr><th>Cloudnet UUID</th><td>{escape(uuid)}</td></tr>
            <tr><th>variables</th><td>{variables}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    """


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": THEME_MUTED},
    )
    fig.update_layout(
        title=title,
        height=560,
        margin={"l": 70, "r": 35, "t": 58, "b": 65},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": THEME_TEXT},
    )
    return fig


def _figure(run_id: str, dataset_id: str, variable: str | None, _clicks: int = 0) -> go.Figure:
    spec = RUNS.get(run_id, {})
    label = str(spec.get("label", run_id))
    path = _dataset_path(run_id, dataset_id)
    dataset_label = _dataset_label(dataset_id)
    if path is None:
        return _empty_figure(label, "No configured dataset path")
    if not path.exists():
        return _empty_figure(label, f"Missing file: {path}")
    if not variable:
        return _empty_figure(label, "No numeric variable selected")
    try:
        with _open_dataset(path) as ds:
            if variable not in ds:
                return _empty_figure(label, f"Variable not found: {variable}")
            da = ds[variable].squeeze(drop=True)
            units = da.attrs.get("units", "")
            title = f"{label} / {dataset_label} / {variable}"
            if da.ndim >= 2:
                dims = list(da.dims)
                time_dim = "time" if "time" in dims else dims[0]
                level_dim = "level" if "level" in dims else next((dim for dim in dims if dim != time_dim), dims[-1])
                da_plot = da.transpose(time_dim, level_dim, ...)
                values = np.asarray(da_plot.values, dtype=float)
                if values.ndim > 2:
                    values = np.nanmean(values, axis=tuple(range(2, values.ndim)))
                if not np.isfinite(values).any():
                    return _empty_figure(title, f"{variable} has no finite values in this file")
                x, x_label = _axis_values(ds, time_dim)
                y, y_label = _height_values(ds, level_dim)
                fig = go.Figure(
                    go.Heatmap(
                        z=values.T,
                        x=x,
                        y=y,
                        colorscale="Viridis",
                        colorbar={"title": units or variable},
                        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{variable}: %{{z:.4g}}<extra></extra>",
                    )
                )
                fig.update_xaxes(title=x_label)
                fig.update_yaxes(title=y_label)
            elif da.ndim == 1:
                dim = da.dims[0]
                x, x_label = _axis_values(ds, dim)
                y_values = np.asarray(da.values, dtype=float)
                if not np.isfinite(y_values).any():
                    return _empty_figure(title, f"{variable} has no finite values in this file")
                fig = go.Figure(
                    go.Scatter(
                        x=x,
                        y=y_values,
                        mode="lines+markers",
                        line={"color": THEME_ACCENT, "width": 2},
                        marker={"size": 5},
                        hovertemplate=f"{x_label}: %{{x}}<br>{variable}: %{{y:.4g}}<extra></extra>",
                    )
                )
                fig.update_xaxes(title=x_label)
                fig.update_yaxes(title=units or variable)
            else:
                fig = go.Figure(go.Indicator(mode="number", value=float(np.asarray(da.values)), title={"text": variable}))
            fig.update_layout(
                title=title,
                height=590,
                margin={"l": 70, "r": 35, "t": 58, "b": 65},
                paper_bgcolor="white",
                plot_bgcolor="white",
                font={"color": THEME_TEXT},
            )
            fig.update_xaxes(showgrid=True, gridcolor=THEME_GRID)
            fig.update_yaxes(showgrid=True, gridcolor=THEME_GRID)
            return fig
    except Exception as exc:
        return _empty_figure(label, f"Could not render {variable}: {exc}")


def _scorecard_panel(run_id: str, _clicks: int = 0) -> pn.Column:
    return _artifact_panel(run_id, "scorecard", _clicks)


def _artifact_panel(run_id: str, dataset_id: str, _clicks: int = 0) -> pn.Column:
    spec = RUNS.get(run_id, {})
    png_path = _artifact_path(run_id, spec, dataset_id, "png")
    markdown_path = _artifact_path(run_id, spec, dataset_id, "markdown")
    title = ARTIFACT_TITLES.get(dataset_id, "Artifact")
    panes: list[object] = []
    if isinstance(png_path, Path) and png_path.exists():
        data_uri = _asset_data_uri(png_path)
        panes.append(
            pn.pane.HTML(
                f"<img class='scorecard-image' src='{data_uri}' alt='{escape(title)}'>",
                sizing_mode="stretch_width",
            )
        )
    else:
        panes.append(pn.pane.Markdown(f"{title} image is missing.", sizing_mode="stretch_width"))
    if isinstance(markdown_path, Path) and markdown_path.exists():
        panes.append(
            pn.Card(
                pn.pane.Markdown(markdown_path.read_text(encoding="utf-8"), sizing_mode="stretch_width"),
                title=f"{title} tables",
                collapsible=True,
                collapsed=False,
                sizing_mode="stretch_width",
            )
        )
    return pn.Column(*panes, sizing_mode="stretch_width")


run_select = pn.widgets.Select(
    name="Run",
    value="cm1_forced_moist_thompson",
    options=_run_options(),
)
dataset_select = pn.widgets.Select(name="Dataset", value="l3_cf", options=DATASETS)
variable_select = pn.widgets.Select(name="Variable", options=OrderedDict())
refresh_button = pn.widgets.Button(name="Refresh", button_type="primary", width=100)
share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
copy_button = pn.widgets.Button(name="Copy link", button_type="default", width=110)


def _sync_variable_options(*_events) -> None:
    options = _variable_options(run_select.value, dataset_select.value)
    variable_select.options = options
    variable_select.disabled = dataset_select.value in ARTIFACT_STEMS
    values = list(options.values())
    if dataset_select.value in ARTIFACT_STEMS:
        variable_select.value = values[0] if values else None
        return
    path = _dataset_path(run_select.value, dataset_select.value)
    finite_values = _finite_numeric_variables(path) if path is not None else set()
    preferred = (
        "cf_V",
        "cf_A",
        "cf_V_adv",
        "cf_A_adv",
        "model_cf",
        "cloud_fraction",
        "model_iwc",
        "iwc",
        "iwc_adv",
        "Z",
        "radar_reflectivity_dbz",
        "radar_detected_mask",
        "Ze",
        "ql",
        "qi",
        "temperature",
        "pressure",
    )
    if variable_select.value in values:
        return
    for name in preferred:
        if name in values and name in finite_values:
            variable_select.value = name
            return
    for name in values:
        if name in finite_values:
            variable_select.value = name
            return
    variable_select.value = values[0] if values else None


def _share_link() -> str:
    proto = _request_header("X-Forwarded-Proto") or "http"
    host = _request_header("Host") or "127.0.0.1:5006"
    query = urlencode(
        {
            "run": run_select.value or "",
            "dataset": dataset_select.value or "",
            "variable": variable_select.value or "",
        }
    )
    return f"{proto}://{host}/model-evaluation?{query}"


def _refresh_share(*_events) -> None:
    share_url.value = _share_link()


def _apply_query_state() -> None:
    args = _request_args()
    if args.get("run") in RUNS:
        run_select.value = args["run"]
    if args.get("dataset") in set(DATASETS.values()):
        dataset_select.value = args["dataset"]
    _sync_variable_options()
    path = _dataset_path(run_select.value, dataset_select.value)
    finite_values = _finite_numeric_variables(path) if path is not None else set()
    if dataset_select.value in ARTIFACT_STEMS and args.get("variable") in list(variable_select.options.values()):
        variable_select.value = args["variable"]
    elif args.get("variable") in list(variable_select.options.values()) and args["variable"] in finite_values:
        variable_select.value = args["variable"]


run_select.param.watch(_sync_variable_options, "value")
dataset_select.param.watch(_sync_variable_options, "value")
refresh_button.on_click(_sync_variable_options)
for widget in (run_select, dataset_select, variable_select):
    widget.param.watch(_refresh_share, "value")

copy_button.js_on_click(
    args={"share": share_url},
    code="""
    const text = share.value || '';
    if (!text) { return; }
    navigator.clipboard.writeText(text);
    """,
)

_sync_variable_options()
_apply_query_state()
_refresh_share()

summary_panel = pn.bind(
    _summary_markup,
    run_select.param.value,
    dataset_select.param.value,
    variable_select.param.value,
    refresh_button.param.clicks,
)
plot_panel = pn.bind(
    lambda run_id, dataset_id, variable, clicks: _artifact_panel(run_id, dataset_id, clicks)
    if dataset_id in ARTIFACT_STEMS
    else pn.pane.Plotly(
        _figure(run_id, dataset_id, variable, clicks),
        config={"responsive": True},
        sizing_mode="stretch_width",
    ),
    run_select.param.value,
    dataset_select.param.value,
    variable_select.param.value,
    refresh_button.param.clicks,
)

CSS = """
body, .bk {
    font-family: "SF Pro Display","SF Pro","-apple-system","BlinkMacSystemFont","Segoe UI",sans-serif;
    font-size: 15px;
    background: #ffffff;
    color: #22313f;
}
.bk.card, .bk-panel-models-card {
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    box-shadow: none;
    background: #ffffff;
}
.bk-btn, button.bk-btn {
    border-radius: 6px;
    border: 1px solid #c5d0da;
    box-shadow: none;
}
.bk-btn-primary, button.bk-btn-primary {
    background: #0b7285;
    border-color: #0b7285;
    color: #ffffff;
}
.model-shell {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.model-headline {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.model-title {
    font-size: 18px;
    font-weight: 650;
    color: #22313f;
}
.model-subtitle {
    margin-top: 3px;
    font-size: 12px;
    color: #5f6c7b;
}
.model-pill {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid #b7e4dc;
    background: #f1fbf8;
    color: #0b6b5d;
    font-size: 12px;
}
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
}
.model-card {
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #fbfcfd;
    padding: 8px 10px;
}
.model-card__label {
    font-size: 11px;
    color: #647283;
}
.model-card__value {
    margin-top: 2px;
    font-size: 18px;
    font-weight: 650;
    color: #22313f;
}
.model-table-wrap {
    overflow-x: auto;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #ffffff;
}
.model-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 640px;
}
.model-table th,
.model-table td {
    border-bottom: 1px solid #e6ebf1;
    padding: 8px 10px;
    vertical-align: top;
    text-align: left;
    font-size: 12px;
}
.model-table th {
    width: 150px;
    color: #3b4a5a;
    background: #f8fafb;
    font-weight: 650;
}
.model-table code {
    word-break: break-all;
    color: #243b53;
}
.scorecard-image {
    display: block;
    width: 100%;
    height: auto;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
}
.model-controls {
    align-items: flex-end;
    gap: 10px;
}
.model-controls > .bk {
    flex: 1 1 210px;
    min-width: 180px;
}
.model-controls .bk-panel-models-widgets-Button {
    flex: 0 0 auto;
}
@media (max-width: 768px) {
    body, .bk { font-size: 14px; }
    .model-controls > .bk {
        flex: 1 1 100%;
        min-width: 0;
    }
    .model-table {
        min-width: 620px;
    }
}
"""

pn.extension(raw_css=[CSS])

template = pn.template.MaterialTemplate(
    title="AURORA Model Evaluation",
    logo=DASHBOARD_LOGO,
    favicon="https://gamb2le.pages.dev/assets/logo.png",
    header_background=THEME_ACCENT,
    header_color="white",
    main_max_width="1800px",
)

controls = pn.Card(
    pn.Row(
        run_select,
        dataset_select,
        variable_select,
        refresh_button,
        sizing_mode="stretch_width",
        css_classes=["model-controls"],
    ),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
)

share = pn.Card(
    pn.Row(copy_button, share_url, sizing_mode="stretch_width", css_classes=["model-controls"]),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
)

template.main[:] = [
    pn.Column(
        controls,
        pn.panel(summary_panel, sizing_mode="stretch_width"),
        pn.panel(plot_panel, sizing_mode="stretch_width"),
        share,
        sizing_mode="stretch_width",
        margin=0,
    )
]

template.servable(location=True)
