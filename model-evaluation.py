"""Standalone AURORA model-evaluation viewer.

This Panel app is intentionally separate from the operational dashboard tabs
while the LES and Cloudnet evaluation workflow is still changing.
"""

from __future__ import annotations

from base64 import b64encode
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
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
OPERATIONAL_CAMPAIGN_ROOT = Path(
    os.environ.get(
        "AURORA_MODEL_EVALUATION_CAMPAIGN_ROOT",
        f"/data/aurora/les/campaigns/{CASE_ID}",
    )
)
ICELAND_CAMPAIGN_ROOT = Path(
    os.environ.get(
        "AURORA_MODEL_EVALUATION_ICELAND_CAMPAIGN_ROOT",
        "/data/aurora/les/campaigns/aurora_iceland_operational_20260706",
    )
)
SHOW_OPERATIONAL_DETAILS = (
    os.environ.get("AURORA_MODEL_EVALUATION_SHOW_OPERATIONAL_DETAILS") == "1"
)
LEEDS_REPLAY_DAYS = tuple(f"2026-05-{day:02d}" for day in range(21, 28))
NEXT_DATA_REQUIRED_INPUTS = (
    "ERA5 pressure levels",
    "ERA5 single levels",
    "Cloudnet categorize",
    "radar at reference point",
    "surface met",
    "ASFS radiation",
    "ASFS sonic/turbulence",
    "ASFS gas",
    "HATPRO/LWP audit or override",
)
CASE_READINESS_POLICY_GATE_STEM = "case_readiness_policy_gate_20260622"
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
PAMTRA_WBAND_SCATTERING_SWEEP_STEM = "pamtra_wband_scattering_sweep_20260622"
ARTIFACT_STEMS = {
    "case_readiness_policy_gate": CASE_READINESS_POLICY_GATE_STEM,
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
    "pamtra_wband_scattering_sweep": PAMTRA_WBAND_SCATTERING_SWEEP_STEM,
}
ARTIFACT_TITLES = {
    "case_readiness_policy_gate": "Case readiness policy gate",
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
    "pamtra_wband_scattering_sweep": "PAMTRA W-band scattering sweep",
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
    ]
)

DEFAULT_RUN_IDS = (
    "era5_reference",
)

INSTRUMENT_COMPARISON_SPECS = (
    {
        "instrument": "Cloudnet CF",
        "model": "ERA5",
        "model_group": "era5",
        "scorecard": "era5_cloud_fraction",
        "comparison": "cf_V",
        "basis": "Cloudnet L3 CF cf_V",
        "occurrence": "contingency",
        "metric_family": "occurrence",
        "caveat": "ready",
    },
    {
        "instrument": "Cloudnet CF",
        "model": "CM1 full LES",
        "model_group": "cm1",
        "scorecard": "cloud_fraction",
        "comparison": "cf_V",
        "basis": "CM1 Cloudnet L3 CF cf_V",
        "occurrence": "contingency",
        "metric_family": "occurrence",
        "caveat": "ready",
    },
    {
        "instrument": "Cloudnet LWC",
        "model": "ERA5",
        "model_group": "era5",
        "scorecard": "era5_lwc",
        "comparison": "lwc",
        "basis": "Cloudnet L3 LWC",
        "occurrence": "liquid_occurrence",
        "metrics": "point_metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "Cloudnet LWC",
        "model": "CM1 full LES",
        "model_group": "cm1",
        "scorecard": "cm1_lwc",
        "comparison": "lwc",
        "basis": "CM1 Cloudnet L3 LWC",
        "occurrence": "liquid_occurrence",
        "metrics": "point_metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "HATPRO/LWP",
        "model": "ERA5",
        "model_group": "era5",
        "scorecard": "era5_lwc",
        "comparison": "lwc",
        "basis": "model LWP vs audit-gated HATPRO LWP",
        "metrics": "lwp_metrics",
        "metric_family": "column",
        "caveat": "ready",
    },
    {
        "instrument": "HATPRO/LWP",
        "model": "CM1 full LES",
        "model_group": "cm1",
        "scorecard": "cm1_lwc",
        "comparison": "lwc",
        "basis": "CM1 LWP vs audit-gated HATPRO LWP",
        "metrics": "lwp_metrics",
        "metric_family": "column",
        "caveat": "ready",
    },
    {
        "instrument": "Cloudnet IWC",
        "model": "ERA5",
        "model_group": "era5",
        "scorecard": "era5_iwc",
        "comparison": "iwc",
        "basis": "Cloudnet L3 IWC",
        "occurrence": "ice_occurrence",
        "metrics": "point_metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "Cloudnet IWC",
        "model": "CM1 full LES",
        "model_group": "cm1",
        "scorecard": "cm1_iwc",
        "comparison": "iwc",
        "basis": "CM1 Cloudnet L3 IWC",
        "occurrence": "ice_occurrence",
        "metrics": "point_metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "W-band radar",
        "model": "CM1 virtual observatory",
        "model_group": "synthetic",
        "scorecard": "wband_radar",
        "basis": "CM1 synthetic radar vs Cloudnet Z",
        "occurrence": "contingency",
        "metrics": "reflectivity_metrics",
        "metric_family": "occurrence",
        "caveat": "ready",
    },
    {
        "instrument": "CL61 lidar",
        "model": "CM1 virtual observatory",
        "model_group": "synthetic",
        "scorecard": "cl61_diagnostic",
        "basis": "diagnostic only; not colocated",
        "occurrence": "contingency",
        "metric_family": "occurrence",
        "caveat": "not_colocated",
    },
    {
        "instrument": "Surface met",
        "model": "CM1/ERA5 surface",
        "model_group": "surface",
        "scorecard": "surface_met",
        "comparison": "air_temperature",
        "basis": "met station air temperature",
        "metrics": "metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "ASFS radiation",
        "model": "CM1 + RRTMGP/SEB",
        "model_group": "surface",
        "scorecard": "asfs_logger_radiation_surface",
        "comparison": "longwave_downwelling",
        "basis": "ASFS logger longwave down",
        "metrics": "metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "ASFS sonic",
        "model": "CM1 surface diagnostics",
        "model_group": "surface",
        "scorecard": "asfs_sonic_turbulence",
        "comparison": "mean_x_wind",
        "basis": "sonic mean wind and turbulence",
        "metrics_group": "mean_comparisons",
        "metrics": "metrics",
        "metric_family": "continuous",
        "caveat": "ready",
    },
    {
        "instrument": "ASFS gas",
        "model": "CM1 diagnostic background",
        "model_group": "surface",
        "scorecard": "asfs_gas",
        "comparison": "co2_molar_density",
        "basis": "LI-COR CO2 diagnostic",
        "metrics": "metrics",
        "metric_family": "continuous",
        "caveat": "diagnostic_only",
    },
)

INSTRUMENT_GALLERY_SCORECARDS = {
    "Cloudnet CF": (("Cloudnet CF: ERA5", "era5_cloud_fraction"), ("Cloudnet CF: CM1 full LES", "cloud_fraction")),
    "Cloudnet LWC": (("Cloudnet LWC: ERA5", "era5_lwc"), ("Cloudnet LWC: CM1 full LES", "cm1_lwc")),
    "HATPRO/LWP": (("LWP context: ERA5", "era5_lwc"), ("LWP context: CM1 full LES", "cm1_lwc")),
    "Cloudnet IWC": (("Cloudnet IWC: ERA5", "era5_iwc"), ("Cloudnet IWC: CM1 full LES", "cm1_iwc")),
    "W-band radar": (("W-band radar", "wband_radar"),),
    "CL61 lidar": (("CL61 lidar diagnostic", "cl61_diagnostic"),),
    "Surface met": (("Surface met", "surface_met"),),
    "ASFS radiation": (("ASFS radiation", "asfs_logger_radiation_surface"),),
    "ASFS sonic": (("ASFS sonic/turbulence", "asfs_sonic_turbulence"),),
    "ASFS gas": (("ASFS gas", "asfs_gas"),),
}

MODEL_FILTERS = OrderedDict(
    [
        ("All model outputs", "all"),
        ("ERA5", "era5"),
        ("CM1 full LES", "cm1"),
        ("Synthetic / forward operator", "synthetic"),
        ("Surface / SEB diagnostics", "surface"),
    ]
)

METRIC_FAMILY_FILTERS = OrderedDict(
    [
        ("All metric families", "all"),
        ("Occurrence skill", "occurrence"),
        ("Continuous values", "continuous"),
        ("Column / base-top", "column"),
        ("Readiness only", "readiness"),
    ]
)

DATASETS = OrderedDict(
    [
        ("Cloudnet L3 CF", "l3_cf"),
        ("Cloudnet L3 IWC", "l3_iwc"),
        ("Cloudnet model", "cloudnet_model"),
        ("Synthetic W-band radar", "wband_radar"),
        ("PAMTRA W-band radar", "pamtra_wband_radar"),
        ("Case readiness policy gate", "case_readiness_policy_gate"),
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
        ("PAMTRA W-band scattering sweep", "pamtra_wband_scattering_sweep"),
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


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _operational_run_paths(limit: int = 14) -> list[Path]:
    days_root = OPERATIONAL_CAMPAIGN_ROOT / "days"
    if not days_root.exists():
        return []
    paths = sorted(days_root.glob("*/operational_run.json"), reverse=True)
    return paths[:limit]


def _direct_scorecard(day: str, name: str) -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "scorecards" / f"{name}.json")


def _campaign_index() -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "campaign_virtual_observatory_index.json")


def _campaign_process_diagnosis() -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "campaign_process_diagnosis.json")


def _campaign_archive_manifest() -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "archive_manifest.json")


def load_campaign_index() -> dict[str, object] | None:
    return _campaign_index()


def load_day_bundle(day: str) -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "lasso_bundle" / "bundle.json")


def load_scorecard(day: str, name: str) -> dict[str, object] | None:
    return _direct_scorecard(day, name)


def _iceland_preflight() -> dict[str, object] | None:
    latest = _read_json(ICELAND_CAMPAIGN_ROOT / "preflight_latest.json")
    if latest:
        return latest
    days_root = ICELAND_CAMPAIGN_ROOT / "days"
    if not days_root.exists():
        return None
    paths = sorted(days_root.glob("*/preflight.json"), reverse=True)
    return _read_json(paths[0]) if paths else None


def _group_status_summary(groups: object) -> str:
    if not isinstance(groups, dict) or not groups:
        return "<div class='model-note'>No grouped readiness checks have been written yet.</div>"
    rows = []
    for group_id, group in sorted(groups.items()):
        if not isinstance(group, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(str(group_id))}</td>"
            f"<td>{escape(str(group.get('status', 'unknown')))}</td>"
            f"<td>{escape(str(group.get('ready', 0)))}</td>"
            f"<td>{escape(str(group.get('missing', 0)))}</td>"
            f"<td>{escape(str(group.get('blocked', 0)))}</td>"
            f"<td>{escape(str(group.get('diagnostic', 0)))}</td>"
            "</tr>"
        )
    return (
        "<table class='model-table'>"
        "<tr><th>group</th><th>status</th><th>ready</th><th>missing</th>"
        "<th>blocked</th><th>diagnostic</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _iceland_audit_summary(preflight: dict[str, object]) -> str:
    audit = preflight.get("site_metadata_audit")
    if not isinstance(audit, dict):
        return (
            "<div class='model-note'>"
            "Site metadata audit has not been written yet. Run the audit before treating "
            "Iceland inputs as production-ready."
            "</div>"
        )
    blockers = audit.get("blockers", [])
    top_blockers = blockers[:5] if isinstance(blockers, list) else []
    blocker_html = "".join(f"<li>{escape(str(blocker))}</li>" for blocker in top_blockers)
    cards = [
        _card("metadata audit", audit.get("status", "unknown")),
        _card("metadata ready", audit.get("production_ready", False)),
        _card(
            "colocated",
            f"{audit.get('required_ready_count', 'n/a')}/{audit.get('required_count', 'n/a')}",
        ),
        _card("reference", audit.get("reference_dataset_id", "missing")),
    ]
    return (
        "<div class='model-subsection-title'>Site Metadata Audit</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        "<div class='model-note'>"
        f"coordinate source: {escape(str(audit.get('coordinate_source', 'missing')))}; "
        f"site: {escape(str(audit.get('site_latitude', 'n/a')))}, "
        f"{escape(str(audit.get('site_longitude', 'n/a')))}"
        "</div>"
        + (f"<ul class='model-compact-list'>{blocker_html}</ul>" if blocker_html else "")
    )


def _iceland_readiness_panel() -> str:
    preflight = _iceland_preflight()
    if not isinstance(preflight, dict):
        cards = [
            _card("Iceland status", "not staged"),
            _card("planned start", "2026-07-06"),
            _card("next action", "run preflight"),
        ]
        return (
            "<div class='model-section-title'>Iceland Readiness</div>"
            f"<div class='model-grid'>{''.join(cards)}</div>"
            "<div class='model-note'>"
            "No Iceland preflight record has been written yet. The Leeds replay remains the "
            "regression baseline until new colocated campaign data arrive."
            "</div>"
        )
    groups = preflight.get("groups", {})
    cards = [
        _card("Iceland day", preflight.get("day", "missing")),
        _card("preflight", preflight.get("status", "missing")),
        _card("daily readiness", preflight.get("readiness_status", "missing")),
        _card("blockers", len(preflight.get("blockers", []) or [])),
    ]
    notes = preflight.get("notes", [])
    note_html = "".join(
        f"<li>{escape(str(note))}</li>" for note in notes if str(note).strip()
    )
    return (
        "<div class='model-section-title'>Iceland Readiness</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        "<div class='model-note'>"
        f"{escape(str(preflight.get('resume_condition', 'Run preflight before execution.')))}"
        "</div>"
        f"{_iceland_audit_summary(preflight)}"
        f"{_group_status_summary(groups)}"
        + (f"<ul class='model-compact-list'>{note_html}</ul>" if note_html else "")
    )


def _bundle_recipe(day: str) -> dict[str, object]:
    bundle = load_day_bundle(day)
    models = bundle.get("models") if isinstance(bundle, dict) else None
    if not isinstance(models, dict):
        return {}
    recipe = models.get("standard_daily_les_recipe")
    return recipe if isinstance(recipe, dict) else {}


def _hours_text(value: object) -> str:
    compact = _compact_float(value)
    return "n/a" if compact == "n/a" else str(compact)


def _bundle_runtime_summary(day: str) -> dict[str, object]:
    recipe = _bundle_recipe(day)
    return {
        "run_hours": _hours_text(recipe.get("configured_run_time_hours")),
        "spinup_hours": _hours_text(
            float(recipe.get("spin_up_seconds", 0.0)) / 3600.0
            if recipe.get("spin_up_seconds") is not None
            else None
        ),
        "evaluation_hours": _hours_text(recipe.get("evaluation_window_hours")),
        "recipe_class": recipe.get("daily_recipe_class", "unknown"),
    }


def _lasso_bundle_paths(limit: int = 31) -> list[Path]:
    days_root = OPERATIONAL_CAMPAIGN_ROOT / "days"
    if not days_root.exists():
        return []
    paths = sorted(days_root.glob("*/lasso_bundle/bundle.json"), reverse=True)
    return paths[:limit]


def _lasso_bundle_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        payload = _read_json(path)
        compliance_path = path.parent / "compliance.json"
        compliance = _read_json(compliance_path)
        compliance_status = (
            str(compliance.get("status", "missing")) if isinstance(compliance, dict) else "missing"
        )
        compliance_detail = _lasso_compliance_detail(compliance)
        if not payload:
            runtime = _bundle_runtime_summary(path.parents[1].name)
            rows.append(
                {
                    "day": path.parents[1].name,
                    "status": "unreadable",
                    "bundle_json": str(path),
                    "compliance": compliance_status,
                    "compliance_detail": compliance_detail,
                    "compliance_json": str(compliance_path),
                    "modf": "missing",
                    "mmdf": "missing",
                    "cloudnet": "missing",
                    "scorecards": "missing",
                    "seb": "missing",
                    "cm1_runtime_h": runtime["run_hours"],
                    "cm1_eval_h": runtime["evaluation_hours"],
                    "cm1_recipe_class": runtime["recipe_class"],
                    "operational_qa": "missing",
                }
            )
            continue
        day = str(payload.get("day", path.parents[1].name))
        runtime = _bundle_runtime_summary(day)
        rows.append(
            {
                "day": day,
                "status": payload.get("status", "unknown"),
                "bundle_json": payload.get("bundle_json", str(path)),
                "bundle_markdown": payload.get("bundle_markdown"),
                "compliance": compliance_status,
                "compliance_detail": compliance_detail,
                "compliance_json": str(compliance_path),
                "modf": _product_status_summary(payload.get("modf_products")),
                "mmdf": _product_status_summary(payload.get("mmdf_products")),
                "cloudnet": _cloudnet_status_summary(payload.get("cloudnet")),
                "scorecards": _scorecard_status_summary(payload.get("scorecards")),
                "seb": _nested_status(payload, ["forward_operators", "rrtmgp_surface_energy_budget"]),
                "cm1_runtime_h": runtime["run_hours"],
                "cm1_eval_h": runtime["evaluation_hours"],
                "cm1_recipe_class": runtime["recipe_class"],
                "scheduler_policy": _nested_status(payload, ["scheduler_policy"]),
                "scheduler_priority": _nested_value(payload, ["scheduler_policy", "priority"], "n/a"),
                "scheduler_actions": _scheduler_action_summary(payload.get("scheduler_policy")),
                "operational_qa": _nested_value(
                    _operational_summary_for_day(day),
                    ["operational_qa", "status"],
                    "missing",
                ),
            }
        )
    return rows


def _lasso_compliance_detail(compliance: dict[str, object] | None) -> str:
    if not isinstance(compliance, dict):
        return "missing"
    failures = compliance.get("failures")
    warnings = compliance.get("warnings")
    failure_count = len(failures) if isinstance(failures, list) else 0
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    level = compliance.get("compliance_level", "unknown")
    return f"{level}; failures:{failure_count}; warnings:{warning_count}"


def _product_status_summary(products: object) -> str:
    if not isinstance(products, dict) or not products:
        return "missing"
    counts: dict[str, int] = {}
    for product in products.values():
        if isinstance(product, dict):
            status = str(product.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
    if not counts:
        return "missing"
    return ", ".join(f"{status}:{count}" for status, count in sorted(counts.items()))


def _cloudnet_status_summary(cloudnet: object) -> str:
    if not isinstance(cloudnet, dict):
        return "missing"
    products = cloudnet.get("products")
    if not isinstance(products, dict):
        return "missing"
    categorize = _dict_status(products.get("categorize"))
    l3_cf = _dict_status(products.get("l3_cf_era5"))
    lwc = _dict_status(products.get("lwc_source"))
    iwc = _dict_status(products.get("iwc_source"))
    return f"cat:{categorize}; cf:{l3_cf}; lwc:{lwc}; iwc:{iwc}"


def _scorecard_status_summary(scorecards: object) -> str:
    if not isinstance(scorecards, dict):
        return "missing"
    written = 0
    diagnostic = 0
    missing = 0
    for scorecard in scorecards.values():
        if not isinstance(scorecard, dict):
            continue
        status = str(scorecard.get("status", "missing"))
        if status == "ready":
            written += 1
        elif status == "diagnostic_only":
            diagnostic += 1
        else:
            missing += 1
    return f"ready:{written}; diagnostic:{diagnostic}; missing:{missing}"


def _nested_status(payload: dict[str, object], keys: list[str]) -> str:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            return "missing"
        value = value.get(key)
    return _dict_status(value)


def _nested_value(payload: dict[str, object], keys: list[str], default: object = "missing") -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _scheduler_action_summary(policy: object, limit: int = 3) -> str:
    if not isinstance(policy, dict):
        return "-"
    actions = policy.get("actions")
    if not isinstance(actions, list) or not actions:
        return "-"
    names = [
        str(action.get("action", "unknown")) if isinstance(action, dict) else str(action)
        for action in actions
    ]
    suffix = "" if len(names) <= limit else f" +{len(names) - limit}"
    return ", ".join(names[:limit]) + suffix


def _dict_status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status", "missing"))
    return "missing"


def _operational_summary_for_day(day: str) -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "scorecards" / "operational_summary.json")


def _latest_operational_summary(
    index: dict[str, object] | None,
    paths: list[Path],
) -> tuple[str | None, dict[str, object] | None]:
    if isinstance(index, dict):
        days = index.get("days")
        if isinstance(days, list) and days:
            for item in reversed(days):
                if not isinstance(item, dict):
                    continue
                day = item.get("day")
                if day:
                    summary = _operational_summary_for_day(str(day))
                    if summary:
                        return str(day), summary
    for path in paths:
        day = path.parent.name
        summary = _operational_summary_for_day(day)
        if summary:
            return day, summary
    return None, None


def _summary_scorecard(summary: dict[str, object] | None, name: str) -> dict[str, object]:
    if not summary:
        return {}
    scorecards = summary.get("scorecards")
    if not isinstance(scorecards, dict):
        return {}
    scorecard = scorecards.get(name)
    return scorecard if isinstance(scorecard, dict) else {}


def _cf_csi(summary: dict[str, object] | None, name: str) -> object:
    scorecard = _summary_scorecard(summary, name)
    comparisons = scorecard.get("comparisons")
    if not isinstance(comparisons, dict):
        return "n/a"
    comparison = comparisons.get("cf_V") or comparisons.get("cf_A")
    if not isinstance(comparison, dict):
        return "n/a"
    return _compact_float(comparison.get("critical_success_index"))


def _cm1_cf_csi(day: str, summary: dict[str, object] | None) -> object:
    value = _cf_csi(summary, "cloud_fraction")
    if value != "n/a":
        return value
    scorecard = _direct_scorecard(day, "cloud_fraction")
    if not isinstance(scorecard, dict):
        return "n/a"
    comparison = scorecard.get("cf_V") or scorecard.get("cf_A")
    if not isinstance(comparison, dict):
        comparison = scorecard.get("comparisons", {}).get("cf_V") if isinstance(scorecard.get("comparisons"), dict) else {}
    if not isinstance(comparison, dict):
        return "n/a"
    return _compact_float(comparison.get("critical_success_index"))


def _index_cf_metric(
    index: dict[str, object] | None,
    scorecard_name: str,
    observed_variable: str,
    metric: str = "critical_success_index_mean",
) -> object:
    if not index:
        return "n/a"
    scorecards = index.get("scorecard_rollup")
    if not isinstance(scorecards, dict):
        return "n/a"
    scorecard = scorecards.get(scorecard_name)
    if not isinstance(scorecard, dict):
        return "n/a"
    metrics = scorecard.get("cloud_fraction_metrics")
    if not isinstance(metrics, dict):
        return "n/a"
    variable = metrics.get(observed_variable)
    if not isinstance(variable, dict):
        return "n/a"
    return _compact_float(variable.get(metric))


def _process_skill_rollup(index: dict[str, object] | None) -> dict[str, object]:
    if not index:
        return {}
    rollup = index.get("process_skill_rollup")
    return rollup if isinstance(rollup, dict) else {}


def _process_skill_metric(
    index: dict[str, object] | None,
    label: str,
    scorecard_name: str,
    observed_variable: str,
    metric: str = "critical_success_index_mean",
) -> object:
    rollup = _process_skill_rollup(index)
    item = rollup.get(label)
    if not isinstance(item, dict):
        return "n/a"
    scorecards = item.get("scorecards")
    if not isinstance(scorecards, dict):
        return "n/a"
    scorecard = scorecards.get(scorecard_name)
    if not isinstance(scorecard, dict):
        return "n/a"
    metrics = scorecard.get("cloud_fraction_metrics")
    if not isinstance(metrics, dict):
        return "n/a"
    variable = metrics.get(observed_variable)
    if not isinstance(variable, dict):
        return "n/a"
    return _compact_float(variable.get(metric))


def _process_day_count(index: dict[str, object] | None, label: str) -> object:
    rollup = _process_skill_rollup(index)
    item = rollup.get(label)
    if not isinstance(item, dict):
        return 0
    return item.get("day_count", 0)


def _process_diagnoses(diagnosis: dict[str, object] | None) -> dict[str, object]:
    if not diagnosis:
        return {}
    processes = diagnosis.get("process_diagnoses")
    return processes if isinstance(processes, dict) else {}


def _diagnosis_metric(
    diagnosis: dict[str, object] | None,
    label: str,
    metric: str,
    scorecard_name: str = "era5_cloud_fraction",
    observed_variable: str = "cf_V",
) -> object:
    process = _process_diagnoses(diagnosis).get(label)
    if not isinstance(process, dict):
        return "n/a"
    scorecards = process.get("scorecards")
    if not isinstance(scorecards, dict):
        return "n/a"
    scorecard = scorecards.get(scorecard_name)
    if not isinstance(scorecard, dict):
        return "n/a"
    comparison = scorecard.get(observed_variable)
    if not isinstance(comparison, dict):
        return "n/a"
    return _compact_float(comparison.get(metric))


def _index_required_pending(index: dict[str, object] | None) -> list[str]:
    if not index:
        return []
    components = index.get("component_rollup")
    if not isinstance(components, dict):
        return []
    pending = []
    for name, component in components.items():
        if not isinstance(component, dict):
            continue
        if not component.get("required_for_full_virtual_observatory"):
            continue
        try:
            not_ready = int(component.get("not_ready_day_count", 0))
        except (TypeError, ValueError):
            not_ready = 0
        if not_ready:
            pending.append(f"{name} ({not_ready} d)")
    return pending


def _operator_policy_rollup_table(index: dict[str, object] | None) -> str:
    if not isinstance(index, dict):
        return ""
    rollup = index.get("operator_policy_rollup")
    if not isinstance(rollup, dict):
        return ""
    body = []
    for name in ("gas", "turbulence", "radiation"):
        policy = rollup.get(name)
        if not isinstance(policy, dict):
            continue
        blockers = policy.get("blocker_counts")
        top_blockers = "-"
        if isinstance(blockers, dict) and blockers:
            ranked = sorted(
                blockers.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )
            top_blockers = ", ".join(f"{key} ({value})" for key, value in ranked[:4])
        body.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(str(policy.get('ready_day_count', 0)))}</td>"
            f"<td>{escape(str(policy.get('not_ready_day_count', 0)))}</td>"
            f"<td>{escape(str(policy.get('latest_status', 'unknown')))}</td>"
            f"<td>{escape(str(policy.get('latest_ready', False)))}</td>"
            f"<td>{escape(top_blockers)}</td>"
            "</tr>"
        )
    if not body:
        return ""
    return (
        "<div class='model-section-title'>Campaign Operator Policy Rollup</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table asfs-detail-table'>"
        "<thead><tr>"
        "<th>operator</th><th>ready days</th><th>not-ready days</th>"
        "<th>latest status</th><th>latest ready</th><th>top blockers</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _scheduler_policy_rollup(index: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(index, dict):
        return {}
    rollup = index.get("scheduler_policy_rollup")
    return rollup if isinstance(rollup, dict) else {}


def _operational_qa_rollup(index: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(index, dict):
        return {}
    rollup = index.get("operational_qa_rollup")
    return rollup if isinstance(rollup, dict) else {}


def _scheduler_policy_rollup_table(index: dict[str, object] | None) -> str:
    rollup = _scheduler_policy_rollup(index)
    if not rollup:
        return ""
    action_counts = rollup.get("action_counts")
    action_days = rollup.get("action_days")
    if not isinstance(action_counts, dict):
        return ""
    action_days = action_days if isinstance(action_days, dict) else {}
    body = []
    for action, count in sorted(
        action_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    ):
        days = action_days.get(action) if isinstance(action_days, dict) else None
        day_text = ", ".join(str(day) for day in days) if isinstance(days, list) else "-"
        body.append(
            "<tr>"
            f"<td>{escape(str(action))}</td>"
            f"<td>{escape(str(count))}</td>"
            f"<td>{escape(day_text)}</td>"
            "</tr>"
        )
    if not body:
        return ""
    status_counts = rollup.get("status_counts")
    priority_counts = rollup.get("priority_counts")
    caveat_counts = rollup.get("caveat_counts")
    status_text = _count_dict_text(status_counts)
    priority_text = _count_dict_text(priority_counts)
    caveat_text = _count_dict_text(caveat_counts)
    return (
        "<div class='model-section-title'>Scheduler Policy Rollup</div>"
        f"<div class='model-subtitle'>status: {escape(status_text)}; "
        f"priority: {escape(priority_text)}; caveats: {escape(caveat_text)}</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table scheduler-policy-table'>"
        "<thead><tr><th>action</th><th>days</th><th>day list</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _operational_qa_rollup_table(index: dict[str, object] | None) -> str:
    rollup = _operational_qa_rollup(index)
    if not rollup:
        return ""
    missing_counts = rollup.get("missing_required_scorecard_counts")
    missing_days = rollup.get("missing_required_scorecard_days")
    blocked_counts = rollup.get("blocked_required_scorecard_counts")
    status_counts = rollup.get("status_counts")
    missing_counts = missing_counts if isinstance(missing_counts, dict) else {}
    missing_days = missing_days if isinstance(missing_days, dict) else {}
    blocked_counts = blocked_counts if isinstance(blocked_counts, dict) else {}
    body = []
    for scorecard, count in sorted(
        missing_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    ):
        days = missing_days.get(scorecard)
        day_text = ", ".join(str(day) for day in days) if isinstance(days, list) else "-"
        body.append(
            "<tr>"
            f"<td>{escape(str(scorecard))}</td>"
            f"<td>{escape(str(count))}</td>"
            f"<td>{escape(day_text)}</td>"
            "</tr>"
        )
    if not body:
        body.append("<tr><td>-</td><td>0</td><td>-</td></tr>")
    blocked_text = _count_dict_text(blocked_counts)
    return (
        "<div class='model-section-title'>Operational QA Rollup</div>"
        f"<div class='model-subtitle'>status: {escape(_count_dict_text(status_counts))}; "
        f"QA-ready days: {escape(str(rollup.get('ready_day_count', 0)))}; "
        f"QA-incomplete days: {escape(str(rollup.get('qa_incomplete_day_count', 0)))}; "
        f"blocked required scorecards: {escape(blocked_text)}</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table scheduler-policy-table'>"
        "<thead><tr><th>missing required scorecard</th><th>days</th><th>day list</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _archive_manifest_cards(manifest: dict[str, object] | None) -> list[str]:
    if not isinstance(manifest, dict):
        return [_card("archive manifest", "missing")]
    counts = manifest.get("class_counts")
    counts = counts if isinstance(counts, dict) else {}
    items = manifest.get("items")
    item_count = len(items) if isinstance(items, list) else "n/a"
    return [
        _card("archive manifest", manifest.get("status", "unknown")),
        _card("manifest items", item_count),
        _card("active", counts.get("active_campaign", 0)),
        _card("reference", counts.get("reference", 0)),
        _card("archived", counts.get("archived_experiment", 0)),
        _card("retired", counts.get("retired_dead_end", 0)),
    ]


def _archive_manifest_table(manifest: dict[str, object] | None) -> str:
    if not isinstance(manifest, dict):
        return "<div class='model-note'>archive manifest missing</div>"
    counts = manifest.get("class_counts")
    if not isinstance(counts, dict):
        return "<div class='model-note'>archive manifest has no class counts</div>"
    descriptions = manifest.get("archive_classes")
    descriptions = descriptions if isinstance(descriptions, dict) else {}
    rows = []
    for name, count in sorted(counts.items()):
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(name))}</code></td>"
            f"<td>{escape(str(count))}</td>"
            f"<td>{escape(str(descriptions.get(name, '')))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-section-title'>Archive Manifest</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table archive-table'>"
        "<thead><tr><th>class</th><th>count</th><th>policy</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _day_archive_class_summary(day: str, manifest: dict[str, object] | None) -> str:
    if not isinstance(manifest, dict):
        return "missing"
    items = manifest.get("items")
    if not isinstance(items, list):
        return "missing"
    marker = f"/days/{day}/"
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        if marker not in path:
            continue
        archive_class = str(item.get("archive_class", "unknown"))
        counts[archive_class] = counts.get(archive_class, 0) + 1
    return _count_dict_text(counts)


def _scheduler_policy_day_table(index: dict[str, object] | None) -> str:
    if not isinstance(index, dict):
        return ""
    days = index.get("days")
    if not isinstance(days, list):
        return ""
    body = []
    for day in days:
        if not isinstance(day, dict):
            continue
        actions = day.get("scheduler_policy_actions")
        action_text = ", ".join(str(item) for item in actions) if isinstance(actions, list) else "-"
        missing_qa = day.get("operational_qa_missing_required_scorecards")
        missing_qa_text = (
            ", ".join(str(item) for item in missing_qa)
            if isinstance(missing_qa, list)
            else "-"
        )
        body.append(
            "<tr>"
            f"<td>{escape(str(day.get('day', '')))}</td>"
            f"<td>{escape(str(day.get('scheduler_policy_status', 'unknown')))}</td>"
            f"<td>{escape(str(day.get('scheduler_policy_priority', 'normal')))}</td>"
            f"<td>{escape(str(day.get('operational_qa_status', 'unknown')))}</td>"
            f"<td>{escape(str(day.get('operational_qa_ready', 'unknown')))}</td>"
            f"<td>{escape(missing_qa_text)}</td>"
            f"<td>{escape(action_text)}</td>"
            "</tr>"
        )
    if not body:
        return ""
    return (
        "<div class='model-section-title'>Daily Scheduler Policy</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table scheduler-policy-table'>"
        "<thead><tr><th>day</th><th>policy</th><th>priority</th>"
        "<th>QA</th><th>QA ready</th><th>missing QA scorecards</th><th>actions</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _day_status(day: str) -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "status.json")


def _day_command_state(day: str) -> dict[str, object] | None:
    return _read_json(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "command_state.json")


def _missing_required_inputs(status: dict[str, object] | None) -> list[str]:
    if not isinstance(status, dict):
        return []
    checks = status.get("checks")
    if not isinstance(checks, list):
        return []
    missing = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("required") and check.get("status") != "ready":
            missing.append(str(check.get("id", "unknown")))
    return missing


def _command_state_summary(command_state: dict[str, object] | None) -> tuple[str, str]:
    if not isinstance(command_state, dict):
        return "not_started", "-"
    status = str(command_state.get("status", "unknown"))
    failed = command_state.get("failed_command_id")
    if failed:
        return status, str(failed)
    resume = command_state.get("resume_command_id")
    if resume:
        return status, f"resume: {resume}"
    completed = command_state.get("completed_command_count")
    total = command_state.get("total_command_count")
    if completed is not None and total is not None:
        return status, f"{completed}/{total}"
    return status, "-"


def _daily_review_queue_rows(index: dict[str, object] | None, limit: int = 10) -> list[dict[str, object]]:
    archive_manifest = _campaign_archive_manifest()
    indexed_days = []
    if isinstance(index, dict) and isinstance(index.get("days"), list):
        indexed_days = [str(day.get("day")) for day in index["days"] if isinstance(day, dict)]
    status_days = []
    days_root = OPERATIONAL_CAMPAIGN_ROOT / "days"
    if days_root.exists():
        status_days = [
            path.parent.name
            for path in days_root.glob("*/status.json")
            if path.parent.name[:4].isdigit()
        ]
    days = sorted({*indexed_days, *status_days}, reverse=True)[:limit]
    indexed = {}
    if isinstance(index, dict) and isinstance(index.get("days"), list):
        indexed = {
            str(day.get("day")): day
            for day in index["days"]
            if isinstance(day, dict) and day.get("day")
        }
    rows = []
    for day in days:
        indexed_day = indexed.get(day, {})
        status = _day_status(day)
        command_state = _day_command_state(day)
        command_status, command_detail = _command_state_summary(command_state)
        instrument_rows = build_instrument_catalog([day])
        diagnostic = sum(
            1
            for row in instrument_rows
            if row.get("caveat") in {"diagnostic_only", "not_colocated"}
        )
        blocked = sum(
            1
            for row in instrument_rows
            if str(row.get("caveat", "")).startswith("blocked")
        )
        rows.append(
            {
                "day": day,
                "bundle": indexed_day.get("lasso_bundle_status", "missing"),
                "qa": indexed_day.get("operational_qa_status", status.get("status") if status else "missing"),
                "missing_inputs": _missing_required_inputs(status),
                "diagnostic_streams": diagnostic,
                "blocked_streams": blocked,
                "failed_operator": command_detail if "fail" in command_status.lower() else "-",
                "command_status": command_status,
                "command_detail": command_detail,
                "archive_classes": _day_archive_class_summary(day, archive_manifest),
                "actions": indexed_day.get("scheduler_policy_actions", []),
            }
        )
    return rows


def _daily_review_queue_table(index: dict[str, object] | None) -> str:
    rows = _daily_review_queue_rows(index)
    if not rows:
        return "<div class='model-note'>No daily review records found yet.</div>"
    body = []
    for row in rows:
        actions = row.get("actions")
        action_text = _list_summary(actions, limit=2)
        missing_text = _list_summary(row.get("missing_inputs"), limit=4)
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('day', '')))}</td>"
            f"<td>{_badge(row.get('bundle'))}</td>"
            f"<td>{_badge(row.get('qa'))}</td>"
            f"<td>{escape(missing_text)}</td>"
            f"<td>{escape(str(row.get('diagnostic_streams', 0)))}</td>"
            f"<td>{escape(str(row.get('blocked_streams', 0)))}</td>"
            f"<td>{escape(str(row.get('failed_operator', '-')))}</td>"
            f"<td>{escape(str(row.get('command_status', 'unknown')))}</td>"
            f"<td>{escape(str(row.get('command_detail', '-')))}</td>"
            f"<td>{escape(str(row.get('archive_classes', 'missing')))}</td>"
            f"<td>{escape(action_text)}</td>"
            "</tr>"
        )
    return (
        "<div class='model-section-title'>Daily Review Queue</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table daily-review-table'>"
        "<thead><tr><th>day</th><th>bundle</th><th>QA</th><th>missing inputs</th>"
        "<th>diagnostic</th><th>blocked</th><th>failed operator</th>"
        "<th>runner</th><th>runner detail</th><th>archive classes</th><th>QA actions</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _indexed_day_map(index: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if not isinstance(index, dict) or not isinstance(index.get("days"), list):
        return {}
    return {
        str(day.get("day")): day
        for day in index["days"]
        if isinstance(day, dict) and day.get("day")
    }


def _scorecard_state(day: str, scorecard_name: str) -> str:
    scorecard = load_scorecard(day, scorecard_name)
    if not isinstance(scorecard, dict):
        return "missing"
    return str(
        scorecard.get("scoring_status")
        or scorecard.get("status")
        or scorecard.get("readiness_state")
        or "present"
    )


def _source_readiness_state(day: str, product: str) -> str:
    path = (
        OPERATIONAL_CAMPAIGN_ROOT
        / "days"
        / day
        / "cloudnet_source_readiness"
        / f"{product}.json"
    )
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return "missing"
    return str(payload.get("status", payload.get("readiness_state", "present")))


def _day_compliance_state(day: str) -> tuple[str, str]:
    payload = _read_json(
        OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "lasso_bundle" / "compliance.json"
    )
    if not isinstance(payload, dict):
        return "missing", "n/a"
    failures = payload.get("failures", [])
    warnings = payload.get("warnings", [])
    failure_count = len(failures) if isinstance(failures, list) else 0
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    return str(payload.get("status", "unknown")), f"{failure_count}/{warning_count}"


def _cm1_output_count(day: str) -> object:
    run_dir = OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "cm1" / "run"
    if not run_dir.exists():
        return "missing"
    return len(list(run_dir.glob("cm1out_*.nc")))


def _row_for_instrument(
    rows: list[dict[str, object]],
    instrument: str,
    model: str,
) -> dict[str, object]:
    for row in rows:
        if row.get("instrument") == instrument and row.get("model") == model:
            return row
    return {}


def _numeric_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_compact(values: list[object]) -> object:
    numeric = [value for value in (_numeric_or_none(item) for item in values) if value is not None]
    if not numeric:
        return "n/a"
    return _compact_float(sum(numeric) / len(numeric))


def _replay_process_labels(day: str) -> str:
    forcing = load_scorecard(day, "forcing_diagnostic")
    if not isinstance(forcing, dict):
        return "missing"
    classification = forcing.get("process_classification")
    if not isinstance(classification, dict):
        return "missing"
    labels = classification.get("labels")
    return _list_summary(labels, limit=3)


def _seven_day_replay_rows(index: dict[str, object] | None) -> list[dict[str, object]]:
    indexed = _indexed_day_map(index)
    rows = []
    for day in LEEDS_REPLAY_DAYS:
        if not (OPERATIONAL_CAMPAIGN_ROOT / "days" / day).exists():
            continue
        instrument_rows = build_instrument_catalog([day])
        era5_cf = _row_for_instrument(instrument_rows, "Cloudnet CF", "ERA5")
        cm1_cf = _row_for_instrument(instrument_rows, "Cloudnet CF", "CM1 full LES")
        wband = _row_for_instrument(
            instrument_rows,
            "W-band radar",
            "CM1 virtual observatory",
        )
        indexed_day = indexed.get(day, {})
        compliance, compliance_detail = _day_compliance_state(day)
        rows.append(
            {
                "day": day,
                "gate": indexed_day.get("release_gate_status", "missing"),
                "bundle": indexed_day.get("lasso_bundle_status", "missing"),
                "qa": indexed_day.get("operational_qa_status", "missing"),
                "compliance": compliance,
                "compliance_detail": compliance_detail,
                "cm1_outputs": _cm1_output_count(day),
                "era5_cf_csi": era5_cf.get("csi", "n/a"),
                "cm1_cf_csi": cm1_cf.get("csi", "n/a"),
                "wband_csi": wband.get("csi", "n/a"),
                "lwc": _source_readiness_state(day, "l3-lwc"),
                "iwc": _source_readiness_state(day, "l3-iwc"),
                "surface": _scorecard_state(day, "surface_met"),
                "asfs_radiation": _scorecard_state(day, "asfs_logger_radiation_surface"),
                "asfs_sonic": _scorecard_state(day, "asfs_sonic_turbulence"),
                "asfs_gas": _scorecard_state(day, "asfs_gas"),
                "process": _replay_process_labels(day),
            }
        )
    return rows


def _seven_day_replay_summary(index: dict[str, object] | None) -> str:
    rows = _seven_day_replay_rows(index)
    if not rows:
        return "<div class='model-note'>No seven-day Leeds replay records found yet.</div>"
    ready_days = sum(
        1
        for row in rows
        if row.get("gate") == "full_virtual_observatory_ready"
        and row.get("bundle") == "ready"
        and row.get("qa") == "ready"
        and row.get("compliance") == "pass"
    )
    cards = [
        _card("Leeds replay days", len(rows)),
        _card("ready days", ready_days),
        _card("ERA5 CF CSI", _mean_compact([row.get("era5_cf_csi") for row in rows])),
        _card("CM1 CF CSI", _mean_compact([row.get("cm1_cf_csi") for row in rows])),
        _card("W-band CSI", _mean_compact([row.get("wband_csi") for row in rows])),
    ]
    body = []
    for row in rows:
        surface_text = (
            f"met:{row.get('surface')}, rad:{row.get('asfs_radiation')}, "
            f"sonic:{row.get('asfs_sonic')}, gas:{row.get('asfs_gas')}"
        )
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('day', '')))}</td>"
            f"<td>{_badge(row.get('gate'))}</td>"
            f"<td>{_badge(row.get('bundle'))}</td>"
            f"<td>{_badge(row.get('qa'))}</td>"
            f"<td>{escape(str(row.get('compliance', '')))} "
            f"({escape(str(row.get('compliance_detail', '')))})</td>"
            f"<td>{escape(str(row.get('cm1_outputs', '')))}</td>"
            f"<td>{escape(str(row.get('era5_cf_csi', 'n/a')))}</td>"
            f"<td>{escape(str(row.get('cm1_cf_csi', 'n/a')))}</td>"
            f"<td>{escape(str(row.get('wband_csi', 'n/a')))}</td>"
            f"<td>{escape(str(row.get('lwc', '')))} / {escape(str(row.get('iwc', '')))}</td>"
            f"<td>{escape(surface_text)}</td>"
            f"<td>{escape(str(row.get('process', '')))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-section-title'>7-Day Leeds Replay</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table seven-day-replay-table'>"
        "<thead><tr><th>day</th><th>gate</th><th>bundle</th><th>QA</th>"
        "<th>compliance f/w</th><th>CM1 outputs</th><th>ERA5 CF CSI</th>"
        "<th>CM1 CF CSI</th><th>W-band CSI</th><th>LWC/IWC</th>"
        "<th>surface and ASFS</th><th>process labels</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _next_day_after(day: str) -> str:
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError:
        return "not staged"
    return (parsed + timedelta(days=1)).isoformat()


def _future_staged_days(latest_day: str) -> list[str]:
    days_root = OPERATIONAL_CAMPAIGN_ROOT / "days"
    if not latest_day or not days_root.exists():
        return []
    return sorted(
        path.name
        for path in days_root.iterdir()
        if path.is_dir() and path.name[:4].isdigit() and path.name > latest_day
    )


def _operational_wait_state(index: dict[str, object] | None) -> str:
    days = []
    if isinstance(index, dict) and isinstance(index.get("days"), list):
        days = [
            str(day.get("day"))
            for day in index["days"]
            if isinstance(day, dict) and day.get("day")
        ]
    latest_day = max(days) if days else ""
    future_days = _future_staged_days(latest_day)
    ready_days = 0
    qa_rollup = index.get("operational_qa_rollup", {}) if isinstance(index, dict) else {}
    if isinstance(qa_rollup, dict):
        ready_days = int(qa_rollup.get("ready_day_count", 0) or 0)
    index_status = index.get("status", "missing") if isinstance(index, dict) else "missing"
    if latest_day and not future_days and index_status == "full_virtual_observatory_ready":
        mode = "waiting_for_new_data"
        detail = "runner should idle with no_ready_day until a future day is staged"
        next_day = _next_day_after(latest_day)
    elif future_days:
        mode = "future_inputs_staged"
        detail = "review staged future days before launching production CM1"
        next_day = future_days[0]
    else:
        mode = "campaign_state_unknown"
        detail = "campaign index or day records are missing"
        next_day = "unknown"
    cards = [
        _card("operating mode", mode),
        _card("regression baseline", f"{LEEDS_REPLAY_DAYS[0]} to {LEEDS_REPLAY_DAYS[-1]}"),
        _card("latest ready day", latest_day or "missing"),
        _card("QA ready days", ready_days),
        _card("next expected day", next_day),
    ]
    checklist = "".join(
        f"<li>{escape(item)}</li>" for item in NEXT_DATA_REQUIRED_INPUTS
    )
    future_text = ", ".join(future_days) if future_days else "none"
    return (
        "<div class='model-section-title'>Operational Wait State</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        "<div class='model-note'>"
        f"{escape(detail)}. Future staged days: {escape(future_text)}."
        "</div>"
        "<div class='model-two-column'>"
        "<div>"
        "<div class='model-subsection-title'>Resume Inputs</div>"
        f"<ul class='model-compact-list'>{checklist}</ul>"
        "</div>"
        "<div>"
        "<div class='model-subsection-title'>Allowed Work</div>"
        "<ul class='model-compact-list'>"
        "<li>keep runner and dashboard healthy</li>"
        "<li>use the seven-day replay as regression coverage</li>"
        "<li>improve W-band, Cloudnet, SEB and ASFS interpretation</li>"
        "<li>do not start new CM1 production runs until a day is planned-ready</li>"
        "</ul>"
        "</div>"
        "</div>"
    )


def _count_dict_text(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}:{count}" for key, count in sorted(value.items()))


def _list_summary(value: object, limit: int = 3) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    items = [str(item) for item in value[:limit]]
    if len(value) > limit:
        items.append(f"+{len(value) - limit} more")
    return ", ".join(items)


def _generic_contingency(scorecard: dict[str, object] | None, variable: str, key: str) -> dict[str, object]:
    if not scorecard:
        return {}
    comparisons = scorecard.get("comparisons")
    if not isinstance(comparisons, dict):
        return {}
    comparison = comparisons.get(variable)
    if not isinstance(comparison, dict):
        comparison = next((item for item in comparisons.values() if isinstance(item, dict)), {})
    occurrence = comparison.get(key) if isinstance(comparison, dict) else {}
    return occurrence if isinstance(occurrence, dict) else {}


def _lwp_policy_summary(scorecard: dict[str, object] | None) -> str:
    if not isinstance(scorecard, dict):
        return "missing"
    context = scorecard.get("lwp_context")
    if not isinstance(context, dict):
        return "missing"
    readiness = str(context.get("readiness_state", "unknown"))
    source_policy = str(context.get("source_policy", "unknown"))
    if source_policy and source_policy != "unknown":
        return f"{readiness}: {source_policy}"
    return readiness


def _campaign_days(limit: int = 7) -> list[str]:
    days_root = OPERATIONAL_CAMPAIGN_ROOT / "days"
    if not days_root.exists():
        return []
    days = sorted(
        path.name for path in days_root.iterdir() if path.is_dir() and path.name[:4].isdigit()
    )
    return list(reversed(days))[:limit]


def _comparison_payload(
    scorecard: dict[str, object] | None,
    spec: dict[str, object],
) -> dict[str, object]:
    if not isinstance(scorecard, dict):
        return {}
    comparison_name = spec.get("comparison")
    metrics_group = spec.get("metrics_group")
    if isinstance(metrics_group, str):
        group = scorecard.get(metrics_group)
        if isinstance(group, dict) and comparison_name in group:
            value = group.get(comparison_name)
            return value if isinstance(value, dict) else {}
        return {}
    comparisons = scorecard.get("comparisons")
    if isinstance(comparisons, dict) and isinstance(comparison_name, str):
        value = comparisons.get(comparison_name)
        return value if isinstance(value, dict) else {}
    return scorecard


def _metric_from(metric_block: object, names: tuple[str, ...]) -> object:
    if not isinstance(metric_block, dict):
        return "n/a"
    for name in names:
        value = metric_block.get(name)
        if value is not None:
            return _compact_float(value)
    return "n/a"


def _scorecard_caveat(scorecard: dict[str, object] | None, spec: dict[str, object]) -> str:
    if not isinstance(scorecard, dict):
        return "blocked_missing_input"
    if scorecard.get("excluded_from_scoring"):
        status = str(scorecard.get("scoring_status", "")).lower()
        if "non_colocated" in status or "not colocated" in status:
            return "not_colocated"
        return "diagnostic_only"
    payload_caveat = str(spec.get("caveat", "ready"))
    if payload_caveat != "ready":
        return payload_caveat
    context = scorecard.get("lwp_context")
    if isinstance(context, dict):
        readiness = str(context.get("readiness_state", "ready"))
        if readiness.startswith("blocked"):
            return readiness
    return "ready"


def _badge(value: object) -> str:
    text = str(value or "unknown")
    css = "badge-ready"
    lower = text.lower()
    if "diagnostic" in lower or "excluded" in lower:
        css = "badge-diagnostic"
    elif "blocked" in lower or "missing" in lower:
        css = "badge-blocked"
    elif "colocated" in lower:
        css = "badge-warning"
    return f"<span class='status-badge {css}'>{escape(text)}</span>"


def _instrument_comparison_row(day: str, spec: dict[str, object]) -> dict[str, object]:
    scorecard_name = str(spec["scorecard"])
    scorecard = load_scorecard(day, scorecard_name)
    payload = _comparison_payload(scorecard, spec)
    occurrence_key = spec.get("occurrence")
    occurrence = {}
    if isinstance(occurrence_key, str):
        if occurrence_key == "contingency" and isinstance(scorecard, dict):
            occurrence = scorecard.get("contingency", {})
        if not isinstance(occurrence, dict) or not occurrence:
            occurrence = payload.get(occurrence_key, {}) if isinstance(payload, dict) else {}
        if occurrence_key == "contingency" and isinstance(payload, dict) and not occurrence:
            occurrence = payload.get("contingency", {})
    occurrence = occurrence if isinstance(occurrence, dict) else {}

    metrics_key = spec.get("metrics")
    metrics = {}
    if isinstance(metrics_key, str):
        metrics = payload.get(metrics_key, {}) if isinstance(payload, dict) else {}
        if not isinstance(metrics, dict) and isinstance(scorecard, dict):
            metrics = scorecard.get(metrics_key, {})
    metrics = metrics if isinstance(metrics, dict) else {}

    status = "missing"
    if isinstance(scorecard, dict):
        status = str(
            scorecard.get("scoring_status")
            or scorecard.get("status")
            or payload.get("status", "available")
        )
        if scorecard.get("excluded_from_scoring"):
            status = str(scorecard.get("scoring_status", "diagnostic_only"))
    valid = occurrence.get("valid_points")
    if valid is None:
        valid = metrics.get("valid_points", metrics.get("valid_times", "n/a"))

    note = ""
    if isinstance(scorecard, dict):
        if scorecard.get("excluded_from_scoring"):
            note = "excluded from ranking"
        elif payload.get("production_ready") is False:
            note = str(payload.get("readiness_note", "diagnostic"))
        elif scorecard_name.endswith("_lwc"):
            note = _lwp_policy_summary(scorecard)
    base_top = payload.get("cloud_base_top") if isinstance(payload, dict) else None
    if not isinstance(base_top, dict) and isinstance(scorecard, dict):
        base_top = scorecard.get("cloud_base_top")
    base_top = base_top if isinstance(base_top, dict) else {}
    runtime = _bundle_runtime_summary(day)

    return {
        "day": day,
        "instrument": spec.get("instrument", ""),
        "model": spec.get("model", ""),
        "model_group": spec.get("model_group", "all"),
        "metric_family": spec.get("metric_family", "readiness"),
        "basis": spec.get("basis", ""),
        "scorecard": scorecard_name,
        "cm1_runtime_h": runtime["run_hours"],
        "cm1_eval_h": runtime["evaluation_hours"],
        "cm1_recipe_class": runtime["recipe_class"],
        "status": status,
        "caveat": _scorecard_caveat(scorecard, spec),
        "valid": valid,
        "pod": _metric_from(occurrence, ("probability_of_detection",)),
        "far": _metric_from(occurrence, ("false_alarm_ratio",)),
        "csi": _metric_from(occurrence, ("critical_success_index",)),
        "bias": _metric_from(
            metrics,
            (
                "bias_mean",
                "mean_bias_db",
                "lwp_bias_mean_kg_m2",
                "iwp_bias_mean_kg_m2",
            ),
        ),
        "rmse": _metric_from(
            metrics,
            (
                "root_mean_square_error",
                "root_mean_square_error_db",
                "lwp_root_mean_square_error_kg_m2",
                "iwp_root_mean_square_error_kg_m2",
            ),
        ),
        "correlation": _metric_from(
            metrics,
            ("pearson_correlation", "lwp_pearson_correlation", "iwp_pearson_correlation"),
        ),
        "base_bias_m": _metric_from(base_top, ("cloud_base_bias_mean_m", "model_cloud_base_bias_mean_m")),
        "top_bias_m": _metric_from(base_top, ("cloud_top_bias_mean_m", "model_cloud_top_bias_mean_m")),
        "note": note,
    }


def _instrument_comparison_rows(days: list[str]) -> list[dict[str, object]]:
    return [
        _instrument_comparison_row(day, spec)
        for day in days
        for spec in INSTRUMENT_COMPARISON_SPECS
    ]


def build_instrument_catalog(days: list[str] | None = None) -> list[dict[str, object]]:
    selected_days = days if days is not None else _campaign_days()
    return _instrument_comparison_rows(selected_days)


def _filter_instrument_rows(
    rows: list[dict[str, object]],
    instrument: str = "all",
    model_group: str = "all",
    metric_family: str = "all",
) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        if instrument != "all" and row.get("instrument") != instrument:
            continue
        if model_group != "all" and row.get("model_group") != model_group:
            continue
        if metric_family != "all" and row.get("metric_family") != metric_family:
            continue
        filtered.append(row)
    return filtered


def build_instrument_detail(
    day: str,
    instrument: str = "all",
    model: str = "all",
    metric_family: str = "all",
) -> dict[str, object]:
    rows = _filter_instrument_rows(
        build_instrument_catalog([day]),
        instrument=instrument,
        model_group=model,
        metric_family=metric_family,
    )
    caveat_counts: dict[str, int] = {}
    for row in rows:
        caveat = str(row.get("caveat", "unknown"))
        caveat_counts[caveat] = caveat_counts.get(caveat, 0) + 1
    return {
        "day": day,
        "instrument": instrument,
        "model": model,
        "metric_family": metric_family,
        "rows": rows,
        "caveat_counts": caveat_counts,
    }


def _instrument_options() -> OrderedDict[str, str]:
    instruments = sorted({str(spec["instrument"]) for spec in INSTRUMENT_COMPARISON_SPECS})
    return OrderedDict([("All instruments", "all"), *[(name, name) for name in instruments]])


def _day_options() -> OrderedDict[str, str]:
    days = _campaign_days(limit=60)
    if not days:
        return OrderedDict([("No campaign days found", "")])
    return OrderedDict((day, day) for day in days)


def _instrument_metric_cards(rows: list[dict[str, object]]) -> str:
    if not rows:
        cards = [_card("selected products", 0), _card("status", "missing")]
        return f"<div class='model-grid'>{''.join(cards)}</div>"
    ready = sum(1 for row in rows if row.get("caveat") == "ready")
    diagnostic = sum(1 for row in rows if row.get("caveat") in {"diagnostic_only", "not_colocated"})
    blocked = sum(1 for row in rows if str(row.get("caveat", "")).startswith("blocked"))
    csi_values = [
        float(row["csi"])
        for row in rows
        if isinstance(row.get("csi"), str) and _is_float_string(str(row.get("csi")))
    ]
    csi_mean = sum(csi_values) / len(csi_values) if csi_values else None
    cards = [
        _card("selected products", len(rows)),
        _card("ready", ready),
        _card("diagnostic", diagnostic),
        _card("blocked", blocked),
        _card("mean CSI", _compact_float(csi_mean) if csi_mean is not None else "n/a"),
    ]
    return f"<div class='model-grid'>{''.join(cards)}</div>"


def _is_float_string(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _instrument_comparison_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p>No instrument comparison records found yet.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row['day']))}</td>"
            f"<td>{escape(str(row['instrument']))}</td>"
            f"<td>{escape(str(row['model']))}</td>"
            f"<td>{_badge(row['caveat'])}</td>"
            f"<td>{escape(str(row['cm1_runtime_h']))}</td>"
            f"<td>{escape(str(row['cm1_eval_h']))}</td>"
            f"<td>{escape(str(row['cm1_recipe_class']))}</td>"
            f"<td>{escape(str(row['status']))}</td>"
            f"<td>{escape(str(row['basis']))}</td>"
            f"<td>{escape(str(row['valid']))}</td>"
            f"<td>{escape(str(row['pod']))}</td>"
            f"<td>{escape(str(row['far']))}</td>"
            f"<td>{escape(str(row['csi']))}</td>"
            f"<td>{escape(str(row['bias']))}</td>"
            f"<td>{escape(str(row['rmse']))}</td>"
            f"<td>{escape(str(row['correlation']))}</td>"
            f"<td>{escape(str(row['base_bias_m']))}</td>"
            f"<td>{escape(str(row['top_bias_m']))}</td>"
            f"<td><code>{escape(str(row['scorecard']))}</code></td>"
            f"<td>{escape(str(row['note']))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table instrument-comparison-table'>"
        "<thead><tr><th>day</th><th>instrument</th><th>model/output</th>"
        "<th>caveat</th><th>CM1 h</th><th>eval h</th><th>recipe</th>"
        "<th>status</th><th>comparison</th><th>valid</th><th>POD</th><th>FAR</th>"
        "<th>CSI</th><th>bias</th><th>RMSE</th><th>corr</th><th>base bias m</th>"
        "<th>top bias m</th><th>scorecard</th><th>note</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _scorecard_png_path(day: str, scorecard_name: str) -> Path | None:
    scorecard = load_scorecard(day, scorecard_name)
    if not isinstance(scorecard, dict):
        return None
    value = scorecard.get("output_png")
    if isinstance(value, str) and value:
        path = Path(value)
        if path.exists():
            return path
    fallback = OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "scorecards" / f"{scorecard_name}.png"
    return fallback if fallback.exists() else None


def render_scorecard_gallery(day: str | None, instrument: str = "all") -> str:
    if not day:
        return ""
    cards = []
    if instrument == "all":
        gallery_items = [
            item
            for items in INSTRUMENT_GALLERY_SCORECARDS.values()
            for item in items
        ]
    else:
        gallery_items = list(INSTRUMENT_GALLERY_SCORECARDS.get(instrument, ()))
    for title, scorecard_name in gallery_items:
        png_path = _scorecard_png_path(day, scorecard_name)
        if png_path is None:
            continue
        data_uri = _asset_data_uri(png_path)
        if not data_uri:
            continue
        cards.append(
            "<div class='instrument-plot-card'>"
            f"<div class='instrument-plot-title'>{escape(title)}</div>"
            f"<img class='instrument-plot-image' src='{data_uri}' alt='{escape(title)}'>"
            "</div>"
        )
    if not cards:
        return "<p>No scorecard plots found for the latest day.</p>"
    return "<div class='instrument-plot-grid'>" + "".join(cards) + "</div>"


def _instrument_comparison_panel(
    day: str,
    instrument: str,
    model_group: str,
    metric_family: str,
    _clicks: int = 0,
) -> pn.Column:
    detail = build_instrument_detail(day, instrument, model_group, metric_family)
    rows = detail["rows"] if isinstance(detail.get("rows"), list) else []
    caveats = detail.get("caveat_counts")
    caveat_text = _count_dict_text(caveats)
    html = (
        "<div class='model-shell operational-shell'>"
        "<div class='model-headline'>"
        "<div>"
        "<div class='model-title'>Instrument Comparisons</div>"
        "<div class='model-subtitle'>Model and synthetic outputs compared with active observation products</div>"
        "</div>"
        f"<div class='model-pill'>{escape(day or 'no day selected')}</div>"
        "</div>"
        f"{_instrument_metric_cards(rows)}"
        f"<div class='model-subtitle'>caveats: {escape(caveat_text)}</div>"
        f"{render_scorecard_gallery(day, instrument)}"
        f"{_instrument_comparison_table(rows)}"
        "</div>"
    )
    return pn.Column(pn.pane.HTML(html, sizing_mode="stretch_width"), sizing_mode="stretch_width")


def _operational_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = _read_json(path)
        if not payload:
            continue
        day = str(payload.get("day") or path.parent.name)
        summary = payload.get("summary")
        summary = summary if isinstance(summary, dict) else None
        gate = summary.get("release_gate") if isinstance(summary, dict) else {}
        gate = gate if isinstance(gate, dict) else {}
        wband = _summary_scorecard(summary, "wband_radar")
        wband_contingency = wband.get("contingency") if isinstance(wband, dict) else {}
        wband_contingency = wband_contingency if isinstance(wband_contingency, dict) else {}
        era5_iwc = _generic_contingency(_direct_scorecard(day, "era5_iwc"), "iwc", "ice_occurrence")
        era5_lwc_scorecard = _direct_scorecard(day, "era5_lwc")
        era5_lwc = _generic_contingency(era5_lwc_scorecard, "lwc", "liquid_occurrence")
        process_labels = summary.get("process_labels") if isinstance(summary, dict) else []
        if not isinstance(process_labels, list):
            process_labels = []
        scheduler_policy = summary.get("scheduler_policy") if isinstance(summary, dict) else {}
        scheduler_policy = scheduler_policy if isinstance(scheduler_policy, dict) else {}
        operational_qa = summary.get("operational_qa") if isinstance(summary, dict) else {}
        operational_qa = operational_qa if isinstance(operational_qa, dict) else {}
        rows.append(
            {
                "day": day,
                "run_status": payload.get("status", "n/a"),
                "summary_status": summary.get("status", "missing") if summary else "missing",
                "gate": gate.get("status", "n/a"),
                "scheduler": scheduler_policy.get("status", "missing"),
                "scheduler_priority": scheduler_policy.get("priority", "n/a"),
                "scheduler_actions": _scheduler_action_summary(scheduler_policy),
                "operational_qa": operational_qa.get("status", "missing"),
                "operational_qa_ready": operational_qa.get("status") in {
                    "ready",
                    "no_targeted_scheduler_actions",
                },
                "operational_qa_missing": _list_summary(
                    operational_qa.get("missing_required_scorecards")
                ),
                "era5_cf_csi": _cf_csi(summary, "era5_cloud_fraction"),
                "cm1_cf_csi": _cm1_cf_csi(day, summary),
                "wband_csi": _compact_float(wband_contingency.get("critical_success_index")),
                "iwc_points": era5_iwc.get("valid_points", "n/a"),
                "iwc_csi": _compact_float(era5_iwc.get("critical_success_index")),
                "lwc_points": era5_lwc.get("valid_points", "n/a"),
                "lwp_policy": _lwp_policy_summary(era5_lwc_scorecard),
                "labels": ", ".join(str(item) for item in process_labels[:4]),
            }
        )
    return rows


def _operational_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row['day']))}</td>"
            f"<td>{escape(str(row['run_status']))}</td>"
            f"<td>{escape(str(row['summary_status']))}</td>"
            f"<td>{escape(str(row['gate']))}</td>"
            f"<td>{escape(str(row['scheduler']))}</td>"
            f"<td>{escape(str(row['scheduler_priority']))}</td>"
            f"<td>{escape(str(row['operational_qa']))}</td>"
            f"<td>{escape(str(row['operational_qa_missing']))}</td>"
            f"<td>{escape(str(row['era5_cf_csi']))}</td>"
            f"<td>{escape(str(row['cm1_cf_csi']))}</td>"
            f"<td>{escape(str(row['wband_csi']))}</td>"
            f"<td>{escape(str(row['iwc_points']))}</td>"
            f"<td>{escape(str(row['iwc_csi']))}</td>"
            f"<td>{escape(str(row['lwc_points']))}</td>"
            f"<td>{escape(str(row['lwp_policy']))}</td>"
            f"<td>{escape(str(row['scheduler_actions']))}</td>"
            f"<td>{escape(str(row['labels']))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table'>"
        "<thead><tr>"
        "<th>day</th><th>run</th><th>summary</th><th>gate</th>"
        "<th>scheduler</th><th>priority</th><th>QA</th><th>missing QA</th>"
        "<th>ERA5 CF CSI</th><th>CM1 LES CF CSI</th><th>W-band CSI</th>"
        "<th>IWC gates</th><th>IWC CSI</th><th>LWC gates</th><th>LWP policy</th>"
        "<th>actions</th><th>labels</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _scorecard_metric(comparison: dict[str, object], name: str) -> object:
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict):
        return "n/a"
    return _compact_float(metrics.get(name, "n/a"))


def _operator_policy_table(summary: dict[str, object] | None) -> str:
    if not isinstance(summary, dict):
        return ""
    policies = summary.get("operator_policies")
    if not isinstance(policies, dict):
        return ""
    body = []
    for name in ("gas", "turbulence", "radiation"):
        policy = policies.get(name)
        if not isinstance(policy, dict):
            continue
        blockers = policy.get("blockers")
        blocker_text = (
            ", ".join(str(item) for item in blockers)
            if isinstance(blockers, list) and blockers
            else "-"
        )
        detail = []
        if "observation_processing_ready" in policy:
            detail.append(
                "obs processing: "
                + str(policy.get("observation_processing_ready", "n/a"))
            )
        if "turbulence_observation_processing_ready" in policy:
            detail.append(
                "turb obs: "
                + str(policy.get("turbulence_observation_processing_ready", "n/a"))
            )
        if "full_operator_ready" in policy:
            detail.append("full operator: " + str(policy.get("full_operator_ready", "n/a")))
        if "comparison_count" in policy:
            detail.append("comparisons: " + str(policy.get("comparison_count", "n/a")))
        body.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(str(policy.get('status', 'unknown')))}</td>"
            f"<td>{escape(str(policy.get('ready', False)))}</td>"
            f"<td>{escape(str(policy.get('policy', '')))}</td>"
            f"<td>{escape('; '.join(detail) or '-')}</td>"
            f"<td>{escape(blocker_text)}</td>"
            "</tr>"
        )
    if not body:
        return ""
    return (
        "<div class='model-section-title'>Operator Policy Gate</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table asfs-detail-table'>"
        "<thead><tr>"
        "<th>operator</th><th>status</th><th>ready</th><th>policy</th>"
        "<th>detail</th><th>blockers</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _asfs_gas_table(summary: dict[str, object] | None) -> str:
    scorecard = _summary_scorecard(summary, "asfs_gas")
    if not scorecard:
        return ""
    comparisons = scorecard.get("comparisons")
    if not isinstance(comparisons, dict):
        return ""
    blockers = scorecard.get("gas_operator_blockers")
    blocker_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else "n/a"
    body = []
    for name in ("h2o_mole_fraction", "co2_molar_density", "licor_cell_pressure", "licor_cell_temperature"):
        comparison = comparisons.get(name)
        if not isinstance(comparison, dict):
            continue
        body.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(str(comparison.get('status', 'n/a')))}</td>"
            f"<td>{escape(str(comparison.get('production_ready', 'n/a')))}</td>"
            f"<td>{escape(str(comparison.get('model_science_policy', '')))}</td>"
            f"<td>{escape(str(comparison.get('model_variable', 'missing')))}</td>"
            f"<td>{escape(str(comparison.get('observed_variable', 'n/a')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'valid_times')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'model_mean')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'observed_mean')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'bias_mean')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'root_mean_square_error')))}</td>"
            f"<td>{escape(str(comparison.get('target_units', '')))}</td>"
            f"<td>{escape(str(comparison.get('reason', '')))}</td>"
            "</tr>"
        )
    if not body:
        return ""
    return (
        "<div class='model-section-title'>ASFS Gas Readiness</div>"
        "<div class='model-subtitle'>"
        f"gas operator ready: {escape(str(scorecard.get('gas_operator_ready', 'n/a')))}; "
        f"status: {escape(str(scorecard.get('scoring_status', 'n/a')))}"
        "</div>"
        "<div class='model-subtitle'>"
        f"blockers: {escape(blocker_text)}"
        "</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table asfs-detail-table'>"
        "<thead><tr>"
        "<th>comparison</th><th>status</th><th>production</th><th>policy</th><th>model var</th><th>obs var</th>"
        "<th>valid</th><th>model mean</th><th>obs mean</th><th>bias</th><th>RMSE</th>"
        "<th>units</th><th>reason</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _asfs_sonic_summary(summary: dict[str, object] | None) -> str:
    scorecard = _summary_scorecard(summary, "asfs_sonic_turbulence")
    if not scorecard:
        return ""
    availability = scorecard.get("input_variable_availability")
    availability = availability if isinstance(availability, dict) else {}
    blockers = scorecard.get("readiness_blockers")
    blocker_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else "n/a"
    availability_rows = []
    for key in (
        "raw_sonic_wind_available",
        "raw_sonic_wind_variables",
        "earth_aligned_wind_available",
        "earth_aligned_wind_variables",
        "quality_flag_variables",
        "inclinometer_variables",
    ):
        value = availability.get(key, "n/a")
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value) if value else "none"
        availability_rows.append(
            "<tr>"
            f"<th>{escape(key)}</th>"
            f"<td>{escape(str(value))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-section-title'>ASFS Sonic Readiness</div>"
        "<div class='model-subtitle'>"
        f"turbulence ready: {escape(str(scorecard.get('turbulence_operator_ready', 'n/a')))}; "
        f"observation processing ready: {escape(str(scorecard.get('observation_processing_ready', 'n/a')))}; "
        f"rotation: {escape(str(scorecard.get('rotation_policy', 'n/a')))}"
        "</div>"
        "<div class='model-subtitle'>"
        f"blockers: {escape(blocker_text)}"
        "</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table asfs-detail-table'>"
        f"<tbody>{''.join(availability_rows)}</tbody>"
        "</table></div>"
        f"{_asfs_sonic_mean_table(scorecard)}"
        f"{_asfs_sonic_turbulence_table(scorecard)}"
    )


def _asfs_sonic_mean_table(scorecard: dict[str, object]) -> str:
    comparisons = scorecard.get("mean_comparisons")
    if not isinstance(comparisons, dict):
        return ""
    body = []
    for name, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            continue
        body.append(
            "<tr>"
            f"<td>{escape(str(name))}</td>"
            f"<td>{escape(str(comparison.get('status', 'n/a')))}</td>"
            f"<td>{escape(str(comparison.get('model_variable', 'missing')))}</td>"
            f"<td>{escape(str(comparison.get('observed_variable', 'n/a')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'valid_times')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'bias_mean')))}</td>"
            f"<td>{escape(str(_scorecard_metric(comparison, 'root_mean_square_error')))}</td>"
            f"<td>{escape(str(comparison.get('target_units', '')))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table asfs-detail-table'>"
        "<thead><tr>"
        "<th>sonic mean</th><th>status</th><th>model var</th><th>obs var</th>"
        "<th>valid</th><th>bias</th><th>RMSE</th><th>units</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _asfs_sonic_turbulence_table(scorecard: dict[str, object]) -> str:
    turbulence = scorecard.get("turbulence_summary")
    if not isinstance(turbulence, dict):
        return ""
    names = (
        "observed_tke_proxy",
        "model_surface_tke",
        "model_surface_wind_variance",
        "model_surface_temperature_variance",
        "tke_proxy_comparison",
    )
    body = []
    for name in names:
        item = turbulence.get(name)
        if not isinstance(item, dict):
            continue
        mean = (
            item.get("window_tke_proxy_mean")
            if "window_tke_proxy_mean" in item
            else item.get("value_mean")
        )
        if mean is None and isinstance(item.get("metrics"), dict):
            mean = item["metrics"].get("model_mean")
        body.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(str(item.get('status', 'n/a')))}</td>"
            f"<td>{escape(str(item.get('science_policy', '')))}</td>"
            f"<td>{escape(str(item.get('model_variable', item.get('observed_variable', 'n/a'))))}</td>"
            f"<td>{escape(str(_compact_float(item.get('valid_times', item.get('valid_windows', 'n/a')))))}</td>"
            f"<td>{escape(str(_compact_float(mean)))}</td>"
            f"<td>{escape(str(item.get('target_units', '')))}</td>"
            "</tr>"
        )
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table asfs-detail-table'>"
        "<thead><tr>"
        "<th>turbulence item</th><th>status</th><th>policy</th><th>variable</th>"
        "<th>valid</th><th>mean</th><th>units</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _campaign_index_table(index: dict[str, object] | None) -> str:
    if not index:
        return ""
    days = index.get("days")
    if not isinstance(days, list) or not days:
        return ""
    body = []
    for day in days:
        if not isinstance(day, dict):
            continue
        missing = day.get("missing_required_components")
        if isinstance(missing, list) and missing:
            missing_text = ", ".join(str(item) for item in missing[:5])
        else:
            missing_text = "none"
        body.append(
            "<tr>"
            f"<td>{escape(str(day.get('day', '')))}</td>"
            f"<td>{escape(str(day.get('summary_status', '')))}</td>"
            f"<td>{escape(str(day.get('release_gate_status', '')))}</td>"
            f"<td>{escape(str(_compact_float(day.get('common_overlap_hours'))))}</td>"
            f"<td>{escape(missing_text)}</td>"
            "</tr>"
        )
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table'>"
        "<thead><tr>"
        "<th>day</th><th>summary</th><th>release gate</th><th>overlap h</th><th>missing required</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _process_skill_rollup_table(index: dict[str, object] | None, limit: int = 16) -> str:
    rollup = _process_skill_rollup(index)
    if not rollup:
        return ""
    rows = []
    for label, item in sorted(
        rollup.items(),
        key=lambda pair: (
            -int(pair[1].get("day_count", 0)) if isinstance(pair[1], dict) else 0,
            str(pair[0]),
        ),
    )[:limit]:
        if not isinstance(item, dict):
            continue
        scorecards = item.get("scorecards")
        scorecards = scorecards if isinstance(scorecards, dict) else {}
        era5 = _metric_block(scorecards, "era5_cloud_fraction", "cf_V")
        cm1 = _metric_block(scorecards, "cloud_fraction", "cf_V")
        cloudnet = _metric_block(scorecards, "cloudnet_cloud_fraction", "cf_V")
        rows.append(
            "<tr>"
            f"<td>{escape(str(label))}</td>"
            f"<td>{escape(str(item.get('day_count', 0)))}</td>"
            f"<td>{escape(str(item.get('full_virtual_observatory_ready_day_count', 0)))}</td>"
            f"<td>{escape(str(era5.get('csi', 'n/a')))}</td>"
            f"<td>{escape(str(era5.get('pod', 'n/a')))}</td>"
            f"<td>{escape(str(era5.get('far', 'n/a')))}</td>"
            f"<td>{escape(str(cm1.get('csi', 'n/a')))}</td>"
            f"<td>{escape(str(cloudnet.get('csi', 'n/a')))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table process-skill-table'>"
        "<thead><tr>"
        "<th>process label</th><th>days</th><th>full VO days</th>"
        "<th>ERA5 CSI</th><th>ERA5 POD</th><th>ERA5 FAR</th>"
        "<th>CM1 LES CSI</th><th>Cloudnet CSI</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _process_evidence_table(
    index: dict[str, object] | None,
    label_limit: int = 10,
    day_limit_per_label: int = 6,
) -> str:
    rollup = _process_skill_rollup(index)
    if not rollup:
        return ""
    day_rows = _index_day_rows(index)
    body = []
    for label, item in sorted(
        rollup.items(),
        key=lambda pair: (
            -int(pair[1].get("day_count", 0)) if isinstance(pair[1], dict) else 0,
            str(pair[0]),
        ),
    )[:label_limit]:
        if not isinstance(item, dict):
            continue
        days = item.get("days")
        if not isinstance(days, list):
            continue
        for day in days[:day_limit_per_label]:
            day_text = str(day)
            row = day_rows.get(day_text, {})
            summary = _operational_summary_for_day(day_text)
            body.append(
                "<tr>"
                f"<td>{escape(str(label))}</td>"
                f"<td>{escape(day_text)}</td>"
                f"<td>{escape(str(row.get('release_gate_status', 'n/a')))}</td>"
                f"<td>{escape(str(_cf_csi(summary, 'era5_cloud_fraction')))}</td>"
                f"<td>{escape(str(_cm1_cf_csi(day_text, summary)))}</td>"
                f"<td><code>{escape(_evidence_bundle_path(row, day_text))}</code></td>"
                f"<td><code>{escape(_evidence_summary_path(row, day_text))}</code></td>"
                f"<td><code>{escape(_evidence_scorecards_path(day_text))}</code></td>"
                "</tr>"
            )
    if not body:
        return ""
    return (
        "<div class='model-section-title'>Process-Regime Evidence Drill-Down</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table process-evidence-table'>"
        "<thead><tr>"
        "<th>process label</th><th>day</th><th>gate</th><th>ERA5 CSI</th><th>CM1 LES CSI</th>"
        "<th>bundle</th><th>summary</th><th>scorecards</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _process_diagnosis_table(diagnosis: dict[str, object] | None, limit: int = 16) -> str:
    processes = _process_diagnoses(diagnosis)
    if not processes:
        return ""
    body = []
    for label, process in sorted(
        processes.items(),
        key=lambda pair: (
            -int(pair[1].get("day_count", 0)) if isinstance(pair[1], dict) else 0,
            str(pair[0]),
        ),
    )[:limit]:
        if not isinstance(process, dict):
            continue
        scorecards = process.get("scorecards")
        scorecards = scorecards if isinstance(scorecards, dict) else {}
        era5 = _diagnosis_comparison(scorecards, "era5_cloud_fraction", "cf_V")
        labels = era5.get("diagnosis_labels", [])
        label_text = ", ".join(str(item) for item in labels) if isinstance(labels, list) else "-"
        body.append(
            "<tr>"
            f"<td>{escape(str(label))}</td>"
            f"<td>{escape(str(process.get('day_count', 0)))}</td>"
            f"<td>{escape(str(_compact_float(era5.get('critical_success_index_mean'))))}</td>"
            f"<td>{escape(str(_compact_float(era5.get('false_alarm_ratio_mean'))))}</td>"
            f"<td>{escape(str(_compact_float(era5.get('cloud_base_bias_mean_m'))))}</td>"
            f"<td>{escape(str(_compact_float(era5.get('cloud_top_bias_mean_m'))))}</td>"
            f"<td>{escape(label_text)}</td>"
            f"<td>{escape(str(era5.get('interpretation', 'n/a')))}</td>"
            "</tr>"
        )
    if not body:
        return ""
    return (
        "<div class='model-section-title'>Process Diagnosis</div>"
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table process-diagnosis-table'>"
        "<thead><tr>"
        "<th>process label</th><th>days</th><th>ERA5 CSI</th><th>ERA5 FAR</th>"
        "<th>base bias m</th><th>top bias m</th><th>dominant labels</th><th>interpretation</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _diagnosis_comparison(
    scorecards: dict[str, object],
    scorecard_name: str,
    observed_variable: str,
) -> dict[str, object]:
    scorecard = scorecards.get(scorecard_name)
    if not isinstance(scorecard, dict):
        return {}
    comparison = scorecard.get(observed_variable)
    return comparison if isinstance(comparison, dict) else {}


def _index_day_rows(index: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if not isinstance(index, dict):
        return {}
    days = index.get("days")
    if not isinstance(days, list):
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row in days:
        if not isinstance(row, dict):
            continue
        day = row.get("day")
        if day is not None:
            rows[str(day)] = row
    return rows


def _evidence_bundle_path(row: dict[str, object], day: str) -> str:
    value = row.get("lasso_bundle_json")
    if value:
        return str(value)
    return str(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "lasso_bundle" / "bundle.json")


def _evidence_summary_path(row: dict[str, object], day: str) -> str:
    value = row.get("output_json")
    if value:
        return str(value)
    return str(OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "scorecards" / "operational_summary.json")


def _evidence_scorecards_path(day: str) -> str:
    scorecard_root = OPERATIONAL_CAMPAIGN_ROOT / "days" / day / "scorecards"
    key_files = (
        scorecard_root / "forcing_diagnostic.json",
        scorecard_root / "era5_cloud_fraction.json",
        scorecard_root / "cloud_fraction.json",
    )
    return " | ".join(str(path) for path in key_files)


def _metric_block(
    scorecards: dict[str, object],
    scorecard_name: str,
    observed_variable: str,
) -> dict[str, object]:
    scorecard = scorecards.get(scorecard_name)
    if not isinstance(scorecard, dict):
        return {}
    metrics = scorecard.get("cloud_fraction_metrics")
    if not isinstance(metrics, dict):
        return {}
    variable = metrics.get(observed_variable)
    if not isinstance(variable, dict):
        return {}
    return {
        "csi": _compact_float(variable.get("critical_success_index_mean")),
        "pod": _compact_float(variable.get("probability_of_detection_mean")),
        "far": _compact_float(variable.get("false_alarm_ratio_mean")),
    }


def _evaluation_schematic() -> str:
    return """
    <div class="schematic">
      <div class="schematic-col">
        <div class="schematic-title">Observations</div>
        <div class="schematic-box">Cloud radar / W-band</div>
        <div class="schematic-box">CL61 lidar</div>
        <div class="schematic-box">HATPRO LWP</div>
        <div class="schematic-box">ASFS met, radiation, sonic, gas</div>
        <div class="schematic-box">Cloudnet source products</div>
      </div>
      <div class="schematic-arrow">-></div>
      <div class="schematic-col">
        <div class="schematic-title">Model Inputs</div>
        <div class="schematic-box">ERA5</div>
        <div class="schematic-box">CM1 / full LES daily recipe</div>
        <div class="schematic-box">CM1 virtual-observatory products</div>
      </div>
      <div class="schematic-arrow">-></div>
      <div class="schematic-col">
        <div class="schematic-title">Forward Operators</div>
        <div class="schematic-box">Cloudnet CF / LWC / IWC</div>
        <div class="schematic-box">PAMTRA / W-band radar</div>
        <div class="schematic-box">CL61 diagnostic lidar</div>
        <div class="schematic-box">RRTMGP / SEB</div>
        <div class="schematic-box">ASFS surface diagnostics</div>
      </div>
      <div class="schematic-arrow">-></div>
      <div class="schematic-col">
        <div class="schematic-title">Review Outputs</div>
        <div class="schematic-box">Scorecards</div>
        <div class="schematic-box">AURORA-LASSO bundle</div>
        <div class="schematic-box">Dashboard</div>
      </div>
    </div>
    """


def _overview_panel(_clicks: int = 0) -> pn.Column:
    index = load_campaign_index()
    days = _campaign_days()
    latest_day = days[0] if days else ""
    latest_runtime = _bundle_runtime_summary(latest_day) if latest_day else {}
    operational_qa = _operational_qa_rollup(index)
    archive_manifest = _campaign_archive_manifest()
    rows = build_instrument_catalog([latest_day]) if latest_day else []
    ready = sum(1 for row in rows if row.get("caveat") == "ready")
    diagnostic = sum(1 for row in rows if row.get("caveat") in {"diagnostic_only", "not_colocated"})
    blocked = sum(1 for row in rows if str(row.get("caveat", "")).startswith("blocked"))
    cards = [
        _card("latest day", latest_day or "missing"),
        _card("campaign", index.get("status", "missing") if isinstance(index, dict) else "missing"),
        _card("QA ready days", operational_qa.get("ready_day_count", "n/a")),
        _card("QA incomplete", operational_qa.get("qa_incomplete_day_count", "n/a")),
        _card("CM1 run h", latest_runtime.get("run_hours", "n/a")),
        _card("CM1 eval h", latest_runtime.get("evaluation_hours", "n/a")),
        _card("ready products", ready),
        _card("diagnostic products", diagnostic),
        _card("blocked products", blocked),
        _card("ERA5 CF CSI", _index_cf_metric(index, "era5_cloud_fraction", "cf_V")),
        _card("CM1 LES CF CSI", _index_cf_metric(index, "cloud_fraction", "cf_V")),
        *_archive_manifest_cards(archive_manifest),
    ]
    html = (
        "<div class='model-shell operational-shell'>"
        "<div class='model-headline'>"
        "<div>"
        "<div class='model-title'>AURORA-LASSO Evaluation Overview</div>"
        "<div class='model-subtitle'>External science-review view of active campaign products</div>"
        "</div>"
        "<div class='model-pill'>active campaign only</div>"
        "</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        f"{_operational_wait_state(index)}"
        f"{_iceland_readiness_panel()}"
        f"{_daily_review_queue_table(index)}"
        f"{_seven_day_replay_summary(index)}"
        f"{_evaluation_schematic()}"
        "</div>"
    )
    return pn.Column(pn.pane.HTML(html, sizing_mode="stretch_width"), sizing_mode="stretch_width")


def _lasso_bundle_table(rows: list[dict[str, object]], include_paths: bool = False) -> str:
    if not rows:
        return ""
    body = []
    for row in rows:
        path_cells = ""
        if include_paths:
            path_cells = (
                f"<td><code>{escape(str(row.get('bundle_json', '')))}</code></td>"
                f"<td><code>{escape(str(row.get('compliance_json', '')))}</code></td>"
            )
        body.append(
            "<tr>"
            f"<td>{escape(str(row.get('day', '')))}</td>"
            f"<td>{escape(str(row.get('status', '')))}</td>"
            f"<td>{escape(str(row.get('compliance', '')))}</td>"
            f"<td>{escape(str(row.get('compliance_detail', '')))}</td>"
            f"<td>{escape(str(row.get('modf', '')))}</td>"
            f"<td>{escape(str(row.get('mmdf', '')))}</td>"
            f"<td>{escape(str(row.get('cloudnet', '')))}</td>"
            f"<td>{escape(str(row.get('seb', '')))}</td>"
            f"<td>{escape(str(row.get('cm1_runtime_h', 'n/a')))}</td>"
            f"<td>{escape(str(row.get('cm1_eval_h', 'n/a')))}</td>"
            f"<td>{escape(str(row.get('cm1_recipe_class', 'unknown')))}</td>"
            f"<td>{escape(str(row.get('scheduler_policy', '')))}</td>"
            f"<td>{escape(str(row.get('scheduler_priority', '')))}</td>"
            f"<td>{escape(str(row.get('operational_qa', '')))}</td>"
            f"<td>{escape(str(row.get('scheduler_actions', '')))}</td>"
            f"<td>{escape(str(row.get('scorecards', '')))}</td>"
            f"{path_cells}"
            "</tr>"
        )
    path_headers = "<th>bundle path</th><th>compliance path</th>" if include_paths else ""
    return (
        "<div class='model-table-wrap'>"
        "<table class='model-table operational-table lasso-table'>"
        "<thead><tr>"
        "<th>day</th><th>bundle</th><th>compliance</th><th>compliance detail</th>"
        "<th>MODF</th><th>MMDF</th><th>Cloudnet</th><th>SEB</th>"
        "<th>CM1 h</th><th>eval h</th><th>recipe</th>"
        "<th>scheduler</th><th>priority</th><th>QA</th><th>actions</th><th>scorecards</th>"
        f"{path_headers}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _lasso_bundle_panel(_clicks: int = 0) -> pn.Column:
    index = _campaign_index()
    diagnosis = _campaign_process_diagnosis()
    archive_manifest = _campaign_archive_manifest()
    paths = _lasso_bundle_paths()
    rows = _lasso_bundle_rows(paths)
    process_rollup = _process_skill_rollup(index)
    process_diagnoses = _process_diagnoses(diagnosis)
    scheduler_rollup = _scheduler_policy_rollup(index)
    operational_qa_rollup = _operational_qa_rollup(index)
    ready_count = sum(1 for row in rows if row.get("status") == "ready")
    compliance_pass_count = sum(1 for row in rows if row.get("compliance") == "pass")
    latest_path = paths[0] if paths else OPERATIONAL_CAMPAIGN_ROOT / "days" / "*" / "lasso_bundle" / "bundle.json"
    cards = [
        _card("bundles", len(rows)),
        _card("ready", ready_count),
        _card("compliance pass", compliance_pass_count),
        _card("latest day", rows[0].get("day", "missing") if rows else "missing"),
        _card("latest status", rows[0].get("status", "missing") if rows else "missing"),
        _card("latest compliance", rows[0].get("compliance", "missing") if rows else "missing"),
        _card("latest CM1 h", rows[0].get("cm1_runtime_h", "n/a") if rows else "n/a"),
        _card("latest eval h", rows[0].get("cm1_eval_h", "n/a") if rows else "n/a"),
        _card("QA incomplete", operational_qa_rollup.get("qa_incomplete_day_count", "n/a")),
        _card("latest updated", _format_mtime(latest_path)),
        *_archive_manifest_cards(archive_manifest),
    ]
    detail_html = ""
    if SHOW_OPERATIONAL_DETAILS:
        detail_html = (
            f"{_scheduler_policy_rollup_table(index)}"
            f"{_operational_qa_rollup_table(index)}"
            f"{_scheduler_policy_day_table(index)}"
            f"{_process_diagnosis_table(diagnosis)}"
            f"{_process_skill_rollup_table(index)}"
            f"{_process_evidence_table(index)}"
            f"{_archive_manifest_table(archive_manifest)}"
        )
    html = (
        "<div class='model-shell operational-shell'>"
        "<div class='model-headline'>"
        "<div>"
        "<div class='model-title'>AURORA-LASSO Case Library</div>"
        "<div class='model-subtitle'>Daily Cloudnet-centred MODF/MMDF virtual-observatory bundles</div>"
        "</div>"
        "<div class='model-pill'>standalone development view</div>"
        "</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        f"{detail_html}"
        f"{_lasso_bundle_table(rows, include_paths=False)}"
        "</div>"
    )
    if not rows:
        html += "<p>No AURORA-LASSO bundles found yet.</p>"
    return pn.Column(pn.pane.HTML(html, sizing_mode="stretch_width"), sizing_mode="stretch_width")


def _details_provenance_panel(_clicks: int = 0) -> pn.Column:
    index = load_campaign_index()
    diagnosis = _campaign_process_diagnosis()
    archive_manifest = _campaign_archive_manifest()
    bundle_rows = _lasso_bundle_rows(_lasso_bundle_paths())
    operational_rows = _operational_rows(_operational_run_paths())
    html = (
        "<div class='model-shell operational-shell'>"
        "<div class='model-headline'>"
        "<div>"
        "<div class='model-title'>Details / Provenance</div>"
        "<div class='model-subtitle'>Technical evidence for scheduler policy, process labels, paths and generated records</div>"
        "</div>"
        "<div class='model-pill'>collapsible evidence</div>"
        "</div>"
        f"{_scheduler_policy_rollup_table(index)}"
        f"{_operational_qa_rollup_table(index)}"
        f"{_scheduler_policy_day_table(index)}"
        f"{_archive_manifest_table(archive_manifest)}"
        f"{_operator_policy_rollup_table(index)}"
        f"{_process_diagnosis_table(diagnosis)}"
        f"{_process_skill_rollup_table(index)}"
        f"{_process_evidence_table(index)}"
        f"{_operational_table(operational_rows)}"
        f"{_lasso_bundle_table(bundle_rows, include_paths=True)}"
        f"<div class='model-subtitle'>campaign root: <code>{escape(str(OPERATIONAL_CAMPAIGN_ROOT))}</code></div>"
        "</div>"
    )
    return pn.Column(pn.pane.HTML(html, sizing_mode="stretch_width"), sizing_mode="stretch_width")


def _operational_panel(_clicks: int = 0) -> pn.Column:
    index = _campaign_index()
    paths = _operational_run_paths()
    rows = _operational_rows(paths)
    latest = rows[0] if rows else {}
    latest_summary_day, latest_summary = _latest_operational_summary(index, paths)
    latest_path = paths[0] if paths else OPERATIONAL_CAMPAIGN_ROOT / "days" / "*" / "operational_run.json"
    index_pending = _index_required_pending(index)
    index_days = index.get("days", []) if isinstance(index, dict) else []
    scheduler_rollup = _scheduler_policy_rollup(index)
    operational_qa_rollup = _operational_qa_rollup(index)
    cards = [
        _card("campaign index", index.get("status", "missing") if index else "missing"),
        _card("indexed days", len(index_days) if isinstance(index_days, list) else 0),
        _card("QA ready", operational_qa_rollup.get("ready_day_count", "n/a")),
        _card("QA incomplete", operational_qa_rollup.get("qa_incomplete_day_count", "n/a")),
        _card("ERA5 CF CSI mean", _index_cf_metric(index, "era5_cloud_fraction", "cf_V")),
        _card("CM1 LES CF CSI mean", _index_cf_metric(index, "cloud_fraction", "cf_V")),
        _card("latest day", latest.get("day", "missing")),
        _card("gate", latest.get("gate", "missing")),
        _card("W-band CSI", latest.get("wband_csi", "n/a")),
        _card("LWP policy", latest.get("lwp_policy", "n/a")),
        _card("records", len(rows)),
    ]
    detail_html = ""
    if SHOW_OPERATIONAL_DETAILS:
        detail_html = (
            f"{_scheduler_policy_rollup_table(index)}"
            f"{_operational_qa_rollup_table(index)}"
            f"{_scheduler_policy_day_table(index)}"
            f"{_operator_policy_rollup_table(index)}"
            f"<div class='model-subtitle'>latest ASFS detail day: {escape(str(latest_summary_day or 'missing'))}</div>"
            f"{_operator_policy_table(latest_summary)}"
            f"{_asfs_sonic_summary(latest_summary)}"
            f"{_asfs_gas_table(latest_summary)}"
        )
    summary = (
        "<div class='model-shell operational-shell'>"
        "<div class='model-headline'>"
        "<div>"
        "<div class='model-title'>Operational Campaign</div>"
        "<div class='model-subtitle'>Daily ERA5/LES virtual-observatory evaluation records</div>"
        "</div>"
        f"<div class='model-pill'>{escape(str(_format_mtime(latest_path)))}</div>"
        "</div>"
        f"<div class='model-grid'>{''.join(cards)}</div>"
        f"{_campaign_index_table(index)}"
        f"{_operational_table(rows)}"
        f"{detail_html}"
        f"<div class='model-subtitle'>pending required: {escape(', '.join(index_pending) if index_pending else 'none')}</div>"
        f"<div class='model-subtitle'>root: <code>{escape(str(OPERATIONAL_CAMPAIGN_ROOT))}</code></div>"
        "</div>"
    )
    if not rows:
        summary += "<p>No operational run records found yet.</p>"
    return pn.Column(pn.pane.HTML(summary, sizing_mode="stretch_width"), sizing_mode="stretch_width")


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
    if dataset_id == "case_readiness_policy_gate":
        candidates.insert(0, _path(f"{stem}.{suffix}"))
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
    run_ids = tuple(run_id for run_id in DEFAULT_RUN_IDS if run_id in RUNS)
    return OrderedDict((str(RUNS[run_id]["label"]), run_id) for run_id in run_ids)


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
    if dataset_id == "case_readiness_policy_gate":
        return _case_readiness_policy_gate_cards(run_id, spec)
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
        "pamtra_wband_scattering_sweep",
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


def _case_readiness_policy_gate_cards(run_id: str, spec: dict[str, object]) -> list[str]:
    readiness = _artifact_json(run_id, spec, "case_readiness_policy_gate")
    if not readiness:
        return []
    summary = readiness.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    datasets = readiness.get("datasets")
    datasets = datasets if isinstance(datasets, list) else []
    hatpro = next(
        (
            item
            for item in datasets
            if isinstance(item, dict) and item.get("dataset_id") == "hatpro"
        ),
        {},
    )
    return [
        _card("status", readiness.get("status", "n/a")),
        _card("required", summary.get("required_total", "n/a")),
        _card("coverage ok", summary.get("required_coverage_ok", "n/a")),
        _card("policy blocked", summary.get("required_policy_blocked", "n/a")),
        _card("HATPRO policy", hatpro.get("policy_status", "n/a")),
        _card("HATPRO site", hatpro.get("site_status", "n/a")),
    ]


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
    ok_count = sum(
        1 for item in products if isinstance(item, dict) and item.get("status", "ok") == "ok"
    )
    failed_count = sum(
        1 for item in products if isinstance(item, dict) and item.get("status") == "failed"
    )
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
        _card("successful variants", ok_count),
        _card("failed variants", failed_count),
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
    value=next(iter(_run_options().values()), None),
    options=_run_options(),
)
dataset_select = pn.widgets.Select(name="Dataset", value="l3_cf", options=DATASETS)
variable_select = pn.widgets.Select(name="Variable", options=OrderedDict())
day_select = pn.widgets.Select(
    name="Day",
    value=next(iter(_day_options().values()), ""),
    options=_day_options(),
)
instrument_select = pn.widgets.Select(
    name="Instrument",
    value="all",
    options=_instrument_options(),
)
model_filter_select = pn.widgets.Select(
    name="Model / output",
    value="all",
    options=MODEL_FILTERS,
)
metric_family_select = pn.widgets.Select(
    name="Metric family",
    value="all",
    options=METRIC_FAMILY_FILTERS,
)
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
            "day": day_select.value or "",
            "instrument": instrument_select.value or "",
            "model": model_filter_select.value or "",
            "metric_family": metric_family_select.value or "",
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
    if args.get("day") in set(day_select.options.values()):
        day_select.value = args["day"]
    if args.get("instrument") in set(instrument_select.options.values()):
        instrument_select.value = args["instrument"]
    if args.get("model") in set(model_filter_select.options.values()):
        model_filter_select.value = args["model"]
    if args.get("metric_family") in set(metric_family_select.options.values()):
        metric_family_select.value = args["metric_family"]
    if args.get("run") in set(run_select.options.values()):
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
for widget in (day_select, instrument_select, model_filter_select, metric_family_select):
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
operational_panel = pn.bind(_operational_panel, refresh_button.param.clicks)
lasso_bundle_panel = pn.bind(_lasso_bundle_panel, refresh_button.param.clicks)
overview_panel = pn.bind(_overview_panel, refresh_button.param.clicks)
instrument_comparison_panel = pn.bind(
    _instrument_comparison_panel,
    day_select.param.value,
    instrument_select.param.value,
    model_filter_select.param.value,
    metric_family_select.param.value,
    refresh_button.param.clicks,
)
details_provenance_panel = pn.bind(_details_provenance_panel, refresh_button.param.clicks)

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
.model-section-title {
    margin-top: 8px;
    font-size: 14px;
    font-weight: 650;
    color: #22313f;
}
.model-subsection-title {
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 650;
    color: #3b4a5a;
}
.model-note {
    color: #5f6c7b;
    font-size: 12px;
    line-height: 1.4;
}
.model-two-column {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
}
.model-compact-list {
    margin: 0;
    padding-left: 18px;
    color: #3b4a5a;
    font-size: 12px;
    line-height: 1.45;
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
.status-badge {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid #d8e1e8;
    font-size: 11px;
    font-weight: 650;
}
.badge-ready {
    border-color: #b7e4dc;
    background: #f1fbf8;
    color: #0b6b5d;
}
.badge-diagnostic {
    border-color: #f0d58c;
    background: #fff8df;
    color: #7a5b00;
}
.badge-warning {
    border-color: #f1b7a8;
    background: #fff3ef;
    color: #8a3c24;
}
.badge-blocked {
    border-color: #d1d8e0;
    background: #f5f7fa;
    color: #4b5563;
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
.leaderboard-table {
    min-width: 860px;
}
.operational-table {
    min-width: 980px;
}
.asfs-detail-table {
    min-width: 920px;
}
.instrument-comparison-table {
    min-width: 1180px;
}
.instrument-plot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
    margin: 12px 0 16px;
}
.instrument-plot-card {
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #ffffff;
    padding: 10px;
}
.instrument-plot-title {
    font-size: 13px;
    font-weight: 650;
    color: #22313f;
    margin-bottom: 8px;
}
.instrument-plot-image {
    display: block;
    width: 100%;
    max-height: 430px;
    object-fit: contain;
}
.schematic {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) auto minmax(180px, 1fr) auto minmax(180px, 1fr) auto minmax(180px, 1fr);
    gap: 10px;
    align-items: stretch;
    margin-top: 12px;
}
.schematic-col {
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #ffffff;
    padding: 10px;
}
.schematic-title {
    font-size: 13px;
    font-weight: 700;
    color: #22313f;
    margin-bottom: 8px;
}
.schematic-box {
    border: 1px solid #e1e7ee;
    border-radius: 6px;
    background: #fbfcfd;
    padding: 7px 8px;
    margin-top: 6px;
    font-size: 12px;
    color: #3b4a5a;
}
.schematic-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0b7285;
    font-weight: 700;
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
    .schematic {
        grid-template-columns: 1fr;
    }
    .schematic-arrow {
        transform: rotate(90deg);
        min-height: 20px;
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
        sizing_mode="stretch_width",
        css_classes=["model-controls"],
    ),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
)

review_controls = pn.Card(
    pn.Row(
        day_select,
        instrument_select,
        model_filter_select,
        metric_family_select,
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

main_sections = [
    pn.Card(
        pn.panel(overview_panel, sizing_mode="stretch_width"),
        title="Overview",
        collapsible=True,
        collapsed=False,
        sizing_mode="stretch_width",
    ),
    pn.Card(
        review_controls,
        pn.panel(instrument_comparison_panel, sizing_mode="stretch_width"),
        share,
        title="Instrument Comparisons",
        collapsible=True,
        collapsed=False,
        sizing_mode="stretch_width",
    ),
    pn.Card(
        pn.panel(lasso_bundle_panel, sizing_mode="stretch_width"),
        title="Case Library",
        collapsible=True,
        collapsed=False,
        sizing_mode="stretch_width",
    ),
    pn.Card(
        pn.panel(details_provenance_panel, sizing_mode="stretch_width"),
        title="Details / Provenance",
        collapsible=True,
        collapsed=True,
        sizing_mode="stretch_width",
    ),
]

template.main[:] = [
    pn.Column(
        *main_sections,
        sizing_mode="stretch_width",
        margin=0,
    )
]

template.servable(location=True)
