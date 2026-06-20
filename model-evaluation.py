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
                "l3_cf": _path(
                    "cloudnet_l3",
                    "cm1_0400_thompson_tall_rh105_25_60",
                    "aurora_multistream_pilot_20260520_20260602_cm1_0400_thompson_tall_rh105_25_60_official_l3-cf_2026-05-21.nc",
                ),
                "run_dir": _path("model", "cm1_0400_thompson_tall_rh105_25_60", "run_20260521"),
                "uuid": "e63672c1-26d5-4bda-8839-84b199ce6823",
                "runtime": "completed normally at 1800 s; 16 native outputs",
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
                    "aurora_multistream_pilot_20260520_20260602_cm1_smoke_l3-cf_2026-05-21.nc",
                ),
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
                    "aurora_multistream_pilot_20260520_20260602_era5_l3-cf_2026-05-21.nc",
                ),
                "uuid": "594d87e1-3bf2-428a-a22d-4beffd9ad344",
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
                    "aurora_multistream_pilot_20260520_20260602_les_bridge_l3-cf_2026-05-21.nc",
                ),
                "uuid": "0ed9db33-2101-4b32-999e-b3fe61315dc5",
            },
        ),
    ]
)

DATASETS = OrderedDict(
    [
        ("Cloudnet L3 CF", "l3_cf"),
        ("Cloudnet model", "cloudnet_model"),
        ("CF scorecard", "scorecard"),
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
    if dataset_id == "scorecard":
        scorecard_path = spec.get("scorecard_png")
        return scorecard_path if isinstance(scorecard_path, Path) else None
    path = spec.get(dataset_id)
    return path if isinstance(path, Path) else None


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
    if dataset_id == "scorecard":
        path = _dataset_path(run_id, dataset_id)
        if path is not None and path.exists():
            return OrderedDict([("Scorecard image", "scorecard")])
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


def _scorecard_json(spec: dict[str, object]) -> dict[str, object] | None:
    path = spec.get("scorecard_json")
    if not isinstance(path, Path) or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scorecard_cards(spec: dict[str, object]) -> list[str]:
    scorecard = _scorecard_json(spec)
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
    if dataset_id == "scorecard":
        cards.extend(_scorecard_cards(spec))
        variables = "scorecard PNG, Markdown and JSON"
        file_state = f"{_format_size(path)}; {_format_mtime(path)}" if path is not None and path.exists() else "missing"
        if not cards:
            cards.append(_card("scorecard", file_state))
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
    scorecard_rows = ""
    if dataset_id == "scorecard":
        markdown_path = spec.get("scorecard_markdown")
        json_path = spec.get("scorecard_json")
        if isinstance(markdown_path, Path):
            scorecard_rows += f"<tr><th>scorecard markdown</th><td><code>{escape(str(markdown_path))}</code></td></tr>"
        if isinstance(json_path, Path):
            scorecard_rows += f"<tr><th>scorecard json</th><td><code>{escape(str(json_path))}</code></td></tr>"
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
            {scorecard_rows}
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
    spec = RUNS.get(run_id, {})
    png_path = spec.get("scorecard_png")
    markdown_path = spec.get("scorecard_markdown")
    panes: list[object] = []
    if isinstance(png_path, Path) and png_path.exists():
        data_uri = _asset_data_uri(png_path)
        panes.append(
            pn.pane.HTML(
                f"<img class='scorecard-image' src='{data_uri}' alt='CM1 cloud-fraction scorecard'>",
                sizing_mode="stretch_width",
            )
        )
    else:
        panes.append(pn.pane.Markdown("Scorecard image is missing.", sizing_mode="stretch_width"))
    if isinstance(markdown_path, Path) and markdown_path.exists():
        panes.append(
            pn.Card(
                pn.pane.Markdown(markdown_path.read_text(encoding="utf-8"), sizing_mode="stretch_width"),
                title="Scorecard tables",
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
    variable_select.disabled = dataset_select.value == "scorecard"
    values = list(options.values())
    if dataset_select.value == "scorecard":
        variable_select.value = values[0] if values else None
        return
    path = _dataset_path(run_select.value, dataset_select.value)
    finite_values = _finite_numeric_variables(path) if path is not None else set()
    preferred = ("cf_V", "cf_A", "cf_V_adv", "cf_A_adv", "cloud_fraction", "model_cf", "ql", "qi", "temperature", "pressure")
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
    if args.get("variable") in list(variable_select.options.values()) and args["variable"] in finite_values:
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
    lambda run_id, dataset_id, variable, clicks: _scorecard_panel(run_id, clicks)
    if dataset_id == "scorecard"
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
