"""Standalone AURORA DART compatibility inventory.

This Panel app sits beside the operational dashboard and model-evaluation page.
It does not run DART. It maps AURORA observation products onto DART concepts:
supported model families, observation converters, observation-sequence tooling,
forward operators, filtering, and observation-space diagnostics.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path
from shutil import which

import panel as pn
import xarray as xr

pn.extension(sizing_mode="stretch_width")

APP_DIR = Path(__file__).resolve().parent
THEME_ACCENT = "#0b7285"


def _asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }.get(suffix, "application/octet-stream")
    return f"data:{mime};base64,{b64encode(path.read_bytes()).decode('utf-8')}"


DASHBOARD_LOGO = _asset_data_uri(APP_DIR / "assets" / "logo.png")
DASHBOARD_FAVICON = "https://gamb2le.pages.dev/assets/logo.png"


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


RAW_ROOT = _env_path("AURORA_RAW_ROOT", "/project/aurora/raw")
DART_ROOT = _env_path("AURORA_DART_ROOT", "/data/aurora/dart")
DART_SOURCE_ROOT = _env_path("DART_ROOT", str(DART_ROOT / "DART"))


@dataclass(frozen=True)
class StreamSpec:
    name: str
    product_path: Path
    raw_path: Path | None
    variables: str
    converter_path: str
    forward_operator: str
    model_families: str
    readiness: str
    readiness_class: str
    recommendation: str


@dataclass(frozen=True)
class ModelFamilySpec:
    family: str
    relevance: str
    aurora_fit: str
    first_question: str


@dataclass(frozen=True)
class ProgramSpec:
    program: str
    dart_role: str
    aurora_use: str


STREAMS = (
    StreamSpec(
        "Vaisala meteorology",
        _env_path("VAISALAMET_ZARR_PATH", "/data/aurora/products/vaisalamet/vaisalamet.zarr"),
        _env_path("VAISALAMET_DIR", "/project/aurora/raw/vaisalamet"),
        "baro_hPa, h1_t, h1_rh, h1_td",
        "Closest to text/MADIS/conventional surface observation converter patterns; likely a small custom converter from Zarr to obs_seq.",
        "Near-surface pressure, temperature and humidity interpolation from the selected model family.",
        "WRF, MPAS-Atmosphere, CAM/CESM; land models for surface coupling context.",
        "candidate for first converter",
        "badge-ready",
        "Use this as the first AURORA-to-DART conversion smoke test because units, location, and uncertainty are tractable.",
    ),
    StreamSpec(
        "HATPRO microwave radiometer",
        _env_path("HATPRO_ZARR_PATH", "/data/aurora/products/hatprog5/hatpro.zarr"),
        RAW_ROOT / "hatprog5",
        "LWP, IWV, SURF_T, T_PROF, T_PROF_CMP",
        "Custom converter likely required; DART can carry custom observation types once preprocessing includes them.",
        "Column water-vapor and liquid-water operators; QCEFF may matter for bounded/cloud-water quantities.",
        "WRF, MPAS-Atmosphere, CAM/CESM for column water; land/surface models only for surface-temperature context.",
        "strong planning candidate",
        "badge-ready",
        "Prioritize IWV/LWP after surface-met conversion, with retrieval uncertainty and metadata gates explicit.",
    ),
    StreamSpec(
        "94 GHz cloud radar",
        _env_path("CLOUD_RADAR_ZARR_PATH", "/data/aurora/products/rpgfmcw94/cloud_radar.zarr"),
        RAW_ROOT / "rpgfmcw94",
        "ZE_dBZ, MeanVel, SpecWidth, ZDR, SLDR, RHV, KDP",
        "DART has radar-observation precedent, but RPG 94 GHz products are not a turnkey NEXRAD converter path.",
        "Radar reflectivity/Doppler operator against hydrometeor state; operator-error and sensitivity assumptions dominate.",
        "WRF and MPAS-Atmosphere are the most plausible atmospheric targets; CAM/CESM only at coarser-scale diagnostic value.",
        "requires forward operator",
        "badge-diagnostic",
        "Keep as high-value science target, but do not make it the first DART assimilation stream.",
    ),
    StreamSpec(
        "CL61 ceilometer",
        _env_path("CEILOMETER_ZARR_PATH", "/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr"),
        _env_path("CEILOMETER_DIR", "/project/aurora/raw/cl61"),
        "beta_att, linear_depol_ratio, cloud_base_heights, visibility/fog flags",
        "Custom lidar/cloud-base observation treatment likely required.",
        "Backscatter/cloud-base/cloud-mask operator; representativeness and siting are current blockers.",
        "WRF and MPAS-Atmosphere for cloud fields; Cloudnet products remain evaluation context.",
        "OSSE design candidate",
        "badge-diagnostic",
        "Use for observation-impact design and cloud-regime diagnostics before direct assimilation.",
    ),
    StreamSpec(
        "ASFS logger and radiation",
        _env_path("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"),
        _env_path("ASFS_LOGGER_DIR", "/project/aurora/raw/asfs/loggernet"),
        "PTemp_Avg, shortwave/longwave radiation, surface temperature, pressure/humidity context",
        "Some overlap with tower/flux-style data patterns, but AURORA schema needs a custom mapping.",
        "Surface energy and radiation operators; may fit land-model DA better than atmospheric DA.",
        "NOAH-MP, CLM, WRF-Hydro, WRF surface schemes; atmospheric models for boundary context.",
        "surface/land candidate",
        "badge-diagnostic",
        "Treat as surface-state and energy-balance readiness work before assimilating into atmosphere.",
    ),
    StreamSpec(
        "ASFS fast sonic",
        _env_path("ASFS_FAST_SONIC_ZARR_PATH", "/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr"),
        RAW_ROOT / "asfs/crd",
        "metek_x_out, metek_y_out, metek_z_out, metek_T_out, quality flags",
        "No direct first-pass converter target; aggregate statistics may become custom observations.",
        "Turbulence and flux representation, not a simple state-variable observation.",
        "WRF/MPAS boundary-layer evaluation; land-atmosphere coupling diagnostics.",
        "diagnostic only for now",
        "badge-blocked",
        "Use for regime tagging and model evaluation; revisit after simpler observation streams work.",
    ),
    StreamSpec(
        "ASFS fast gas",
        _env_path("ASFS_FAST_GAS_ZARR_PATH", "/data/aurora/products/asfs_fast_gas/asfs_fast_gas.zarr"),
        RAW_ROOT / "asfs/crd",
        "licor_co2_out, licor_h2o_out, diagnostics, pressure, temperature",
        "Custom gas/flux treatment required; not a standard weather-observation path.",
        "Gas-state or flux operator only if the selected model carries compatible CO2/H2O variables.",
        "CAM/CESM chemistry/carbon contexts or land-surface flux diagnostics; not v1 atmospheric DA.",
        "diagnostic only for now",
        "badge-blocked",
        "Keep as evaluation and QC context until a model family with matching gas state is selected.",
    ),
    StreamSpec(
        "WXcam",
        _env_path("WXCAM_ZARR_PATH", "/data/aurora/products/wxcam/wxcam.zarr"),
        RAW_ROOT / "wxcam",
        "HDR image pixels, filenames, image dimensions, media catalog context",
        "No direct DART converter target; derived cloud-cover classes would be custom observations.",
        "Image classifier or cloud-cover operator needed before DART can use it.",
        "Any atmospheric model only after calibrated derived products exist.",
        "QC and regime context",
        "badge-blocked",
        "Use for human QC, case selection, and regime tagging rather than DART assimilation.",
    ),
    StreamSpec(
        "Aurora power and operations",
        _env_path("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"),
        _env_path("POWER_DIR", "/project/aurora/raw/power/level1"),
        "BatterySOC, BatteryVolts, solar power, inverter power, source-lag diagnostics",
        "Not a DART observation target.",
        "No geophysical forward operator; operational context only.",
        "None.",
        "not DART-relevant",
        "badge-blocked",
        "Keep on the DART page only as health context for whether observation streams are trustworthy.",
    ),
    StreamSpec(
        "Cloudnet-derived products",
        Path(os.environ.get("AURORA_CLOUDNET_SOURCE_ROOT", "/data/aurora/les/cloudnet/source")),
        None,
        "categorize, IWC, LWC and related retrieval products when available",
        "Potential custom derived-observation path; use carefully because retrieval assumptions are already model-like.",
        "Cloud-mask, water-content and retrieval-error operators; best first as verification/diagnostics.",
        "WRF, MPAS-Atmosphere, CAM/CESM as cloud-state evaluation targets.",
        "evaluation first",
        "badge-diagnostic",
        "Use as a bridge between raw instrument products and DART, but avoid treating retrievals as raw truth.",
    ),
)


MODEL_FAMILIES = (
    ModelFamilySpec(
        "WRF",
        "Most practical atmospheric DART target for mesoscale weather and cloud-impact studies.",
        "Best match to surface met, radiometer columns, radar, and ceilometer planning if a WRF setup exists or is created.",
        "Can Aurora build or obtain WRF ensembles over Leeds/Iceland windows with variables needed by HATPRO/radar operators?",
    ),
    ModelFamilySpec(
        "MPAS-Atmosphere",
        "Strong DART atmospheric model family for global-to-regional workflows.",
        "Plausible for broader campaign context; more infrastructure work than a narrow local inventory.",
        "Is there an MPAS workflow already accessible for these cases, or would this become a separate modeling project?",
    ),
    ModelFamilySpec(
        "CAM/CESM/WACCM",
        "Relevant for larger-scale atmosphere, chemistry, and climate contexts.",
        "Useful for water-vapor/cloud regime studies but likely too coarse for direct W-band/CL61 interpretation.",
        "Are we asking DART a campaign-scale context question rather than a local observatory question?",
    ),
    ModelFamilySpec(
        "NOAH-MP / CLM / WRF-Hydro",
        "Land, hydrology and surface-state DART targets.",
        "Best fit for ASFS radiation, surface temperature, energy balance and possibly power/weather context.",
        "Should Aurora DART first target surface-state estimation rather than atmospheric cloud state?",
    ),
    ModelFamilySpec(
        "ROMS and ocean/ice model families",
        "DART supports model families outside the atmosphere.",
        "Low direct fit to current Aurora atmospheric observations, but relevant if coupled cryosphere/ocean questions enter scope.",
        "Is any non-atmospheric AURORA dataset meant to be assimilated, or is this out of scope?",
    ),
)


DART_PROGRAMS = (
    ProgramSpec(
        "preprocess",
        "Build DART with the observation types needed by a selected experiment.",
        "Would register AURORA-specific observation definitions after the inventory chooses a first stream.",
    ),
    ProgramSpec(
        "text_to_obs / custom converters",
        "Create DART observation-sequence files from real observations.",
        "First concrete Aurora coding target: convert a simple stream such as Vaisala met into an obs_seq-compatible form.",
    ),
    ProgramSpec(
        "obs_sequence_tool, obs_info, obs_seq_coverage",
        "Subset, inspect and quality-control obs_seq files.",
        "Dashboard-ready summaries of time coverage, observation count, rejected values and variable mix.",
    ),
    ProgramSpec(
        "perfect_model_obs",
        "Generate synthetic observations for OSSEs.",
            "Use after a DART-supported model family is selected to test observation types before real-data assimilation.",
    ),
    ProgramSpec(
        "filter",
        "Run ensemble assimilation.",
        "Out of scope until a model family, ensemble, converter and forward operator are chosen.",
    ),
    ProgramSpec(
        "obs_diag / obs_seq_to_netcdf",
        "Produce observation-space diagnostics and NetCDF products.",
        "Best eventual dashboard integration point: prior/posterior fit, bias, spread and coverage by stream.",
    ),
)


def _fmt_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _path_status(path: Path) -> tuple[str, str, str]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return "missing", "Missing", "badge-blocked"
    except OSError as exc:
        return "error", f"Unreadable: {exc}", "badge-warning"
    if path.is_dir():
        return "present", f"Directory, updated {_fmt_time(stat.st_mtime)}", "badge-ready"
    return "present", f"File, updated {_fmt_time(stat.st_mtime)}", "badge-ready"


def _dataset_summary(path: Path) -> str:
    if not path.exists():
        return "No product found"
    try:
        ds = xr.open_zarr(path, consolidated=False)
    except Exception as exc:
        return f"Metadata unavailable: {exc}"
    try:
        dims = ", ".join(f"{name}={size}" for name, size in ds.sizes.items()) or "no dimensions"
        variables = ", ".join(list(ds.data_vars)[:8]) or "no data variables"
        suffix = ""
        if len(ds.data_vars) > 8:
            suffix = f" (+{len(ds.data_vars) - 8} more)"
        return f"{dims}; variables: {variables}{suffix}"
    finally:
        ds.close()


def _badge(label: str, klass: str) -> str:
    return f"<span class='status-badge {klass}'>{escape(label)}</span>"


def _table(headers: list[str], rows: list[list[str]], table_class: str = "") -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f"<div class='dart-table-wrap'><table class='dart-table {table_class}'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _overview_markup() -> pn.pane.HTML:
    return pn.pane.HTML(
        """
        <div class="dart-shell">
          <div class="dart-headline">
            <div>
              <div class="dart-title">AURORA DART Compatibility Inventory</div>
              <div class="dart-subtitle">
                A docs-driven map from the existing Aurora observing system to DART model families,
                observation converters, forward operators, filtering and diagnostics.
              </div>
            </div>
            <div class="dart-pill">Planning only</div>
          </div>
          <div class="dart-grid">
            <div class="dart-card"><div class="dart-card__label">First deliverable</div><div class="dart-card__value">Inventory</div></div>
            <div class="dart-card"><div class="dart-card__label">First coding target</div><div class="dart-card__value">Converter smoke test</div></div>
            <div class="dart-card"><div class="dart-card__label">Experiment path</div><div class="dart-card__value">OSSE before OSE</div></div>
            <div class="dart-card"><div class="dart-card__label">Current status</div><div class="dart-card__value">No DART run</div></div>
          </div>
          <p>
            DART can help Aurora only after three choices are explicit: the target model family,
            the observation types and converters, and the forward operators that turn a model state
            into expected observations. This page is therefore an inventory and decision aid, not a
            control panel for running assimilation.
          </p>
        </div>
        """,
        sizing_mode="stretch_width",
    )


def _runtime_status(_clicks: int = 0) -> pn.pane.HTML:
    executable_checks = []
    for name in ("preprocess", "filter", "obs_diag", "obs_sequence_tool", "obs_seq_to_netcdf"):
        found = which(name)
        executable_checks.append(
            [
                escape(name),
                _badge("present" if found else "not found", "badge-ready" if found else "badge-blocked"),
                f"<code>{escape(found or name)}</code>",
            ]
        )
    source_status, source_label, source_class = _path_status(DART_SOURCE_ROOT)
    work_status, work_label, work_class = _path_status(DART_ROOT)
    rows = [
        ["DART source tree", _badge(source_status, source_class), f"<code>{escape(str(DART_SOURCE_ROOT))}</code>", escape(source_label)],
        ["DART working root", _badge(work_status, work_class), f"<code>{escape(str(DART_ROOT))}</code>", escape(work_label)],
    ]
    rows.extend(executable_checks)
    return pn.pane.HTML(_table(["Check", "State", "Evidence", "Interpretation"], rows), sizing_mode="stretch_width")


def _stream_inventory(_clicks: int = 0) -> pn.pane.HTML:
    rows: list[list[str]] = []
    for stream in STREAMS:
        status, product_label, klass = _path_status(stream.product_path)
        raw_label = "Not tracked"
        if stream.raw_path is not None:
            _, raw_label, _ = _path_status(stream.raw_path)
        rows.append(
            [
                escape(stream.name),
                _badge(stream.readiness, stream.readiness_class),
                _badge(status, klass),
                escape(stream.variables),
                escape(stream.converter_path),
                escape(stream.forward_operator),
                escape(stream.model_families),
                escape(stream.recommendation),
                escape(product_label),
                escape(raw_label),
                escape(_dataset_summary(stream.product_path)),
            ]
        )
    return pn.pane.HTML(
        _table(
            [
                "Stream",
                "DART readiness",
                "Product",
                "Variables",
                "Converter path",
                "Forward operator",
                "Model-family fit",
                "Recommendation",
                "Product state",
                "Raw state",
                "Metadata",
            ],
            rows,
            "stream-table",
        ),
        sizing_mode="stretch_width",
    )


def _model_family_markup() -> pn.pane.HTML:
    rows = [
        [
            escape(item.family),
            escape(item.relevance),
            escape(item.aurora_fit),
            escape(item.first_question),
        ]
        for item in MODEL_FAMILIES
    ]
    return pn.pane.HTML(
        _table(["DART model family", "DART relevance", "Aurora fit", "First planning question"], rows, "model-family-table"),
        sizing_mode="stretch_width",
    )


def _program_markup() -> pn.pane.HTML:
    rows = [
        [f"<code>{escape(item.program)}</code>", escape(item.dart_role), escape(item.aurora_use)]
        for item in DART_PROGRAMS
    ]
    return pn.pane.HTML(
        _table(["DART program / path", "DART role", "Aurora use"], rows, "program-table"),
        sizing_mode="stretch_width",
    )


def _roadmap_markup() -> pn.pane.HTML:
    rows = [
        [
            "1. Inventory",
            "Finish stream-by-stream readiness, converter, model-family and operator classification.",
            "Current page.",
        ],
        [
            "2. Choose model family",
            "Pick a DART-supported model target before designing any experiment.",
            "Decision needed: atmospheric model, surface/land model, or broader context model.",
        ],
        [
            "3. Converter smoke test",
            "Build the smallest real-observation conversion path and inspect the resulting obs_seq product.",
            "Vaisala met is the recommended first stream.",
        ],
        [
            "4. OSSE",
            "Use synthetic observations to prove model-state interpolation, observation definitions and diagnostics.",
            "Run before any real-data assimilation claim.",
        ],
        [
            "5. OSE",
            "Assimilate real observations only after converter, model ensemble, forward operator and diagnostics are proven.",
            "Future work; not active in v1.",
        ],
    ]
    return pn.pane.HTML(
        _table(["Phase", "Purpose", "Aurora default"], rows, "roadmap-table"),
        sizing_mode="stretch_width",
    )


def _research_markup() -> pn.pane.HTML:
    return pn.pane.HTML(
        """
        <div class="dart-shell">
          <div class="dart-section-title">DART concepts used here</div>
          <ul class="dart-list">
            <li>Observation converters create DART observation-sequence files from external data.</li>
            <li>Forward operators map a model state to expected observation values.</li>
            <li><code>filter</code> is the assimilation program; its outputs are meaningful only after a model family, ensemble and observation definitions exist.</li>
            <li>Observation-space diagnostics are the most natural dashboard integration target after a run.</li>
            <li>QCEFF is relevant to bounded or non-Gaussian geophysical quantities, including several cloud and water-content candidates.</li>
          </ul>
          <div class="dart-section-title">Primary documentation links</div>
          <div class="dart-link-grid">
            <a href="https://docs.dart.ucar.edu/en/latest/" target="_blank">DART documentation</a>
            <a href="https://docs.dart.ucar.edu/en/latest/models/README.html" target="_blank">Supported model interfaces</a>
            <a href="https://docs.dart.ucar.edu/en/latest/guide/available-observation-converters.html" target="_blank">Observation converters</a>
            <a href="https://docs.dart.ucar.edu/en/latest/guide/adding-your-observations-to-dart.html" target="_blank">Adding observations</a>
            <a href="https://docs.dart.ucar.edu/en/latest/guide/forward_operator.html" target="_blank">Forward operators</a>
            <a href="https://docs.dart.ucar.edu/en/latest/assimilation_code/programs/readme.html" target="_blank">DART programs</a>
            <a href="/model-evaluation" target="_blank">AURORA model evaluation</a>
          </div>
        </div>
        """,
        sizing_mode="stretch_width",
    )


refresh_button = pn.widgets.Button(name="Refresh status", button_type="primary", width=140)
runtime_status = pn.bind(_runtime_status, refresh_button.param.clicks)
stream_inventory = pn.bind(_stream_inventory, refresh_button.param.clicks)

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
.dart-shell {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.dart-headline {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.dart-title {
    font-size: 18px;
    font-weight: 650;
    color: #22313f;
}
.dart-subtitle {
    margin-top: 3px;
    font-size: 12px;
    color: #5f6c7b;
}
.dart-section-title {
    margin-top: 8px;
    font-size: 14px;
    font-weight: 650;
    color: #22313f;
}
.dart-pill {
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
.dart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
}
.dart-card {
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #fbfcfd;
    padding: 8px 10px;
}
.dart-card__label {
    font-size: 11px;
    color: #647283;
}
.dart-card__value {
    margin-top: 2px;
    font-size: 17px;
    font-weight: 650;
    color: #22313f;
}
.dart-table-wrap {
    overflow-x: auto;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    background: #ffffff;
}
.dart-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 920px;
}
.dart-table th,
.dart-table td {
    border-bottom: 1px solid #e6ebf1;
    padding: 8px 10px;
    vertical-align: top;
    text-align: left;
    font-size: 12px;
}
.dart-table th {
    color: #3b4a5a;
    background: #f8fafb;
    font-weight: 650;
}
.dart-table code {
    word-break: break-all;
    color: #243b53;
}
.stream-table {
    min-width: 1780px;
}
.model-family-table {
    min-width: 1120px;
}
.program-table,
.roadmap-table {
    min-width: 980px;
}
.dart-list {
    margin: 0;
    padding-left: 19px;
    color: #3b4a5a;
}
.dart-list li {
    margin: 5px 0;
}
.dart-link-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px;
}
.dart-link-grid a {
    display: block;
    border: 1px solid #d8e1e8;
    border-radius: 8px;
    padding: 9px 10px;
    color: #0b7285;
    text-decoration: none;
    background: #fbfcfd;
    font-size: 12px;
    font-weight: 650;
}
@media (max-width: 768px) {
    body, .bk { font-size: 14px; }
    .dart-table {
        min-width: 760px;
    }
}
"""

pn.extension(raw_css=[CSS])

template = pn.template.MaterialTemplate(
    title="AURORA DART Compatibility Inventory",
    logo=DASHBOARD_LOGO,
    favicon=DASHBOARD_FAVICON,
    header_background=THEME_ACCENT,
    header_color="white",
    main_max_width="1800px",
)

template.main[:] = [
    pn.Column(
        pn.Card(_overview_markup(), title="Overview", collapsible=True, collapsed=False),
        pn.Card(
            refresh_button,
            pn.panel(runtime_status, sizing_mode="stretch_width"),
            title="DART Runtime Checks",
            collapsible=True,
            collapsed=False,
        ),
        pn.Card(
            pn.panel(stream_inventory, sizing_mode="stretch_width"),
            title="Observation Compatibility Inventory",
            collapsible=True,
            collapsed=False,
        ),
        pn.Card(_model_family_markup(), title="Relevant DART Model Families", collapsible=True, collapsed=False),
        pn.Card(_program_markup(), title="DART Program Map", collapsible=True, collapsed=False),
        pn.Card(_roadmap_markup(), title="Research Roadmap", collapsible=True, collapsed=False),
        pn.Card(_research_markup(), title="Documentation Notes", collapsible=True, collapsed=False),
        sizing_mode="stretch_width",
        margin=0,
    )
]

template.servable(location=True)
