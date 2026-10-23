"""Forecast-only publication must stay coherent across atomic generation swaps."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import app
import mobile_catalog as catalog


@pytest.fixture
def serving(tmp_path, monkeypatch):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")
    active = tmp_path / "active"
    current = tmp_path / "continuity" / "current.zarr"
    current.parent.mkdir()
    xr.Dataset(
        {"BatterySOC": ("time", [77., 76.])},
        coords={"time": [now - pd.Timedelta(minutes=2), now - pd.Timedelta(minutes=1)]},
    ).to_zarr(current, consolidated=True, zarr_format=2)

    generations = []
    for index, soc in enumerate((81., 71.)):
        root = tmp_path / f"generation-{index}"
        root.mkdir()
        anchor = now - pd.Timedelta(minutes=10 - index)
        times = pd.date_range(anchor, periods=3, freq="1h")
        forecast = xr.Dataset(
            {
                "BatterySOCForecastP10": ("time", [soc, soc-10, soc-20]),
                "BatterySOCForecastP50": ("time", [soc, soc-5, soc-10]),
                "BatterySOCForecastP90": ("time", [soc, soc-2, soc-4]),
                "OperatingCL61OptimizedSOCP50": ("time", [99., 99., 99.]),
            },
            coords={"time": times},
            attrs={"forecast_initial_soc_time": anchor.isoformat(),
                   "forecast_refresh_kind": "stale_embedded_metadata",
                   "operating_planning_status": "unavailable"},
        )
        ids = ["current_mode", "optimized_cl61", "uas_tier_1"]
        scenarios = xr.Dataset(
            {
                **{f"ScenarioSOC{q}": (("scenario", "time"), np.tile([soc, soc-5, soc-10], (3, 1))) for q in ("P10", "P50", "P90")},
                "ScenarioBelow40Probability": (("scenario", "time"), np.zeros((3, 3))),
                "ScenarioModeCode": (("scenario", "time"), np.ones((3, 3), dtype=np.int16)),
                "SolarEnsembleWatts": (("member", "time"), np.ones((1, 3))),
                "ComponentLoadWatts": (("member", "component"), np.ones((1, 1))),
            },
            coords={"time": times, "scenario": ids, "member": [0], "component": ["DC"]},
            attrs={"initial_soc_time": anchor.isoformat(), "generated_at_utc": now.isoformat(),
                   "planning_forecast_refresh_kind": "cached_reanchor", "planning_status": "ready"},
        )
        products = {}
        for name, filename, dataset in (
            ("forecastDisplay", "forecast.zarr", forecast),
            ("operatingScenarios", "scenarios.zarr", scenarios),
        ):
            dataset.to_zarr(root / filename, consolidated=True, zarr_format=2)
            products[name] = {"relativePath": filename, "sha256": catalog._power_bundle_artifact_digest(root / filename)}
        manifest = {
            "status": "complete", "controlAuthority": "advisory_only", "cl61ActuationEnabled": False,
            "dataUpdatedAt": now.isoformat() + "Z", "validUntil": (now + pd.Timedelta(days=3)).isoformat() + "Z",
            "forecastIdentityID": f"identity-{index}", "sourceCycleSetID": f"cycle-{index}",
            "observationCutoffUTC": anchor.isoformat() + "Z", "forecastRefreshKind": "cached_reanchor",
            "independentCycle": False, "degradedModeCode": "cached_reanchor", "products": products,
        }
        (root / "generation.json").write_text(json.dumps(manifest))
        generations.append(root)
    active.symlink_to(generations[0], target_is_directory=True)
    env = {
        "AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED": "true",
        "AURORA_POWER_FORECAST_PUBLICATION_ACTIVE": "false",
        "AURORA_POWER_FORECAST_ONLY_PUBLICATION_ACTIVE": "true",
        "AURORA_POWER_FORECAST_BUNDLE_READY_PATH": str(active / "generation.json"),
        "AURORA_POWER_FORECAST_BUNDLE_STATUS_PATH": str(tmp_path / "status.json"),
        "POWER_CURRENT_DISPLAY_ZARR_PATH": str(current),
        "POWER_FORECAST_DISPLAY_ZARR_PATH": str(tmp_path / "old-continuity-forecast.zarr"),
        "POWER_OPERATING_SCENARIOS_ZARR_PATH": str(active / "scenarios.zarr"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    catalog._POWER_BUNDLE_VALIDATION_CACHE.clear()
    app._refresh_power_display_energy_dataset()
    yield dict(root=tmp_path, now=now, active=active, generations=generations, current=current)
    app._refresh_power_display_energy_dataset()
    catalog._POWER_BUNDLE_VALIDATION_CACHE.clear()


def rotate(serving):
    temporary = serving["root"] / "next-active"
    temporary.symlink_to(serving["generations"][1], target_is_directory=True)
    temporary.replace(serving["active"])


def test_forecast_only_flag_preserves_shadow_and_current_defaults(serving, monkeypatch):
    assert catalog.power_display_section_path("current") == serving["current"]
    monkeypatch.setenv("AURORA_POWER_FORECAST_ONLY_PUBLICATION_ACTIVE", "false")
    assert catalog.power_forecast_serving_selection() is None
    assert catalog.power_forecast_bundle_status(Path("unused"))["status"] == "shadow_unpublished"
    monkeypatch.setenv("AURORA_POWER_FORECAST_ONLY_PUBLICATION_ACTIVE", "true")
    monkeypatch.setenv("AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED", "false")
    assert catalog.power_forecast_serving_selection() is None


def test_api_curves_scenarios_and_provenance_use_one_completed_generation(serving):
    payload = catalog.power(window="96h", group="forecast")
    assert payload["source"]["path"] == str(serving["generations"][0] / "forecast.zarr")
    assert payload["forecastIdentityID"] == "identity-0"
    assert payload["forecastRefreshKind"] == "cached_reanchor"
    assert payload["independentCycle"] is False
    panels = {panel["id"]: panel for panel in payload["panels"]}
    assert {"soc_ecmwf_forecast", "operating_plan_scenarios", "uas_tier_scenarios", "operating_plan_schedule"} <= panels.keys()
    for panel in payload["panels"]:
        context = panel.get("forecastContext")
        if context:
            assert context["forecastIdentityID"] == "identity-0"
            assert context["forecastRefreshKind"] == "cached_reanchor"
            assert context["independentCycle"] is False
            assert context["observationCutoffUTC"] == payload["observationCutoffUTC"]
    assert panels["soc_ecmwf_forecast"]["traces"][0]["points"][0]["value"] == 81.
    current = catalog.power(group="current")
    assert current["source"]["path"] == str(serving["current"])
    assert any(trace["points"][-1]["value"] == 76 for panel in current["panels"] for trace in panel["traces"])
    combined = catalog.power(group="all")
    assert combined["observedSource"]["path"] == str(serving["current"])
    assert len(combined["panels"]) == len(current["panels"]) + len(payload["panels"])


@pytest.mark.parametrize("reader", ["api", "browser"])
def test_active_pointer_rotation_cannot_mix_forecast_scenarios_or_provenance(serving, reader):
    original = xr.open_zarr
    rotated = False

    def open_and_rotate(path, *args, **kwargs):
        nonlocal rotated
        result = original(path, *args, **kwargs)
        if Path(path) == serving["generations"][0] / "forecast.zarr" and not rotated:
            rotate(serving)
            rotated = True
        return result

    with patch.object(xr, "open_zarr", side_effect=open_and_rotate):
        if reader == "api":
            first = catalog.power(group="forecast")
            assert first["forecastIdentityID"] == "identity-0"
            assert first["panels"][0]["traces"][0]["points"][0]["value"] == 81.
        else:
            first = app._open_power_display_summary_window(serving["now"] - pd.Timedelta(hours=1), serving["now"], "forecast")
            assert first.attrs["forecast_identity_id"] == "identity-0"
            assert float(first.OperatingCL61OptimizedSOCP50.dropna("time")[0]) == 81.
            assert first.attrs["operating_planning_status"] == "ready"
    assert rotated
    if reader == "api":
        second = catalog.power(group="forecast")
        assert second["forecastIdentityID"] == "identity-1"
        assert second["panels"][0]["traces"][0]["points"][0]["value"] == 71.
    else:
        second = app._open_power_display_summary_window(serving["now"] - pd.Timedelta(hours=1), serving["now"], "forecast")
        assert second.attrs["forecast_identity_id"] == "identity-1"
        assert float(second.OperatingCL61OptimizedSOCP50.dropna("time")[0]) == 71.
        assert app._POWER_DISPLAY_SECTION_SOURCE_PATH["forecast"].parent == serving["generations"][1]
        assert app._POWER_OPERATING_SCENARIOS_SOURCE_PATH.parent == serving["generations"][1]


@pytest.mark.parametrize("failure", ["missing", "digest", "incomplete", "control", "escape"])
def test_invalid_forecast_fails_closed_without_hiding_current_telemetry(serving, failure):
    root = serving["generations"][0]
    manifest_path = root / "generation.json"
    manifest = json.loads(manifest_path.read_text())
    if failure == "missing":
        manifest_path.unlink()
    elif failure == "digest":
        (root / "forecast.zarr" / ".zattrs").write_text('{}')
    else:
        if failure == "incomplete":
            manifest["status"] = "incomplete"
        elif failure == "control":
            manifest["cl61ActuationEnabled"] = True
        else:
            manifest["products"]["operatingScenarios"]["relativePath"] = "../outside.zarr"
        manifest_path.write_text(json.dumps(manifest))
    payload = catalog.power(group="forecast")
    assert payload["status"] == "unavailable"
    assert payload["panels"] == []
    assert catalog.power(group="current")["panels"]
    assert catalog.power(group="all")["panels"]
    assert app._open_power_display_summary_window(serving["now"] - pd.Timedelta(hours=1), serving["now"], "forecast") is None
    assert app._open_power_display_summary_window(serving["now"] - pd.Timedelta(hours=1), serving["now"], "current") is not None


def test_forecast_only_browser_does_not_use_legacy_prewarm(serving):
    with patch.object(app, "power_view_select", SimpleNamespace(value="forecast")):
        assert app._prewarmed_interactive_path("power").name == ".forecast-prewarm-not-published.json"
    assert catalog.power_prewarm_path("forecast").name == ".forecast-prewarm-not-published.json"
    assert catalog.power_prewarm_path("current").name == "power_current_latest_interactive.json"


@pytest.mark.parametrize("missing_power", [False, True])
@pytest.mark.parametrize("group", ["current", "all", "forecast"])
def test_enclosure_wrapper_adds_one_independent_panel(serving, missing_power, group, monkeypatch):
    if missing_power:
        monkeypatch.setenv("POWER_CURRENT_DISPLAY_ZARR_PATH", str(serving["root"] / "missing-current.zarr"))
        monkeypatch.setenv("POWER_DISPLAY_SUMMARY_ZARR_PATH", str(serving["root"] / "missing-summary.zarr"))
        (serving["generations"][0] / "generation.json").unlink()
    panel = {"id": "aps_enclosure", "traces": [{"id": "APSBLETemperature", "points": [{"time": "2026-10-23T09:00:00Z", "value": 12}]}]}
    with patch.object(catalog, "enclosure_power_panel", return_value=panel) as enclosure:
        result = catalog.power(group=group)
    enclosure_panels = [item for item in result["panels"] if item["id"] == "aps_enclosure"]
    if group == "forecast":
        assert enclosure_panels == []
        enclosure.assert_not_called()
    else:
        assert enclosure_panels == [panel]
        enclosure.assert_called_once()
