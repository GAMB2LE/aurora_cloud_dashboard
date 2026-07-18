#!/usr/bin/env python3
"""Generate development operating-state and named SOC scenario products."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from power_operating_scenarios import DEFAULT_EVENTS_PATH, build_operating_scenarios, fit_operating_model, load_operating_events

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
PDU_ZARR_PATH = Path(os.environ.get("PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))
FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
ENSEMBLE_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_forecast.zarr")
)
STATE_OUTPUT_PATH = Path(
    os.environ.get("POWER_OPERATING_STATE_ZARR_PATH", "/data/aurora/dev-products/power/power_operating_state.zarr")
)
SCENARIO_OUTPUT_PATH = Path(
    os.environ.get(
        "POWER_OPERATING_SCENARIOS_ZARR_PATH",
        "/data/aurora/dev-products/power/power_operating_scenarios.zarr",
    )
)
MODEL_STATE_PATH = Path(
    os.environ.get(
        "POWER_OPERATING_MODEL_STATE_PATH",
        "/data/aurora/dev-products/power/power_operating_model_state.json",
    )
)
RECOMMENDATION_ARCHIVE_PATH = Path(
    os.environ.get(
        "POWER_OPERATING_RECOMMENDATION_ARCHIVE_PATH",
        "/data/aurora/dev-products/power/power_operating_recommendations.json",
    )
)
LEGACY_STATE_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_STATE_PATH", "/data/aurora/products/power/power_soc_forecast_state.json")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_zarr_atomic(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    chunks: dict[str, int] = {}
    for dimension, size in dataset.sizes.items():
        if dimension == "time":
            chunks[dimension] = min(max(int(size), 1), 672)
        elif dimension == "member":
            chunks[dimension] = min(max(int(size), 1), 10)
        else:
            chunks[dimension] = min(max(int(size), 1), 16)
    dataset.chunk(chunks).to_zarr(temporary, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    temporary.replace(path)


def _archive_recommendation(path: Path, scenarios: xr.Dataset, *, retention: int = 256) -> None:
    current = _read_json(path)
    records = current.get("recommendations", [])
    if not isinstance(records, list):
        records = []
    scenario_ids = [str(value) for value in scenarios["scenario"].values]
    if "optimized_cl61" not in scenario_ids:
        return
    index = scenario_ids.index("optimized_cl61")
    start = scenarios["ScenarioStartTime"].values[index]
    stop = scenarios["ScenarioStopTime"].values[index]
    record = {
        "issued_at_utc": _utc_now(),
        "initial_soc_time": str(scenarios.attrs.get("initial_soc_time", "")),
        "initial_soc_pct": float(scenarios.attrs.get("initial_soc_pct", "nan")),
        "model": str(scenarios.attrs.get("model", "")),
        "model_version": int(scenarios.attrs.get("model_version", 0)),
        "current_mode": str(scenarios.attrs.get("current_mode", "")),
        "start_time": None if str(start) == "NaT" else str(start),
        "stop_time": None if str(stop) == "NaT" else str(stop),
        "collection_hours": float(scenarios["ScenarioCollectionHours"].values[index]),
        "minimum_p10_soc": float(scenarios["ScenarioMinimumP10SOC"].values[index]),
        "final_p10_soc": float(scenarios["ScenarioFinalP10SOC"].values[index]),
        "starts": int(scenarios["ScenarioStarts"].values[index]),
        "safe": bool(scenarios["ScenarioSafe"].values[index] >= 0.5),
        "control_authority": "advisory_only",
    }
    if records:
        latest = records[-1]
        comparable = ("initial_soc_time", "current_mode", "start_time", "stop_time", "collection_hours", "safe")
        if isinstance(latest, dict) and all(latest.get(name) == record.get(name) for name in comparable):
            records[-1] = record
        else:
            records.append(record)
    else:
        records.append(record)
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "updated_at_utc": _utc_now(),
            "recommendations": records[-max(int(retention), 1) :],
        },
    )


def _validate_operating_inputs(
    power: xr.Dataset,
    forecast: xr.Dataset,
    *,
    planning_hours: int,
    max_power_age_minutes: float | None,
    minimum_horizon_hours: int = 96,
    now: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, float, float]:
    """Require a fresh physical anchor and enough ECMWF solar coverage to plan."""
    if "time" not in power or "BatterySOC" not in power:
        raise ValueError("Operating scenarios require power time and BatterySOC inputs")
    times = pd.DatetimeIndex(power["time"].values)
    values = np.asarray(power["BatterySOC"].values, dtype=np.float64)
    valid = np.flatnonzero((~times.isna()) & np.isfinite(values))
    if not valid.size:
        raise ValueError("Operating scenarios require a finite BatterySOC anchor")
    latest_index = int(valid[-1])
    anchor_time = pd.Timestamp(times[latest_index])
    anchor_soc = float(values[latest_index])
    reference = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if reference.tzinfo is not None:
        reference = reference.tz_convert("UTC").tz_localize(None)
    if anchor_time.tzinfo is not None:
        anchor_time = anchor_time.tz_convert("UTC").tz_localize(None)
    power_age_minutes = float((reference - anchor_time) / pd.Timedelta(minutes=1))
    if max_power_age_minutes is not None and power_age_minutes > float(max_power_age_minutes):
        raise ValueError(
            "Refusing to publish operating scenarios from stale SOC/load input: "
            f"latest BatterySOC is {power_age_minutes:.1f} minutes old "
            f"(limit {float(max_power_age_minutes):.1f} minutes)"
        )
    if "time" not in forecast or forecast.sizes.get("time", 0) == 0:
        raise ValueError("Operating scenarios require an ECMWF planning forecast with time coverage")
    forecast_times = pd.DatetimeIndex(forecast["time"].values)
    required_hours = min(int(planning_hours), max(int(minimum_horizon_hours), 1))
    required_end = anchor_time + pd.Timedelta(hours=required_hours)
    if forecast_times.max() < required_end:
        raise ValueError(
            "Refusing to publish operating scenarios without the minimum decision horizon: "
            f"coverage ends at {forecast_times.max().isoformat()}, "
            f"but {required_end.isoformat()} is required for {required_hours} h planning"
        )
    return anchor_time, anchor_soc, max(power_age_minutes, 0.0)


def _planning_forecast_provenance(forecast: xr.Dataset) -> dict[str, str]:
    """Capture the exact planning-cycle identity used for a scenario product."""
    times = pd.DatetimeIndex(forecast["time"].values)
    return {
        "planning_forecast_generated_at_utc": str(forecast.attrs.get("generated_at_utc", "")),
        "planning_forecast_initial_soc_time": str(forecast.attrs.get("initial_soc_time", "")),
        "planning_forecast_refresh_kind": str(forecast.attrs.get("forecast_refresh_kind", "")),
        "planning_forecast_verification_eligible": str(forecast.attrs.get("forecast_verification_eligible", "")),
        "planning_forecast_time_coverage_start": times.min().isoformat() if len(times) else "",
        "planning_forecast_time_coverage_end": times.max().isoformat() if len(times) else "",
    }


def generate(
    *,
    power_zarr: Path = POWER_ZARR_PATH,
    pdu_zarr: Path = PDU_ZARR_PATH,
    forecast_zarr: Path = FORECAST_ZARR_PATH,
    ensemble_zarr: Path | None = ENSEMBLE_ZARR_PATH,
    state_output: Path = STATE_OUTPUT_PATH,
    scenario_output: Path = SCENARIO_OUTPUT_PATH,
    model_state: Path = MODEL_STATE_PATH,
    bootstrap_state: Path | None = LEGACY_STATE_PATH,
    recommendation_archive: Path | None = RECOMMENDATION_ARCHIVE_PATH,
    planning_hours: int = 240,
    optimization_hours: int = 96,
    lookback_days: float = 30.0,
    events_path: Path | None = DEFAULT_EVENTS_PATH,
    max_power_age_minutes: float | None = None,
) -> tuple[Path, Path]:
    state = _read_json(model_state)
    if not state and bootstrap_state is not None:
        state = _read_json(bootstrap_state)
    power = xr.open_zarr(power_zarr, chunks={})
    pdu = xr.open_zarr(pdu_zarr, chunks={}) if pdu_zarr.exists() else None
    forecast = xr.open_zarr(forecast_zarr, chunks={})
    ensemble = xr.open_zarr(ensemble_zarr, chunks={}) if ensemble_zarr is not None and ensemble_zarr.exists() else None
    try:
        input_time, input_soc, input_age_minutes = _validate_operating_inputs(
            power,
            forecast,
            planning_hours=planning_hours,
            minimum_horizon_hours=optimization_hours,
            max_power_age_minutes=max_power_age_minutes,
        )
        events = load_operating_events(events_path)
        model = fit_operating_model(power, pdu, raw_state=state, lookback_days=lookback_days, events=events)
        scenarios = build_operating_scenarios(
            power,
            forecast,
            model,
            ensemble=ensemble,
            horizon_hours=planning_hours,
            optimization_hours=optimization_hours,
        )
        scenarios.attrs.update(
            {
                "input_power_time": input_time.isoformat(),
                "input_power_soc_pct": f"{input_soc:.6g}",
                "input_power_age_minutes": f"{input_age_minutes:.6g}",
                "input_validation": "fresh_power_anchor_and_complete_solar_coverage",
                "operating_events_path": str(events_path or ""),
                "operating_event_count": str(len(events)),
                **_planning_forecast_provenance(forecast),
            }
        )
        _write_zarr_atomic(model.state_dataset, state_output)
        _write_zarr_atomic(scenarios, scenario_output)
        _write_json_atomic(model_state, model.state)
        if recommendation_archive is not None:
            _archive_recommendation(recommendation_archive, scenarios)
    finally:
        power.close()
        forecast.close()
        if pdu is not None:
            pdu.close()
        if ensemble is not None:
            ensemble.close()
    print(
        f"Wrote {state_output} and {scenario_output}; "
        f"mode={model.current_mode} confidence={model.current_confidence:.3f}"
    )
    return state_output, scenario_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learned APS operating-state SOC scenarios")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=PDU_ZARR_PATH)
    parser.add_argument("--forecast-zarr", type=Path, default=FORECAST_ZARR_PATH)
    parser.add_argument("--ensemble-zarr", type=Path, default=ENSEMBLE_ZARR_PATH)
    parser.add_argument("--state-output", type=Path, default=STATE_OUTPUT_PATH)
    parser.add_argument("--scenario-output", type=Path, default=SCENARIO_OUTPUT_PATH)
    parser.add_argument("--model-state", type=Path, default=MODEL_STATE_PATH)
    parser.add_argument("--bootstrap-state", type=Path, default=LEGACY_STATE_PATH)
    parser.add_argument("--recommendation-archive", type=Path, default=RECOMMENDATION_ARCHIVE_PATH)
    parser.add_argument("--planning-hours", type=int, default=240)
    parser.add_argument("--optimization-hours", type=int, default=96)
    parser.add_argument("--lookback-days", type=float, default=30.0)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument(
        "--max-power-age-minutes",
        type=float,
        default=20.0,
        help="Reject scenarios when the latest SOC/load observation is older than this limit",
    )
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        pdu_zarr=args.pdu_zarr,
        forecast_zarr=args.forecast_zarr,
        ensemble_zarr=args.ensemble_zarr,
        state_output=args.state_output,
        scenario_output=args.scenario_output,
        model_state=args.model_state,
        bootstrap_state=args.bootstrap_state,
        recommendation_archive=args.recommendation_archive,
        planning_hours=args.planning_hours,
        optimization_hours=args.optimization_hours,
        lookback_days=args.lookback_days,
        events_path=args.events,
        max_power_age_minutes=args.max_power_age_minutes,
    )


if __name__ == "__main__":
    main()
