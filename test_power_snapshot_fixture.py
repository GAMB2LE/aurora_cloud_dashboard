from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
    ISSUE_SNAPSHOT_DIGEST_MARKER,
    _archive_row_from_forecast,
    _atomic_write_archive,
    append_forecast_archive,
    build_forecast_skill_dataset,
    write_immutable_issue_snapshot,
)
from power_archive_io import prepare_archive_for_storage
from repair_power_forecast_archive import recover_archive


def issue(hour: int = 0) -> xr.Dataset:
    anchor = pd.Timestamp("2026-09-01") + pd.Timedelta(hours=hour)
    stamp = anchor.isoformat()
    return xr.Dataset(
        {
            "BatterySOCForecast": ("time", np.array([89.0, 87.0], dtype=np.float32)),
            "ForecastLoadWatts": ("time", np.array([400.0, 450.0], dtype=np.float32)),
            "ForecastSolarWatts": ("time", np.array([0.0, 100.0], dtype=np.float32)),
        },
        coords={"time": pd.DatetimeIndex([anchor + pd.Timedelta(hours=1), anchor + pd.Timedelta(hours=3)])},
        attrs={
            "initial_soc_time": stamp, "initial_soc_pct": "90",
            "soc_anchor_time_utc": stamp, "observation_cutoff_utc": stamp,
            "training_cutoff_utc": stamp, "ecmwf_cycle_time": stamp,
            "forecast_verification_eligible": "true", "independent_cycle": "true",
            "forecast_refresh_kind": "ecmwf_cycle", "load_model_version": "10",
            "load_mode": "DC-Only + CL61 + Radar + HATPRO", "load_mode_learning_ready": "true",
            "forecast_model_contract_id": "forecast-model-v2-test-long-contract",
            "forecast_identity_id": f"forecast-identity-v1-test-{hour}",
            "forecast_system_version": "power-v10",
            "feature_set_version": "ecmwf_ssrd_scalar_v1+finite_operating_state_phases_v2",
            "feature_set_digest": "7" * 64, "forecast_code_revision": "a" * 40,
            "source_cycle_set_id": f"ecmwf:legacy:{stamp}:sha256:{str(hour).zfill(20)}",
            "source_manifest_digest": "sha256:" + "b" * 64,
            "publication_signature": f"test-signature-{hour}",
        },
    )


def snapshot(root: Path, hour: int, *, soc_adjustment: float = 0.0, directory_name: str | None = None) -> Path:
    forecast = issue(hour)
    forecast["BatterySOCForecast"] = forecast.BatterySOCForecast + soc_adjustment
    directory = root / (directory_name or forecast.attrs["forecast_identity_id"])
    path = write_immutable_issue_snapshot(forecast, directory / "forecast.zarr")
    marker_bytes = (path / ISSUE_SNAPSHOT_DIGEST_MARKER).read_bytes()
    marker = json.loads(marker_bytes)
    manifest = {
        "schemaVersion": 1, "status": "complete", "relativePath": "forecast.zarr",
        "digestAlgorithm": ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
        "publicationSignature": forecast.attrs["publication_signature"],
        "forecastIdentityID": forecast.attrs["forecast_identity_id"],
        "sourceCycleSetID": forecast.attrs["source_cycle_set_id"],
        "sourceCycleUTC": forecast.attrs["ecmwf_cycle_time"],
        "contentDigest": marker["contentDigest"],
        "snapshotMarkerDigest": "sha256:" + hashlib.sha256(
            ISSUE_SNAPSHOT_DIGEST_MARKER.encode("utf-8") + b"\0" + marker_bytes + b"\0"
        ).hexdigest(),
    }
    (directory / "issue_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory
