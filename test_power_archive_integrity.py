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


class PowerArchiveIntegrityTests(unittest.TestCase):
    def test_multiple_appends_widen_legacy_encoding_and_keep_distinct_cycles(self) -> None:
        with TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "archive.zarr"
            legacy = _archive_row_from_forecast(issue(-6))
            for name in (
                "ForecastSystemVersion", "FeatureSetVersion", "FeatureSetDigest",
                "ForecastCodeRevision", "SourceCycleSetID",
            ):
                legacy[name] = ("issue_time", np.array(["nan"], dtype="U3"))
            legacy["ForecastVerificationEligible"] = ("issue_time", [np.nan])
            legacy["IndependentCycle"] = ("issue_time", [np.nan])
            _atomic_write_archive(legacy, archive_path)
            for hour in (0, 6):
                append_forecast_archive(issue(hour), archive_path)
                with xr.open_zarr(archive_path, consolidated=True) as stored:
                    for name, attribute in (
                        ("ForecastSystemVersion", "forecast_system_version"),
                        ("FeatureSetVersion", "feature_set_version"),
                        ("FeatureSetDigest", "feature_set_digest"),
                        ("ForecastCodeRevision", "forecast_code_revision"),
                        ("SourceCycleSetID", "source_cycle_set_id"),
                    ):
                        self.assertEqual(stored[name].values[-1], issue(hour).attrs[attribute])
                    self.assertEqual(stored.LoadMode.values[-1], issue(hour).attrs["load_mode"])
            with xr.open_zarr(archive_path, consolidated=True) as opened:
                archive = opened.load()
            self.assertEqual(len(set(archive.SourceCycleSetID.values[1:])), 2)
            observed_time = pd.date_range("2026-09-01", periods=13, freq="h")
            power = xr.Dataset(
                {"BatterySOC": ("time", np.arange(90.0, 77.0, -1.0))},
                coords={"time": observed_time},
            )
            skill = build_forecast_skill_dataset(archive, power)
            self.assertEqual(float(skill.ForecastSOCMAECycles_0_6h.isel(time=-1)), 2.0)
            self.assertEqual(float(skill.ForecastSOCMAESamples_0_6h.isel(time=-1)), 4.0)
            self.assertTrue(np.isfinite(skill.ForecastSOCMAE_0_6h_Verified.values[-1]))

    def test_rechunked_archive_ignores_inherited_encoding_and_preserves_nanosecond_time(self) -> None:
        with TemporaryDirectory() as temporary:
            archive = _archive_row_from_forecast(issue())
            archive = xr.concat([archive] * 65, dim="issue_time")
            archive["ForecastValidTime"] = archive.ForecastValidTime + np.timedelta64(123456789, "ns")
            for name in archive:
                archive[name].encoding["chunks"] = tuple(6 for _ in archive[name].dims)
            archive["ForecastSystemVersion"].encoding["dtype"] = "<U3"
            _atomic_write_archive(archive, Path(temporary) / "archive.zarr")
            with xr.open_zarr(Path(temporary) / "archive.zarr", consolidated=True) as stored:
                np.testing.assert_array_equal(stored.ForecastValidTime.values, archive.ForecastValidTime.values)
                self.assertEqual(stored.ForecastSystemVersion.values[-1], "power-v10")

    def test_validation_failure_preserves_previous_archive(self) -> None:
        with TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "archive.zarr"
            _atomic_write_archive(_archive_row_from_forecast(issue()), archive_path)
            with patch("power_archive_io.validate_archive_roundtrip", side_effect=ValueError("bad identity")):
                with self.assertRaisesRegex(ValueError, "bad identity"):
                    _atomic_write_archive(_archive_row_from_forecast(issue(6)), archive_path)
            with xr.open_zarr(archive_path, consolidated=True) as stored:
                self.assertEqual(stored.SourceCycleSetID.values[0], issue().attrs["source_cycle_set_id"])

    def test_metadata_preparation_does_not_mutate_input_encoding(self) -> None:
        archive = _archive_row_from_forecast(issue())
        archive.ForecastSystemVersion.encoding["dtype"] = "<U3"
        prepared = prepare_archive_for_storage(archive)
        self.assertEqual(archive.ForecastSystemVersion.encoding["dtype"], "<U3")
        self.assertEqual(prepared.ForecastSystemVersion.encoding, {})

    def test_recovery_uses_verified_snapshots_and_reports_corruption(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "issues"
            first = snapshot(sources, 0)
            snapshot(sources, 6)
            damaged = snapshot(sources, 12)
            metadata = damaged / "forecast.zarr" / ".zattrs"
            metadata.write_bytes(metadata.read_bytes() + b" ")
            unproven = sources / "unproven"
            unproven.mkdir()
            before = (first / "forecast.zarr" / ISSUE_SNAPSHOT_DIGEST_MARKER).read_bytes()
            output = root / "recovered.zarr"
            report_path = root / "report.json"
            report = recover_archive(sources, output, report_path)
            self.assertEqual(report["recovered_issue_count"], 2)
            self.assertEqual(report["distinct_source_cycle_count"], 2)
            self.assertEqual(len(report["excluded"]), 2)
            self.assertIn("content digest", report["excluded"][0]["reason"])
            self.assertEqual((first / "forecast.zarr" / ISSUE_SNAPSHOT_DIGEST_MARKER).read_bytes(), before)
            with xr.open_zarr(output, consolidated=True) as stored:
                self.assertEqual(stored.ForecastCodeRevision.values.tolist(), ["a" * 40] * 2)
            with self.assertRaises(FileExistsError):
                recover_archive(sources, output, root / "second-report.json")

    def test_recovery_rejects_manifest_mismatch_and_output_overlap(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = snapshot(root / "issues", 0)
            manifest_path = directory / "issue_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sourceCycleSetID"] = "wrong-cycle"
            manifest_path.write_text(json.dumps(manifest))
            report = recover_archive(root / "issues", root / "empty.zarr", root / "report.json")
            self.assertEqual(report["status"], "no_verified_issues")
            self.assertFalse((root / "empty.zarr").exists())
            self.assertIn("identity mismatch", report["excluded"][0]["reason"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                recover_archive(root / "issues", root / "issues" / "output.zarr", root / "another.json")

    def test_recovery_excludes_all_conflicting_issues_and_deduplicates_identical_snapshots(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "issues"
            snapshot(sources, 0)
            snapshot(sources, 0, soc_adjustment=1.0, directory_name="conflicting-issue")
            identical = snapshot(sources, 6)
            shutil.copytree(identical, sources / "identical-issue")
            report = recover_archive(sources, root / "recovered.zarr", root / "report.json")
            self.assertEqual(report["recovered_issue_count"], 1)
            self.assertEqual(len(report["excluded"]), 3)
            self.assertEqual(sum("Conflicting" in row["reason"] for row in report["excluded"]), 2)
            self.assertEqual(sum("Identical duplicate" in row["reason"] for row in report["excluded"]), 1)


if __name__ == "__main__":
    unittest.main()
