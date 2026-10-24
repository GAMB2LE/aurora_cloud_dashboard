from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from snapshot_power_baseline import _copy_product, snapshot_baseline


class BaselineSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "power"
        self.source.mkdir()
        for name in ("power_soc_forecast.zarr", "power_soc_forecast_archive.zarr"):
            product = self.source / name
            product.mkdir()
            (product / ".zattrs").write_text('{"description":"baseline"}')
            (product / "array-bytes").write_bytes(b"forecast-data")
        (self.source / "power_soc_forecast_state.json").write_text('{"solar_factor":5.0}')

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_snapshot_verifies_all_bytes_and_excludes_raw_and_caches(self) -> None:
        for name in ("power.zarr", "ecmwf_solar_forecast", "power_soc_forecast.zarr.tmp", "secret.env"):
            (self.source / name).write_bytes(b"excluded")
        output = self.root / "baseline"
        state = self.source / "power_soc_forecast_state.json"
        state_mode = state.stat().st_mode
        result = snapshot_baseline(self.source, output, code_revision="abc123", config_references=["operator-approved-json-v1"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["product_count"], 3)
        self.assertEqual(set(path.name for path in output.iterdir()), {
            "power_soc_forecast.zarr", "power_soc_forecast_archive.zarr",
            "power_soc_forecast_state.json", "manifest.json",
        })
        self.assertEqual(state.stat().st_mode, state_mode)
        self.assertEqual(json.loads((output / "manifest.json").read_text())["code_revision"], "abc123")
        self.assertEqual(result["verification"], "source_before_equals_source_after_equals_copy")

    def test_source_race_preserves_diagnostic_stage_and_never_publishes(self) -> None:
        state = self.source / "power_soc_forecast_state.json"
        copied = 0

        def copy_with_race(source: Path, target: Path) -> None:
            nonlocal copied
            _copy_product(source, target)
            copied += 1
            if copied == 2:
                state.write_text('{"solar_factor":6.0}')

        output = self.root / "baseline"
        with patch("snapshot_power_baseline._copy_product", side_effect=copy_with_race):
            result = snapshot_baseline(self.source, output)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(output.exists())
        staging = Path(result["staging_root"])
        self.assertTrue((staging / "manifest.json").exists())
        self.assertIn("power_soc_forecast_state.json", result["changed_source_products"])
        self.assertEqual(state.read_text(), '{"solar_factor":6.0}')

    def test_copy_corruption_fails_without_touching_source(self) -> None:
        def corrupt(source: Path, target: Path) -> None:
            _copy_product(source, target)
            if target.is_file():
                target.write_bytes(b"corrupt copy")

        with patch("snapshot_power_baseline._copy_product", side_effect=corrupt):
            result = snapshot_baseline(self.source, self.root / "baseline")
        self.assertEqual(result["status"], "failed")
        self.assertIn("copy_mismatch_products", result)
        self.assertEqual((self.source / "power_soc_forecast_state.json").read_text(), '{"solar_factor":5.0}')

    def test_refuses_overlap_existing_output_and_symlink_products(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            snapshot_baseline(self.source, self.source / "snapshot")
        with self.assertRaises(FileExistsError):
            snapshot_baseline(self.source, self.root)
        (self.source / "power_soc_forecast_skill.zarr").symlink_to(self.source / "power_soc_forecast.zarr", target_is_directory=True)
        result = snapshot_baseline(self.source, self.root / "baseline")
        self.assertEqual(result["status"], "failed")
        self.assertIn("symbolic link", result["reason"])


if __name__ == "__main__":
    unittest.main()
