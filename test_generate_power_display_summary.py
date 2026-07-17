from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from generate_power_display_summary import _write_metadata


class PowerDisplaySummaryMetadataTests(unittest.TestCase):
    def test_metadata_records_time_bounds_and_dimensions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "power_display_summary.zarr"
            display = xr.Dataset(
                {"power": ("time", np.asarray([1.0, 2.0]))},
                coords={"time": np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")},
            )
            path = _write_metadata(output, display)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["time_count"], 2)
        self.assertEqual(payload["variable_count"], 1)
        self.assertEqual(str(np.datetime64(payload["time_start_utc"])), "2026-07-17T00:00:00")
        self.assertEqual(str(np.datetime64(payload["time_end_utc"])), "2026-07-17T00:01:00")


if __name__ == "__main__":
    unittest.main()
