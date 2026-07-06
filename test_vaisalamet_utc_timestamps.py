import tempfile
import unittest
from pathlib import Path

import numpy as np

from append_new_vaisalamet_to_zarr import _read_file


class VaisalaMetTimestampTests(unittest.TestCase):
    def test_naive_source_timestamps_are_utc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vaisala_met_level0_06-07-2026.dat"
            path.write_text("timestamp,temp_C\n2026-07-06 12:26:46,9.5\n", encoding="utf-8")

            ds = _read_file(path)

        self.assertEqual(np.datetime_as_string(ds["time"].values[0], unit="s"), "2026-07-06T12:26:46")

    def test_offset_aware_source_timestamps_still_convert_to_utc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vaisala_met_level0_06-07-2026.dat"
            path.write_text("timestamp,temp_C\n2026-07-06T12:26:46+01:00,9.5\n", encoding="utf-8")

            ds = _read_file(path)

        self.assertEqual(np.datetime_as_string(ds["time"].values[0], unit="s"), "2026-07-06T11:26:46")


if __name__ == "__main__":
    unittest.main()
