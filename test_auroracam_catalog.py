from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import auroracam_catalog


class AuroracamCatalogTests(unittest.TestCase):
    def _write_image(self, root: Path, camera: str, day: str, hhmm: str) -> Path:
        path = root / camera / day / f"{camera}_{day}_{hhmm}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
        return path

    def test_parse_new_camera_filename(self) -> None:
        path = Path("end-south-array-cam_2026-07-07_11-48.jpg")

        self.assertEqual(auroracam_catalog.camera_from_filename(path), "end-south-array-cam")
        self.assertEqual(auroracam_catalog.parse_timestamp(path).strftime("%Y-%m-%d %H:%M"), "2026-07-07 11:48")

    def test_old_mx4_filename_is_ignored(self) -> None:
        self.assertIsNone(auroracam_catalog.parse_timestamp(Path("188-11-48-12.jpg")))

    def test_latest_records_and_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_image(root, "end-south-array-cam", "2026-07-07", "11-48")
            latest_path = self._write_image(root, "end-south-array-cam", "2026-07-07", "11-49")
            self._write_image(root, "radar-cam", "2026-07-06", "09-30")

            self.assertEqual(auroracam_catalog.available_days(root), ["2026-07-06", "2026-07-07"])
            latest = auroracam_catalog.latest_records(root)

        self.assertEqual(latest["end-south-array-cam"].filename, latest_path.name)
        self.assertEqual(latest["radar-cam"].day_utc, "2026-07-06")

    def test_hourly_representative_prefers_half_past(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_image(root, "fence-post-cam", "2026-07-07", "10-05")
            target_path = self._write_image(root, "fence-post-cam", "2026-07-07", "10-31")
            self._write_image(root, "fence-post-cam", "2026-07-07", "10-55")

            rows = auroracam_catalog.representative_hourly_records(root, "fence-post-cam", "2026-07-07")

        self.assertEqual(rows[10].filename, target_path.name)

    def test_retired_camera_is_hidden_but_remains_in_archive_index(self) -> None:
        from index_auroracam_zarr import _records_to_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retired = self._write_image(root, "mid-south-array-cam", "2026-10-02", "09-55")
            self._write_image(root, "radar-cam", "2026-10-22", "00-01")
            before = retired.read_bytes()

            self.assertNotIn("mid-south-array-cam", auroracam_catalog.ACTIVE_AURORACAM_CAMERAS)
            self.assertEqual(auroracam_catalog.active_available_days(root), ["2026-10-22"])
            self.assertEqual(auroracam_catalog.available_days(root), ["2026-10-02", "2026-10-22"])
            records = list(auroracam_catalog.iter_image_records(root))
            archive = _records_to_dataset(records, root)

            self.assertIn(b"mid-south-array-cam", archive.camera_id.values)
            self.assertIn(b"mid-south-array-cam", archive.camera.values)
            self.assertEqual(auroracam_catalog.latest_record(root, "mid-south-array-cam").filename, retired.name)
            self.assertEqual(retired.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
