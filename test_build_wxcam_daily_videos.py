from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from build_wxcam_daily_videos import (
    _existing_thumbnail_hours,
    _representative_hourly_images,
    _thumbnail_build_before,
    _unpublished_hourly_images,
)


class WxcamDailyVideoTests(unittest.TestCase):
    def test_thumbnail_cutoff_includes_an_hour_only_after_settle_grace(self) -> None:
        self.assertEqual(
            _thumbnail_build_before(datetime(2026, 8, 18, 14, 9, tzinfo=timezone.utc)),
            datetime(2026, 8, 18, 13, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            _thumbnail_build_before(datetime(2026, 8, 18, 14, 10, tzinfo=timezone.utc)),
            datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            _thumbnail_build_before(datetime(2026, 8, 19, 0, 10, tzinfo=timezone.utc)),
            datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc),
        )

    def test_existing_thumbnail_hours_ignore_malformed_and_wrong_day_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "HDR_20260818_023000.jpg").touch()
            (directory / "HDR_20260818_141539.jpg").touch()
            (directory / "HDR_20260817_143000.jpg").touch()
            (directory / "sample.jpg").touch()

            hours = _existing_thumbnail_hours(directory, "20260818")

        self.assertEqual(hours, {2, 14})

    def test_existing_hour_is_not_republished_when_a_late_candidate_arrives(self) -> None:
        hourly_images = {
            13: Path("HDR_20260818_133000.jpg"),
            14: Path("HDR_20260818_143000.jpg"),
        }

        unpublished = _unpublished_hourly_images(hourly_images, {14})

        self.assertEqual(unpublished, {13: Path("HDR_20260818_133000.jpg")})

    def test_hourly_thumbnails_exclude_the_in_progress_utc_hour(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day = Path(tmp) / "20260818"
            hour_13 = day / "13"
            hour_14 = day / "14"
            hour_13.mkdir(parents=True)
            hour_14.mkdir(parents=True)
            (hour_13 / "HDR_20260818_132500.jpg").touch()
            (hour_13 / "HDR_20260818_133000.jpg").touch()
            for minute in range(0, 31, 5):
                (hour_14 / f"HDR_20260818_14{minute:02d}39.jpg").touch()

            during_hour = _representative_hourly_images(
                day,
                "HDR_*.jpg",
                before=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
            )
            after_hour = _representative_hourly_images(
                day,
                "HDR_*.jpg",
                before=datetime(2026, 8, 18, 15, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(list(during_hour), [13])
        self.assertEqual(during_hour[13].name, "HDR_20260818_133000.jpg")
        self.assertEqual(list(after_hour), [13, 14])
        self.assertEqual(after_hour[14].name, "HDR_20260818_143039.jpg")


if __name__ == "__main__":
    unittest.main()
