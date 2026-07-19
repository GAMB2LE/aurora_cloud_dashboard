import unittest
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from data_services import WindowRequest, coarsen_targets, prepare_dataset_window


def _valid_time_mask(values):
    return ~np.isnat(values)


class DataServicesTests(unittest.TestCase):
    def setUp(self):
        times = np.arange(
            np.datetime64("2026-07-19T00:00"),
            np.datetime64("2026-07-19T04:00"),
            np.timedelta64(1, "m"),
        )
        heights = np.arange(0, 2000, 10)
        self.dataset = xr.Dataset(
            {"value": (("time", "range"), np.ones((times.size, heights.size)))},
            coords={"time": times, "range": heights},
        )

    def test_invalid_window_is_empty(self):
        start = datetime(2026, 7, 19, 2)
        metrics = {}
        result = prepare_dataset_window(
            self.dataset,
            WindowRequest(start, start),
            valid_time_mask=_valid_time_mask,
            perf=metrics,
        )
        self.assertEqual(result.sizes, {})
        self.assertEqual(metrics["status"], "invalid_window")

    def test_no_match_is_empty(self):
        start = datetime(2026, 7, 20)
        result = prepare_dataset_window(
            self.dataset,
            WindowRequest(start, start + timedelta(hours=1)),
            valid_time_mask=_valid_time_mask,
        )
        self.assertEqual(result.sizes, {})

    def test_height_window_is_applied(self):
        start = datetime(2026, 7, 19)
        result = prepare_dataset_window(
            self.dataset,
            WindowRequest(
                start,
                start + timedelta(hours=1),
                bottom_m=500,
                top_m=1000,
            ),
            valid_time_mask=_valid_time_mask,
        )
        self.assertGreaterEqual(float(result["range"].min()), 500)
        self.assertLessEqual(float(result["range"].max()), 1000)

    def test_coarse_render_reduces_time_samples(self):
        start = datetime(2026, 7, 19)
        metrics = {}
        result = prepare_dataset_window(
            self.dataset,
            WindowRequest(
                start,
                start + timedelta(hours=4),
                render_quality="coarse",
            ),
            valid_time_mask=_valid_time_mask,
            perf=metrics,
        )
        self.assertLess(result.sizes["time"], self.dataset.sizes["time"])
        self.assertEqual(metrics["status"], "ok")

    def test_short_window_targets_retain_detail(self):
        self.assertEqual(
            coarsen_targets(timedelta(hours=1), 500),
            (1, 1200, 400),
        )


if __name__ == "__main__":
    unittest.main()
