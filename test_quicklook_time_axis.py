from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quicklook_time_axis import apply_quicklook_time_axis


class QuicklookTimeAxisTests(unittest.TestCase):
    def test_wide_time_limits_do_not_generate_hourly_ticks(self) -> None:
        start = pd.Timestamp("2024-01-01 00:00:00")
        end = pd.Timestamp("2026-07-05 15:29:00")

        fig, ax = plt.subplots()
        try:
            apply_quicklook_time_axis(
                ax,
                pd.DatetimeIndex([start, end]),
                x_limits=(start, end),
                max_ticks=8,
            )
            ticks = ax.xaxis.get_major_locator().tick_values(start.to_pydatetime(), end.to_pydatetime())
        finally:
            plt.close(fig)

        self.assertLess(len(ticks), 1000)
        self.assertLessEqual(len(ticks), 12)


if __name__ == "__main__":
    unittest.main()
