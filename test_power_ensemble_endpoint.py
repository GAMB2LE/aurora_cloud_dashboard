import unittest
import numpy as np
import pandas as pd
from generate_power_soc_ensemble import _clip_member_horizon


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series([100., 200., 300.], index=pd.date_range('2026-09-21', periods=3, freq='3h'))

    def test_partial_endpoint_uses_enclosing_interval_mean(self):
        end = pd.Timestamp('2026-09-21T03:08:53.731445557')
        result = _clip_member_horizon(self.series, self.series.index[0], end)
        self.assertEqual(result.index[-1], end)
        self.assertEqual(result.iloc[-1], 300.)
        self.assertEqual(result.iloc[-2], 200.)

    def test_exact_endpoint_is_unchanged(self):
        pd.testing.assert_series_equal(_clip_member_horizon(self.series, self.series.index[0], self.series.index[-1]), self.series)

    def test_missing_tail_is_not_extrapolated(self):
        with self.assertRaises(ValueError):
            _clip_member_horizon(self.series, self.series.index[0], pd.Timestamp('2026-09-21T07:00'))

    def test_invalid_final_interval_fails(self):
        self.series.iloc[-1] = np.nan
        with self.assertRaises(ValueError):
            _clip_member_horizon(self.series, self.series.index[0], pd.Timestamp('2026-09-21T05:00'))


if __name__ == '__main__':
    unittest.main()
