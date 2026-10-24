import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from generate_power_soc_forecast import _atomic_write_archive
from power_archive_io import write_forecast_archive


class ArchiveEncodingTests(unittest.TestCase):
    def test_inherited_short_encoding_does_not_truncate_new_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'archive.zarr'
            old = xr.Dataset({'ForecastIdentityID': ('issue_time', ['nan'])},
                             coords={'issue_time': [0]})
            old.to_zarr(path, consolidated=True)
            with xr.open_zarr(path) as source:
                previous = source.load()
            full = 'forecast-identity-v1-' + 'a' * 64
            new = xr.Dataset({'ForecastIdentityID': ('issue_time', [full])},
                             coords={'issue_time': [1]})
            combined = xr.concat([previous, new], dim='issue_time')
            _atomic_write_archive(combined, path)
            with xr.open_zarr(path) as actual:
                self.assertEqual(actual.ForecastIdentityID.values.tolist(), ['nan', full])

    def test_failed_validation_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'archive.zarr'
            old = xr.Dataset({'identity': ('issue_time', ['original'])}, coords={'issue_time': [0]})
            write_forecast_archive(old, path)
            new = old.assign(identity=('issue_time', ['replacement']))
            with patch('power_archive_io.validate_archive_roundtrip', side_effect=ValueError('bad identity')):
                with self.assertRaises(ValueError):
                    write_forecast_archive(new, path)
            with xr.open_zarr(path) as actual:
                self.assertEqual(actual.identity.values.tolist(), ['original'])

    def test_datetime_precision_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'archive.zarr'
            times = np.array(['2026-09-21T21:17:51.123456789'], dtype='datetime64[ns]')
            original = xr.Dataset({'valid': ('issue_time', times)}, coords={'issue_time': times})
            write_forecast_archive(original, path)
            with xr.open_zarr(path) as actual:
                np.testing.assert_array_equal(actual.valid.values, times)


if __name__ == '__main__':
    unittest.main()
