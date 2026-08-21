from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import wxcam_catalog


class WxcamCatalogTests(TestCase):
    def test_readonly_open_retries_immutable_after_generic_sqlite_failure(self) -> None:
        connection = MagicMock(spec=sqlite3.Connection)
        connection.execute.return_value.fetchone.return_value = ("images",)

        with patch.object(
            wxcam_catalog.sqlite3,
            "connect",
            side_effect=[sqlite3.OperationalError("unable to open database file"), connection],
        ) as connect:
            result = wxcam_catalog._open_readonly_catalog(Path("/mirror/wxcam_catalog.sqlite"))

        self.assertIs(result, connection)
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(
            connect.call_args_list[1].args[0],
            "file:/mirror/wxcam_catalog.sqlite?mode=ro&immutable=1",
        )

    def test_readonly_open_can_read_wal_database_without_sidecars(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wxcam_catalog.sqlite"
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO images DEFAULT VALUES")
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            Path(f"{path}-wal").unlink(missing_ok=True)
            Path(f"{path}-shm").unlink(missing_ok=True)
            with wxcam_catalog.open_catalog(path, readonly=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

        self.assertEqual(count, 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
