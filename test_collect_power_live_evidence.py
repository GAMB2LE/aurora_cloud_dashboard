import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import fcntl

from collect_power_live_evidence import collect, validate_ensemble_pair, discard_staging
from test_power_snapshot_fixture import snapshot, issue


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        snapshot(self.source, 0)
        self.target = self.root / 'candidate'

    def tearDown(self):
        self.tmp.cleanup()

    def test_enrolls_new_issues_once_without_recovery_archive_gate(self):
        first = collect(self.source, self.target)
        self.assertEqual(first['enrolled_this_run'], 1)
        self.assertEqual(first['missing_ensemble_pairs'], 1)
        snapshot(self.source, 6)
        second = collect(self.source, self.target)
        self.assertEqual(second['enrolled_this_run'], 1)
        self.assertEqual(second['captured_issues'], 2)
        third = collect(self.source, self.target)
        self.assertEqual(third['enrolled_this_run'], 0)
        self.assertEqual(third['candidate_status'], 'not_generated')
        self.assertFalse(third['cl61_actuation_enabled'])
        self.assertEqual(len((self.target / 'history.jsonl').read_text().splitlines()), 3)

    def test_rejects_modified_snapshot(self):
        path = next(self.source.iterdir()) / 'issue_manifest.json'
        data = json.loads(path.read_text())
        data['contentDigest'] = 'sha256:' + '0' * 64
        path.write_text(json.dumps(data))
        result = collect(self.source, self.target)
        self.assertEqual(result['captured_issues'], 0)
        self.assertEqual(result['rejected_snapshots'], 1)

    def test_rejects_source_overlap(self):
        with self.assertRaises(ValueError):
            collect(self.source, self.source / 'candidate')

    def test_rejects_output_symlink(self):
        self.target.mkdir()
        (self.target / 'history.jsonl').symlink_to(self.root / 'protected')
        with self.assertRaises(ValueError):
            collect(self.source, self.target)

    def test_inventory_bound(self):
        snapshot(self.source, 6)
        with self.assertRaises(ValueError):
            collect(self.source, self.target, maximum_snapshots=1)

    def test_concurrent_run_defers(self):
        self.target.mkdir()
        with (self.target / '.collector.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(collect(self.source, self.target)['reason'], 'collector_already_running')

    def test_pair_requires_exact_anchor_and_cycle(self):
        baseline = issue(0)
        ensemble = baseline.copy(deep=True)
        validate_ensemble_pair(baseline, ensemble)
        for field, value in [('initial_soc_pct', '91'), ('initial_soc_time', '2026-09-02'),
                             ('ecmwf_cycle_time', '2026-09-02'), ('initial_soc_pct', 'nan')]:
            altered = ensemble.copy(deep=True)
            altered.attrs[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_ensemble_pair(baseline, altered)

    def test_readonly_staging_cleanup_is_scoped(self):
        staging = self.root / '.ensemble-test'
        nested = staging / 'data'
        nested.mkdir(parents=True)
        (nested / 'chunk').write_text('test')
        nested.chmod(0o555)
        discard_staging(staging, self.root)
        self.assertFalse(staging.exists())
        with self.assertRaises(ValueError):
            discard_staging(self.source, self.root)


if __name__ == '__main__':
    unittest.main()
