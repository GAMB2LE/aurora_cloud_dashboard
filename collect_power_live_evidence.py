#!/usr/bin/env python3
"""Enroll checksum-verified issues without mutating a forecast or its archive.

Capture is not candidate generation, skill evidence, or promotion. Missing or
unmatched ensemble inputs stay explicitly blocked. Existing captures are never
overwritten. Only the supplied candidate root is writable.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import pandas as pd
import xarray as xr

from generate_power_soc_forecast import _issue_snapshot_tree_digest
from repair_power_forecast_archive import verified_issue_snapshot
from power_v12_ensemble import baseline_ensemble_signature


def validate_ensemble_pair(baseline, ensemble):
    for name in ('initial_soc_time', 'ecmwf_cycle_time'):
        if not baseline.attrs.get(name) or not ensemble.attrs.get(name):
            raise ValueError('Missing paired ' + name)
        if pd.Timestamp(baseline.attrs[name]) != pd.Timestamp(ensemble.attrs[name]):
            raise ValueError('Different paired ' + name)
    a, b = float(baseline.attrs['initial_soc_pct']), float(ensemble.attrs['initial_soc_pct'])
    if not (0 <= a <= 100 and 0 <= b <= 100 and abs(a - b) <= 1e-6):
        raise ValueError('Different or invalid SOC anchor')
    if pd.Timestamp(ensemble.time.values[-1]) < pd.Timestamp(baseline.time.values[-1]):
        raise ValueError('Ensemble does not cover paired horizon')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def collect(issues_root: Path, candidate_root: Path, ensemble_zarr: Path | None = None,
            *, maximum_snapshots=10000):
    source, root = issues_root.resolve(), candidate_root.resolve()
    if issues_root.is_symlink() or candidate_root.is_symlink():
        raise ValueError('Issue and candidate roots must be direct directories')
    inputs = [source] + ([ensemble_zarr.resolve()] if ensemble_zarr else [])
    if any(root == p or root in p.parents or p in root.parents for p in inputs):
        raise ValueError('Candidate output overlaps a read-only input')
    if maximum_snapshots < 1:
        raise ValueError('maximum_snapshots must be positive')
    directories = sorted(source.iterdir())
    if len(directories) > maximum_snapshots:
        raise ValueError('Issue inventory exceeds bound')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('.collector.lock', 'history.jsonl', 'status.json', 'status.json.tmp', 'issues'):
        if (root / name).is_symlink():
            raise ValueError('Collector output may not be a symbolic link')
    with (root / '.collector.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'deferred', 'reason': 'collector_already_running'}
        ensemble_anchor = None
        if ensemble_zarr and ensemble_zarr.exists():
            with xr.open_zarr(ensemble_zarr.resolve(), chunks={}) as ds:
                if ds.sizes.get('member', 0) > 100 or ds.sizes.get('time', 0) > 1000:
                    raise ValueError('Only bounded site-level ensembles are permitted')
                ensemble_anchor = pd.Timestamp(ds.attrs.get('initial_soc_time'))
        enrolled = rejected = ensembles = 0
        reasons = {}
        destinations = root / 'issues'
        destinations.mkdir(exist_ok=True)
        for directory in directories:
            if not directory.is_dir():
                continue
            try:
                _, evidence = verified_issue_snapshot(directory)
            except Exception as exc:
                rejected += 1
                reason = str(exc)
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            key = evidence['content_digest'].removeprefix('sha256:')
            target = destinations / key
            if target.is_symlink():
                raise ValueError('Captured issue may not be a symbolic link')
            if not target.exists():
                staging = Path(tempfile.mkdtemp(prefix='.capture-', dir=root))
                try:
                    shutil.copytree(directory, staging / 'baseline')
                    _, copied = verified_issue_snapshot(staging / 'baseline')
                    if copied['content_digest'] != evidence['content_digest']:
                        raise ValueError('Issue changed during capture')
                    atomic_json(staging / 'manifest.json', {
                        'schema_version': 1, 'status': 'captured', **evidence,
                        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                        'authority': 'evidence_only', 'cl61_actuation_enabled': False,
                        'candidate_status': 'not_generated',
                        'issue_time_feature_status': 'not_captured_by_this_collector',
                    })
                    staging.rename(target)
                    enrolled += 1
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            # A later invocation may fill a missing ensemble, but never replace
            # a comparator already admitted for this immutable issue.
            final_ensemble = target / 'ensemble'
            if final_ensemble.is_symlink():
                raise ValueError('Captured ensemble may not be a symbolic link')
            if (ensemble_anchor is not None and pd.Timestamp(evidence['issue_time']) == ensemble_anchor
                    and ensemble_zarr and not final_ensemble.exists()):
                staging = Path(tempfile.mkdtemp(prefix='.ensemble-', dir=root))
                try:
                    resolved = ensemble_zarr.resolve()
                    before = _issue_snapshot_tree_digest(resolved)[0]
                    shutil.copytree(resolved, staging / 'forecast.zarr')
                    after = _issue_snapshot_tree_digest(resolved)[0]
                    copied = _issue_snapshot_tree_digest(staging / 'forecast.zarr')[0]
                    if before != after or before != copied:
                        raise ValueError('Ensemble changed during capture')
                    with xr.open_zarr(target / 'baseline' / 'forecast.zarr', chunks={}) as d:
                        baseline = d.load()
                    with xr.open_zarr(staging / 'forecast.zarr', chunks={}) as d:
                        ensemble = d.load()
                    # Validate anchors and cycle even for a matching time grid.
                    validate_ensemble_pair(baseline, ensemble)
                    signature = baseline_ensemble_signature(ensemble)
                    atomic_json(staging / 'manifest.json', {
                        'status': 'captured_matching_anchor_cycle', 'baseline_content_digest': evidence['content_digest'],
                        'grid_alignment_status': 'requires_candidate_validation',
                        'ensemble_content_digest': 'sha256:' + copied,
                        'ensemble_signature': signature,
                    })
                    staging.rename(final_ensemble)
                    ensembles += 1
                except (ValueError, KeyError, OSError) as exc:
                    # A current ensemble ordinarily cannot match old issues.
                    reason = 'ensemble_unavailable_or_unmatched:' + str(exc)
                    reasons[reason] = reasons.get(reason, 0) + 1
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
        total = sum(p.is_dir() for p in destinations.iterdir())
        paired = sum((p / 'ensemble' / 'manifest.json').is_file() for p in destinations.iterdir())
        status = {'schema_version': 1, 'status': 'capture_complete',
                  'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'enrolled_this_run': enrolled, 'ensembles_this_run': ensembles,
                  'captured_issues': total, 'captured_ensemble_pairs': paired,
                  'missing_ensemble_pairs': total - paired,
                  'rejected_snapshots': rejected, 'reasons': reasons,
                  'candidate_status': 'not_generated', 'promotion_status': 'not_eligible',
                  'authority': 'evidence_only', 'cl61_actuation_enabled': False}
        atomic_json(root / 'status.json', status)
        with (root / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(status, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issues-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--ensemble-zarr', type=Path)
    parser.add_argument('--maximum-snapshots', type=int, default=10000)
    print(json.dumps(collect(**vars(parser.parse_args())), sort_keys=True))
