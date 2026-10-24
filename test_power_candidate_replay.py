from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from generate_power_soc_v12_candidate import LANES
from repair_power_forecast_archive import recover_archive, verified_issue_snapshot
from run_power_candidate_replay import run_replay
from test_power_archive_integrity import snapshot


class CandidateReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "issues"
        for hour in (12, 0, 6):
            snapshot(self.sources, hour)
        self.archive = self.root / "recovered.zarr"
        recover_archive(self.sources, self.archive, self.root / "recovery.json")
        self.config = self.root / "physical.json"
        self.config.write_text("{}")
        self.output = self.root / "candidate"
        self.calls = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self) -> dict:
        return {
            "issues_root": self.sources, "recovered_archive_zarr": self.archive,
            "candidate_root": self.output, "power_zarr": self.root / "power.zarr",
            "physical_config": self.config,
        }

    def candidate(self, **kwargs):
        self.calls.append(kwargs)
        _, evidence = verified_issue_snapshot(kwargs["baseline_issue_zarr"].parent)
        work_root = kwargs["candidate_root"]
        paths = {lane: work_root / "lanes" / lane / "forecast.zarr" for lane in LANES}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "complete", "baseline_issue_content_digest": evidence["content_digest"],
            "baseline_publication_signature": evidence["publication_signature"],
        }
        (work_root / "status.json").write_text(json.dumps(status))
        return paths

    def history(self):
        return [json.loads(line) for line in (self.output / "replay_history.jsonl").read_text().splitlines()]

    @patch("run_power_candidate_replay._model_evaluation_state", return_value="inactive")
    def test_chronological_bounded_replay_skips_only_successes(self, _state) -> None:
        with patch("run_power_candidate_replay.run_candidate", side_effect=self.candidate):
            first = run_replay(**self.arguments(), max_issues=1)
            second = run_replay(**self.arguments(), max_issues=1)
        self.assertEqual(first["processed"], 1)
        self.assertEqual(second["processed"], 1)
        self.assertTrue(str(self.calls[0]["baseline_issue_zarr"]).endswith("test-0/forecast.zarr"))
        self.assertTrue(str(self.calls[1]["baseline_issue_zarr"]).endswith("test-6/forecast.zarr"))
        self.assertEqual(self.calls[0]["baseline_archive_zarr"], self.archive)
        self.assertIsNone(self.calls[0]["baseline_ensemble_zarr"])
        self.assertFalse(self.calls[0]["public_source_manifest_root"].exists())
        self.assertEqual(sum(event["status"] == "succeeded" for event in self.history()), 2)

    def test_active_or_unknown_model_evaluation_defers_without_processing(self) -> None:
        with patch("run_power_candidate_replay.run_candidate") as candidate:
            for state in ("activating", "active", "unknown"):
                with patch("run_power_candidate_replay._model_evaluation_state", return_value=state):
                    result = run_replay(**self.arguments())
                    self.assertEqual(result["status"], "deferred")
            candidate.assert_not_called()
        self.assertFalse(any(event["status"] == "succeeded" for event in self.history()))

    @patch("run_power_candidate_replay._model_evaluation_state", return_value="inactive")
    def test_failure_remains_retryable_and_stops_before_next_issue(self, _state) -> None:
        with patch("run_power_candidate_replay.run_candidate", side_effect=ValueError("pair mismatch")):
            failed = run_replay(**self.arguments(), max_issues=3)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["processed"], 0)
        with patch("run_power_candidate_replay.run_candidate", side_effect=self.candidate):
            retried = run_replay(**self.arguments(), max_issues=1)
        self.assertEqual(retried["processed"], 1)
        started = [event for event in self.history() if event["status"] == "started"]
        self.assertEqual(started[0]["replay_key"], started[1]["replay_key"])

    @patch("run_power_candidate_replay._model_evaluation_state", return_value="inactive")
    def test_implementation_or_configuration_change_replays_same_issue(self, _state) -> None:
        with patch("run_power_candidate_replay.run_candidate", side_effect=self.candidate):
            with patch("run_power_candidate_replay.forecast_implementation_digest", return_value="sha256:" + "a" * 64):
                first = run_replay(**self.arguments())
            with patch("run_power_candidate_replay.forecast_implementation_digest", return_value="sha256:" + "b" * 64):
                changed = run_replay(**self.arguments())
                self.config.write_text('{"changed":true}')
                config_changed = run_replay(**self.arguments())
        self.assertEqual(first["processed"], changed["processed"])
        self.assertEqual(config_changed["processed"], 1)
        self.assertNotEqual(first["candidate_work_root"], changed["candidate_work_root"])
        self.assertNotEqual(changed["candidate_work_root"], config_changed["candidate_work_root"])
        self.assertEqual(len({call["baseline_issue_zarr"] for call in self.calls}), 1)

    @patch("run_power_candidate_replay._model_evaluation_state", return_value="inactive")
    def test_runner_deferral_is_not_a_success(self, _state) -> None:
        with patch("run_power_candidate_replay.run_candidate", return_value={}):
            result = run_replay(**self.arguments())
        self.assertEqual(result["status"], "deferred")
        with patch("run_power_candidate_replay.run_candidate", side_effect=self.candidate):
            retry = run_replay(**self.arguments())
        self.assertEqual(retry["processed"], 1)
        self.assertTrue(str(self.calls[0]["baseline_issue_zarr"]).endswith("test-0/forecast.zarr"))

    def test_evaluator_becoming_active_between_issues_defers_queue(self) -> None:
        with patch("run_power_candidate_replay._model_evaluation_state", side_effect=["inactive", "inactive", "active"]):
            with patch("run_power_candidate_replay.run_candidate", side_effect=self.candidate):
                result = run_replay(**self.arguments(), max_issues=3)
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
