from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_model_evaluation_module():
    path = Path(__file__).resolve().parent / "model-evaluation.py"
    spec = importlib.util.spec_from_file_location("model_evaluation_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_partial_evidence_panel_uses_day_review_index(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day_root = tmp_path / "2026" / "07" / "06"
    (day_root / "dashboard").mkdir(parents=True)
    (day_root / "dashboard" / "day_review_index.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "partial_evidence_brief": {
                        "status": "partial_evidence_review_ready",
                        "headline": "Partial evidence is ready for review.",
                        "reviewable_item_count": 2,
                        "diagnostic_item_count": 1,
                        "blocked_item_count": 1,
                        "do_not_claim": ["Do not claim full v1 readiness."],
                        "items": [
                            {
                                "item_id": "direct_model_variable_partial_review",
                                "review_use": "partial",
                                "status": "partial_direct_science_summary_ready",
                                "statement": "ERA5 direct evidence is reviewable.",
                                "caveat": "No all-model ranking.",
                                "metrics": {"scored_variable_count": 8},
                            },
                            {
                                "item_id": "carra2_parent_model_input_wait",
                                "review_use": "blocked",
                                "status": "waiting_upstream_not_available",
                                "statement": "CARRA2 is unavailable.",
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    html = module._partial_evidence_brief_panel("2026-07-06")

    assert "Partial Evidence Review" in html
    assert "partial_evidence_review_ready" in html
    assert "direct_model_variable_partial_review" in html
    assert "carra2_parent_model_input_wait" in html
    assert "Do not claim full v1 readiness." in html


def test_parent_model_inputs_panel_shows_ready_era5_and_waiting_carra2(
    tmp_path,
) -> None:
    module = _load_model_evaluation_module()
    day_root = tmp_path / "2026" / "07" / "06"
    (day_root / "scorecards").mkdir(parents=True)
    (day_root / "dashboard").mkdir(parents=True)
    (day_root / "scorecards" / "model_input_availability.json").write_text(
        json.dumps(
            {
                "status": "partial_parent_model_inputs_ready",
                "ready_model_count": 1,
                "waiting_model_count": 1,
                "next_retry_after_utc": "2026-07-16T05:19:39Z",
                "next_actions": [
                    "Run downstream products for ready parent models.",
                    "Run guarded model-input retry after 2026-07-16T05:19:39Z.",
                ],
                "models": [
                    {
                        "model_id": "era5",
                        "status": "ready_full_target_day",
                        "target_full_day_available": True,
                        "target_available_hours": 24.0,
                        "latest_complete_day": "2026-07-06",
                        "latest_fetch_status": "downloaded",
                        "retry_trend_status": "single_record",
                    },
                    {
                        "model_id": "carra2",
                        "status": "waiting_upstream_not_available",
                        "target_full_day_available": False,
                        "target_available_hours": 0.0,
                        "latest_complete_day": "2026-04-29",
                        "latest_fetch_status": "blocked_upstream_not_available_yet",
                        "retry_trend_status": "stalled",
                        "recommended_retry_after_utc": "2026-07-16T05:19:39Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (day_root / "dashboard" / "day_review_index.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "comparison_readiness": {
                        "paths": {
                            "direct_model_variable_path": {"state": "ready"},
                            "full_les_virtual_observatory_path": {
                                "state": "blocked_missing_input"
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    html = module._parent_model_inputs_panel("2026-07-06")

    assert "Parent Model Inputs" in html
    assert "partial_parent_model_inputs_ready" in html
    assert "Ready parent models: era5." in html
    assert "Waiting parent models: carra2." in html
    assert "Direct path: ready" in html
    assert "CM1/CARRA2 virtual observatory: blocked_missing_input" in html
    assert "ready_full_target_day" in html
    assert "waiting_upstream_not_available" in html
