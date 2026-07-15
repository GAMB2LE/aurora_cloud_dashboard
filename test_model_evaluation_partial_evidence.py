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
