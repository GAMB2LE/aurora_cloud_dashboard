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


def test_goal_progress_panel_shows_v1_tracks(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day_root = tmp_path / "2026" / "07" / "06"
    (day_root / "dashboard").mkdir(parents=True)
    (day_root / "dashboard" / "day_review_index.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "goal_progress": {
                        "status": "partial_review_ready_with_production_blocks",
                        "goal_complete": False,
                        "complete_track_count": 1,
                        "reviewable_track_count": 4,
                        "diagnostic_track_count": 2,
                        "production_blocked_track_count": 3,
                        "policy": "Do not claim production completion yet.",
                        "tracks": [
                            {
                                "track_id": "direct_model_variable_evaluation",
                                "title": "Direct model-variable evaluation",
                                "completion_state": "partial_review_ready",
                                "comparison_state": "ready",
                                "reviewable_now": True,
                                "production_ready": False,
                                "diagnostic_only": False,
                                "next_action": "Keep ERA5 conclusions partial.",
                            },
                            {
                                "track_id": "full_les_virtual_observatory",
                                "title": "Full LES virtual observatory",
                                "completion_state": "diagnostic_only",
                                "comparison_state": "blocked_missing_input",
                                "reviewable_now": True,
                                "production_ready": False,
                                "diagnostic_only": True,
                                "next_action": "Stage CARRA2 forcing for CM1.",
                            },
                            {
                                "track_id": "cloud_seb_process_understanding",
                                "title": "Cloud and SEB process diagnostics",
                                "completion_state": "diagnostic_only",
                                "comparison_state": "blocked_regime_mismatch",
                                "reviewable_now": True,
                                "production_ready": False,
                                "diagnostic_only": True,
                                "next_action": "Recover comparable regimes.",
                            },
                            {
                                "track_id": "daily_bundle_provenance_readiness",
                                "title": "Daily bundle and provenance",
                                "completion_state": "complete",
                                "comparison_state": None,
                                "reviewable_now": True,
                                "production_ready": True,
                                "diagnostic_only": False,
                                "next_action": "Keep records current.",
                            },
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    html = module._goal_progress_panel("2026-07-06")

    assert "Goal Progress" in html
    assert "partial_review_ready_with_production_blocks" in html
    assert "Direct model-variable evaluation" in html
    assert "Full LES virtual observatory" in html
    assert "Cloud and SEB process diagnostics" in html
    assert "Daily bundle and provenance" in html
    assert "blocked_missing_input" in html
    assert "blocked_regime_mismatch" in html
    assert "Do not claim production completion yet." in html


def test_surface_met_row_uses_full_day_virtual_instrument_scorecard(
    tmp_path,
) -> None:
    module = _load_model_evaluation_module()
    day = "2026-07-06"
    day_root = tmp_path / "2026" / "07" / "06"
    scorecard_root = day_root / "scorecards"
    plot_root = day_root / "plots"
    dashboard_root = day_root / "dashboard"
    scorecard_root.mkdir(parents=True)
    plot_root.mkdir(parents=True)
    dashboard_root.mkdir(parents=True)
    plot = plot_root / "surface_met_cm1_era5_full_day_20260706.svg"
    plot.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'></svg>",
        encoding="utf-8",
    )
    (scorecard_root / "surface_met_cm1_era5_full_day.json").write_text(
        json.dumps(
            {
                "status": "scored_diagnostic_vertical_support_mismatch",
                "scorecard": {
                    "status": "scored_diagnostic_vertical_support_mismatch",
                    "comparison_readiness": (
                        "diagnostic_vertical_support_mismatch"
                    ),
                    "output_plot": str(plot),
                    "variables": {
                        "temperature": {
                            "status": (
                                "scored_diagnostic_vertical_support_mismatch"
                            ),
                            "score": {
                                "sample_count": 89,
                                "bias": -0.405,
                                "rmse": 1.35,
                                "correlation": -0.708,
                            },
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (dashboard_root / "day_review_index.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "review_tracks": {
                        "tracks": {
                            "full_les_virtual_observatory": {
                                "status": (
                                    "diagnostic_full_day_virtual_observatory_ready"
                                ),
                                "diagnostic_ready": True,
                                "can_review_now": True,
                                "production_ready": False,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    spec = next(
        item
        for item in module.INSTRUMENT_COMPARISON_SPECS
        if item["instrument"] == "Surface met"
    )
    row = module._instrument_comparison_row(day, spec)
    gallery = module.render_scorecard_gallery(day, "Surface met")

    assert row["scorecard"] == "surface_met_cm1_era5_full_day"
    assert row["model"] == "CM1 full LES virtual instrument"
    assert row["status"] == "scored_diagnostic_vertical_support_mismatch"
    assert row["caveat"] == "diagnostic_only"
    assert row["path_readiness"] == (
        "diagnostic_full_day_virtual_observatory_ready"
    )
    assert row["path_review_ready"] is True
    assert row["path_production_ready"] is False
    assert row["valid"] == 89
    assert row["bias"] == "-0.405"
    assert row["rmse"] == "1.35"
    assert row["correlation"] == "-0.708"
    assert "CM1 full LES surface meteorology" in gallery
    assert "data:image/svg+xml;base64," in gallery


def test_hogan_cloud_fraction_row_and_svg_gallery(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day = "2026-07-06"
    day_root = tmp_path / "2026" / "07" / "06"
    scorecard_root = day_root / "scorecards"
    plot_root = day_root / "plots"
    scorecard_root.mkdir(parents=True)
    plot_root.mkdir(parents=True)
    plot_path = plot_root / "cloud_fraction_hogan_20260706.svg"
    plot_path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'></svg>",
        encoding="utf-8",
    )
    (scorecard_root / "cloud_fraction_hogan.json").write_text(
        json.dumps(
            {
                "status": "scored_with_qc_caveat",
                "primary_comparison_id": "model_cf_cirrus__vs__cf_V_adv",
                "plot_file": str(plot_path),
                "qc_caveats": [
                    "High-rain exclusion is pending collocated surface rain rate."
                ],
                "comparisons": [
                    {
                        "comparison_id": "model_cf_cirrus__vs__cf_V_adv",
                        "score": {
                            "status": "scored",
                            "sample_qc": {"valid_pair_count": 88},
                            "support_compliance": {
                                "headline_eligibility": {
                                    "status": (
                                        "method_support_ready_with_high_rain_qc_caveat"
                                    ),
                                    "ranking_policy": "blocked_pending_high_rain_qc",
                                }
                            },
                            "primary_threshold_score": {
                                "probability_of_detection": 0.8667,
                                "false_alarm_ratio": 0.8194,
                                "critical_success_index": 0.1757,
                                "heidke_skill_score_height_aware": {
                                    "status": "scored",
                                    "value": 0.01168,
                                },
                                "log_odds_ratio_climatology_corrected": {
                                    "status": "scored",
                                    "value": 0.226,
                                    "standard_error": 0.8157,
                                },
                                "symmetric_extreme_dependency_score": {
                                    "status": "scored",
                                    "value": 0.0147,
                                    "standard_error": 0.0537,
                                },
                            },
                            "continuous_skill": {
                                "mean_absolute_error_skill_score": {
                                    "status": "scored",
                                    "value": -0.00512,
                                }
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    module._direct_model_variable_readiness = lambda _day: {
        "status": "partial_review_ready",
        "production_ready": False,
        "current_work": {
            "can_review_now": True,
            "can_rank_all_models_now": False,
        },
        "usable_models": ["era5"],
        "missing_models": ["carra2"],
    }
    rows = module.build_instrument_catalog([day])
    row = next(
        item
        for item in rows
        if item["scorecard"] == "cloud_fraction_hogan"
    )

    assert row["valid"] == 88
    assert row["hss"] == "0.0117"
    assert row["log_odds"] == "0.226 (SE 0.816)"
    assert row["seds"] == "0.0147 (SE 0.0537)"
    assert row["maess"] == "-0.00512"
    assert row["path_review_ready"] is True
    assert row["path_model_ranking_ready"] is False
    assert row["caveat"] == "diagnostic_only_high_rain_qc"
    assert "blocked_pending_high_rain_qc" in row["note"]

    gallery = module.render_scorecard_gallery(day, "Cloudnet CF")
    assert "Hogan CF verification: ERA5" in gallery
    assert "data:image/svg+xml;base64," in gallery
