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
    assert "GFS-forced CM1 virtual observatory: blocked_missing_input" in html
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
    day = "2026-08-01"
    day_root = tmp_path / "2026" / "08" / "01"
    scorecard_root = day_root / "scorecards"
    plot_root = day_root / "plots"
    dashboard_root = day_root / "dashboard"
    scorecard_root.mkdir(parents=True)
    plot_root.mkdir(parents=True)
    dashboard_root.mkdir(parents=True)
    plot = plot_root / "surface_met_cm1_gfs_full_day_v1_20260801.svg"
    plot.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'></svg>",
        encoding="utf-8",
    )
    (scorecard_root / "surface_met_cm1_gfs_full_day_v1.json").write_text(
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
        if item["instrument"] == "Surface met" and item["model_group"] == "surface"
    )
    row = module._instrument_comparison_row(day, spec)
    gallery = module.render_scorecard_gallery(day, "Surface met")

    assert row["scorecard"] == "surface_met_cm1_gfs_full_day_v1"
    assert row["model"] == "GFS-forced CM1 virtual instrument"
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
    assert "GFS-forced CM1 surface meteorology" in gallery
    assert "data:image/svg+xml;base64," in gallery


def test_hogan_cloud_fraction_row_and_svg_gallery(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day = "2026-08-01"
    day_root = tmp_path / "2026" / "08" / "01"
    scorecard_root = day_root / "scorecards"
    plot_root = day_root / "plots"
    scorecard_root.mkdir(parents=True)
    plot_root.mkdir(parents=True)
    plot_path = plot_root / "direct_scorecard_20260801.svg"
    plot_path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'></svg>",
        encoding="utf-8",
    )
    (scorecard_root / "direct_scorecard.json").write_text(
        json.dumps({"status": "scored", "output_svg": str(plot_path)}),
        encoding="utf-8",
    )
    (scorecard_root / "direct_gfs.json").write_text(
        json.dumps(
            {
                "scorecard": {
                    "status": "scored_partial_model_input_coverage",
                    "model_id": "gfs",
                    "metrics": {
                        "variable_scores": [
                            {
                                "variable_name": "cloud_fraction",
                                "status": "scored",
                                "notes": ["Mapped to model support."],
                                "binary_occurrence": {
                                    "sample_count": 88,
                                    "probability_of_detection": 0.8667,
                                    "false_alarm_ratio": 0.8194,
                                    "critical_success_index": 0.1757,
                                },
                                "continuous": {"sample_count": 88},
                                "hogan_2009_cloud_fraction": {
                                    "status": "scored",
                                    "support_compliance": {
                                        "headline_eligibility": {
                                            "status": "diagnostic_only_incomplete_hogan_support",
                                            "ranking_policy": "excluded_from_hogan_headline_ranking",
                                            "support_blockers": [
                                                "missing_radar_sensitivity_cirrus_filter",
                                                "high_rain_qc_not_applied_missing_categorize",
                                            ],
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
                        ]
                    },
                }
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
        "usable_models": ["gfs"],
        "missing_models": ["icon"],
    }
    rows = module.build_instrument_catalog([day])
    row = next(
        item
        for item in rows
        if item["scorecard"] == "gfs_cloud_fraction"
    )

    assert row["valid"] == 88
    assert row["hss"] == "0.0117"
    assert row["log_odds"] == "0.226 (SE 0.816)"
    assert row["seds"] == "0.0147 (SE 0.0537)"
    assert row["maess"] == "-0.00512"
    assert row["path_review_ready"] is True
    assert row["path_model_ranking_ready"] is False
    assert row["caveat"] == "diagnostic_only_hogan_support"
    assert "missing_radar_sensitivity_cirrus_filter" in row["note"]

    gallery = module.render_scorecard_gallery(day, "Cloudnet CF")
    assert "IFS / GFS / ICON direct cloud evaluation" in gallery
    assert "data:image/svg+xml;base64," in gallery


def test_evaluation_schematic_shows_current_two_paths() -> None:
    module = _load_model_evaluation_module()

    html = module._evaluation_schematic()

    assert "IFS / GFS / ICON MMDF" in html
    assert "MODF observations at native cadence" in html
    assert "GFS boundary forcing" in html
    assert "CM1 nonhydrostatic full LES" in html
    assert "ALCF CL61" in html
    assert "ERA5" not in html
    assert "diagnostic lidar" not in html


def test_iceland_panel_uses_collocated_deployment_and_production_overlap(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day_root = tmp_path / "2026" / "08" / "01"
    (day_root / "scorecards").mkdir(parents=True)
    (day_root / "bundle.json").write_text(
        json.dumps(
            {
                "day": "2026-08-01",
                "site_name": "AURORA Iceland deployment",
                "readiness": {
                    "model_input_availability": {
                        "models": {
                            "ifs": {"status": "downloaded", "target_available_hours": 24.0},
                            "gfs": {"status": "downloaded", "target_available_hours": 24.0},
                            "icon": {"status": "partial_day", "target_available_hours": 12.0},
                        }
                    },
                    "v1_path_summary": {
                        "paths": {
                            "full_les_virtual_observatory_path": {
                                "status": "running",
                                "diagnostic_review": {
                                    "artifact_statuses": {
                                        "cm1_gfs_full_day_launch": "running"
                                    }
                                },
                            }
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (day_root / "status.json").write_text(
        json.dumps({"day": "2026-08-01", "status": "running"}),
        encoding="utf-8",
    )
    (day_root / "scorecards" / "cloudnet_instrument_coverage.json").write_text(
        json.dumps(
            {
                "status": "partial_day_coverage",
                "production_common_overlap": {
                    "hours": 23.900556,
                    "status": "partial_day",
                },
                "site": {
                    "name": "AURORA Iceland deployment",
                    "deployments": [
                        {
                            "colocated_instruments": [
                                "rpgfmcw94",
                                "cl61",
                                "hatpro",
                                "vaisalamet",
                                "asfs_logger",
                                "asfs_fast_sonic",
                                "asfs_fast_gas",
                            ]
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    html = module._iceland_readiness_panel()

    assert "August 1 Iceland Case" in html
    assert "collocated instruments" in html
    assert ">7<" in html
    assert "23.900556" in html
    assert "CL61 backscatter" in html
    assert "production-eligible" in html
    assert "depolarisation remains observation-only" in html


def test_cloudnet_backbone_panel_selects_requested_day(tmp_path) -> None:
    module = _load_model_evaluation_module()
    provenance = tmp_path / "2026" / "08" / "01" / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "cloudnet_categorize_run.json").write_text(
        json.dumps(
            {
                "common_overlap": {
                    "hours": 22.936667,
                    "start": "2026-08-01T00:59:58Z",
                    "end": "2026-08-01T23:56:10Z",
                    "limiting_roles": ["radar", "lidar"],
                }
            }
        ),
        encoding="utf-8",
    )
    index = {
        "days": [
            {
                "day": "2026-07-06",
                "cloudnet_backbone_status": "ready",
                "cloudnet_instrument_source_common_overlap_hours": 3.998889,
                "cloudnet_instrument_source_common_overlap_start": "2026-07-06T19:59:59Z",
                "cloudnet_instrument_source_common_overlap_end": "2026-07-06T23:59:55Z",
            },
            {
                "day": "2026-08-01",
                "cloudnet_backbone_status": "ready",
                "cloudnet_instrument_source_common_overlap_hours": 23.900556,
                "cloudnet_instrument_source_common_overlap_start": "2026-08-01T00:00:01Z",
                "cloudnet_instrument_source_common_overlap_end": "2026-08-01T23:56:10Z",
            },
        ]
    }

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    html = module._cloudnet_backbone_review_panel(index, "2026-08-01")

    assert "23.900556 h" in html
    assert "22.936667 h" in html
    assert "2026-08-01T00:00:01Z to 2026-08-01T23:56:10Z" in html
    assert "2026-08-01T00:59:58Z to 2026-08-01T23:56:10Z" in html
    assert "2026-07-06" not in html


def test_direct_process_verdict_uses_active_iceland_models(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day = "2026-08-01"
    scorecards = tmp_path / "2026" / "08" / "01" / "scorecards"
    scorecards.mkdir(parents=True)
    verdicts = []
    for model_id, hits, misses in (("ifs", 30, 10), ("gfs", 0, 40), ("icon", 25, 15)):
        verdicts.append(
            {
                "role": f"direct_{model_id}",
                "model_id": model_id,
                "status": "window_ready" if model_id == "ifs" else "diagnostic_ready",
                "paired_sample_count": 40,
                "production_use": (
                    "process_ranking_ready" if model_id == "ifs" else "diagnostic_only"
                ),
                "metrics": {
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": 0,
                    "probability_of_detection": hits / 40,
                    "false_alarm_ratio": 0.0,
                    "critical_success_index": hits / 40,
                },
            }
        )
    (scorecards / "direct_process_verdict.json").write_text(
        json.dumps(
            {
                "status": "ready_direct_process_verdict",
                "scorecard": {
                    "status": "ready_direct_process_verdict",
                    "statement": "Active direct-model process verdicts are ready.",
                    "ready_direct_roles": ["direct_ifs"],
                    "diagnostic_direct_roles": ["direct_gfs", "direct_icon"],
                    "primary_verdict": verdicts[0],
                    "verdicts": verdicts,
                },
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    verdict = module._direct_process_verdict(day)
    html = module._direct_process_verdict_panel(day)
    evidence = module._direct_model_evidence_panel(day)

    assert verdict["exists"] is True
    assert [item["role"] for item in verdict["verdicts"]] == [
        "direct_ifs",
        "direct_gfs",
        "direct_icon",
    ]
    assert "IFS direct" in html
    assert "GFS direct" in html
    assert "ICON direct" in html
    assert "ERA5" not in html
    assert "CARRA2" not in html
    assert "Direct Model-Evaluation Evidence" in evidence


def test_cloud_seb_panels_only_enumerate_active_process_roles(tmp_path) -> None:
    module = _load_model_evaluation_module()
    day = "2026-08-01"
    scorecards = tmp_path / "2026" / "08" / "01" / "scorecards"
    scorecards.mkdir(parents=True)

    def write_process_scorecard(name: str, model_id: str, clear: int, cloudy: int) -> None:
        (scorecards / f"{name}.json").write_text(
            json.dumps(
                {
                    "status": "scored",
                    "scorecard": {
                        "status": "scored",
                        "model_id": model_id,
                        "sample_count": clear + cloudy,
                        "clear_count": clear,
                        "cloudy_count": cloudy,
                        "output_json": str(scorecards / f"{name}.json"),
                        "figure_evidence_summary": {
                            "production_ready": bool(clear and cloudy),
                            "support_status": (
                                "clear_and_cloudy_sampled"
                                if clear and cloudy
                                else "only_clear_sampled"
                            ),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    write_process_scorecard("direct_ifs_cloud_seb_process", "ifs", 2, 2)
    write_process_scorecard("direct_gfs_cloud_seb_process", "gfs", 4, 0)
    write_process_scorecard("direct_icon_cloud_seb_process", "icon", 1, 1)
    write_process_scorecard("cloud_seb_process_cm1_gfs", "cm1_gfs", 2, 2)
    (scorecards / "cloud_seb_process_window.json").write_text(
        json.dumps(
            {
                "status": "process_window_ready",
                "diagnostic": {
                    "pairs": {
                        "direct_ifs": {
                            "comparison_ready": True,
                            "regime_contingency": {
                                "status": "diagnostic_ready",
                                "paired_sample_count": 4,
                                "dominant_pair": "observed_cloudy_model_cloudy",
                                "metrics": {"hits": 2, "misses": 0, "false_alarms": 0},
                                "table": {
                                    "observed_cloudy_model_cloudy": 2,
                                    "observed_cloudy_model_clear": 0,
                                    "observed_clear_model_cloudy": 0,
                                    "observed_clear_model_clear": 2,
                                },
                            },
                        },
                        "direct_era5": {
                            "comparison_ready": True,
                            "regime_contingency": {"status": "legacy"},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    module.OPERATIONAL_CAMPAIGN_ROOT = tmp_path
    process = module._cloud_seb_process_summary(day)
    roles = [row["role"] for row in module._cloud_seb_role_rows(process)]
    contingency = module._cloud_seb_process_window_contingency(process)

    assert roles == ["direct_ifs", "direct_gfs", "direct_icon", "les_cm1_gfs"]
    assert "direct_ifs" in contingency
    assert "direct_era5" not in contingency
    assert "CARRA2" not in contingency
