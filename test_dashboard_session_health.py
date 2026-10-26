"""Regression coverage for the September 2026 dashboard availability incident."""

from pathlib import Path
from unittest.mock import patch

from bokeh.application import Application
from bokeh.application.handlers.script import ScriptHandler
import panel as pn
import param

import mobile_catalog


def test_repeated_browser_sessions_do_not_register_new_component_classes():
    """Exercise the server's per-session script runner, retaining open sessions."""
    application = Application(ScriptHandler(filename=str(Path(__file__).with_name("app.py"))))
    names = {"WxcamVideoPlayer", "AuroracamCameraCard", "BrowserPerformanceProbe", "OverviewSnapshotHTML"}

    def components():
        return {cls for cls in param.descendents(pn.reactive.ReactiveCustomBase)
                if cls.__name__ in names}

    with patch.dict("os.environ", {"AURORA_DASHBOARD_PERF_ENABLED": "0"}):
        documents = [application.create_document()]
        original = components()
        assert {cls.__name__ for cls in original} == names
        for _ in range(5):
            documents.append(application.create_document())
            assert not application.handlers[0].failed
            assert components() == original


def test_failed_float_probe_is_red_in_the_public_dashboard_group():
    snapshot = {
        "dashboard_http_ok_state": 1.0,
        "failover_primary_dashboard_http_ok_state": 0.0,
        "failover_standby_dashboard_http_ok_state": 1.0,
    }
    group = next(group for group in mobile_catalog._root_cause_groups(snapshot, [])
                 if group["id"] == "dashboard")
    assert group["level"] == "red"
    assert "healthy" not in group["detail"]


def test_unknown_probe_values_are_not_healthy():
    for values in ([float("nan")], [float("inf")], [-1.0], [None], ["unexpected"]):
        assert mobile_catalog.level_from_booleans(values) == "unknown"


def test_probe_normalization_preserves_boolean_and_numeric_states():
    for value in (False, 0, 0.0, "0", "0.0", "false"):
        assert mobile_catalog.level_from_booleans([1.0, value]) == "red"
    for value in (True, 1, 1.0, "1", "1.0", "true"):
        assert mobile_catalog.level_from_booleans([value]) == "green"


def test_shared_probe_class_keeps_session_callbacks_separate():
    from dashboard_components import BrowserPerformanceProbe

    events = [[], []]
    probes = [BrowserPerformanceProbe(log_callback=lambda event, _events=bucket, **fields:
                                     _events.append((event, fields)))
              for bucket in events]
    probes[0]._handle_msg({"event": "browser_document_ready", "duration_ms": 12})
    probes[1]._handle_msg({"event": "browser_document_ready", "duration_ms": 34})
    assert events[0][0][1]["duration_ms"] == 12
    assert events[1][0][1]["duration_ms"] == 34
    assert len(events[0]) == len(events[1]) == 1
