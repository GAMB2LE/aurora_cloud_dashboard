"""HTTP bootstrap health is useful, but it does not prove browser rendering."""

from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

import collect_operations_snapshot as collector


BOOTSTRAP = b'''<!doctype html><html><head><title>AURORA Data Viewer</title></head>
<body><div id="dashboard-root"></div><script type="application/json" id="docs">{}</script>
<script>
const docs_json = document.getElementById('docs').textContent;
const render_items = [{"token":"test-only","roots":{"p1":"dashboard-root"},"root_ids":["p1"]}];
root.Bokeh.embed.embed_items(docs_json, render_items);
</script></body></html>'''


def test_small_panel_bootstrap_is_ready_without_a_size_threshold():
    assert len(BOOTSTRAP) < 10_000
    assert collector._dashboard_document_state(200, BOOTSTRAP) == 1


@pytest.mark.parametrize("status,body", [
    (500, BOOTSTRAP),
    (302, BOOTSTRAP),
    (200, BOOTSTRAP.replace(b"<title>AURORA Data Viewer</title>", b"<title>Bokeh Application</title>")),
    (200, b"<title>AURORA Data Viewer</title><p>Loading...</p>"),
    (200, b"<title>AURORA Data Viewer</title>" + b" " * 100_001),
    (200, BOOTSTRAP.replace(b'<div id="dashboard-root"></div>', b"")),
    (200, BOOTSTRAP.replace(b'"roots":{"p1":"dashboard-root"}', b'"roots":{}')),
    (200, BOOTSTRAP.replace(b'"roots":{"p1":"dashboard-root"}', b'"roots":["dashboard-root"]')),
    (200, BOOTSTRAP.replace(b'const render_items = [', b'const render_items = [invalid')),
    (200, BOOTSTRAP.replace(b'root.Bokeh.embed.embed_items(docs_json, render_items);', b"")),
])
def test_error_and_placeholder_responses_are_not_ready(status, body):
    assert collector._dashboard_document_state(status, body) == 0


def test_http_probe_keeps_legacy_field_as_bootstrap_alias():
    response = MagicMock()
    response.status = 200
    response.read.return_value = BOOTSTRAP
    response.__enter__.return_value = response
    with patch.object(collector.urllib.request, "urlopen", return_value=response):
        probe = collector._probe_http("https://example.test/app")
    assert probe["ok_state"] == 1
    assert probe["bootstrap_ready_state"] == probe["full_document_state"] == 1
    assert probe["content_bytes"] == len(BOOTSTRAP)
    assert probe["title"] == "AURORA Data Viewer"


@pytest.mark.parametrize("error", [
    HTTPError("https://example.test/app", 500, "Server error", {}, BytesIO(BOOTSTRAP)),
    URLError("Connection failed"),
])
def test_http_failures_keep_both_readiness_fields_false(error):
    with patch.object(collector.urllib.request, "urlopen", side_effect=error):
        probe = collector._probe_http("https://example.test/app")
    assert probe["ok_state"] == 0
    assert probe["bootstrap_ready_state"] == probe["full_document_state"] == 0
    assert probe["error"]
