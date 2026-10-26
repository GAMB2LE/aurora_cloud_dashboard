"""Execute the actual browser clock without a server or device connection."""
import json
import shutil
import subprocess

import pytest
from panel.models.reactive_html import ReactiveHTML

from dashboard_components import OverviewSnapshotHTML


@pytest.mark.parametrize("sample_key,value,prefix,unavailable", [
    ("chargerSample", "Charging", "1.78 kW", "Charging telemetry unavailable"),
    ("gridSample", "1.93 kW", "Grid AC input", "Grid telemetry unavailable"),
    ("gridSample", "0.0 W", "Grid AC input", "Grid telemetry unavailable"),
    ("gridSample", "-20.0 W", "Grid AC input", "Grid telemetry unavailable"),
])
def test_power_signal_browser_clock_and_cleanup(sample_key, value, prefix, unavailable):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser clock test")
    config = {"sampleKey": sample_key, "value": value, "prefix": prefix, "unavailable": unavailable}
    script = ("const scripts = " + json.dumps(OverviewSnapshotHTML._scripts) + ";\n"
              + "const config = " + json.dumps(config) + ";\n") + r"""
    import assert from 'node:assert/strict';
    const now = Date.parse('2026-10-26T20:00:00Z');
    let currentTime = now;
    Date.now = () => currentTime;
    let cards = [];
    const content = {innerHTML: '', querySelectorAll: selector => {
      assert.equal(selector, '[data-charger-sample], [data-grid-sample]'); return cards;
    }};
    const data = {markup: {html: '<div>snapshot</div>'}};
    const state = {};
    const view = {run_script: name => {
      new Function('data', 'state', 'view', 'content', scripts[name])(data, state, view, content);
    }};
    const events = {};
    const intervals = new Map();
    let nextInterval = 0;
    globalThis.document = {
      addEventListener: (name, fn) => {assert(!events[name]); events[name] = fn;},
      removeEventListener: (name, fn) => {assert.equal(events[name], fn); delete events[name];}
    };
    globalThis.window = document;
    globalThis.fetch = () => {throw new Error('The local clock must not make network requests');};
    globalThis.setInterval = (fn, ms) => {
      assert.equal(ms, 60000); intervals.set(++nextInterval, fn); return nextInterval;
    };
    globalThis.clearInterval = id => {assert(intervals.delete(id));};
    const ageCharger = c => {cards = [c]; view.run_script('age');};
    function card(minutes, state = config.value) {
      const values = {value: {textContent: state}, meta: {textContent: `${config.prefix} · Sample 0m ago`}};
      const classes = new Set(['mobile-status-card--green']);
      return {dataset: {[config.sampleKey]: new Date(now - minutes * 60000).toISOString()}, values, classes,
        classList: {add: c => classes.add(c), remove: c => classes.delete(c)},
        querySelector: selector => values[selector.endsWith('__value') ? 'value' : 'meta']};
    }
    for (const minutes of [0, 8, 30]) {
      const c = card(minutes);
      ageCharger(c);
      assert.equal(c.values.value.textContent, config.value);
      assert.equal(c.values.meta.textContent, `${config.prefix} · Sample ${minutes}m ago`);
    }
    for (const minutes of [30.01, 35, -1]) {
      const c = card(minutes);
      ageCharger(c);
      assert.equal(c.values.value.textContent, 'Unknown');
      assert(!c.classes.has('mobile-status-card--green'));
      assert(c.classes.has('mobile-status-card--gray'));
    }
    const future = card(-1);
    ageCharger(future);
    assert.equal(future.values.meta.textContent, config.unavailable);
    currentTime = now + 120000;
    ageCharger(future);
    assert.equal(future.values.value.textContent, 'Unknown');
    currentTime = now;
    for (const sample of ['', 'not-a-date']) {
      const c = card(8); c.dataset[config.sampleKey] = sample;
      ageCharger(c);
      assert.equal(c.values.value.textContent, 'Unknown');
      assert.equal(c.values.meta.textContent, config.unavailable);
    }
    for (const state of ['Not charging', 'Unknown']) {
      const c = card(8, state);
      ageCharger(c);
      assert.equal(c.values.value.textContent, state);
    }
    const c = card(8);
    cards = [c];
    view.run_script('render');
    assert.equal(content.innerHTML, data.markup.html);
    assert.equal(intervals.size, 1);
    // Re-rendering replaces the clock and listeners rather than duplicating them.
    view.run_script('render');
    assert.equal(intervals.size, 1);
    currentTime = now + 31 * 60000;
    events.visibilitychange();
    assert.equal(c.values.value.textContent, 'Unknown');
    assert.equal(c.values.meta.textContent, 'Last sample 39m ago');
    for (const callback of intervals.values()) callback();
    events.pageshow();
    data.markup = {html: '<div>replacement snapshot</div>'};
    view.run_script('markup');
    assert.equal(content.innerHTML, data.markup.html);
    assert.equal(intervals.size, 1);
    view.run_script('remove');
    view.run_script('remove');
    assert.equal(intervals.size, 0);
    assert.deepEqual(events, {});
    """
    subprocess.run([node, "--input-type=module"], input=script, text=True, check=True, capture_output=True)


def test_overview_snapshot_uses_existing_reactive_html_runtime():
    component = OverviewSnapshotHTML(object="<div>initial</div>")
    model = component.get_root()
    try:
        assert isinstance(model, ReactiveHTML)
        assert model.data.markup == {"html": "<div>initial</div>"}
        assert set(model.scripts) == {"render", "object", "markup", "age", "remove"}
        component.object = "<div>updated</div>"
        assert model.data.markup == {"html": "<div>updated</div>"}
    finally:
        component._cleanup(model)
