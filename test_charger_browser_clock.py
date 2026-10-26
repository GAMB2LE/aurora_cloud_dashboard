"""Execute the actual browser clock without a server or device connection."""
import json
import shutil
import subprocess

import pytest
from panel.models.reactive_html import ReactiveHTML

from dashboard_components import OverviewSnapshotHTML


def test_charger_browser_clock_and_cleanup():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser clock test")
    script = "const scripts = " + json.dumps(OverviewSnapshotHTML._scripts) + ";\n" + r"""
    import assert from 'node:assert/strict';
    const now = Date.parse('2026-10-26T20:00:00Z');
    let currentTime = now;
    Date.now = () => currentTime;
    let cards = [];
    const content = {innerHTML: '', querySelectorAll: selector => {
      assert.equal(selector, '[data-charger-sample]'); return cards;
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
    globalThis.setInterval = (fn, ms) => {
      assert.equal(ms, 60000); intervals.set(++nextInterval, fn); return nextInterval;
    };
    globalThis.clearInterval = id => {assert(intervals.delete(id));};
    const ageCharger = c => {cards = [c]; view.run_script('age');};
    function card(minutes, state = 'Charging') {
      const values = {value: {textContent: state}, meta: {textContent: '1.78 kW · Sample 0m ago'}};
      const classes = new Set(['mobile-status-card--green']);
      return {dataset: {chargerSample: new Date(now - minutes * 60000).toISOString()}, values, classes,
        classList: {add: c => classes.add(c), remove: c => classes.delete(c)},
        querySelector: selector => values[selector.endsWith('__value') ? 'value' : 'meta']};
    }
    for (const minutes of [0, 8, 30]) {
      const c = card(minutes);
      ageCharger(c);
      assert.equal(c.values.value.textContent, 'Charging');
      assert.equal(c.values.meta.textContent, `1.78 kW · Sample ${minutes}m ago`);
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
    currentTime = now + 120000;
    ageCharger(future);
    assert.equal(future.values.value.textContent, 'Unknown');
    currentTime = now;
    for (const sample of ['', 'not-a-date']) {
      const c = card(8); c.dataset.chargerSample = sample;
      ageCharger(c);
      assert.equal(c.values.value.textContent, 'Unknown');
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
