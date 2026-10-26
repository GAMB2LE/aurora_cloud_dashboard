"""Execute the actual browser clock without a server or device connection."""
import shutil
import subprocess

import pytest

from dashboard_components import OverviewSnapshotHTML


def test_charger_browser_clock_and_cleanup():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser clock test")
    script = OverviewSnapshotHTML._esm + r"""
    import assert from 'node:assert/strict';
    const now = Date.parse('2026-10-26T20:00:00Z');
    function card(minutes, state = 'Charging') {
      const values = {value: {textContent: state}, meta: {textContent: '1.78 kW · Sample 0m ago'}};
      const classes = new Set(['mobile-status-card--green']);
      return {dataset: {chargerSample: new Date(now - minutes * 60000).toISOString()}, values, classes,
        classList: {add: c => classes.add(c), remove: c => classes.delete(c)},
        querySelector: selector => values[selector.endsWith('__value') ? 'value' : 'meta']};
    }
    for (const minutes of [0, 8, 30]) {
      const c = card(minutes);
      ageCharger(c, now);
      assert.equal(c.values.value.textContent, 'Charging');
      assert.equal(c.values.meta.textContent, `1.78 kW · Sample ${minutes}m ago`);
    }
    for (const minutes of [30.01, 35, -1]) {
      const c = card(minutes);
      ageCharger(c, now);
      assert.equal(c.values.value.textContent, 'Unknown');
      assert(!c.classes.has('mobile-status-card--green'));
      assert(c.classes.has('mobile-status-card--gray'));
    }
    const future = card(-1);
    ageCharger(future, now);
    ageCharger(future, now + 120000);
    assert.equal(future.values.value.textContent, 'Unknown');
    for (const sample of ['', 'not-a-date']) {
      const c = card(8); c.dataset.chargerSample = sample;
      ageCharger(c, now);
      assert.equal(c.values.value.textContent, 'Unknown');
    }
    for (const state of ['Not charging', 'Unknown']) {
      const c = card(8, state);
      ageCharger(c, now);
      assert.equal(c.values.value.textContent, state);
    }
    const c = card(8);
    const events = {};
    const callbacks = {};
    let interval, cleared = false;
    Date.now = () => now;
    globalThis.document = {
      createElement: () => ({innerHTML: '', querySelectorAll: () => [c]}),
      addEventListener: (name, fn) => events[name] = fn,
      removeEventListener: (name, fn) => {assert.equal(events[name], fn); delete events[name];}
    };
    globalThis.window = document;
    globalThis.setInterval = (fn, ms) => {assert.equal(ms, 60000); interval = fn; return 1;};
    globalThis.clearInterval = id => {assert.equal(id, 1); cleared = true;};
    render({model: {object: '<div>snapshot</div>', on: (name, fn) => callbacks[name] = fn}});
    Date.now = () => now + 31 * 60000;
    events.visibilitychange();
    assert.equal(c.values.value.textContent, 'Unknown');
    assert.equal(c.values.meta.textContent, 'Last sample 39m ago');
    interval(); events.pageshow(); callbacks['change:object']();
    callbacks.remove();
    assert(cleared);
    assert.deepEqual(events, {});
    """
    subprocess.run([node, "--input-type=module"], input=script, text=True, check=True, capture_output=True)
