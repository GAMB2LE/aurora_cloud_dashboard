from __future__ import annotations

import app


def test_desktop_shell_has_full_named_tabs() -> None:
    labels = [label for label, _slug, _panel in app.DESKTOP_TAB_SPECS]

    assert labels == [
        "Interactive Data Browser",
        "Science Quicklooks",
        "House Keeping Quicklooks",
        "AURORACam",
        "UAS",
        "Operations Dashboard",
    ]
    assert len(app.desktop_tabs) == len(labels)
    assert app.desktop_tabs.dynamic


def test_desktop_tab_labels_scroll_without_abbreviating() -> None:
    assert ":host(.desktop-tabs) .bk-header" in app.css
    assert "overflow-x: auto" in app.css
    assert ":host(.desktop-tabs) .bk-tab" in app.css
    assert "white-space: nowrap" in app.css


def test_desktop_controls_keep_compact_navigation_rows() -> None:
    controls_body = app.controls.objects[0]
    first_row_names = [widget.name for widget in controls_body.objects[0].objects]
    second_row_names = [widget.name for widget in controls_body.objects[1].objects]

    assert first_row_names == ["Instrument", "Start (UTC)", "End (UTC)", "Live Off"]
    assert second_row_names == ["Previous Day", "Reset View Defaults", "Next Day/Current Day"]


def test_phone_shell_keeps_operational_groups() -> None:
    assert list(app.MOBILE_TAB_OPTIONS) == ["Overview", "Power", "Plots", "Camera", "Ops"]
