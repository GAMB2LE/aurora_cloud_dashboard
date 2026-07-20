from browser_icons import instrument_icon_svg


def test_instrument_icons_are_labelled_and_portable():
    icon = instrument_icon_svg("airplane")

    assert "overview-instrument-row__icon-symbol" in icon
    assert "data-instrument-icon='airplane'" in icon
    assert "data:image/png;base64," in icon


def test_unknown_instrument_icon_uses_safe_fallback():
    assert "overview-instrument-row__icon-fallback" in instrument_icon_svg("unknown")
