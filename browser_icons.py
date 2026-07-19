"""Portable SVG equivalents for the instrument symbols used by the iOS app."""

from __future__ import annotations

from html import escape


_SVG_PATHS = {
    "cloud.sun": (
        "M7 16a4 4 0 0 1 1-7.9A5.5 5.5 0 0 1 18.6 10H19a3 3 0 0 1 0 6H7Z"
        " M8 4V2 M4.2 5.6 2.8 4.2 M11.8 5.6l1.4-1.4"
    ),
    "sun.max": (
        "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z"
        " M12 2v2 M12 20v2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4"
        " M2 12h2 M20 12h2 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4"
    ),
    "airplane": "M3 13.5 21 5l-5.5 8.5L21 19l-18-5.5Z M9.5 11.5 13 15",
    "laser.burst": (
        "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z"
        " M12 2v3 M12 19v3 M2 12h3 M19 12h3 M4.9 4.9 7 7 M17 17l2.1 2.1"
        " M19.1 4.9 17 7 M7 17l-2.1 2.1"
    ),
    "dot.radiowaves.left.and.right": (
        "M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4Z"
        " M7.8 7.8a6 6 0 0 0 0 8.4 M16.2 7.8a6 6 0 0 1 0 8.4"
        " M4.9 4.9a10 10 0 0 0 0 14.2 M19.1 4.9a10 10 0 0 1 0 14.2"
    ),
    "antenna.radiowaves.left.and.right": (
        "M12 5v13 M8 21h8 M10 18h4 M9 9l3-4 3 4"
        " M6.5 8.5a7.5 7.5 0 0 0 0 7 M17.5 8.5a7.5 7.5 0 0 1 0 7"
    ),
}

_FALLBACK_PATH = "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z M12 8v5 M12 16h.01"


def instrument_icon_svg(system_image: str | None) -> str:
    """Return a labelled inline SVG that inherits the status colour from CSS."""
    icon_id = str(system_image or "instrument")
    paths = _SVG_PATHS.get(icon_id, _FALLBACK_PATH)
    return (
        f"<svg class='overview-instrument-row__icon-svg' data-instrument-icon='{escape(icon_id, quote=True)}' "
        "viewBox='0 0 24 24' aria-hidden='true' fill='none' stroke='currentColor' "
        "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>"
        f"<path d='{paths}' /></svg>"
    )
