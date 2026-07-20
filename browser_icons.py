"""Rendered SF Symbols shared with the iOS dashboard."""

from __future__ import annotations

from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path


_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "sf-symbols"


@lru_cache(maxsize=None)
def _symbol_mask_uri(system_image: str) -> str | None:
    """Return the alpha mask from an Apple-rendered SF Symbol PNG."""
    path = _ASSET_DIR / f"{system_image}.png"
    if not path.is_file():
        return None
    return f"data:image/png;base64,{b64encode(path.read_bytes()).decode('ascii')}"


def instrument_icon_svg(system_image: str | None) -> str:
    """Return the exact iOS symbol as a CSS-tinted alpha mask."""
    icon_id = str(system_image or "instrument")
    uri = _symbol_mask_uri(icon_id)
    if uri is None:
        return "<span class='overview-instrument-row__icon-fallback' aria-hidden='true'>!</span>"
    return (
        f"<span class='overview-instrument-row__icon-symbol' data-instrument-icon='{escape(icon_id, quote=True)}' "
        f"style=\"--instrument-symbol: url('{uri}')\" aria-hidden='true'></span>"
    )
