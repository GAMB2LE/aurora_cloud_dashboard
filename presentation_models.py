"""Framework-neutral presentation decisions shared by dashboard clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from instrument_registry import INSTRUMENT_BY_BROWSER_KEY


@dataclass(frozen=True)
class EmptyDataState:
    instrument_title: str
    reason: str
    detail: str | None
    start: datetime | None
    end: datetime | None
    intentionally_powered_off: bool = False
    eyebrow: str | None = None


def empty_data_state(
    instrument_key: str,
    reason: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    detail: str | None = None,
    pdu_status: dict[str, object] | None = None,
) -> EmptyDataState:
    """Resolve an empty data window without coupling the decision to Plotly."""
    instrument = INSTRUMENT_BY_BROWSER_KEY.get(instrument_key)
    title = instrument.title if instrument is not None else instrument_key
    intentionally_off = pdu_status is not None and pdu_status.get("state") == "Off"
    if not intentionally_off:
        return EmptyDataState(title, reason, detail, start, end)

    pdu_detail = str(pdu_status.get("detail") or "PDU status is current")
    pdu_title = instrument.pdu_title if instrument is not None else title
    return EmptyDataState(
        instrument_title=title,
        reason="Data collection is paused because this instrument is intentionally powered off.",
        detail=f"{pdu_title} is off at its assigned PDU outlet. {pdu_detail}.",
        start=start,
        end=end,
        intentionally_powered_off=True,
        eyebrow="INTENTIONAL POWER-OFF",
    )
