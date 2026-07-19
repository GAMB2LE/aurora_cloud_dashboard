"""Small, defensive accessors for the current Panel request and session."""

from __future__ import annotations

import panel as pn


def _document():
    return pn.state.curdoc


def _request():
    doc = _document()
    return doc.session_context.request if doc and doc.session_context else None


def session_id() -> str | None:
    try:
        doc = _document()
        context = None if doc is None else doc.session_context
        return None if context is None else context.id
    except (AttributeError, RuntimeError):
        return None


def request_header(name: str) -> str | None:
    try:
        headers = pn.state.headers or {}
    except (AttributeError, RuntimeError):
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return None


def request_path() -> str | None:
    try:
        request = _request()
        return None if request is None else str(request.path)
    except (AttributeError, RuntimeError):
        return None


def request_query_args() -> dict[str, str]:
    try:
        request = _request()
        query_args = {} if request is None else (getattr(request, "query_arguments", {}) or {})
    except (AttributeError, RuntimeError):
        return {}
    parsed: dict[str, str] = {}
    for key, values in query_args.items():
        if not values:
            continue
        raw = values[0]
        normalized_key = key.decode("utf-8", errors="ignore") if isinstance(key, bytes) else str(key)
        parsed[normalized_key] = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    return parsed


def request_base_url() -> str:
    proto = request_header("X-Forwarded-Proto") or "http"
    host = request_header("Host") or "127.0.0.1:5006"
    return f"{proto}://{host}{request_path() or '/app'}"


def client_ip() -> str | None:
    forwarded = request_header("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request_header("X-Real-Ip")
    if real_ip:
        return real_ip.strip()
    try:
        request = _request()
        remote_ip = None if request is None else getattr(request, "remote_ip", None)
        return None if remote_ip is None else str(remote_ip)
    except (AttributeError, RuntimeError):
        return None


def server_session_count() -> int | None:
    try:
        doc = _document()
        context = doc.session_context.server_context if doc and doc.session_context else None
        return None if context is None else int(len(context.sessions))
    except (AttributeError, RuntimeError, TypeError):
        return None


def live_session_count() -> int | None:
    try:
        return int((pn.state.session_info or {}).get("live", 0))
    except (AttributeError, TypeError, ValueError):
        return None


def total_session_count() -> int | None:
    try:
        return int((pn.state.session_info or {}).get("total", 0))
    except (AttributeError, TypeError, ValueError):
        return None
