"""FastAPI wrapper for the Aurora Dashboard mobile API."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse

import mobile_catalog as catalog


app = FastAPI(
    title="Aurora Dashboard Mobile API",
    version="0.1.0",
    root_path=os.environ.get("AURORA_MOBILE_API_ROOT_PATH", ""),
)


def _token() -> str | None:
    value = os.environ.get("AURORA_MOBILE_API_TOKEN")
    return value.strip() if value else None


def _allow_public() -> bool:
    return os.environ.get("AURORA_MOBILE_API_ALLOW_PUBLIC", "").strip().lower() in {"1", "true", "yes", "on"}


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if _allow_public():
        return
    expected = _token()
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Mobile API token is not configured")
    scheme, _, value = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or value != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid mobile API token")


def _not_found(message: str = "Resource not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def _file_response(path: Path) -> FileResponse:
    if not path.exists() or not path.is_file():
        raise _not_found()
    stat_result = path.stat()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {
        "Cache-Control": "private, max-age=60",
        "ETag": f'W/"{stat_result.st_mtime_ns}-{stat_result.st_size}"',
        "Last-Modified": catalog.datetime.fromtimestamp(stat_result.st_mtime, catalog.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT"),
    }
    return FileResponse(path, media_type=media_type, headers=headers)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "serverTime": catalog.utc_now_iso(),
        "authRequired": not _allow_public(),
        "tokenConfigured": bool(_token()),
    }


@app.get("/manifest", dependencies=[Depends(require_auth)])
def manifest() -> dict:
    return catalog.manifest()


@app.get("/operations", dependencies=[Depends(require_auth)])
def operations() -> dict:
    return catalog.operations()


@app.get("/instruments/{instrument_id}/summary", dependencies=[Depends(require_auth)])
def instrument_summary(instrument_id: str, window: str = Query("24h", pattern="^(24h|7d)$")) -> dict:
    try:
        return catalog.instrument_summary(instrument_id, window)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/quicklooks", dependencies=[Depends(require_auth)])
def quicklooks(kind: str = Query("science", pattern="^(science|housekeeping)$"), instrument: str = "power") -> dict:
    try:
        return catalog.quicklooks(kind, instrument)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/wxcam", dependencies=[Depends(require_auth)])
def wxcam(stream: str = Query("fish_hdr"), day: str = Query("latest")) -> dict:
    try:
        return catalog.wxcam(stream, day)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/media/quicklook/{kind}/{instrument_id}/{token}", dependencies=[Depends(require_auth)])
def quicklook_media(kind: str, instrument_id: str, token: str) -> FileResponse:
    try:
        path = catalog.resolve_quicklook_path(kind, instrument_id, token)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    if not path:
        raise _not_found("Quicklook image not found")
    return _file_response(path)


@app.get("/media/wxcam/video/{stream}/{day}", dependencies=[Depends(require_auth)])
def wxcam_video(stream: str, day: str) -> FileResponse:
    try:
        resolved = catalog.wxcam(stream, day)["selectedDay"]
        path = catalog.resolve_wxcam_video_path(stream, resolved)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    if not path:
        raise _not_found("WXcam video not found")
    return _file_response(path)


@app.get("/media/wxcam/thumb/{stream}/{day_token}/{filename}", dependencies=[Depends(require_auth)])
def wxcam_thumbnail(stream: str, day_token: str, filename: str) -> FileResponse:
    path = catalog.resolve_wxcam_thumbnail_path(stream, day_token, filename)
    if not path:
        raise _not_found("WXcam thumbnail not found")
    return _file_response(path)
