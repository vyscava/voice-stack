"""The agent's small web app: a call page, call tokens, health and metrics.

The page is served from the same origin as the token endpoint. Browsers may
refuse to let a page on a public site open a connection to a private address,
so a self-hosted deployment needs its own page rather than a hosted one.

Anyone who can load this page can start a call. Put it behind whatever
restricts access in your deployment (a LAN-only route, a VPN, an auth proxy).
"""

from __future__ import annotations

import json
import secrets
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from core.settings import Settings, get_settings
from voice_agent import metrics

_PAGE = Path(__file__).parent / "static" / "index.html"
_URL_PLACEHOLDER = "__LIVEKIT_URL__"


def new_room_name() -> str:
    return "call-" + secrets.token_hex(8)


def mint_call_token(settings: Settings, room: str) -> str:
    from livekit import api

    if not (settings.LIVEKIT_API_KEY and settings.LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set to mint call tokens")
    token = (
        api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(settings.AGENT_CALLER_IDENTITY)
        .with_name(settings.AGENT_CALLER_IDENTITY)
        .with_ttl(timedelta(minutes=settings.AGENT_TOKEN_TTL_MIN))
        .with_grants(api.VideoGrants(room_join=True, room=room))
    )
    return token.to_jwt()


def render_page(settings: Settings) -> str:
    # Empty = the page derives wss://<its own host>, for a LiveKit served on the same host.
    # Inserted as a JSON string literal, escaped so the value can never close the script.
    literal = json.dumps(settings.AGENT_PUBLIC_LIVEKIT_URL).replace("<", "\\u003c")
    return _PAGE.read_text(encoding="utf-8").replace(f'"{_URL_PLACEHOLDER}"', literal)


app = FastAPI(title="Voice Stack Agent", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/", response_class=HTMLResponse)
def page() -> HTMLResponse:
    return HTMLResponse(render_page(get_settings()), headers={"Cache-Control": "no-store"})


@app.get("/token")
def token() -> Response:
    settings = get_settings()
    room = new_room_name()
    try:
        jwt = mint_call_token(settings, room)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    return JSONResponse({"token": jwt, "room": room}, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> PlainTextResponse:
    # Liveness only. LiveKit or a brain being down is something a call handles
    # by speaking; restarting this process would not fix it.
    return PlainTextResponse("ok")


@app.get("/metrics")
def metrics_endpoint() -> Response:
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)
