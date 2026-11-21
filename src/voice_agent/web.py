"""The agent's small web app: a call page, call tokens, health and metrics.

The page is served from the same origin as the token endpoint. Browsers may
refuse to let a page on a public site open a connection to a private address,
so a self-hosted deployment needs its own page rather than a hosted one.

Anyone who can load this page can start a call. Put it behind whatever
restricts access in your deployment (a LAN-only route, a VPN, an auth proxy).
Outbound calls under /v1/calls are additionally gated by AGENT_CALL_TOKEN.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Body, FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from core.settings import Settings, get_settings
from voice_agent import calls, metrics
from voice_agent.outbound import check_policy, parse_quiet_hours, parse_request

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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _disabled(settings: Settings) -> bool:
    return not settings.AGENT_CALL_TOKEN or not settings.AGENT_CALL_TO


def _authorized(value: str, token: str) -> bool:
    return hmac.compare_digest((value or "").encode(), f"Bearer {token}".encode())


def _record_json(record: calls.CallRecord) -> dict[str, Any]:
    return {
        "call_id": record.call_id,
        "reason": record.reason,
        "requested_by": record.requested_by,
        "status": record.status,
        "detail": record.detail,
        "started_at": record.started_at.isoformat(),
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
    }


async def create_outbound_room(settings: Settings, record: calls.CallRecord) -> None:
    from livekit import api

    lk = api.LiveKitAPI(settings.LIVEKIT_URL, settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
    try:
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=record.call_id,
                metadata=json.dumps(
                    {
                        "outbound": {
                            "call_id": record.call_id,
                            "reason": record.reason,
                            "requested_by": record.requested_by,
                        }
                    }
                ),
                empty_timeout=30,
                agents=[api.RoomAgentDispatch(agent_name="")],
            )
        )
    finally:
        await lk.aclose()


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


@app.post("/v1/calls")
def create_call(
    body: Any = Body(default=None), auth_value: str = Header(default=None, alias="Authorization")
) -> Response:
    settings = get_settings()
    if _disabled(settings):
        return JSONResponse({"error": "disabled"}, status_code=404)
    if not _authorized(auth_value, settings.AGENT_CALL_TOKEN):
        metrics.OUTBOUND_CALLS.labels("rejected_auth").inc()
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        request = parse_request(body)
    except ValueError as exc:
        metrics.OUTBOUND_CALLS.labels("invalid").inc()
        return JSONResponse({"error": "invalid", "detail": str(exc)}, status_code=422)

    now = _now()
    if calls.REGISTRY.has_active(now, settings.AGENT_CALL_RING_TIMEOUT_S, settings.AGENT_CALL_MAX_DURATION_S):
        metrics.OUTBOUND_CALLS.labels("in_progress").inc()
        return JSONResponse({"error": "call_in_progress"}, status_code=409)
    try:
        quiet = parse_quiet_hours(settings.AGENT_CALL_QUIET_HOURS)
        ZoneInfo(settings.AGENT_CALL_TIMEZONE)
    except (ValueError, ZoneInfoNotFoundError):
        metrics.OUTBOUND_CALLS.labels("failed").inc()
        return JSONResponse({"error": "misconfigured"}, status_code=503)
    refusal = check_policy(
        now,
        calls.REGISTRY.attempt_times(),
        quiet=quiet,
        tz=settings.AGENT_CALL_TIMEZONE,
        max_per_day=settings.AGENT_CALL_MAX_PER_DAY,
        min_interval_min=settings.AGENT_CALL_MIN_INTERVAL_MIN,
    )
    if refusal is not None:
        metrics.OUTBOUND_CALLS.labels(f"refused_{refusal}").inc()
        return JSONResponse({"error": "refused", "detail": refusal}, status_code=429)
    try:
        record = calls.REGISTRY.start(
            request, now, settings.AGENT_CALL_RING_TIMEOUT_S, settings.AGENT_CALL_MAX_DURATION_S
        )
    except calls.CallInProgress:
        metrics.OUTBOUND_CALLS.labels("in_progress").inc()
        return JSONResponse({"error": "call_in_progress"}, status_code=409)
    try:
        asyncio.run(create_outbound_room(settings, record))
    except Exception:
        calls.REGISTRY.update(record.call_id, "failed", "room_create", now=now)
        metrics.OUTBOUND_CALLS.labels("failed").inc()
        return JSONResponse({"error": "room_create_failed"}, status_code=503)
    metrics.OUTBOUND_CALLS.labels("dialing").inc()
    return JSONResponse({"call_id": record.call_id, "status": "dialing"}, status_code=202)


@app.get("/v1/calls/{call_id}")
def get_call(call_id: str, auth_value: str = Header(default=None, alias="Authorization")) -> Response:
    settings = get_settings()
    if _disabled(settings):
        return JSONResponse({"error": "disabled"}, status_code=404)
    if not _authorized(auth_value, settings.AGENT_CALL_TOKEN):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    record = calls.REGISTRY.get(call_id, _now(), settings.AGENT_CALL_RING_TIMEOUT_S, settings.AGENT_CALL_MAX_DURATION_S)
    if record is None:
        return JSONResponse({"error": "unknown_call"}, status_code=404)
    return JSONResponse(_record_json(record))
