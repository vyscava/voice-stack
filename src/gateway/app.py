"""The voice-gateway service: gives the agent fleet ears and a mouth.

It adds no brain. Agents already exist and are reachable over NATS; this
service turns an utterance into a question for one of them and its answer back
into speech. Agent code changes by zero lines.

Deliberately not here: any model. The gateway calls the ASR and TTS services
over HTTP, so it starts in milliseconds, holds no GPU memory, and can be
restarted without evicting a 1.8 GB model from anyone's cache.
"""

from __future__ import annotations

import importlib.metadata as md
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from core.error_handler import register_exception_handlers
from core.logging import logger_gateway as logger
from core.settings import get_settings
from gateway.api.api_v1.api import api_router
from gateway.deps import build_agent, shutdown_agent

settings = get_settings()


@asynccontextmanager
async def app_init(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Initializing voice gateway...")
    app.include_router(api_router, prefix=settings.API_V1_STR)

    agent = build_agent()
    logger.info(f"Agent route: {agent.subject} (timeout {settings.GATEWAY_AGENT_TIMEOUT_S}s)")
    logger.info(f"ASR: {settings.GATEWAY_ASR_URL} | TTS: {settings.GATEWAY_TTS_URL}")
    logger.info(f"Reply voice: {settings.GATEWAY_VOICE} ({settings.GATEWAY_AUDIO_FORMAT})")

    # The bus is NOT dialled here on purpose. A NATS outage must not stop the
    # gateway from starting: it connects lazily on the first exchange, so the
    # service comes up, answers /health, and speaks an error rather than
    # crash-looping while NATS is down.

    yield

    logger.info("Shutting down voice gateway...")
    await shutdown_agent()


# Create the FastAPI application instance
projectMetadata = md.metadata("voice-stack")
app = FastAPI(
    title=settings.PROJECT_NAME or projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    license_info={"name": "MIT License"},
    lifespan=app_init,
)

register_exception_handlers(app)

if settings.CORS_ORIGINS:
    cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
    logger.info(f"Configuring CORS with allowed origins: {cors_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_STATIC_DIR = Path(__file__).parent / "static"
_INDEX = _STATIC_DIR / "index.html"


@app.get("/", include_in_schema=False)
async def root() -> Response:
    """Serve the push-to-talk page, or fall back to the API docs.

    The page is the only surface a person can actually speak into, so it lives
    at the root rather than behind a path nobody would guess. If the file is
    missing (a partial image, say) fall back to /docs instead of 500ing --
    a broken UI should not make the service look dead.

    NOTE: browsers expose getUserMedia only in a SECURE CONTEXT, so this page
    can only reach a microphone over HTTPS or on localhost. Served over plain
    http on a LAN address it loads and then refuses to record, which is why it
    says so itself rather than failing silently.
    """
    if _INDEX.is_file():
        return FileResponse(_INDEX, media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/health")
@app.head("/health")
@app.get("/healthz")
@app.head("/healthz")
@app.get("/healthcheck")
@app.head("/healthcheck")
async def healthcheck() -> JSONResponse:
    """Liveness only.

    Deliberately does NOT probe ASR, TTS or NATS. A dependency being down is
    something the gateway handles by speaking an error, not a reason for an
    orchestrator to kill and restart this process. Use /health/detailed to see
    what it can currently reach.
    """
    return JSONResponse({"status": "ok", "service": "gateway"})


@app.get("/health/detailed")
async def detailed_health() -> JSONResponse:
    """What the gateway is configured to talk to, plus local resources."""
    mem = psutil.virtual_memory()
    agent = build_agent()

    data: dict[str, Any] = {
        "status": "ok",
        "service": "gateway",
        "memory": {
            "percent": round(mem.percent, 1),
            "available_mb": round(mem.available / (1024 * 1024), 1),
        },
        "routes": {
            "asr_url": settings.GATEWAY_ASR_URL,
            "tts_url": settings.GATEWAY_TTS_URL,
            "agent_subject": agent.subject,
            "nats_url": settings.GATEWAY_NATS_URL,
        },
        "voice": {
            "voice": settings.GATEWAY_VOICE,
            "format": settings.GATEWAY_AUDIO_FORMAT,
            "language": settings.GATEWAY_LANGUAGE or "auto",
        },
        "timeouts_s": {
            "agent": settings.GATEWAY_AGENT_TIMEOUT_S,
            "asr": settings.GATEWAY_ASR_TIMEOUT_S,
            "tts": settings.GATEWAY_TTS_TIMEOUT_S,
        },
    }
    return JSONResponse(data)
