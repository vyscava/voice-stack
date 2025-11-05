from __future__ import annotations

import asyncio
import importlib.metadata as md
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from core.logging import logger_tts as logger
from core.middleware import ResourceGuardMiddleware
from core.settings import get_settings
from tts.api.api_v1.api import api_router
from utils.debugpy_helper import maybe_enable_debugpy

# Core Services Settings
settings = get_settings()


async def idle_timeout_checker():
    """
    Background task that periodically checks if the TTS model should be unloaded due to inactivity.

    Runs every 60 seconds and checks if the model has been idle for longer than
    TTS_IDLE_TIMEOUT_MINUTES. If so, unloads the model to free resources.
    """
    from tts.engine_factory import _engine

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            if _engine is not None and settings.TTS_IDLE_TIMEOUT_MINUTES > 0:
                # Check and unload if idle
                was_unloaded = _engine.check_and_unload_if_idle()
                if was_unloaded:
                    logger.info("TTS model unloaded due to inactivity")
        except Exception as e:  # noqa: PERF203
            logger.error(f"Error in idle timeout checker: {e}")


@asynccontextmanager
async def app_init(app: FastAPI) -> AsyncIterator[None]:
    """
    Performs initialization tasks for the application during startup.

    Tasks:
        - Loads API routes
        - Starts background task for idle model checking

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Allows the application to continue initialization.
    """
    logger.info("Initializing web server...")

    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)
    logger.info("API routes loaded...")

    # Start background task for idle timeout checking
    if settings.TTS_IDLE_TIMEOUT_MINUTES > 0:
        idle_task = asyncio.create_task(idle_timeout_checker())
        logger.info(f"Idle timeout checker started (timeout: {settings.TTS_IDLE_TIMEOUT_MINUTES} minutes)")
    else:
        idle_task = None
        logger.info("Idle timeout disabled (TTS_IDLE_TIMEOUT_MINUTES=0)")

    yield  # Allow further initialization steps

    # Cleanup on shutdown
    logger.info("Shutting down TTS service...")

    # Stop idle timeout checker
    if idle_task is not None:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            logger.info("Idle timeout checker stopped")

    # Clean up semaphore to prevent resource leak warnings
    try:
        from tts import engine_factory

        if hasattr(engine_factory, "_semaphore") and engine_factory._semaphore is not None:
            # Release any acquired slots to prevent resource leak warning
            # This ensures the semaphore is fully released on shutdown
            _semaphore = engine_factory._semaphore
            max_concurrent = settings.TTS_MAX_CONCURRENT_REQUESTS
            current_available = _semaphore._value
            acquired_slots = max_concurrent - current_available

            if acquired_slots > 0:
                logger.info(f"Releasing {acquired_slots} acquired semaphore slots on shutdown")
                for _ in range(acquired_slots):
                    _semaphore.release()

            logger.info("Semaphore cleanup completed")
    except Exception as e:
        logger.debug(f"Semaphore cleanup skipped: {e}")


if settings.DEBUGPY_ENABLE:
    maybe_enable_debugpy(
        host=settings.DEBUGPY_HOST, port=settings.DEBUGPY_PORT, wait_for_client=settings.DEBUGPY_WAIT_FOR_CLIENT
    )

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

# Configure CORS (Cross-Origin Resource Sharing) settings
if settings.CORS_ORIGINS:
    # Parse CORS_ORIGINS from a comma-separated string into a list
    cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
    logger.info(f"Configuring CORS with allowed origins: {cors_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,  # Allow specified origins
        allow_credentials=True,  # Allow cookies and credentials
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )
else:
    logger.warning("CORS_ORIGINS is not set. No CORS configuration applied.")

# Add resource guard middleware (memory/swap/file size checks)
app.add_middleware(ResourceGuardMiddleware)
logger.info("Resource guard middleware enabled")


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index() -> Any:
    return "/docs"


@app.get("/health")
@app.get("/healthz")
@app.get("/healthcheck")
async def healthcheck() -> JSONResponse:
    """
    Healthcheck endpoint to verify that the server is running.

    Returns:
        dict: A JSON response with the key "status" and value "ok".
    """
    return JSONResponse({"status": "ok"})


@app.get("/health/detailed")
async def detailed_health() -> JSONResponse:
    """
    Detailed health endpoint with system resource metrics.

    Provides comprehensive system information including:
    - Memory usage (RAM and swap)
    - Model loading status
    - Concurrency slots availability

    This endpoint is useful for monitoring and debugging resource issues.

    Returns:
        JSONResponse: System metrics and service status
    """
    from datetime import datetime, timezone

    import psutil

    from tts.engine_factory import _engine, _semaphore

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Calculate active/total request slots
    active_requests = 0
    max_concurrent = settings.TTS_MAX_CONCURRENT_REQUESTS

    if _semaphore is not None:
        # _value represents available slots, so used = max - available
        active_requests = max_concurrent - _semaphore._value

    response_data = {
        "status": "healthy",
        "service": "tts",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "memory": {
            "percent": round(mem.percent, 1),
            "available_mb": round(mem.available / (1024 * 1024), 1),
            "total_mb": round(mem.total / (1024 * 1024), 1),
            "threshold_percent": settings.MEMORY_THRESHOLD_PERCENT,
        },
        "swap": {
            "percent": round(swap.percent, 1),
            "used_mb": round(swap.used / (1024 * 1024), 1),
            "total_mb": round(swap.total / (1024 * 1024), 1),
            "threshold_percent": settings.SWAP_THRESHOLD_PERCENT,
        },
        "model": {
            "loaded": _engine is not None,
            "engine": settings.TTS_ENGINE,
            "model_name": settings.TTS_MODEL,
            "device": settings.TTS_DEVICE,
        },
        "concurrency": {
            "active_requests": active_requests,
            "max_concurrent": max_concurrent,
            "available_slots": max_concurrent - active_requests,
        },
        "config": {
            "idle_timeout_minutes": settings.TTS_IDLE_TIMEOUT_MINUTES,
            "max_upload_mb": settings.MAX_UPLOAD_SIZE_MB,
        },
    }

    return JSONResponse(response_data)
