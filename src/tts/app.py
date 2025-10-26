from __future__ import annotations

import importlib.metadata as md
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from core.logging import logger_tts as logger
from core.settings import get_settings
from tts.api.api_v1.api import api_router
from utils.debugpy_helper import maybe_enable_debugpy

# Core Services Settings
settings = get_settings()


@asynccontextmanager
async def app_init(app: FastAPI) -> AsyncIterator[None]:
    """
    Performs initialization tasks for the application during startup.

    Tasks:
        - Verifies database connectivity using a ping command.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Allows the application to continue initialization.
    """
    logger.info("Initializing web server...")

    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)
    logger.info("API routes loaded...")

    yield  # Allow further initialization steps


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
