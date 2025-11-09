"""
Error handling utilities for Voice Stack services.

This module provides utilities for consistent error handling across all endpoints:
- ErrorContext: Captures request context for logging and debugging
- Exception handlers: Convert custom exceptions to structured HTTP responses
"""

from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions import NotFoundError, ProcessingError, ResourceError, ValidationError, VoiceStackError
from core.settings import get_settings

settings = get_settings()


class ErrorContext:
    """
    Captures and manages request context for error logging and debugging.

    This class helps collect relevant information about a request so that
    when errors occur, we have rich context for debugging.

    Example:
        ctx = ErrorContext.create(request_id="req_123", endpoint="/audio/transcriptions")
        ctx.add_file_info(filename="audio.mp3", size_bytes=1048576, content_type="audio/mpeg")
        ctx.add_params({"model": "whisper-1", "language": "en"})

        # Later, when an error occurs:
        logger.error("Transcription failed", extra=ctx.to_log_dict())
    """

    def __init__(self, request_id: str, endpoint: str) -> None:
        """
        Initialize an ErrorContext.

        Args:
            request_id: Unique identifier for this request
            endpoint: API endpoint being called
        """
        self.request_id = request_id
        self.endpoint = endpoint
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.data: dict[str, Any] = {}

    @classmethod
    def create(cls, request_id: str | None = None, endpoint: str = "") -> ErrorContext:
        """
        Factory method to create an ErrorContext with auto-generated request ID.

        Args:
            request_id: Optional request ID (auto-generated if not provided)
            endpoint: API endpoint being called

        Returns:
            New ErrorContext instance
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"
        return cls(request_id=request_id, endpoint=endpoint)

    def add_file_info(
        self,
        filename: str | None = None,
        size_bytes: int | None = None,
        content_type: str | None = None,
        **kwargs: Any,
    ) -> ErrorContext:
        """
        Add file information to the context.

        Args:
            filename: Name of the uploaded file
            size_bytes: File size in bytes
            content_type: MIME type of the file
            **kwargs: Additional file-related metadata

        Returns:
            Self for method chaining
        """
        file_info: dict[str, Any] = {}
        if filename is not None:
            file_info["filename"] = filename
        if size_bytes is not None:
            file_info["size_bytes"] = size_bytes
            file_info["size_mb"] = round(size_bytes / (1024 * 1024), 2)
        if content_type is not None:
            file_info["content_type"] = content_type

        file_info.update(kwargs)
        self.data["file"] = file_info
        return self

    def add_params(self, params: dict[str, Any]) -> ErrorContext:
        """
        Add request parameters to the context.

        Args:
            params: Dictionary of request parameters

        Returns:
            Self for method chaining
        """
        self.data["params"] = params
        return self

    def add_model_info(self, model: str | None = None, engine: str | None = None, **kwargs: Any) -> ErrorContext:
        """
        Add model information to the context.

        Args:
            model: Model name being used
            engine: Engine type (e.g., "faster-whisper", "coqui")
            **kwargs: Additional model-related metadata

        Returns:
            Self for method chaining
        """
        model_info: dict[str, Any] = {}
        if model is not None:
            model_info["model"] = model
        if engine is not None:
            model_info["engine"] = engine

        model_info.update(kwargs)
        self.data["model"] = model_info
        return self

    def add_custom(self, key: str, value: Any) -> ErrorContext:
        """
        Add custom data to the context.

        Args:
            key: Data key
            value: Data value

        Returns:
            Self for method chaining
        """
        self.data[key] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to a dictionary.

        Returns:
            Dictionary representation of the context
        """
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
            **self.data,
        }

    def to_log_dict(self) -> dict[str, Any]:
        """
        Convert context to a dictionary suitable for structured logging.

        Returns:
            Dictionary with flattened structure for logging
        """
        log_dict = {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
        }

        # Flatten nested structures for easier log parsing
        for key, value in self.data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    log_dict[f"{key}_{sub_key}"] = sub_value
            else:
                log_dict[key] = value

        return log_dict


def build_error_response(
    error: VoiceStackError,
    status_code: int,
    request_id: str | None = None,
    include_traceback: bool = False,
) -> dict[str, Any]:
    """
    Build a structured error response from a VoiceStackError.

    Args:
        error: The exception to convert
        status_code: HTTP status code
        request_id: Request ID to include in response
        include_traceback: Whether to include stack trace (for debugging)

    Returns:
        Dictionary ready for JSON serialization
    """
    response: dict[str, Any] = {
        "error": error.to_dict(),
        "status_code": status_code,
    }

    if request_id:
        response["request_id"] = request_id

    if include_traceback and settings.DEBUG:
        response["traceback"] = traceback.format_exc()

    return response


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers for the FastAPI application.

    This function sets up handlers for all custom exceptions and ensures
    consistent error response format across the application.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle ValidationError exceptions."""
        request_id = getattr(exc, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=build_error_response(exc, status.HTTP_422_UNPROCESSABLE_ENTITY, request_id),
        )

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        """Handle NotFoundError exceptions."""
        request_id = getattr(exc, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=build_error_response(exc, status.HTTP_404_NOT_FOUND, request_id),
        )

    @app.exception_handler(ResourceError)
    async def resource_error_handler(request: Request, exc: ResourceError) -> JSONResponse:
        """Handle ResourceError exceptions."""
        request_id = getattr(exc, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=build_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request_id),
        )

    @app.exception_handler(ProcessingError)
    async def processing_error_handler(request: Request, exc: ProcessingError) -> JSONResponse:
        """Handle ProcessingError exceptions."""
        request_id = getattr(exc, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=build_error_response(
                exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request_id, include_traceback=True
            ),
        )

    @app.exception_handler(VoiceStackError)
    async def voicestack_error_handler(request: Request, exc: VoiceStackError) -> JSONResponse:
        """Handle generic VoiceStackError exceptions (fallback)."""
        request_id = getattr(exc, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=build_error_response(
                exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request_id, include_traceback=True
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle FastAPI request validation errors."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "error_code": "REQUEST_VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                },
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "request_id": request_id,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle standard HTTP exceptions."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "error_code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                },
                "status_code": exc.status_code,
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle any unhandled exceptions (last resort)."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        error_response = {
            "error": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": f"{type(exc).__name__}: {str(exc)}",
            },
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "request_id": request_id,
        }

        if settings.DEBUG:
            error_response["traceback"] = traceback.format_exc()

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response,
        )
