"""
Resource Guard Middleware for Voice Stack.

This middleware provides the first line of defense against resource exhaustion
by checking system resources BEFORE accepting requests.

Checks performed:
1. Memory usage (RAM) - Rejects if > configured threshold (default 90%)
2. Memory pressure - Rejects if RAM > 70% AND swap > threshold (default 80%)
   (High swap alone is not a concern if there's plenty of free RAM)
3. File upload size - Rejects if Content-Length > configured max (default 100MB)

All thresholds are configurable via environment variables.
"""

from __future__ import annotations

from typing import Any

import psutil
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from core.logging import logger_asr as logger
from core.settings import get_settings

settings = get_settings()


class ResourceGuardMiddleware(BaseHTTPMiddleware):
    """
    Middleware to guard against resource exhaustion.

    This middleware runs BEFORE FastAPI processes any request, providing
    fast rejection when system resources are constrained.

    Configuration (via .env):
        MEMORY_THRESHOLD_PERCENT: Reject when RAM usage exceeds this % (default: 90)
        SWAP_THRESHOLD_PERCENT: Reject when swap > threshold AND RAM > 70% (default: 80)
        MAX_UPLOAD_SIZE_MB: Maximum file upload size in MB (default: 100)

    HTTP Status Codes:
        503 Service Unavailable: Memory/swap pressure (client should retry)
        413 Payload Too Large: File exceeds size limit (client error)

    Note:
        High swap usage alone is not considered a problem if RAM is available.
        Only when BOTH RAM > 70% AND swap exceeds threshold will requests be rejected.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.memory_threshold = getattr(settings, "MEMORY_THRESHOLD_PERCENT", 90)
        self.swap_threshold = getattr(settings, "SWAP_THRESHOLD_PERCENT", 80)
        self.max_upload_mb = getattr(settings, "MAX_UPLOAD_SIZE_MB", 100)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Check system resources before allowing request to proceed.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/endpoint in the chain

        Returns:
            Response: Either an error response (503/413) or the result of call_next()
        """
        # 1. Check memory availability
        mem = psutil.virtual_memory()

        if mem.percent > self.memory_threshold:
            logger.warning(
                f"Memory usage critical: {mem.percent:.1f}% > {self.memory_threshold}% threshold - rejecting request"
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "detail": "Insufficient memory available",
                    "memory_percent": round(mem.percent, 1),
                    "threshold": self.memory_threshold,
                },
                headers={"Retry-After": "60"},  # Try again in 60 seconds
            )

        # 2. Check swap usage (only if memory is also moderately high)
        # High swap alone isn't a problem if there's plenty of free RAM.
        # Only reject if BOTH memory is >70% AND swap exceeds threshold.
        # This indicates actual memory pressure, not just old swapped pages.
        swap = psutil.swap_memory()

        if mem.percent > 70 and swap.percent > self.swap_threshold:
            logger.warning(
                f"Memory pressure detected: RAM {mem.percent:.1f}%, "
                f"Swap {swap.percent:.1f}% > {self.swap_threshold}% threshold - rejecting request"
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "detail": "System under memory pressure",
                    "memory_percent": round(mem.percent, 1),
                    "swap_percent": round(swap.percent, 1),
                    "swap_threshold": self.swap_threshold,
                },
                headers={"Retry-After": "120"},  # Try again in 2 minutes
            )

        # 3. Check content-length for file uploads (prevent huge files)
        content_length = request.headers.get("content-length")

        if content_length:
            size_mb = int(content_length) / (1024 * 1024)

            if size_mb > self.max_upload_mb:
                logger.warning(f"File too large: {size_mb:.1f}MB > {self.max_upload_mb}MB limit - rejecting request")
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "detail": f"File size {size_mb:.1f}MB exceeds {self.max_upload_mb}MB limit",
                        "file_size_mb": round(size_mb, 1),
                        "max_size_mb": self.max_upload_mb,
                    },
                )

        # All checks passed - proceed with request
        response = await call_next(request)
        return response
