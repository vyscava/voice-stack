"""
Core exception classes for the Voice Stack application.

This module defines the base exception hierarchy that all service-specific
exceptions should inherit from. These exceptions are designed to provide
rich error context for logging and debugging.
"""

from __future__ import annotations

from typing import Any


class VoiceStackError(Exception):
    """
    Base exception class for all Voice Stack errors.

    All custom exceptions should inherit from this class to maintain
    a consistent error handling interface across the application.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for categorization
        details: Additional error details (e.g., original exception message)
        context: Dictionary containing request context (file info, params, etc.)
        original_exception: The original exception that was wrapped, if any
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        """
        Initialize a VoiceStackError.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code (defaults to class name in UPPER_SNAKE_CASE)
            details: Additional error details
            context: Request context dictionary
            original_exception: The original exception that was wrapped
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._default_error_code()
        self.details = details
        self.context = context or {}
        self.original_exception = original_exception

    def _default_error_code(self) -> str:
        """Generate default error code from class name."""
        # Convert class name like "AudioDecodingError" to "AUDIO_DECODING_ERROR"
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).upper()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the error
        """
        error_dict: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
        }

        if self.details:
            error_dict["details"] = self.details

        if self.context:
            error_dict["context"] = self.context

        if self.original_exception:
            error_dict["original_exception"] = {
                "type": type(self.original_exception).__name__,
                "module": type(self.original_exception).__module__,
                "message": str(self.original_exception),
            }

        return error_dict


class ValidationError(VoiceStackError):
    """
    Exception raised when input validation fails.

    This typically maps to HTTP 400 or 422 status codes.
    Examples: invalid parameters, malformed requests, unsupported formats.
    """


class ResourceError(VoiceStackError):
    """
    Exception raised when system resources are exhausted.

    This typically maps to HTTP 503 status codes.
    Examples: out of memory, GPU memory exhausted, all worker slots busy.
    """


class ProcessingError(VoiceStackError):
    """
    Exception raised when processing fails unexpectedly.

    This typically maps to HTTP 500 status codes.
    Examples: model inference failures, unexpected runtime errors.
    """


class NotFoundError(VoiceStackError):
    """
    Exception raised when a requested resource is not found.

    This typically maps to HTTP 404 status codes.
    Examples: model not found, voice not found, file not found.
    """
