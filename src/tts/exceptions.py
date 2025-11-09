"""
TTS-specific exception classes.

This module defines exceptions specific to Text-to-Speech (TTS) operations.
All exceptions inherit from the core VoiceStackError hierarchy.
"""

from __future__ import annotations

from typing import Any

from core.exceptions import NotFoundError, ProcessingError, ValidationError


class SynthesisError(ProcessingError):
    """
    Exception raised when TTS synthesis fails.

    This typically indicates:
    - Model inference failure
    - Out of memory during processing
    - Unexpected runtime error during synthesis

    Maps to HTTP 500 (Internal Server Error)
    """

    def __init__(
        self,
        message: str = "TTS synthesis failed",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="SYNTHESIS_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class AudioGenerationError(ProcessingError):
    """
    Exception raised when audio generation fails.

    This typically indicates:
    - Failed to generate audio waveform
    - Audio encoding failure
    - Vocoder failure

    Maps to HTTP 500 (Internal Server Error)
    """

    def __init__(
        self,
        message: str = "Audio generation failed",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUDIO_GENERATION_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class VoiceNotFoundError(NotFoundError):
    """
    Exception raised when a requested voice is not available.

    This typically indicates:
    - Requested voice name does not exist
    - Voice not configured in the system

    Maps to HTTP 404 (Not Found)
    """

    def __init__(
        self,
        message: str = "Voice not found",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="VOICE_NOT_FOUND",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class ModelLoadError(ProcessingError):
    """
    Exception raised when TTS model loading fails.

    This typically indicates:
    - Model files not found
    - Corrupted model files
    - Insufficient memory to load model
    - CUDA/GPU initialization failure

    Maps to HTTP 500 (Internal Server Error)
    """

    def __init__(
        self,
        message: str = "Failed to load TTS model",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class ModelNotFoundError(NotFoundError):
    """
    Exception raised when a requested TTS model is not available.

    This typically indicates:
    - Requested model name does not exist
    - Model not configured in the system

    Maps to HTTP 404 (Not Found)
    """

    def __init__(
        self,
        message: str = "TTS model not found",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class TextValidationError(ValidationError):
    """
    Exception raised when text input validation fails.

    This typically indicates:
    - Text too long
    - Empty text input
    - Invalid characters or encoding

    Maps to HTTP 422 (Unprocessable Entity)
    """

    def __init__(
        self,
        message: str = "Text validation failed",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TEXT_VALIDATION_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )
