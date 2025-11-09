"""
ASR-specific exception classes.

This module defines exceptions specific to Automatic Speech Recognition (ASR)
operations. All exceptions inherit from the core VoiceStackError hierarchy.
"""

from __future__ import annotations

from typing import Any

from core.exceptions import NotFoundError, ProcessingError, ValidationError


class AudioDecodingError(ValidationError):
    """
    Exception raised when audio file decoding fails.

    This typically indicates:
    - Unsupported audio codec or format
    - Corrupted audio file
    - Invalid file structure

    Maps to HTTP 422 (Unprocessable Entity)
    """

    def __init__(
        self,
        message: str = "Failed to decode audio file",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUDIO_DECODING_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class TranscriptionError(ProcessingError):
    """
    Exception raised when transcription processing fails.

    This typically indicates:
    - Model inference failure
    - Out of memory during processing
    - Unexpected runtime error during transcription

    Maps to HTTP 500 (Internal Server Error)
    """

    def __init__(
        self,
        message: str = "Transcription processing failed",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TRANSCRIPTION_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class LanguageDetectionError(ProcessingError):
    """
    Exception raised when language detection fails.

    This typically indicates:
    - Model inference failure during language detection
    - Unable to detect language from audio
    - Unexpected runtime error during detection

    Maps to HTTP 500 (Internal Server Error)
    """

    def __init__(
        self,
        message: str = "Language detection failed",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="LANGUAGE_DETECTION_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )


class ModelLoadError(ProcessingError):
    """
    Exception raised when ASR model loading fails.

    This typically indicates:
    - Model files not found
    - Corrupted model files
    - Insufficient memory to load model
    - CUDA/GPU initialization failure

    Maps to HTTP 500 (Internal Server Error)
    """

    def __init__(
        self,
        message: str = "Failed to load ASR model",
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
    Exception raised when a requested ASR model is not available.

    This typically indicates:
    - Requested model name does not exist
    - Model not configured in the system

    Maps to HTTP 404 (Not Found)
    """

    def __init__(
        self,
        message: str = "ASR model not found",
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


class AudioValidationError(ValidationError):
    """
    Exception raised when audio file validation fails.

    This typically indicates:
    - Audio file too large
    - Audio file too short/empty
    - Invalid audio parameters (sample rate, channels, etc.)

    Maps to HTTP 422 (Unprocessable Entity)
    """

    def __init__(
        self,
        message: str = "Audio validation failed",
        details: str | None = None,
        context: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUDIO_VALIDATION_ERROR",
            details=details,
            context=context,
            original_exception=original_exception,
        )
