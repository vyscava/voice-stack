"""
OpenAI-compatible transcription endpoint.

- Accepts multipart/form-data with `file`, plus optional fields
  (model, language, temperature, response_format, prompt, timestamp_granularities).
- Returns **exactly** the shape OpenAI expects for JSON: {"text": "..."}.
- Everything else (decode, resample, VAD, model inference) happens in AudioEngine.

If you want additional metadata, expose another route (e.g., /v1/audio/transcriptions/verbose)
that returns the full dict from the engine.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import PlainTextResponse

from asr.engine_factory import acquire_engine, get_audio_engine, release_engine
from asr.exceptions import AudioDecodingError, TranscriptionError
from asr.schemas.audio_engine import ListModelsResponse
from asr.schemas.openai import (
    OpenAITranscription,
    OpenAITranscriptionRequest,
    OpenAITranscriptionVerboseRequest,
    OpenAITranslationsRequest,
    TranscribeVerboseResponse,
)
from core.error_handler import ErrorContext
from core.logging import logger_asr as logger
from core.settings import get_settings

router = APIRouter()

settings = get_settings()
# Note: get_audio_engine() is used for model listing (read-only),
# but inference endpoints use acquire_engine() for concurrency control
_engine_for_listing = get_audio_engine()


@router.post(
    "/audio/transcriptions",
    response_model=OpenAITranscription,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": OpenAITranscription.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
            "description": "JSON (OpenAI-compatible) or plain text transcript",
        }
    },
)
async def transcriptions(
    request: OpenAITranscriptionRequest = Depends(OpenAITranscriptionRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    # Create error context for better logging
    ctx = ErrorContext.create(endpoint="/audio/transcriptions")
    ctx.add_file_info(
        filename=file.filename,
        size_bytes=file.size if hasattr(file, "size") and file.size else None,
        content_type=file.content_type,
    )
    ctx.add_params(
        {
            "model": request.model,
            "language": request.language,
            "temperature": request.temperature,
            "response_format": request.response_format,
        }
    )
    ctx.add_model_info(model=settings.ASR_MODEL, engine=settings.ASR_ENGINE)

    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        # Read entire uploaded audio bytes into memory
        audio = await file.read()

        # Log request details
        logger.info("Processing transcription request", extra=ctx.to_log_dict())

        result = engine.transcribe_file(
            file_bytes=audio, filename=file.filename, request_language=request.language, temperature=request.temperature
        )

        # If return  is text lets give it a plan text
        if request.response_format == "text":
            return PlainTextResponse(content=result.text, status_code=status.HTTP_200_OK)

        # or we default to JSON
        return OpenAITranscription(**result.to_dict())

    except (OSError, ValueError) as e:
        # File decoding or validation errors (client error)
        logger.error(
            "Audio decoding failed",
            extra={**ctx.to_log_dict(), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise AudioDecodingError(
            message="Failed to decode audio file",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e

    except Exception as e:
        # Unexpected errors (server error)
        logger.error(
            "Transcription processing failed",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise TranscriptionError(
            message="Transcription processing failed",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e
    finally:
        # Always release the concurrency slot
        release_engine()


@router.post(
    "/audio/transcriptions/verbose",
    response_model=TranscribeVerboseResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": TranscribeVerboseResponse.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
            "description": "JSON (OpenAI-compatible) or plain text transcript",
        }
    },
)
async def transcriptions_verbose(
    request: OpenAITranscriptionVerboseRequest = Depends(OpenAITranscriptionVerboseRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    """
    Verbose response: returns the full dict from the engine, including:
      text, segments, language, language_probability, durations, timings, vad_used, model.
    """
    # Create error context for better logging
    ctx = ErrorContext.create(endpoint="/audio/transcriptions/verbose")
    ctx.add_file_info(
        filename=file.filename,
        size_bytes=file.size if hasattr(file, "size") and file.size else None,
        content_type=file.content_type,
    )
    ctx.add_params(
        {
            "model": request.model,
            "language": request.language,
            "task": request.task,
            "beam_size": request.beam_size,
            "temperature": request.temperature,
            "best_of": request.best_of,
            "word_timestamps": request.word_timestamps,
            "vad": request.vad,
            "response_format": request.response_format,
        }
    )
    ctx.add_model_info(model=settings.ASR_MODEL, engine=settings.ASR_ENGINE)

    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        audio = await file.read()

        # Log request details
        logger.info("Processing verbose transcription request", extra=ctx.to_log_dict())

        result = engine.transcribe_file(
            file_bytes=audio,
            filename=file.filename,
            request_language=request.language,
            task=request.task,
            beam_size=request.beam_size,
            temperature=request.temperature,
            best_of=request.best_of,
            word_timestamps=request.word_timestamps,
            vad=request.vad,
        )

        if request.response_format == "text":
            # If someone still asks for text, mirror the minimal behavior.
            return PlainTextResponse(content=result.text, status_code=status.HTTP_200_OK)

        # Full JSON payload for debugging/analytics.
        return TranscribeVerboseResponse(**result.to_dict())

    except (OSError, ValueError) as e:
        # File decoding or validation errors (client error)
        logger.error(
            "Audio decoding failed (verbose endpoint)",
            extra={**ctx.to_log_dict(), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise AudioDecodingError(
            message="Failed to decode audio file",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e

    except Exception as e:
        # Unexpected errors (server error)
        logger.error(
            "Verbose transcription processing failed",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise TranscriptionError(
            message="Verbose transcription processing failed",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e
    finally:
        # Always release the concurrency slot
        release_engine()


@router.post(
    "/audio/translations",
    response_model=OpenAITranscription,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": OpenAITranscription.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
            "description": "JSON (OpenAI-compatible) or plain text transcript",
        }
    },
)
async def translations(
    request: OpenAITranslationsRequest = Depends(OpenAITranslationsRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    """
    OpenAI-compatible Translation endpoint.

    Behavior:
      - Runs Whisper/Faster-Whisper with task="translate" (source → English).
      - Returns OpenAI minimal shape for JSON: {"text": "..."}.
      - If response_format="text", returns plain text instead of JSON.
    """
    # Create error context for better logging
    ctx = ErrorContext.create(endpoint="/audio/translations")
    ctx.add_file_info(
        filename=file.filename,
        size_bytes=file.size if hasattr(file, "size") and file.size else None,
        content_type=file.content_type,
    )
    ctx.add_params(
        {
            "model": request.model,
            "task": "translate",
            "temperature": request.temperature,
            "response_format": request.response_format,
        }
    )
    ctx.add_model_info(model=settings.ASR_MODEL, engine=settings.ASR_ENGINE)

    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        audio = await file.read()

        # Log request details
        logger.info("Processing translation request", extra=ctx.to_log_dict())

        result = engine.transcribe_file(
            file_bytes=audio,
            filename=file.filename,
            # Let the engine auto-detect source language; translation target is English.
            task="translate",
            temperature=request.temperature,
            # Intentionally don't pass `request_language` here; Whisper detects source.
        )

        if request.response_format == "text":
            return PlainTextResponse(content=result.text, status_code=status.HTTP_200_OK)

        return OpenAITranscription(**result.to_dict())

    except (OSError, ValueError) as e:
        # File decoding or validation errors (client error)
        logger.error(
            "Audio decoding failed (translation endpoint)",
            extra={**ctx.to_log_dict(), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise AudioDecodingError(
            message="Failed to decode audio file",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e

    except Exception as e:
        # Unexpected errors (server error)
        logger.error(
            "Translation processing failed",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise TranscriptionError(
            message="Translation processing failed",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e
    finally:
        # Always release the concurrency slot
        release_engine()


@router.get("/models", response_model=ListModelsResponse)
def list_models() -> Any:
    """
    OpenAI-compatible models listing. Delegates to the engine so other
    API mocks (or future services) can reuse the same definition.

    Note: This endpoint uses get_audio_engine() directly (not acquire_engine())
    because it's just reading model metadata, not performing inference.
    """
    return ListModelsResponse(**_engine_for_listing.list_models())
