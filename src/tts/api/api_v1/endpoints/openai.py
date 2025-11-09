"""
OpenAI-compatible TTS endpoint.

- Accepts JSON with text input, voice selection, and audio format options
- Returns audio data in the requested format (mp3, opus, aac, flac, wav, pcm)
- Supports streaming or complete audio responses
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.error_handler import ErrorContext
from core.logging import logger_tts as logger
from core.settings import get_settings
from tts.engine_factory import acquire_engine, get_audio_engine, release_engine
from tts.exceptions import ModelNotFoundError, SynthesisError, VoiceNotFoundError
from tts.schemas.audio_engine import ModelResponse, ModelsResponse, VoicesResponse
from tts.schemas.openai import OpenAICreateSpeechRequest

router = APIRouter()

settings = get_settings()
# Note: get_audio_engine() is used for model/voice listing (read-only),
# but synthesis endpoints use acquire_engine() for concurrency control
_engine_for_listing = get_audio_engine()


@router.get("/audio/voices", response_model=VoicesResponse)
async def list_voices() -> Any:
    """
    List all available voices for the current TTS engine.
    Returns a list of voice objects with their IDs and names.

    Note: This endpoint uses get_audio_engine() directly (not acquire_engine())
    because it's just reading voice metadata, not performing synthesis.
    """
    ctx = ErrorContext.create(endpoint="/audio/voices")
    ctx.add_model_info(model=settings.TTS_MODEL, engine=settings.TTS_ENGINE)

    try:
        logger.info("Listing available voices", extra=ctx.to_log_dict())
        return _engine_for_listing.list_voices()
    except Exception as e:
        logger.error(
            "Failed to list voices",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise SynthesisError(
            message="Failed to list available voices",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e


# Reference: https://platform.openai.com/docs/api-reference/models/list
@router.get("/models", response_model=ModelsResponse)
async def list_models() -> Any:
    """
    List all available TTS models.

    Note: This endpoint uses get_audio_engine() directly (not acquire_engine())
    because it's just reading model metadata, not performing synthesis.
    """
    ctx = ErrorContext.create(endpoint="/models")
    ctx.add_model_info(model=settings.TTS_MODEL, engine=settings.TTS_ENGINE)

    try:
        logger.info("Listing available models", extra=ctx.to_log_dict())
        return _engine_for_listing.list_models()
    except Exception as e:
        logger.error(
            "Failed to list models",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise SynthesisError(
            message="Failed to list available models",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e


# Reference: https://platform.openai.com/docs/api-reference/models/retrieve
@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str) -> Any:
    """
    Get information about a specific TTS model.

    Note: This endpoint uses get_audio_engine() directly (not acquire_engine())
    because it's just reading model metadata, not performing synthesis.
    """
    ctx = ErrorContext.create(endpoint=f"/models/{model_id}")
    ctx.add_model_info(model=settings.TTS_MODEL, engine=settings.TTS_ENGINE)
    ctx.add_params({"requested_model_id": model_id})

    try:
        logger.info("Retrieving model info", extra=ctx.to_log_dict())
        return _engine_for_listing.get_model(model_id=model_id)
    except KeyError as e:
        # Model not found
        logger.warning(
            "Model not found",
            extra={**ctx.to_log_dict(), "error_type": "KeyError"},
        )
        raise ModelNotFoundError(
            message=f"Model '{model_id}' not found",
            details="The requested model ID does not exist",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e
    except Exception as e:
        logger.error(
            "Failed to retrieve model info",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise SynthesisError(
            message="Failed to retrieve model information",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e


# Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post("/audio/speech")
async def tts(request: OpenAICreateSpeechRequest) -> Any:
    """
    Generate speech audio from text input.

    Supports multiple output formats (mp3, opus, aac, flac, wav, pcm) and
    both streaming and complete audio responses.
    """
    # Create error context for better logging
    ctx = ErrorContext.create(endpoint="/audio/speech")
    ctx.add_params(
        {
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
            "stream_format": request.stream_format,
            "input_length": len(request.input),
        }
    )
    ctx.add_model_info(model=settings.TTS_MODEL, engine=settings.TTS_ENGINE)

    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        # Log request details
        logger.info("Processing TTS synthesis request", extra=ctx.to_log_dict())

        result = engine.speech(
            input=request.input,
            voice=request.voice,
            response_format=request.response_format,
            speed=request.speed,
            stream_format=request.stream_format,
        )

        if request.stream_format == "audio":
            return engine.helper_return_audio_file(audio=result, response_format=request.response_format)
        else:
            return engine.helper_return_sse_stream(audio=result, already_sse=True)

    except KeyError as e:
        # Voice not found
        logger.error(
            "Voice not found",
            extra={**ctx.to_log_dict(), "error_type": "KeyError"},
            exc_info=True,
        )
        raise VoiceNotFoundError(
            message=f"Voice '{request.voice}' not found",
            details="The requested voice is not available",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e

    except (ValueError, TypeError) as e:
        # Input validation errors
        logger.error(
            "TTS input validation failed",
            extra={**ctx.to_log_dict(), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise SynthesisError(
            message="Invalid input for TTS synthesis",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e

    except Exception as e:
        # Unexpected errors (server error)
        logger.error(
            "TTS synthesis failed",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise SynthesisError(
            message="TTS synthesis failed",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e
    finally:
        # Always release the concurrency slot
        release_engine()
