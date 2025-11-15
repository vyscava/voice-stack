from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, Query, UploadFile, status
from fastapi.responses import JSONResponse

from asr.engine_factory import acquire_engine, get_audio_engine, release_engine
from asr.exceptions import AudioDecodingError, LanguageDetectionError, TranscriptionError
from asr.schemas.audio_engine import DetectedLanguage, ListModelsResponse, Output
from core.error_handler import ErrorContext
from core.logging import logger_asr as logger
from core.settings import get_settings

router = APIRouter()
settings = get_settings()
# Note: get_audio_engine() is used for model listing (read-only),
# but inference endpoints use acquire_engine() for concurrency control
_engine_for_listing = get_audio_engine()

# Simple in-memory stats tracking
_stats: dict[str, Any] = {
    "started_at": datetime.now().isoformat(),
    "endpoints": {
        "/asr": defaultdict(int),
        "/detect-language": defaultdict(int),
    },
    "total_requests": 0,
    "total_success": 0,
    "total_errors": 0,
}


def _track_request(endpoint: str, status_code: int) -> None:
    """Track request statistics."""
    _stats["total_requests"] += 1
    _stats["endpoints"][endpoint][f"status_{status_code}"] += 1

    if 200 <= status_code < 300:
        _stats["total_success"] += 1
        _stats["endpoints"][endpoint]["success"] += 1
    else:
        _stats["total_errors"] += 1
        _stats["endpoints"][endpoint]["errors"] += 1


@router.post(
    "/asr",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "application/x-subrip": {},
                "text/vtt; charset=utf-8": {},
                "text/plain; charset=utf-8": {},
                "text/tab-separated-values; charset=utf-8": {},
                "application/jsonl; charset=utf-8": {},
            },
            "description": "Bazarr-compatible ASR outputs (JSON subtitles payload or subtitle text).",
        }
    },
)
async def bazarr_asr(
    audio_file: UploadFile = File(...),
    task: str = Query(default="transcribe", description="Operation: transcribe or translate"),
    language: str | None = Query(default=None, description="Language of the audio if known"),
    initial_prompt: str | None = Query(default=None, description="Prompt text; ignored for Whisper"),
    encode: bool = Query(default=True, description="IGNORED: Encode audio first through ffmpeg"),
    output: Output | None = Query(default=Output.TXT, description="Output formats: txt|vtt|srt|tsv|json|jsonl"),
    video_file: str | None = Query(default=None, description="Original video file path for logging purposes"),
    vad: bool = Query(
        default=False,
        description=(
            "Enable Voice Activity Detection (VAD) to remove silence. Disabled by default for subtitles "
            "to ensure accurate timing and capture all audio including music/songs."
        ),
    ),
) -> Any:
    """
    Returns segments suitable for creating subtitles in Bazarr workflows.
     JSON shape (when output=json):
      { "language": "en", "segments": [ { "start": 0.0, "end": 1.2, "text": "..." }, ... ] }

    Note: VAD is disabled by default for subtitle generation because:
    - It ensures timestamps stay synchronized with the original video
    - Music and songs are not incorrectly classified as non-voice
    - All audio content is transcribed, not just detected speech
    """
    # default to TXT if output is None (defensive)
    output_format: Output = output or Output.TXT

    # Create error context for better logging
    ctx = ErrorContext.create(endpoint="/bazarr/asr")

    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        # Read entire uploaded audio bytes into memory
        audio = await audio_file.read()
        size_b = len(audio)

        ctx.add_file_info(
            filename=audio_file.filename,
            size_bytes=size_b,
            content_type=audio_file.content_type,
        )
        ctx.add_params(
            {
                "task": task,
                "language": language,
                "output_format": str(output_format),
                "video_file": video_file,
                "beam_size": 10,
                "temperature": 0.0,
                "vad": vad,
            }
        )
        ctx.add_model_info(model=settings.ASR_MODEL, engine=settings.ASR_ENGINE)

        logger.info("Bazarr ASR request", extra=ctx.to_log_dict())

        result = engine.transcribe_file(
            file_bytes=audio,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            request_language=language,
            beam_size=5,
            temperature=0.0,
            best_of=1,
            vad=vad,
            expect_raw_pcm=True,  # Bazarr always sends headerless PCM
        )

        _track_request("/asr", 200)
        return engine.helper_write_output(file=audio_file, result=result, output=output_format)

    except (OSError, ValueError) as e:
        # File decoding or validation errors (client error)
        _track_request("/asr", 422)
        logger.error(
            "Audio decoding failed (Bazarr endpoint)",
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
        _track_request("/asr", 500)
        logger.error(
            "Bazarr ASR processing failed",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise TranscriptionError(
            message="Bazarr ASR processing failed",
            details=f"{type(e).__name__}: {str(e)}",
            context=ctx.to_dict(),
            original_exception=e,
        ) from e
    finally:
        # Always release the concurrency slot
        release_engine()


@router.post(
    "/detect-language",
    response_model=DetectedLanguage,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": DetectedLanguage.model_json_schema()},
            },
            "description": "JSON (Bazarr-compatible) for detected language response.",
        }
    },
)
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="IGNORED: Encode audio first through ffmpeg"),
    detect_lang_length: int | None = Query(default=None, description="Detect language on X seconds of the file"),
    detect_lang_offset: int | None = Query(default=None, description="Start Detect language X seconds into the file"),
    video_file: str | None = Query(default=None, description="Original video file path for logging purposes"),
) -> Any:
    """
    Detect the Language of Audio for Bazarr
    """
    # Create error context for better logging
    ctx = ErrorContext.create(endpoint="/bazarr/detect-language")

    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        audio = await audio_file.read()
        size_b = len(audio)

        ctx.add_file_info(
            filename=audio_file.filename,
            size_bytes=size_b,
            content_type=audio_file.content_type,
        )
        ctx.add_params(
            {
                "detect_lang_length": detect_lang_length,
                "detect_lang_offset": detect_lang_offset,
                "video_file": video_file,
            }
        )
        ctx.add_model_info(model=settings.ASR_MODEL, engine=settings.ASR_ENGINE)

        logger.info("Bazarr language detection request", extra=ctx.to_log_dict())

        result = engine.detect_language_file(
            file_bytes=audio,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            detect_lang_length=detect_lang_length,
            detect_lang_offset=detect_lang_offset,
            expect_raw_pcm=True,  # Bazarr always sends headerless PCM
        )

        _track_request("/detect-language", 200)
        return DetectedLanguage(**result.to_dict())

    except (OSError, ValueError) as e:
        # File decoding or validation errors (client error)
        _track_request("/detect-language", 422)
        logger.error(
            "Audio decoding failed (language detection endpoint)",
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
        _track_request("/detect-language", 500)
        logger.error(
            "Language detection failed",
            extra={
                **ctx.to_log_dict(),
                "error_type": type(e).__name__,
                "error_module": type(e).__module__,
            },
            exc_info=True,
        )
        raise LanguageDetectionError(
            message="Language detection failed",
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


@router.get("/stats")
def get_stats() -> Any:
    """
    Get statistics about Bazarr endpoint usage.

    Returns request counts by endpoint and status code, along with
    success/error rates. Useful for monitoring endpoint health.
    """
    # Convert defaultdict to regular dict for JSON serialization
    stats_output = {
        "started_at": _stats["started_at"],
        "total_requests": _stats["total_requests"],
        "total_success": _stats["total_success"],
        "total_errors": _stats["total_errors"],
        "success_rate": (
            round(_stats["total_success"] / _stats["total_requests"] * 100, 2) if _stats["total_requests"] > 0 else 0.0
        ),
        "endpoints": {endpoint: dict(counts) for endpoint, counts in _stats["endpoints"].items()},
    }

    return JSONResponse(content=stats_output)
